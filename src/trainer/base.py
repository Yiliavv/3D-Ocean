import os
import torch
import wandb
import arrow
import tempfile
import platform
import numpy as np
import matplotlib.pyplot as plt

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Subset

from src.plot.sst import plot_sst, plot_sst_diff, plot_nino
from src.config.area import Area, PROJECT_PATH
from src.trainer.wandb import Wandb

class BaseTrainer:
    """
    训练器基类 - 集成性能优化和 Checkpoint 机制
    
    参数:
        area: Area, 区域
        model_class: LightningModule, 模型类
        checkpoint_path: str, checkpoint 路径 (用于恢复训练)
        dataset_params: dict, 数据集参数
        trainer_params: dict, 训练参数
        model_params: dict, 模型参数
        use_checkpoint: bool, 是否使用 checkpoint 机制 (默认: True)
        use_wandb: bool, 是否使用 wandb 日志 (默认: False)
        
    dataset_params:
        seq_len: int, 序列长度
        offset: int, 偏移量
        resolution: float, 分辨率
        
    trainer_params:
        epochs: int, 训练轮数
        batch_size: int, 批量大小
        split_ratio: list, 训练集和验证集的分割比例
        
        # Checkpoint 参数
        save_top_k: int, 保存最好的 k 个模型 (默认: 3)
        monitor: str, 监控的指标 (默认: "val_loss")
        mode: str, 监控模式 (默认: "min")
        
        # 性能优化参数
        num_workers: int, 数据加载工作进程数 (默认: 8, 推荐CPU核心数/2)
        pin_memory: bool, 是否固定GPU内存 (默认: True)
        prefetch_factor: int, 预取因子 (默认: 2)
        precision: str, 训练精度 (默认: "16-mixed", 可选: "32", "bf16-mixed")
        compile_model: bool, 是否编译模型-PyTorch2.0+ (默认: False)
        compile_mode: str, 编译模式 (默认: "default")
        
    使用示例:
        # 第一次训练
        trainer = BaseTrainer(
            area=area,
            model_class=YourModel,
            dataset_class=YourDataset,
            trainer_params={'epochs': 100},
            use_checkpoint=True
        )
        model = trainer.train()
        
        # 从 checkpoint 恢复并继续训练
        trainer = BaseTrainer(
            area=area,
            model_class=YourModel,
            dataset_class=YourDataset,
            checkpoint_path=f'{PROJECT_PATH}/out/checkpoints/YourModel/last.ckpt',  # 加载 checkpoint（使用PROJECT_PATH）
            trainer_params={'epochs': 150},  # 可以修改超参数
            use_checkpoint=True
        )
        model = trainer.train()

    """
    
    def __init__(self,
                 area: Area,
                 model_class = None,
                 dataset_class = None,
                 checkpoint_path: str = None,  # checkpoint 路径
                 dataset_params: dict = {},
                 trainer_params: dict = {},
                 model_params: dict = {},
                 use_checkpoint: bool = True,  # 是否使用 checkpoint
                 use_wandb: bool = True):  # 是否使用 wandb 日志
        
        self.trainer_uid = arrow.now().format('YYYY-MM-DD-HH-mm')

        self.area = area
        
        # 工厂类型
        self.model_class = model_class
        self.dataset_class = dataset_class

        # 参数
        self.dataset_params = dataset_params
        self.trainer_params = trainer_params
        self.model_params = model_params
        
        # Checkpoint配置
        self.checkpoint_path = checkpoint_path
        self.use_checkpoint = use_checkpoint
        
        # 模型状态
        self.model = None
        self.trained = False
        
        # Wandb 配置
        if use_wandb:
            self.wandb = Wandb(
                uid=self.trainer_uid,
                model_class=model_class,
                dataset_class=dataset_class,
                area=area,
                model_params=model_params,
                dataset_params=dataset_params,
                trainer_params=trainer_params,
                enabled=True
            )
        else:
            self.wandb = None
        
        # checkpoint callback
        self.checkpoint_callback = None
    
    def split(self, dataset):
        split_ratio = self.trainer_params.get('split_ratio', [0.9, 0.1])
        batch_size = self.trainer_params.get('batch_size', 20)
        
        # 计算训练集大小（按时间顺序有序分割）
        total_size = len(dataset)
        train_size = int(total_size * split_ratio[0])
        
        # 按时间顺序分割：前train_size个样本作为训练集，后val_size个样本作为验证集
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, total_size))
        
        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)
        
        # 优化的DataLoader配置
        # Windows系统上多进程DataLoader可能有问题，默认使用单进程
        is_windows = platform.system() == 'Windows'
        default_workers = 0 if is_windows else 8
        
        num_workers = self.trainer_params.get('num_workers', default_workers)
        pin_memory = self.trainer_params.get('pin_memory', True)
        prefetch_factor = self.trainer_params.get('prefetch_factor', 2)
        
        dataloader_kwargs = {
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
        }
        
        if num_workers > 0:
            # num_workers > 0 时，默认使用 persistent_workers=True
            persistent_workers = self.trainer_params.get('persistent_workers', True)
            dataloader_kwargs['persistent_workers'] = persistent_workers
            if prefetch_factor:
                dataloader_kwargs['prefetch_factor'] = prefetch_factor
        
        train_loader = DataLoader(train_set, **dataloader_kwargs)
        
        # 验证集使用较少的workers
        val_dataloader_kwargs = dataloader_kwargs.copy()
        val_dataloader_kwargs['num_workers'] = max(1, num_workers // 2)
        val_loader = DataLoader(val_set, **val_dataloader_kwargs)
        
        return train_loader, val_loader
        
    def _create_checkpoint_callback(self):
        """创建 checkpoint callback（参考 PyTorch Lightning 官方示例）"""
        if not self.use_checkpoint:
            return None
        
        checkpoint_dir = f'{PROJECT_PATH}/out/checkpoints/{self.model_class.__name__}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=self.model_class.__name__,
            monitor=self.trainer_params.get('monitor', 'val_loss'),
            mode=self.trainer_params.get('mode', 'min'),
            save_top_k=self.trainer_params.get('save_top_k', 1),
            save_last=False,
            verbose=False,
        )
        
        print(f"\n💾 Checkpoint: {checkpoint_dir}\n")
        
        return checkpoint_callback
    
    def train(self):
        lon = self.area.lon
        lat = self.area.lat
        
        # 启用 Tensor Cores 优化（适用于 RTX 系列 GPU）
        if torch.cuda.is_available() and hasattr(torch, 'set_float32_matmul_precision'):
            matmul_precision = self.trainer_params.get('matmul_precision', 'high')
            torch.set_float32_matmul_precision(matmul_precision)
        
        dataset = self.dataset_class(
            lon=lon,
            lat=lat,
            **self.dataset_params
        )
        
        train_loader, val_loader = self.split(dataset)
        
        # 创建模型（PyTorch Lightning 会自动从 checkpoint 恢复状态）
        self.model = self.model_class(**self.model_params)
        
        # PyTorch 2.0+ 模型编译
        if self.trainer_params.get('compile_model', False):
            if hasattr(torch, 'compile'):
                compile_mode = self.trainer_params.get('compile_mode', 'default')
                print(f"🚀 编译模型 (模式: {compile_mode})...")
                self.model = torch.compile(self.model, mode=compile_mode)
            else:
                print("⚠️  PyTorch版本 < 2.0, 模型编译不可用")
        
        # 检查 checkpoint 路径
        ckpt_path = None
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            ckpt_path = self.checkpoint_path
            print(f"\n🔄 从 checkpoint 恢复: {self.checkpoint_path}\n")
        
        epochs = self.trainer_params.get('epochs', 100)
        
        # 初始化 wandb
        if self.wandb:
            self.wandb.init(self.model)
        
        # 创建 checkpoint callback
        self.checkpoint_callback = self._create_checkpoint_callback()
        
        # 优化的Trainer配置
        trainer_config = {
            'max_epochs': epochs,
            'accelerator': 'gpu',
            'enable_checkpointing': self.use_checkpoint,
            'num_sanity_val_steps': 0,
            'precision': self.trainer_params.get('precision', '16-mixed'),
        }
        
        # 添加 callbacks
        callbacks = []
        if self.checkpoint_callback:
            callbacks.append(self.checkpoint_callback)
        if callbacks:
            trainer_config['callbacks'] = callbacks
        
        # 添加 wandb logger
        if self.wandb and self.wandb.logger:
            trainer_config['logger'] = self.wandb.logger
        # 禁用 Lightning 默认日志目录（使用临时目录，不会在项目根目录创建 lightning_logs）
        trainer_config['default_root_dir'] = tempfile.gettempdir()
        
        
        trainer = Trainer(**trainer_config)
        
        # 打印优化配置摘要
        self._print_optimization_summary()
        
        start_time = arrow.Arrow.now().format('YYYY-MM-DD HH:mm:ss')
        print(f"================================================")
        print(f"Model: {self.model_class.__name__} Training Started at: {start_time}")

        import time
        train_start = time.time()
        
        # 训练（PyTorch Lightning 自动处理 checkpoint 恢复）
        trainer.fit(self.model, train_loader, val_loader, ckpt_path=ckpt_path)
        
        train_time = time.time() - train_start
        
        end_time = arrow.Arrow.now().format('YYYY-MM-DD HH:mm:ss')
        print(f"Model: {self.model_class.__name__} Training Ended at: {end_time}")
        
        spend_time = arrow.get(end_time) - arrow.get(start_time)
        print(f"Model: {self.model_class.__name__} Training Duration: {spend_time}")
        
        print(f"================================================")

        self.trained = True

        # 处理 wandb 训练结束
        if self.wandb:
            self.wandb.finish(train_time, self.checkpoint_callback)

        return self.model
    
    def _print_optimization_summary(self):
        """打印优化配置摘要"""
        is_windows = platform.system() == 'Windows'
        
        print("\n" + "="*60)
        print("🚀 训练优化配置")
        print("="*60)
        
        # 系统信息
        if is_windows:
            print(f"\n💻 系统: Windows (多进程数据加载已禁用)")
        
        # Checkpoint 信息
        if self.use_checkpoint:
            print(f"\n💾 Checkpoint:")
            print(f"  • 启用: True")
            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                print(f"  • 恢复自: {self.checkpoint_path}")
            else:
                from src.config.params import PROJECT_PATH
                print(f"  • 保存路径: {PROJECT_PATH}/out/checkpoints/{self.model_class.__name__}")
                print(f"  • 监控指标: {self.trainer_params.get('monitor', 'val_loss')}")
                print(f"  • 保存最优: Top-{self.trainer_params.get('save_top_k', 3)}")
        
        # 数据加载优化
        print("\n📦 数据加载:")
        num_workers = self.trainer_params.get('num_workers', 0 if is_windows else 8)
        print(f"  • num_workers: {num_workers}")
        if is_windows and num_workers == 0:
            print(f"    ⚠️  Windows系统默认禁用多进程，避免兼容性问题")
        print(f"  • pin_memory: {self.trainer_params.get('pin_memory', True)}")
        if num_workers > 0:
            print(f"  • persistent_workers: {self.trainer_params.get('persistent_workers', True)}")
        print(f"  • prefetch_factor: {self.trainer_params.get('prefetch_factor', 2 if num_workers > 0 else 'N/A')}")
        
        # 训练优化
        print("\n⚡ 训练配置:")
        precision = self.trainer_params.get('precision', '16-mixed')
        print(f"  • precision: {precision}")
        if precision == '16-mixed':
            print(f"    ✅ 混合精度训练已启用 (FP16+FP32)")
        
        if torch.cuda.is_available():
            matmul_precision = self.trainer_params.get('matmul_precision', 'high')
            print(f"  • tensor_cores: {matmul_precision} precision")
            print(f"    ✅ Tensor Cores 优化已启用 (RTX GPU)")
        
        print(f"  • batch_size: {self.trainer_params.get('batch_size', 20)}")
        
        # 模型编译
        if self.trainer_params.get('compile_model', False):
            print(f"\n🔧 模型编译:")
            print(f"  • 已启用: {self.trainer_params.get('compile_mode', 'default')} 模式")
        
        # Wandb 信息
        if self.wandb:
            print(f"\n📊 Wandb:")
            print(f"  • 已启用")
        
        print("="*60 + "\n")


class BasePrediction:
    """
    预测器基类 - 独立于训练器的预测功能
    
    支持从本地 checkpoint 或 wandb artifacts 加载模型进行预测
    
    参数:
        area: Area, 区域
        model_class: LightningModule, 模型类
        dataset_class: Dataset, 数据集类
        wandb_run_id: str, wandb run ID（可选，用于从 wandb 加载）
        wandb_version: str, wandb artifact version（可选，如 "latest" 或 "v0"）
        dataset_params: dict, 数据集参数
        model_params: dict, 模型参数
        use_wandb: bool, 是否使用 wandb 日志（默认: False）
        
    使用示例:
        # 从本地 checkpoint 加载
        predictor = BasePrediction(
            area=area,
            model_class=RGTransformer,
            dataset_class=OISSTMonthlyDataset,
            dataset_params={'seq_len': 2, 'resolution': 1},
            model_params={'width': 360, 'height': 160, ...}
        )
        result = predictor.predict(offset=520, plot=True)
        
        # 从 wandb 加载
        predictor = BasePrediction(
            area=area,
            model_class=RGTransformer,
            dataset_class=OISSTMonthlyDataset,
            wandb_run_id='2025-01-15-10-30',
            wandb_version='latest',  # 或 'v0', 'v1' 等
            dataset_params={'seq_len': 2, 'resolution': 1},
            model_params={'width': 360, 'height': 160, ...}
        )
        result = predictor.predict(offset=520, plot=True)
    """
    
    def __init__(self,
                 area: Area,
                 model_class=None,
                 dataset_class=None,
                 wandb_run_id: str = None,
                 wandb_version: str = 'latest',
                 dataset_params: dict = {},
                 model_params: dict = {},
                 use_wandb: bool = True):
        
        self.area = area
        self.model_class = model_class
        self.dataset_class = dataset_class
        self.wandb_run_id = wandb_run_id
        self.wandb_version = wandb_version
        self.dataset_params = dataset_params
        self.model_params = model_params
        
        self.model = None
        self.model_loaded = False
        
        # Wandb 配置（用于记录预测结果）
        if use_wandb:
            # 如果提供了 wandb_run_id，使用原来的 run ID（在原来的 run 上更新）
            # 否则创建新的 run
            if wandb_run_id:
                uid = wandb_run_id
            else:
                uid = arrow.now().format('YYYY-MM-DD-HH-mm')
            
            self.wandb = Wandb(
                uid=uid,
                model_class=model_class,
                dataset_class=dataset_class,
                area=area,
                model_params=model_params,
                dataset_params=dataset_params,
                trainer_params={},
                enabled=use_wandb
            )
        else:
            self.wandb = None
    
    def _load_model_from_wandb(self, run_id: str, version: str = 'latest'):
        """
        从 wandb artifacts 加载模型（使用官方 API 方法）
        
        参考: https://docs.wandb.ai/ref/python/api/run#logged_artifacts
        """
        from src.config.params import WANDB_PROJECT, WANDB_ENTITY, PROJECT_PATH
        
        print(f"📦 从 wandb 加载模型...")
        print(f"  • Run ID: {run_id}")
        print(f"  • Version: {version}")
        print(f"  • Project: {WANDB_PROJECT}")
        print(f"  • Entity: {WANDB_ENTITY}")
        
        # 使用 wandb API（官方方法）
        api = wandb.Api()
        
        # 构建完整的 artifact 路径：entity/project/artifact_name:version
        artifact_base_name = f"{self.model_class.__name__}_{self.wandb_run_id}"
        artifact_full_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{artifact_base_name}:{version}"

        # 构建本地缓存目录（使用项目目录下的 artifacts 文件夹）
        cache_dir = os.path.join(PROJECT_PATH, 'src', 'artifacts', f"{artifact_base_name}_{version}")
        
        # 检查本地缓存目录是否存在且包含 checkpoint 文件
        if os.path.exists(cache_dir):
            checkpoint_files = [f for f in os.listdir(cache_dir) if f.endswith('.ckpt')]
            if checkpoint_files:
                print(f"  • 使用本地缓存: {cache_dir}")
                download_dir = cache_dir
            else:
                # 目录存在但没有 checkpoint 文件，需要重新下载
                print(f"  • 查找 Artifact: {artifact_full_path}")
                artifact = api.artifact(artifact_full_path)
                # 下载到指定的缓存目录
                download_dir = artifact.download(root=cache_dir)
                print(f"  • 已下载到: {download_dir}")
        else:
            # 目录不存在，需要从 wandb 下载
            print(f"  • 查找 Artifact: {artifact_full_path}")
            artifact = api.artifact(artifact_full_path)
            # 下载到指定的缓存目录
            download_dir = artifact.download(root=cache_dir)
            print(f"  • 已下载到: {download_dir}")

        
        # 查找 checkpoint 文件（可能是 last.ckpt 或带 epoch 信息的文件名）
        checkpoint_files = [f for f in os.listdir(download_dir) if f.endswith('.ckpt')]
        
        if not checkpoint_files:
            raise FileNotFoundError(f"在下载的 artifact 中未找到 .ckpt 文件: {download_dir}")
        
        # 优先使用 last.ckpt，否则使用第一个找到的
        if 'last.ckpt' in checkpoint_files:
            checkpoint_path = os.path.join(download_dir, 'last.ckpt')
        else:
            checkpoint_path = os.path.join(download_dir, checkpoint_files[0])
            print(f"  • 使用找到的 checkpoint: {checkpoint_files[0]}")
        
        print(f"  • Checkpoint 路径: {checkpoint_path}")

        self.model = self.model_class.load_from_checkpoint(
            checkpoint_path, 
            strict=True,
            **self.model_params
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()
        self.model_loaded = True
    
    def _ensure_model_loaded(self):
        """确保模型已加载"""
        if self.model_loaded and self.model is not None:
            return
        
        # 优先使用 wandb 加载
        if self.wandb_run_id:
            self._load_model_from_wandb(self.wandb_run_id, self.wandb_version)
        else:
            raise ValueError("必须提供 wandb_run_id 来加载模型")
        
    def predict(self, offset: int, plot: bool = False) -> tuple:
        """
        预测
        
        :param offset: 数据偏移量
        :param plot: 是否绘制预测结果图
        :return: (input, output, pred_output, rmse, r2, ssta) 元组
        """
        # 确保模型已加载
        self._ensure_model_loaded()
        
        # 确定模型所在的设备
        device = next(self.model.parameters()).device
        
        dataset_params = {
            **self.dataset_params,
            'offset': offset,
        }
        
        pred_dataset = self.dataset_class(
            lon=self.area.lon,
            lat=self.area.lat,
            **dataset_params
        )
        
        pred_loader = DataLoader(pred_dataset, batch_size=1, shuffle=False)
        
        input, output = next(iter(pred_loader))
        ssta = pred_dataset.read_ssta(offset)
        
        # 将数据移动到模型所在的设备
        input = input.to(device)
        output = output.to(device)
        
        # 模型预测
        pred_output = self.model(input)
        
        # 转换为numpy之前，先移回CPU
        input = input.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        pred_output = pred_output.detach().cpu().numpy()
        
        input = input[0, 0, :, :]
        output = output[0, :, :]
        pred_output = pred_output[0, :, :]
        
        masked = np.isnan(output)
        pred_output[masked] = np.nan
        
        pred_diff = pred_output - output
        
        rmse = np.sqrt(np.nanmean((pred_diff) ** 2))
        r2 = 1 - np.nanmean((pred_diff) ** 2) / np.nanmean((output - np.nanmean(output)) ** 2)
        
        print(f"--------------------------------")
        print(f"Model: {self.model_class.__name__} Prediction RMSE: {rmse}")

        # 绘制模型可视化

        position_encoding = self.model.viz['position_encoding'].detach().cpu().numpy()
        sst_after_position_encoding = self.model.viz['sst_after_position_encoding'].detach().cpu().numpy()
        sst_after_attention = self.model.viz['sst_after_attention'].detach().cpu().numpy()
        sst_after_ffn = self.model.viz['sst_after_ffn'].detach().cpu().numpy()

        print(f"--------------------------------")
        print(f" 📊 Model: {self.model_class.__name__} Prediction Position Encoding:")
        print(f"position_encoding: {position_encoding.shape}")
        print(f"sst_after_position_encoding: {sst_after_position_encoding.shape}")
        print(f"sst_after_attention: {sst_after_attention.shape}")
        print(f"sst_after_ffn: {sst_after_ffn.shape}")
        print(f"--------------------------------")


        # 其他参数
        spatial_enc_scale = self.model.spatial_enc_scale.detach().cpu().numpy()
        harmonic_weights = self.model.spatial_pos_encoding.harmonic_weights.detach().cpu().numpy()
        spatial_bias = self.model.spatial_pos_encoding.spatial_bias.detach().cpu().numpy()

        print(f" 📊 Model: {self.model_class.__name__} Prediction Other Parameters:")
        print(f"spatial_enc_scale: {spatial_enc_scale}")
        print(f"harmonic_weights: {harmonic_weights}")
        print(f"spatial_bias: {spatial_bias}")
        print(f"--------------------------------")
        
        if plot:
            resolution = self.dataset_params.get('resolution', 1)
            
            # 绘制图像并获取 figure 对象
            ax_nino = plot_nino(ssta, step=resolution)
            ax_sst = plot_sst(pred_output, self.area.lon, self.area.lat, step=resolution)
            ax_diff = plot_sst_diff(pred_diff, self.area.lon, self.area.lat, step=resolution)
            
            # 将图像记录到 wandb（如果 wandb 可用）
            if self.wandb:
                # 确保 wandb logger 已初始化
                if not self.wandb.logger:
                    self.wandb.init_for_prediction(self.model)
                
                if self.wandb.logger:
                    self.wandb.log_prediction_images(
                        offset=offset,
                        rmse=rmse,
                        r2=r2,
                        nino_fig=ax_nino.figure,
                        sst_fig=ax_sst.figure,
                        diff_fig=ax_diff.figure
                    )
                
                # 在记录后再关闭图像
                plt.close(ax_nino.figure)
                plt.close(ax_sst.figure)
                plt.close(ax_diff.figure)
            
        return input, output, pred_output, rmse, r2, ssta