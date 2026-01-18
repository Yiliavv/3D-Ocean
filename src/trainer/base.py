import os
import torch
import wandb
import arrow
import tempfile
import platform
import numpy as np
import matplotlib.pyplot as plt

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, Subset

from src.plot.sst import plot_sst, plot_sst_diff, plot_nino, plot_sequence, plot_attention
from src.config.area import Area
from src.config.params import CHECKPOINT_SAVE_PATH
from src.trainer.wandb import Wandb

class BaseTrainer:
    """
    训练器基类 - 集成性能优化和 Checkpoint 机制
    
    参数:
        area: Area, 区域
        model_class: LightningModule, 模型类
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
            trainer_params={'epochs': 150},  # 可以修改超参数
            use_checkpoint=True
        )
        model = trainer.train()

    """
    
    def __init__(self,
                 area: Area,
                 model_class = None,
                 dataset_class = None,
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
        num_workers = self.trainer_params.get('num_workers', 8)
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
        
        os.makedirs(CHECKPOINT_SAVE_PATH, exist_ok=True)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath= f"{CHECKPOINT_SAVE_PATH}/{self.trainer_uid}",
            filename=f'{self.model_class.__name__}',
            monitor=self.trainer_params.get('monitor', 'val_loss'),
            mode=self.trainer_params.get('mode', 'min'),
            save_top_k=self.trainer_params.get('save_top_k', 1),
            save_last=False,
            verbose=False,
        )
        
        print(f"\n💾 Checkpoint: {CHECKPOINT_SAVE_PATH}/{self.trainer_uid}/{f'{self.model_class.__name__}'}\n")
        
        return checkpoint_callback
    
    def train(self, run_id: str = None):

        run_id = run_id or self.trainer_uid

        ckpt_path = f"{CHECKPOINT_SAVE_PATH}/{run_id}/{self.model_class.__name__}"

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
        
        # 梯度累积（模拟大批量训练）
        accumulate_grad_batches = self.trainer_params.get('accumulate_grad_batches', 1)
        if accumulate_grad_batches > 1:
            trainer_config['accumulate_grad_batches'] = accumulate_grad_batches
            effective_batch = self.trainer_params.get('batch_size', 8) * accumulate_grad_batches
            print(f"📊 梯度累积: {accumulate_grad_batches}x, 有效批量大小: {effective_batch}")
        
        # 梯度裁剪（防止梯度爆炸）
        gradient_clip_val = self.trainer_params.get('gradient_clip_val', 1.0)
        if gradient_clip_val:
            trainer_config['gradient_clip_val'] = gradient_clip_val
        
        # 添加 callbacks
        callbacks = []
        if self.checkpoint_callback:
            callbacks.append(self.checkpoint_callback)
        
        # 早停策略
        early_stopping_patience = self.trainer_params.get('early_stopping_patience', None)
        if early_stopping_patience:
            early_stopping = EarlyStopping(
                monitor=self.trainer_params.get('monitor', 'val_loss'),
                patience=early_stopping_patience,
                mode=self.trainer_params.get('mode', 'min'),
                verbose=True
            )
            callbacks.append(early_stopping)
            print(f"⏱️ 早停策略: {early_stopping_patience} 轮无改善则停止")
        
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
        if os.path.exists(ckpt_path):
            print(f"\n🔄 使用 checkpoint: {ckpt_path}\n")
            trainer.fit(self.model, train_loader, val_loader, ckpt_path=ckpt_path)
        else:
            print(f"\n🔄 从头开始训练\n")
            trainer.fit(self.model, train_loader, val_loader)
        
        train_time = time.time() - train_start
        
        end_time = arrow.Arrow.now().format('YYYY-MM-DD HH:mm:ss')
        print(f"Model: {self.model_class.__name__} Training Ended at: {end_time}")
        
        spend_time = arrow.get(end_time) - arrow.get(start_time)
        print(f"Model: {self.model_class.__name__} Training Duration: {spend_time}")
        
        print(f"================================================")

        self.trained = True

        # 处理 wandb 训练结束
        if self.wandb:
            # 默认训练结束后关闭 wandb run，除非在 trainer_params 中显式设置为 False
            close_run = self.trainer_params.get('close_wandb', True)
            self.wandb.finish(train_time, self.checkpoint_callback, close_run=close_run)

        return self.model
    
    def _print_optimization_summary(self):
        """打印优化配置摘要"""
        print("\n" + "="*60)
        print("🚀 训练优化配置（V3 精度优化版）")
        print("="*60)
        
        # 系统信息
        print(f"\n💻 系统: {platform.system()}")
        
        # GPU 信息
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Checkpoint 信息
        if self.use_checkpoint:
            print(f"\n💾 Checkpoint:")
            print(f"  • 启用: True")
            print(f"  • 监控指标: {self.trainer_params.get('monitor', 'val_loss')}")
            print(f"  • 保存最优: Top-{self.trainer_params.get('save_top_k', 3)}")
        
        # 数据加载优化
        print("\n📦 数据加载:")
        num_workers = self.trainer_params.get('num_workers', 8)
        batch_size = self.trainer_params.get('batch_size', 8)
        print(f"  • batch_size: {batch_size}")
        print(f"  • num_workers: {num_workers}")
        print(f"  • pin_memory: {self.trainer_params.get('pin_memory', True)}")
        if num_workers > 0:
            print(f"  • persistent_workers: {self.trainer_params.get('persistent_workers', True)}")
        
        # 梯度累积
        accumulate = self.trainer_params.get('accumulate_grad_batches', 1)
        if accumulate > 1:
            print(f"\n📈 梯度累积:")
            print(f"  • 累积步数: {accumulate}")
            print(f"  • 有效批量: {batch_size * accumulate}")
        
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
        
        # 梯度裁剪
        grad_clip = self.trainer_params.get('gradient_clip_val', 1.0)
        if grad_clip:
            print(f"  • gradient_clip: {grad_clip}")
        
        # 模型参数信息
        if hasattr(self, 'model_params'):
            print(f"\n🧠 模型配置:")
            print(f"  • d_model: {self.model_params.get('d_model', 'N/A')}")
            print(f"  • num_heads: {self.model_params.get('num_heads', 'N/A')}")
            print(f"  • dim_feedforward: {self.model_params.get('dim_feedforward', 'N/A')}")
            print(f"  • ffn_activation: {self.model_params.get('ffn_activation', 'gelu')}")
            print(f"  • use_se_attention: {self.model_params.get('use_se_attention', False)}")
            print(f"  • loss_type: {self.model_params.get('loss_type', 'mse')}")
            
            if self.model_params.get('use_gradient_checkpointing', False):
                print(f"  • gradient_checkpointing: ✅ 已启用")
        
        # 学习率配置
        print(f"\n📉 学习率配置:")
        print(f"  • initial_lr: {self.model_params.get('learning_rate', 1e-4)}")
        if self.model_params.get('use_lr_scheduler', False):
            print(f"  • scheduler: CosineAnnealingLR")
            print(f"  • warmup_epochs: {self.model_params.get('warmup_epochs', 10)}")
            print(f"  • min_lr: {self.model_params.get('min_lr', 1e-6)}")
        
        # 早停策略
        patience = self.trainer_params.get('early_stopping_patience', None)
        if patience:
            print(f"\n⏱️ 早停策略:")
            print(f"  • patience: {patience} epochs")
        
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
    
    优先从本地 checkpoint 加载模型，如果本地不存在则尝试从 wandb artifacts 加载
    
    参数:
        area: Area, 区域
        model_class: LightningModule, 模型类
        dataset_class: Dataset, 数据集类
        run_id: str, checkpoint 的 run ID（用于定位本地 checkpoint 或 wandb artifact）
        wandb_version: str, wandb artifact version（可选，如 "latest" 或 "v0"）
        dataset_params: dict, 数据集参数
        model_params: dict, 模型参数
        use_wandb: bool, 是否使用 wandb 日志（默认: False）
        
    使用示例:
        # 从本地 checkpoint 加载（优先）
        predictor = BasePrediction(
            area=area,
            model_class=RGTransformer,
            dataset_class=OISSTMonthlyDataset,
            run_id='2025-01-15-10-30',  # out/checkpoints/2025-01-15-10-30/
            dataset_params={'seq_len': 2, 'resolution': 1},
            model_params={'width': 360, 'height': 160, ...}
        )
        result = predictor.predict(offset=520, plot=True)
        
        # 如果本地不存在，会自动尝试从 wandb 加载
        predictor = BasePrediction(
            area=area,
            model_class=RGTransformer,
            dataset_class=OISSTMonthlyDataset,
            run_id='2025-01-15-10-30',
            wandb_version='latest',  # 或 'v0', 'v1' 等
            dataset_params={'seq_len': 2, 'resolution': 1},
            model_params={'width': 360, 'height': 160, ...},
            use_wandb=True
        )
        result = predictor.predict(offset=520, plot=True)
    """
    
    def __init__(self,
                 area: Area,
                 model_class=None,
                 dataset_class=None,
                 run_id: str = None,
                 wandb_version: str = 'latest',
                 dataset_params: dict = {},
                 model_params: dict = {},
                 use_wandb: bool = True):
        
        self.area = area
        self.model_class = model_class
        self.dataset_class = dataset_class
        self.run_id = run_id
        self.wandb_version = wandb_version
        self.dataset_params = dataset_params
        self.model_params = model_params
        
        self.model = None
        self.model_loaded = False
        
        # Wandb 配置（用于记录预测结果）
        if use_wandb:
            # 如果提供了 run_id，使用该 ID（在原来的 run 上更新）
            # 否则创建新的 run
            uid = run_id if run_id else arrow.now().format('YYYY-MM-DD-HH-mm')
            
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
    
    def _load_model_from_local(self, run_id: str) -> bool:
        """
        从本地 checkpoint 加载模型
        
        参数:
            run_id: checkpoint 的 run ID
            
        返回:
            bool: 是否加载成功
        """
        checkpoint_file = f"{CHECKPOINT_SAVE_PATH}/{run_id}/{self.model_class.__name__}.ckpt"
        
        if not os.path.exists(checkpoint_file):
            return False
        
        print(f"📦 从本地 checkpoint 加载模型...")
        print(f"  • Run ID: {run_id}")
        print(f"  • 路径: {checkpoint_file}")
        
        # 先创建模型实例（不加载权重），用于初始化延迟初始化的层
        temp_model = self.model_class(**self.model_params)
        
        # 如果模型有 RGAttention，需要先初始化投影层
        if hasattr(temp_model, 'attention') and hasattr(temp_model.attention, '_init_projections'):
            width = self.model_params.get('width', 360)
            height = self.model_params.get('height', 160)
            seq_len = self.model_params.get('seq_len', 2)
            
            dummy_input = torch.zeros(1, seq_len - 1, width, height)
            try:
                _ = temp_model.attention(dummy_input)
                print(f"  • 已初始化延迟投影层")
            except:
                pass
        
        # 加载 checkpoint
        self.model = self.model_class.load_from_checkpoint(
            checkpoint_file, 
            strict=False,
            **self.model_params
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()
        self.model_loaded = True
        
        print(f"  ✅ 本地 checkpoint 加载成功")
        return True
    
    def _load_model_from_wandb(self, run_id: str, version: str = 'latest'):
        """
        从 wandb artifacts 加载模型（使用官方 API 方法）
        
        参考: https://docs.wandb.ai/ref/python/api/run#logged_artifacts
        """
        from src.config.params import WANDB_PROJECT, WANDB_ENTITY
        
        print(f"📦 从 wandb 加载模型...")
        print(f"  • Run ID: {run_id}")
        print(f"  • Version: {version}")
        print(f"  • Project: {WANDB_PROJECT}")
        print(f"  • Entity: {WANDB_ENTITY}")
        
        # 使用 wandb API（官方方法）
        api = wandb.Api()

        # 构建完整的 artifact 路径：entity/project/artifact_name:version
        artifact_base_name = f"{self.model_class.__name__}_{run_id}"
        artifact_full_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{artifact_base_name}:{version}"

        # 构建本地缓存目录
        cache_dir = os.path.join(CHECKPOINT_SAVE_PATH, run_id)
        checkpoint_file = f"{cache_dir}/{self.model_class.__name__}.ckpt"
        
        if os.path.exists(checkpoint_file):
            print(f"  • 使用本地缓存: {checkpoint_file}")
        else:
            print(f"  • 查找 Artifact: {artifact_full_path}")
            artifact = api.artifact(artifact_full_path)
            
            os.makedirs(cache_dir, exist_ok=True)
            artifact.download(root=cache_dir)
            
            ckpt_files = [f for f in os.listdir(cache_dir) if f.endswith('.ckpt')]
            
            if not ckpt_files:
                raise FileNotFoundError(f"在下载的 artifact 中未找到 .ckpt 文件: {cache_dir}")
            
            downloaded_ckpt = os.path.join(cache_dir, ckpt_files[0])

            if downloaded_ckpt != checkpoint_file:
                print(f"  • 重命名文件: {ckpt_files[0]} -> {self.model_class.__name__}.ckpt")
                os.rename(downloaded_ckpt, checkpoint_file)
        
        # 先创建模型实例（不加载权重），用于初始化延迟初始化的层
        temp_model = self.model_class(**self.model_params)
        
        # 如果模型有 RGAttention，需要先初始化投影层
        if hasattr(temp_model, 'attention') and hasattr(temp_model.attention, '_init_projections'):
            width = self.model_params.get('width', 360)
            height = self.model_params.get('height', 160)
            seq_len = self.model_params.get('seq_len', 2)
            
            dummy_input = torch.zeros(1, seq_len - 1, width, height)
            try:
                _ = temp_model.attention(dummy_input)
                print(f"  • 已初始化延迟投影层")
            except:
                pass
        
        # 加载 checkpoint
        self.model = self.model_class.load_from_checkpoint(
            checkpoint_file, 
            strict=False,
            **self.model_params
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()
        self.model_loaded = True
        
        print(f"  ✅ wandb artifact 加载成功")
    
    def _ensure_model_loaded(self):
        """
        确保模型已加载
        
        加载优先级：
        1. 本地 checkpoint (out/checkpoints/{run_id}/)
        2. wandb artifacts (如果 use_wandb=True)
        """
        if self.model_loaded and self.model is not None:
            return
        
        if not self.run_id:
            raise ValueError("必须提供 run_id 来加载模型")
        
        # 优先从本地加载
        if self._load_model_from_local(self.run_id):
            return
        
        # 本地不存在，尝试从 wandb 加载
        if self.wandb:
            print(f"  ⚠️ 本地 checkpoint 不存在，尝试从 wandb 加载...")
            self._load_model_from_wandb(self.run_id, self.wandb_version)
        else:
            raise FileNotFoundError(
                f"本地 checkpoint 不存在: {CHECKPOINT_SAVE_PATH}/{self.run_id}/{self.model_class.__name__}.ckpt\n"
                f"如需从 wandb 加载，请设置 use_wandb=True"
            )
        
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
        
        if plot:
            resolution = self.dataset_params.get('resolution', 1)
            
            # 绘制图像并获取 figure 对象
            ax_nino = plot_nino(ssta, step=resolution)
            ax_sst = plot_sst(pred_output, self.area.lon, self.area.lat, step=resolution)
            ax_diff = plot_sst_diff(pred_diff, self.area.lon, self.area.lat, step=resolution)

            # plot_attention(position_encoding, self.area.lon, self.area.lat, step=resolution, title='Position Encoding')
            # plot_sequence(x_normed, self.area.lon, self.area.lat, step=resolution, title='X Normed', plot_type='attention')
            # plot_sequence(attention_out, self.area.lon, self.area.lat, step=resolution, title='Attention out', plot_type='attention')
            # plot_sequence(ffn_out, self.area.lon, self.area.lat, step=resolution, title='FFN Output', plot_type='ffn')

            
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