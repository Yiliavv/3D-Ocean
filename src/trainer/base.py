import arrow
import torch
import numpy as np
import platform
import os
import tempfile
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Subset
from typing import List, Optional

from src.plot.sst import plot_sst, plot_sst_diff, plot_nino

from src.config.area import Area
from src.trainer.plugins import PluginManager, WandbPlugin, BasePlugin

class BaseTrainer:
    """
    训练器基类 - 集成性能优化和 Checkpoint 机制
    
    参数:
        title: str, 模型名称
        uid: str, 训练器唯一标识
        area: Area, 区域
        model_class: LightningModule, 模型类
        checkpoint_path: str, checkpoint 路径 (用于恢复训练)
        dataset_params: dict, 数据集参数
        trainer_params: dict, 训练参数
        model_params: dict, 模型参数
        use_checkpoint: bool, 是否使用 checkpoint 机制 (默认: True)
        use_wandb: bool, 是否使用 wandb 日志 (默认: True)
        
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
        persistent_workers: bool, 是否保持工作进程 (默认: True)
        prefetch_factor: int, 预取因子 (默认: 2)
        precision: str, 训练精度 (默认: "16-mixed", 可选: "32", "bf16-mixed")
        accumulate_grad_batches: int, 梯度累积步数 (默认: 1)
        gradient_clip_val: float, 梯度裁剪值 (默认: None)
        compile_model: bool, 是否编译模型-PyTorch2.0+ (默认: False)
        compile_mode: str, 编译模式 (默认: "default")
        
    使用示例:
        # 第一次训练
        trainer = BaseTrainer(
            title='SST_Model',
            uid='run_001',
            area=area,
            model_class=YourModel,
            dataset_class=YourDataset,
            trainer_params={'epochs': 100},
            use_checkpoint=True
        )
        model = trainer.train()
        
        # 从 checkpoint 恢复并继续训练
        trainer = BaseTrainer(
            title='SST_Model',
            uid='run_002',
            area=area,
            model_class=YourModel,
            dataset_class=YourDataset,
            checkpoint_path=f'{PROJECT_PATH}/out/checkpoints/SST_Model/last.ckpt',  # 加载 checkpoint（使用PROJECT_PATH）
            trainer_params={'epochs': 150},  # 可以修改超参数
            use_checkpoint=True
        )
        model = trainer.train()

    """
    
    def __init__(self,
                 title: str,
                 uid: str,
                 area: Area,
                 model_class = None,
                 dataset_class = None,
                 checkpoint_path: str = None,  # checkpoint 路径
                 dataset_params: dict = {},
                 trainer_params: dict = {},
                 model_params: dict = {},
                 use_checkpoint: bool = True,  # 是否使用 checkpoint
                 use_wandb: bool = False,  # 是否使用 wandb（向后兼容，将自动注册 WandbPlugin）
                 plugins: Optional[List[BasePlugin]] = None):  # 自定义插件列表
        
        self.trainer_uid = uid

        self.title = title
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
        
        # 插件系统
        self.plugin_manager = PluginManager()
        
        # 向后兼容：如果 use_wandb=True，自动注册 WandbPlugin
        if use_wandb:
            wandb_plugin = WandbPlugin(enabled=True)
            self.plugin_manager.register(wandb_plugin)
        
        # 注册用户自定义插件
        if plugins:
            self.plugin_manager.register_all(plugins)
        
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
        persistent_workers = self.trainer_params.get('persistent_workers', True) and num_workers > 0
        prefetch_factor = self.trainer_params.get('prefetch_factor', 2)
        
        dataloader_kwargs = {
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
        }
        
        if num_workers > 0:
            dataloader_kwargs['persistent_workers'] = persistent_workers
            if prefetch_factor:
                dataloader_kwargs['prefetch_factor'] = prefetch_factor
        
        train_loader = DataLoader(train_set, **dataloader_kwargs)
        
        # 验证集使用较少的workers
        val_dataloader_kwargs = dataloader_kwargs.copy()
        val_dataloader_kwargs['num_workers'] = max(1, num_workers // 2)
        val_loader = DataLoader(val_set, **val_dataloader_kwargs)
        
        return train_loader, val_loader
        
    @staticmethod
    def find_latest_checkpoint(title: str) -> str:
        """
        查找指定模型的最新 checkpoint
        
        Args:
            title: 模型名称
            
        Returns:
            checkpoint_path: 最新的 checkpoint 路径，如果不存在则返回 None
        """
        from src.config.params import PROJECT_PATH
        checkpoint_dir = f'{PROJECT_PATH}/out/checkpoints/{title}'
        
        if not os.path.exists(checkpoint_dir):
            return None
        
        # 查找 last.ckpt（总是最新的）
        last_ckpt = os.path.join(checkpoint_dir, 'last.ckpt')
        if os.path.exists(last_ckpt):
            return last_ckpt
        
        # 如果没有 last.ckpt，查找最新的 epoch checkpoint
        ckpt_files = []
        for f in os.listdir(checkpoint_dir):
            if f.endswith('.ckpt'):
                ckpt_path = os.path.join(checkpoint_dir, f)
                ckpt_files.append((os.path.getmtime(ckpt_path), ckpt_path))
        
        if ckpt_files:
            # 按修改时间排序，返回最新的
            ckpt_files.sort(reverse=True)
            return ckpt_files[0][1]
        
        return None
    
    def _create_checkpoint_callback(self):
        """创建 checkpoint callback"""
        if not self.use_checkpoint:
            return None
        
        # checkpoint 保存目录（使用项目根目录的绝对路径）
        from src.config.params import PROJECT_PATH
        checkpoint_dir = f'{PROJECT_PATH}/out/checkpoints/{self.title}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 配置 checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='{epoch}-{val_loss:.4f}',
            monitor=self.trainer_params.get('monitor', 'val_loss'),
            mode=self.trainer_params.get('mode', 'min'),
            save_top_k=self.trainer_params.get('save_top_k', 3),
            save_last=True,  # 保存最后一个 checkpoint (last.ckpt)
            verbose=False,  # 关闭保存时的详细打印
        )
        
        print(f"\n💾 Checkpoint 配置:")
        print(f"  • 保存路径: {checkpoint_dir}")
        print(f"  • 监控指标: {checkpoint_callback.monitor}")
        print(f"  • 保存最优: Top-{checkpoint_callback.save_top_k}")
        print(f"  • 保存最新: True (last.ckpt)")
        
        # 检查是否有已有的 checkpoint
        latest_ckpt = self.find_latest_checkpoint(self.title)
        if latest_ckpt:
            print(f"  • 发现已有 checkpoint: {latest_ckpt}")
            print(f"    提示: 设置 checkpoint_path='{latest_ckpt}' 可恢复训练\n")
        else:
            print()
        
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
        
        # 判断是否从 checkpoint 恢复
        resume_from_checkpoint = None
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            resume_from_checkpoint = self.checkpoint_path
            print(f"\n🔄 从 checkpoint 恢复训练: {self.checkpoint_path}\n")
        
        if not resume_from_checkpoint:
            # 创建新模型
            self.model = self.model_class(
                **self.model_params
            )
            
            # PyTorch 2.0+ 模型编译
            if self.trainer_params.get('compile_model', False):
                if hasattr(torch, 'compile'):
                    compile_mode = self.trainer_params.get('compile_mode', 'default')
                    print(f"🚀 编译模型 (模式: {compile_mode})...")
                    self.model = torch.compile(self.model, mode=compile_mode)
                else:
                    print("⚠️  PyTorch版本 < 2.0, 模型编译不可用")
        elif resume_from_checkpoint:
            # 从 checkpoint 加载时，先创建模型结构
            self.model = self.model_class(
                **self.model_params
            )
            
            # 如果需要修改学习率等超参数，在这里处理
            # 注意：这些修改会在 checkpoint 加载后生效
            if 'learning_rate' in self.model_params:
                print(f"⚙️  设置新的学习率: {self.model_params['learning_rate']}")
                self.model.learning_rate = self.model_params['learning_rate']
        
        epochs = self.trainer_params.get('epochs', 100)
        
        # 调用插件的 on_train_start 钩子
        self.plugin_manager.on_train_start(
            self, self.model,
            dataset=dataset,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
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
        # 添加插件提供的 callbacks
        plugin_callbacks = self.plugin_manager.get_all_lightning_callbacks()
        callbacks.extend(plugin_callbacks)
        if callbacks:
            trainer_config['callbacks'] = callbacks
        
        # 添加插件提供的 loggers
        loggers = self.plugin_manager.get_all_lightning_loggers()
        if loggers:
            # 如果只有一个 logger，直接使用；否则使用列表
            trainer_config['logger'] = loggers[0] if len(loggers) == 1 else loggers
        # 禁用 Lightning 默认日志目录（使用临时目录，不会在项目根目录创建 lightning_logs）
        trainer_config['default_root_dir'] = tempfile.gettempdir()
        
        # 梯度累积
        accumulate_grad_batches = self.trainer_params.get('accumulate_grad_batches', 1)
        if accumulate_grad_batches > 1:
            trainer_config['accumulate_grad_batches'] = accumulate_grad_batches
        
        # 梯度裁剪
        gradient_clip_val = self.trainer_params.get('gradient_clip_val', None)
        if gradient_clip_val:
            trainer_config['gradient_clip_val'] = gradient_clip_val
            trainer_config['gradient_clip_algorithm'] = self.trainer_params.get(
                'gradient_clip_algorithm', 'norm'
            )

        trainer = Trainer(**trainer_config)
        
        # 打印优化配置摘要
        self._print_optimization_summary(accumulate_grad_batches)
        
        start_time = arrow.Arrow.now().format('YYYY-MM-DD HH:mm:ss')
        print(f"================================================")
        print(f"Model: {self.model_class.__name__} Training Started at: {start_time}")

        import time
        train_start = time.time()
        
        # 从 checkpoint 恢复训练
        if resume_from_checkpoint:
            trainer.fit(self.model, train_loader, val_loader, ckpt_path=resume_from_checkpoint)
        else:
            trainer.fit(self.model, train_loader, val_loader)
        
        train_time = time.time() - train_start
        
        end_time = arrow.Arrow.now().format('YYYY-MM-DD HH:mm:ss')
        print(f"Model: {self.model_class.__name__} Training Ended at: {end_time}")
        
        spend_time = arrow.get(end_time) - arrow.get(start_time)
        print(f"Model: {self.model_class.__name__} Training Duration: {spend_time}")
        
        # 计算吞吐量
        total_samples = len(train_loader.dataset) * epochs
        throughput = total_samples / train_time if train_time > 0 else 0
        self.total_samples = total_samples  # 保存供插件使用
        print(f"Training Throughput: {throughput:.2f} samples/second")
        print(f"================================================")

        self.trained = True

        # 调用插件的 on_train_end 钩子
        self.plugin_manager.on_train_end(
            self, self.model,
            train_time=train_time,
            throughput=throughput,
            total_samples=total_samples
        )

        return self.model
    
    def _print_optimization_summary(self, accumulate_grad_batches):
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
                print(f"  • 保存路径: {PROJECT_PATH}/out/checkpoints/{self.title}")
                print(f"  • 监控指标: {self.trainer_params.get('monitor', 'val_loss')}")
                print(f"  • 保存最优: Top-{self.trainer_params.get('save_top_k', 3)}")
        
        # 数据加载优化
        print("\n📦 数据加载:")
        num_workers = self.trainer_params.get('num_workers', 0 if is_windows else 8)
        print(f"  • num_workers: {num_workers}")
        if is_windows and num_workers == 0:
            print(f"    ⚠️  Windows系统默认禁用多进程，避免兼容性问题")
        print(f"  • pin_memory: {self.trainer_params.get('pin_memory', True)}")
        print(f"  • persistent_workers: {self.trainer_params.get('persistent_workers', True) and num_workers > 0}")
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
        
        if accumulate_grad_batches > 1:
            effective_bs = self.trainer_params.get('batch_size', 20) * accumulate_grad_batches
            print(f"  • gradient_accumulation: {accumulate_grad_batches} (有效batch_size: {effective_bs})")
        
        if self.trainer_params.get('gradient_clip_val'):
            print(f"  • gradient_clipping: {self.trainer_params.get('gradient_clip_val')}")
        
        # 模型编译
        if self.trainer_params.get('compile_model', False):
            print(f"\n🔧 模型编译:")
            print(f"  • 已启用: {self.trainer_params.get('compile_mode', 'default')} 模式")
        
        # 插件信息
        enabled_plugins = self.plugin_manager.get_enabled_plugins()
        if enabled_plugins:
            print(f"\n🔌 插件:")
            for plugin in enabled_plugins:
                print(f"  • {plugin.name}: 已启用")
        
        print("="*60 + "\n")

    def predict(self, offset: int, plot: bool = False) -> tuple:
        """
        预测
        
        :param offset: 数据偏移量
        :param plot: 是否绘制预测结果图
        :return: (input, output, pred_output, rmse, r2, ssta) 元组
        """
        # 加载模型（从 checkpoint）
        if not self.trained:
            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                # 从 checkpoint 加载（适用于 Lightning 模型）
                print(f"📦 从 checkpoint 加载模型: {self.checkpoint_path}")
                self.model = self.model_class.load_from_checkpoint(
                    self.checkpoint_path,
                    **self.model_params
                )
                # 确保模型在正确的设备上
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model = self.model.to(device)
                self.model.eval()
                self.trained = True
            else:
                raise ValueError(
                    f'无已训练模型。请提供 checkpoint_path: {self.checkpoint_path}'
                )
            
        if not self.model:
            raise ValueError('模型加载失败')
        
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
        
        pred_output = self.model(input)
        
        # 转换为numpy之前，先移回CPU
        input = input.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        pred_output = pred_output.detach().cpu().numpy()
        
        input = input[0, 0, 0, :, :]
        output = output[0, 0, :, :]
        pred_output = pred_output[0, 0, :, :]
        
        masked = np.isnan(output)
        pred_output[masked] = np.nan
        
        pred_diff = pred_output - output
        
        rmse = np.sqrt(np.nanmean((pred_diff) ** 2))
        r2 = 1 - np.nanmean((pred_diff) ** 2) / np.nanmean((output - np.nanmean(output)) ** 2)
        
        print(f"--------------------------------")
        
        print(f"Model: {self.model_class.__name__} Prediction RMSE: {rmse}")
        
        if plot:
            resolution = self.dataset_params.get('resolution', 1)
            plot_nino(ssta, step=resolution)
            plot_sst(pred_output, self.area.lon, self.area.lat, step=resolution)
            plot_sst_diff(pred_diff, self.area.lon, self.area.lat, step=resolution)
            
        return input, output, pred_output, rmse, r2, ssta
    
    def evaluate(self, offsets: list, plot: bool = False) -> dict:
        """
        批量评估模型在多个时间点的表现
        
        :param offsets: 数据偏移量列表
        :param plot: 是否绘制每个时间点的预测结果
        :return: 包含所有评估结果的字典
        """
        results = {
            'offsets': offsets,
            'rmse': [],
            'r2': [],
            'details': []
        }
        
        print(f"\n📊 开始评估 {len(offsets)} 个时间点...")
        print("=" * 60)
        
        for i, offset in enumerate(offsets):
            print(f"\n[{i+1}/{len(offsets)}] 评估 offset={offset}")
            try:
                input, output, pred_output, rmse, r2, ssta = self.predict(
                    offset=offset, 
                    plot=plot
                )
                results['rmse'].append(rmse)
                results['r2'].append(r2)
                results['details'].append({
                    'offset': offset,
                    'rmse': rmse,
                    'r2': r2
                })
            except Exception as e:
                print(f"⚠️  offset={offset} 评估失败: {str(e)}")
                results['rmse'].append(np.nan)
                results['r2'].append(np.nan)
        
        # 计算统计信息
        valid_rmse = [r for r in results['rmse'] if not np.isnan(r)]
        valid_r2 = [r for r in results['r2'] if not np.isnan(r)]
        
        if valid_rmse:
            results['mean_rmse'] = np.mean(valid_rmse)
            results['std_rmse'] = np.std(valid_rmse)
            results['min_rmse'] = np.min(valid_rmse)
            results['max_rmse'] = np.max(valid_rmse)
        
        if valid_r2:
            results['mean_r2'] = np.mean(valid_r2)
            results['std_r2'] = np.std(valid_r2)
            results['min_r2'] = np.min(valid_r2)
            results['max_r2'] = np.max(valid_r2)
        
        print("\n" + "=" * 60)
        print("📊 评估摘要:")
        if valid_rmse:
            print(f"  RMSE: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}")
            print(f"   范围: [{results['min_rmse']:.4f}, {results['max_rmse']:.4f}]")
        if valid_r2:
            print(f"  R²:   {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
            print(f"   范围: [{results['min_r2']:.4f}, {results['max_r2']:.4f}]")
        print("=" * 60 + "\n")
        
        return results
