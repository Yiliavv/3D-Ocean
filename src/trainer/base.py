import arrow
import torch
import numpy as np
import platform
from torch import save, load
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Subset
import wandb

from src.plot.sst import plot_sst, plot_sst_diff, plot_2d_kernel_density, plot_nino

from src.config.area import Area
from src.config.params import WANDB_PROJECT, WANDB_ENTITY

class BaseTrainer:
    """
    训练器基类 - 集成性能优化
    
    参数:
        title: str, 模型名称
        uid: str, 训练器唯一标识
        area: Area, 区域
        model_class: LightningModule, 模型类
        save_path: str, 保存路径
        pre_model: LightningModule, 预训练模型
        dataset_params: dict, 数据集参数
        trainer_params: dict, 训练参数
        model_params: dict, 模型参数
        
    dataset_params:
        seq_len: int, 序列长度
        offset: int, 偏移量
        resolution: float, 分辨率
        
    trainer_params:
        epochs: int, 训练轮数
        batch_size: int, 批量大小
        split_ratio: list, 训练集和验证集的分割比例
        
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

    """
    
    def __init__(self,
                 title: str,
                 uid: str,
                 area: Area,
                 model_class = None,
                 dataset_class = None,
                 save_path: str = None,
                 pre_model: bool = False,
                 dataset_params: dict = {},
                 trainer_params: dict = {},
                 model_params: dict = {},
                 use_wandb: bool = True):
        
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
        
        # 保存路径和预训练模型
        self.save_path = save_path
        self.pre_model = pre_model
        
        # wandb 配置
        self.use_wandb = use_wandb
        self.wandb_logger = None
        
        # 模型
        if pre_model:
            self.model = load(self.save_path, weights_only=False)
            self.trained = True
        else:
            self.model = None
    
        self.trained = False
    
    def split(self, dataset):
        split_ratio = self.trainer_params.get('split_ratio', [0.8, 0.2])
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
        
        if not self.pre_model:
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
        
        epochs = self.trainer_params.get('epochs', 100)
        
        # 初始化 wandb logger
        if self.use_wandb:
            self.wandb_logger = self._init_wandb_logger()
        
        # 优化的Trainer配置
        trainer_config = {
            'max_epochs': epochs,
            'accelerator': 'gpu',
            'enable_checkpointing': False,
            'num_sanity_val_steps': 0,
            'precision': self.trainer_params.get('precision', '16-mixed'),
        }
        
        # 添加 wandb logger
        if self.wandb_logger:
            trainer_config['logger'] = self.wandb_logger
        
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
        
        trainer.fit(self.model, train_loader, val_loader)
        
        train_time = time.time() - train_start
        
        end_time = arrow.Arrow.now().format('YYYY-MM-DD HH:mm:ss')
        print(f"Model: {self.model_class.__name__} Training Ended at: {end_time}")
        
        spend_time = arrow.get(end_time) - arrow.get(start_time)
        print(f"Model: {self.model_class.__name__} Training Duration: {spend_time}")
        
        # 计算吞吐量
        total_samples = len(train_loader.dataset) * epochs
        throughput = total_samples / train_time if train_time > 0 else 0
        print(f"Training Throughput: {throughput:.2f} samples/second")
        print(f"================================================")

        # 记录最终指标到 wandb
        if self.use_wandb and self.wandb_logger:
            self._log_final_metrics(train_time, throughput)

        self.trained = True

        if self.save_path:
            self.save()
            
            # 保存模型到 wandb artifacts
            if self.use_wandb and self.wandb_logger:
                self._save_model_artifact()

        # 关闭 wandb run
        if self.use_wandb:
            wandb.finish()

        return self.model
    
    def _init_wandb_logger(self):
        """初始化 wandb logger"""
        try:
            # 构建配置字典
            config = {
                'model': self.model_class.__name__,
                'dataset': self.dataset_class.__name__,
                'area': {
                    'lon': self.area.lon.tolist() if hasattr(self.area.lon, 'tolist') else self.area.lon,
                    'lat': self.area.lat.tolist() if hasattr(self.area.lat, 'tolist') else self.area.lat,
                    'title': self.area.title,
                },
                'model_params': self.model_params,
                'dataset_params': self.dataset_params,
                'trainer_params': self.trainer_params,
            }
            
            # 获取模型的损失函数和优化器配置
            if self.model:
                optimizer_config = self._get_optimizer_config()
                if optimizer_config:
                    config['optimizer'] = optimizer_config
                
                loss_function_info = self._get_loss_function_info()
                if loss_function_info:
                    config['loss_function'] = loss_function_info
            
            # 创建 wandb logger
            logger = WandbLogger(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=f"{self.title}_{self.model_class.__name__}",
                id=self.trainer_uid,
                config=config,
                save_dir='./out/wandb_logs',
            )
            
            print(f"\n📊 Wandb 已启用")
            print(f"  • Project: {WANDB_PROJECT}")
            print(f"  • Run ID: {self.trainer_uid}")
            print(f"  • Run URL: {logger.experiment.url}\n")
            
            return logger
            
        except Exception as e:
            print(f"\n⚠️  Wandb 初始化失败: {str(e)}")
            print(f"  训练将继续，但不记录到 wandb\n")
            return None
    
    def _get_optimizer_config(self):
        """获取优化器配置信息"""
        try:
            # Lightning 模型通过 configure_optimizers() 返回优化器
            if hasattr(self.model, 'configure_optimizers'):
                optimizers_config = self.model.configure_optimizers()
                
                # 处理不同的返回格式
                optimizer = None
                scheduler = None
                
                if isinstance(optimizers_config, tuple):
                    # 返回 (optimizer, scheduler) 或 ([optimizers], [schedulers])
                    optimizers, schedulers = optimizers_config
                    optimizer = optimizers[0] if isinstance(optimizers, list) else optimizers
                    scheduler = schedulers[0] if isinstance(schedulers, list) and schedulers else None
                elif isinstance(optimizers_config, list):
                    # 返回 [optimizer]
                    optimizer = optimizers_config[0]
                else:
                    # 直接返回 optimizer
                    optimizer = optimizers_config
                
                config = {}
                
                if optimizer:
                    # 获取优化器类型和参数
                    config['type'] = optimizer.__class__.__name__
                    
                    # 获取参数组（包含学习率等）
                    if hasattr(optimizer, 'param_groups') and optimizer.param_groups:
                        param_group = optimizer.param_groups[0]
                        config['learning_rate'] = param_group.get('lr', 'N/A')
                        config['weight_decay'] = param_group.get('weight_decay', 0)
                        
                        # 获取其他常见参数
                        if 'momentum' in param_group:
                            config['momentum'] = param_group['momentum']
                        if 'betas' in param_group:
                            config['betas'] = param_group['betas']
                        if 'eps' in param_group:
                            config['eps'] = param_group['eps']
                
                # 记录学习率调度器
                if scheduler:
                    config['scheduler'] = {
                        'type': scheduler.__class__.__name__,
                    }
                    # 获取调度器参数
                    if hasattr(scheduler, 'T_max'):
                        config['scheduler']['T_max'] = scheduler.T_max
                    if hasattr(scheduler, 'gamma'):
                        config['scheduler']['gamma'] = scheduler.gamma
                    if hasattr(scheduler, 'step_size'):
                        config['scheduler']['step_size'] = scheduler.step_size
                
                return config
                
        except Exception as e:
            print(f"⚠️  获取优化器配置失败: {str(e)}")
            return None
    
    def _get_loss_function_info(self):
        """获取损失函数信息"""
        try:
            loss_info = {}
            
            # 检查是否有自定义损失函数
            if hasattr(self.model, 'custom_mse_loss'):
                loss_info['type'] = 'Custom MSE Loss'
                loss_info['description'] = 'Custom MSE with NaN handling'
                loss_info['handles_nan'] = True
            elif hasattr(self.model, 'loss_fn'):
                loss_info['type'] = self.model.loss_fn.__class__.__name__
            else:
                # 默认 MSE
                loss_info['type'] = 'MSE'
            
            # 检查是否有损失权重或其他配置
            if hasattr(self.model, 'loss_weight'):
                loss_info['weight'] = self.model.loss_weight
            
            return loss_info
            
        except Exception as e:
            print(f"⚠️  获取损失函数信息失败: {str(e)}")
            return None
    
    def _log_final_metrics(self, train_time, throughput):
        """记录最终训练指标"""
        try:
            final_metrics = {
                'train_time_seconds': train_time,
                'throughput_samples_per_second': throughput,
            }
            
            # 记录最终的损失值
            if hasattr(self.model, 'train_loss') and self.model.train_loss:
                final_metrics['final_train_loss'] = self.model.train_loss[-1]
            
            if hasattr(self.model, 'val_loss') and self.model.val_loss:
                final_metrics['final_val_loss'] = self.model.val_loss[-1]
                # 计算最佳验证损失
                final_metrics['best_val_loss'] = min(self.model.val_loss)
            
            wandb.log(final_metrics)
            
        except Exception as e:
            print(f"⚠️  记录最终指标失败: {str(e)}")
    
    def _save_model_artifact(self):
        """保存模型到 wandb artifacts"""
        try:
            # 创建 artifact
            artifact = wandb.Artifact(
                name=f"{self.title}_model",
                type='model',
                description=f"{self.model_class.__name__} trained on {self.area.title}",
                metadata={
                    'model_class': self.model_class.__name__,
                    'dataset_class': self.dataset_class.__name__,
                    'epochs': self.trainer_params.get('epochs', 100),
                    'batch_size': self.trainer_params.get('batch_size', 20),
                }
            )
            
            # 添加模型文件
            if self.save_path:
                artifact.add_file(self.save_path)
                wandb.log_artifact(artifact)
                print(f"\n✅ 模型已保存到 wandb artifacts: {artifact.name}")
            
        except Exception as e:
            print(f"\n⚠️  保存模型到 wandb 失败: {str(e)}")
    
    def _print_optimization_summary(self, accumulate_grad_batches):
        """打印优化配置摘要"""
        is_windows = platform.system() == 'Windows'
        
        print("\n" + "="*60)
        print("🚀 训练优化配置")
        print("="*60)
        
        # 系统信息
        if is_windows:
            print(f"\n💻 系统: Windows (多进程数据加载已禁用)")
        
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
        
        # Wandb
        if self.use_wandb:
            print(f"\n📊 Wandb 日志:")
            print(f"  • 已启用: 实时记录训练指标")
        
        print("="*60 + "\n")

    def predict(self, offset: int, plot: bool = False) -> tuple:
        
        """
        预测
        
        :param offset: 数据偏移量
        :return: 输入和预测输出
        """

        print(self.save_path)

        if not self.trained:
            self.model = load(self.save_path, weights_only=False)
            self.trained = True
            
        if not self.model:
            raise ValueError('无已训练模型')
        
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
        
        pred_output = self.model(input)
        
        input = input.detach().numpy()
        output = output.detach().numpy()
        pred_output = pred_output.detach().numpy()
        
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
            # 使用增强版海表温度分析，传递二维海表温度数据
            plot_2d_kernel_density(pred_output, self.area.lon, self.area.lat)
            
        return input, output, pred_output, rmse, r2, ssta
    

    def save(self):
        save(self.model, self.save_path)
