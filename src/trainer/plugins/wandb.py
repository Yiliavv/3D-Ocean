"""
Wandb 日志插件

将 wandb 功能从 BaseTrainer 中提取出来，作为独立插件
"""

import os
import wandb
from typing import Any, Dict, Optional
from lightning.pytorch.loggers import WandbLogger

from src.config.params import WANDB_PROJECT, WANDB_ENTITY, PROJECT_PATH
from .base import BasePlugin


class WandbPlugin(BasePlugin):
    """
    Wandb 日志插件
    
    提供训练过程的 wandb 日志记录功能，包括：
    - 超参数配置记录
    - 训练指标记录
    - 模型 artifact 保存
    
    使用示例:
        plugin = WandbPlugin(
            enabled=True,
            project="my-project",
            entity="my-entity"
        )
        manager.register(plugin)
    """
    
    def __init__(self, 
                 enabled: bool = True,
                 project: str = None,
                 entity: str = None,
                 save_dir: str = None,
                 save_checkpoint_artifact: bool = True,
                 **kwargs):
        """
        Args:
            enabled: 是否启用插件
            project: Wandb 项目名称（默认使用配置中的 WANDB_PROJECT）
            entity: Wandb 实体名称（默认使用配置中的 WANDB_ENTITY）
            save_dir: 本地日志保存目录（None 表示不保存本地日志）
            save_checkpoint_artifact: 是否保存 checkpoint 到 wandb artifacts
            **kwargs: 其他 wandb 配置（如 tags, group 等）
        """
        super().__init__(enabled, name="WandbPlugin")
        self.project = project or WANDB_PROJECT
        self.entity = entity or WANDB_ENTITY
        self.save_dir = save_dir
        self.save_checkpoint_artifact = save_checkpoint_artifact
        self.wandb_config = kwargs
        
        self.logger: Optional[WandbLogger] = None
        self.trainer_ref: Any = None
    
    def on_train_start(self, trainer: Any, model: Any, **kwargs) -> None:
        """初始化 wandb logger"""
        if not self.enabled:
            return
        
        try:
            self.trainer_ref = trainer
            
            # 构建配置字典
            config = self._build_config(trainer, model, **kwargs)
            
            # 创建 wandb logger
            self.logger = WandbLogger(
                project=self.project,
                entity=self.entity,
                name=f"{trainer.title}_{trainer.model_class.__name__}",
                id=trainer.trainer_uid,
                config=config,
                save_dir=self.save_dir,  # None 表示不保存本地日志
                **self.wandb_config
            )
            
            print(f"\n📊 Wandb 已启用")
            print(f"  • Project: {self.project}")
            print(f"  • Run ID: {trainer.trainer_uid}")
            print(f"  • Run URL: {self.logger.experiment.url}\n")
            
        except Exception as e:
            print(f"\n⚠️  Wandb 初始化失败: {str(e)}")
            print(f"  训练将继续，但不记录到 wandb\n")
            self.enabled = False
            self.logger = None
    
    def on_train_end(self, trainer: Any, model: Any, train_time: float = None, **kwargs) -> None:
        """训练结束后记录最终指标并保存 checkpoint"""
        if not self.enabled or not self.logger:
            return
        
        try:
            # 记录最终指标
            final_metrics = self._build_final_metrics(trainer, model, train_time)
            if final_metrics:
                wandb.log(final_metrics)
            
            # 保存 checkpoint 到 wandb artifacts
            if self.save_checkpoint_artifact:
                self._save_checkpoint_artifact(trainer)
            
            # 关闭 wandb run
            wandb.finish()
            
        except Exception as e:
            print(f"\n⚠️  训练结束时 wandb 操作失败: {str(e)}")
    
    def get_lightning_logger(self) -> Optional[WandbLogger]:
        """返回 WandbLogger 实例"""
        return self.logger if self.enabled else None
    
    def _build_config(self, trainer: Any, model: Any, **kwargs) -> Dict:
        """构建 wandb 配置字典"""
        config = {
            'model': trainer.model_class.__name__,
            'dataset': trainer.dataset_class.__name__,
            'area': {
                'lon': trainer.area.lon.tolist() if hasattr(trainer.area.lon, 'tolist') else trainer.area.lon,
                'lat': trainer.area.lat.tolist() if hasattr(trainer.area.lat, 'tolist') else trainer.area.lat,
                'title': trainer.area.title,
            },
            'model_params': trainer.model_params,
            'dataset_params': trainer.dataset_params,
            'trainer_params': trainer.trainer_params,
        }
        
        # 获取模型的优化器和损失函数配置
        if model:
            optimizer_config = self._get_optimizer_config(model)
            if optimizer_config:
                config['optimizer'] = optimizer_config
            
            loss_function_info = self._get_loss_function_info(model)
            if loss_function_info:
                config['loss_function'] = loss_function_info
        
        return config
    
    def _build_final_metrics(self, trainer: Any, model: Any, train_time: float = None, **kwargs) -> Dict:
        """构建最终指标字典"""
        final_metrics = {}
        
        if train_time is not None:
            final_metrics['train_time_seconds'] = train_time
            
            # 计算吞吐量
            total_samples = kwargs.get('total_samples')
            if total_samples is not None:
                throughput = total_samples / train_time if train_time > 0 else 0
                final_metrics['throughput_samples_per_second'] = throughput
        
        # 记录最终的损失值
        if hasattr(model, 'train_loss') and model.train_loss:
            final_metrics['final_train_loss'] = model.train_loss[-1]
        
        if hasattr(model, 'val_loss') and model.val_loss:
            final_metrics['final_val_loss'] = model.val_loss[-1]
            final_metrics['best_val_loss'] = min(model.val_loss)
        
        return final_metrics
    
    def _save_checkpoint_artifact(self, trainer: Any) -> None:
        """保存最后一个 checkpoint 到 wandb artifacts"""
        try:
            last_checkpoint = f'{PROJECT_PATH}/out/checkpoints/{trainer.title}/last.ckpt'
            
            if not os.path.exists(last_checkpoint):
                print(f"\n⚠️  最后一个checkpoint不存在: {last_checkpoint}")
                return
            
            # 创建 artifact
            artifact = wandb.Artifact(
                name=f"{trainer.title}_checkpoint",
                type='model',
                description=f"{trainer.model_class.__name__} trained on {trainer.area.title} - Last checkpoint",
                metadata={
                    'model_class': trainer.model_class.__name__,
                    'dataset_class': trainer.dataset_class.__name__,
                    'epochs': trainer.trainer_params.get('epochs', 100),
                    'batch_size': trainer.trainer_params.get('batch_size', 20),
                    'checkpoint_type': 'last',
                }
            )
            
            # 添加checkpoint文件
            artifact.add_file(last_checkpoint)
            wandb.log_artifact(artifact)
            print(f"\n✅ 最后一个checkpoint已保存到 wandb artifacts: {artifact.name}")
            
        except Exception as e:
            print(f"\n⚠️  保存checkpoint到 wandb 失败: {str(e)}")
    
    def _get_optimizer_config(self, model: Any) -> Optional[Dict]:
        """获取优化器配置信息"""
        try:
            if hasattr(model, 'configure_optimizers'):
                optimizers_config = model.configure_optimizers()
                
                optimizer = None
                scheduler = None
                
                if isinstance(optimizers_config, tuple):
                    optimizers, schedulers = optimizers_config
                    optimizer = optimizers[0] if isinstance(optimizers, list) else optimizers
                    scheduler = schedulers[0] if isinstance(schedulers, list) and schedulers else None
                elif isinstance(optimizers_config, list):
                    optimizer = optimizers_config[0]
                else:
                    optimizer = optimizers_config
                
                config = {}
                
                if optimizer:
                    config['type'] = optimizer.__class__.__name__
                    
                    if hasattr(optimizer, 'param_groups') and optimizer.param_groups:
                        param_group = optimizer.param_groups[0]
                        config['learning_rate'] = param_group.get('lr', 'N/A')
                        config['weight_decay'] = param_group.get('weight_decay', 0)
                        
                        if 'momentum' in param_group:
                            config['momentum'] = param_group['momentum']
                        if 'betas' in param_group:
                            config['betas'] = param_group['betas']
                        if 'eps' in param_group:
                            config['eps'] = param_group['eps']
                
                if scheduler:
                    config['scheduler'] = {
                        'type': scheduler.__class__.__name__,
                    }
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
    
    def _get_loss_function_info(self, model: Any) -> Optional[Dict]:
        """获取损失函数信息"""
        try:
            loss_info = {}
            
            if hasattr(model, 'custom_mse_loss'):
                loss_info['type'] = 'Custom MSE Loss'
                loss_info['description'] = 'Custom MSE with NaN handling'
                loss_info['handles_nan'] = True
            elif hasattr(model, 'loss_fn'):
                loss_info['type'] = model.loss_fn.__class__.__name__
            else:
                loss_info['type'] = 'MSE'
            
            if hasattr(model, 'loss_weight'):
                loss_info['weight'] = model.loss_weight
            
            return loss_info
            
        except Exception as e:
            print(f"⚠️  获取损失函数信息失败: {str(e)}")
            return None

