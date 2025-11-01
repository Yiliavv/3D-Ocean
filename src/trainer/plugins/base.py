"""
插件基类

定义所有训练器插件必须实现的接口和生命周期钩子
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from lightning.pytorch.loggers import Logger
from lightning.pytorch.callbacks import Callback


class BasePlugin(ABC):
    """
    训练器插件基类
    
    所有插件必须继承此类并实现必要的方法。
    插件可以通过生命周期钩子在训练的不同阶段执行自定义逻辑。
    
    使用示例:
        class MyPlugin(BasePlugin):
            def __init__(self, enabled=True, **kwargs):
                super().__init__(enabled)
                self.config = kwargs
            
            def on_train_start(self, trainer, model, **kwargs):
                print("训练开始!")
            
            def on_train_end(self, trainer, model, **kwargs):
                print("训练结束!")
    """
    
    def __init__(self, enabled: bool = True, name: str = None):
        """
        Args:
            enabled: 是否启用此插件
            name: 插件名称（默认使用类名）
        """
        self.enabled = enabled
        self.name = name or self.__class__.__name__
    
    # ========== 生命周期钩子 ==========
    
    def on_train_start(self, trainer: Any, model: Any, **kwargs) -> None:
        """
        训练开始前调用
        
        Args:
            trainer: BaseTrainer 实例
            model: 训练模型
            **kwargs: 其他上下文信息（如 dataset, train_loader, val_loader 等）
        """
        pass
    
    def on_train_end(self, trainer: Any, model: Any, train_time: float = None, **kwargs) -> None:
        """
        训练结束后调用
        
        Args:
            trainer: BaseTrainer 实例
            model: 训练模型
            train_time: 训练总时间（秒）
            **kwargs: 其他上下文信息
        """
        pass
    
    def on_epoch_start(self, trainer: Any, model: Any, epoch: int, **kwargs) -> None:
        """
        Epoch 开始前调用
        
        Args:
            trainer: BaseTrainer 实例
            model: 训练模型
            epoch: 当前 epoch 编号
            **kwargs: 其他上下文信息
        """
        pass
    
    def on_epoch_end(self, trainer: Any, model: Any, epoch: int, metrics: Dict[str, float] = None, **kwargs) -> None:
        """
        Epoch 结束后调用
        
        Args:
            trainer: BaseTrainer 实例
            model: 训练模型
            epoch: 当前 epoch 编号
            metrics: 当前 epoch 的指标字典（如 {'train_loss': 0.5, 'val_loss': 0.6}）
            **kwargs: 其他上下文信息
        """
        pass
    
    def on_batch_end(self, trainer: Any, model: Any, batch_idx: int, metrics: Dict[str, float] = None, **kwargs) -> None:
        """
        Batch 结束后调用（可选，可能影响性能）
        
        Args:
            trainer: BaseTrainer 实例
            model: 训练模型
            batch_idx: 当前 batch 编号
            metrics: 当前 batch 的指标字典
            **kwargs: 其他上下文信息
        """
        pass
    
    # ========== Lightning 集成接口 ==========
    
    def get_lightning_logger(self) -> Optional[Logger]:
        """
        返回 PyTorch Lightning Logger 实例（如果需要）
        
        例如：WandbPlugin 可以返回 WandbLogger
        
        Returns:
            Logger 实例，如果不需要则返回 None
        """
        return None
    
    def get_lightning_callbacks(self) -> List[Callback]:
        """
        返回 PyTorch Lightning Callback 列表（如果需要）
        
        例如：可以返回自定义的 MetricCallback
        
        Returns:
            Callback 列表，如果不需要则返回空列表
        """
        return []
    
    # ========== 工具方法 ==========
    
    def __repr__(self) -> str:
        """插件字符串表示"""
        status = "✓ 启用" if self.enabled else "✗ 禁用"
        return f"{self.name}({status})"

