"""
可视化插件基类

为未来可视化功能预留接口，如训练过程可视化、注意力图可视化等
"""

from typing import Any
from .base import BasePlugin


class VisualizationPlugin(BasePlugin):
    """
    可视化插件基类
    
    提供训练过程的可视化功能，如：
    - 损失曲线可视化
    - 注意力图可视化
    - 预测结果可视化
    
    子类应该实现具体的可视化逻辑。
    
    使用示例:
        class AttentionVisualizationPlugin(VisualizationPlugin):
            def on_epoch_end(self, trainer, model, epoch, metrics=None, **kwargs):
                # 可视化注意力图
                self.visualize_attention(model, epoch)
    """
    
    def __init__(self, 
                 enabled: bool = True,
                 save_dir: str = None,
                 save_format: str = 'png',
                 **kwargs):
        """
        Args:
            enabled: 是否启用插件
            save_dir: 可视化结果保存目录
            save_format: 保存格式（png, pdf, svg等）
            **kwargs: 其他配置参数
        """
        super().__init__(enabled, name="VisualizationPlugin")
        self.save_dir = save_dir
        self.save_format = save_format
        self.config = kwargs
    
    def visualize_loss_curves(self, trainer: Any, model: Any, **kwargs) -> None:
        """
        可视化损失曲线（子类实现）
        
        Args:
            trainer: BaseTrainer 实例
            model: 训练模型
            **kwargs: 其他上下文信息
        """
        pass
    
    def visualize_attention_maps(self, trainer: Any, model: Any, epoch: int, **kwargs) -> None:
        """
        可视化注意力图（子类实现）
        
        Args:
            trainer: BaseTrainer 实例
            model: 训练模型
            epoch: 当前 epoch
            **kwargs: 其他上下文信息
        """
        pass
    
    def visualize_predictions(self, trainer: Any, model: Any, epoch: int, **kwargs) -> None:
        """
        可视化预测结果（子类实现）
        
        Args:
            trainer: BaseTrainer 实例
            model: 训练模型
            epoch: 当前 epoch
            **kwargs: 其他上下文信息
        """
        pass

