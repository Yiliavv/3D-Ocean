"""
训练器插件系统

提供可扩展的插件架构，支持各种训练增强功能：
- 日志记录插件（如 Wandb）
- 可视化插件（如训练过程可视化）
- 自定义回调插件
"""

from .base import BasePlugin
from .manager import PluginManager
from .wandb import WandbPlugin
from .visualization import VisualizationPlugin

__all__ = ['BasePlugin', 'PluginManager', 'WandbPlugin', 'VisualizationPlugin']

