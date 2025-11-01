"""
插件管理器

负责管理所有插件的注册、启用、禁用和生命周期调用
"""

from typing import List, Optional, Any, Dict
from .base import BasePlugin


class PluginManager:
    """
    插件管理器
    
    统一管理所有训练器插件，提供生命周期钩子的统一调用接口。
    
    使用示例:
        manager = PluginManager()
        manager.register(WandbPlugin(enabled=True))
        manager.register(VisualizationPlugin(enabled=False))
        
        # 在训练开始时调用
        manager.on_train_start(trainer, model, train_loader=train_loader)
    """
    
    def __init__(self):
        """初始化插件管理器"""
        self.plugins: List[BasePlugin] = []
    
    def register(self, plugin: BasePlugin) -> None:
        """
        注册插件
        
        Args:
            plugin: 插件实例
        """
        if not isinstance(plugin, BasePlugin):
            raise TypeError(f"插件必须是 BasePlugin 的实例，收到: {type(plugin)}")
        
        # 检查是否已注册同名插件
        for existing in self.plugins:
            if existing.name == plugin.name:
                raise ValueError(f"插件 '{plugin.name}' 已注册")
        
        self.plugins.append(plugin)
    
    def register_all(self, plugins: List[BasePlugin]) -> None:
        """
        批量注册插件
        
        Args:
            plugins: 插件实例列表
        """
        for plugin in plugins:
            self.register(plugin)
    
    def unregister(self, plugin_name: str) -> Optional[BasePlugin]:
        """
        取消注册插件
        
        Args:
            plugin_name: 插件名称
        
        Returns:
            被取消注册的插件，如果不存在则返回 None
        """
        for i, plugin in enumerate(self.plugins):
            if plugin.name == plugin_name:
                return self.plugins.pop(i)
        return None
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """
        获取指定插件
        
        Args:
            plugin_name: 插件名称
        
        Returns:
            插件实例，如果不存在则返回 None
        """
        for plugin in self.plugins:
            if plugin.name == plugin_name:
                return plugin
        return None
    
    def enable(self, plugin_name: str) -> bool:
        """
        启用插件
        
        Args:
            plugin_name: 插件名称
        
        Returns:
            是否成功启用
        """
        plugin = self.get_plugin(plugin_name)
        if plugin:
            plugin.enabled = True
            return True
        return False
    
    def disable(self, plugin_name: str) -> bool:
        """
        禁用插件
        
        Args:
            plugin_name: 插件名称
        
        Returns:
            是否成功禁用
        """
        plugin = self.get_plugin(plugin_name)
        if plugin:
            plugin.enabled = False
            return True
        return False
    
    def get_enabled_plugins(self) -> List[BasePlugin]:
        """
        获取所有启用的插件
        
        Returns:
            启用的插件列表
        """
        return [p for p in self.plugins if p.enabled]
    
    # ========== 生命周期钩子调用 ==========
    
    def on_train_start(self, trainer: Any, model: Any, **kwargs) -> None:
        """调用所有启用插件的 on_train_start"""
        for plugin in self.get_enabled_plugins():
            try:
                plugin.on_train_start(trainer, model, **kwargs)
            except Exception as e:
                print(f"⚠️  插件 {plugin.name}.on_train_start() 执行失败: {str(e)}")
    
    def on_train_end(self, trainer: Any, model: Any, train_time: float = None, **kwargs) -> None:
        """调用所有启用插件的 on_train_end"""
        for plugin in self.get_enabled_plugins():
            try:
                plugin.on_train_end(trainer, model, train_time=train_time, **kwargs)
            except Exception as e:
                print(f"⚠️  插件 {plugin.name}.on_train_end() 执行失败: {str(e)}")
    
    def on_epoch_start(self, trainer: Any, model: Any, epoch: int, **kwargs) -> None:
        """调用所有启用插件的 on_epoch_start"""
        for plugin in self.get_enabled_plugins():
            try:
                plugin.on_epoch_start(trainer, model, epoch, **kwargs)
            except Exception as e:
                print(f"⚠️  插件 {plugin.name}.on_epoch_start() 执行失败: {str(e)}")
    
    def on_epoch_end(self, trainer: Any, model: Any, epoch: int, metrics: Dict[str, float] = None, **kwargs) -> None:
        """调用所有启用插件的 on_epoch_end"""
        for plugin in self.get_enabled_plugins():
            try:
                plugin.on_epoch_end(trainer, model, epoch, metrics=metrics, **kwargs)
            except Exception as e:
                print(f"⚠️  插件 {plugin.name}.on_epoch_end() 执行失败: {str(e)}")
    
    def on_batch_end(self, trainer: Any, model: Any, batch_idx: int, metrics: Dict[str, float] = None, **kwargs) -> None:
        """调用所有启用插件的 on_batch_end"""
        for plugin in self.get_enabled_plugins():
            try:
                plugin.on_batch_end(trainer, model, batch_idx, metrics=metrics, **kwargs)
            except Exception as e:
                print(f"⚠️  插件 {plugin.name}.on_batch_end() 执行失败: {str(e)}")
    
    # ========== Lightning 集成 ==========
    
    def get_all_lightning_loggers(self) -> List[Any]:
        """
        获取所有插件提供的 Lightning Logger
        
        Returns:
            Logger 列表
        """
        loggers = []
        for plugin in self.get_enabled_plugins():
            logger = plugin.get_lightning_logger()
            if logger:
                loggers.append(logger)
        return loggers
    
    def get_all_lightning_callbacks(self) -> List[Any]:
        """
        获取所有插件提供的 Lightning Callback
        
        Returns:
            Callback 列表
        """
        callbacks = []
        for plugin in self.get_enabled_plugins():
            callbacks.extend(plugin.get_lightning_callbacks())
        return callbacks
    
    # ========== 工具方法 ==========
    
    def __repr__(self) -> str:
        """管理器字符串表示"""
        enabled_count = len(self.get_enabled_plugins())
        total_count = len(self.plugins)
        return f"PluginManager({enabled_count}/{total_count} 插件启用)"
    
    def list_plugins(self) -> List[str]:
        """列出所有插件名称和状态"""
        return [str(plugin) for plugin in self.plugins]

