"""
SST (Sea Surface Temperature) 模型模块

主要组件:
- RGTransformer: 优化版递归泛化注意力 Transformer 模型
- ConvStem: 卷积干模块（替代 Patch Embedding）
"""

from .RGTransformer import RGTransformer, RGTransformerV2, RGTransformerOptimized

# 可选导入 Legacy 版本
try:
    from .RGTransformerLegacy import RGTransformer as RGTransformerLegacy
except ImportError:
    RGTransformerLegacy = None

__all__ = [
    'RGTransformer',
    'RGTransformerV2',
    'RGTransformerOptimized',
    'RGTransformerLegacy',
]

