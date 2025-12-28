"""
SST (Sea Surface Temperature) 模型模块

主要组件:
- RGTransformer: 优化版递归泛化注意力 Transformer 模型
- ConvStem: 卷积干模块（替代 Patch Embedding）
"""

from .RGTransformer import RGTransformer, RGTransformerV2, RGTransformerOptimized

__all__ = [
    'RGTransformer',
    'RGTransformerV2',
    'RGTransformerOptimized',
]

