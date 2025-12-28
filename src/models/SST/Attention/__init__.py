"""
注意力机制模块

主要组件:
- EfficientRGAttention: 高效版递归泛化自注意力模块
- RGAttention: EfficientRGAttention 的别名（向后兼容）
"""

from .RGAttention import EfficientRGAttention, RGAttention

__all__ = [
    'EfficientRGAttention',
    'RGAttention',
]

