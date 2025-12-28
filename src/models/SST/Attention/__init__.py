"""
注意力机制模块

主要组件:
- EfficientRGAttention: 高效版递归泛化自注意力模块
- RGAttention: EfficientRGAttention 的别名（向后兼容）
"""

from .RGAttention import EfficientRGAttention, RGAttention

# 可选导入 Legacy 版本
try:
    from .RGAttentionLegacy import RGAttention as RGAttentionLegacy
except ImportError:
    RGAttentionLegacy = None

__all__ = [
    'EfficientRGAttention',
    'RGAttention',
    'RGAttentionLegacy',
]

