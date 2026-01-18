"""
模型变体工厂模块

用于创建不同的消融变体模型，支持组件启用/禁用。

包含:
- create_variant_model: 根据配置创建模型变体
- PatchEmbedding: ConvStem 的简单替代
- StandardAttentionRGTransformer: 使用标准 MHA 的变体
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from src.analysis.ablation.config import AblationConfig


class PatchEmbedding(nn.Module):
    """
    简单的 Patch Embedding 模块
    
    用于替代 ConvStem，作为 w/o ConvStem 消融变体。
    直接将输入划分为 patch 并线性投影。
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 256,
        patch_size: int = 4
    ):
        """
        Args:
            in_channels: 输入通道数
            embed_dim: 嵌入维度
            patch_size: Patch 大小
        """
        super().__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # 使用卷积实现 patch embedding
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # 初始化
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch, channels, height, width]
            
        Returns:
            [batch, embed_dim, height/patch_size, width/patch_size]
        """
        return self.proj(x)
    
    def get_output_size(self, input_size: tuple) -> tuple:
        """计算输出尺寸"""
        h, w = input_size
        return h // self.patch_size, w // self.patch_size


class NoSphericalEncoding(nn.Module):
    """
    空的球谐波编码替代
    
    用于 w/o SHPE 变体，返回零张量。
    """
    
    def __init__(self, height: int, width: int):
        super().__init__()
        self.height = height
        self.width = width
        # 注册一个零缓冲区
        self.register_buffer('zero_encoding', torch.zeros(height, width))
    
    def forward(self) -> torch.Tensor:
        """返回零编码"""
        return self.zero_encoding


class SimpleDecoder(nn.Module):
    """
    简单的单层解码器
    
    用于 w/o MultiScale 变体，替代多尺度解码器。
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 1,
        scale_factor: int = 4
    ):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            scale_factor: 上采样倍数
        """
        super().__init__()
        
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=scale_factor,
            stride=scale_factor
        )
        
        nn.init.xavier_uniform_(self.deconv.weight)
        if self.deconv.bias is not None:
            nn.init.zeros_(self.deconv.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        skip_features: Optional[list] = None
    ) -> torch.Tensor:
        """
        前向传播（忽略跳跃连接）
        
        Args:
            x: [batch, in_channels, H, W]
            skip_features: 忽略
            
        Returns:
            [batch, out_channels, H*scale, W*scale]
        """
        return self.deconv(x)


class StandardAttention(nn.Module):
    """
    标准多头注意力模块
    
    用于 w/o RGAttention 变体，使用简单的残差连接。
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        num_layers: int = 1
    ):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: Dropout 比例
            num_layers: 层数
        """
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(
                d_model, 
                num_heads, 
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（简单残差连接，无门控）
        
        Args:
            x: [batch, seq_len, d_model]
            
        Returns:
            [batch, seq_len, d_model]
        """
        for attn, norm in zip(self.layers, self.layer_norms):
            # Pre-norm
            normed = norm(x)
            # Attention
            attn_out, _ = attn(normed, normed, normed)
            # 简单残差（无门控）
            x = x + self.dropout(attn_out)
        
        return x


def create_variant_model(
    config: AblationConfig,
    base_model_params: Dict[str, Any],
    model_class = None
) -> nn.Module:
    """
    根据消融配置创建模型变体
    
    通过修改模型组件实现不同的消融变体。
    
    Args:
        config: 消融配置
        base_model_params: 基础模型参数
        model_class: 模型类（默认使用 RGTransformer）
        
    Returns:
        配置好的模型实例
    """
    # 延迟导入避免循环依赖
    if model_class is None:
        from src.models.SST.RGTransformer import RGTransformer
        model_class = RGTransformer
    
    # 合并参数
    model_params = config.get_model_kwargs(base_model_params)
    
    # 创建基础模型
    model = model_class(**model_params)
    
    # 应用消融修改
    model = _apply_ablation_modifications(model, config)
    
    return model


def _apply_ablation_modifications(
    model: nn.Module,
    config: AblationConfig
) -> nn.Module:
    """
    应用消融修改到模型
    
    Args:
        model: 基础模型
        config: 消融配置
        
    Returns:
        修改后的模型
    """
    # w/o ConvStem: 替换为 PatchEmbedding
    if not config.use_conv_stem:
        if hasattr(model, 'conv_stem'):
            embed_dim = model.d_model
            patch_size = model.patch_size
            model.conv_stem = PatchEmbedding(
                in_channels=1,
                embed_dim=embed_dim,
                patch_size=patch_size
            )
    
    # w/o RGAttention: 替换为标准注意力
    if not config.use_efficient_attention:
        if hasattr(model, 'attention'):
            d_model = model.d_model
            num_heads = model.hparams.get('num_heads', 8)
            dropout = model.hparams.get('dropout', 0.1)
            num_layers = model.hparams.get('num_attn_layers', 1)
            model.attention = StandardAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                num_layers=num_layers
            )
    
    # w/o SHPE: 移除球谐波编码
    if not config.use_spherical_encoding:
        if hasattr(model, 'spatial_pos_encoding'):
            height = model.height
            width = model.width
            model.spatial_pos_encoding = NoSphericalEncoding(height, width)
            # 设置缩放因子为 0
            if hasattr(model, 'spatial_enc_scale'):
                model.spatial_enc_scale.data.fill_(0.0)
    
    # w/o MultiScaleDecoder: 替换为简单解码器
    if not config.use_multiscale_decoder:
        if hasattr(model, 'decoder') and model.decoder is not None:
            d_model = model.d_model
            patch_size = model.patch_size
            model.decoder = SimpleDecoder(
                in_channels=d_model,
                out_channels=1,
                scale_factor=patch_size
            )
            model.use_multiscale = False
        elif hasattr(model, 'patch_recovery'):
            # 已经是简单解码器，无需修改
            pass
    
    # w/o GatedResidual: 禁用门控
    if not config.use_gated_residual:
        if hasattr(model, 'attention') and hasattr(model.attention, 'use_gate'):
            model.attention.use_gate = False
            # 如果有 gates 模块，将其替换为 Identity
            if hasattr(model.attention, 'gates'):
                model.attention.gates = nn.ModuleList([
                    nn.Identity() for _ in model.attention.gates
                ])
    
    return model


def get_variant_description(config: AblationConfig) -> str:
    """
    获取变体的详细描述
    
    Args:
        config: 消融配置
        
    Returns:
        描述字符串
    """
    modifications = []
    
    if not config.use_conv_stem:
        modifications.append("ConvStem → PatchEmbedding")
    if not config.use_efficient_attention:
        modifications.append("RGAttention → StandardAttention")
    if not config.use_spherical_encoding:
        modifications.append("SphericalHarmonicEncoding → None")
    if not config.use_multiscale_decoder:
        modifications.append("MultiScaleDecoder → SimpleDecoder")
    if not config.use_gated_residual:
        modifications.append("GatedResidual → SimpleResidual")
    
    if not modifications:
        return "Baseline (no modifications)"
    
    return ", ".join(modifications)

