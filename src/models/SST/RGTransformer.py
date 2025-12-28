"""
RG-Transformer V2: 优化版递归泛化注意力 Transformer 模型
专为海表温度时空序列预测任务设计

相比 V1 的改进：
1. ConvStem 替代 Patch Embedding - 更好的局部特征提取，解决边界效应
2. EfficientRGAttention 替代 RGAttention - 参数减少 33%，移除冗余计算
3. einops 简化张量操作 - 代码更清晰，潜在性能优化
4. torch.compile 支持 - 利用 PyTorch 2.0+ 编译优化

预期性能提升：
- 训练速度提升 ≥20%
- 显存占用减少 ≥15%
- 推理速度提升 ≥15%
- 参数量减少约 11%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from lightning import LightningModule
from torch import optim

try:
    from einops import rearrange
except ImportError:
    raise ImportError("Please install einops: pip install einops")

# 导入优化后的模块
from src.models.SST.PE.SphericalHarmonicEncoding import (
    SpatialSphericalHarmonicEncoding,
)
from src.models.SST.Attention.RGAttention import EfficientRGAttention
from src.models.SST.ConvStem import ConvStem, MultiScaleConvStem
from src.models.SST.MultiScaleDecoder import MultiScaleDecoder


class ChannelFeedForward(nn.Module):
    """
    基于通道的前馈网络（Channel-Mixing FFN）
    
    结构: Linear(d_model) -> GELU -> Dropout -> Linear(d_model) -> Dropout
    作用于特征维度，对空间位置共享权重
    """
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d_model]
        return self.net(x)


class RGTransformer(LightningModule):
    """
    RG-Transformer 海表温度预测模型（优化版）
    
    主要优化:
    1. ConvStem 替代 Patch Embedding - 多层卷积，更好的特征提取
    2. EfficientRGAttention 替代 RGAttention - 轻量门控，无 Global Token
    3. einops.rearrange 简化张量操作
    4. torch.compile 兼容设计
    """
    
    def __init__(
        self, 
        width: int, 
        height: int, 
        seq_len: int,
        d_model: int = 256, 
        num_heads: int = 8, 
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_attn_layers: int = 1,
        learning_rate: float = 1e-4,
        lat_range: Optional[List[float]] = None,
        lon_range: Optional[List[float]] = None,
        resolution: float = 1.0,
        patch_size: int = 4,
        use_compile: bool = False,
        compile_mode: str = "reduce-overhead",
        # 多尺度特征增强参数
        use_multiscale: bool = False,
        num_skip_connections: int = 2,
        skip_fusion: str = "add",
        **kwargs
    ):
        """
        Args:
            width: 海表温度图像宽度
            height: 海表温度图像高度
            seq_len: 输入序列长度
            d_model: 模型维度
            num_heads: 注意力头数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比例
            num_attn_layers: 注意力层数（替代 recursion_depth）
            learning_rate: 学习率
            lat_range: [lat_min, lat_max] 纬度范围
            lon_range: [lon_min, lon_max] 经度范围
            resolution: 空间分辨率
            patch_size: Patch大小（分块大小），默认4x4
            use_compile: 是否启用 torch.compile
            compile_mode: torch.compile 模式
            use_multiscale: 是否启用多尺度特征增强（跳跃连接）
            num_skip_connections: 跳跃连接数量（1 或 2）
            skip_fusion: 跳跃连接融合方式 "add" 或 "concat"
        """
        super().__init__()
        
        # 保存超参数
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.train_loss = []
        self.val_loss = []
        self.use_compile = use_compile
        self.compile_mode = compile_mode
        
        # 多尺度特征增强配置
        self.use_multiscale = use_multiscale
        self.num_skip_connections = num_skip_connections
        self.skip_fusion = skip_fusion
        
        if not (width and height):
            raise ValueError("Must specify width and height")
        
        self.width = width
        self.height = height
        self.seq_len = seq_len
        self.d_model = d_model
        self.patch_size = patch_size
        
        # 计算 patch 后的特征图尺寸
        self.w_feat = width // patch_size
        self.h_feat = height // patch_size
        
        # ======= 空间位置编码 =======
        self.spatial_pos_encoding = SpatialSphericalHarmonicEncoding(
            lat_range=lat_range,
            lon_range=lon_range,
            max_degree=2,
            resolution=resolution
        )
        self.spatial_enc_scale = nn.Parameter(torch.tensor(0.5))
        
        # ======= ConvStem 模块 =======
        if use_multiscale:
            # 多尺度 ConvStem，支持跳跃连接
            self.conv_stem = MultiScaleConvStem(
                in_channels=1,
                embed_dim=d_model,
                num_skip_outputs=num_skip_connections,
                use_bn=True
            )
            # 获取跳跃连接通道数
            self._skip_channels = self.conv_stem.get_skip_channels()
        else:
            # 原始 ConvStem
            self.conv_stem = ConvStem(
                in_channels=1,
                embed_dim=d_model,
                target_reduction=patch_size,
                use_bn=True
            )
            self._skip_channels = []
        
        # ======= Decoder 模块 =======
        if use_multiscale:
            # 多尺度解码器，支持跳跃连接融合
            # 跳跃连接通道需要反转顺序（从深到浅）
            skip_channels_reversed = list(reversed(self._skip_channels))
            self.decoder = MultiScaleDecoder(
                in_channels=d_model,
                out_channels=1,
                skip_channels=skip_channels_reversed,
                num_stages=num_skip_connections,
                fusion=skip_fusion
            )
            # 用于向后兼容的 patch_recovery（实际不使用）
            self.patch_recovery = None
        else:
            # 原始 Patch Recovery
            self.patch_recovery = nn.ConvTranspose2d(
                in_channels=d_model,
                out_channels=1,
                kernel_size=patch_size,
                stride=patch_size
            )
            self.decoder = None
    
        # ======= Transformer 组件 =======
        
        # 高效版 RG 注意力模块（时序注意力）
        self.attention = EfficientRGAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_attn_layers,
            use_gate=True
        )
        
        # Channel-Mixing FFN
        self.ffn = ChannelFeedForward(d_model, dim_feedforward, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
        # LayerNorm
        self.layer_norm = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 可学习的时序权重向量
        self.temporal_weights = nn.Parameter(torch.ones(seq_len - 1) / (seq_len - 1))

        # 可视化数据
        self.viz = {}
        
        # torch.compile 在训练开始时应用
        self._compiled = False
    
    def _maybe_compile(self):
        """在首次前向传播时应用 torch.compile"""
        if self.use_compile and not self._compiled and hasattr(torch, 'compile'):
            try:
                # 编译核心前向传播部分
                self._forward_impl = torch.compile(
                    self._forward_impl, 
                    mode=self.compile_mode
                )
                self._compiled = True
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}")
                self._compiled = True  # 标记为已尝试，避免重复
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch, seq_len-1, width, height]
        
        Returns:
            output: [batch, width, height]
        """
        self._maybe_compile()
        return self._forward_impl(x)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """
        实际的前向传播实现（可被 torch.compile 编译）
        
        Args:
            x: [batch, seq_len-1, width, height]
        
        Returns:
            output: [batch, width, height]
        """
        batch_size, seq_len_minus_1, width, height = x.shape
        
        # 1. 输入归一化和位置编码
        x = self._normalize_sst(x)
        
        # 生成空间位置编码
        # SpatialSphericalHarmonicEncoding 返回 [height, width]（lat, lon）
        # 输入 x 的形状是 [batch, seq, height, width]（lat, lon），无需转置
        spatial_enc = self.spatial_pos_encoding()  # [height, width]
        
        # 添加位置编码到输入数据中
        x = x + spatial_enc * self.spatial_enc_scale
        
        # 2. ConvStem（替代 Patch Embedding）
        # 使用 einops 简化张量操作
        # [B, S, W, H] -> [B*S, 1, W, H]
        x_flat = rearrange(x, 'b s w h -> (b s) 1 w h')
        
        # 根据是否使用多尺度模式选择不同的处理路径
        if self.use_multiscale:
            # 多尺度模式: 获取主特征和跳跃连接特征
            # [B*S, d_model, W', H'], List[[B*S, C_i, H_i, W_i]]
            x_embed, skip_features = self.conv_stem(x_flat, return_skip_features=True)
        else:
            # 原始模式
            x_embed = self.conv_stem(x_flat)
            skip_features = None
        
        _, d_model, w_feat, h_feat = x_embed.shape
        
        # 3. 准备 Transformer 输入
        # [B*S, D, W', H'] -> [B, S, D, W', H'] -> [B*W'*H', S, D]
        x_embed = rearrange(
            x_embed, 
            '(b s) d w h -> (b w h) s d', 
            b=batch_size, 
            s=seq_len_minus_1
        )
        
        # 4. Transformer Block
        
        # LayerNorm & Dropout
        x_tokens = self.layer_norm(x_embed)
        x_tokens = self.dropout(x_tokens)
        
        # Attention Block (EfficientRGAttention 内部已包含残差连接)
        x_tokens = self.attention(x_tokens)
        
        # FFN Block
        ffn_out = self.ffn(self.norm2(x_tokens))
        x_tokens = x_tokens + ffn_out  # Residual
        
        # 5. 时序聚合
        # [N, S, D] -> [N, D] (Weighted Sum)
        normalized_weights = F.softmax(self.temporal_weights, dim=0)
        weights = normalized_weights.view(-1, 1)
        
        # [N, S, D] * [S, 1] -> sum(dim=1) -> [N, D]
        output_tokens = (x_tokens * weights).sum(dim=1)
        
        # 6. 恢复空间结构和分辨率
        # [B*W'*H', D] -> [B, W', H', D] -> [B, D, W', H']
        output_tokens = rearrange(
            output_tokens, 
            '(b w h) d -> b d w h', 
            b=batch_size, 
            w=w_feat, 
            h=h_feat
        )
        
        # 7. Decoder
        if self.use_multiscale and skip_features is not None:
            # 多尺度解码器
            # 跳跃特征需要在时序维度上聚合（取最后一帧）
            # skip_features: List[[B*S, C_i, H_i, W_i]]
            aggregated_skips = []
            for skip in skip_features:
                # 重塑为 [B, S, C, H, W] 然后取最后一帧 [B, C, H, W]
                _, c, h, w = skip.shape
                skip_reshaped = rearrange(
                    skip, 
                    '(b s) c h w -> b s c h w', 
                    b=batch_size, 
                    s=seq_len_minus_1
                )
                # 使用加权平均聚合时序维度
                skip_aggregated = (skip_reshaped * normalized_weights.view(1, -1, 1, 1, 1)).sum(dim=1)
                aggregated_skips.append(skip_aggregated)
            
            # 反转跳跃特征顺序（从浅到深 -> 从深到浅）
            aggregated_skips_reversed = list(reversed(aggregated_skips))
            
            # 多尺度解码
            output_map = self.decoder(output_tokens, aggregated_skips_reversed)
        else:
            # 原始模式: 单次上采样
            output_map = self.patch_recovery(output_tokens)
        
        # Remove channel dim: [B, 1, W, H] -> [B, W, H]
        output = output_map.squeeze(1)
        
        # 保存可视化数据
        if batch_size > 0:
            self.viz['temporal_weights'] = normalized_weights
        
        return output
    
    def _normalize_sst(self, x: torch.Tensor) -> torch.Tensor:
        """
        海表温度数据归一化
        
        将 NaN 值（陆地区域）替换为 0
        """
        x_mask = torch.isnan(x)
        x_processed = x.clone()
        x_processed[x_mask] = 0.0
        return x_processed

    def custom_mse_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        处理 NaN 值的 MSE 损失函数
        
        Args:
            y_pred: [batch, width, height] 预测值
            y_true: [batch, width, height] 真实值
        
        Returns:
            loss: 标量损失值
        """
        # 创建有效值掩码（排除 NaN 区域）
        y_true_mask = torch.isnan(y_true)
        y_pred_mask = torch.isnan(y_pred)
        valid_mask = ~(y_true_mask | y_pred_mask)
        
        num_valid = valid_mask.sum()
        
        if num_valid > 0:
            y_pred_valid = y_pred[valid_mask]
            y_true_valid = y_true[valid_mask]
            loss = F.mse_loss(y_pred_valid, y_true_valid, reduction='mean')
            return loss
        else:
            # 返回零损失（但保持计算图）
            return y_pred.sum() * 0.0
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.custom_mse_loss(y_pred, y)
        self.train_loss.append(loss.detach().cpu().item())
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        val_loss = self.custom_mse_loss(y_pred, y)
        self.val_loss.append(val_loss.detach().cpu().item())
        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
        return val_loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
        )
        return optimizer
    
    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """
        获取模型参数量
        
        Args:
            trainable_only: 是否只统计可训练参数
        
        Returns:
            参数总数
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# 兼容别名（向后兼容旧代码引用）
RGTransformerV2 = RGTransformer
RGTransformerOptimized = RGTransformer

