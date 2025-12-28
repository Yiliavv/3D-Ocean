"""
RG-Transformer: 递归泛化注意力 Transformer 模型
专为海表温度时空序列预测任务设计

架构设计：
- 采用 Patch Embedding 方案解决分辨率锁定问题
- 注意力机制：使用 RGAttentionWithGlobalQuery（递归泛化自注意力 + 全局查询注意力）
- 空间处理：使用 Patch Embedding 将空间网格转换为特征向量，实现分辨率无关
- 时序处理：对每个 Patch 位置独立进行时序建模，但在全空间共享权重
- 全局查询：通过可学习的全局查询向量引入历史上下文信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning import LightningModule
from torch import optim

# 导入注意力模块
from src.models.SST.PE.SphericalHarmonicEncoding import (
    SpatialSphericalHarmonicEncoding,
)
from src.models.SST.Attention.RGAttention import RGAttention


class ChannelFeedForward(nn.Module):
    """
    基于通道的前馈网络（Channel-Mixing FFN）
    
    结构: Linear(d_model) -> GELU -> Dropout -> Linear(d_model) -> Dropout
    作用于特征维度，对空间位置共享权重
    """
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [..., d_model]
        return self.net(x)
        
class RGTransformer(LightningModule):
    """递归泛化注意力 Transformer - 海表温度预测模型"""
    
    def __init__(self, 
                 width, height, seq_len,
                 d_model=256, 
                 num_heads=8, 
                 dim_feedforward=1024,
                 dropout=0.1,
                 recursion_depth=2,
                 learning_rate=1e-4,
                 lat_range=None,
                 lon_range=None,
                 resolution=1.0,
                 patch_size=4,
                 **kwargs):
        """
        Args:
            width: 海表温度图像宽度
            height: 海表温度图像高度
            seq_len: 输入序列长度
            d_model: 模型维度
            num_heads: 注意力头数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比例
            recursion_depth: RG注意力的递归深度
            learning_rate: 学习率
            lat_range: [lat_min, lat_max] 纬度范围
            lon_range: [lon_min, lon_max] 经度范围
            resolution: 空间分辨率
            patch_size: Patch大小（分块大小），默认4x4
            global_context_size: 全局查询向量数量
            use_global_query: 是否启用全局查询
        """
        super().__init__()
        
        self.learning_rate = learning_rate
        self.train_loss = []
        self.val_loss = []
        
        if not (width and height):
            raise ValueError("Must specify width and height")
        
        self.width = width
        self.height = height
        self.seq_len = seq_len
        self.d_model = d_model
        self.patch_size = patch_size
        
        # 空间位置编码（在原始分辨率上计算）
        self.spatial_pos_encoding = SpatialSphericalHarmonicEncoding(
            lat_range=lat_range,
            lon_range=lon_range,
            max_degree=2,
            resolution=resolution
        )
        self.spatial_enc_scale = nn.Parameter(torch.tensor(0.5))
        
        # ======= Patch Embedding 模块 =======
        # 将 [1, H, W] -> [d_model, H/p, W/p]
        self.patch_embed = nn.Conv2d(
            in_channels=1, 
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # ======= Patch Recovery 模块 =======
        # 将 [d_model, H/p, W/p] -> [1, H, W]
        self.patch_recovery = nn.ConvTranspose2d(
            in_channels=d_model,
            out_channels=1,
            kernel_size=patch_size,
            stride=patch_size
        )
    
        # ======= Transformer 组件 =======
        
        # RG递归注意力模块（时序注意力）
        self.attention = RGAttention(
            d_model=d_model,
            num_heads=num_heads,
            recursion_depth=recursion_depth,
            dropout=dropout,
            use_global_token=True # 启用轻量级 Global Token
        )
        
        # Channel-Mixing FFN
        self.ffn = ChannelFeedForward(d_model, dim_feedforward, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
        # LayerNorm 现在作用于 d_model 维度
        self.layer_norm = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 可学习的时序权重向量
        self.temporal_weights = nn.Parameter(torch.ones(seq_len - 1) / (seq_len - 1))

        self.viz = {}
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len-1, width, height]
        Returns:
            output: [batch, width, height]
        """        
        batch_size, seq_len_minus_1, width, height = x.shape
        
        # 1. 输入归一化和位置编码
        # 这一步在原始分辨率进行，利用球谐编码的物理特性
        x = self._normalize_sst(x)
        
        # 生成空间位置编码
        # SpatialSphericalHarmonicEncoding 返回 [height, width]
        # 输入 x 的形状是 [batch, seq, width, height]，需要转置
        spatial_enc = self.spatial_pos_encoding()  # [height, width]
        spatial_enc = spatial_enc.T  # [width, height]
        spatial_weight = self.spatial_enc_scale
        
        # 添加位置编码到输入数据中（广播加法）
        # [batch, seq, W, H] + [W, H]
        x = x + spatial_enc * spatial_weight
        
        # 2. Patch Embedding
        # reshape to combine batch and seq for Conv2d: [B*S, 1, W, H]
        x_flat = x.view(-1, 1, width, height)
        
        # [B*S, d_model, W', H']
        x_embed = self.patch_embed(x_flat)
        
        _, d_model, w_feat, h_feat = x_embed.shape
        
        # 3. 准备 Transformer 输入
        # 我们需要在时序维度上进行 Attention，对每个空间位置独立处理
        # Reshape: [B, S, D, W', H']
        x_embed = x_embed.view(batch_size, seq_len_minus_1, d_model, w_feat, h_feat)
        
        # Permute to [B, W', H', S, D] -> Flatten spatial: [B*W'*H', S, D]
        # 这样每个 (batch_idx, spatial_loc) 都有一个独立的时间序列
        x_tokens = x_embed.permute(0, 3, 4, 1, 2).contiguous()
        x_tokens = x_tokens.view(-1, seq_len_minus_1, d_model)
        
        # 4. Transformer Block
        
        # LayerNorm & Dropout
        x_tokens = self.layer_norm(x_tokens)
        x_tokens = self.dropout(x_tokens)
        
        # Attention Block
        # [N, S, D]
        attn_out = self.attention(self.norm1(x_tokens))
        x_tokens = x_tokens + attn_out  # Residual
        
        # FFN Block
        ffn_out = self.ffn(self.norm2(x_tokens))
        x_tokens = x_tokens + ffn_out   # Residual
        
        # 5. 时序聚合
        # [N, S, D] -> [N, D] (Weighted Sum)
        
        normalized_weights = F.softmax(self.temporal_weights, dim=0)
        # weights: [S, 1]
        weights = normalized_weights.view(-1, 1)
        
        # [N, S, D] * [S, 1] -> sum(dim=1) -> [N, D]
        # x_tokens: [batch*w'*h', seq, d_model]
        output_tokens = (x_tokens * weights).sum(dim=1)
        
        # 6. 恢复空间结构和分辨率
        # [B*W'*H', D] -> [B, W', H', D]
        output_tokens = output_tokens.view(batch_size, w_feat, h_feat, d_model)
        
        # [B, D, W', H'] for ConvTranspose2d
        output_tokens = output_tokens.permute(0, 3, 1, 2).contiguous()
        
        # Upsample: [B, D, W', H'] -> [B, 1, W, H]
        output_map = self.patch_recovery(output_tokens)
        
        # Remove channel dim: [B, W, H]
        output = output_map.squeeze(1)
        
        # 保存一些可视化数据 (取第一个样本的均值等，避免数据量过大)
        if batch_size > 0:
            self.viz['temporal_weights'] = normalized_weights
        
        return output
    
    def _normalize_sst(self, x):
        """海表温度数据归一化"""
        x_mask = torch.isnan(x)
        x_processed = x.clone()
        x_processed[x_mask] = 0.0
        return x_processed

    def custom_mse_loss(self, y_pred, y_true):
        """
        处理NaN值的MSE损失函数
        
        Args:
            y_pred: [batch, width, height]
            y_true: [batch, width, height]
        """
        # 创建有效值掩码
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
