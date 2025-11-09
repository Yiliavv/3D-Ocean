"""
RG-Transformer: 递归泛化注意力 Transformer 模型
专为海表温度时空序列预测任务设计

架构设计：
- 采用标准 Transformer 模式：自己组装 Block，而不是依赖外部 Block 类
- 注意力机制：使用 RGAttention（递归泛化自注意力，参数共享）
- FFN：使用标准前馈网络（GELU 激活）
- 归一化：Pre-LN 模式（训练更稳定）
- 残差连接：可学习的缩放因子

优势：
- 灵活性高：可以自由调整每层的组成
- 职责清晰：注意力模块只负责注意力，Transformer 负责组装
- 易于调试：Block 的组装逻辑都在一处
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning import LightningModule
from torch import optim

# 导入注意力模块（只包含注意力计算，不包含 FFN）
from src.models.SST.PE.SphericalHarmonicEncoding import (
    SpatialSphericalHarmonicEncoding,
)
from src.models.SST.Attention.RGAttention import RGAttention


class FeedForward(nn.Module):
    """
    标准的前馈网络（FFN）
    
    结构: Linear -> GELU -> Dropout -> Linear -> Dropout
    这是 Transformer 的标准 FFN 实现
    """
    def __init__(self, width, height, dim_feedforward, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(width * height, dim_feedforward)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(dim_feedforward, width * height)

    def forward(self, x):
        batch_size, seq_len, width, height = x.shape

        # [batch, seq_len, width, height] -> [batch, seq_len, dim_feedforward]
        x = x.view(batch_size, seq_len, width * height)

        # [batch, seq_len, width * height] -> [batch, seq_len, dim_feedforward]
        x = self.linear(x)
        x = self.gelu(x)

        # [batch, seq_len, dim_feedforward] -> [batch, seq_len, width * height]
        x = self.linear2(x)
        x = x.view(batch_size, seq_len, width, height)

        return x
        
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
                 resolution=1.0):
        """
        Args:
            width: 海表温度图像宽度
            height: 海表温度图像高度
            seq_len: 输入序列长度
            d_model: 模型维度
            num_heads: 注意力头数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比例
            recursion_depth: RG注意力的递归深度（递归多少次）
            learning_rate: 学习率
            lat_range: [lat_min, lat_max] 纬度范围（度），用于空间位置编码
            lon_range: [lon_min, lon_max] 经度范围（度），用于空间位置编码
            resolution: 空间分辨率（度）
        """
        super().__init__()
        
        self.learning_rate = learning_rate
        
        # 输入处理 - 专为海表温度任务设计
        if not (width and height):
            raise ValueError("Must specify width and height for sea surface temperature data")
        
        self.width = width
        self.height = height
        self.seq_len = seq_len
        
        self.d_model = d_model
        
        self.spatial_pos_encoding = SpatialSphericalHarmonicEncoding(
            lat_range=lat_range,
            lon_range=lon_range,
            max_degree=2,
            resolution=resolution
        )
        # 可学习的空间编码缩放因子（用于稳定训练）
        self.spatial_enc_scale = nn.Parameter(torch.tensor(0.1))
    
        # Transformer 层的组件（直接在这里组装，不使用外部 Block 类）
        # 单层 Transformer：注意力 + FFN + 归一化 + 残差连接
        
        # RG递归注意力模块
        self.attention = RGAttention(
            d_model=d_model,
            num_heads=num_heads,
            recursion_depth=recursion_depth,
            dropout=dropout
        )
        
        # 前馈网络（标准 FFN with GELU）
        self.ffn = FeedForward(width, height, dim_feedforward, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
        # LayerNorm 层（在__init__中初始化，确保设备一致性）
        # 对空间维度 [width, height] 进行归一化
        self.layer_norm = nn.LayerNorm((width, height), eps=1e-5)
        self.norm1 = nn.LayerNorm((width, height), eps=1e-5)
        self.norm2 = nn.LayerNorm((width, height), eps=1e-5)
        
        # 可学习的时序权重向量：用于加权聚合所有时间步的信息
        self.temporal_weights = nn.Parameter(torch.ones(seq_len - 1) / (seq_len - 1))

        # 用来记录可能需要可视化的数据
        self.viz = {}
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch, seq_len-1, width, height]
        
        Returns:
            output: [batch, width, height]
        """        
        # 海表温度输入处理 - 先处理 NaN，确保后续计算不会产生 NaN
        x = self._normalize_sst(x)

        # ======= 生成空间位置编码 =======
        
        # [width, height]
        spatial_enc = self.spatial_pos_encoding()
            
        # 确保spatial_weight都在正确的设备上
        spatial_weight = self.spatial_enc_scale if hasattr(self, 'spatial_enc_scale') and self.spatial_enc_scale is not None else 0.01

        self.viz['position_encoding'] = spatial_enc

        # ======= 添加空间位置编码 =======
        
        # [batch, seq_len-1, width, height]
        # 将空间位置编码添加到输入中
        x = x + spatial_enc * spatial_weight

        self.viz['sst_after_position_encoding'] = x

        # ======= LayerNorm 和 Dropout =======

        # [batch, seq_len-1, width, height]
        # 注意：LayerNorm 在 normalize 之后
        x = self.layer_norm(x)
        x = self.dropout(x)

        self.viz['x_normed'] = x

        # ======= 注意力计算 =======
    
        # [batch, seq_len-1, width, height]
        # 不使用 mask，因为 NaN 已经通过 normalize 处理了
        attn_out = self.attention(self.norm1(x))

        self.viz['attention'] = attn_out

        # ======= 计算 FFN =======

        ffn_out = self.ffn(self.norm2(x))

        self.viz['ffn'] = ffn_out

        # ======= 添加注意力输出（标准残差连接） =======
        
        # [batch, seq_len-1, width, height]
        x = x + attn_out
        
        self.viz['sst_after_attention'] = x

        # ======= 添加 FFN 输出（标准残差连接） =======
        
        # [batch, seq_len-1, width, height]
        x = x + ffn_out

        self.viz['sst_after'] = x

        # ======= 可学习的时序聚合 =======
        # [batch, seq_len-1, width, height] -> [batch, width, height]
        
        # 对权重进行 softmax 归一化，确保权重和为1
        normalized_weights = F.softmax(self.temporal_weights, dim=0)  # [seq_len-1]
        
        # 记录时序权重用于可视化
        self.viz['temporal_weights'] = normalized_weights
        
        # 对每个时间步应用权重并加权求和
        # normalized_weights: [seq_len-1] -> [1, seq_len-1, 1, 1]
        weights = normalized_weights.view(1, -1, 1, 1)
        
        # 加权求和：[batch, seq_len-1, width, height] -> [batch, width, height]
        output = (x * weights).sum(dim=1)
        
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
        
        海洋数据中陆地区域为NaN，此函数只计算有效海洋区域的损失
        
        Args:
            y_pred: 模型预测值 [batch, channels, height, width]
            y_true: 真实值 [batch, channels, height, width]
        
        Returns:
            loss: MSE损失值，如果没有有效值则返回0（保持在计算图中）
        """
        # 创建有效值掩码：同时考虑y_true和y_pred的NaN
        # 只有当y_true和y_pred都不是NaN的位置才参与损失计算
        y_true_mask = torch.isnan(y_true)
        y_pred_mask = torch.isnan(y_pred)
        valid_mask = ~(y_true_mask | y_pred_mask)  # 两个都不是NaN才有效
        
        # 统计有效值数量
        num_valid = valid_mask.sum()
        
        if num_valid > 0:
            # 只对有效区域计算损失
            y_pred_valid = y_pred[valid_mask]
            y_true_valid = y_true[valid_mask]
            loss = F.mse_loss(y_pred_valid, y_true_valid, reduction='mean')
            return loss
        else:
            # 没有有效值时返回0，但保持在计算图中
            return y_pred.sum() * 0.0
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        
        # 使用自定义MSE损失（自动处理NaN值）
        loss = self.custom_mse_loss(y_pred, y)
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        
        # 使用自定义MSE损失处理NaN值
        val_loss = self.custom_mse_loss(y_pred, y)
        
        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)

        return val_loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=0.01,  # 添加权重衰减
            betas=(0.9, 0.999),
            eps=1e-8
        )

        return optimizer