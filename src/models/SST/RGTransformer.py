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
from lightning import LightningModule, Callback
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
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Linear -> GELU -> Dropout -> Linear -> Dropout
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
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
        
        # 线性投影层（使用瓶颈结构减少参数量）
        # 计算中间维度：使用 d_model/2 作为瓶颈，可减少约50%参数
        bottleneck_dim = d_model // 2
        self.input_projection = nn.Sequential(
            nn.Linear(width * height, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, d_model)
        )
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, width * height)
        )
        
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
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)
        
        # 层归一化（Pre-LN模式：每个子层前归一化）
        # norm1: 注意力层前的归一化
        # norm2: FFN层前的归一化
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-5)
        
        # 时间位置编码（可学习的嵌入）
        # 用于区分序列中的不同时间步，解决注意力权重均匀分布的问题
        # max_seq_len 取 seq_len 和 seq_len-1 的最大值（处理两种情况）
        max_seq_len = max(seq_len, seq_len - 1)
        self.temporal_pos_encoding = nn.Embedding(max_seq_len, d_model)
        # 可学习的时间编码缩放因子（用于稳定训练）
        self.temporal_enc_scale = nn.Parameter(torch.tensor(0.1))
        
        # 输出聚合：可学习的查询向量，用于从所有时间步提取信息
        # 替代简单的 "只使用最后一个时间步" 的方式
        self.output_query = nn.Parameter(torch.randn(1, 1, d_model))
        # 初始化查询向量（使用 Xavier 初始化）
        nn.init.xavier_uniform_(self.output_query, gain=0.1)


        # 用来记录可能需要可视化的数据
        self.viz = {
            'position_encoding': None,
            'attention_weights': None,
            'sst_after_ffn': None,
            'sst_after_attention': None,
            'sst_after_position_encoding': None,
            'sst_after_temporal_encoding': None,
        }
        
    def _create_mask(self, x):
        """
        创建掩码（用于海表温度的NaN值）
        
        对于4-D输入 [batch, seq_len, width, height]，返回 [batch, seq_len]
        对于其他形状，返回 None（不使用掩码）
        """
        if len(x.shape) == 4:
            # [batch, seq_len, width, height]
            nan_mask = torch.isnan(x)
            # 聚合到序列维度：如果某个时间步的空间位置中有NaN，则该时间步被掩码
            mask = nan_mask.any(dim=(2, 3))  # [batch, seq_len]
            return mask
        else:
            # 对于其他形状，不使用掩码
            return None
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入数据 [batch, seq_len, width, height] 或 [batch, seq_len-1, width, height]
        
        Returns:
            output: 预测结果 [batch, 1, width, height]
        """
        batch_size = x.shape[0]
        
        # 创建掩码（在输入处理之前）
        original_input = x
        mask = self._create_mask(original_input)
        
        # 海表温度输入处理
        x = self._normalize_sst(x)


        # ======= 生成空间位置编码 =======
            
        spatial_enc = self.spatial_pos_encoding()  # [lat点数, lon点数]
            
        # 确保spatial_weight都在正确的设备上
        spatial_weight = self.spatial_enc_scale if hasattr(self, 'spatial_enc_scale') and self.spatial_enc_scale is not None else 0.01

        self.viz['position_encoding'] = spatial_enc

        # ======= 添加空间位置编码 =======
            
        x = x + spatial_enc * spatial_weight

        self.viz['sst_after_position_encoding'] = x

        # ======= 输入处理 =======

        if len(x.shape) == 4:  # [batch, seq_len, width, height]
            x = x.view(batch_size, x.shape[1], -1)
        else:  # [batch, seq_len-1, width, height]
            x = x.view(batch_size, self.seq_len - 1, -1)
        
        # ======= 投影到 d_model 维度 =======
    
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # ======= 添加时间位置编码 =======

        seq_len_actual = x.shape[1]
        temporal_pos = torch.arange(seq_len_actual, device=x.device)  # [seq_len]
        temporal_enc = self.temporal_pos_encoding(temporal_pos)  # [seq_len, d_model]
        # 使用可学习的缩放因子，并添加到所有batch中
        x = x + self.temporal_enc_scale * temporal_enc.unsqueeze(0)  # [batch, seq_len, d_model]

        self.viz['sst_after_temporal_encoding'] = x
        
        # ======= 添加 dropout =======
        x = self.layer_norm(x)
        x = self.dropout(x)

        # ======= 注意力计算 =======
    
        attn_out = self.attention(self.norm1(x), mask)

        self.viz['attention_weights'] = attn_out

        # ======= 添加注意力输出 =======

        x = x + self.dropout(attn_out)

        self.viz['sst_after_attention'] = x
        

        # ======= 添加 FFN =======

        ffn_out = self.ffn(self.norm2(x))

        x = x + ffn_out

        self.viz['sst_after_ffn'] = x
        
        # ======= 使用可学习的查询向量与所有时间步计算注意力权重 =======
        query = self.output_query.expand(batch_size, -1, -1)  # [batch, 1, d_model]
        
        # 计算查询对序列的注意力分数
        # 使用点积注意力：query @ x^T / sqrt(d_model)
        query_attention_scores = torch.matmul(query, x.transpose(1, 2)) / (self.d_model ** 0.5)  # [batch, 1, seq_len]
        query_attention = F.softmax(query_attention_scores, dim=-1)  # [batch, 1, seq_len]
        
        # 加权聚合所有时间步
        x_aggregated = torch.matmul(query_attention, x).squeeze(1)  # [batch, d_model]

        # ======= 投影到空间维度 =======
        output = self.output_projection(x_aggregated)  # [batch, width*height]
        # 重塑回空间维度
        output = output.view(batch_size, 1, self.width, self.height)
        
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
        return optim.AdamW(self.parameters(), lr=self.learning_rate)