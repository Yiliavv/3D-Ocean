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
                 resolution=1.0,
                 gradient_clip_val=1.0):
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
            gradient_clip_val: 梯度裁剪阈值，None表示不裁剪（默认: 1.0）
        """
        super().__init__()
        
        self.learning_rate = learning_rate
        self.gradient_clip_val = gradient_clip_val
        
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
        
        # 空间位置编码（基于经纬度的球谐波编码）
        if lat_range is not None and lon_range is not None:
            self.spatial_pos_encoding = SpatialSphericalHarmonicEncoding(
                lat_range=lon_range,
                lon_range=lat_range,
                d_model=d_model,
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
        
        # 层归一化（两个：一个用于attention，一个用于FFN）
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        
        self.dropout = nn.Dropout(dropout)

        # 输入归一化
        self.input_norm = nn.LayerNorm(d_model, eps=1e-5)
        
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
        batch_size = x.shape[0]
        
        # 创建掩码（在输入处理之前）
        original_input = x
        mask = self._create_mask(original_input)
        
        # 海表温度输入处理
        x_processed = self._normalize_sst(x)
        
        # 线性投影方式
        if len(x.shape) == 4:  # [batch, seq_len, width, height]
            x_reshaped = x_processed.view(batch_size, x.shape[1], -1)
        else:  # [batch, seq_len-1, width, height]
            x_reshaped = x_processed.view(batch_size, self.seq_len - 1, -1)
        
        # 在投影前嵌入空间位置编码（如果启用）
        if self.spatial_pos_encoding is not None:
            # 获取输入数据的设备
            device = x_reshaped.device
            
            spatial_enc = self.spatial_pos_encoding()  # [enc_height, enc_width, d_model] = [lat点数, lon点数, d_model]
            
            # 确保spatial_enc在正确的设备上
            spatial_enc = spatial_enc.to(device)
            
            # 尺寸匹配，直接使用
            spatial_enc_flat = spatial_enc.view(self.height * self.width, self.d_model)
            spatial_feature_1d = spatial_enc_flat[:, 0]  # [height*width] 取第一维
            
            # 确保spatial_feature_1d和spatial_weight都在正确的设备上
            spatial_weight = self.spatial_enc_scale if hasattr(self, 'spatial_enc_scale') and self.spatial_enc_scale is not None else 0.01
            if isinstance(spatial_weight, torch.Tensor):
                spatial_weight = spatial_weight.to(device)
            
            x_reshaped = x_reshaped + spatial_weight * spatial_feature_1d.unsqueeze(0).unsqueeze(0)
        
        # 投影到 d_model 维度
        x = self.input_projection(x_reshaped)  # [batch, seq_len, d_model]
        
        x = self.input_norm(x)  # [batch, seq_len, reduced_dim]
        
        x = self.dropout(x)
        
        # Transformer 单层（Pre-LN 模式）
        # LN -> Attention -> Residual -> LN -> FFN -> Residual
        
        # 注意力子层：LN -> Attention -> Dropout -> Residual
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        # FFN 子层：LN -> FFN -> Residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        # 输出投影 - 海表温度预测
        # 只使用最后一个时间步进行预测
        x = x[:, -1, :]  # [batch, d_model]
        output = self.output_projection(x)  # [batch, width*height]
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
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def on_before_optimizer_step(self, optimizer):
        # 梯度裁剪，防止梯度爆炸导致训练不稳定
        # 注意：如果 BaseTrainer 通过 trainer_params['gradient_clip_val'] 设置了裁剪，
        # Lightning Trainer 会在回调之前自动执行裁剪，所以这里只作为备用/默认值
        # 为了避免重复裁剪，这里检查是否已经有 trainer 级别的裁剪
        # （通过检查 trainer.current_epoch 是否可用来判断是否在训练中）
        if hasattr(self, 'trainer') and self.trainer is not None:
            # 如果 Trainer 配置了 gradient_clip_val，它会在回调前执行，这里不再重复
            # 只有在 Trainer 未配置时才使用模型的默认值
            if not hasattr(self.trainer, 'gradient_clip_val') or self.trainer.gradient_clip_val is None:
                if self.gradient_clip_val is not None and self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.gradient_clip_val)
        else:
            # 没有 trainer 的情况下（如直接调用模型），使用模型的默认值
            if self.gradient_clip_val is not None and self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.gradient_clip_val)