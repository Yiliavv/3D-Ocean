"""
递归注意力 Transformer 模型
专为海表温度预测任务设计，具有创新的递归式自注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch import optim
import math

# 导入分离的模块
from src.models.PE.SphericalHarmonicEncoding import SphericalHarmonicEncoding
from src.models.Attention.RGAttention import RecursiveAttentionLayer

class RecursiveAttentionTransformer(LightningModule):
    """递归注意力 Transformer - 海表温度预测模型"""
    
    def __init__(self, 
                 width, height, seq_len,
                 d_model=256, 
                 num_heads=8, 
                 num_layers=4,
                 dim_feedforward=1024,
                 dropout=0.1,
                 recursion_depth=2,
                 learning_rate=1e-4,
                 spatial_aware=False,
                 use_spherical_harmonics=True,
                 max_harmonic_degree=4):
        super().__init__()
        
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.spatial_aware = spatial_aware
        self.use_spherical_harmonics = use_spherical_harmonics
        
        # 训练稳定性配置
        self.gradient_clip_val = 1.0
        
        # 输入处理 - 专为海表温度任务设计
        if not (width and height):
            raise ValueError("Must specify width and height for sea surface temperature data")
            
        self.input_projection = nn.Linear(width * height, d_model)
        self.width = width
        self.height = height
        self.seq_len = seq_len
        
        # 空间位置编码 - 在投影前应用
        if spatial_aware:
            self.spatial_encoding = nn.Parameter(
                torch.randn(width * height) * 0.02
            )
            
        # 球谐波位置编码神经网络
        if seq_len and use_spherical_harmonics:
            self.pos_encoding = SphericalHarmonicEncoding(
                seq_len, d_model, max_harmonic_degree
            )
        elif seq_len:
            # 保留原始位置编码作为备选
            self.pos_encoding = self._create_positional_encoding(seq_len, d_model)
        
        # 递归注意力 Transformer 层
        self.layers = nn.ModuleList([
            RecursiveAttentionLayer(
                d_model=d_model, 
                num_heads=num_heads, 
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                recursion_depth=recursion_depth
            )
            for _ in range(num_layers)
        ])
        
        # 输出处理 - 映射到海表温度空间维度
        self.output_projection = nn.Linear(d_model, width * height)
        
        self.dropout = nn.Dropout(dropout)
        
        # 损失记录
        self.train_loss = []
        self.val_loss = []
        
        # EMA 相关属性初始化
        self.ema_params = {}
        self.ema_decay = 0.999
        
        # 参数初始化
        self._init_parameters()
        
    def _create_positional_encoding(self, max_len, d_model):
        """保留原始位置编码作为备选"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def _init_parameters(self):
        """参数初始化 - 针对递归注意力和球谐波编码优化"""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if 'recursion_weights' in name:
                    # 递归权重使用较小的初始化
                    nn.init.constant_(p, 0.5)
                elif 'spatial_encoding' in name:
                    # 空间编码使用更小的初始化
                    nn.init.normal_(p, 0, 0.01)
                elif 'weight' in name and ('input_projection' in name or 'output_projection' in name):
                    # 投影层使用Xavier初始化但缩放更小
                    nn.init.xavier_uniform_(p, gain=0.5)
                else:
                    # 其他层使用标准Xavier初始化
                    nn.init.xavier_uniform_(p)
            elif p.dim() == 1:
                # 偏置项初始化为0
                nn.init.constant_(p, 0)
    
    def _create_mask(self, x):
        """创建掩码（用于海表温度的NaN值）"""
        # 为海表温度数据创建NaN掩码
        if len(x.shape) == 4:  # [batch, seq_len, width, height]
            nan_mask = torch.isnan(x)
            # 聚合到序列维度
            mask = nan_mask.any(dim=(2, 3))  # [batch, seq_len]
            return mask
        elif len(x.shape) == 3:  # [batch, seq_len, features] (处理后的格式)
            # 对于已经处理过的输入，不需要掩码
            return None
        return None
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 创建掩码（在输入处理之前）
        original_input = x
        mask = self._create_mask(original_input)
        
        # 海表温度输入处理
        x_processed = self._normalize_sst(x)
        # 重塑输入
        if len(x.shape) == 4:  # [batch, seq_len, width, height]
            x_reshaped = x_processed.view(batch_size, x.shape[1], -1)
        else:  # [batch, seq_len-1, width, height]
            x_reshaped = x_processed.view(batch_size, self.seq_len - 1, -1)
        
        # 在投影前添加空间编码
        if self.spatial_aware:
            # spatial_encoding: [width*height] -> [1, 1, width*height]
            spatial_bias = self.spatial_encoding.unsqueeze(0).unsqueeze(0)  # [1, 1, width*height]
            x_reshaped = x_reshaped + spatial_bias  # 广播添加空间位置编码
        
        x = self.input_projection(x_reshaped)
        
        # 位置编码
        if hasattr(self, 'pos_encoding'):
            if isinstance(self.pos_encoding, SphericalHarmonicEncoding):
                # 使用球谐波神经网络编码
                pos_enc = self.pos_encoding()  # [1, seq_len, d_model]
                x = x + pos_enc
            else:
                # 使用传统位置编码
                seq_len = x.shape[1]
                x = x + self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x)
        
        # 通过所有层
        for layer in self.layers:
            x = layer(x, mask)
        
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
        """处理NaN值的MSE损失函数"""
        y_mask = torch.isnan(y_true)
        valid_mask = ~y_mask
        
        if valid_mask.sum() > 0:
            y_pred_valid = y_pred[valid_mask]
            y_true_valid = y_true[valid_mask]
            loss = F.mse_loss(y_pred_valid, y_true_valid, reduction='mean')
            return loss
        else:
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        
        # 海表温度特殊处理 - 处理NaN值
        y_mask = torch.isnan(y)
        y_pred[y_mask] = float('nan')
        
        # 使用简单稳定的损失函数
        loss = self.custom_mse_loss(y_pred, y)
        
        # 梯度裁剪
        if self.trainer is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.gradient_clip_val)
        
        self.log('train_loss', loss, prog_bar=True)
        self.train_loss.append(loss.item())
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """训练批次结束时更新EMA"""
        if not self.ema_params:
            # 初始化EMA参数
            for name, param in self.named_parameters():
                if param.requires_grad:
                    self.ema_params[name] = param.data.clone()
        else:
            # 更新EMA参数
            for name, param in self.named_parameters():
                if param.requires_grad and name in self.ema_params:
                    self.ema_params[name] = (
                        self.ema_decay * self.ema_params[name] + 
                        (1 - self.ema_decay) * param.data
                    )
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        
        # 使用自定义MSE损失处理NaN值
        val_loss = self.custom_mse_loss(y_pred, y)
        
        # 添加更多的监控指标
        with torch.no_grad():
            y_mask = torch.isnan(y)
            valid_mask = ~y_mask
            
            if valid_mask.sum() > 0:
                y_pred_valid = y_pred[valid_mask]
                y_true_valid = y[valid_mask]
                
                # 计算MAE
                mae = F.l1_loss(y_pred_valid, y_true_valid)
                
                # 计算RMSE
                rmse = torch.sqrt(F.mse_loss(y_pred_valid, y_true_valid))
                
                self.log('val_mae', mae, prog_bar=True)
                self.log('val_rmse', rmse, prog_bar=True)
        
        self.log('val_loss', val_loss, prog_bar=True)
        self.val_loss.append(val_loss.item())
        return val_loss
    
    def configure_optimizers(self):
        # 简化的优化器配置，专注于稳定训练
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 使用简单的指数衰减学习率调度
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=0.99  # 每个epoch学习率衰减1%
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
