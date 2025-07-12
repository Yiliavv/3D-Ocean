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

class RecursiveAttention(nn.Module):
    """递归式自注意力模块 - 核心创新"""
    def __init__(self, d_model, num_heads, recursion_depth=2):
        super().__init__()
        self.d_model = d_model
        self.recursion_depth = recursion_depth
        
        # 递归注意力层
        self.recursive_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            for _ in range(recursion_depth)
        ])
        
        # 递归权重 - 使用温度参数控制权重分布
        self.recursion_weights = nn.Parameter(torch.ones(recursion_depth))
        self.temperature = nn.Parameter(torch.tensor(1.0))  # 可学习的温度参数
        
        # 特征细化器 - 添加Layer Norm提高稳定性
        self.refiners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(0.1)
            ) for _ in range(recursion_depth)
        ])
        
    def forward(self, x, mask=None):
        current_output = x
        recursive_outputs = []
        
        # 递归注意力计算
        for attn_layer, refiner in zip(self.recursive_layers, self.refiners):
            attn_output, _ = attn_layer(
                current_output, current_output, current_output,
                key_padding_mask=mask
            )
            
            refined_output = refiner(attn_output)
            current_output = refined_output + current_output
            recursive_outputs.append(current_output)
        
        # 权重融合 - 使用温度缩放提高稳定性
        weights = F.softmax(self.recursion_weights / self.temperature.clamp(min=0.1), dim=0)
        final_output = sum(w * output for w, output in zip(weights, recursive_outputs))
        
        # 添加残差连接提高稳定性
        final_output = final_output + x
        
        return final_output

class RecursiveAttentionLayer(nn.Module):
    """递归注意力 Transformer 层"""
    def __init__(self, d_model, num_heads, dim_feedforward=1024, dropout=0.1, recursion_depth=2):
        super().__init__()
        
        # 递归注意力机制
        self.recursive_attention = RecursiveAttention(d_model, num_heads, recursion_depth)
        
        # 标准组件
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 递归注意力 + 残差连接
        attn_output = self.recursive_attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # 前馈网络 + 残差连接
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output
        
        return x

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
                 spatial_aware=False):
        super().__init__()
        
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.spatial_aware = spatial_aware
        
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
            
        # 位置编码
        if seq_len:
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
        
        # 参数初始化
        self._init_parameters()
        
        # 添加EMA（指数移动平均）提高模型稳定性
        self.ema_decay = 0.999
        self.ema_params = {}
        
        # 训练监控
        self.save_hyperparameters()
        
    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def _init_parameters(self):
        """参数初始化 - 针对递归注意力优化"""
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
    
    def create_ensemble_predictions(self, x, num_models=5):
        """
        创建集成预测以提高精度
        
        Args:
            x: 输入数据
            num_models: 集成模型数量
            
        Returns:
            torch.Tensor: 集成预测结果
        """
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(num_models):
                # 使用不同的dropout配置
                self.train()
                pred = self(x)
                predictions.append(pred)
        
        # 计算集成预测
        ensemble_pred = torch.stack(predictions)
        
        # 使用加权平均，中间的预测给予更高权重
        weights = torch.softmax(torch.linspace(0.5, 1.0, num_models), dim=0)
        weighted_pred = (ensemble_pred * weights.view(-1, 1, 1, 1, 1)).sum(dim=0)
        
        # 计算预测不确定性
        uncertainty = torch.std(ensemble_pred, dim=0)
        
        self.eval()
        return weighted_pred, uncertainty