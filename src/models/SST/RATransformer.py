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
from src.models.SST.PE.SphericalHarmonicEncoding import SphericalHarmonicEncoding
from src.models.SST.Attention.RGAttention import RecursiveAttentionLayer

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
                 use_spherical_harmonics=True,
                 max_harmonic_degree=4,
                 attention_mode='hierarchical',
                 norm_first=True,
                 use_conv_embedding=True,
                 warmup_epochs=10):
        """
        Args:
            width: 海表温度图像宽度
            height: 海表温度图像高度
            seq_len: 输入序列长度
            d_model: 模型维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比例
            recursion_depth: 递归/层次深度
            learning_rate: 学习率
            use_spherical_harmonics: 是否使用球谐波时序位置编码
            max_harmonic_degree: 球谐波最大阶数（已弃用）
            attention_mode: 注意力模式 ('true_recursive' 或 'hierarchical')
            norm_first: 是否使用Pre-LN归一化（推荐True）
            use_conv_embedding: 是否使用卷积嵌入层（保留空间结构）
            warmup_epochs: 学习率预热周期数
        """
        super().__init__()
        
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.use_spherical_harmonics = use_spherical_harmonics
        self.attention_mode = attention_mode
        self.norm_first = norm_first
        self.use_conv_embedding = use_conv_embedding
        self.warmup_epochs = warmup_epochs
        
        # 训练稳定性配置
        self.gradient_clip_val = 1.0
        
        # 输入处理 - 专为海表温度任务设计
        if not (width and height):
            raise ValueError("Must specify width and height for sea surface temperature data")
        
        self.width = width
        self.height = height
        self.seq_len = seq_len
        
        # 使用卷积嵌入层来保留空间局部性（类似ConvLSTM的优势）
        if use_conv_embedding:
            self.conv_embedding = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((16, 16))  # 降维到固定大小
            )
            # 卷积后的特征大小
            conv_out_size = 64 * 16 * 16
            self.input_projection = nn.Linear(conv_out_size, d_model)
        else:
            self.input_projection = nn.Linear(width * height, d_model)
        
        # 添加输入 BatchNorm
        self.input_norm = nn.LayerNorm(d_model)
        
        # 时序位置编码（用于Transformer序列维度）
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
                recursion_depth=recursion_depth,
                attention_mode=attention_mode,
                norm_first=norm_first
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
        """参数初始化 - 针对递归注意力优化"""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if 'recursion_weights' in name or 'step_weights' in name or 'level_weights' in name:
                    # 递归/层次权重使用较小的初始化
                    nn.init.constant_(p, 0.5)
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
        
        # 使用卷积嵌入保留空间结构
        if self.use_conv_embedding:
            # x_processed: [batch, seq_len-1, height, width]
            seq_len = x_processed.shape[1]
            # 重塑为 [batch*seq_len, 1, height, width] 用于卷积
            x_reshaped = x_processed.view(batch_size * seq_len, 1, self.height, self.width)
            # 通过卷积嵌入
            x_conv = self.conv_embedding(x_reshaped)  # [batch*seq_len, 64, 16, 16]
            # 展平卷积特征
            x_conv_flat = x_conv.view(batch_size * seq_len, -1)  # [batch*seq_len, 64*16*16]
            # 投影到模型维度
            x = self.input_projection(x_conv_flat)  # [batch*seq_len, d_model]
            # 重塑回序列格式
            x = x.view(batch_size, seq_len, self.d_model)  # [batch, seq_len, d_model]
        else:
            # 原始线性投影方式
            if len(x.shape) == 4:  # [batch, seq_len, width, height]
                x_reshaped = x_processed.view(batch_size, x.shape[1], -1)
            else:  # [batch, seq_len-1, width, height]
                x_reshaped = x_processed.view(batch_size, self.seq_len - 1, -1)
            # 投影到模型维度
            x = self.input_projection(x_reshaped)
        
        # 输入归一化
        x = self.input_norm(x)
        
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
        """
        处理NaN值的MSE损失函数
        
        海洋数据中陆地区域为NaN，此函数只计算有效海洋区域的损失
        
        Args:
            y_pred: 模型预测值 [batch, channels, height, width]
            y_true: 真实值 [batch, channels, height, width]
        
        Returns:
            loss: MSE损失值，如果没有有效值则返回0（保持在计算图中）
        """
        # 创建有效值掩码（非NaN的位置）
        y_mask = torch.isnan(y_true)
        valid_mask = ~y_mask
        
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
            # 使用 y_pred.sum() * 0.0 确保梯度流动
            return y_pred.sum() * 0.0
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        
        # 使用自定义MSE损失（自动处理NaN值）
        loss = self.custom_mse_loss(y_pred, y)
        
        # 检查损失有效性
        if torch.isnan(loss) or torch.isinf(loss):
            # 如果损失异常，记录警告并返回零损失
            if batch_idx % 100 == 0:  # 避免日志过多
                print(f"⚠️  Warning: Invalid loss at batch {batch_idx}")
            return y_pred.sum() * 0.0
        
        # Lightning 会自动处理梯度裁剪（如果在 Trainer 中配置了）
        # 但我们也可以在这里显式裁剪
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.gradient_clip_val)
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
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
        
        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.val_loss.append(val_loss.item())
        return val_loss
    
    def configure_optimizers(self):
        # 改进的优化器配置，借鉴ConvLSTM的稳定性
        # 使用更小的学习率和更大的权重衰减
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.05,  # 增加正则化
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 使用带预热的余弦退火学习率调度器
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                # 预热阶段：线性增加学习率
                return (epoch + 1) / self.warmup_epochs
            else:
                # 余弦退火阶段
                progress = (epoch - self.warmup_epochs) / (200 - self.warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
