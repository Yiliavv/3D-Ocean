import numpy as np

import torch
from torch import manual_seed, optim, nn
from torch.nn import Transformer, Linear
from lightning import LightningModule

class SSTTransformer(LightningModule):
    """ 
    SSTTransformer 模型
    
    参数:
        width: int, 宽度
        height: int, 高度
        seq_len: int, 序列长度
        d_model: int, 模型维度
        learning_rate: float, 学习率
        optimizer: optim.Optimizer, 优化器
        **kwargs: Transformer 模型参数
        
    主要的模型参数:
        nhead: 多头注意力机制的head数
        num_encoder_layers: 编码器层数
        num_decoder_layers: 解码器层数
        dim_feedforward: 前馈神经网络的维度
        dropout: 丢弃率
        activation: 激活函数
    """
    def __init__(self,
                 width,
                 height,
                 seq_len,
                 d_model = 512,
                 learning_rate = 1e-4,
                 optimizer = optim.AdamW,
                 **kwargs):

        manual_seed(1)
        super().__init__()
        
        # 修改模型参数
        self.learning_rate = learning_rate
        self.d_model = d_model
        self.optimizer = optimizer

        self.transformer = Transformer(
            d_model=self.d_model,
            batch_first=True,
            norm_first=True,
            **kwargs
        )
        
        self.width = width
        self.height = height
        self.seq_len = seq_len
        
        # 输入投影层：将空间特征映射到d_model维度
        self.input_projection = Linear(width * height, self.d_model)
        # 输出投影层：将d_model维度映射回(20,20)空间
        self.output_projection = Linear(self.d_model, width * height)
        
        # 训练损失
        self.train_loss = []
        # 验证损失
        self.val_loss = []
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        x_processed = self.__normalize__(x)

        # 重塑输入
        x_processed = x_processed.view(batch_size, self.seq_len - 1, -1)
        
        # 投影到transformer维度
        x_viewed = self.input_projection(x_processed)

        # 创建一个目标序列（用最后一个时间步作为初始目标）
        tgt = x_viewed[:, -1:, :]  # [batch, 1, d_model]

        # 通过transformer
        output = self.transformer(x_viewed, tgt)  # [1, batch, d_model]
        
        # 投影回空间维度
        output = self.output_projection(output)  # [batch, 1, 400]
        
        # 重塑回空间维度 [batch, 1, 20, 20]
        output = output.view(batch_size, 1, self.width, self.height)
        
        return output
    
    def custom_mse_loss(self, y_pred, y):
        """
        自定义MSE损失函数，忽略nan值
        """
        
        y_mask = torch.isnan(y)
        
        # 创建掩码，标记非nan值
        mask = ~y_mask
        
        # 只计算非nan值的MSE
        if mask.sum() > 0:  # 确保有非nan值
            # 将y_pred中的nan值替换为0，以便计算损失
            y_pred_processed = y_pred.clone()
            y_pred_processed[y_mask] = 0.0
            
            # 将y中的nan值替换为0
            y_processed = y.clone()
            y_processed[y_mask] = 0.0
            
            # 计算MSE，只考虑非nan位置
            return nn.MSELoss()(y_pred_processed, y_processed)
        else:
            # 如果所有值都是nan，返回0
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
        
    
    def training_step(self, batch):
        x, y = batch  # x: [batch, 14, 20, 20], y: [batch, 1, 20, 20]
        
        y_mask = torch.isnan(y)
        
        y_pred = self(x)
        
        y_pred[y_mask] = np.nan
        
        loss = self.custom_mse_loss(y_pred, y)
        
        self.log('train_loss', loss, prog_bar=True)
        
        self.train_loss.append(loss.item())
        
        return loss

    def validation_step(self, batch):
        x, y = batch
        
        y_pred = self(x)
        
        val_loss = self.custom_mse_loss(y_pred, y)
        
        self.log('val_loss', val_loss)
        
        self.val_loss.append(val_loss.item())
        
        return val_loss
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        
        # 余弦退火学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
    
    def __normalize__(self, x):
        # 保存原始nan掩码
        x_mask = torch.isnan(x)
        # 将nan值替换为0，以便模型处理
        x_processed = x.clone()
        x_processed[x_mask] = 0.0
        
        return x_processed

