import numpy as np

import torch
from torch import manual_seed, optim, nn
from torch.nn import Transformer, Linear
from lightning import LightningModule

from src.utils.log import CSVLogger
from src.config.params import BASE_CSV_PATH

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
    """
    def __init__(self,
                 width,
                 height,
                 seq_len,
                 d_model = 2048,
                 learning_rate = 1e-4,
                 optimizer = optim.AdamW,
                 **kwargs):

        manual_seed(1)
        super().__init__()
        
        # 修改模型参数
        self.learning_rate = learning_rate
        self.d_model = d_model  # 增加模型维度以处理复杂的空间信息
        self.optimizer = optimizer

        self.transformer = Transformer(
            d_model=self.d_model, 
            batch_first=True,
            **kwargs
        )
        
        self.width = width
        self.height = height
        self.seq_len = seq_len
        
        self.batch_norm = nn.BatchNorm1d(1)
        
        # 输入投影层：将(20,20)的空间特征映射到d_model维度
        self.input_projection = Linear(width * height, self.d_model)
        # 输出投影层：将d_model维度映射回(20,20)空间
        self.output_projection = Linear(self.d_model, width * height)
    
    def forward(self, x):
        batch_size = x.shape[0]
            
        # 保存原始nan掩码
        x_mask = torch.isnan(x)
        
        # 将nan值替换为0，以便模型处理
        x_processed = x.clone()
        x_processed[x_mask] = 0.0

        # 重塑输入: [batch, 14, 20, 20] -> [batch, 14, 400]
        x_processed = x_processed.view(batch_size, self.seq_len - 1, -1)
        
        # 投影到transformer维度: [batch, 14, 400] -> [batch, 14, d_model]
        x_processed = self.input_projection(x_processed)
        
        # 创建一个目标序列（用最后一个时间步作为初始目标）
        tgt = x_processed[:, -1:, :]  # [batch, 1, d_model]
        
        tgt = self.batch_norm(tgt)
        
        # 通过transformer
        output = self.transformer(x_processed, tgt)  # [1, batch, d_model]
        
        # 投影回空间维度
        output = self.output_projection(output)  # [batch, 1, 400]
        
        # 重塑回空间维度 [batch, 1, 20, 20]
        output = output.view(batch_size, 1, self.width, self.height)
        
        return output
    
    def custom_mse_loss(self, y_pred, y):
        """
        自定义MSE损失函数，忽略nan值
        """
        
        # 创建掩码，标记非nan值
        mask = ~torch.isnan(y)
        
        # 只计算非nan值的MSE
        if mask.sum() > 0:  # 确保有非nan值
            # 将y_pred中的nan值替换为0，以便计算损失
            y_pred_processed = y_pred.clone()
            y_pred_processed[torch.isnan(y_pred)] = 0.0
            
            # 将y中的nan值替换为0
            y_processed = y.clone()
            y_processed[torch.isnan(y)] = 0.0
            
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
        
        return loss

    def validation_step(self, batch):
        x, y = batch
        
        y_pred = self(x)
        
        val_loss = self.custom_mse_loss(y_pred, y)
        
        self.log('val_loss', val_loss)
        
        return val_loss
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.scheduler.step()

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        return { "optimizer": optimizer }

