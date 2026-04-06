"""
LSTM: 海表温度时空序列预测模型

纯 LSTM 实现，将空间维度展平后处理时序信息

输入格式: [B, S-1, W, H] - 与 RGTransformer/ConvLSTM 一致
输出格式: [B, W, H]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from lightning import LightningModule


class LSTM(LightningModule):
    """
    纯 LSTM 海表温度预测模型
    
    将空间维度展平后，使用标准 LSTM 处理时序信息
    
    与 RGTransformer/ConvLSTM 接口一致：
    - 输入: [B, S-1, W, H]
    - 输出: [B, W, H]
    
    Parameters
    ----------
    width: int
        空间宽度 (经度方向)
    height: int
        空间高度 (纬度方向)
    seq_len: int
        序列长度
    hidden_dim: int
        LSTM 隐藏层维度
    num_layers: int
        LSTM 层数
    learning_rate: float
        学习率
    weight_decay: float
        权重衰减
    dropout: float
        Dropout 比例
    """

    def __init__(
        self,
        width: int,
        height: int,
        seq_len: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 2,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.width = width
        self.height = height
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.spatial_dim = width * height

        self.input_proj = nn.Sequential(
            nn.Linear(self.spatial_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, self.spatial_dim),
        )

        self.train_loss = []
        self.val_loss = []

    def forward(self, x):
        """
        前向传播
        
        Input: [B, S-1, W, H]
        Output: [B, W, H]
        """
        x = self._normalize(x)
        
        batch_size, seq_len, w, h = x.shape

        x = x.view(batch_size, seq_len, -1)

        x = self.input_proj(x)

        lstm_out, (h_n, c_n) = self.lstm(x)

        final_hidden = lstm_out[:, -1, :]

        output = self.output_proj(final_hidden)

        output = output.view(batch_size, w, h)

        return output

    def _normalize(self, x):
        """将 NaN 替换为 0"""
        x = x.clone()
        x[torch.isnan(x)] = 0.0
        return x

    def compute_loss(self, y_pred, y_true):
        """计算损失（处理 NaN）"""
        valid_mask = ~(torch.isnan(y_true) | torch.isnan(y_pred))
        
        if valid_mask.sum() == 0:
            return y_pred.sum() * 0.0
        
        y_pred_valid = y_pred[valid_mask]
        y_true_valid = y_true[valid_mask]
        
        return F.mse_loss(y_pred_valid, y_true_valid)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.compute_loss(y_pred, y)
        
        self.train_loss.append(loss.detach().cpu().item())
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.compute_loss(y_pred, y)
        
        self.val_loss.append(loss.detach().cpu().item())
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_rmse', torch.sqrt(loss), prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=500, eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss"
            }
        }

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """获取参数量"""
        return sum(p.numel() for p in self.parameters() if (not trainable_only or p.requires_grad))
