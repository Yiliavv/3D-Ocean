"""
ConvLSTM: 海表温度时空序列预测模型

输入格式: [B, S-1, W, H] - 与 RGTransformer 一致
输出格式: [B, W, H]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from lightning import LightningModule


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM 单元
    
    Parameters
    ----------
    input_dim: int
        输入通道数
    hidden_dim: int
        隐藏状态通道数
    kernel_size: tuple
        卷积核大小
    bias: bool
        是否使用偏置
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding='same',
            bias=bias
        )

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        )


class ConvLSTM(LightningModule):
    """
    ConvLSTM 海表温度预测模型
    
    与 RGTransformer 接口一致：
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
        隐藏层维度
    kernel_size: tuple
        卷积核大小
    num_layers: int
        ConvLSTM 层数
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
        hidden_dim: int = 64,
        kernel_size: tuple = (3, 3),
        num_layers: int = 2,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        dropout: float = 0.1,
        **kwargs  # 忽略其他参数
    ):
        super().__init__()
        self.save_hyperparameters()

        self.width = width
        self.height = height
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # ConvLSTM 层
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = 1 if i == 0 else hidden_dim
            self.cells.append(
                ConvLSTMCell(in_dim, hidden_dim, kernel_size)
            )

        # 输出层：用卷积代替全连接，避免空间尺寸硬编码
        self.output_conv = nn.Sequential(
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )

        # 损失记录
        self.train_loss = []
        self.val_loss = []

    def forward(self, x):
        """
        前向传播
        
        Input: [B, S-1, W, H]
        Output: [B, W, H]
        """
        # 处理 NaN
        x = self._normalize(x)
        
        batch_size, seq_len, w, h = x.shape
        device = x.device

        # 添加通道维度: [B, S, W, H] -> [B, S, 1, H, W]
        # 注意：ConvLSTM 期望 [B, T, C, H, W]，这里 H=height, W=width
        x = x.unsqueeze(2)  # [B, S, 1, W, H]
        x = x.permute(0, 1, 2, 4, 3)  # [B, S, 1, H, W]

        # 初始化隐藏状态
        hidden_states = []
        for cell in self.cells:
            hidden_states.append(cell.init_hidden(batch_size, h, w, device))

        # 逐时间步处理
        for t in range(seq_len):
            input_t = x[:, t]  # [B, 1, H, W]
            
            for layer_idx, cell in enumerate(self.cells):
                h_state, c_state = hidden_states[layer_idx]
                h_state, c_state = cell(input_t, (h_state, c_state))
                hidden_states[layer_idx] = (h_state, c_state)
                input_t = h_state  # 下一层的输入

        # 最后一层的隐藏状态作为输出
        output = hidden_states[-1][0]  # [B, hidden_dim, H, W]

        # 输出层
        output = self.output_conv(output)  # [B, 1, H, W]
        
        # 调整输出格式: [B, 1, H, W] -> [B, W, H]
        output = output.squeeze(1)  # [B, H, W]
        output = output.permute(0, 2, 1)  # [B, W, H]

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
