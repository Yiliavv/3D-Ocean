"""
ResNet 深度重建器 - 基于残差网络的3D温度场重建
使用全局特征提取 + 深度生成的方式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch import optim


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class ResNetDepthReconstructor(LightningModule):
    """
    ResNet 深度重建器
    
    策略：先提取空间特征，再生成深度维度
    
    输入：海表温度 [batch, 1, height, width]
    输出：三维温度场 [batch, depth, height, width]
    
    参数：
        n_channels: 输入通道数（默认1）
        n_depth: 输出深度层数
        base_channels: 基础通道数
        n_residual_blocks: 残差块数量
        learning_rate: 学习率
    """
    
    def __init__(self,
                 n_channels=1,
                 n_depth=10,
                 base_channels=128,
                 n_residual_blocks=8,
                 dropout=0.1,
                 learning_rate=1e-4):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_depth = n_depth
        self.learning_rate = learning_rate
        
        # 初始卷积
        self.init_conv = nn.Sequential(
            nn.Conv2d(n_channels, base_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # 残差块堆叠
        self.res_blocks = nn.ModuleList([
            ResidualBlock(base_channels, dropout) 
            for _ in range(n_residual_blocks)
        ])
        
        # 深度生成分支
        self.depth_generator = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, n_depth, kernel_size=1)
        )
        
        # 损失记录
        self.train_loss = []
        self.val_loss = []
        
        self._init_weights()
    
    def _init_weights(self):
        """参数初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch, 1, height, width]
        
        Returns:
            output: [batch, depth, height, width]
        """
        # 初始特征提取
        x = self.init_conv(x)
        
        # 残差块处理
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # 生成深度维度
        output = self.depth_generator(x)
        
        return output
    
    def custom_mse_loss(self, y_pred, y_true):
        """处理NaN值的MSE损失"""
        # 调整维度
        if len(y_true.shape) == 4 and y_true.shape[-1] < y_true.shape[1]:
            y_true = y_true.permute(0, 3, 1, 2)
        
        y_mask = torch.isnan(y_true)
        valid_mask = ~y_mask
        
        num_valid = valid_mask.sum()
        
        if num_valid > 0:
            y_pred_valid = y_pred[valid_mask]
            y_true_valid = y_true[valid_mask]
            loss = F.mse_loss(y_pred_valid, y_true_valid, reduction='mean')
            return loss
        else:
            return y_pred.sum() * 0.0
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        y_pred = self(x)
        loss = self.custom_mse_loss(y_pred, y)
        
        if torch.isnan(loss) or torch.isinf(loss):
            if batch_idx % 100 == 0:
                print(f"⚠️  Warning: Invalid loss at batch {batch_idx}")
            return y_pred.sum() * 0.0
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.train_loss.append(loss.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        y_pred = self(x)
        val_loss = self.custom_mse_loss(y_pred, y)
        
        with torch.no_grad():
            if len(y.shape) == 4 and y.shape[-1] < y.shape[1]:
                y = y.permute(0, 3, 1, 2)
            
            y_mask = torch.isnan(y)
            valid_mask = ~y_mask
            
            if valid_mask.sum() > 0:
                y_pred_valid = y_pred[valid_mask]
                y_true_valid = y[valid_mask]
                
                mae = F.l1_loss(y_pred_valid, y_true_valid)
                rmse = torch.sqrt(F.mse_loss(y_pred_valid, y_true_valid))
                
                self.log('val_mae', mae, prog_bar=True, on_step=False, on_epoch=True)
                self.log('val_rmse', rmse, prog_bar=True, on_step=False, on_epoch=True)
        
        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.val_loss.append(val_loss.item())
        return val_loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
    
    def predict(self, x):
        """预测函数（兼容旧接口）"""
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x).float()
            
            if len(x.shape) == 3:
                x = x.unsqueeze(1)
            
            x = x.to(self.device)
            output = self(x)
            
            # 转换回 [batch, height, width, depth]
            output = output.permute(0, 2, 3, 1)
            
            return output.cpu().numpy()

