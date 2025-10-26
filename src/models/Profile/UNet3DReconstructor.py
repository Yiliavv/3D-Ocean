"""
U-Net 3D 重建器 - 从海表温度重建三维温度场
专为海洋温度场重建任务设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch import optim


class DoubleConv(nn.Module):
    """双卷积块 - U-Net的基本构建单元"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """下采样块 - MaxPooling + DoubleConv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样块 - 转置卷积 + 拼接 + DoubleConv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 处理尺寸不匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3DReconstructor(LightningModule):
    """
    U-Net 3D 温度场重建器
    
    输入：海表温度 [batch, 1, height, width]
    输出：三维温度场 [batch, depth, height, width]
    
    参数：
        n_channels: 输入通道数（默认1，海表温度）
        n_depth: 输出深度层数
        base_channels: 基础通道数（默认64）
        bilinear: 是否使用双线性插值上采样（vs 转置卷积）
        learning_rate: 学习率
    """
    
    def __init__(self, 
                 n_channels=1, 
                 n_depth=10,
                 base_channels=64,
                 bilinear=False,
                 learning_rate=1e-4):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_depth = n_depth
        self.learning_rate = learning_rate
        
        # 编码器（下采样）
        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        
        # 解码器（上采样）
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        
        # 输出层 - 生成深度维度
        self.outc = nn.Conv2d(base_channels, n_depth, kernel_size=1)
        
        # 损失记录
        self.train_loss = []
        self.val_loss = []
        
        # 参数初始化
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch, 1, height, width] 海表温度
        
        Returns:
            output: [batch, depth, height, width] 三维温度场
        """
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 解码器
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 输出
        output = self.outc(x)
        
        return output
    
    def custom_mse_loss(self, y_pred, y_true):
        """
        处理NaN值的MSE损失函数
        
        Args:
            y_pred: [batch, depth, height, width]
            y_true: [batch, height, width, depth]
        
        Returns:
            loss: MSE损失
        """
        # 调整 y_true 维度顺序以匹配 y_pred
        if len(y_true.shape) == 4 and y_true.shape[-1] < y_true.shape[1]:
            # [batch, height, width, depth] -> [batch, depth, height, width]
            y_true = y_true.permute(0, 3, 1, 2)
        
        # 创建有效值掩码
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
        
        # 确保输入是 [batch, 1, height, width]
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        y_pred = self(x)
        loss = self.custom_mse_loss(y_pred, y)
        
        # 检查损失有效性
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
        
        # 计算额外指标
        with torch.no_grad():
            # 调整维度以匹配
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
            weight_decay=0.01,
            betas=(0.9, 0.999)
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
        """
        预测函数（兼容旧接口）
        
        Args:
            x: [batch, height, width] 或 [batch, 1, height, width]
        
        Returns:
            output: numpy array [batch, height, width, depth]
        """
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x).float()
            
            if len(x.shape) == 3:
                x = x.unsqueeze(1)
            
            # 移到相同设备
            x = x.to(self.device)
            
            output = self(x)
            
            # 转换回 [batch, height, width, depth] 格式
            output = output.permute(0, 2, 3, 1)
            
            return output.cpu().numpy()

