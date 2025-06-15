import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from lightning import LightningModule
from torch import optim

class SST3DUNet(LightningModule):
    def __init__(self, width=20, height=20, depth=58, learning_rate=1e-4):
        super().__init__()
        
        self.width = width
        self.height = height 
        self.depth = depth
        self.learning_rate = learning_rate
        
        # 编码器：从2D SST提取特征
        self.encoder = nn.Sequential(
            # 首先将2D扩展为3D
            nn.Conv2d(1, 1024, 5, padding=1),
            nn.Conv2d(1024, 512, 5, padding=1),
            nn.ReLU(),
        )
        
        # 深度特征生成器：从2D特征生成3D特征
        # 计算编码器输出的实际尺寸
        with torch.no_grad():
            test_input = torch.randn(1, 1, width, height)
            test_encoded = self.encoder(test_input)
            compressed_size = test_encoded.numel() // test_encoded.shape[0]  # 除去batch维度
        
        self.depth_expander = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(compressed_size, 64),
            nn.Linear(64, 16 * depth),
        )
        
        # 3D解码器：重建3D温度场
        self.decoder_3d = nn.Sequential(
            nn.Conv3d(16, 512, 3, padding=1),
            nn.Conv3d(512, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 1, 3, padding=1),
        )
        
        # 训练记录
        self.train_loss = []
        self.val_loss = []
    
    def forward(self, sst):
        """
        Args:
            sst: 海表温度数据，形状为 (batch, w, h)
        Returns:
            ocean_3d: 三维海洋温度数据，形状为 (batch, w, h, d)
        """
        batch_size = sst.shape[0]
        
        # 添加通道维度: (batch, w, h) -> (batch, 1, w, h)
        sst_input = sst.unsqueeze(1)
        
        # 2D编码器提取空间特征
        encoded_2d = self.encoder(sst_input)  # (batch, 256, w, h)
        
        # 展平并生成深度特征
        flattened = encoded_2d.view(batch_size, -1)
        depth_features = self.depth_expander(flattened)  # (batch, 16*depth)
        
        # 重塑为3D特征体积
        depth_features = depth_features.view(batch_size, 16, self.depth, 1, 1)
        depth_features = depth_features.expand(-1, -1, -1, self.width, self.height)
        
        # 3D解码器重建温度场
        ocean_3d = self.decoder_3d(depth_features)  # (batch, 1, depth, w, h)
        
        # 移除通道维度并调整维度顺序: (batch, 1, depth, w, h) -> (batch, w, h, depth)
        ocean_3d = ocean_3d.squeeze(1).permute(0, 2, 3, 1)
        
        return ocean_3d
    
    def custom_mse_loss(self, y_pred, y_true):
        """
        自定义MSE损失函数，只对有效像素计算损失
        """
        y_mask = torch.isnan(y_true)
        valid_mask = ~y_mask
        
        if valid_mask.sum() > 0:
            # 只对有效像素计算损失
            y_pred_processed = y_pred.clone()
            y_pred_processed[y_mask] = 0.0
            
            y_true_processed = y_true.clone()
            y_true_processed[y_mask] = 0.0
            
            return nn.MSELoss()(y_pred_processed, y_true_processed)
        else:
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
    
    def training_step(self, batch):
        sst, temp_3d = batch  # sst: (batch, w, h), temp_3d: (batch, w, h, d)
        
        sst = self.__normalize__(sst)
        temp_3d = self.__normalize__(temp_3d)
        
        # 前向传播
        temp_pred = self(sst)
        
        # 计算损失 - 直接使用MSE损失
        loss = self.custom_mse_loss(temp_pred, temp_3d)
        
        self.log('train_loss', loss, prog_bar=True)
        self.train_loss.append(loss.item())
        
        return loss
    
    def validation_step(self, batch):
        sst, temp_3d = batch
        
        temp_pred = self(sst)
        val_loss = self.custom_mse_loss(temp_pred, temp_3d)
        
        self.log('val_loss', val_loss)
        self.val_loss.append(val_loss.item())
        
        return val_loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        return { "optimizer": optimizer }
        
    def __normalize__(self, x):
        # 保存原始nan掩码
        x_mask = torch.isnan(x)
        # 将nan值替换为0，以便模型处理
        x_processed = x.clone()
        x_processed[x_mask] = 0.0
        
        return x_processed