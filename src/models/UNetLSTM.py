import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import manual_seed, optim
from lightning import LightningModule

class ConvBlock(nn.Module):
    """
    UNet的基础卷积块
    包含两个卷积层 + 批归一化 + ReLU激活
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class EncoderBlock(nn.Module):
    """
    UNet编码器块
    卷积块 + 最大池化下采样
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        skip = self.conv_block(x)
        x = self.pool(skip)
        return x, skip

class DecoderBlock(nn.Module):
    """
    UNet解码器块
    上采样 + 跳跃连接 + 卷积块
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # 转置卷积只处理输入特征，不考虑跳跃连接
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, 2)
        # 卷积块处理拼接后的特征
        self.conv_block = ConvBlock(in_channels // 2 + skip_channels, out_channels)
        
    def forward(self, x, skip):
        x = self.up(x)
        # 确保尺寸匹配
        if x.size() != skip.size():
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        x = self.conv_block(x)
        return x

class ConvLSTMCell(nn.Module):
    """
    卷积LSTM单元
    在空间维度上保持卷积操作，处理时序信息
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        
        # 输入门、遗忘门、输出门、候选值的卷积层
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=kernel_size//2
        )
        
    def forward(self, x, hidden_state):
        h, c = hidden_state
        
        # 拼接输入和隐藏状态
        combined = torch.cat([x, h], dim=1)
        combined_conv = self.conv(combined)
        
        # 分离四个门的输出
        i, f, o, g = torch.chunk(combined_conv, 4, dim=1)
        
        # 应用激活函数
        i = torch.sigmoid(i)  # 输入门
        f = torch.sigmoid(f)  # 遗忘门
        o = torch.sigmoid(o)  # 输出门
        g = torch.tanh(g)     # 候选值
        
        # 更新细胞状态和隐藏状态
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        
        return h_new, (h_new, c_new)
    
    def init_hidden(self, batch_size, height, width, device):
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        )

class UNetLSTM(LightningModule):
    """
    UNet-LSTM模型
    结合UNet的空间特征提取和LSTM的时序建模
    适用于海表温度时序预测任务
    
    模型架构：
    1. 对每个时间步使用UNet编码器提取空间特征
    2. 在瓶颈层使用ConvLSTM处理时序信息
    3. 使用UNet解码器重建空间特征
    4. 输出下一时刻的海表温度预测
    """
    def __init__(self, 
                 input_channels=1,
                 output_channels=1, 
                 features=[64, 128, 256, 512],
                 lstm_hidden_channels=256,
                 learning_rate=1e-4):
        
        manual_seed(1)
        super().__init__()
        
        self.learning_rate = learning_rate
        self.lstm_hidden_channels = lstm_hidden_channels
        self.features = features
        
        # UNet编码器
        self.encoders = nn.ModuleList()
        in_channels = input_channels
        for feature in features:
            self.encoders.append(EncoderBlock(in_channels, feature))
            in_channels = feature
            
        # 瓶颈层的ConvLSTM
        self.conv_lstm = ConvLSTMCell(
            input_channels=features[-1],
            hidden_channels=lstm_hidden_channels
        )
        
        # UNet解码器 - 完全重构，确保通道数正确匹配
        self.decoders = nn.ModuleList()
        
        # 在UNet中，跳跃连接的使用顺序是从深到浅：[512, 256, 128, 64]
        # 对应的解码器输入输出设计：
        # 解码器1: LSTM(256) + skip(512) → 256
        # 解码器2: 256 + skip(256) → 128
        # 解码器3: 128 + skip(128) → 64  
        # 解码器4: 64 + skip(64) → 64
        
        # 构建解码器配置
        decoder_configs = [
            # (输入通道, 跳跃连接通道, 输出通道)
            (lstm_hidden_channels, features[-1], features[-2]),  # (256, 512, 256)
            (features[-2], features[-2], features[-3]),          # (256, 256, 128)
            (features[-3], features[-3], features[-4]),          # (128, 128, 64)  
            (features[-4], features[-4], features[-4])           # (64, 64, 64)
        ]
        
        # 创建解码器
        for in_ch, skip_ch, out_ch in decoder_configs:
            self.decoders.append(DecoderBlock(in_ch, skip_ch, out_ch))
            
        # 最终输出层
        self.final_conv = nn.Conv2d(features[0], output_channels, 1)
        
        # 训练和验证损失记录
        self.train_loss = []
        self.val_loss = []
        
    def forward(self, x):
        """
        前向传播
        x: [batch_size, seq_len, channels, height, width]
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # 数据预处理
        x_processed = self._normalize(x)
        
        # 初始化LSTM隐藏状态
        # 计算瓶颈层的空间尺寸
        bottleneck_h = height // (2 ** len(self.encoders))
        bottleneck_w = width // (2 ** len(self.encoders))
        
        lstm_hidden = self.conv_lstm.init_hidden(
            batch_size, bottleneck_h, bottleneck_w, x.device
        )
        
        # 处理序列中的每一帧
        for t in range(seq_len):
            current_frame = x_processed[:, t, :, :, :]  # [batch, channels, H, W]
            
            # UNet编码器 - 提取空间特征
            skips = []
            encoder_out = current_frame
            
            for encoder in self.encoders:
                encoder_out, skip = encoder(encoder_out)
                skips.append(skip)
                
            # ConvLSTM处理时序信息
            lstm_out, lstm_hidden = self.conv_lstm(encoder_out, lstm_hidden)
            
        # 使用最后一个时间步的特征进行解码
        decoder_input = lstm_out
        skips = skips[::-1]  # 反转跳跃连接列表
        
        # UNet解码器 - 重建空间特征
        for i, decoder in enumerate(self.decoders):
            skip = skips[i]
            decoder_input = decoder(decoder_input, skip)
            
        # 最终输出
        output = self.final_conv(decoder_input)
        # 修复：添加时间维度，确保输出格式为 [batch, 1, H, W]
        
        return output
    
    def _normalize(self, x):
        """数据归一化处理，处理NaN值"""
        x_mask = torch.isnan(x)
        x_processed = x.clone()
        x_processed[x_mask] = 0.0
        return x_processed
    
    def custom_mse_loss(self, y_pred, y_true):
        """
        自定义MSE损失函数，忽略NaN值
        适用于海表温度数据中的缺失值处理
        """
        y_mask = torch.isnan(y_true)
        mask = ~y_mask
        
        if mask.sum() > 0:
            y_pred_processed = y_pred.clone()
            y_pred_processed[y_mask] = 0.0
            
            y_true_processed = y_true.clone()
            y_true_processed[y_mask] = 0.0
            
            # 只计算非NaN位置的MSE
            mse = F.mse_loss(y_pred_processed, y_true_processed, reduction='none')
            weighted_mse = (mse * mask.float()).sum() / mask.float().sum()
            return weighted_mse
        else:
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
    
    def training_step(self, batch):
        x, y = batch  # x: [batch, seq_len, C, H, W], y: [batch, 1, C, H, W]
        
        y_pred = self(x)
        
        # 处理预测结果中的NaN值
        y_mask = torch.isnan(y)
        y_pred[y_mask] = float('nan')
        
        loss = self.custom_mse_loss(y_pred, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.train_loss.append(loss.item())
        
        return loss
    
    def validation_step(self, batch):
        x, y = batch  # x: [batch, seq_len, C, H, W], y: [batch, 1, C, H, W]
        
        y_pred = self(x)
        
        # 处理预测结果中的NaN值
        y_mask = torch.isnan(y)
        y_pred[y_mask] = float('nan')
        
        loss = self.custom_mse_loss(y_pred, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.val_loss.append(loss.item())
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        
        # 余弦退火学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

# 便捷的模型创建函数
def create_unet_lstm(input_channels=1, 
                    output_channels=1,
                    features=[64, 128, 256, 512],
                    lstm_hidden_channels=256,
                    learning_rate=1e-4):
    """
    创建UNet-LSTM模型的便捷函数
    
    参数:
        input_channels: 输入通道数 (默认1，单通道海表温度)
        output_channels: 输出通道数 (默认1)
        features: UNet编码器的特征数列表
        lstm_hidden_channels: ConvLSTM的隐藏通道数
        learning_rate: 学习率
    
    返回:
        UNetLSTM模型实例
    """
    return UNetLSTM(
        input_channels=input_channels,
        output_channels=output_channels,
        features=features,
        lstm_hidden_channels=lstm_hidden_channels,
        learning_rate=learning_rate
    )

# 轻量级版本
def create_lightweight_unet_lstm():
    """创建轻量级UNet-LSTM模型"""
    return UNetLSTM(
        features=[32, 64, 128, 256],
        lstm_hidden_channels=128,
        learning_rate=1e-3
    )

# 深度版本
def create_deep_unet_lstm():
    """创建深度UNet-LSTM模型"""
    return UNetLSTM(
        features=[64, 128, 256, 512, 1024],
        lstm_hidden_channels=512,
        learning_rate=5e-5
    )
