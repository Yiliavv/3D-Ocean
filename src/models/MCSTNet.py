import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ConvLSTM import ConvLSTM

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn = nn.GroupNorm(2, out_channels)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.leaky_relu(x)
        return x

class TimeTransferModule(nn.Module):
    def __init__(self, channels):
        super(TimeTransferModule, self).__init__()
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 两层全连接网络
        self.fc1 = nn.Linear(channels, channels // 2)
        self.fc2 = nn.Linear(channels // 2, channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        batch, seq_len, channels, height, width = x.size()
        
        # 重塑张量以合并时间和通道维度
        x = x.view(batch, seq_len * channels, height, width)
        
        # 全局池化
        max_out = self.global_max_pool(x).view(batch, -1)
        avg_out = self.global_avg_pool(x).view(batch, -1)
        
        # 非线性变换
        max_out = self.fc2(self.relu(self.fc1(max_out)))
        avg_out = self.fc2(self.relu(self.fc1(avg_out)))
        
        out = max_out + avg_out
        out = out.view(batch, seq_len, channels, 1, 1)
        
        return out

class MCSTNet(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, num_layers=4):
        super(MCSTNet, self).__init__()
        
        # 编码器
        self.encoder_blocks = nn.ModuleList()
        current_channels = input_channels
        for i in range(num_layers):
            self.encoder_blocks.append(ConvBlock(current_channels, hidden_channels * (2**i)))
            current_channels = hidden_channels * (2**i)
        
        # 解码器
        self.decoder_blocks = nn.ModuleList()
        for i in range(num_layers-1, -1, -1):
            self.decoder_blocks.append(ConvBlock(hidden_channels * (2**i) * 2, hidden_channels * (2**max(i-1, 0))))
        
        # 时间迁移模块
        self.time_transfer = TimeTransferModule(hidden_channels * (2**(num_layers-1)))
        
        # 记忆上下文模块 (ConvLSTM)
        self.memory_contextual = ConvLSTM(
            input_channels=hidden_channels,
            hidden_channels=[hidden_channels * 2, hidden_channels * 4, hidden_channels * 8],
            kernel_size=kernel_size,
            num_layers=3
        )
        
        # 多任务生成模块
        self.sst_head = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 1, kernel_size=1)
        )
        
        self.front_head = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        # x shape: [batch, seq_len, channels, height, width]
        batch_size, seq_len = x.size(0), x.size(1)
        
        # 展平时间维度
        x = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))
        
        # 编码器前向传播
        encoder_features = []
        for block in self.encoder_blocks:
            x = block(x)
            encoder_features.append(x)
            x = self.max_pool(x)
            
        # 重塑回时间维度用于时间迁移
        x = x.view(batch_size, seq_len, -1, x.size(2), x.size(3))
        
        # 时间迁移模块
        temporal_features = self.time_transfer(x)
        x = x * temporal_features
        
        # 记忆上下文模块
        x = x.view(batch_size, seq_len, -1, x.size(3), x.size(4))
        memory_output, _ = self.memory_contextual(x)
        x = memory_output[-1]  # 使用最后一层的输出
        
        # 解码器前向传播
        x = x.view(batch_size * seq_len, -1, x.size(3), x.size(4))
        for i, block in enumerate(self.decoder_blocks):
            x = self.upsample(x)
            x = torch.cat([x, encoder_features[-(i+1)]], dim=1)
            x = block(x)
            
        # 重塑回原始维度
        x = x.view(batch_size, seq_len, -1, x.size(2), x.size(3))
        
        # 生成SST和前沿预测
        sst_pred = self.sst_head(x[:, -1])  # 只使用最后一个时间步
        front_pred = self.front_head(x[:, -1])
        
        return sst_pred, front_pred

    def compute_loss(self, sst_pred, front_pred, sst_target, front_target, lambda_weight=0.2):
        # MSE损失
        mse_loss_sst = F.mse_loss(sst_pred, sst_target)
        mse_loss_front = F.mse_loss(front_pred, front_target)
        
        # 上下文损失 (这里使用简化版本，实际应该使用预训练的ResNet特征)
        ctx_loss_sst = F.l1_loss(sst_pred, sst_target)
        ctx_loss_front = F.l1_loss(front_pred, front_target)
        
        # 组合损失
        sst_loss = lambda_weight * ctx_loss_sst + (1 - lambda_weight) * mse_loss_sst
        front_loss = lambda_weight * ctx_loss_front + (1 - lambda_weight) * mse_loss_front
        
        total_loss = sst_loss + front_loss
        return total_loss, sst_loss, front_loss