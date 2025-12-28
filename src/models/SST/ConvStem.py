"""
卷积前处理模块 (ConvStem)

替代原有的单层 Patch Embedding，提供：
1. 更好的局部特征提取
2. 自然的平移等变性
3. 避免 patch 边界硬切割

参考：
- LeViT (ICCV 2021): Conv Stem 的有效性
- ConvNeXt (CVPR 2022): 现代卷积网络设计
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ConvStem(nn.Module):
    """
    卷积前处理模块，替代直接的 Patch Embedding
    
    结构:
        Conv2d(3×3, stride=2) -> BN -> GELU
        Conv2d(3×3, stride=2) -> BN -> GELU  
        Conv2d(1×1) -> 调整通道数
    
    优势:
    - 多层小卷积捕获局部特征
    - 自然引入平移等变性
    - 避免 patch 边界硬切割
    - 特征提取更平滑
    """
    
    def __init__(
        self, 
        in_channels: int = 1, 
        embed_dim: int = 256,
        target_reduction: int = 4,
        use_bn: bool = True
    ):
        """
        Args:
            in_channels: 输入通道数 (SST 为 1)
            embed_dim: 输出嵌入维度
            target_reduction: 总降采样倍数 (对应原 patch_size)
            use_bn: 是否使用 BatchNorm
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.target_reduction = target_reduction
        
        # 计算中间通道数
        mid_channels_1 = embed_dim // 4  # 64 for embed_dim=256
        mid_channels_2 = embed_dim // 2  # 128 for embed_dim=256
        
        # 构建卷积层
        layers = []
        
        # 第一层: stride=2 降采样
        layers.append(nn.Conv2d(
            in_channels, 
            mid_channels_1, 
            kernel_size=3, 
            stride=2, 
            padding=1
        ))
        if use_bn:
            layers.append(nn.BatchNorm2d(mid_channels_1))
        layers.append(nn.GELU())
        
        # 第二层: stride=2 降采样 (总降采样 4x)
        layers.append(nn.Conv2d(
            mid_channels_1, 
            mid_channels_2, 
            kernel_size=3, 
            stride=2, 
            padding=1
        ))
        if use_bn:
            layers.append(nn.BatchNorm2d(mid_channels_2))
        layers.append(nn.GELU())
        
        # 投影层: 1x1 卷积调整通道数
        layers.append(nn.Conv2d(
            mid_channels_2, 
            embed_dim, 
            kernel_size=1
        ))
        
        self.stem = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch, channels, height, width] 或 [batch, 1, height, width]
        
        Returns:
            output: [batch, embed_dim, height/4, width/4]
        """
        return self.stem(x)
    
    def get_output_size(self, input_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        计算输出尺寸
        
        Args:
            input_size: (height, width)
        
        Returns:
            (output_height, output_width)
        """
        h, w = input_size
        # 两次 stride=2 的降采样
        return h // self.target_reduction, w // self.target_reduction


class ConvStemV2(nn.Module):
    """
    增强版卷积前处理模块
    
    添加了残差连接和更多的中间层，适用于需要更强特征提取的场景
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 256,
        target_reduction: int = 4,
        num_blocks: int = 2
    ):
        """
        Args:
            in_channels: 输入通道数
            embed_dim: 输出嵌入维度
            target_reduction: 降采样倍数
            num_blocks: 残差块数量
        """
        super().__init__()
        
        # 初始卷积
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU()
        )
        
        # 残差块
        self.blocks = nn.ModuleList()
        current_channels = embed_dim // 4
        
        for i in range(num_blocks):
            next_channels = min(current_channels * 2, embed_dim)
            stride = 2 if i == 0 else 1  # 第一个块降采样
            
            self.blocks.append(ResidualBlock(
                current_channels, 
                next_channels, 
                stride=stride
            ))
            current_channels = next_channels
        
        # 最终投影
        self.proj = nn.Conv2d(current_channels, embed_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.init_conv(x)
        for block in self.blocks:
            x = block(x)
        return self.proj(x)


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        
        # 残差连接（如果维度不匹配）
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity
        out = self.gelu(out)
        
        return out

