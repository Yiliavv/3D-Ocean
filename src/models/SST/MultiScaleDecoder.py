"""
多尺度解码器模块 (MultiScaleDecoder)

提供从低分辨率特征图逐层上采样重建的能力，支持跳跃连接融合。

主要组件:
- DecoderStage: 单层解码器，包含上采样、跳跃连接融合和精炼卷积
- MultiScaleDecoder: 组合多个 DecoderStage，实现多尺度特征融合

参考:
- UNet (MICCAI 2015): 跳跃连接架构
- UNet++ (DLMIA 2018): 密集跳跃连接
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class DecoderStage(nn.Module):
    """
    解码器单层模块
    
    功能:
    1. 双线性上采样 2x
    2. 跳跃连接特征融合（加法或拼接）
    3. 精炼卷积（3x3 Conv + BN + GELU）
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        skip_channels: 跳跃连接通道数
        scale_factor: 上采样倍数 (默认 2)
        fusion: 融合方式 "add" | "concat"
    
    Input:
        x: [batch, in_channels, H, W] 解码器特征
        skip: [batch, skip_channels, H*scale, W*scale] 跳跃连接特征 (可选)
    
    Output:
        out: [batch, out_channels, H*scale, W*scale]
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int = 0,
        scale_factor: int = 2,
        fusion: str = "add"
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels
        self.scale_factor = scale_factor
        self.fusion = fusion
        
        # 通道对齐层（将输入通道数调整为输出通道数）
        self.channel_adjust = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # 跳跃连接通道对齐
        if skip_channels > 0 and skip_channels != out_channels:
            self.skip_align = nn.Conv2d(skip_channels, out_channels, 1)
        else:
            self.skip_align = nn.Identity() if skip_channels == out_channels else None
        
        # 计算精炼卷积的输入通道数
        if fusion == "concat" and skip_channels > 0:
            refine_in_channels = out_channels + out_channels  # 调整后的通道数相加
        else:
            refine_in_channels = out_channels
        
        # 精炼卷积
        self.refine = nn.Sequential(
            nn.Conv2d(refine_in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
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
    
    def forward(
        self, 
        x: torch.Tensor, 
        skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch, in_channels, H, W] 解码器特征
            skip: [batch, skip_channels, H*scale, W*scale] 跳跃连接特征 (可选)
        
        Returns:
            out: [batch, out_channels, H*scale, W*scale]
        """
        # 1. 上采样
        x = F.interpolate(
            x, 
            scale_factor=self.scale_factor, 
            mode='bilinear', 
            align_corners=False
        )
        
        # 2. 通道调整
        x = self.channel_adjust(x)
        
        # 3. 跳跃连接融合
        if skip is not None and self.skip_align is not None:
            # 对齐跳跃连接通道
            skip_aligned = self.skip_align(skip)
            
            # 确保空间尺寸匹配
            if x.shape[2:] != skip_aligned.shape[2:]:
                skip_aligned = F.interpolate(
                    skip_aligned, 
                    size=x.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # 融合
            if self.fusion == "add":
                x = x + skip_aligned
            else:  # concat
                x = torch.cat([x, skip_aligned], dim=1)
        
        # 4. 精炼卷积
        out = self.refine(x)
        
        return out


class MultiScaleDecoder(nn.Module):
    """
    多尺度解码器
    
    组合多个 DecoderStage，逐层上采样并融合跳跃连接特征。
    
    Args:
        in_channels: 输入通道数 (来自 Attention 输出，默认 256)
        out_channels: 最终输出通道数 (SST 预测为 1)
        skip_channels: 各层跳跃连接通道数列表 [skip_2_ch, skip_1_ch]
        num_stages: 解码器层数 (默认 2)
        fusion: 融合方式 "add" | "concat"
    
    Input:
        x: [batch, in_channels, H, W] Attention 输出
        skip_features: List[[batch, C_i, H_i, W_i]] 跳跃连接特征列表
                       顺序: [skip_2, skip_1]（从深到浅）
    
    Output:
        out: [batch, out_channels, H_final, W_final]
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 1,
        skip_channels: Optional[List[int]] = None,
        num_stages: int = 2,
        fusion: str = "add"
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_stages = num_stages
        self.fusion = fusion
        
        # 默认跳跃连接通道数: [128, 64]
        if skip_channels is None:
            skip_channels = [128, 64]
        self.skip_channels = skip_channels
        
        # 计算每层的通道数
        # Stage 1: 256 -> 128, Stage 2: 128 -> 64
        stage_channels = [in_channels]
        for i in range(num_stages):
            stage_channels.append(skip_channels[i] if i < len(skip_channels) else stage_channels[-1] // 2)
        
        # 构建解码器阶段
        self.stages = nn.ModuleList()
        for i in range(num_stages):
            skip_ch = skip_channels[i] if i < len(skip_channels) else 0
            self.stages.append(DecoderStage(
                in_channels=stage_channels[i],
                out_channels=stage_channels[i + 1],
                skip_channels=skip_ch,
                scale_factor=2,
                fusion=fusion
            ))
        
        # 最终输出卷积
        self.final_conv = nn.Conv2d(stage_channels[-1], out_channels, 1)
        
        # 初始化最终卷积
        nn.init.xavier_uniform_(self.final_conv.weight)
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        skip_features: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch, in_channels, H, W] Attention 输出
            skip_features: 跳跃连接特征列表，顺序从深到浅 [skip_2, skip_1]
        
        Returns:
            out: [batch, out_channels, H_final, W_final]
        """
        if skip_features is None:
            skip_features = [None] * self.num_stages
        
        # 逐层解码
        for i, stage in enumerate(self.stages):
            skip = skip_features[i] if i < len(skip_features) else None
            x = stage(x, skip)
        
        # 最终输出
        out = self.final_conv(x)
        
        return out
    
    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """
        获取参数量
        
        Args:
            trainable_only: 是否只统计可训练参数
        
        Returns:
            参数总数
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

