"""
递归注意力模块
专为海表温度预测任务设计，提供创新的递归式自注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RecursiveAttention(nn.Module):
    """递归式自注意力模块 - 核心创新"""
    def __init__(self, d_model, num_heads, recursion_depth=2):
        super().__init__()
        self.d_model = d_model
        self.recursion_depth = recursion_depth
        
        # 递归注意力层
        self.recursive_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            for _ in range(recursion_depth)
        ])
        
        # 递归权重 - 使用温度参数控制权重分布
        self.recursion_weights = nn.Parameter(torch.ones(recursion_depth))
        self.temperature = nn.Parameter(torch.tensor(1.0))  # 可学习的温度参数
        
        # 特征细化器 - 添加Layer Norm提高稳定性
        self.refiners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(0.1)
            ) for _ in range(recursion_depth)
        ])
        
    def forward(self, x, mask=None):
        current_output = x
        recursive_outputs = []
        
        # 递归注意力计算
        for attn_layer, refiner in zip(self.recursive_layers, self.refiners):
            attn_output, _ = attn_layer(
                current_output, current_output, current_output,
                key_padding_mask=mask
            )
            
            refined_output = refiner(attn_output)
            current_output = refined_output + current_output
            recursive_outputs.append(current_output)
        
        # 权重融合 - 使用温度缩放提高稳定性
        weights = F.softmax(self.recursion_weights / self.temperature.clamp(min=0.1), dim=0)
        final_output = sum(w * output for w, output in zip(weights, recursive_outputs))
        
        # 添加残差连接提高稳定性
        final_output = final_output + x
        
        return final_output

class RecursiveAttentionLayer(nn.Module):
    """递归注意力 Transformer 层"""
    def __init__(self, d_model, num_heads, dim_feedforward=1024, dropout=0.1, recursion_depth=2):
        super().__init__()
        
        # 递归注意力机制
        self.recursive_attention = RecursiveAttention(d_model, num_heads, recursion_depth)
        
        # 标准组件
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 递归注意力 + 残差连接
        attn_output = self.recursive_attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # 前馈网络 + 残差连接
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output
        
        return x
