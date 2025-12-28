"""
高效版递归泛化自注意力模块 (EfficientRGAttention)

相比原 RGAttention 的改进：
1. 移除 Global Token - 输出被丢弃，计算浪费
2. 轻量 Scalar Gate - 从 131K 参数降至 256 参数
3. 可配置注意力层数 - 替代固定递归深度

参数对比 (d_model=256):
- 原版 RGAttention: ~394K 参数
- EfficientRGAttention: ~263K 参数
- 节省: ~131K 参数 (33%)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EfficientRGAttention(nn.Module):
    """
    高效版递归泛化自注意力模块
    
    改进点：
    - 无 Global Token（节省计算）
    - 轻量 Scalar Gate（256 params vs 131K）
    - 可配置层数（替代递归深度）
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        dropout: float = 0.1,
        num_layers: int = 1,
        use_gate: bool = True
    ):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: Dropout 比例
            num_layers: 注意力层数（替代 recursion_depth）
            use_gate: 是否使用轻量门控
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_gate = use_gate
        
        # 多层注意力（每层独立参数，比递归更有效）
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                d_model, 
                num_heads, 
                dropout=dropout, 
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # 轻量 Scalar Gate：从 d_model*2 -> d_model 简化为 d_model -> 1
        # 原版: nn.Linear(d_model * 2, d_model) = 131,072 参数
        # 新版: nn.Linear(d_model, 1) = 257 参数
        if use_gate:
            self.gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, 1),
                    nn.Sigmoid()
                )
                for _ in range(num_layers)
            ])
            
            # 初始化门控偏置，使初始 gate ≈ 0.5
            for gate in self.gates:
                nn.init.zeros_(gate[0].weight)
                nn.init.zeros_(gate[0].bias)
        
        # Layer Norm for pre-norm architecture
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # 用于可视化
        self.viz = {
            'attention_weights': None,
            'gate_values': None
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch, seq_len, d_model] 输入特征序列
        
        Returns:
            output: [batch, seq_len, d_model]
        """
        current = x
        
        for i in range(self.num_layers):
            # Pre-norm
            normed = self.layer_norms[i](current)
            
            # Self-Attention
            attn_output, attn_weights = self.attention_layers[i](
                normed, normed, normed
            )
            
            # 保存最后一层的注意力权重用于可视化
            if i == self.num_layers - 1:
                self.viz['attention_weights'] = attn_weights.detach()
            
            # 门控残差连接
            if self.use_gate:
                # Scalar gate: [batch, seq, 1]
                gate = self.gates[i](attn_output)
                
                # 保存门控值用于可视化
                if i == self.num_layers - 1:
                    self.viz['gate_values'] = gate.detach().mean().item()
                
                # 门控融合: gate * attn + (1 - gate) * residual
                current = gate * attn_output + (1 - gate) * current
            else:
                # 简单残差连接
                current = current + self.dropout(attn_output)
        
        return current
    
    def get_num_parameters(self) -> int:
        """获取参数量"""
        return sum(p.numel() for p in self.parameters())


class MultiLayerAttention(nn.Module):
    """
    多层注意力模块（无门控，更简洁版本）
    
    适用于需要更简单架构的场景
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            num_layers: 注意力层数
            dropout: Dropout 比例
        """
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch, seq_len, d_model]
        
        Returns:
            output: [batch, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x)
        return x


# 兼容别名
RGAttention = EfficientRGAttention
