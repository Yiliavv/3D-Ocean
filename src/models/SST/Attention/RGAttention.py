"""RGAttention: 门控自注意力"""

import torch
import torch.nn as nn


class RGAttention(nn.Module):
    """门控自注意力
    
    特点：
    - 单层 MultiheadAttention
    - 可学习门控：动态融合注意力输出和残差
    - 门控初始化为 0，训练初期接近恒等映射
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, use_gate: bool = True, **kwargs):
        super().__init__()
        self.use_gate = use_gate
        
        # 注意力层
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 门控机制
        if use_gate:
            self.gate = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
            # 初始化为 0，训练初期 gate ≈ 0.5
            nn.init.zeros_(self.gate[0].weight)
            nn.init.zeros_(self.gate[0].bias)
        
        # 可视化
        self.viz = {'attention_weights': None, 'gate_values': None}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S, D] -> [B, S, D]"""
        normed = self.layer_norm(x)
        attn_output, attn_weights = self.attention(normed, normed, normed)
        
        # 保存用于可视化
        self.viz['attention_weights'] = attn_weights.detach()
        
        if self.use_gate:
            gate = self.gate(attn_output)
            self.viz['gate_values'] = gate.detach().mean().item()
            return gate * attn_output + (1 - gate) * x
        else:
            return x + self.dropout(attn_output)
