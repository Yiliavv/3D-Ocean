"""EfficientRGAttention: 轻量门控多层自注意力"""

import torch
import torch.nn as nn


class EfficientRGAttention(nn.Module):
    """轻量门控多层自注意力"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, num_layers: int = 1, use_gate: bool = True):
        super().__init__()
        self.num_layers = num_layers
        self.use_gate = use_gate
        
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        if use_gate:
            self.gates = nn.ModuleList([
                nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
                for _ in range(num_layers)
            ])
            for gate in self.gates:
                nn.init.zeros_(gate[0].weight)
                nn.init.zeros_(gate[0].bias)
        
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.viz = {'attention_weights': None, 'gate_values': None}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S, D] -> [B, S, D]"""
        current = x
        
        for i in range(self.num_layers):
            normed = self.layer_norms[i](current)
            attn_output, attn_weights = self.attention_layers[i](normed, normed, normed)
            
            if i == self.num_layers - 1:
                self.viz['attention_weights'] = attn_weights.detach()
            
            if self.use_gate:
                gate = self.gates[i](attn_output)
                if i == self.num_layers - 1:
                    self.viz['gate_values'] = gate.detach().mean().item()
                current = gate * attn_output + (1 - gate) * current
            else:
                current = current + self.dropout(attn_output)
        
        return current


RGAttention = EfficientRGAttention
