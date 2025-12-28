"""
递归泛化自注意力模块 (RGAttention)
集成轻量级全局上下文 (Global Token)

核心改进：
- 移除了复杂的 GlobalQuery 模块，改用类似 ViT 的 [CLS] Token 机制。
- 在输入序列中拼接可学习的 Global Token，利用 Self-Attention 自动完成全局信息的聚合与分发。
- 优势：零额外显存开销，计算速度最快，且已被证明在 Transformer 中极其有效。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGAttention(nn.Module):
    """
    带全局 Token 的递归泛化自注意力模块
    """
    
    def __init__(self, 
                 d_model, 
                 num_heads, 
                 recursion_depth=2,
                 dropout=0.1,
                 use_global_token=True):
        super().__init__()
        
        self.d_model = d_model
        self.recursion_depth = recursion_depth
        self.use_global_token = use_global_token
        
        # 轻量级全局上下文：可学习的 Global Token
        # 类似 BERT/ViT 的 [CLS] Token
        # [1, 1, d_model]
        if use_global_token:
            self.global_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # 核心自注意力模块
        self.shared_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # 门控机制 (用于递归融合)
        self.gate_projection = nn.Linear(d_model * 2, d_model)
        
        # 递归步骤权重
        self.step_weights = nn.Parameter(torch.ones(recursion_depth))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # 参数初始化
        nn.init.xavier_uniform_(self.gate_projection.weight, gain=0.5)
        nn.init.constant_(self.gate_projection.bias, 0.0)
        
        self.viz = {
            'shared_attention_weights': None,
        }
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch, seq_len, d_model] 输入特征序列
        
        Returns:
            output: [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 拼接 Global Token
        if self.use_global_token:
            # 扩展到 batch 维度: [B, 1, D]
            global_token = self.global_token.expand(batch_size, -1, -1)
            # 拼接: [B, 1+S, D]
            x_with_global = torch.cat([global_token, x], dim=1)
        else:
            x_with_global = x
            
        # 2. 递归泛化自注意力
        current_state = x_with_global
        accumulated_output = torch.zeros_like(x_with_global)
        
        # 温度缩放的步骤权重
        temp = self.temperature.clamp(min=0.1, max=10.0)
        step_weights_norm = F.softmax(self.step_weights / temp, dim=0)
        
        for step in range(self.recursion_depth):
            # Self-Attention: Global Token 会在这里与所有 Patch 交互
            attn_output, attn_weights = self.shared_attention(
                current_state, current_state, current_state
            )
            
            # 门控融合
            gate_input = torch.cat([current_state, attn_output], dim=-1)
            gate_logits = self.gate_projection(gate_input)
            gate = torch.sigmoid(gate_logits)
            
            current_state = gate * attn_output + (1 - gate) * current_state
            
            accumulated_output = accumulated_output + step_weights_norm[step] * current_state
        
        # 保存权重用于可视化 (只保存最后一层的)
        self.viz['shared_attention_weights'] = attn_weights
        
        # 3. 分离 Global Token，只返回数据部分的特征
        if self.use_global_token:
            # [B, 1+S, D] -> [B, S, D]
            # 我们这里暂时只利用 Global Token 在 Attention 中作为信息枢纽的作用
            # 而不直接使用它的输出
            output = accumulated_output[:, 1:, :]
        else:
            output = accumulated_output
            
        return output
