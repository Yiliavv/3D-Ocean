"""
RG-SA (Recursive-Generalization Self-Attention) 递归注意力模块

核心思想：
参数共享的递归注意力机制 - 单个注意力层循环调用多次，逐步细化特征表示

关键特性：
1. 参数高效：通过参数共享，用更少参数达到更深网络的效果
2. 迭代细化：多次递归逐步优化特征，类似迭代优化算法  
3. 门控融合：自适应控制每次递归中新旧信息的融合比例
4. 温度缩放：动态调整不同递归步骤的权重分布

设计理念：
- 受启发于递归神经网络的参数共享思想
- 每次递归相当于对特征进行一次"修正"
- 可学习的步骤权重自动发现最优递归策略

使用示例：
    attn = RGAttention(d_model=256, num_heads=8, recursion_depth=3)
    output = attn(x, mask)  # 经过3次递归细化的特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGAttention(nn.Module):
    """
    递归泛化自注意力（Recursive-Generalization Self-Attention）
    
    核心机制：
    - 参数共享：单个注意力层循环调用，大幅减少参数量
    - 迭代细化：每次递归逐步优化特征表示
    - 门控融合：自适应控制新旧信息的融合比例
    - 温度缩放：动态调整不同递归步骤的权重分布
    
    工作流程：
    1. 初始化：输入特征 x
    2. 递归循环（depth次）：
       a. 自注意力计算
       b. 特征细化网络
       c. 门控融合（决定保留多少旧特征和新特征）
       d. 加权累积到输出
    3. 最终残差连接
    """
    def __init__(self, d_model, num_heads, recursion_depth=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.recursion_depth = recursion_depth
        
        # 共享的注意力层（真正的递归，所有步骤共享参数）
        self.shared_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # 特征细化网络（共享，用于后处理注意力输出）
        self.refiner = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        
        # 门控机制 - 自适应融合新旧特征
        self.gate_projection = nn.Linear(d_model * 2, d_model)
        
        # 递归步骤权重（可学习）- 决定每个递归步骤的重要性
        self.step_weights = nn.Parameter(torch.ones(recursion_depth))
        # 温度参数 - 控制权重分布的平滑度
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # 参数初始化
        nn.init.xavier_uniform_(self.gate_projection.weight, gain=0.5)
        nn.init.constant_(self.gate_projection.bias, 0.0)
        
    def forward(self, x, mask=None):
        """
        递归注意力前向传播
        
        Args:
            x: [batch, seq_len, d_model] 输入特征
            mask: [batch, seq_len] 可选的padding掩码
        
        Returns:
            output: [batch, seq_len, d_model] 经过递归细化的特征
        """
        batch_size, seq_len, _ = x.shape
        
        # 初始状态
        current_state = x
        accumulated_output = torch.zeros_like(x)
        
        # 温度缩放的步骤权重（避免权重过于集中或分散）
        temp = self.temperature.clamp(min=0.1, max=10.0)
        step_weights_norm = F.softmax(self.step_weights / temp, dim=0)
        
        # 递归迭代：每次迭代都细化特征表示
        for step in range(self.recursion_depth):
            # 步骤1: 自注意力计算（使用共享的注意力层）
            attn_output, _ = self.shared_attention(
                current_state, current_state, current_state,
                key_padding_mask=mask
            )
            
            # 步骤2: 特征细化（通过小型FFN进一步处理）
            refined = self.refiner(attn_output)
            
            # 步骤3: 门控融合（自适应决定保留多少旧特征vs新特征）
            gate_input = torch.cat([current_state, refined], dim=-1)
            gate_logits = self.gate_projection(gate_input)
            gate = torch.sigmoid(gate_logits)
            
            # 融合：gate接近1时更新多，接近0时保留多
            current_state = gate * refined + (1 - gate) * current_state
            
            # 步骤4: 加权累积（不同递归步骤可以有不同的重要性）
            accumulated_output = accumulated_output + step_weights_norm[step] * current_state
        
        # 最终残差连接（确保梯度流畅）
        output = accumulated_output + x
        
        return output
