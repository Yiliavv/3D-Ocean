"""
递归注意力模块（重构版）
专为海表温度预测任务设计，提供两种注意力机制：
1. TrueRecursiveAttention: 真正的递归（参数共享，迭代细化）
2. HierarchicalAttention: 层次注意力（多尺度特征融合）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrueRecursiveAttention(nn.Module):
    """
    真正的递归自注意力模块
    
    特点：
    - 参数共享：单个注意力层循环调用
    - 迭代细化：每次递归逐步优化特征表示
    - 门控机制：自适应控制信息流动
    """
    def __init__(self, d_model, num_heads, recursion_depth=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.recursion_depth = recursion_depth
        
        # 共享的注意力层（真正的递归）
        self.shared_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # 特征细化网络（共享）
        self.refiner = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 门控机制 - 控制每次递归的信息融合
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # 递归步骤权重（可学习）
        self.step_weights = nn.Parameter(torch.ones(recursion_depth) / recursion_depth)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [batch, seq_len] 可选的padding掩码
        Returns:
            output: [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # 初始状态
        current_state = x
        accumulated_output = torch.zeros_like(x)
        
        # 归一化步骤权重
        step_weights_norm = F.softmax(self.step_weights, dim=0)
        
        # 递归迭代
        for step in range(self.recursion_depth):
            # 自注意力计算
            attn_output, _ = self.shared_attention(
                current_state, current_state, current_state,
                key_padding_mask=mask
            )
            
            # 特征细化
            refined = self.refiner(attn_output)
            
            # 门控融合 - 决定保留多少旧信息和新信息
            gate_input = torch.cat([current_state, refined], dim=-1)
            gate = self.gate_net(gate_input)
            current_state = gate * refined + (1 - gate) * current_state
            
            # 加权累积
            accumulated_output = accumulated_output + step_weights_norm[step] * current_state
        
        # 最终残差连接
        output = accumulated_output + x
        
        return output


class HierarchicalAttention(nn.Module):
    """
    层次注意力模块（多尺度特征融合）
    
    特点：
    - 多层独立：每层捕获不同粒度的特征
    - 自适应融合：学习最优的层间组合权重
    - 渐进式残差：逐层累积特征，避免梯度消失
    """
    def __init__(self, d_model, num_heads, num_levels=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_levels = num_levels
        
        # 多层注意力（每层独立，捕获不同尺度）
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_levels)
        ])
        
        # 每层的特征变换网络
        self.level_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(d_model)
            ) for _ in range(num_levels)
        ])
        
        # 层间融合权重（可学习，带温度参数）
        self.level_weights = nn.Parameter(torch.ones(num_levels))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # 跨层注意力融合网络
        self.fusion_attention = nn.MultiheadAttention(
            d_model, num_heads // 2, dropout=dropout, batch_first=True
        )
        
        # 最终输出门控
        self.output_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [batch, seq_len] 可选的padding掩码
        Returns:
            output: [batch, seq_len, d_model]
        """
        level_outputs = []
        current_input = x
        
        # 逐层处理（渐进式特征提取）
        for level_idx, (attn_layer, transform) in enumerate(
            zip(self.attention_layers, self.level_transforms)
        ):
            # 注意力计算
            attn_output, _ = attn_layer(
                current_input, current_input, current_input,
                key_padding_mask=mask
            )
            
            # 特征变换
            transformed = transform(attn_output)
            
            # 残差连接（逐层累积）
            level_output = transformed + current_input
            level_outputs.append(level_output)
            
            # 更新输入到下一层
            current_input = level_output
        
        # 加权融合所有层的输出
        weights = F.softmax(
            self.level_weights / self.temperature.clamp(min=0.1, max=10.0),
            dim=0
        )
        weighted_sum = sum(w * output for w, output in zip(weights, level_outputs))
        
        # 跨层注意力融合（让不同层的特征相互交互）
        fused_output, _ = self.fusion_attention(
            weighted_sum, weighted_sum, weighted_sum,
            key_padding_mask=mask
        )
        
        # 门控输出（决定保留多少原始信息）
        gate_input = torch.cat([x, fused_output], dim=-1)
        gate = self.output_gate(gate_input)
        output = gate * fused_output + (1 - gate) * x
        
        return output


# 为了兼容性，保留原名称但使用改进的实现
class RecursiveAttention(nn.Module):
    """
    递归注意力模块（兼容接口）
    
    根据 mode 参数选择实现：
    - 'true_recursive': 真正的递归（参数共享）
    - 'hierarchical': 层次注意力（多层融合）
    """
    def __init__(self, d_model, num_heads, recursion_depth=2, dropout=0.1, mode='hierarchical'):
        super().__init__()
        self.mode = mode
        
        if mode == 'true_recursive':
            self.attention = TrueRecursiveAttention(
                d_model, num_heads, recursion_depth, dropout
            )
        elif mode == 'hierarchical':
            self.attention = HierarchicalAttention(
                d_model, num_heads, recursion_depth, dropout
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'true_recursive' or 'hierarchical'")
    
    def forward(self, x, mask=None):
        return self.attention(x, mask)

class RecursiveAttentionLayer(nn.Module):
    """
    递归注意力 Transformer 层（完整版）
    
    结合递归注意力机制和标准 Transformer 组件
    支持 Pre-LN 和 Post-LN 两种归一化模式
    """
    def __init__(self, 
                 d_model, 
                 num_heads, 
                 dim_feedforward=1024, 
                 dropout=0.1, 
                 recursion_depth=2,
                 attention_mode='hierarchical',
                 norm_first=True):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dim_feedforward: 前馈网络隐藏层维度
            dropout: Dropout 比例
            recursion_depth: 递归/层次深度
            attention_mode: 'true_recursive' 或 'hierarchical'
            norm_first: True为Pre-LN（更稳定），False为Post-LN（原始Transformer）
        """
        super().__init__()
        self.norm_first = norm_first
        
        # 递归注意力机制
        self.recursive_attention = RecursiveAttention(
            d_model, num_heads, recursion_depth, dropout, mode=attention_mode
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        # 改进的前馈网络（使用 GLU 变体）
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward * 2, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # 可学习的残差缩放因子（提高训练稳定性）
        self.residual_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [batch, seq_len] 可选的padding掩码
        Returns:
            output: [batch, seq_len, d_model]
        """
        if self.norm_first:
            # Pre-LN 模式（推荐，训练更稳定）
            # 递归注意力 + 残差
            attn_out = self.recursive_attention(self.norm1(x), mask)
            x = x + self.dropout(attn_out) * self.residual_scale
            
            # 前馈网络 + 残差
            ffn_out = self.ffn(self.norm2(x))
            x = x + ffn_out * self.residual_scale
        else:
            # Post-LN 模式（原始Transformer）
            attn_out = self.recursive_attention(x, mask)
            x = self.norm1(x + self.dropout(attn_out))
            
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)
        
        return x
