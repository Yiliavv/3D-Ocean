import torch
import torch.nn as nn
import torch.nn.functional as F

class RGAttention(nn.Module):
    """
    递归泛化自注意力（Recursive-Generalization Self-Attention）
    
    专为时空序列数据设计，处理 [batch, seq_len, width, height] 格式的输入
    在时间步序列上应用递归注意力机制
    
    核心机制：
    - 参数共享：单个注意力层循环调用，大幅减少参数量
    - 迭代细化：每次递归逐步优化时间序列特征表示
    - 门控融合：自适应控制新旧信息的融合比例
    - 温度缩放：动态调整不同递归步骤的权重分布
    
    工作流程：
    1. 输入特征 x [batch, seq_len, width, height]
    2. 展平空间维度：x -> [batch, seq_len, width*height]
    3. 投影到模型维度：[batch, seq_len, width*height] -> [batch, seq_len, d_model]
    4. 递归循环（depth次）：
       a. 时间步自注意力计算
       b. 特征细化网络
       c. 门控融合
       d. 加权累积到输出
    5. 投影回空间维度：[batch, seq_len, d_model] -> [batch, seq_len, width*height]
    6. 重塑回原始形状：[batch, seq_len, width*height] -> [batch, seq_len, width, height]
    7. 最终残差连接
    """
    def __init__(self, d_model, num_heads, recursion_depth=2, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.recursion_depth = recursion_depth
        
        # 输入投影层：将空间展平后的维度投影到模型维度
        # 注意：输入维度在forward中动态确定，这里先设为None，在首次forward时初始化
        self.input_proj = None
        self.output_proj = None
        self._input_dim = None
        
        # 共享的注意力层（真正的递归，所有步骤共享参数）
        self.shared_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
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

        self.viz = {
            'shared_attention': None,
            'shared_attention_weights': None,
        }
    
    def _init_projections(self, input_dim):
        """
        初始化输入和输出投影层（延迟初始化，因为输入维度在运行时才确定）
        
        Args:
            input_dim: 空间展平后的维度 (width * height)
        """
        if self._input_dim == input_dim:
            return  # 已经初始化过了
        
        self._input_dim = input_dim
        
        # 获取设备（从已初始化的模块获取）
        device = self.gate_projection.weight.device
        
        # 输入投影：将空间维度投影到模型维度
        self.input_proj = nn.Linear(input_dim, self.d_model).to(device)
        # 输出投影：将模型维度投影回空间维度
        self.output_proj = nn.Linear(self.d_model, input_dim).to(device)
        
        # 注册为子模块（确保参数能被优化器识别）
        self.add_module('input_proj', self.input_proj)
        self.add_module('output_proj', self.output_proj)
        
        # 初始化投影层
        nn.init.xavier_uniform_(self.input_proj.weight, gain=0.1)
        nn.init.constant_(self.input_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
        nn.init.constant_(self.output_proj.bias, 0.0)
        
    def forward(self, x):
        """
        递归注意力前向传播（时空序列版本）
        
        Args:
            x: [batch, seq_len, width, height] 输入时空特征
        
        Returns:
            output: [batch, seq_len, width, height] 经过递归细化的时空特征
        """

        batch_size, seq_len, width, height = x.shape
        
        # 展平空间维度：[batch, seq_len, width, height] -> [batch, seq_len, width*height]
        x_flat = x.view(batch_size, seq_len, width * height)
        
        # 初始化投影层（如果还未初始化）
        self._init_projections(width * height)
        
        # 投影到模型维度：[batch, seq_len, width*height] -> [batch, seq_len, d_model]
        x_proj = self.input_proj(x_flat)
        
        # 初始状态
        current_state = x_proj
        accumulated_output = torch.zeros_like(x_proj)
        
        # 温度缩放的步骤权重（避免权重过于集中或分散）
        temp = self.temperature.clamp(min=0.1, max=10.0)
        step_weights_norm = F.softmax(self.step_weights / temp, dim=0)
        
        # 递归迭代：每次迭代都细化特征表示
        for step in range(self.recursion_depth):
            # 步骤1: 时间步自注意力计算（使用共享的注意力层）
            attn_output, _ = self.shared_attention(current_state, current_state, current_state)
            
            # 步骤3: 门控融合（自适应决定保留多少旧特征vs新特征）
            gate_input = torch.cat([current_state, attn_output], dim=-1)
            gate_logits = self.gate_projection(gate_input)
            gate = torch.sigmoid(gate_logits)
            
            # 融合：gate接近1时更新多，接近0时保留多
            current_state = gate * attn_output + (1 - gate) * current_state
            
            # 步骤4: 加权累积（不同递归步骤可以有不同的重要性）
            accumulated_output = accumulated_output + step_weights_norm[step] * current_state
        
        # 投影回空间维度：[batch, seq_len, d_model] -> [batch, seq_len, width*height]
        output_flat = self.output_proj(accumulated_output)
        
        # 重塑回原始形状：[batch, seq_len, width*height] -> [batch, seq_len, width, height]
        output = output_flat.view(batch_size, seq_len, width, height)
        
        return output