"""
球谐波位置编码模块
专为海表温度预测任务设计，提供球面几何感知的位置编码
"""

import torch
import torch.nn as nn
import math

def spherical_harmonics(l, m, theta, phi):
    """计算球谐波函数 Y_l^m(theta, phi)"""
    if m == 0:
        # Y_l^0 = sqrt((2l+1)/(4pi)) * P_l(cos(theta))
        if l == 0:
            return torch.ones_like(theta) / torch.sqrt(4 * math.pi)
        elif l == 1:
            return torch.sqrt(3 / (4 * math.pi)) * torch.cos(theta)
        elif l == 2:
            return torch.sqrt(5 / (4 * math.pi)) * (3 * torch.cos(theta)**2 - 1) / 2
        else:
            # 简化实现，对于更高阶使用近似
            return torch.cos(l * theta) / torch.sqrt(4 * math.pi)
    elif m > 0:
        # Y_l^m = sqrt((2l+1)(l-m)!/(4pi(l+m)!)) * P_l^m(cos(theta)) * cos(m*phi)
        if l == 1 and m == 1:
            return -torch.sqrt(3 / (8 * math.pi)) * torch.sin(theta) * torch.cos(phi)
        elif l == 2 and m == 1:
            return -torch.sqrt(15 / (8 * math.pi)) * torch.sin(theta) * torch.cos(theta) * torch.cos(phi)
        elif l == 2 and m == 2:
            return torch.sqrt(15 / (32 * math.pi)) * torch.sin(theta)**2 * torch.cos(2 * phi)
        else:
            # 简化实现
            return torch.cos(m * phi) * torch.sin(theta)**m / torch.sqrt(4 * math.pi)
    else:
        # Y_l^(-m) = (-1)^m * conj(Y_l^m)
        return (-1)**abs(m) * spherical_harmonics(l, abs(m), theta, phi)

class SphericalHarmonicEncoding(nn.Module):
    """球谐波位置编码神经网络模型 - 简化版本"""
    def __init__(self, seq_len, d_model, max_degree=4, hidden_dim=64):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.max_degree = max_degree
        
        # 位置嵌入层
        self.position_embedding = nn.Embedding(seq_len, hidden_dim)
        
        # 球谐波特征网络 - 简化版本
        self.harmonic_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 球面坐标网络
        self.spherical_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2)  # theta, phi
        )
        
        # 球谐波基础函数网络
        self.basis_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model // 2)
        )
        
        # 融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.02)
    
    def forward(self, positions=None):
        """前向传播"""
        if positions is None:
            positions = torch.arange(self.seq_len, device=self.position_embedding.weight.device)
        
        # 位置嵌入
        pos_emb = self.position_embedding(positions)  # [seq_len, hidden_dim]
        
        # 基础球谐波特征
        harmonic_feat = self.harmonic_net(pos_emb)  # [seq_len, d_model]
        
        # 球面坐标特征
        spherical_coords = self.spherical_net(pos_emb)  # [seq_len, 2]
        theta = torch.sigmoid(spherical_coords[:, 0]) * math.pi  # [0, pi]
        phi = torch.sigmoid(spherical_coords[:, 1]) * 2 * math.pi  # [0, 2pi]
        
        # 球谐波基础函数
        coords = torch.stack([theta, phi], dim=1)  # [seq_len, 2]
        basis_feat = self.basis_net(coords)  # [seq_len, d_model//2]
        
        # 融合特征
        combined_feat = torch.cat([harmonic_feat, basis_feat], dim=1)  # [seq_len, d_model + d_model//2]
        encoding = self.fusion_net(combined_feat)  # [seq_len, d_model]
        
        return encoding.unsqueeze(0)  # [1, seq_len, d_model]
