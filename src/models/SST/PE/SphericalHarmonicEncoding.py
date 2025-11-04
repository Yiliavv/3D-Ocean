"""
球谐波位置编码模块（重构版）
提供真正的球谐波编码，基于空间坐标（经纬度）
适用于全球海洋数据的球面几何特性
"""

import torch
import torch.nn as nn
import math
import numpy as np


def legendre_polynomial(l, m, x):
    """
    计算关联勒让德多项式 P_l^m(x)
    使用递推公式实现，支持 l <= 4
    
    Args:
        l: 阶数 (degree)
        m: 序数 (order)，|m| <= l
        x: cos(theta)，范围 [-1, 1]
    Returns:
        P_l^m(x)
    """
    # 确保 m >= 0，负数情况由球谐波函数处理
    m = abs(m)
    
    if l == 0:
        return torch.ones_like(x)
    
    # P_m^m(x) = (-1)^m * (2m-1)!! * (1-x^2)^(m/2)
    if l == m:
        pmm = torch.ones_like(x)
        somx2 = torch.sqrt((1 - x) * (1 + x))  # sqrt(1 - x^2)
        fact = 1.0
        for i in range(1, m + 1):
            pmm = pmm * (-fact) * somx2
            fact += 2.0
        return pmm
    
    # P_{m+1}^m(x) = x * (2m+1) * P_m^m(x)
    pmmp1 = x * (2 * m + 1) * legendre_polynomial(m, m, x)
    if l == m + 1:
        return pmmp1
    
    # 递推公式: (l-m) * P_l^m = x * (2l-1) * P_{l-1}^m - (l+m-1) * P_{l-2}^m
    pll = torch.zeros_like(x)
    pmm_val = legendre_polynomial(m, m, x)
    pmmp1_val = pmmp1
    
    for ll in range(m + 2, l + 1):
        pll = (x * (2 * ll - 1) * pmmp1_val - (ll + m - 1) * pmm_val) / (ll - m)
        pmm_val = pmmp1_val
        pmmp1_val = pll
    
    return pll


def spherical_harmonics(l, m, theta, phi):
    """
    计算实数球谐波函数 Y_l^m(theta, phi)
    
    定义：
        Y_l^m(θ, φ) = N_l^m * P_l^|m|(cos θ) * T_m(φ)
        
        其中：
        - N_l^m = sqrt[(2l+1) / (4π) * (l-|m|)! / (l+|m|)!] 归一化系数
        - P_l^|m| 是关联勒让德多项式
        - T_m(φ) = cos(m*φ) if m >= 0, sin(|m|*φ) if m < 0
    
    Args:
        l: 阶数 (degree), l >= 0
        m: 序数 (order), -l <= m <= l
        theta: 极角（余纬度），范围 [0, π]，theta = π/2 - latitude
        phi: 方位角（经度），范围 [0, 2π]
    
    Returns:
        Y_l^m(theta, phi): 实数球谐波值
    """
    # 计算归一化系数
    abs_m = abs(m)
    
    # 计算阶乘比 (l-|m|)! / (l+|m|)!
    factorial_ratio = 1.0
    for i in range(l - abs_m + 1, l + abs_m + 1):
        factorial_ratio /= i
    
    normalization = math.sqrt((2 * l + 1) / (4 * math.pi) * factorial_ratio)
    
    # 计算关联勒让德多项式
    cos_theta = torch.cos(theta)
    plm = legendre_polynomial(l, abs_m, cos_theta)
    
    # 计算三角函数部分
    if m > 0:
        trig_part = torch.cos(m * phi)
    elif m < 0:
        trig_part = torch.sin(abs_m * phi)
    else:
        trig_part = torch.ones_like(phi)
    
    return normalization * plm * trig_part


def compute_all_spherical_harmonics(theta, phi, max_degree):
    """
    计算所有球谐波基函数 Y_l^m，l=0...max_degree, m=-l...l
    
    Args:
        theta: [N] 极角张量
        phi: [N] 方位角张量
        max_degree: 最大阶数
    
    Returns:
        harmonics: [N, num_harmonics] 所有球谐波特征
                   num_harmonics = (max_degree + 1)^2
    """
    harmonics = []
    
    for l in range(max_degree + 1):
        for m in range(-l, l + 1):
            Y_lm = spherical_harmonics(l, m, theta, phi)
            harmonics.append(Y_lm)
    
    return torch.stack(harmonics, dim=-1)  # [N, num_harmonics]

class SpatialSphericalHarmonicEncoding(nn.Module):
    """
    空间球谐波编码（基于经纬度坐标）
    
    为每个空间网格点计算球谐波特征，提供球面几何感知的位置编码
    适用于全球海洋温度数据的空间特征提取
    """
    def __init__(self, lat_range, lon_range, max_degree=4, resolution=1.0):
        """
        Args:
            lat_range: [lat_min, lat_max] 纬度范围（度）
            lon_range: [lon_min, lon_max] 经度范围（度）
            d_model: 输出编码维度
            max_degree: 球谐波最大阶数
            resolution: 空间分辨率（度）
        """
        super().__init__()
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.max_degree = max_degree
        self.resolution = resolution
        
        # 计算球谐波特征数量: (max_degree + 1)^2
        self.num_harmonics = (max_degree + 1) ** 2
        
        # 生成空间网格
        self.lats = torch.arange(lat_range[0], lat_range[1], resolution)
        self.lons = torch.arange(lon_range[0], lon_range[1], resolution)
        self.height = len(self.lats)
        self.width = len(self.lons)
        
        # 预计算球谐波基函数（可缓存）
        self._precompute_harmonics()
        
         # 可学习的球谐波权重（每个基函数一个权重）
        self.harmonic_weights = nn.Parameter(
            torch.ones(self.num_harmonics) * 0.1  # 初始化为小的正值
        )
        
        # 可学习的空间偏置（每个位置独立）
        self.spatial_bias = nn.Parameter(
            torch.zeros(self.height, self.width) * 0.01
        )
    
    def _precompute_harmonics(self):
        """预计算所有空间位置的球谐波特征"""
        # 转换经纬度到球面坐标
        # theta (极角) = π/2 - latitude，范围 [0, π]
        # phi (方位角) = longitude，范围 [0, 2π]
        
        lat_grid, lon_grid = torch.meshgrid(self.lats, self.lons, indexing='ij')
        
        # 转换为弧度并计算球面坐标
        lat_rad = lat_grid * math.pi / 180.0
        lon_rad = lon_grid * math.pi / 180.0
        
        theta = math.pi / 2 - lat_rad  # 余纬度
        phi = lon_rad  # 方位角（需要归一化到 [0, 2π]）
        phi = torch.where(phi < 0, phi + 2 * math.pi, phi)
        
        # 展平为 [height * width]
        theta_flat = theta.flatten()
        phi_flat = phi.flatten()
        
        # 计算所有球谐波基函数
        harmonics = compute_all_spherical_harmonics(
            theta_flat, phi_flat, self.max_degree
        )  # [height*width, num_harmonics]
        
        # 重塑为 [height, width, num_harmonics]
        harmonics = harmonics.view(self.height, self.width, self.num_harmonics)
        
        # 注册为buffer（不参与梯度计算但会保存）
        self.register_buffer('harmonics', harmonics)
    
    def forward(self):
        """
        Returns:
            encoding: [height, width] 空间位置编码
        """
        
        # 加权求和：对 num_harmonics 维度进行加权求和
        weighted_harmonics = (self.harmonics * self.harmonic_weights).sum(dim=-1)
        
        # 添加空间偏置
        encoding = weighted_harmonics + self.spatial_bias
        # encoding: [height, width]
        
        return encoding