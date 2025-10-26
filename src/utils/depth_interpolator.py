"""
深度插值工具 - 将非均匀深度层插值到均匀5m网格
用于平滑三维温度场的垂直剖面
"""

import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
from typing import Literal

from src.dataset.Argo import ArgoDepthMap
from src.utils.log import Log


class DepthInterpolator:
    """
    深度插值器 - 将模型输出的非均匀深度层插值到均匀网格
    
    例如：
        模型输出20层，对应 depth_map 的索引 [0:20]
        实际深度：[0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
        插值到5m间隔：[0, 5, 10, 15, 20, 25, 30, ..., 180]
    
    用法：
        interpolator = DepthInterpolator(
            depth_indices=[0, 20],  # 使用 depth_map 的前20层
            target_interval=5,       # 插值到5m间隔
            method='pchip'           # 使用平滑插值
        )
        
        # temp_original: [batch, n_depth, height, width] 或 [n_depth, height, width]
        temp_interpolated = interpolator.interpolate(temp_original)
        # temp_interpolated: [..., n_interpolated_depth, height, width]
        
        # 获取插值后的深度列表
        depths = interpolator.get_target_depths()
    """
    
    def __init__(self, 
                 depth_indices: list[int] | int,
                 target_interval: float = 5.0,
                 method: Literal['linear', 'pchip', 'cubic'] = 'pchip',
                 extrapolate: bool = False):
        """
        初始化深度插值器
        
        Args:
            depth_indices: 深度索引范围 [start, end] 或单个深度层数
                          如果是整数，表示从0到该值的范围 [0, depth_indices]
            target_interval: 目标插值间隔（米），默认5米
            method: 插值方法
                - 'linear': 线性插值（最快，但可能不够平滑）
                - 'pchip': 分段三次Hermite插值（推荐，平滑且保形）
                - 'cubic': 三次样条插值（最平滑，但可能有振荡）
            extrapolate: 是否外推到目标范围外的深度（不推荐）
        """
        # 解析深度索引
        if isinstance(depth_indices, int):
            self.depth_start = 0
            self.depth_end = depth_indices
        else:
            self.depth_start = depth_indices[0]
            self.depth_end = depth_indices[1]
        
        self.target_interval = target_interval
        self.method = method
        self.extrapolate = extrapolate
        
        # 获取原始深度（非均匀）
        self.source_depths = np.array(ArgoDepthMap.get([self.depth_start, self.depth_end]))
        
        # 生成目标深度（均匀间隔）
        max_depth = self.source_depths[-1]
        self.target_depths = np.arange(0, max_depth + target_interval, target_interval)
        
        # 确保不超过最大深度
        self.target_depths = self.target_depths[self.target_depths <= max_depth]
        
        Log.i(f"DepthInterpolator 初始化:")
        Log.i(f"  原始深度层数: {len(self.source_depths)}")
        Log.i(f"  原始深度范围: {self.source_depths[0]}m - {self.source_depths[-1]}m")
        Log.i(f"  目标插值间隔: {target_interval}m")
        Log.i(f"  目标深度层数: {len(self.target_depths)}")
        Log.i(f"  插值方法: {method}")
    
    def interpolate(self, temperature: np.ndarray) -> np.ndarray:
        """
        对温度场进行深度插值
        
        Args:
            temperature: 温度数据，形状可以是：
                - [n_depth, height, width]
                - [batch, n_depth, height, width]
        
        Returns:
            插值后的温度数据，深度维度被插值到目标深度网格
            形状与输入相同，只是深度维度大小改变
        """
        original_shape = temperature.shape
        is_batched = len(original_shape) == 4
        
        if is_batched:
            batch_size = original_shape[0]
            n_depth = original_shape[1]
            height = original_shape[2]
            width = original_shape[3]
        else:
            n_depth = original_shape[0]
            height = original_shape[1]
            width = original_shape[2]
        
        # 验证深度维度
        if n_depth != len(self.source_depths):
            raise ValueError(
                f"温度数据的深度维度 ({n_depth}) 与配置的深度层数 ({len(self.source_depths)}) 不匹配"
            )
        
        # 准备输出数组
        if is_batched:
            temp_interpolated = np.full(
                (batch_size, len(self.target_depths), height, width),
                np.nan,
                dtype=np.float32
            )
        else:
            temp_interpolated = np.full(
                (len(self.target_depths), height, width),
                np.nan,
                dtype=np.float32
            )
        
        # 执行插值
        if is_batched:
            for b in range(batch_size):
                temp_interpolated[b] = self._interpolate_single(temperature[b])
        else:
            temp_interpolated = self._interpolate_single(temperature)
        
        return temp_interpolated
    
    def _interpolate_single(self, temperature: np.ndarray) -> np.ndarray:
        """
        对单个样本进行插值（不含batch维度）
        
        Args:
            temperature: [n_depth, height, width]
        
        Returns:
            插值后的温度: [n_target_depth, height, width]
        """
        n_depth, height, width = temperature.shape
        temp_interpolated = np.full(
            (len(self.target_depths), height, width),
            np.nan,
            dtype=np.float32
        )
        
        # 对每个空间位置独立插值
        for i in range(height):
            for j in range(width):
                # 提取该位置的垂直剖面
                profile = temperature[:, i, j]
                
                # 跳过全是NaN的剖面（陆地）
                if np.all(np.isnan(profile)):
                    continue
                
                # 找到有效数据点
                valid_mask = ~np.isnan(profile)
                
                if valid_mask.sum() < 2:
                    # 至少需要2个有效点才能插值
                    continue
                
                valid_depths = self.source_depths[valid_mask]
                valid_temps = profile[valid_mask]
                
                # 执行插值
                try:
                    if self.method == 'linear':
                        interpolator = interp1d(
                            valid_depths, valid_temps,
                            kind='linear',
                            bounds_error=False,
                            fill_value=np.nan
                        )
                    elif self.method == 'pchip':
                        # PCHIP 保形插值，避免非物理振荡
                        interpolator = PchipInterpolator(
                            valid_depths, valid_temps,
                            extrapolate=self.extrapolate
                        )
                    elif self.method == 'cubic':
                        interpolator = interp1d(
                            valid_depths, valid_temps,
                            kind='cubic',
                            bounds_error=False,
                            fill_value=np.nan
                        )
                    else:
                        raise ValueError(f"不支持的插值方法: {self.method}")
                    
                    # 计算插值结果
                    temp_interpolated[:, i, j] = interpolator(self.target_depths)
                    
                except Exception as e:
                    # 插值失败，保持为NaN
                    Log.w(f"插值失败 at ({i}, {j}): {e}")
                    continue
        
        return temp_interpolated
    
    def get_source_depths(self) -> np.ndarray:
        """获取原始深度列表（米）"""
        return self.source_depths.copy()
    
    def get_target_depths(self) -> np.ndarray:
        """获取目标深度列表（米）"""
        return self.target_depths.copy()
    
    def get_depth_info(self) -> dict:
        """
        获取深度信息字典
        
        Returns:
            包含深度信息的字典：
            - source_depths: 原始深度数组
            - target_depths: 目标深度数组
            - source_count: 原始深度层数
            - target_count: 目标深度层数
            - interval: 目标间隔
            - method: 插值方法
        """
        return {
            'source_depths': self.source_depths,
            'target_depths': self.target_depths,
            'source_count': len(self.source_depths),
            'target_count': len(self.target_depths),
            'interval': self.target_interval,
            'method': self.method,
        }
    
    def plot_comparison(self, 
                       original_profile: np.ndarray,
                       interpolated_profile: np.ndarray,
                       title: str = "Temperature Profile Comparison"):
        """
        绘制单个位置的原始剖面和插值剖面对比图
        
        Args:
            original_profile: 原始温度剖面 [n_depth]
            interpolated_profile: 插值后的温度剖面 [n_target_depth]
            title: 图表标题
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 10))
        
        # 绘制原始数据（散点）
        valid_mask = ~np.isnan(original_profile)
        ax.scatter(original_profile[valid_mask], 
                  -self.source_depths[valid_mask],
                  c='red', s=50, label='Original (model output)', 
                  zorder=3, alpha=0.7)
        
        # 绘制插值数据（线）
        valid_interp = ~np.isnan(interpolated_profile)
        ax.plot(interpolated_profile[valid_interp],
               -self.target_depths[valid_interp],
               'b-', linewidth=2, label=f'Interpolated ({self.target_interval}m)', 
               alpha=0.8)
        
        # 设置标签和标题
        ax.set_xlabel('Temperature (°C)', fontsize=12)
        ax.set_ylabel('Depth (m)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)
        
        # 反转y轴（深度向下）
        ax.invert_yaxis()
        
        plt.tight_layout()
        return fig, ax

