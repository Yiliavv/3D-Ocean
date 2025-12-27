# -*- coding: utf-8 -*-
"""
区域分析模块 - 对关键海洋区域进行针对性分析

支持的区域：
- NINO 3.4: 厄尔尼诺/拉尼娜监测区（5°S-5°N, 170°W-120°W）
- NINO 3: 东太平洋暖池（5°S-5°N, 150°W-90°W）
- 赤道太平洋暖池: Western Pacific Warm Pool（5°S-5°N, 120°E-180°E）
- 墨西哥湾暖流: Gulf Stream（25°N-45°N, 80°W-40°W）
- 黑潮区域: Kuroshio Current（25°N-40°N, 125°E-170°E）
- 南大洋西风漂流: Antarctic Circumpolar Current（40°S-60°S, 全经度）
- 北印度洋: North Indian Ocean（0°-25°N, 50°E-100°E）
- 北大西洋副极地: North Atlantic Subpolar（45°N-65°N, 60°W-10°W）
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from cmocean import cm

from src.plot.base import create_carto_ax


# 定义关键海洋区域
OCEAN_REGIONS = {
    'nino34': {
        'name': 'NINO 3.4',
        'name_cn': '厄尔尼诺监测区',
        'lon': [-170, -120],
        'lat': [-5, 5],
        'color': '#E53935',  # 红色
        'description': '厄尔尼诺/拉尼娜现象的核心监测区域，海表温度异常直接反映ENSO状态'
    },
    'nino3': {
        'name': 'NINO 3',
        'name_cn': '东太平洋暖池',
        'lon': [-150, -90],
        'lat': [-5, 5],
        'color': '#FF7043',  # 橙红色
        'description': '赤道东太平洋冷舌区域，温度变化幅度大，对ENSO响应敏感'
    },
    'warm_pool': {
        'name': 'Western Pacific Warm Pool',
        'name_cn': '赤道太平洋暖池',
        'lon': [120, 180],
        'lat': [-5, 5],
        'color': '#FFA726',  # 橙色
        'description': '全球海表温度最高区域(>28°C)，是大气对流和降水的主要热源'
    },
    'gulf_stream': {
        'name': 'Gulf Stream',
        'name_cn': '墨西哥湾暖流',
        'lon': [-80, -40],
        'lat': [25, 45],
        'color': '#42A5F5',  # 蓝色
        'description': '北大西洋最强暖流，每秒输送约3000万立方米温暖海水向北，对欧洲气候有重要影响'
    },
    'kuroshio': {
        'name': 'Kuroshio Current',
        'name_cn': '黑潮区域',
        'lon': [125, 170],
        'lat': [25, 40],
        'color': '#7E57C2',  # 紫色
        'description': '西北太平洋边界流，与墨西哥湾暖流齐名的全球最强暖流之一'
    },
    'acc': {
        'name': 'Antarctic Circumpolar Current',
        'name_cn': '南大洋西风漂流',
        'lon': [-180, 180],
        'lat': [-60, -40],
        'color': '#26A69A',  # 青色
        'description': '全球唯一环绕地球的洋流，连接三大洋，温度变化相对平缓'
    },
    'north_indian': {
        'name': 'North Indian Ocean',
        'name_cn': '北印度洋',
        'lon': [50, 100],
        'lat': [0, 25],
        'color': '#66BB6A',  # 绿色
        'description': '季风影响显著区域，夏季西南季风带来强烈上升流'
    },
    'north_atlantic_subpolar': {
        'name': 'North Atlantic Subpolar',
        'name_cn': '北大西洋副极地',
        'lon': [-60, -10],
        'lat': [45, 65],
        'color': '#5C6BC0',  # 靛蓝色
        'description': '北大西洋深层水形成区域，对全球热盐环流至关重要'
    }
}


class RegionalAnalysis:
    """
    区域分析类 - 对海洋预测结果进行区域性统计分析
    
    使用示例:
        analyzer = RegionalAnalysis(
            pred_output=pred_output,  # 预测结果 [height, width]
            true_output=true_output,  # 真实值 [height, width]
            lon=[-180, 180],
            lat=[-80, 80],
            resolution=1
        )
        
        # 计算所有区域的统计指标
        stats = analyzer.compute_all_regions()
        
        # 绘制区域对比图
        analyzer.plot_regional_comparison()
        
        # 绘制单个区域的详细分析
        analyzer.plot_region_detail('nino34')
    """
    
    def __init__(self, pred_output: np.ndarray, true_output: np.ndarray,
                 lon: list, lat: list, resolution: float = 1,
                 attention_weights: np.ndarray = None,
                 ssta: np.ndarray = None):
        """
        初始化区域分析器
        
        :param pred_output: 预测结果，形状 [height, width]
        :param true_output: 真实值，形状 [height, width]
        :param lon: 经度范围 [lon_min, lon_max]
        :param lat: 纬度范围 [lat_min, lat_max]
        :param resolution: 空间分辨率（度）
        :param attention_weights: 注意力权重（可选），形状 [height, width]
        :param ssta: 海表温度异常（可选），形状 [height, width]
        """
        self.pred = pred_output
        self.true = true_output
        self.lon_range = lon
        self.lat_range = lat
        self.resolution = resolution
        self.attention = attention_weights
        self.ssta = ssta
        
        # 生成完整的经纬度数组
        self.height, self.width = pred_output.shape
        self.lon_array = np.linspace(lon[0], lon[1], self.width, endpoint=False)
        self.lat_array = np.linspace(lat[0], lat[1], self.height)
        
        # 计算预测误差
        self.error = pred_output - true_output
        
    def _get_region_mask(self, region_key: str) -> np.ndarray:
        """
        获取指定区域的掩码
        
        :param region_key: 区域键名
        :return: 布尔掩码数组
        """
        region = OCEAN_REGIONS[region_key]
        lon_range = region['lon']
        lat_range = region['lat']
        
        # 处理跨越180度经线的情况（如南大洋）
        if lon_range[0] == -180 and lon_range[1] == 180:
            lon_mask = np.ones(self.width, dtype=bool)
        elif lon_range[0] > lon_range[1]:
            # 跨越日期变更线
            lon_mask = (self.lon_array >= lon_range[0]) | (self.lon_array <= lon_range[1])
        else:
            lon_mask = (self.lon_array >= lon_range[0]) & (self.lon_array <= lon_range[1])
        
        lat_mask = (self.lat_array >= lat_range[0]) & (self.lat_array <= lat_range[1])
        
        # 创建二维掩码
        mask = np.outer(lat_mask, lon_mask)
        
        return mask
    
    def _extract_region_data(self, data: np.ndarray, region_key: str) -> np.ndarray:
        """
        提取指定区域的数据
        
        :param data: 原始数据
        :param region_key: 区域键名
        :return: 区域内的有效数据（一维数组）
        """
        mask = self._get_region_mask(region_key)
        region_data = data[mask]
        
        # 排除 NaN 值（陆地区域）
        valid_data = region_data[~np.isnan(region_data)]
        
        return valid_data
    
    def compute_region_stats(self, region_key: str) -> dict:
        """
        计算单个区域的统计指标
        
        :param region_key: 区域键名
        :return: 统计指标字典
        """
        region = OCEAN_REGIONS[region_key]
        
        # 提取区域数据
        pred_data = self._extract_region_data(self.pred, region_key)
        true_data = self._extract_region_data(self.true, region_key)
        error_data = self._extract_region_data(self.error, region_key)
        
        if len(pred_data) == 0 or len(true_data) == 0:
            return {
                'region': region['name'],
                'region_cn': region['name_cn'],
                'valid': False,
                'message': '该区域无有效数据'
            }
        
        # 基本统计
        stats = {
            'region': region['name'],
            'region_cn': region['name_cn'],
            'valid': True,
            'pixel_count': len(pred_data),
            
            # 温度统计
            'pred_mean': np.mean(pred_data),
            'pred_std': np.std(pred_data),
            'pred_min': np.min(pred_data),
            'pred_max': np.max(pred_data),
            
            'true_mean': np.mean(true_data),
            'true_std': np.std(true_data),
            'true_min': np.min(true_data),
            'true_max': np.max(true_data),
            
            # 误差统计
            'rmse': np.sqrt(np.mean(error_data ** 2)),
            'mae': np.mean(np.abs(error_data)),
            'bias': np.mean(error_data),  # 系统偏差
            'error_std': np.std(error_data),
            'error_min': np.min(error_data),
            'error_max': np.max(error_data),
            
            # R² 决定系数
            'r2': 1 - np.sum(error_data ** 2) / np.sum((true_data - np.mean(true_data)) ** 2),
            
            # 温度梯度（空间变异性）
            'spatial_variability': np.std(true_data),
        }
        
        # 如果有 SSTA 数据，计算区域 SSTA 统计
        if self.ssta is not None:
            ssta_data = self._extract_region_data(self.ssta, region_key)
            if len(ssta_data) > 0:
                stats['ssta_mean'] = np.mean(ssta_data)
                stats['ssta_std'] = np.std(ssta_data)
        
        # 如果有注意力权重，计算区域平均注意力
        if self.attention is not None:
            attn_data = self._extract_region_data(self.attention, region_key)
            if len(attn_data) > 0:
                stats['attention_mean'] = np.mean(attn_data)
                stats['attention_std'] = np.std(attn_data)
                stats['attention_max'] = np.max(attn_data)
        
        return stats
    
    def compute_all_regions(self, regions: list = None) -> dict:
        """
        计算所有（或指定）区域的统计指标
        
        :param regions: 区域键名列表，默认为所有区域
        :return: 区域统计字典
        """
        if regions is None:
            regions = list(OCEAN_REGIONS.keys())
        
        all_stats = {}
        for region_key in regions:
            if region_key in OCEAN_REGIONS:
                all_stats[region_key] = self.compute_region_stats(region_key)
        
        return all_stats
    
    def print_regional_report(self, regions: list = None):
        """
        打印区域分析报告
        
        :param regions: 区域键名列表
        """
        stats = self.compute_all_regions(regions)
        
        print("\n" + "=" * 80)
        print("📊 海洋区域分析报告")
        print("=" * 80)
        
        for region_key, region_stats in stats.items():
            if not region_stats['valid']:
                continue
            
            region_info = OCEAN_REGIONS[region_key]
            print(f"\n▶ {region_stats['region_cn']} ({region_stats['region']})")
            print(f"  经度范围: {region_info['lon'][0]}° ~ {region_info['lon'][1]}°")
            print(f"  纬度范围: {region_info['lat'][0]}° ~ {region_info['lat'][1]}°")
            print(f"  有效像素: {region_stats['pixel_count']}")
            print("-" * 60)
            
            print(f"  【温度统计】")
            print(f"    真实值 - 均值: {region_stats['true_mean']:.2f}°C, "
                  f"标准差: {region_stats['true_std']:.2f}°C, "
                  f"范围: [{region_stats['true_min']:.2f}, {region_stats['true_max']:.2f}]°C")
            print(f"    预测值 - 均值: {region_stats['pred_mean']:.2f}°C, "
                  f"标准差: {region_stats['pred_std']:.2f}°C, "
                  f"范围: [{region_stats['pred_min']:.2f}, {region_stats['pred_max']:.2f}]°C")
            
            print(f"  【误差分析】")
            print(f"    RMSE: {region_stats['rmse']:.4f}°C")
            print(f"    MAE: {region_stats['mae']:.4f}°C")
            print(f"    系统偏差 (Bias): {region_stats['bias']:+.4f}°C")
            print(f"    R²: {region_stats['r2']:.4f}")
            print(f"    误差范围: [{region_stats['error_min']:.3f}, {region_stats['error_max']:.3f}]°C")
            
            if 'ssta_mean' in region_stats:
                print(f"  【温度异常 (SSTA)】")
                print(f"    均值: {region_stats['ssta_mean']:+.3f}°C, "
                      f"标准差: {region_stats['ssta_std']:.3f}°C")
            
            if 'attention_mean' in region_stats:
                print(f"  【注意力权重】")
                print(f"    均值: {region_stats['attention_mean']:.4f}, "
                      f"最大值: {region_stats['attention_max']:.4f}")
            
            print(f"  【区域特征】")
            print(f"    {region_info['description']}")
        
        print("\n" + "=" * 80)
        
        return stats
    
    def plot_regional_comparison(self, save_path: str = None, figsize=(12, 10)):
        """
        绘制区域对比图 - Nature 风格综合图（上下布局）
        
        :param save_path: 保存路径
        :param figsize: 图像尺寸
        """
        stats = self.compute_all_regions()
        
        # 过滤有效区域
        valid_stats = {k: v for k, v in stats.items() if v['valid']}
        
        # 准备数据
        regions = list(valid_stats.keys())
        region_names_en = [valid_stats[r]['region'] for r in regions]
        rmse_values = np.array([valid_stats[r]['rmse'] for r in regions])
        r2_values = np.array([valid_stats[r]['r2'] for r in regions])
        mae_values = np.array([valid_stats[r]['mae'] for r in regions])
        bias_values = np.array([valid_stats[r]['bias'] for r in regions])
        spatial_var = np.array([valid_stats[r]['spatial_variability'] for r in regions])
        true_mean = np.array([valid_stats[r]['true_mean'] for r in regions])
        colors = [OCEAN_REGIONS[r]['color'] for r in regions]
        
        # 按 RMSE 排序
        sort_idx = np.argsort(rmse_values)
        
        # ============ Nature 风格设置 ============
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
            'font.size': 9,
            'axes.linewidth': 0.8,
            'axes.labelsize': 10,
            'axes.titlesize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 8,
            'figure.dpi': 300,
        })
        
        # 创建图像 - 上下布局
        fig = plt.figure(figsize=figsize, facecolor='white')
        gs = GridSpec(2, 1, figure=fig, height_ratios=[1.4, 1], hspace=0.25)
        
        # ============ (a) 上方：地图展示区域性能 ============
        ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
        ax_map.set_extent([self.lon_range[0], self.lon_range[1], 
                          self.lat_range[0], self.lat_range[1]], 
                         crs=ccrs.PlateCarree())
        
        # 添加地理要素
        ax_map.coastlines(resolution='110m', linewidth=0.4, color='#404040')
        ax_map.add_feature(cfeature.LAND, facecolor='#f0f0f0', edgecolor='none')
        
        # 添加网格线
        gl = ax_map.gridlines(draw_labels=True, linewidth=0.3, color='gray', 
                             alpha=0.5, linestyle='-')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
        
        # 绘制误差分布背景
        lon_grid, lat_grid = np.meshgrid(self.lon_array, self.lat_array)
        abs_max = min(1.2, max(abs(np.nanmin(self.error)), abs(np.nanmax(self.error))))
        levels = np.linspace(-abs_max, abs_max, 25)
        
        # 使用更柔和的配色
        error_colors = ['#2166ac', '#67a9cf', '#d1e5f0', '#f7f7f7', 
                       '#fddbc7', '#ef8a62', '#b2182b']
        error_cmap = LinearSegmentedColormap.from_list('error', error_colors, N=256)
        
        cf = ax_map.contourf(lon_grid, lat_grid, self.error, levels=levels,
                           cmap=error_cmap, extend='both', transform=ccrs.PlateCarree(),
                           alpha=0.85)
        
        # 绘制各区域边界和标注
        short_names = {
            'nino34': 'N3.4', 'nino3': 'N3', 'warm_pool': 'WP',
            'gulf_stream': 'GS', 'kuroshio': 'KS', 'acc': 'ACC',
            'north_indian': 'NIO', 'north_atlantic_subpolar': 'NASP'
        }
        
        for region_key in regions:
            region_info = OCEAN_REGIONS[region_key]
            lon = region_info['lon']
            lat = region_info['lat']
            color = region_info['color']
            rmse = valid_stats[region_key]['rmse']
            r2 = valid_stats[region_key]['r2']
            
            # 处理全经度情况
            if lon[0] == -180 and lon[1] == 180:
                width = 358
                lon_start = -179
            else:
                width = lon[1] - lon[0]
                lon_start = lon[0]
            
            height = lat[1] - lat[0]
            
            # 绘制边界框
            rect = mpatches.Rectangle(
                (lon_start, lat[0]), width, height,
                linewidth=1.5, edgecolor=color, facecolor='none',
                transform=ccrs.PlateCarree(), zorder=10
            )
            ax_map.add_patch(rect)
            
            # 在区域中心添加标签
            center_lon = lon_start + width / 2
            center_lat = lat[0] + height / 2
            
            # 使用圆圈大小表示 RMSE，颜色表示 R²
            size = 80 + rmse * 150
            ax_map.scatter(center_lon, center_lat, s=size, c=[r2], cmap='RdYlGn',
                          vmin=0, vmax=1, edgecolor='black', linewidth=0.8,
                          transform=ccrs.PlateCarree(), zorder=11)
            
            ax_map.text(center_lon, center_lat, short_names.get(region_key, ''),
                       ha='center', va='center', fontsize=7, fontweight='bold',
                       color='white', transform=ccrs.PlateCarree(), zorder=12)
        
        # 添加误差色标（地图右侧）
        cbar_ax = fig.add_axes([0.92, 0.52, 0.015, 0.35])
        cbar = fig.colorbar(cf, cax=cbar_ax, orientation='vertical')
        cbar.set_label('Prediction Error (°C)', fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        
        ax_map.set_title('(a) Regional prediction performance overview', 
                        fontsize=11, fontweight='bold', loc='left', pad=8)
        
        # ============ (b) 下方：水平统计图 ============
        ax_stats = fig.add_subplot(gs[1])
        
        # 排序数据（按 RMSE 从小到大）
        sorted_names = [region_names_en[i] for i in sort_idx]
        sorted_rmse = rmse_values[sort_idx]
        sorted_r2 = r2_values[sort_idx]
        sorted_bias = bias_values[sort_idx]
        sorted_colors = [colors[i] for i in sort_idx]
        
        n_regions = len(regions)
        x_pos = np.arange(n_regions)
        
        # 全球 RMSE 参考线
        global_rmse = np.sqrt(np.nanmean(self.error ** 2))
        
        # 绘制垂直 Lollipop chart
        for i, (rmse_val, r2_val, bias_val, color) in enumerate(
            zip(sorted_rmse, sorted_r2, sorted_bias, sorted_colors)):
            
            # 主线（垂直）
            ax_stats.vlines(x=i, ymin=0, ymax=rmse_val, color=color, 
                           linewidth=3, alpha=0.85)
            
            # 圆点 - 颜色编码 R²
            marker_size = 180
            ax_stats.scatter(i, rmse_val, s=marker_size, 
                           c=[r2_val], cmap='RdYlGn', vmin=0, vmax=1,
                           edgecolor='#333333', linewidth=1.5, zorder=5)
            
            # 在圆点上方显示 R² 值
            ax_stats.text(i, rmse_val + 0.08, f'{r2_val:.2f}', ha='center', va='bottom',
                         fontsize=8, fontweight='bold', color='#333333')
        
        # 添加全球平均参考线
        ax_stats.axhline(y=global_rmse, color='#D32F2F', linestyle='--', 
                        linewidth=1.5, alpha=0.8, zorder=1)
        ax_stats.text(n_regions - 0.5, global_rmse + 0.03, f'Global: {global_rmse:.3f}°C', 
                     ha='right', va='bottom', fontsize=8, color='#D32F2F', 
                     fontweight='bold')
        
        # 设置 X 轴
        ax_stats.set_xticks(x_pos)
        ax_stats.set_xticklabels(sorted_names, fontsize=9, rotation=25, ha='right')
        ax_stats.set_ylabel('RMSE (°C)', fontsize=10)
        ax_stats.set_xlim(-0.5, n_regions - 0.5)
        ax_stats.set_ylim(0, max(sorted_rmse) * 1.25)
        
        # 美化轴线
        ax_stats.spines['top'].set_visible(False)
        ax_stats.spines['right'].set_visible(False)
        ax_stats.spines['left'].set_linewidth(0.5)
        ax_stats.spines['bottom'].set_linewidth(0.5)
        
        # 添加网格线
        ax_stats.yaxis.grid(True, linestyle='-', alpha=0.2, zorder=0)
        ax_stats.set_axisbelow(True)
        
        # 在底部添加 Bias 迷你条（垂直向下）
        ax_bias = ax_stats.twiny()
        ax_bias.set_xlim(ax_stats.get_xlim())
        ax_bias.set_xticks([])
        ax_bias.spines['top'].set_visible(False)
        
        # 绘制 Bias 指示条（在 X 轴下方）
        bias_max = max(abs(sorted_bias.min()), abs(sorted_bias.max()))
        
        for i, bias_val in enumerate(sorted_bias):
            bar_color = '#E57373' if bias_val > 0 else '#64B5F6'
            bar_height = abs(bias_val) / bias_max * 0.15 if bias_max > 0 else 0
            
            # 在柱子底部绘制 Bias 条
            ax_stats.bar(i, bar_height, bottom=-0.02, width=0.5,
                        color=bar_color, alpha=0.8, edgecolor='none', zorder=3)
            
            # 添加 Bias 数值
            ax_stats.text(i, -0.22, f'{bias_val:+.2f}', ha='center', va='top',
                         fontsize=7, color='#555555')
        
        ax_stats.set_title('(b) Regional prediction performance ranking (sorted by RMSE)', 
                          fontsize=10, fontweight='bold', loc='left', pad=10)
        
        # R² 色标
        sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar_ax2 = fig.add_axes([0.92, 0.12, 0.015, 0.25])
        cbar2 = fig.colorbar(sm, cax=cbar_ax2)
        cbar2.set_label('R²', fontsize=9)
        cbar2.ax.tick_params(labelsize=7)
        
        # 添加 Bias 图例
        legend_elements = [
            mpatches.Patch(facecolor='#E57373', edgecolor='none', label='Bias > 0 (warm)'),
            mpatches.Patch(facecolor='#64B5F6', edgecolor='none', label='Bias < 0 (cold)'),
        ]
        ax_stats.legend(handles=legend_elements, loc='upper right', fontsize=8,
                       framealpha=0.95, edgecolor='gray')
        
        # 调整布局
        plt.subplots_adjust(left=0.08, right=0.90, top=0.95, bottom=0.12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white',
                       edgecolor='none', pad_inches=0.1)
            print(f"✅ 区域对比图已保存: {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_regional_map(self, save_path: str = None, show_error: bool = True):
        """
        在地图上绘制各区域位置和误差分布
        
        :param save_path: 保存路径
        :param show_error: 是否显示误差分布
        """
        fig = plt.figure(figsize=(18, 10), dpi=150)
        
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent([self.lon_range[0], self.lon_range[1], 
                      self.lat_range[0], self.lat_range[1]], 
                     crs=ccrs.PlateCarree())
        
        # 添加海岸线
        ax.coastlines(resolution='110m', linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor='white', alpha=0.3)
        
        # 生成网格
        lon_grid, lat_grid = np.meshgrid(self.lon_array, self.lat_array)
        
        if show_error:
            # 绘制误差分布
            abs_max = min(1.5, max(abs(np.nanmin(self.error)), abs(np.nanmax(self.error))))
            levels = np.linspace(-abs_max, abs_max, 30)
            
            # 创建误差色标
            colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7',
                     '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
            cmap = LinearSegmentedColormap.from_list('error', colors, N=256)
            
            im = ax.contourf(lon_grid, lat_grid, self.error, levels=levels,
                           cmap=cmap, extend='both', transform=ccrs.PlateCarree())
            cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
            cbar.set_label('预测误差 (°C)', fontsize=12)
        
        # 绘制各区域边界框
        stats = self.compute_all_regions()
        
        for region_key, region_info in OCEAN_REGIONS.items():
            lon = region_info['lon']
            lat = region_info['lat']
            color = region_info['color']
            
            # 处理全经度范围的情况
            if lon[0] == -180 and lon[1] == 180:
                width = 360
                lon_start = -180
            else:
                width = lon[1] - lon[0]
                lon_start = lon[0]
            
            height = lat[1] - lat[0]
            
            # 绘制边界框
            rect = mpatches.Rectangle(
                (lon_start, lat[0]), width, height,
                linewidth=2.5, edgecolor=color, facecolor='none',
                transform=ccrs.PlateCarree(), zorder=10
            )
            ax.add_patch(rect)
            
            # 添加区域标签和 RMSE
            center_lon = lon_start + width / 2
            center_lat = lat[0] + height / 2
            
            if region_key in stats and stats[region_key]['valid']:
                rmse = stats[region_key]['rmse']
                label_text = f"{region_info['name_cn']}\nRMSE={rmse:.3f}°C"
            else:
                label_text = region_info['name_cn']
            
            ax.text(center_lon, center_lat, label_text,
                   transform=ccrs.PlateCarree(),
                   fontsize=9, fontweight='bold',
                   ha='center', va='center',
                   color=color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            alpha=0.85, edgecolor=color, linewidth=1.5),
                   zorder=11)
        
        # 添加网格线
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                         alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        ax.set_title('海洋关键区域预测误差分布', fontsize=16, fontweight='bold', pad=15)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ 区域地图已保存: {save_path}")
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_region_detail(self, region_key: str, save_path: str = None):
        """
        绘制单个区域的详细分析图 - Nature期刊级别专业图表
        
        设计参考 Nature 期刊规范：
        - 字体: 5-7pt
        - 线条: 0.25-1pt
        - 单栏宽度: 89mm, 双栏: 183mm
        - 色盲友好配色
        
        :param region_key: 区域键名
        :param save_path: 保存路径
        """
        if region_key not in OCEAN_REGIONS:
            print(f"❌ Unknown region: {region_key}")
            return None
        
        region_info = OCEAN_REGIONS[region_key]
        stats = self.compute_region_stats(region_key)
        
        if not stats['valid']:
            print(f"❌ Region {region_key} has no valid data")
            return None
        
        # 提取区域范围
        lon_range = region_info['lon']
        lat_range = region_info['lat']
        
        if lon_range[0] == -180 and lon_range[1] == 180:
            lon_idx = slice(None)
        else:
            lon_idx = np.where((self.lon_array >= lon_range[0]) & 
                              (self.lon_array <= lon_range[1]))[0]
        
        lat_idx = np.where((self.lat_array >= lat_range[0]) & 
                          (self.lat_array <= lat_range[1]))[0]
        
        if len(lat_idx) == 0:
            print(f"❌ Cannot extract data for region {region_key}")
            return None
        
        # 提取子区域数据
        pred_region = self.pred[lat_idx[0]:lat_idx[-1]+1, :]
        true_region = self.true[lat_idx[0]:lat_idx[-1]+1, :]
        error_region = self.error[lat_idx[0]:lat_idx[-1]+1, :]
        
        if not isinstance(lon_idx, slice):
            pred_region = pred_region[:, lon_idx]
            true_region = true_region[:, lon_idx]
            error_region = error_region[:, lon_idx]
        
        # ============ Nature 风格设置 ============
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
            'font.size': 7,
            'axes.linewidth': 0.5,
            'axes.labelsize': 7,
            'axes.titlesize': 8,
            'xtick.labelsize': 6,
            'ytick.labelsize': 6,
            'legend.fontsize': 6,
            'lines.linewidth': 0.75,
            'patch.linewidth': 0.5,
        })
        
        # Nature 双栏图尺寸 (183mm ≈ 7.2 inch)
        fig = plt.figure(figsize=(7.2, 5.5), dpi=300, facecolor='white')
        
        # 紧凑布局：2行3列
        gs = GridSpec(2, 3, figure=fig, height_ratios=[1.1, 1], 
                     hspace=0.35, wspace=0.7,
                     left=0.06, right=0.96, top=0.90, bottom=0.10)
        
        # 提取绘图范围
        plot_lon = self.lon_array if isinstance(lon_idx, slice) else self.lon_array[lon_idx]
        plot_lat = self.lat_array[lat_idx]
        lon_grid, lat_grid = np.meshgrid(plot_lon, plot_lat)
        
        # 统一温度范围
        vmin = min(np.nanmin(true_region), np.nanmin(pred_region))
        vmax = max(np.nanmax(true_region), np.nanmax(pred_region))
        
        # 色盲友好配色 (viridis 代替 thermal)
        cmap_sst = 'cmo.thermal'
        cmap_error = 'cmo.balance'  # 替代 RdBu_r，更专业
        
        # ========== (a) Observed SST ==========
        ax_obs = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        ax_obs.set_extent([plot_lon[0], plot_lon[-1], plot_lat[0], plot_lat[-1]], 
                         crs=ccrs.PlateCarree())
        ax_obs.coastlines(resolution='110m', linewidth=0.3, color='#333333')
        ax_obs.add_feature(cfeature.LAND, facecolor='#EEEEEE', edgecolor='none')
        
        im_obs = ax_obs.pcolormesh(lon_grid, lat_grid, true_region,
                                   cmap=cmap_sst, transform=ccrs.PlateCarree(),
                                   vmin=vmin, vmax=vmax, shading='auto')
        
        gl = ax_obs.gridlines(draw_labels=True, linewidth=0.2, color='gray', 
                             alpha=0.5, linestyle='-')
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {'size': 5}
        
        ax_obs.text(0.02, 0.98, 'a', transform=ax_obs.transAxes, fontsize=9, 
                   fontweight='bold', va='top')
        ax_obs.set_title('Observed', fontsize=7, pad=3)
        
        # ========== (b) Predicted SST ==========
        ax_pred = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
        ax_pred.set_extent([plot_lon[0], plot_lon[-1], plot_lat[0], plot_lat[-1]], 
                          crs=ccrs.PlateCarree())
        ax_pred.coastlines(resolution='110m', linewidth=0.3, color='#333333')
        ax_pred.add_feature(cfeature.LAND, facecolor='#EEEEEE', edgecolor='none')
        
        im_pred = ax_pred.pcolormesh(lon_grid, lat_grid, pred_region,
                                     cmap=cmap_sst, transform=ccrs.PlateCarree(),
                                     vmin=vmin, vmax=vmax, shading='auto')
        
        gl2 = ax_pred.gridlines(draw_labels=True, linewidth=0.2, color='gray', 
                               alpha=0.5, linestyle='-')
        gl2.top_labels = gl2.right_labels = gl2.left_labels = False
        gl2.xlabel_style = {'size': 5}
        
        ax_pred.text(0.02, 0.98, 'b', transform=ax_pred.transAxes, fontsize=9, 
                    fontweight='bold', va='top')
        ax_pred.set_title('Predicted', fontsize=7, pad=3)
        
        # ========== (c) Prediction Error ==========
        ax_err = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
        ax_err.set_extent([plot_lon[0], plot_lon[-1], plot_lat[0], plot_lat[-1]], 
                         crs=ccrs.PlateCarree())
        ax_err.coastlines(resolution='110m', linewidth=0.3, color='#333333')
        ax_err.add_feature(cfeature.LAND, facecolor='#EEEEEE', edgecolor='none')
        
        err_max = max(0.5, min(1.5, np.nanpercentile(np.abs(error_region), 99)))
        im_err = ax_err.pcolormesh(lon_grid, lat_grid, error_region,
                                   cmap=cmap_error, transform=ccrs.PlateCarree(),
                                   vmin=-err_max, vmax=err_max, shading='auto')
        
        gl3 = ax_err.gridlines(draw_labels=True, linewidth=0.2, color='gray', 
                              alpha=0.5, linestyle='-')
        gl3.top_labels = gl3.right_labels = gl3.left_labels = False
        gl3.xlabel_style = {'size': 5}
        
        ax_err.text(0.02, 0.98, 'c', transform=ax_err.transAxes, fontsize=9, 
                   fontweight='bold', va='top')
        ax_err.set_title(f'Error (RMSE={stats["rmse"]:.3f}°C)', fontsize=7, pad=3)
        
        # 绘制完所有地图后，获取实际位置来放置 colorbar
        fig.canvas.draw()
        
        # 获取子图位置
        pos_obs = ax_obs.get_position()
        pos_pred = ax_pred.get_position()
        pos_err = ax_err.get_position()
        
        # a,b 共享 colorbar - 居中于 a 和 b 之间
        cb1_left = pos_obs.x0
        cb1_right = pos_pred.x1
        cb1_width = cb1_right - cb1_left
        cb1_y = pos_obs.y0 - 0.06  # 在地图下方，稍微往下
        cax1 = fig.add_axes([cb1_left, cb1_y, cb1_width, 0.012])
        cb1 = plt.colorbar(im_obs, cax=cax1, orientation='horizontal')
        cb1.set_label('SST (°C)', fontsize=6, labelpad=2)
        cb1.ax.tick_params(labelsize=5, length=2, width=0.5)
        
        # c 的 colorbar - 与 c 对齐
        cb2_left = pos_err.x0
        cb2_width = pos_err.width
        cb2_y = pos_err.y0 - 0.04
        cax2 = fig.add_axes([cb2_left, cb2_y, cb2_width, 0.012])
        cb2 = plt.colorbar(im_err, cax=cax2, orientation='horizontal', extend='both')
        cb2.set_label('Error (°C)', fontsize=6, labelpad=2)
        cb2.ax.tick_params(labelsize=5, length=2, width=0.5)
        
        # ========== (d) Scatter Plot with Density ==========
        ax_scatter = fig.add_subplot(gs[1, 0])
        pred_flat = self._extract_region_data(self.pred, region_key)
        true_flat = self._extract_region_data(self.true, region_key)
        
        # 2D 直方图/密度图 - 更专业
        from matplotlib.colors import LogNorm
        h = ax_scatter.hist2d(true_flat, pred_flat, bins=50, cmap='cmo.dense', 
                             norm=LogNorm(), cmin=1)
        
        # 1:1 线
        lims = [min(true_flat.min(), pred_flat.min()), max(true_flat.max(), pred_flat.max())]
        ax_scatter.plot(lims, lims, 'k--', linewidth=0.75, label='1:1', zorder=10)
        
        # 线性拟合
        coef = np.polyfit(true_flat, pred_flat, 1)
        fit_fn = np.poly1d(coef)
        ax_scatter.plot(lims, fit_fn(lims), color='#E74C3C', linewidth=0.75, 
                       label=f'Fit (y={coef[0]:.2f}x{coef[1]:+.2f})', zorder=10)
        
        ax_scatter.set_xlabel('Observed (°C)', fontsize=7)
        ax_scatter.set_ylabel('Predicted (°C)', fontsize=7)
        ax_scatter.set_aspect('equal', adjustable='box')
        ax_scatter.legend(loc='lower right', fontsize=5, framealpha=0.9, 
                         handlelength=1.5, borderpad=0.3)
        
        # 统计信息
        ax_scatter.text(0.03, 0.97, f'R²={stats["r2"]:.3f}\nN={stats["pixel_count"]:,}', 
                       transform=ax_scatter.transAxes, fontsize=6, va='top',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                alpha=0.8, linewidth=0.3))
        
        ax_scatter.text(0.02, 1.02, 'd', transform=ax_scatter.transAxes, fontsize=9, 
                       fontweight='bold', va='bottom')
        
        # d 的密度 colorbar - 放在图的右侧而不是下方，避免与 x 轴重合
        fig.canvas.draw()
        pos_scatter = ax_scatter.get_position()
        cax3 = fig.add_axes([pos_scatter.x1 + 0.008, pos_scatter.y0 + 0.02, 0.008, pos_scatter.height * 0.8])
        cb3 = plt.colorbar(h[3], cax=cax3, orientation='vertical')
        cb3.set_label('Density', fontsize=5, labelpad=2)
        cb3.ax.tick_params(labelsize=4, length=1.5, width=0.3)
        
        # ========== (e) Error Histogram with KDE ==========
        ax_hist = fig.add_subplot(gs[1, 1])
        error_flat = self._extract_region_data(self.error, region_key)
        
        # 直方图 + KDE
        n, bins, patches = ax_hist.hist(error_flat, bins=50, density=True, 
                                        color='#3498DB', alpha=0.6, 
                                        edgecolor='white', linewidth=0.3)
        
        # KDE 曲线
        from scipy import stats as scipy_stats
        kde = scipy_stats.gaussian_kde(error_flat)
        x_kde = np.linspace(error_flat.min(), error_flat.max(), 200)
        ax_hist.plot(x_kde, kde(x_kde), color='#2C3E50', linewidth=1, label='KDE')
        
        # 零线和偏差线
        ax_hist.axvline(0, color='#E74C3C', linestyle='-', linewidth=0.75, label='Zero')
        ax_hist.axvline(stats['bias'], color='#27AE60', linestyle='--', linewidth=0.75,
                       label=f'Bias={stats["bias"]:+.3f}')
        
        ax_hist.set_xlabel('Error (°C)', fontsize=7)
        ax_hist.set_ylabel('Density', fontsize=7)
        ax_hist.legend(loc='upper right', fontsize=5, framealpha=0.9, 
                      handlelength=1.2, borderpad=0.3)
        
        ax_hist.text(0.02, 1.02, 'e', transform=ax_hist.transAxes, fontsize=9, 
                    fontweight='bold', va='bottom')
        
        # ========== (f) Taylor Diagram Style Metrics ==========
        ax_metrics = fig.add_subplot(gs[1, 2])
        ax_metrics.axis('off')
        
        # 极简统计表格
        metrics = [
            ('RMSE', f'{stats["rmse"]:.4f}', '°C'),
            ('MAE', f'{stats["mae"]:.4f}', '°C'),
            ('Bias', f'{stats["bias"]:+.4f}', '°C'),
            ('R²', f'{stats["r2"]:.4f}', ''),
            ('σ_obs', f'{np.nanstd(true_flat):.3f}', '°C'),
            ('σ_pred', f'{np.nanstd(pred_flat):.3f}', '°C'),
        ]
        
        # 绘制极简表格
        y_start = 0.85
        for i, (name, val, unit) in enumerate(metrics):
            y = y_start - i * 0.12
            ax_metrics.text(0.05, y, name, fontsize=7, fontweight='bold', 
                           transform=ax_metrics.transAxes, va='center')
            ax_metrics.text(0.45, y, val, fontsize=7, transform=ax_metrics.transAxes, 
                           va='center', ha='right', family='monospace')
            ax_metrics.text(0.48, y, unit, fontsize=6, transform=ax_metrics.transAxes, 
                           va='center', color='#666666')
        
        # 分隔线
        ax_metrics.axhline(y=0.90, xmin=0.02, xmax=0.55, color='#CCCCCC', 
                          linewidth=0.5, transform=ax_metrics.transAxes)
        ax_metrics.axhline(y=0.15, xmin=0.02, xmax=0.55, color='#CCCCCC', 
                          linewidth=0.5, transform=ax_metrics.transAxes)
        
        # 区域信息
        ax_metrics.text(0.05, 0.05, f'{region_info["name"]}', fontsize=7, 
                       fontweight='bold', transform=ax_metrics.transAxes)
        ax_metrics.text(0.05, -0.02, 
                       f'{region_info["lon"][0]}°~{region_info["lon"][1]}°E, '
                       f'{region_info["lat"][0]}°~{region_info["lat"][1]}°N',
                       fontsize=5, color='#666666', transform=ax_metrics.transAxes)
        
        ax_metrics.text(0.02, 1.02, 'f', transform=ax_metrics.transAxes, fontsize=9, 
                       fontweight='bold', va='bottom')
        
        # 总标题
        fig.suptitle(f'{region_info["name"]}', fontsize=9, fontweight='bold', y=0.96)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white',
                       edgecolor='none', pad_inches=0.05)
            print(f"✅ Figure saved: {save_path}")
        
        plt.show()
        return fig


def run_regional_analysis(predictor, area, resolution: float = 1,
                          test_offset: int = 520, 
                          save_dir: str = None,
                          detail_regions: list = None,
                          show_plots: bool = True):
    """
    完整的区域性分析 - 一键完成所有区域分析、绘图和报告生成
    
    :param predictor: BasePrediction 或 BaseTrainer 对象（需要有 predict 方法）
    :param area: Area 对象，包含经纬度范围
    :param resolution: 空间分辨率（度）
    :param test_offset: 测试时间点，默认 520
    :param save_dir: 保存目录，默认 'out/sst/regional'
    :param detail_regions: 需要详细分析的区域列表，默认 ['nino34', 'gulf_stream', 'kuroshio', 'acc']
    :param show_plots: 是否显示图表，默认 True
    :return: (analyzer, stats, summary_df) 分析器、统计结果、汇总表格
    
    使用示例:
        from src.analysis.regional import run_regional_analysis
        
        analyzer, stats, df = run_regional_analysis(
            predictor=eval_trainer,
            area=area,
            resolution=1,
            test_offset=520,
            save_dir='out/sst/regional'
        )
    """
    import os
    import pandas as pd
    
    # 默认参数
    if save_dir is None:
        save_dir = 'out/sst/regional'
    if detail_regions is None:
        detail_regions = ['nino34', 'gulf_stream', 'kuroshio', 'acc', 'warm_pool']
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("🌊 海洋区域性分析")
    print("=" * 80)
    print(f"分析区域: {area.title}")
    print(f"空间分辨率: {resolution}°")
    print(f"测试时间点: {test_offset}")
    print(f"保存目录: {save_dir}")
    print("=" * 80 + "\n")
    
    # ========== 1. 单时间点详细分析 ==========
    print(f"📊 正在对时间点 {test_offset} 进行详细分析...")
    
    _, true_output, pred_output, global_rmse, global_r2, ssta = predictor.predict(test_offset, plot=False)
    
    print(f"\n🌍 全球预测性能: RMSE={global_rmse:.4f}°C, R²={global_r2:.4f}")
    
    # 创建区域分析器
    analyzer = RegionalAnalysis(
        pred_output=pred_output,
        true_output=true_output,
        lon=area.lon,
        lat=area.lat,
        resolution=resolution,
        ssta=ssta
    )
    
    # 打印详细的区域分析报告
    stats = analyzer.print_regional_report()
    
    # ========== 2. 绘制区域对比图 ==========
    print("\n📈 正在绘制区域对比图...")
    fig_comparison = analyzer.plot_regional_comparison(
        save_path=f'{save_dir}/regional_comparison.png',
        figsize=(16, 12)
    )
    if not show_plots:
        plt.close(fig_comparison)
    
    # ========== 3. 单个区域详细分析 ==========
    print(f"\n🔍 正在生成各区域详细分析图...")
    for region_key in detail_regions:
        if region_key in OCEAN_REGIONS:
            try:
                fig_detail = analyzer.plot_region_detail(
                    region_key, 
                    save_path=f'{save_dir}/region_{region_key}_detail.png'
                )
                if not show_plots:
                    plt.close(fig_detail)
            except Exception as e:
                print(f"⚠️  区域 {region_key} 分析失败: {e}")
    
    # ========== 4. 生成汇总表格 ==========
    print("\n📋 正在生成汇总表格...")
    summary_data = []
    
    for region_key, region_stats in stats.items():
        if region_stats['valid']:
            summary_data.append({
                '区域': region_stats['region_cn'],
                '英文名': region_stats['region'],
                'RMSE (°C)': region_stats['rmse'],
                'MAE (°C)': region_stats['mae'],
                'R²': region_stats['r2'],
                '系统偏差 (°C)': region_stats['bias'],
                '真实温度均值 (°C)': region_stats['true_mean'],
                '空间变异性 (°C)': region_stats['spatial_variability'],
            })
    
    df = pd.DataFrame(summary_data)
    df_sorted = df.sort_values(by='RMSE (°C)')
    
    # 保存 CSV
    csv_path = f'{save_dir}/regional_analysis_summary.csv'
    df_sorted.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 汇总表格已保存: {csv_path}")
    
    # 打印汇总表格
    print("\n📊 各海域预测性能汇总（按 RMSE 排序）:")
    print("-" * 100)
    
    # 格式化输出
    df_display = df_sorted.copy()
    df_display['RMSE (°C)'] = df_display['RMSE (°C)'].apply(lambda x: f'{x:.4f}')
    df_display['MAE (°C)'] = df_display['MAE (°C)'].apply(lambda x: f'{x:.4f}')
    df_display['R²'] = df_display['R²'].apply(lambda x: f'{x:.4f}')
    df_display['系统偏差 (°C)'] = df_display['系统偏差 (°C)'].apply(lambda x: f'{x:+.4f}')
    df_display['真实温度均值 (°C)'] = df_display['真实温度均值 (°C)'].apply(lambda x: f'{x:.2f}')
    df_display['空间变异性 (°C)'] = df_display['空间变异性 (°C)'].apply(lambda x: f'{x:.3f}')
    
    print(df_display.to_string(index=False))
    print("-" * 100)
    
    # ========== 5. 分析结论 ==========
    print("\n" + "=" * 80)
    print("📝 区域分析结论")
    print("=" * 80)
    
    # 找出表现最好和最差的区域
    best_region = df_sorted.iloc[0]['区域']
    worst_region = df_sorted.iloc[-1]['区域']
    best_rmse = df_sorted.iloc[0]['RMSE (°C)']
    worst_rmse = df_sorted.iloc[-1]['RMSE (°C)']
    
    print(f"\n✅ 预测效果最好的区域: {best_region}, RMSE = {best_rmse:.4f}°C")
    print(f"⚠️  预测效果最差的区域: {worst_region}, RMSE = {worst_rmse:.4f}°C")
    
    print("""
【区域预测性能差异分析】

1. 赤道太平洋区域（NINO 3.4、NINO 3、暖池）:
   - 这些区域是厄尔尼诺/拉尼娜现象的核心区域
   - 温度变化幅度大，时空变异性强
   - 模型对这些区域的注意力较高，预测精度取决于对ENSO信号的捕捉能力

2. 西边界流区域（墨西哥湾暖流、黑潮）:
   - 强烈的海洋锋面和温度梯度
   - 局部涡旋和中尺度过程活跃
   - 由于复杂的动力学过程，预测难度较大

3. 南大洋西风漂流区域:
   - 温度变化相对平缓，季节性变化明显
   - 主要受西风带影响，变化规律性较强
   - 预测误差通常较低

4. 北印度洋区域:
   - 受季风影响显著，季节性变化明显
   - 上升流和混合层深度变化大
   - 预测需要考虑季风周期
""")
    
    print("=" * 80)
    print(f"✅ 区域分析完成！所有结果已保存至: {save_dir}/")
    print("=" * 80 + "\n")
    
    return analyzer, stats, df_sorted


# 保留旧函数名以兼容
def quick_regional_analysis(pred_output: np.ndarray, true_output: np.ndarray,
                           lon: list = [-180, 180], lat: list = [-80, 80],
                           resolution: float = 1, save_dir: str = None):
    """
    快速区域分析 - 一键生成所有区域的分析报告和图表（简化版本）
    
    :param pred_output: 预测结果
    :param true_output: 真实值
    :param lon: 经度范围
    :param lat: 纬度范围
    :param resolution: 空间分辨率
    :param save_dir: 保存目录
    """
    import os
    
    analyzer = RegionalAnalysis(pred_output, true_output, lon, lat, resolution)
    
    # 打印文本报告
    stats = analyzer.print_regional_report()
    
    # 生成图表
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        analyzer.plot_regional_comparison(save_path=f'{save_dir}/regional_comparison.png')
        
        # 为每个区域生成详细分析图
        for region_key in OCEAN_REGIONS.keys():
            try:
                analyzer.plot_region_detail(region_key, 
                                           save_path=f'{save_dir}/region_{region_key}.png')
            except Exception as e:
                print(f"⚠️  区域 {region_key} 分析失败: {e}")
    else:
        analyzer.plot_regional_comparison()
    
    return analyzer, stats

