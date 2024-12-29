# 画海表温度图的函数

from numpy import arange, meshgrid, nanmin, nanmax

from matplotlib import ticker as tk
from matplotlib import pyplot as plt

from src.plot.base import create_ax, create_shared_axes

COLOR_MAP = 'thermal'

def _range(range, step=1):
    """
    根据给定范围生成一个列表
    """
    
    return arange(range[0], range[1], step)

def set_ticker(ax, lon, lat):
    """
    设置经纬度刻度
    
    :param ax: 子图对象
    :param lon: 经度范围 [起始经度, 结束经度]
    :param lat: 纬度范围 [起始纬度, 结束纬度]
    """
    
    ax.set_xticks(_range(lon))
    ax.set_yticks(_range(lat))
    ax.xaxis.set_major_formatter(tk.LongitudeFormatter())
    ax.yaxis.set_major_formatter(tk.LatitudeFormatter())

def plot_sst(sst, lon, lat):
    """
    绘制海表温度分布图
    
    :param sst: 海表温度数据,二维数组
    :param lon: 经度范围 [起始经度, 结束经度]
    :param lat: 纬度范围 [起始纬度, 结束纬度]
    :return: 返回图像对象和子图对象
    """
    ax = create_ax()
    
    set_ticker(ax, lon, lat)
    
    # 生成网格点
    lon_grid, lat_grid = meshgrid(_range(lon), _range(lat))
    contour = ax.contourf(lon_grid, lat_grid, sst, cmap=COLOR_MAP)
    
    plt.colorbar(contour, ax=ax,
                orientation='horizontal',
                pad=0.05,
                fraction=0.05,
                label='temperature (°C)')
    
    return ax

def plot_sst_l(sst, lon, lat):
    """
    绘制海表温度图，标注等高线以及数值
    """
    ax = create_ax()
    
    set_ticker(ax, lon, lat)
    
    # 生成网格点
    lon_grid, lat_grid = meshgrid(_range(lon), _range(lat))
    contour = ax.contourf(lon_grid, lat_grid, sst, cmap=COLOR_MAP)
    
    # 添加等高线
    contour_lines = ax.contour(lon_grid, lat_grid, sst, colors='black', linewidths=0.5)
    
    # 在等高线上标注数值
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
    
    plt.colorbar(contour, ax=ax,
                orientation='horizontal',
                pad=0.05,
                fraction=0.05,
                label='temperature (°C)')
    
    return ax
    

def plot_sst_comparison(sst1, sst2, lon, lat):
    """
    绘制两个海表温度分布图的对比图
    
    :param sst1: 第一个海表温度数据,二维数组
    :param sst2: 第二个海表温度数据,二维数组 
    :param lon: 经度范围 [起始经度, 结束经度]
    :param lat: 纬度范围 [起始纬度, 结束纬度]
    :return: 返回包含两个子图的列表
    """
    axes = create_shared_axes(1, 2)
    
    # 设置两个子图的刻度
    for ax in axes:
        set_ticker(ax, lon, lat)
    
    # 生成网格点
    lon_grid, lat_grid = meshgrid(_range(lon), _range(lat))
    
    # 计算共同的色标范围
    vmin = min(nanmin(sst1), nanmin(sst2))
    vmax = max(nanmax(sst1), nanmax(sst2))
    
    # 绘制第一个海表温度分布图
    _ = axes[0].contourf(lon_grid, lat_grid, sst1, 
                               cmap=COLOR_MAP,
                               vmin=vmin, vmax=vmax)
    
    # 绘制第二个海表温度分布图
    _ = axes[1].contourf(lon_grid, lat_grid, sst2,
                               cmap=COLOR_MAP, 
                               vmin=vmin, vmax=vmax)
    
    # 添加共享色标
    plt.colorbar(_,
                ax=axes,
                orientation='horizontal',
                pad=0.05, 
                fraction=0.05,
                label='temperature (°C)')
    
    return axes
    