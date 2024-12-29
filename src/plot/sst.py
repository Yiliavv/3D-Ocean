# 画海表温度图的函数

from numpy import arange, meshgrid, linspace, nanmin, nanmax, floor, ceil
from cmocean import cm

from matplotlib import pyplot as plt
from cartopy.mpl import ticker as tk
from cartopy import crs as ccrs

from src.utils.log import Log
from src.plot.base import create_ax, create_shared_axes, create_carto_ax

COLOR_MAP = cm.thermal

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
    # 计算合适的经纬度刻度
    lon_ticks = linspace(lon[0], lon[1], 10)
    lat_ticks = linspace(lat[0], lat[1], 9)
    
    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)
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
    lon_grid, lat_grid = meshgrid(_range(lat), _range(lon))
    contour = ax.contourf(lon_grid, lat_grid, sst, cmap=COLOR_MAP)
    
    plt.colorbar(contour, ax=ax,
                orientation='horizontal',
                pad=0.05,
                fraction=0.05,
                label='temperature (°C)')
    
    return ax

def plot_sst_l(sst, lon, lat):
    """
    使用 cartopy 投影地图绘制海表温度图，标注等高线以及数值
    """
    ax = create_carto_ax()
    
    projection = ccrs.PlateCarree(central_longitude=-180)
    
    ax.set_extent([*lon, *lat], crs=projection)
    
    set_ticker(ax, lon, lat)
    
    sst = sst.T # 转置
    
    # 生成网格点
    lon_grid, lat_grid = meshgrid( _range(lon), _range(lat))
    contour = ax.contourf(lon_grid, lat_grid, sst, cmap=COLOR_MAP, transform=projection, levels=30)
    
    # 添加等高线, 每 1 度一个浅色等高线，每 5 度一个深色等高线
    # 绘制等高线
    ax.contour(lon_grid, lat_grid, sst, 
                colors='black', alpha=0.2, linewidths=0.2,
                levels=arange(floor(nanmin(sst)), ceil(nanmax(sst)), 1),
                transform=projection)
    
    # 绘制主要等高线(每5度)
    contour_lines_major = ax.contour(lon_grid, lat_grid, sst,
                                    colors='black', alpha=0.9, linewidths=0.5,
                                    transform=projection)
    
    
    # 在深色等高线上标注数值
    ax.clabel(contour_lines_major, inline=True, fontsize=5, fmt='%d')
    
    plt.colorbar(contour, ax=ax,
                orientation='horizontal',
                pad=0.1,
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
    lon_grid, lat_grid = meshgrid(_range(lat), _range(lon))
    
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
                pad=0.1, 
                fraction=0.05,
                label='temperature (°C)')
    
    return axes
    