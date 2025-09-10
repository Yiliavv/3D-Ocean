# 绘制三维海洋图
from numpy import linspace, meshgrid, transpose, nanmin, nanmax, arange

from matplotlib import pyplot as plt

from src.utils.log import Log
from src.plot.base import create_3d_ax, create_shared_3d_axes, create_ax
from src.plot.sst import _range

def plot_3d_temperature(temp, lon, lat, depth, step=1):
    """
    绘制三维海温分布图
    
    :param temp: 三维温度数据,形状为 (lon, lat, depth)
    :param lon: 经度范围 [起始经度, 结束经度]
    :param lat: 纬度范围 [起始纬度, 结束纬度] 
    :param depth: 深度列表
    :return: 返回3D图像对象
    """
    ax = create_3d_ax()
    
    # 生成网格点
    lon_grid, lat_grid = meshgrid(_range(lat, step), _range(lon, step))
    
    # 计算色标范围
    vmin = nanmin(temp)
    vmax = nanmax(temp)
    
    # 绘制每一层的等温面
    for i, d in enumerate(depth):        
        temp_layer = transpose(temp[:, :, i], (1, 0))
        
        _ = ax.contourf(lon_grid, lat_grid, temp_layer,
                       zdir='z', offset=-d,
                       vmin=vmin, vmax=vmax)
    
    # 设置坐标轴范围
    # ax.set_xlim(lon)
    # ax.set_ylim(lat)
    ax.set_zlim(-max(depth), -min(depth))
    
    # 添加色标
    plt.colorbar(_, ax=ax,
                orientation='vertical',
                pad=0.1,
                fraction=0.02,
                label='temperature [°C]')
    
    return ax

def plot_3d_temperature_comparison(temp1, temp2, lon, lat, depth, step=1):
    """
    绘制两个三维海温分布图的对比图

    :param temp1: 第一个三维温度数据,形状为 (lon, lat, depth)
    :param temp2: 第二个三维温度数据,形状为 (lon, lat, depth) 
    :param lon: 经度范围 [起始经度, 结束经度]
    :param lat: 纬度范围 [起始纬度, 结束纬度]
    :param depth: 深度列表
    :return: 返回包含两个3D子图的列表
    """
    # 创建两个共享坐标轴的3D子图
    axes = create_shared_3d_axes(1, 2, shared='all')

    # 生成网格点
    x_grid, y_grid, z_grid = meshgrid(_range(lat, step), _range(lon, step), -depth)

    # 计算两个数据集的共同色标范围
    vmin = min(nanmin(temp1), nanmin(temp2))
    vmax = max(nanmax(temp1), nanmax(temp2))
    
    Log.d(f"min: {vmin}, max: {vmax}")
    
    kw = {
            'vmin': vmin,
            'vmax': vmax,
            'levels': linspace(vmin, vmax, 20),
        }
    
    print(x_grid.shape, y_grid.shape, z_grid.shape)

    # 在两个子图上分别绘制等温面
    for ax, temp in zip(axes, [temp1, temp2]):
        for d_index in range(len(depth)):
            # 绘制每一层的等温面
            _ = ax.contourf(x_grid[:, :, d_index], y_grid[:, :, d_index], temp[:, :, d_index],
                           zdir='z', offset=-depth[d_index],
                           **kw)
        
            # 设置坐标轴范围
            ax.set_xlim(x_grid.min(), x_grid.max())
            ax.set_ylim(y_grid.min(), y_grid.max())
            ax.set_zlim(z_grid.min(), z_grid.max())

    # 添加色标
    plt.colorbar(_, ax=axes,
                orientation='vertical',
                pad=0.1, 
                fraction=0.02,
                label='temperature [°C]')

    return axes

def plot_profile_rmses(rmses_profile, dates):
    """
    绘制剖面分布图的均方根误差
    
    :param rmses_profile: 三维海温分布图的均方根误差
    """
    for area, rmses in rmses_profile.items():
        ax = create_ax()
        print(f"{area}: {rmses}")
        
        for date, rmse in zip(dates, rmses):
            ax.plot(rmse, label=date)
        
        ax.legend()
        ax.set_xlabel('depth (F)')
        ax.set_ylabel('rmse (°C)')
        
        plt.title(area)