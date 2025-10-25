# 绘制混合层相关的函数

from numpy import meshgrid, arange
from cmocean import cm
from matplotlib import pyplot as plt

from src.plot.base import create_ax, create_axes

def plot_mld(mld, lon, lat):
    """
    绘制混合层深度
    
    :param mld: 混合层深度
    :param lon: 经度
    :param lat: 纬度
    """
    
    ax = create_ax()
    
    lon_indices = arange(lon[0], lon[1], 1)
    lat_indices = arange(lat[0], lat[1], 1)
    
    X, Y = meshgrid(lon_indices, lat_indices)
    
    contour = ax.contourf(X, Y, mld, cmap=cm.haline)
    
    plt.colorbar(contour, ax=ax,
                orientation='horizontal',
                pad=0.1,
                fraction=0.05,
                label='mld (m)')


def plot_mld_rmse(mld_rmse, lon, lat, title):
    """
    绘制混合层深度误差
    
    :param mld_rmse: 混合层深度误差
    :param lon: 经度
    :param lat: 纬度
    :param title: 标题
    """
    
    ax = create_axes(3, 1, 'all')
    plt.title(title)
    
    lon_indices = arange(lon[0], lon[1], 1)
    lat_indices = arange(lat[0], lat[1], 1)
    
    X, Y = meshgrid(lon_indices, lat_indices)
    
    for i in range(3):
        rmse = mld_rmse[i]
        contour = ax[i].contourf(X, Y, rmse, cmap=cm.haline)
        ax[i].figure.set_size_inches(10, 15)
    
    plt.colorbar(contour, ax=ax,
                orientation='horizontal',
                pad=0.05,
                fraction=0.05,
                label='mld rmse (°C)')
    
    

