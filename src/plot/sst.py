# 画海表温度图的函数

from numpy import meshgrid, nanmin, nanmax, floor, ceil, arange, array, linspace
from cmocean import cm

from matplotlib import cm as cm_plt
from matplotlib import pyplot as plt

from cartopy.mpl import ticker as tk
from cartopy import crs as ccrs

from config import area

from src.utils.log import Log
from src.plot.base import create_ax, create_axes, create_carto_ax

COLOR_MAP_PROFILE = cm.thermal
COLOR_MAP_SST = cm_plt.jet

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
    width = lon[1] - lon[0]
    height = lat[1] - lat[0]
    
    if ( width < 20):    
        lon_ticks = arange(lon[0], lon[1] + 1, 5)
    elif ( width < 100):
        lon_ticks = arange(lon[0], lon[1] + 1, 10)
    elif ( width < 200):
        lon_ticks = arange(lon[0], lon[1] + 1, 20)
    else:
        lon_ticks = arange(lon[0], lon[1] + 1, 40)
    
    if (height < 20):
        lat_ticks = arange(lat[0], lat[1] + 1, 5)
    elif ( height < 100):
        lat_ticks = arange(lat[0], lat[1] + 1, 10)
    elif ( height < 200):
        lat_ticks = arange(lat[0], lat[1] + 1, 20)
    else:
        lat_ticks = arange(lat[0], lat[1] + 1, 40)
    
    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)
    ax.xaxis.set_major_formatter(tk.LongitudeFormatter())
    ax.yaxis.set_major_formatter(tk.LatitudeFormatter())

def plot_sst(sst, lon, lat, step=1, filename='sst.png', title=''):
    """
    绘制海表温度分布图
    
    :param sst: 海表温度数据,二维数组
    :param lon: 经度范围 [起始经度, 结束经度]
    :param lat: 纬度范围 [起始纬度, 结束纬度]
    :return: 返回图像对象和子图对象
    """
    from src.config.params import PREDICT_SAVE_PATH
    
    ax = create_carto_ax()
    
    projection = ccrs.PlateCarree()
    
    ax.set_extent([*lon, *lat], crs=projection)
    
    ax.figure.set_size_inches(10, 4)
    
    set_ticker(ax, lon, lat)
    
    # 生成网格点
    lon_grid, lat_grid = meshgrid(_range(lon, step), _range(lat, step))
    
    # vmin = max(floor(nanmin(sst)), 0)
    # vmax = min(ceil(nanmax(sst)), 30)
    
    levels = arange(0, 30, 1)
    
    im = ax.contourf(
        lon_grid, lat_grid, sst, 
        levels=levels,
        extend='both',
        cmap=COLOR_MAP_SST,
        transform=projection)
    
    ax.figure.colorbar(im, 
                ax=ax,
                orientation='vertical',
                label='temperature (°C)')
    
    ax.text(0.97, 0.97, '(e)', transform=ax.transAxes, fontsize=16,
                ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 去掉网格
    ax.grid(False)
    
    plt.title(title)
    
    plt.savefig(f'{PREDICT_SAVE_PATH}/{filename}')
    
    return ax

def plot_sst_diff(sst_diff, lon, lat, step=1, filename='sst_diff.png', title=''):
    """
    绘制海表温度分布图
    
    :param sst: 海表温度数据,二维数组
    :param lon: 经度范围 [起始经度, 结束经度]
    :param lat: 纬度范围 [起始纬度, 结束纬度]
    :return: 返回图像对象和子图对象
    """
    from src.config.params import ERROR_SAVE_PATH
    
    ax = create_carto_ax()
    
    projection = ccrs.PlateCarree()
    
    ax.set_extent([*lon, *lat], crs=projection)
    
    ax.figure.set_size_inches(10, 4)
    
    set_ticker(ax, lon, lat)
    
    # 生成网格点
    lon_grid, lat_grid = meshgrid(_range(lon, step), _range(lat, step))
    
    # vmin = max(floor(nanmin(sst_diff)), -3)
    # vmax = min(ceil(nanmax(sst_diff)), )
    
    levels = arange(-1.5, 1.5, 0.1)
    
    im = ax.contourf(
        lon_grid, lat_grid, sst_diff, 
        levels=levels,
        cmap=COLOR_MAP_SST,
        extend='both',
        transform=projection)
    
    ax.figure.colorbar(im, 
                ax=ax,
                orientation='vertical',
                label='temperature (°C)')
    
    ax.text(0.97, 0.97, '(f)', transform=ax.transAxes, fontsize=16,
                ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.grid(False)
    
    plt.title(title)
    plt.savefig(f'{ERROR_SAVE_PATH}/{filename}')
    
    return ax

def plot_sst_l(sst, lon, lat, step=1):
    """
    使用 cartopy 投影地图绘制海表温度图，标注等高线以及数值
    """
    ax = create_carto_ax()
    
    projection = ccrs.PlateCarree()
    
    ax.set_extent([*lon, *lat], crs=projection)
    
    set_ticker(ax, lon, lat)
    
    # 生成网格点
    lon_grid, lat_grid = meshgrid( _range(lon, step), _range(lat, step))
    contour = ax.contourf(lon_grid, lat_grid, sst, cmap=COLOR_MAP_PROFILE, transform=projection, levels=30)
    
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
    
    # 去掉网格
    ax.grid(False)
    
    plt.colorbar(contour, ax=ax,
                orientation='vertical',
                fraction=0.05,
                label='temperature (°C)')
    
    return ax
    

def plot_sst_comparison(sst1, sst2, lon, lat, title='', step=1):
    """
    绘制两个海表温度分布图的对比图
    
    :param sst1: 第一个海表温度数据,二维数组
    :param sst2: 第二个海表温度数据,二维数组 
    :param lon: 经度范围 [起始经度, 结束经度]
    :param lat: 纬度范围 [起始纬度, 结束纬度]
    :return: 返回包含两个子图的列表
    """
    
    axes = create_axes(2, 1, 'all')
    
    lat = array(lat)
    lon = array(lon)
    
    # 生成网格点
    lon_grid, lat_grid = meshgrid(_range(lon, step), _range(lat, step))
    
    # 计算共同的色标范围
    vmin = min(nanmin(sst1), nanmin(sst2))
    vmax = max(nanmax(sst1), nanmax(sst2))
    
    vmin = max(floor(vmin), 0)
    vmax = min(ceil(vmax), 30)
    
    levels = arange(vmin, vmax, 1)
    
    # 绘制第一个海表温度分布图
    _ = axes[0].contourf(lon_grid, lat_grid, sst1,
                         extend='both',
                         cmap=COLOR_MAP_SST,
                         levels=levels)
    

    # 绘制第二个海表温度分布图
    _ = axes[1].contourf(lon_grid, lat_grid, sst2,
                        extend='both',
                        cmap=COLOR_MAP_SST, 
                        levels=levels)

    # 设置两个子图的刻度
    for ax in axes:
        set_ticker(ax, lon, lat)
        ax.figure.set_size_inches(10, 8)
    
    plt.title(title)

    # 添加共享色标
    plt.colorbar(_,
                ax=axes,
                orientation='horizontal',
                pad=0.1, 
                fraction=0.05,
                label='temperature (°C)')
    
    return axes

def plot_sst_month(sst, ax, levels, label, lon, lat):
        
    ax.set_xticks(_range([0, lon[1] - lon[0]], 5))
    ax.set_yticks(_range([0, lat[1] - lat[0]], 5))
    ax.tick_params(axis='both', which='major', labelsize=8)  # 设置刻度标签大小
    # 计算刻度标签，转换为经纬度
    x_labels = _range(lon, 5)
    y_labels = _range(lat, 5)
    
    # 转换坐标
    x_labels = [f"{x:.0f}°E" if x >= 0 else f"{abs(x):.0f}°W" for x in x_labels]
    y_labels = [f"{y:.0f}°N" if y >= 0 else f"{abs(y):.0f}°S" for y in y_labels]
    
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    
    # 修改文本位置到左上角，使用相对坐标
    ax.text(0.02, 0.12, label, 
            fontsize=8, 
            color='orange',
            transform=ax.transAxes,  # 使用相对坐标系统
            verticalalignment='bottom'
    )

    _ = ax.contourf(sst, levels=levels, cmap=COLOR_MAP_SST)
    ax.contour(sst, colors='black', alpha=0.5, linewidths=0.2, linestyles='--', levels=30)
    contour_lines = ax.contour(sst, colors='black', linewidths=0.5)
    
    ax.clabel(contour_lines, inline=True, colors='black', fontsize=5, fmt='%d', manual=False)
    
    return _

def plot_sst_seq(sst_seq, lon, lat):
    """
    绘制海表温度序列图
    
    :param sst_seq: 海表温度序列,二维数组
    :param lon: 经度范围 [起始经度, 结束经度]
    :param lat: 纬度范围 [起始纬度, 结束纬度]
    """
    length = sst_seq.shape[0]
    
    cols = 6
    rows = int(ceil(length / cols))
    
    axs = create_axes(rows, cols, 'all')

    levels = linspace(nanmin(sst_seq), nanmax(sst_seq), 15)

    for i in range(length):
        ax = axs[i // 6, i % 6]
        
        _ = plot_sst_month(sst_seq[i], ax, levels, f'{i} month', lon, lat)
        
def plot_sst_rmses(rmses_sst, areas: list[area.Area], labels: list[str]):
    """
    绘制海表温度序列的均方根误差柱状图
    
    :param rmses_sst: 海表温度序列的均方根误差
    """
    
    ax = create_ax()
    
    # x轴为时间，第 n 天
    x = arange(len(areas))
    width = 0.2
    multiplier = 0
     
    for month, rmse in rmses_sst.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, rmse, width, label=month)
        ax.bar_label(rects, padding=3)
        multiplier += 1
    
    # 设置 y 轴标签
    ax.set_ylabel('rmse (°C)')
    ax.set_title('RMSE of SST')
    # 设置 x 轴标签
    ax.set_xticks(x + width, [area.title for area in areas])
    ax.legend(labels=labels, loc='upper left', ncols=2)
    ax.set_ylim(0, 5)
    
    return ax
        
def plot_sst_rmses_seq(sst_rmses):
    """
    绘制海表温度序列的均方根误差柱状图
    
    :param sst_rmses: 海表温度序列的均方根误差
    """
    
    ax = create_ax()
    
    # x 轴为时间
    x = arange(len(sst_rmses))
    
    # y 轴为均方根误差
    y = sst_rmses
    
    ax.plot(x, y)
    
    # 设置 x 轴标签
    ax.set_xlabel('time (day)')
    # 设置 y 轴标签
    ax.set_ylabel('rmse (°C)')
    
    return ax

