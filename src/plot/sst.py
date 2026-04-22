# 画海表温度图的函数

import numpy as np
from cmocean import cm

from matplotlib import cm as cm_plt
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

from cartopy.mpl import ticker as tk
from cartopy import crs as ccrs

from config import area

from src.utils.log import Log
from src.plot.base import create_axes, create_carto_ax, create_carto_axes, apply_plot_style
from src.config.params import PREDICT_SAVE_PATH

apply_plot_style()

COLOR_MAP_PROFILE = cm.thermal
COLOR_MAP_SST = cm_plt.jet
COLOR_MAP_ATTENTION = cm_plt.viridis

# 创建自定义的误差色标：蓝色（负误差）-> 浅白色（零误差）-> 红色（正误差）
# 使用柔和的颜色，避免过于鲜艳
def create_error_colormap():
    """
    创建用于误差可视化的自定义色标
    - 负误差：深蓝 -> 浅蓝
    - 零误差：浅白色
    - 正误差：浅红 -> 深红
    颜色配置和谐，不会过于鲜艳
    """
    colors = [
        '#2166ac',  # 深蓝（大负误差）
        '#4393c3',  # 中蓝
        '#92c5de',  # 浅蓝
        '#d1e5f0',  # 极浅蓝
        '#f7f7f7',  # 浅白色（零误差）
        '#fddbc7',  # 极浅红
        '#f4a582',  # 浅红
        '#d6604d',  # 中红
        '#b2182b',  # 深红（大正误差）
    ]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('error_cmap', colors, N=n_bins)
    return cmap

COLOR_MAP_ERROR = create_error_colormap()

def _range(range, step=1):
    """
    根据给定范围生成一个列表
    """
    
    return np.arange(range[0], range[1], step)

def set_ticker(ax, lon, lat):
    """
    设置经纬度刻度
    
    :param ax: 子图对象
    :param lon: 经度范围 [起始经度, 结束经度]
    :param lat: 纬度范围 [起始纬度, 结束纬度]
    """

    if lon[0] > lon[1]:
        segment_f = [lon[0], 180]
        segment_b = [-180, lon[1]]
        lon_ticks = np.concatenate([_range(segment_f, 10), _range(segment_b, 10)])
    
    else:
        width = lon[1] - lon[0]
        
        if ( width < 20):    
            lon_ticks = np.arange(lon[0], lon[1] + 1, 5)
        elif ( width < 100):
            lon_ticks = np.arange(lon[0], lon[1] + 1, 10)
        elif ( width < 200):
            lon_ticks = np.arange(lon[0], lon[1] + 1, 20)
        else:
            lon_ticks = np.arange(lon[0], lon[1] + 1, 40)

    if lat[0] > lat[1]:
        segment_f = [lat[0], 90]
        segment_b = [-90, lat[1]]
        lat_ticks = np.concatenate([_range(segment_f, 10), _range(segment_b, 10)])

    else:
        height = lat[1] - lat[0]
    
        if (height < 20):
            lat_ticks = np.arange(lat[0], lat[1] + 1, 5)
        elif ( height < 100):
            lat_ticks = np.arange(lat[0], lat[1] + 1, 10)
        elif ( height < 200):
            lat_ticks = np.arange(lat[0], lat[1] + 1, 20)
        else:
            lat_ticks = np.arange(lat[0], lat[1] + 1, 40)
    
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
    lon_grid, lat_grid = np.meshgrid(_range(lon, step), _range(lat, step))
    
    # vmin = max(floor(nanmin(sst)), 0)
    # vmax = min(ceil(nanmax(sst)), 30)
    
    levels = np.arange(0, 30, 1)
    
    im = ax.contourf(
        lon_grid, lat_grid, sst, 
        levels=levels,
        extend='both',
        cmap=COLOR_MAP_SST,
        transform=projection)
    
    cbar = ax.figure.colorbar(im, 
                ax=ax,
                orientation='vertical',
                label='temperature (°C)')
    
    # 设置坐标轴刻度标签字体大小
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # 去掉网格
    ax.grid(False)
    
    plt.title('')
    
    plt.savefig(f'{PREDICT_SAVE_PATH}/{filename}')
    
    return ax

def plot_attention(attention, lon, lat, step=1, filename='attention.png', title='Attention', ax=None):
    """
    绘制海表温度分布图
    
    :param sst: 海表温度数据,二维数组
    :param lon: 经度范围 [起始经度, 结束经度]
    :param lat: 纬度范围 [起始纬度, 结束纬度]
    :return: 返回图像对象和子图对象
    """

    print(f'attention shape: {attention.shape}, max: {np.max(attention)}, min: {np.min(attention)}')

    ax = ax or create_carto_ax()
    
    projection = ccrs.PlateCarree()
    
    ax.set_extent([*lon, *lat], crs=projection)
    
    set_ticker(ax, lon, lat)
    
    # 生成网格点
    lon_grid, lat_grid = np.meshgrid(_range(lon, step), _range(lat, step))
    
    im = ax.contourf(
        lon_grid, lat_grid, attention, 
        extend='both',
        cmap=COLOR_MAP_ATTENTION,
        transform=projection)
    
    cbar = ax.figure.colorbar(im, 
                ax=ax,
                orientation='vertical',
                shrink=0.6,  # 缩小 colorbar 高度到 60%
                aspect=20,    # 控制宽高比，使 colorbar 更细
                pad=0.05)     # 减小与地图的间距
    cbar.set_label('Attention weight', fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    
    # 设置坐标轴刻度标签字体大小
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # 去掉网格
    ax.grid(False)
    
    plt.title('')
    
    return ax

def plot_sst_diff(sst_diff, lon, lat, step=1, filename='sst_diff.png', title=''):
    """
    绘制海表温度误差分布图
    
    :param sst_diff: 海表温度误差数据,二维数组
    :param lon: 经度范围 [起始经度, 结束经度]
    :param lat: 纬度范围 [起始纬度, 结束纬度]
    :param step: 网格步长
    :param filename: 保存文件名
    :param title: 图表标题
    :return: 返回图像对象和子图对象
    """
    from src.config.params import ERROR_SAVE_PATH
    
    ax = create_carto_ax()
    
    projection = ccrs.PlateCarree()
    
    ax.set_extent([*lon, *lat], crs=projection)
    
    ax.figure.set_size_inches(10, 4)
    
    set_ticker(ax, lon, lat)
    
    # 生成网格点
    lon_grid, lat_grid = np.meshgrid(_range(lon, step), _range(lat, step))
    
    # 计算误差的范围，确保色标以0为中心
    abs_max = max(abs(np.nanmin(sst_diff)), abs(np.nanmax(sst_diff)))
    abs_max = min(abs_max, 1.5)  # 限制最大范围为±1.5°C
    
    levels = np.linspace(-abs_max, abs_max, 30)
    
    im = ax.contourf(
        lon_grid, lat_grid, sst_diff, 
        levels=levels,
        cmap=COLOR_MAP_ERROR,
        extend='both',
        transform=projection)
    
    ax.figure.colorbar(im, 
                ax=ax,
                orientation='vertical',
                label='temperature error (°C)')
    
    # 设置坐标轴刻度标签字体大小
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    ax.grid(False)
    
    plt.title('')
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
    lon_grid, lat_grid = np.meshgrid( _range(lon, step), _range(lat, step))
    contour = ax.contourf(lon_grid, lat_grid, sst, cmap=COLOR_MAP_PROFILE, transform=projection, levels=30)
    
    # 添加等高线, 每 1 度一个浅色等高线，每 5 度一个深色等高线
    # 绘制等高线
    ax.contour(lon_grid, lat_grid, sst, 
                colors='black', alpha=0.2, linewidths=0.2,
                levels=np.arange(np.floor(np.nanmin(sst)), np.ceil(np.nanmax(sst)), 1),
                transform=projection)
    
    # 绘制主要等高线(每5度)
    contour_lines_major = ax.contour(lon_grid, lat_grid, sst,
                                    colors='black', alpha=0.9, linewidths=0.5,
                                    transform=projection)
    
    
    # 在深色等高线上标注数值
    ax.clabel(contour_lines_major, inline=True, fontsize=16, fmt='%d')
    
    # 去掉网格
    ax.grid(False)
    
    plt.colorbar(contour, ax=ax,
                orientation='vertical',
                fraction=0.05,
                label='temperature (°C)')
    
    return ax

def plot_nino(ssta, step=1):
    '''
    绘制 NINO 指数图
    
    NINO3.4 区域: 5°S-5°N, 170°W-120°W
    NINO3 区域: 5°S-5°N, 150°W-90°W

    :param ssta: 海表温度异常,二维数组 [纬度, 经度]
                 假设经纬度范围为 [-180, 180], [-80, 80]
                 shape: [160/step, 360/step] 对于1°分辨率
    :param step: 空间分辨率（度）
    '''
    
    # 计算数据的形状和坐标
    lat_size, lon_size = ssta.shape
    
    # 生成完整的经纬度数组
    lon_full = np.linspace(-180, 180, lon_size, endpoint=False)
    lat_full = np.linspace(-80, 80, lat_size)
    
    # 定义绘图区域和 NINO 区域
    plot_lat_range = [-20, 20]
    plot_lon_range = [-180, -80]
    
    # NINO3.4: 5°S-5°N, 170°W-120°W
    nino34_lon_range = [-170, -120]
    nino34_lat_range = [-5, 5]
    
    # NINO3: 5°S-5°N, 150°W-90°W  
    nino3_lon_range = [-150, -90]
    nino3_lat_range = [-5, 5]
    
    # 提取绘图区域的数据索引
    lat_mask = (lat_full >= plot_lat_range[0]) & (lat_full <= plot_lat_range[1])
    lon_mask = (lon_full >= plot_lon_range[0]) & (lon_full <= plot_lon_range[1])
    
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    
    # 提取数据
    ssta_plot = ssta[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
    lon_plot = lon_full[lon_idx]
    lat_plot = lat_full[lat_idx]
    
    # 计算 NINO3.4 指数
    nino34_lat_mask = (lat_full >= nino34_lat_range[0]) & (lat_full <= nino34_lat_range[1])
    nino34_lon_mask = (lon_full >= nino34_lon_range[0]) & (lon_full <= nino34_lon_range[1])
    nino34_lat_idx = np.where(nino34_lat_mask)[0]
    nino34_lon_idx = np.where(nino34_lon_mask)[0]
    
    if len(nino34_lat_idx) > 0 and len(nino34_lon_idx) > 0:
        nino34_data = ssta[nino34_lat_idx[0]:nino34_lat_idx[-1]+1, 
                           nino34_lon_idx[0]:nino34_lon_idx[-1]+1]
        nino34_index = np.nanmean(nino34_data)
    else:
        nino34_index = np.nan
    
    # 计算 NINO3 指数
    nino3_lat_mask = (lat_full >= nino3_lat_range[0]) & (lat_full <= nino3_lat_range[1])
    nino3_lon_mask = (lon_full >= nino3_lon_range[0]) & (lon_full <= nino3_lon_range[1])
    nino3_lat_idx = np.where(nino3_lat_mask)[0]
    nino3_lon_idx = np.where(nino3_lon_mask)[0]
    
    if len(nino3_lat_idx) > 0 and len(nino3_lon_idx) > 0:
        nino3_data = ssta[nino3_lat_idx[0]:nino3_lat_idx[-1]+1,
                         nino3_lon_idx[0]:nino3_lon_idx[-1]+1]
        nino3_index = np.nanmean(nino3_data)
    else:
        nino3_index = np.nan
    
    print(f'NINO3.4 指数: {nino34_index:.3f}°C')
    print(f'NINO3 指数: {nino3_index:.3f}°C')
    
    # 绘制 NINO 指数图
    ax = create_carto_ax()
    projection = ccrs.PlateCarree()
    
    # 使用 try-except 避免调试器与 Cartopy Cython 扩展的兼容性问题
    # 这个问题通常出现在使用 PyCharm/PyDev 等调试器时
    try:
        # 将范围展开为列表，避免使用 * 展开操作符
        lon_min, lon_max = plot_lon_range[0], plot_lon_range[1]
        lat_min, lat_max = plot_lat_range[0], plot_lat_range[1]
        extent = [lon_min, lon_max, lat_min, lat_max]
        ax.set_extent(extent, crs=projection)
    except (AssertionError, KeyError, Exception) as e:
        # 如果 set_extent 失败（通常是调试器兼容性问题），跳过设置 extent
        # Cartopy 会自动使用数据范围
        print(f"⚠️  Cartopy set_extent 失败（调试器兼容性问题），跳过: {str(e)[:100]}")
        # 不设置 extent，让 Cartopy 使用默认范围
    
    ax.figure.set_size_inches(10, 4)
    
    # 图像设置黑色边框
    ax.spines['top'].set_color('#444444')
    ax.spines['right'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')
    ax.spines['left'].set_color('#444444')
    
    # 生成网格用于绘图
    lon_grid, lat_grid = np.meshgrid(lon_plot, lat_plot)
    
    # 使用误差色标绘制 SSTA
    abs_max = max(abs(np.nanmin(ssta_plot)), abs(np.nanmax(ssta_plot)))
    levels = np.linspace(-abs_max, abs_max, 30)
    ax.contourf(lon_grid, lat_grid, ssta_plot, 
                cmap=COLOR_MAP_ERROR, transform=projection, 
                levels=levels, extend='both')
    
    # 添加色标
    cbar = ax.figure.colorbar(ax.collections[0], ax=ax, 
                               orientation='horizontal',
                               pad=0.05, fraction=0.05, shrink=0.8)
    cbar.set_label('SSTA (°C)', fontsize=16)
    
    # 绘制矩形边界框
    import matplotlib.patches as mpatches
    
    # 创建 NINO3.4 边界框
    rect34 = mpatches.Rectangle(
        (nino34_lon_range[0], nino34_lat_range[0]), 
        nino34_lon_range[1] - nino34_lon_range[0], 
        nino34_lat_range[1] - nino34_lat_range[0],
        linewidth=2,
        edgecolor='red',
        facecolor='none',
        transform=projection,
        zorder=10
    )

    # 创建 NINO3 边界框
    rect3 = mpatches.Rectangle(
        (nino3_lon_range[0], nino3_lat_range[0]), 
        nino3_lon_range[1] - nino3_lon_range[0], 
        nino3_lat_range[1] - nino3_lat_range[0],
        linewidth=2,
        edgecolor='blue',
        facecolor='none',
        transform=projection,
        zorder=10
    )
    
    ax.add_patch(rect34)
    ax.add_patch(rect3)
    
    # 添加标签和指数值
    # NINO3.4 标签
    nino34_center_lon = (nino34_lon_range[0] + nino34_lon_range[1]) / 2
    nino34_center_lat = (nino34_lat_range[0] + nino34_lat_range[1]) / 2
    ax.text(nino34_center_lon, nino34_center_lat, 
            f'NINO3.4\n{nino34_index:.2f}°C', 
            transform=projection, 
            fontsize=16, 
            fontweight='bold',
            color='red',
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='red'))

    # NINO3 标签
    nino3_center_lon = (nino3_lon_range[0] + nino3_lon_range[1]) / 2
    nino3_center_lat = (nino3_lat_range[0] + nino3_lat_range[1]) / 2
    ax.text(nino3_center_lon, nino3_center_lat, 
            f'NINO3\n{nino3_index:.2f}°C', 
            transform=projection, 
            fontsize=16, 
            fontweight='bold',
            color='blue',
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='blue'))
    
    # Keep figures title-free per visualization convention.
    ax.set_title('')
    
    return ax
    
def plot_sst_month(sst, ax, levels, lon, lat):
        
    set_ticker(ax, lon, lat)

    _ = ax.contourf(sst, levels=levels, cmap=COLOR_MAP_SST)
    ax.contour(sst, colors='black', alpha=0.5, linewidths=0.2, linestyles='--', levels=30)
    contour_lines = ax.contour(sst, colors='black', linewidths=0.5)
    
    ax.clabel(contour_lines, inline=True, colors='black', fontsize=16, fmt='%d', manual=False)
    
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
    rows = int(np.ceil(length / cols))
    
    axs = create_axes(rows, cols, 'all')

    levels = np.linspace(np.nanmin(sst_seq), np.nanmax(sst_seq), 15)

    for i in range(length):
        ax = axs[i // 6, i % 6]
        
        _ = plot_sst_month(sst_seq[i], ax, levels, f'{i} month', lon, lat)

def plot_sequence(sequence, lon, lat, step=1, filename='sequence.png', title='Sequence Visualization', plot_type='attention'):
    """
    绘制序列图
    
    :param sequence: 序列，形状为 [seq_len, width, height]
    :param lon: 经度范围 [起始经度, 结束经度]
    :param lat: 纬度范围 [起始纬度, 结束纬度]
    :param step: 空间分辨率（度）
    :param filename: 保存文件名
    :param title: 图表标题
    :return: 返回图像对象
    """

    seq_len, width, height = sequence.shape

    axs = create_carto_axes(seq_len, 1).flatten()

    for i in range(seq_len):
        ax = axs[i]

        match plot_type:
            case 'ffn':
                plot_attention(sequence[i], lon, lat, step=step, ax=ax, title=title)
                break
            case 'attention':
                plot_attention(sequence[i], lon, lat, step=step, ax=ax, title=title)
                break
            case _:
                plot_sst(sequence[i], lon, lat, step=step, title=title)