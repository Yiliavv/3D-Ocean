# 画海表温度图的函数

import numpy as np
from cmocean import cm

from matplotlib import cm as cm_plt
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from cartopy.mpl import ticker as tk
from cartopy import crs as ccrs

from config import area

from src.utils.log import Log
from src.plot.base import create_ax, create_axes, create_carto_ax

COLOR_MAP_PROFILE = cm.thermal
COLOR_MAP_SST = cm_plt.jet

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

    print(lon_ticks, lat_ticks)
    
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
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # 去掉网格
    ax.grid(False)
    
    plt.title(title, fontsize=16)
    
    plt.savefig(f'{PREDICT_SAVE_PATH}/{filename}')
    
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
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    ax.grid(False)
    
    plt.title(title, fontsize=16)
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
    ax.clabel(contour_lines_major, inline=True, fontsize=5, fmt='%d')
    
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
    
    ax.set_extent([*plot_lon_range, *plot_lat_range], crs=projection)
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
    cbar.set_label('SSTA (°C)', fontsize=12)
    
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
            fontsize=12, 
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
            fontsize=12, 
            fontweight='bold',
            color='blue',
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='blue'))
    
    # 添加标题
    ax.set_title('NINO Regions and Sea Surface Temperature Anomaly', 
                 fontsize=14, fontweight='bold', pad=15)
    
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
    
    lat = np.array(lat)
    lon = np.array(lon)
    
    # 生成网格点
    lon_grid, lat_grid = np.meshgrid(_range(lon, step), _range(lat, step))
    
    # 计算共同的色标范围
    vmin = min(np.nanmin(sst1), np.nanmin(sst2))
    vmax = max(np.nanmax(sst1), np.nanmax(sst2))
    
    vmin = max(np.floor(vmin), 0)
    vmax = min(np.ceil(vmax), 30)
    
    levels = np.arange(vmin, vmax, 1)
    
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
    rows = int(np.ceil(length / cols))
    
    axs = create_axes(rows, cols, 'all')

    levels = np.linspace(np.nanmin(sst_seq), np.nanmax(sst_seq), 15)

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
    x = np.arange(len(areas))
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
    x = np.arange(len(sst_rmses))
    
    # y 轴为均方根误差
    y = sst_rmses
    
    ax.plot(x, y)
    
    # 设置 x 轴标签
    ax.set_xlabel('time (day)')
    # 设置 y 轴标签
    ax.set_ylabel('rmse (°C)')
    
    return ax

def plot_prediction_error_analysis(pred_output, true_output, lon, lat, title='Prediction Error Analysis'):
    """
    绘制预测误差分析图（包含误差分布和统计信息）
    
    :param pred_output: 预测输出,二维数组
    :param true_output: 真实输出,二维数组
    :param lon: 经度范围 [起始经度, 结束经度]
    :param lat: 纬度范围 [起始纬度, 结束纬度]
    :param title: 图表标题
    :return: 返回图形对象
    """
    
    # 创建包含四个子图的图形
    fig = plt.figure(figsize=(20, 10), dpi=1200)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.3)
    
    # 计算预测误差
    error = pred_output - true_output
    
    # 清理数据，去除NaN和无穷大值
    error_clean = error[~np.isnan(error) & ~np.isinf(error)]
    
    if len(error_clean) == 0:
        for i in range(2):
            for j in range(2):
                ax = fig.add_subplot(gs[i, j])
                ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes, 
                        ha='center', va='center', fontsize=14)
        return fig
    
    # 根据数据形状生成正确的网格
    lat_size, lon_size = pred_output.shape
    
    # 生成经度和纬度数组
    lon_array = np.linspace(lon[0], lon[1], lon_size)
    lat_array = np.linspace(lat[0], lat[1], lat_size)
    
    # 生成网格点
    lon_grid, lat_grid = np.meshgrid(lon_array, lat_array)
    
    # 左上图：预测结果
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.contourf(lon_grid, lat_grid, pred_output, levels=30, cmap=COLOR_MAP_SST, extend='both')
    ax1.contour(lon_grid, lat_grid, pred_output, colors='black', alpha=0.3, linewidths=0.5, levels=10)
    ax1.set_xlabel('Longitude (°E)', fontsize=12)
    ax1.set_ylabel('Latitude (°N)', fontsize=12)
    ax1.set_title('Predicted SST', fontsize=13, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)
    cbar1.set_label('Temperature (°C)', fontsize=10)
    
    # 右上图：真实值
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.contourf(lon_grid, lat_grid, true_output, levels=30, cmap=COLOR_MAP_SST, extend='both')
    ax2.contour(lon_grid, lat_grid, true_output, colors='black', alpha=0.3, linewidths=0.5, levels=10)
    ax2.set_xlabel('Longitude (°E)', fontsize=12)
    ax2.set_ylabel('Latitude (°N)', fontsize=12)
    ax2.set_title('True SST', fontsize=13, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)
    cbar2.set_label('Temperature (°C)', fontsize=10)
    
    # 左下图：预测误差分布
    ax3 = fig.add_subplot(gs[1, 0])
    
    # 计算误差的范围，确保色标以0为中心，使用对称的范围
    abs_max_error = max(abs(np.nanmin(error)), abs(np.nanmax(error)))
    error_levels = np.linspace(-abs_max_error, abs_max_error, 30)
    
    im3 = ax3.contourf(lon_grid, lat_grid, error, levels=error_levels, cmap=COLOR_MAP_ERROR, extend='both')
    ax3.contour(lon_grid, lat_grid, error, colors='black', alpha=0.3, linewidths=0.5, levels=10)
    ax3.set_xlabel('Longitude (°E)', fontsize=12)
    ax3.set_ylabel('Latitude (°N)', fontsize=12)
    ax3.set_title('Prediction Error', fontsize=13, fontweight='bold')
    cbar3 = plt.colorbar(im3, ax=ax3, orientation='vertical', fraction=0.046, pad=0.04)
    cbar3.set_label('Error (°C)', fontsize=10)
    
    # 右下图：误差直方图
    ax4 = fig.add_subplot(gs[1, 1])
    
    # 计算误差统计
    mean_error = np.mean(error_clean)
    std_error = np.std(error_clean)
    rmse = np.sqrt(np.mean(error_clean**2))
    mae = np.mean(np.abs(error_clean))
    
    # 绘制误差直方图
    n_bins = min(50, int(np.sqrt(len(error_clean))))
    ax4.hist(error_clean, bins=n_bins, density=True, alpha=0.7, 
             color='lightcoral', edgecolor='black', linewidth=0.5)
    
    # 添加正态分布拟合
    from scipy import stats
    x_range = np.linspace(error_clean.min(), error_clean.max(), 200)
    normal_fit = stats.norm.pdf(x_range, mean_error, std_error)
    ax4.plot(x_range, normal_fit, 'b-', linewidth=2, label='Normal Fit')
    
    # 添加零线
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero Error')
    
    ax4.set_xlabel('Error (°C)', fontsize=12)
    ax4.set_ylabel('Density', fontsize=12)
    ax4.set_title('Error Distribution', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 添加统计信息
    info_text = f'Mean Error: {mean_error:.3f}°C\nStd Error: {std_error:.3f}°C\nRMSE: {rmse:.3f}°C\nMAE: {mae:.3f}°C'
    ax4.text(0.02, 0.98, info_text, transform=ax4.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 设置总标题
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
    
    return fig

def plot_2d_kernel_density(sst_data, lon, lat):
    """
    绘制二维核密度分布图
    
    :param sst_data: 海表温度数据,二维数组
    :param lon: 经度范围 [起始经度, 结束经度]
    :param lat: 纬度范围 [起始纬度, 结束纬度]
    :return: 返回图形对象
    """
    from scipy import stats
    from scipy.ndimage import gaussian_filter
    
    # 创建单个子图的图形
    fig = plt.figure(figsize=(12, 6), dpi=1200)
    ax = fig.add_subplot(111)
    
    # 清理数据，去除NaN和无穷大值
    sst_clean = sst_data[~np.isnan(sst_data) & ~np.isinf(sst_data)]
    
    if len(sst_clean) == 0:
        ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes, 
                ha='center', va='center', fontsize=14)
        return fig
    
    # 根据数据形状生成正确的网格
    lat_size, lon_size = sst_data.shape
    
    # 生成经度和纬度数组
    lon_array = np.linspace(lon[0], lon[1], lon_size)
    lat_array = np.linspace(lat[0], lat[1], lat_size)
    
    # 生成网格点
    lon_grid, lat_grid = np.meshgrid(lon_array, lat_array)
    
    # 准备二维核密度估计的数据
    # 创建有效的坐标点（排除NaN值）
    valid_mask = ~np.isnan(sst_data) & ~np.isinf(sst_data)
    valid_lon = lon_grid[valid_mask]
    valid_lat = lat_grid[valid_mask]
    
    # 创建数据点数组
    data_points = np.column_stack([valid_lon, valid_lat])
    
    # 计算二维核密度估计
    kde = stats.gaussian_kde(data_points.T)
    
    # 创建用于评估的网格
    x_min, x_max = lon[0], lon[1]
    y_min, y_max = lat[0], lat[1]
    
    # 使用更密集的网格以获得更平滑的结果
    x_grid = np.linspace(x_min, x_max, 100)
    y_grid = np.linspace(y_min, y_max, 100)
    xx, yy = np.meshgrid(x_grid, y_grid)
    
    # 评估核密度
    positions = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(positions)
    density = density.reshape(xx.shape)
    
    
    # 散点图
    im = ax.scatter(xx, yy, c=density, s=2, alpha=0.6, cmap=COLOR_MAP_SST, 
               edgecolors='none', zorder=1)

    ax.figure.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    
    ax.set_xlabel('Longitude (°E)', fontsize=16)
    ax.set_ylabel('Latitude (°N)', fontsize=16)
    
    return fig

