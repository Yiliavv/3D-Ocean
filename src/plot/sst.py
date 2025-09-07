# 画海表温度图的函数

import numpy as np
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
    lon_grid, lat_grid = np.meshgrid(_range(lon, step), _range(lat, step))
    
    # vmin = max(floor(nanmin(sst_diff)), -3)
    # vmax = min(ceil(nanmax(sst_diff)), )
    
    levels = np.arange(-1.5, 1.5, 0.1)
    
    im = ax.contourf(
        lon_grid, lat_grid, sst_diff, 
        levels=levels,
        cmap=COLOR_MAP_SST,
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

    :param ssta: 海表温度异常,二维数组, 经纬度范围为 [-180, 180, -80, 80]
    '''
    
    # 截取 NINO 区域
    ssta_nino = ssta[30:55, 0:60]

    lon_grid, lat_grid = np.meshgrid(_range([-180, -60], step), _range([-20, 30], step))

    # 绘制 NINO 指数图
    ax = create_carto_ax()

    projection = ccrs.PlateCarree()

    ax.set_extent([-180, -80, -20, 20], crs=projection)

    ax.figure.set_size_inches(10, 4)
    # 图像设置黑色边框
    ax.spines['top'].set_color('#444444')
    ax.spines['right'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')
    ax.spines['left'].set_color('#444444')
    
    ax.contourf(lon_grid, lat_grid, ssta_nino, cmap=COLOR_MAP_SST, transform=projection, levels=30)
    
    # 添加 NINO3.4 区域边界框 (西经150°到90°，南纬5°到北纬5°)
    nino34_lon = [-170, -120]  # 西经150°到90°
    nino34_lat = [-5, 5] # 南纬5°到北纬5°
    nini34 = np.nanmean(ssta_nino[15:20, 5:30])

    nino3_lon = [-150, -90]
    nino3_lat = [-5, 5]

    nino3 = np.nanmean(ssta_nino[15:20, 30:55])

    print('NINO3.4: ', nini34, 'NINO3: ', nino3)
    
    
    # 绘制矩形边界框
    import matplotlib.patches as mpatches
    
    # 创建矩形边界框
    rect34 = mpatches.Rectangle(
        (nino34_lon[0], nino34_lat[0]), 
        nino34_lon[1] - nino34_lon[0], 
        nino34_lat[1] - nino34_lat[0],
        linewidth=1,
        edgecolor='#444444',
        facecolor='none',
        transform=projection
    )

    rect3 = mpatches.Rectangle(
        (nino3_lon[0], nino3_lat[0]), 
        nino3_lon[1] - nino3_lon[0], 
        nino3_lat[1] - nino3_lat[0],
        linewidth=1,
        edgecolor='#444444',
        facecolor='none',
        transform=projection
    )
    
    ax.add_patch(rect34)
    ax.add_patch(rect3)
    
    # 添加标签
    ax.text(-160, 0, 'NINO3.4', 
            transform=projection, 
            fontsize=14, 
            fontweight='bold',
            color='black',
            ha='center',
            va='center')

    ax.text(-100, 0, 'NINO3', 
            transform=projection, 
            fontsize=14, 
            fontweight='bold',
            color='black',
            ha='center',
            va='center')
    
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
    im3 = ax3.contourf(lon_grid, lat_grid, error, levels=20, cmap='RdBu_r', extend='both')
    ax3.contour(lon_grid, lat_grid, error, colors='black', alpha=0.5, linewidths=0.5, levels=10)
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

