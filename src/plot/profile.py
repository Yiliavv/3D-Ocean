# 绘制三维海洋图
from numpy import linspace, meshgrid, transpose, nanmin, nanmax, arange, ma
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.utils.log import Log
from src.plot.base import create_3d_ax, create_shared_3d_axes, create_ax
from src.plot.sst import _range
from src.utils.depth_interpolator import DepthInterpolator

# 创建3维温度场误差的自定义色标：深蓝（负误差）-> 浅白（零误差）-> 深红（正误差）
# 使用与2D海表温度误差图相同的配色方案，保持视觉一致性
def create_3d_error_colormap():
    """
    创建用于3维温度场误差可视化的自定义色标
    与2D海表温度误差图使用相同的配色方案
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
    cmap = LinearSegmentedColormap.from_list('3d_error_cmap', colors, N=n_bins)
    return cmap

COLOR_MAP_3D_ERROR = create_3d_error_colormap()

def plot_3d_temperature(temp, lon, lat, depth, step=1, cmap=None, label='temperature [°C]',
                        interpolate=False, interp_interval=5.0, interp_method='pchip'):
    """
    绘制三维海温分布图
    
    :param temp: 三维温度数据,形状为 (lon, lat, depth)
    :param lon: 经度范围 [起始经度, 结束经度]
    :param lat: 纬度范围 [起始纬度, 结束纬度] 
    :param depth: 深度列表或深度层数
    :param step: 网格步长
    :param cmap: 色标（可选）
    :param label: 色标标签
    :param interpolate: 是否进行深度插值（默认False）
    :param interp_interval: 插值间隔（米），默认5米
    :param interp_method: 插值方法 'linear', 'pchip', 'cubic'
    :return: 返回3D图像对象
    """
    # 处理插值
    if interpolate:
        if isinstance(depth, int):
            depth_indices = depth
        elif isinstance(depth, list):
            depth_indices = len(depth)
        else:
            depth_indices = len(depth)
        
        interpolator = DepthInterpolator(
            depth_indices=depth_indices,
            target_interval=interp_interval,
            method=interp_method
        )
        
        # DepthInterpolator 期望格式: (depth, height, width) = (depth, lat, lon)
        # 当前格式: (lon, lat, depth)
        # 需要转置: (lon, lat, depth) -> (depth, lat, lon)
        temp_for_interp = np.transpose(temp, (2, 1, 0))
        
        # 对温度数据插值
        temp_interpolated = interpolator.interpolate(temp_for_interp)
        
        # 插值后格式: (depth_new, lat, lon)
        # 转换回: (lon, lat, depth_new)
        temp = np.transpose(temp_interpolated, (2, 1, 0))
        
        depth = interpolator.get_target_depths()
        
        Log.i(f"已应用深度插值: {len(depth)} 层，间隔 {interp_interval}m")
    
    ax = create_3d_ax()
    
    # 生成网格点
    # meshgrid 第一个参数对应 X 轴（经度），第二个参数对应 Y 轴（纬度）
    lon_grid, lat_grid = meshgrid(_range(lon, step), _range(lat, step))
    
    # 计算色标范围
    vmin = nanmin(temp)
    vmax = nanmax(temp)
    
    # 检查是否所有值都是 NaN
    if np.isnan(vmin) or np.isnan(vmax):
        Log.w("警告：温度数据全部为 NaN，无法绘制")
        return None
    
    # 绘制每一层的等温面
    for i, d in enumerate(depth):
        # temp 的形状是 (lon, lat, depth) = (180, 80, depth)
        # temp[:, :, i] 是 (lon, lat) = (180, 80)
        # meshgrid 产生的网格是 (n_lat, n_lon) = (80, 180)
        # 需要转置数据为 (lat, lon) 以匹配网格形状
        temp_layer = temp[:, :, i].T  # 转置: (180, 80) -> (80, 180)
        
        # 使用 masked array 来处理 NaN 值（陆地区域）
        temp_layer_masked = ma.masked_invalid(temp_layer)
        
        # 检查当前层是否有有效数据
        if temp_layer_masked.count() == 0:
            # 如果当前层全是 masked/NaN，跳过绘制
            continue
        
        kw = {'zdir': 'z', 'offset': -d, 'vmin': vmin, 'vmax': vmax}
        if cmap is not None:
            kw['cmap'] = cmap
            
        _ = ax.contourf(lon_grid, lat_grid, temp_layer_masked, **kw)
    
    # 设置坐标轴范围，确保与数据范围一致
    ax.set_xlim(lon[0], lon[1])
    ax.set_ylim(lat[0], lat[1])
    ax.set_zlim(-max(depth), -min(depth))
    ax.set_xlabel('Longitude (°E)', fontsize=10)
    ax.set_ylabel('Latitude (°N)', fontsize=10)
    ax.set_zlabel('Depth (m)', fontsize=10)
    
    # 添加色标
    plt.colorbar(_, ax=ax,
                orientation='vertical',
                pad=0.1,
                fraction=0.02,
                label=label)
    
    return ax

def plot_3d_temperature_error(temp_error, lon, lat, depth, step=1, 
                               filename='3d_temp_error.png', title='3D Temperature Error',
                               interpolate=False, interp_interval=5.0, interp_method='pchip'):
    """
    绘制三维温度场误差分布图
    
    :param temp_error: 三维温度误差数据,形状为 (lon, lat, depth)
    :param lon: 经度范围 [起始经度, 结束经度]
    :param lat: 纬度范围 [起始纬度, 结束纬度]
    :param depth: 深度列表或深度层数
    :param step: 网格步长
    :param filename: 保存文件名
    :param title: 图表标题
    :param interpolate: 是否进行深度插值（默认False）
    :param interp_interval: 插值间隔（米），默认5米
    :param interp_method: 插值方法 'linear', 'pchip', 'cubic'
    :return: 返回3D图像对象
    """
    from src.config.params import ERROR_SAVE_PATH
    
    # 处理插值
    if interpolate:
        if isinstance(depth, int):
            depth_indices = depth
        elif isinstance(depth, list):
            depth_indices = len(depth)
        else:
            depth_indices = len(depth)
        
        interpolator = DepthInterpolator(
            depth_indices=depth_indices,
            target_interval=interp_interval,
            method=interp_method
        )
        
        # DepthInterpolator 期望格式: (depth, height, width) = (depth, lat, lon)
        # 当前格式: (lon, lat, depth)
        # 需要转置: (lon, lat, depth) -> (depth, lat, lon)
        temp_error_for_interp = np.transpose(temp_error, (2, 1, 0))
        
        # 对误差数据插值
        temp_error_interpolated = interpolator.interpolate(temp_error_for_interp)
        
        # 插值后格式: (depth_new, lat, lon)
        # 转换回: (lon, lat, depth_new)
        temp_error = np.transpose(temp_error_interpolated, (2, 1, 0))
        
        depth = interpolator.get_target_depths()
        
        Log.i(f"已应用深度插值: {len(depth)} 层，间隔 {interp_interval}m")
    
    ax = create_3d_ax()
    
    # 生成网格点
    # meshgrid 第一个参数对应 X 轴（经度），第二个参数对应 Y 轴（纬度）
    lon_grid, lat_grid = meshgrid(_range(lon, step), _range(lat, step))
    
    # 计算误差范围，确保色标以0为中心，使用对称的范围
    # 使用 nanmin 和 nanmax 忽略 NaN 值
    temp_min = nanmin(temp_error)
    temp_max = nanmax(temp_error)
    
    # 检查是否所有值都是 NaN
    if np.isnan(temp_min) or np.isnan(temp_max):
        Log.w("警告：温度误差数据全部为 NaN，无法绘制")
        return None
    
    abs_max_error = max(abs(temp_min), abs(temp_max))
    vmin = -abs_max_error
    vmax = abs_max_error
    
    # 绘制每一层的误差等值面
    for i, d in enumerate(depth):
        # temp_error 的形状是 (lon, lat, depth) = (180, 80, depth)
        # temp_error[:, :, i] 是 (lon, lat) = (180, 80)
        # meshgrid(_range(lon, step), _range(lat, step)) 产生:
        # lon_grid: (n_lat, n_lon) = (80, 180), lat_grid: (n_lat, n_lon) = (80, 180)
        # 需要转置数据为 (lat, lon) 以匹配网格形状
        error_layer = temp_error[:, :, i].T  # 转置: (180, 80) -> (80, 180)
        
        # 使用 masked array 来处理 NaN 值（陆地区域）
        # 这样 matplotlib 可以正确处理 NaN 值，在陆地区域不绘制
        error_layer_masked = ma.masked_invalid(error_layer)
        
        # 检查当前层是否有有效数据
        if error_layer_masked.count() == 0:
            # 如果当前层全是 masked/NaN，跳过绘制
            continue
        
        _ = ax.contourf(lon_grid, lat_grid, error_layer_masked,
                       zdir='z', offset=-d,
                       vmin=vmin, vmax=vmax,
                       cmap=COLOR_MAP_3D_ERROR,
                       levels=linspace(vmin, vmax, 30))
    
    # 设置坐标轴范围，确保与数据范围一致
    ax.set_xlim(lon[0], lon[1])
    ax.set_ylim(lat[0], lat[1])
    ax.set_zlim(-max(depth), -min(depth))
    ax.set_xlabel('Longitude (°E)', fontsize=10)
    ax.set_ylabel('Latitude (°N)', fontsize=10)
    ax.set_zlabel('Depth (m)', fontsize=10)
    
    # 添加色标
    cbar = plt.colorbar(_, ax=ax,
                        orientation='vertical',
                        pad=0.1,
                        fraction=0.02,
                        label='Temperature Error (°C)')
    
    # 设置标题
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 保存图像
    plt.savefig(f'{ERROR_SAVE_PATH}/{filename}', dpi=300, bbox_inches='tight')
    
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