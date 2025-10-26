# 绘制三维海洋图
from numpy import linspace, meshgrid, transpose, nanmin, nanmax, arange

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.utils.log import Log
from src.plot.base import create_3d_ax, create_shared_3d_axes, create_ax
from src.plot.sst import _range

# 创建3维温度场误差的自定义色标：蓝色（负误差）-> 浅绿色（零误差）-> 红色（正误差）
# 与海表温度误差色标有所区别，使用蓝-绿-红的连续过渡
def create_3d_error_colormap():
    """
    创建用于3维温度场误差可视化的自定义色标
    - 负误差：深蓝 -> 浅蓝
    - 零误差：明显的绿色
    - 正误差：黄 -> 橙 -> 深红
    配色连续过渡，适合3D可视化
    """
    colors = [
        '#313695',  # 深蓝（大负误差）
        '#4575b4',  # 中蓝
        '#74add1',  # 浅蓝
        '#abd9e9',  # 极浅蓝
        '#d1e5f0',  # 浅青蓝
        '#c7e9c0',  # 浅绿色（接近零）
        '#74c476',  # 明绿色（零误差）- 更明显
        '#41ab5d',  # 深绿色（接近零）
        '#fee08b',  # 浅黄
        '#fdae61',  # 浅橙
        '#f46d43',  # 橙色
        '#d73027',  # 橙红
        '#a50026',  # 深红（大正误差）
    ]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('3d_error_cmap', colors, N=n_bins)
    return cmap

COLOR_MAP_3D_ERROR = create_3d_error_colormap()

def plot_3d_temperature(temp, lon, lat, depth, step=1, cmap=None, label='temperature [°C]'):
    """
    绘制三维海温分布图
    
    :param temp: 三维温度数据,形状为 (lon, lat, depth)
    :param lon: 经度范围 [起始经度, 结束经度]
    :param lat: 纬度范围 [起始纬度, 结束纬度] 
    :param depth: 深度列表
    :param cmap: 色标（可选）
    :param label: 色标标签
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
        
        kw = {'zdir': 'z', 'offset': -d, 'vmin': vmin, 'vmax': vmax}
        if cmap is not None:
            kw['cmap'] = cmap
            
        _ = ax.contourf(lon_grid, lat_grid, temp_layer, **kw)
    
    # 设置坐标轴范围
    # ax.set_xlim(lon)
    # ax.set_ylim(lat)
    ax.set_zlim(-max(depth), -min(depth))
    
    # 添加色标
    plt.colorbar(_, ax=ax,
                orientation='vertical',
                pad=0.1,
                fraction=0.02,
                label=label)
    
    return ax

def plot_3d_temperature_error(temp_error, lon, lat, depth, step=1, 
                               filename='3d_temp_error.png', title='3D Temperature Error'):
    """
    绘制三维温度场误差分布图
    
    :param temp_error: 三维温度误差数据,形状为 (lon, lat, depth)
    :param lon: 经度范围 [起始经度, 结束经度]
    :param lat: 纬度范围 [起始纬度, 结束纬度]
    :param depth: 深度列表
    :param step: 网格步长
    :param filename: 保存文件名
    :param title: 图表标题
    :return: 返回3D图像对象
    """
    from src.config.params import ERROR_SAVE_PATH
    
    ax = create_3d_ax()
    
    # 生成网格点
    lon_grid, lat_grid = meshgrid(_range(lat, step), _range(lon, step))
    
    # 计算误差范围，确保色标以0为中心，使用对称的范围
    abs_max_error = max(abs(nanmin(temp_error)), abs(nanmax(temp_error)))
    vmin = -abs_max_error
    vmax = abs_max_error
    
    # 绘制每一层的误差等值面
    for i, d in enumerate(depth):
        error_layer = transpose(temp_error[:, :, i], (1, 0))
        
        _ = ax.contourf(lon_grid, lat_grid, error_layer,
                       zdir='z', offset=-d,
                       vmin=vmin, vmax=vmax,
                       cmap=COLOR_MAP_3D_ERROR,
                       levels=linspace(vmin, vmax, 30))
    
    # 设置坐标轴
    ax.set_zlim(-max(depth), -min(depth))
    ax.set_xlabel('Latitude (°N)', fontsize=10)
    ax.set_ylabel('Longitude (°E)', fontsize=10)
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