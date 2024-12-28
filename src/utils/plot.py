# For plotting.

import os, io, glob, enum, zipfile, requests, matplotlib

import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import cartopy.mpl.ticker as tk
import matplotlib.pyplot as plt
import matplotlib.colors as cls
import matplotlib.artist as art
import cartopy.io.shapereader as sreader

import src.utils.log as l
import src.config.params as parm

# 基础绘图方法

def create_high_quality_figure():
    """
    创建固定 dpi 为 600 的高分辨率画布
    """

    dpi = 600
    
    figure = plt.figure(dpi=dpi)
    
    return figure

def create_two_col_figure(projection=None):
    """
    创建一个横向的双图画布(两列)
    
    :param projection: 使用的 projection, 可用于创建 cartopy 底图
    """

    figure = create_high_quality_figure();
    axes = []
    axes.append(figure.add_subplot(121, projection=projection))
    axes.append(figure.add_subplot(122, projection=projection))
    
    return axes

def create_figure(row=1, col=1, projection=None):
    figure = create_high_quality_figure()
    
    axes = []
    
    for m in range(row):
        for n in range(col):
            index = m * row + n
            axes.append(figure.add_subplot(row, col, index, projection=projection))
    
    return axes

# cartopy 绘制地图

class Carto:
    """
    封装常用的 cartopy 地图绘制方法
    """
    Feature = enum.Enum('Feature', [
        'COASTLINE', 'LAND', 'OCEAN', 'LAKES', 'RIVERS', 'BORDERS', 'STATES'
    ])
    
    # 使用类属性存储特征映射,避免每个实例都创建一份
    _feature_map = {
        Feature.LAND     : cfeat.LAND,
        Feature.OCEAN    : cfeat.OCEAN,
        Feature.LAKES    : cfeat.LAKES,
        Feature.RIVERS   : cfeat.RIVERS,
        Feature.BORDERS  : cfeat.BORDERS,
        Feature.STATES   : cfeat.STATES,
        Feature.COASTLINE : cfeat.COASTLINE,
    }
    
    def __init__(self, row=1, col=1, projection=None) -> None:
        """
        创建一个 cartopy 绘图对象
        
        :param row: 子图的行数
        :param col: 子图的列数
        :param projection: 投影方式,默认为 PlateCarree
        """
        self.projection = projection or ccrs.PlateCarree()
        self.axes = create_figure(row, col, self.projection)
        
    def _get_target_axes(self, ax_indices):
        """
        根据索引获取目标子图列表
        
        :param ax_indices: 要操作的子图索引,可以是None、单个索引或索引列表
        :return: 目标子图列表
        """
        if ax_indices is None:
            return self.axes
        elif isinstance(ax_indices, (list, tuple)):
            return [self.axes[i] for i in ax_indices]
        else:
            return [self.axes[ax_indices]]
            
    def add_feature(self, feature: Feature, ax_indices=None):
        """
        添加地图特征
        
        :param feature: 要添加的特征,从 Feature 枚举中选择
        :param ax_indices: 要添加到哪些子图,可以是单个索引或索引列表,None 表示所有子图
        """
        for ax in self._get_target_axes(ax_indices):
            ax.add_feature(self._feature_map[feature])
    
    def add_features(self, features: list[Feature], ax_indices=None):
        """
        批量添加多个地图特征
        
        :param features: 要添加的特征列表
        :param ax_indices: 要添加到哪些子图,可以是单个索引或索引列表,None 表示所有子图
        """
        for feature in features:
            self.add_feature(feature, ax_indices)
    
    def set_extent(self, lon_range, lat_range, ax_indices=None):
        """
        设置地图显示范围
        
        :param lon_range: 经度范围,格式为 [min_lon, max_lon]
        :param lat_range: 纬度范围,格式为 [min_lat, max_lat] 
        :param ax_indices: 要设置的子图索引,可以是单个索引或索引列表,None表示所有子图
        """
        for ax in self._get_target_axes(ax_indices):
            ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], 
                         crs=self.projection)
            
    def set_title(self, title, ax_indices=None, **kwargs):
        """
        设置标题
        
        :param title: 标题文本
        :param ax_indices: 要设置的子图索引,可以是单个索引或索引列表,None表示所有子图
        :param kwargs: 传递给 set_title 的其他参数
        """
        for ax in self._get_target_axes(ax_indices):
            ax.set_title(title, **kwargs)
            
class BathCarto(Carto):
    """
    带有水深数据的地图类
    """
    def __init__(self, row=1, col=1, projection=None):
        super().__init__(row, col, projection)
        
        # 添加水深
        self._add_bath()
        
    def _add_bath(self):
        """添加水深数据到地图"""
        depths_str, shape_dict = self._load_bath()
        depths = depths_str.astype(int)
        
        # 设置颜色映射
        n_levels = len(depths)
        boundaries = [min(depths)] + sorted(depths + 0.01)  # 添加小偏移以包含边界数据
        norm = cls.BoundaryNorm(boundaries, n_levels)
        cmap = matplotlib.colormaps['Blues_r'].resampled(n_levels)
        colors = cmap(norm(depths))
        
        # 添加水深数据到每个子图
        for i, depth_str in enumerate(depths_str):
            geometries = shape_dict[depth_str].geometries()
            for ax in self.axes:
                ax.add_geometries(geometries,
                                crs=self.projection,
                                color=colors[i],
                                alpha=0.8)  # 添加透明度提升视觉效果
        
    def _load_bath(self):
        """加载水深数据文件"""
        # 缓存目录
        cache_dir = "bath_data"
        os.makedirs(cache_dir, exist_ok=True)
        
        bath_file = ('https://naturalearth.s3.amazonaws.com/'
                    '10m_physical/ne_10m_bathymetry_all.zip')
        
        # 下载并解压数据
        try:
            r = requests.get(bath_file, timeout=30)
            r.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extractall(cache_dir)
        except (requests.RequestException, zipfile.BadZipFile) as e:
            raise RuntimeError(f"无法加载水深数据: {e}")
            
        # 读取shape文件
        shp_dict = {}
        files = sorted(glob(os.path.join(cache_dir, '*.shp')))
        if not files:
            raise FileNotFoundError("未找到水深数据文件")
            
        depths = []
        for f in files:
            depth = '-' + f.split('_')[-1].split('.')[0]
            depths.append(depth)
            bbox = (-180, -90, 180, 90)
            shp_dict[depth] = sreader.Reader(f, bbox=bbox)
            
        return np.array(depths)[::-1], shp_dict

        

# -------------------------- CADC 绘图 --------------------------
# 绘图的所有函数都返回一个figure和一个axes对象， 可用于将不同的图像绘制在同一个figure上

def plot_cdac_argo_data_one_day(one_day_data, figure: plt.Figure = None, ax: plt.Axes = None):
    """
    绘制一天的浮标位置分布图
    """
    plt.style.use('_mpl-gallery')
    if figure is None:
        figure = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    if ax is None:
        ax = figure.add_subplot(111, projection=ccrs.PlateCarree())

    ax.set_extent([parm.LON_RANGE[0], parm.LON_RANGE[1], parm.LAT_RANGE[0], parm.LAT_RANGE[1]], crs=ccrs.PlateCarree())
    ax.stock_img()
    ax.coastlines()

    gl = ax.gridlines(draw_labels=True)
    gl.xlocator = plt.MaxNLocator(18)
    gl.ylocator = plt.MaxNLocator(18)
    ax.set_title('CDCA Argo Data One Day Float Position Distribution', pad=20)

    # 提取经纬度坐标
    lon = []
    lat = []

    for i, data in enumerate(one_day_data):
        pos = data['pos']
        lon.append(pos['lat'])
        lat.append(pos['lon'])

    ax.scatter(lat, lon, s=20, transform=ccrs.PlateCarree(), marker="*", color='#2ca02c')

    plt.show()

    return figure, ax


def plot_cdac_float_temperature_profile(float_data, figure: plt.Figure = None, ax: plt.Axes = None):
    """
    绘制单个浮标数据的剖面温度图
    :param float_data: Only part of the data is used
    :param figure:  If None, a new figure will be created
    :param ax:  If None, a new axes will be created
    :return: figure, ax
    """

    plt.style.use('_mpl-gallery')
    if figure is None:
        figure = plt.figure(figsize=(8, 10))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    if ax is None:
        ax = figure.add_subplot(111)

    ax.set_title('CDCA Argo Data Profile')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Pressure (dbar)')

    # 提取经纬度坐标
    pressures = -np.array(float_data['pres'][::-1])
    adjusted_pressures = -np.array(float_data['pres_adj'][::-1])
    temperatures = np.array(float_data['temp'][::-1])
    adjusted_temperatures = np.array(float_data['temp_adj'][::-1])

    ax.plot(temperatures, pressures, label='Temperature', color='#1f77b4')

    plt.show()

    return figure, ax


# 绘制一天的浮标数据温度剖面图
def plot_cdac_argo_data_one_day_temperature_profile(one_day_data, figure: plt.Figure = None, ax: plt.Axes = None):
    """
    绘制一天的浮标数据温度剖面图
    :param one_day_data: Only part of the data is used
    :param figure:  If None, a new figure will be created
    :param ax:  If None, a new axes will be created
    :return: figure, ax
    """
    plt.style.use('_mpl-gallery')
    if figure is None:
        figure = plt.figure(figsize=(8, 10))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    if ax is None:
        ax = figure.add_subplot(111)

    ax.set_title('CDCA Argo Data One Day Profile')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Pressure (dbar)')

    # 提取经纬度坐标
    for i, data in enumerate(one_day_data):
        pressures = -np.array(data['data']['pres'][::-1])
        temperatures = np.array(data['data']['temp'][::-1])

        ax.plot(temperatures, pressures, label='Temperature', color='#1f77b4')

    plt.show()

    return figure, ax


# 绘制一天的浮标数据盐度剖面图
def plot_cdac_argo_data_one_day_salinity_profile(one_day_data, figure: plt.Figure = None, ax: plt.Axes = None):
    """
    绘制一天的浮标数据盐度剖面图
    :param one_day_data: Only part of the data is used
    :param figure:  If None, a new figure will be created
    :param ax:  If None, a new axes will be created
    :return: figure, ax
    """
    plt.style.use('_mpl-gallery')
    if figure is None:
        figure = plt.figure(figsize=(8, 10))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    if ax is None:
        ax = figure.add_subplot(111)

    ax.set_title('CDCA Argo Data One Day Profile')
    ax.set_xlabel('Salinity (PSU)')
    ax.set_ylabel('Pressure (dbar)')

    # 提取经纬度坐标
    for i, data in enumerate(one_day_data):
        pressures = -np.array(data['data']['pres'][::-1])
        salinities = np.array(data['data']['psal'][::-1])

        ax.plot(salinities, pressures, label='Salinity', color='#ff7f0e')

    plt.show()

    return figure, ax


# 绘制 CDAC 数据的混合层深度
def plot_cdac_mld_profile(mld, figure: plt.Figure = None, ax: plt.Axes = None):
    """
    绘制 CDAC 数据的混合层深度水平分布图
    """
    plt.style.use('_mpl-gallery')
    if figure is None:
        figure = plt.figure(figsize=(10, 8))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    if ax is None:
        ax = figure.add_subplot(111)

    ax.set_title('Argo Data MLD')

    mld = np.array(mld)
    mld_num = [i for i in range(len(mld))]
    ax.plot(mld_num, mld, label='MLD')
    ax.set_xlabel('Float Number')
    ax.set_ylabel('Depth (m)')

    plt.show()

    return figure, ax


def plot_cdac_mld_distribution(mld, positions, figure: plt.Figure = None, ax: plt.Axes = None):
    """
    绘制 CDAC 数据的混合层深度水平分布图
    """
    plt.style.use('_mpl-gallery')
    if figure is None:
        figure = plt.figure(figsize=(8, 10))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    if ax is None:
        ax = figure.add_subplot(111, projection=ccrs.PlateCarree())

    ax.set_title('Argo Data MLD')
    ax.set_extent([parm.LON_RANGE[0], parm.LON_RANGE[1], parm.LAT_RANGE[0], parm.LAT_RANGE[1]], crs=ccrs.PlateCarree())
    ax.stock_img()
    ax.coastlines()
    # 设置地图刻度
    ax.set_xticks(np.arange(parm.LON_RANGE[0], parm.LON_RANGE[1], 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(parm.LAT_RANGE[0], parm.LAT_RANGE[1], 10), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(tk.LongitudeFormatter())
    ax.yaxis.set_major_formatter(tk.LatitudeFormatter())

    mld = np.array(mld)

    # 划分混合层深度
    # 0-100m
    mld_100 = []
    positions_100 = []
    # 100-300m
    mld_300 = []
    positions_300 = []
    # > 300m
    mld_300_plus = []
    positions_300_plus = []

    for i, mld_value in enumerate(mld):
        if mld_value <= 100:
            mld_100.append(mld_value)
            positions_100.append(positions[i])
        elif mld_value <= 300:
            mld_300.append(mld_value)
            positions_300.append(positions[i])
        else:
            mld_300_plus.append(mld_value)
            positions_300_plus.append(positions[i])

    # 不同深度绘制不同颜色和形状的点
    ax.scatter([pos['lon'] for pos in positions_100], [pos['lat'] for pos in positions_100], s=20,
               transform=ccrs.PlateCarree(), marker="*", color='#2ca02c')
    ax.scatter([pos['lon'] for pos in positions_300], [pos['lat'] for pos in positions_300], s=20,
               transform=ccrs.PlateCarree(), marker="o", color='#ff7f0e')
    ax.scatter([pos['lon'] for pos in positions_300_plus], [pos['lat'] for pos in positions_300_plus], s=20,
               transform=ccrs.PlateCarree(), marker="x", color='#1f77b4')
    plt.show()

    return figure, ax


# -------------------------- BOA Argo 绘图 --------------------------

# 绘制 Argo 在浮标位置的温度剖面图
def plot_argo_float_temperature_profile(temperatures, float_lon, float_lat, figure: plt.Figure = None,
                                        ax: plt.Axes = None):
    """
    绘制 Argo 在浮标位置的温度剖面图
    :param temperatures: 3D array
    :param float_lon: Float position longitude
    :param float_lat: Float position latitude
    :param figure:  If None, a new figure will be created
    :param ax:  If None, a new axes will be created
    :return: figure, ax
    """
    plt.style.use('_mpl-gallery')
    if figure is None:
        figure = plt.figure(figsize=(8, 10))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    if ax is None:
        ax = figure.add_subplot(111)

    ax.set_title('Argo Data Profile')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Depth (m)')

    float_lon = int(float_lon)
    float_lat = int(float_lat)
    l.Log.d("Float Position ", float_lon, float_lat)

    temperature = temperatures[float_lon, float_lat]
    l.Log.d("Temperature ", temperature)
    deeps = -np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 200,
                       220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 500, 550, 600, 650, 700,
                       750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1400, 1500, 1600, 1700, 1800,
                       1900, 1975])
    l.Log.d("Deeps ", deeps)

    ax.plot(temperature, deeps, label='Temperature', color='#1f77b4')

    plt.show()

    return figure, ax


# 绘制 Argo 数据的混合层深度
def plot_argo_mld(mld, figure: plt.Figure = None, ax: plt.Axes = None):
    """
    绘制 Argo 数据的混合层深度水平分布图
    """
    plt.style.use('_mpl-gallery')
    if figure is None:
        figure = plt.figure(figsize=(10, 8))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    if ax is None:
        ax = figure.add_subplot(111, projection=ccrs.PlateCarree())

    ax.set_title('Argo Data MLD')
    ax.set_extent([parm.LON_RANGE[0], parm.LON_RANGE[1], parm.LAT_RANGE[0], parm.LAT_RANGE[1]], crs=ccrs.PlateCarree())
    ax.stock_img()
    ax.coastlines()
    # 设置地图刻度
    ax.set_xticks(np.arange(parm.LON_RANGE[0], parm.LON_RANGE[1], 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(parm.LAT_RANGE[0], parm.LAT_RANGE[1], 10), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(tk.LongitudeFormatter())
    ax.yaxis.set_major_formatter(tk.LatitudeFormatter())

    mld = np.array(mld)

    # 划分混合层深度
    # 0-100m
    mld_100 = []
    positions_100 = []
    # 100-300m
    mld_300 = []
    positions_300 = []
    # > 300m
    mld_300_plus = []
    positions_300_plus = []

    for i in range(mld.shape[0]):
        for j in range(mld.shape[1]):
            if mld[i][j] <= 100:
                mld_100.append(mld[i][j])
                positions_100.append((i + parm.LON_RANGE[0], j + parm.LAT_RANGE[0]))
            elif mld[i][j] <= 300:
                mld_300.append(mld[i][j])
                positions_300.append((i + parm.LON_RANGE[0], j + parm.LAT_RANGE[0]))
            else:
                mld_300_plus.append(mld[i][j])
                positions_300_plus.append((i + parm.LON_RANGE[0], j + parm.LAT_RANGE[0]))

    # 不同深度绘制不同颜色和形状的点
    ax.scatter([pos[0] for pos in positions_100], [pos[1] for pos in positions_100], s=20, transform=ccrs.PlateCarree(),
               marker="*", color='#2ca02c')
    ax.scatter([pos[0] for pos in positions_300], [pos[1] for pos in positions_300], s=20, transform=ccrs.PlateCarree(),
               marker="o", color='#ff7f0e')
    ax.scatter([pos[0] for pos in positions_300_plus], [pos[1] for pos in positions_300_plus], s=20,
               transform=ccrs.PlateCarree(), marker="x", color='#1f77b4')
    plt.show()

    return figure, ax


# -------------------------- 通用绘图方法 --------------------------

def plot_sst_distribution(sst, title='Sea Surface Temperature (°C)', figure: plt.Figure = None, ax: plt.Axes = None, precision=1):
    """
    绘制海表温度分布图
    :param precision: precision of the plot
    :param title: title of the plot
    :param sst: 2D array representing sea surface temperature
    :param figure: If None, a new figure will be created
    :param ax: If None, a new axes will be created
    :return: figure, ax
    """
    plt.style.use('_mpl-gallery')
    if figure is None:
        figure = plt.figure(figsize=(10, 8))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    if ax is None:
        ax = figure.add_subplot(111, projection=ccrs.PlateCarree())

    ax.set_title(title)
    # 设置地图刻度
    ax.set_xticks(np.arange(150, 170, 5), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(20, 40, 5), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(tk.LongitudeFormatter())
    ax.yaxis.set_major_formatter(tk.LatitudeFormatter())

    # 绘制海表温度
    lon = np.arange(150, 170, 1 / precision)
    lat = np.arange(20,40, 1 / precision)
    lon, lat = np.meshgrid(lon, lat)
    contour = ax.contourf(lon, lat, sst, cmap='viridis', transform=ccrs.PlateCarree(), levels=50)

    # 添加颜色条
    cbar = figure.colorbar(contour, ax=ax, orientation='horizontal', pad=0.05, fraction=0.05)

    plt.show()

    return figure, ax


def plot_sst_distribution_compare(sst1, sst2, title='Sea Surface Temperature (°C)', precision=1):
    """
    绘制海表温度分布图
    :param title: title of the plot
    :param sst1: 2D array representing sea surface temperature
    :param sst2: 2D array representing sea surface temperature
    """
    plt.style.use('_mpl-gallery')
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    figure.suptitle(title)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3)

    # 设置地图刻度
    for ax in [ax1, ax2]:
        ax.set_xticks(np.arange(20, 40, 5), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-40, -20, 5), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(tk.LongitudeFormatter())
        ax.yaxis.set_major_formatter(tk.LatitudeFormatter())

    # 绘制第一个海表温度
    lon = np.arange(20, 40,  1 / precision)
    lat = np.arange(-40, -20, 1 / precision)
    lon, lat = np.meshgrid(lon, lat)
    levels = np.arange(min(np.nanmin(sst1), np.nanmin(sst2)), max(np.nanmax(sst1), np.nanmax(sst2)), 0.05)
    contour1 = ax1.contourf(lon, lat, sst1, cmap='coolwarm', transform=ccrs.PlateCarree(),levels=levels)
    ax1.set_title('SST1')

    # 绘制第二个海表温度
    contour2 = ax2.contourf(lon, lat, sst2, cmap='coolwarm', transform=ccrs.PlateCarree(),levels=levels)
    ax2.set_title('SST2')

    # 添加共享颜色条
    cbar = figure.colorbar(contour1, ax=[ax1, ax2], orientation='horizontal', pad=0.05, fraction=0.05)

    plt.show()

    return figure, (ax1, ax2)


def plot_temperature_profile_compare(argo_profile, ear_profile, title="", figrue: plt.Figure = None,
                                     ax: plt.Axes = None):
    """
    绘制温度剖面图
    :param title:
    :param ear_profile:  1D array representing temperature profile
    :param argo_profile:  1D array representing temperature profile
    :param ax: If None, a new axes will be created
    :param figrue: If None, a new figure will be created
    """
    plt.style.use('_mpl-gallery')
    if figrue is None:
        figrue = plt.figure(figsize=(8, 10))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    if ax is None:
        ax = figrue.add_subplot(111)

    ax.set_xlabel('Depth (m)')
    ax.set_ylabel('Temperature (°C)')
    # 限制y轴范围 0 ~ 40
    ax.set_ylim(0, 40)

    ax.plot(argo_profile, label='Argo Source', color='#1f77b4')
    ax.plot(ear_profile, label='EAR5 Predicted', color='#ff7f0e', marker='o', markevery=[0])
    # 增加图例
    ax.legend()

    plt.title(title, pad=20)
    plt.show()

    return figrue, ax


def plot_profile_for_predicted_in_lat(pres, figure=None, ax=None):
    """
    绘制沿纬度方向的预测海温剖面图
    """
    plt.style.use('_mpl-gallery')
    if figure is None:
        figure = plt.figure(figsize=(8, 10))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    if ax is None:
        ax = figure.add_subplot(111)

    deeps = -np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 200,
                       220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 500, 550, 600, 650, 700,
                       750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1400, 1500, 1600, 1700, 1800,
                       1900, 1975])

    lon = np.arange(160, 180, 0.5)

    X, Y = np.meshgrid(lon, deeps)

    pres_in_lon = np.transpose(pres.reshape(40, 40, 58)[:, 39, :], (1, 0))
    contour = ax.contourf(X, Y, pres_in_lon, cmap='coolwarm', levels=50)
    figure.colorbar(contour, ax=ax)

    plt.show()

    return figure, ax


def plot_profile_for_predicted_in_lon(pres, figure=None, ax=None):
    """
    绘制沿经度方向的预测海温剖面图
    """
    plt.style.use('_mpl-gallery')
    if figure is None:
        figure = plt.figure(figsize=(8, 10))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    if ax is None:
        ax = figure.add_subplot(111)


def plot_compared_profile_for_predicted(origin, predicted):
    """
    绘制对比的预测海温剖面图
    """

    plt.style.use('_mpl-gallery')

    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3)

    deeps = -np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 200,
                       220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 500, 550, 600, 650, 700,
                       750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1400, 1500, 1600, 1700, 1800,
                       1900, 1975])

    lon = np.arange(160, 180)
    lat = 5

    figure.suptitle('Profile compare in Latitude {}°S'.format(19 - lat))

    X, Y = np.meshgrid(lon, deeps)

    predicted = predicted.reshape(20, 20, 58)

    # 绘制对比
    origin_in_lon = np.transpose(origin[:, lat, :], (1, 0))
    predicted_in_lon = np.transpose(predicted[:, lat, :], (1, 0))

    contour1 = ax1.contourf(X, Y, origin_in_lon, cmap='coolwarm', levels=20)
    ax1.set_title('Origin')

    contour2 = ax2.contourf(X, Y, predicted_in_lon, cmap='coolwarm', levels=20)
    ax2.set_title('Predicted')

    # 添加共享颜色条
    figure.colorbar(contour1, ax=[ax1, ax2], orientation='horizontal', pad=0.05, fraction=0.05)

    plt.show()


def plot_compared_profile_for_predicted_with_different(low_, high_):
    """
    绘制对比的预测海温剖面图
    """

    plt.style.use('_mpl-gallery')

    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3)

    deeps = -np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 200,
                       220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 500, 550, 600, 650, 700,
                       750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1400, 1500, 1600, 1700, 1800,
                       1900, 1975])

    if low_.shape[0] == 400:
        low_ = low_.reshape(20, 20, 58)
    high_ = high_.reshape(40, 40, 58)

    low_lon = np.arange(160, 180)
    low_lat = 19
    high_lon = np.arange(160, 180, 0.5)
    high_lat = low_lat * 2 + 1

    figure.suptitle('Profile compare in Latitude {}°S'.format(20 - low_lat - 1))

    low_X, low_Y = np.meshgrid(low_lon, deeps)
    high_X, high_Y = np.meshgrid(high_lon, deeps)

    # 绘制对比
    low_in_lon = np.transpose(low_[:, low_lat, :], (1, 0))
    high_in_lon = np.transpose(high_[:, high_lat, :], (1, 0))

    contour1 = ax1.contourf(low_X, low_Y, low_in_lon, cmap='coolwarm', levels=50)
    ax1.set_title('Low Resolution Origin')

    contour2 = ax2.contourf(high_X, high_Y, high_in_lon, cmap='coolwarm', levels=50)
    ax2.set_title('High Resolution Predicted')

    # 添加共享颜色条
    figure.colorbar(contour1, ax=[ax1, ax2], orientation='horizontal', pad=0.05, fraction=0.05)

    plt.show()
    
    
def draw_3d_temperature(temperature, X, Y, Z, fig=None, ax=None):
    '''绘制三维温度图
    '''
    
    if fig is None:
        fig = plt.figure(figsize=(10, 8))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')

    new_deep_map = [
        0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
        110, 120, 130, 140, 150, 160, 170, 180, 200,
        220, 240, 260, 280, 300, 320, 340, 360, 380, 400,
        420, 440, 460, 500, 550, 600, 650, 700, 750, 800,
        850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250,
        1300, 1400, 1500, 1600, 1700, 1800, 1900, 1975
    ]
    max_v = np.nanmax(temperature)
    min_v = np.nanmin(temperature)


    kw = {
            'vmin': min_v,
            'vmax': max_v,
            'levels': np.linspace(min_v, max_v, 10),
        }
    _ = None
    
    for i in range(58):
        
        temp = temperature[:, :, i]
        temp = np.transpose(temp, (1, 0))
        
        _ = ax.contourf(
                X[:, :, i], Y[:, :, i], temp,
                zdir='z', offset=-new_deep_map[i], **kw
            )

        # Set limits of the plot from coord limits
        xmin, xmax = X.min(), X.max()
        ymin, ymax = Y.min(), Y.max()
        zmin, zmax = Z.min(), Z.max()
        ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

        # Plot edges
        edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
        ax.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
        ax.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
        ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

        # Set labels and zticks
        ax.set(
            xlabel='Lon [°]',
            ylabel='Lat [°]',
            zlabel='Depth [m]',
        )

        # Set zoom and angle view
        ax.set_box_aspect(None, zoom=0.9)

        # Colorbar
    fig.colorbar(_, ax=ax, fraction=0.02, pad=0.1, label='Name [units]')