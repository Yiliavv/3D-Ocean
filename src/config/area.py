from glob import glob
from numpy import array,arange, abs, sqrt, ceil
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm
from cartopy import crs
from cartopy.io import shapereader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from src.plot.base import create_base_figure

class Area:
    """
    Area 封装了地图的一个区域
    """
    def __init__(self, title: str, lon: list[float], lat: list[float], description: str):
        self.title = title
        self.lon = lon
        self.lat = lat
        self.description = description
        
        self.width = lat[1] - lat[0]
        self.height = lon[1] - lon[0]
        
    def where(self):
        global_map = Global()
        global_map.draw_area(self)


class Global:
    """
    Global 封装了全球的地图
    """
    def __init__(self):
        self.projection = crs.PlateCarree()
        self.fig = create_base_figure()
        # init ax
        self.ax = self.fig.add_subplot(1, 1, 1, projection = self.projection)
        self.ax.set_global()
        self.ax.stock_img()
        self.ax.coastlines(resolution='50m')
        
        self._patch_bathymetry()
        self._set_ticks()

        # nino1+2
        self.nino12 = [-90, -80, -10, 0]
        self.nino3 = [-150, -90, -5, 5]
        self.nino34 = [160, -150, -5, 5]
        self.nino4 = [-170, -120, -5, 5]
        
    """
    在地图上绘制区域
    """
    def draw_area(self, area: Area):
        x, y = self._rectangle(area)
        self._draw_segment(x, y)
        
    def _rectangle(self, area: Area):
        lon_start = area.lon[0]
        lon_end = area.lon[1] 
        lat_start = area.lat[0]
        lat_end = area.lat[1]
        
        x = [lon_start, lon_start,lon_end, lon_end, lon_start]
        y = [lat_start, lat_end, lat_end, lat_start, lat_start]
        
        return x, y
    
    def _draw_segment(self, x, y):
        segment_length = 2.5
    
        # 遍历每个点对绘制黑白相间的边框
        for i in range(len(x)-1):
            # 计算当前线段的总长度
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            total_length = sqrt(dx**2 + dy**2)
            
            # 计算需要多少个线段
            n_segments = int(ceil(total_length/segment_length))
            
            # 计算每个小线段的x和y增量
            dx_segment = dx/n_segments
            dy_segment = dy/n_segments
            
            # 绘制黑白相间的线段
            for j in range(n_segments):
                x_start = x[i] + j*dx_segment
                y_start = y[i] + j*dy_segment
                x_end = x[i] + (j+1)*dx_segment
                y_end = y[i] + (j+1)*dy_segment
                color = 'black' if j % 2 == 0 else 'white'
                self.ax.plot([x_start, x_end], [y_start, y_end], 
                            color=color, linewidth=1, 
                            transform=self.projection)
        
    def _patch_bathymetry(self):
        depths_str, shp_dict = self._load_shapefile()
        
        depths = depths_str.astype(int)
        depths = abs(depths)  # 将深度值转换为正数

        N = len(depths)
        nudge = 0.01
        boundaries = [min(depths)] + sorted(depths+nudge)  # 从浅到深排序
        norm = BoundaryNorm(boundaries, N)
        blues_cm = colormaps['Blues'].resampled(N)  # 移除 '_r' 反转
        colors_depths = blues_cm(norm(depths))
        
        for i, depth_str in enumerate(depths_str):
            self.ax.add_geometries(shp_dict[depth_str].geometries(),
                                    crs=self.projection,
                                    color=colors_depths[i])
        
        depths = depths[depths != 200]
        # 选择较少的刻度点显示  # 每隔一个深度值显示一个刻度
        
        # Add custom colorbar
        sm = ScalarMappable(cmap=blues_cm, norm=norm)
        cb = self.fig.colorbar(mappable=sm,
                        ax=self.ax,
                        spacing='proportional',
                        extend='max',
                        orientation='horizontal',
                        fraction = 0.05,
                        pad=0.1
                        )
        cb.set_ticks(depths)

        # 将刻度标签格式化为千米单位
        tick_labels = [f'{int(d/1000)}k' if d >= 1000 else str(d) for d in depths]
        cb.set_ticklabels(tick_labels)

        cb.set_label(label='Depth (m)', fontsize=8)
        cb.ax.tick_params(labelsize=6)
        
    def _load_shapefile(self):
        shp_dict = {}
        
        files = glob('X:/WorkSpace/tensorflow/src/config/ne_10m_bathymetry_all/*.shp')
        
        assert len(files) > 0
        files.sort()
        
        depths = []
        for f in files:
            depth = '-' + f.split('_')[-1].split('.')[0]  # depth from file name
            depths.append(depth)
            bbox = (-180, -90, 180, 90)  # (x0, y0, x1, y1)
            nei = shapereader.Reader(f, bbox=bbox)
            shp_dict[depth] = nei
        
        depths = array(depths)[::-1]  # sort from surface to bottom
        
        return depths, shp_dict
    
    def _set_ticks(self):
        # 设置经纬度刻度
        self.ax.set_xticks(arange(-180, 180 + 60, 60), crs=self.projection)
        self.ax.set_xticks(arange(-180, 180 + 30, 30), minor=True, crs=self.projection)
        self.ax.set_yticks(arange(-90, 90 + 30, 30), crs=self.projection)
        self.ax.set_yticks(arange(-90, 90 + 15, 15), minor=True, crs=self.projection)

        # 设置经纬度格式
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        self.ax.xaxis.set_major_formatter(lon_formatter)
        self.ax.yaxis.set_major_formatter(lat_formatter)

        # 添加网格线
        self.ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        
        self.ax.set_rasterized(True)
        
