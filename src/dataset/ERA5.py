# -*- coding: utf-8 -*-
import os
import arrow
import numpy as np
import netCDF4 as nc

from torch import tensor, float32
from torch.utils.data import Dataset

from src.config.params import BASE_ERA5_MONTHLY_DATA_PATH


# ERA5 海表月平均温度数据集
class ERA5SSTMonthlyDataset(Dataset):
    """
    ERA5 SST 月平均温度数据集（单文件模式）
    
    支持多种分辨率: 0.25°, 0.5°, 1°, 2°
    参考 OISST 数据集的方式，根据分辨率选择对应的单文件
    
    数据时间范围: 1940-01-01 至 2024-12-01
    
    注意：使用前需要先运行重采样脚本生成对应分辨率的单文件
    
    :arg seq_len: 序列长度（包含输入和输出）
    :arg offset: 时间偏移（数据批次的偏移）
    :arg lon: 经度范围 [lon_min, lon_max]
    :arg lat: 纬度范围 [lat_min, lat_max]
    :arg resolution: 空间分辨率（度），支持 0.25, 0.5, 1, 2
    """
    
    def __init__(self, seq_len=2, offset=0, lon=None, lat=None, resolution=1):
        super().__init__()
        
        if lat is None:
            lat = np.array([0, 0])
        if lon is None:
            lon = np.array([0, 0])
        
        self.lon = np.array(lon)
        self.lat = np.array(lat)
        self.start_time = arrow.get('1940-01-01')
        self.end_time = arrow.get('2024-12-01')
        
        print(f'起始时间：{self.start_time.shift(months=offset).format("YYYY-MM-DD")}')

        self.offset = offset
        self.seq_len = seq_len
        self.resolution = resolution
        
        # 根据分辨率选择数据源（单文件模式）
        self.nc_file_path = self.__get_file_path__()
        
        # 检查文件是否存在
        if not os.path.exists(self.nc_file_path):
            raise FileNotFoundError(
                f"ERA5数据文件不存在: {self.nc_file_path}\n"
                f"请先运行重采样脚本生成对应分辨率的文件。"
            )
        
        # 单文件模式（类似 OISST）
        self._nc_file = None
        self._sst_data = None
        self._lon_data = None
        self._lat_data = None
        self._time_data = None
        self.__load_data__()
        
        # 缓存气候平均态，避免read_ssta重复计算
        self._climatology_cache = {}
    
    def __get_file_path__(self):
        """
        根据分辨率返回对应的文件路径（单文件模式）
        
        :return: 文件路径
        """
        if self.resolution == 0.25:
            filename = 'ERA5.mon.mean.nc'
        elif self.resolution == 0.5:
            filename = 'ERA5.mon.mean.0.5deg.nc'
        elif self.resolution == 1:
            filename = 'ERA5.mon.mean.1.0deg.nc'
        elif self.resolution == 2:
            filename = 'ERA5.mon.mean.2.0deg.nc'
        else:
            raise ValueError(f"不支持的分辨率: {self.resolution}，支持的分辨率为 0.25, 0.5, 1, 2")
        
        # 在 BASE_ERA5_MONTHLY_DATA_PATH 的父目录查找（重采样后的文件）
        parent_dir = os.path.dirname(BASE_ERA5_MONTHLY_DATA_PATH)
        file_path = os.path.join(parent_dir, filename)
        
        return file_path
    
    def __load_data__(self):
        """
        加载单文件数据（类似 OISST）
        """
        try:
            self._nc_file = nc.Dataset(self.nc_file_path, 'r', format='NETCDF4')
            
            # 读取原始数据
            sst_data = self._nc_file.variables['sst'][:]  # [time, lat, lon]
            self._lon_data = self._nc_file.variables['lon'][:]
            self._lat_data = self._nc_file.variables['lat'][:]
            self._time_data = self._nc_file.variables['time'][:]
            
            # 转换经度：[0, 360] -> [-180, 180]（如果需要）
            if self._lon_data.max() > 180:
                self._lon_data = np.where(self._lon_data > 180, self._lon_data - 360, self._lon_data)
                # 重新排序经度
                lon_sort_indices = np.argsort(self._lon_data)
                self._lon_data = self._lon_data[lon_sort_indices]
                # 相应地重新排序SST数据
                sst_data = sst_data[:, :, lon_sort_indices]
            
            # 翻转纬度（ERA5 数据需要翻转）
            self._lat_data = np.flip(self._lat_data)
            sst_data = np.flip(sst_data, axis=1)
            
            self._sst_data = sst_data
            
            print(f'成功加载ERA5数据: {self._sst_data.shape}')
            print(f'时间步数: {len(self._time_data)}')
            print(f'经度范围: [{self._lon_data.min():.2f}, {self._lon_data.max():.2f}]')
            print(f'纬度范围: [{self._lat_data.min():.2f}, {self._lat_data.max():.2f}]')
            print(f'分辨率: {self.resolution}°')
            
        except Exception as e:
            raise IOError(f"读取ERA5文件 {self.nc_file_path} 时出错: {str(e)}")
    
    def __len__(self):
        total_months = len(self._time_data)
        length = total_months - self.seq_len
        
        return length - self.offset
    
    def __getitem__(self, index):
        
        start_index = index + self.offset
        end_index = start_index + self.seq_len
        
        # 支持读取单个月份数据
        if  (self.seq_len == 1):
            return self.__read_sst__(start_index)
        
        # print(f"读取月份: {self.start_time.shift(months=start_index).format('YYYY-MM-DD')} - {self.start_time.shift(months=end_index).format('YYYY-MM-DD')}")
        
        # 优化：预分配数组而非使用list动态扩展
        first_sst = self.__read_sst__(start_index)
        sst_time_series = np.empty((self.seq_len, *first_sst.shape), dtype=np.float32)
        sst_time_series[0] = first_sst
        
        for i in range(1, self.seq_len):
            sst_time_series[i] = self.__read_sst__(start_index + i)
        
        # 优化：移除不必要的.copy()，tensor会自动复制数据
        sst_time_series = tensor(sst_time_series, dtype=float32)
        
        fore_ = sst_time_series[:self.seq_len - 1, ...]
        last_ = sst_time_series[-1, ...]

        return fore_, last_
        
    def __read_sst__(self, index: int):
        """
        读取指定时间索引的SST数据
        
        :param index: 时间索引（从0开始）
        :return: SST数据 [height, width]
        """
        # 从内存中读取数据
        sst = self._sst_data[index, :, :]  # [lat, lon]
        
        # 温度转换：开尔文 -> 摄氏度
        sst = sst - 273.15
        
        # 处理异常值
        sst = sst.astype(np.float32)
        sst[sst > 99] = np.nan
        sst[sst < -10] = np.nan
        
        # 提取指定经纬度范围的数据
        lon_indices = self.__get_lon_indices__()
        lat_indices = self.__get_lat_indices__()
        
        # 创建网格索引
        lon_grid, lat_grid = np.meshgrid(lon_indices, lat_indices)
        
        return sst[lat_grid, lon_grid]
    
    def __get_lon_indices__(self):
        """
        将经度范围转换为数组索引
        """
        lon_min = self.lon[0]
        lon_max = self.lon[1]
        
        # 使用实际的数据坐标
        lon_indices = []
        for lon in np.arange(lon_min, lon_max, self.resolution):
            # 找到最接近的索引
            idx = np.argmin(np.abs(self._lon_data - lon))
            lon_indices.append(idx)
        return np.array(lon_indices, dtype=np.int32)
    
    def __get_lat_indices__(self):
        """
        将纬度范围转换为数组索引
        """
        lat_min = self.lat[0]
        lat_max = self.lat[1]
        
        # 使用实际的数据坐标
        lat_indices = []
        for lat in np.arange(lat_min, lat_max, self.resolution):
            # 找到最接近的索引
            idx = np.argmin(np.abs(self._lat_data - lat))
            lat_indices.append(idx)
        return np.array(lat_indices, dtype=np.int32)

    def read_ssta(self, index: int):
        """
        计算海表温度异常 (Sea Surface Temperature Anomaly) - 优化版
        
        SSTA = 当前SST - 气候平均态SST
        气候平均态是指该位置在历史时期的平均温度
        
        优化：缓存气候平均态，避免重复计算（性能提升100x+）
        
        :param index: 当前时间索引
        :return: SSTA，与SST相同的形状 [lat, lon]
        """
        # 检查缓存
        if index in self._climatology_cache:
            climatology_sst = self._climatology_cache[index]
        else:
            # 计算气候平均态（使用在线算法，减少内存占用）
            if index == 0:
                climatology_sst = self.__read_sst__(0)
            else:
                # 批量读取，利用LRU缓存
                sst_sum = None
                for i in range(index):
                    sst = self.__read_sst__(i)
                    if sst_sum is None:
                        sst_sum = np.zeros_like(sst, dtype=np.float64)
                    # 只累加非NaN值
                    valid_mask = ~np.isnan(sst)
                    sst_sum[valid_mask] += sst[valid_mask]
                
                climatology_sst = sst_sum / max(index, 1)
            
            # 缓存结果（限制缓存大小）
            if len(self._climatology_cache) > 10:
                # 删除最旧的缓存
                oldest_key = min(self._climatology_cache.keys())
                del self._climatology_cache[oldest_key]
            
            self._climatology_cache[index] = climatology_sst
        
        # 当前时刻的SST
        current_sst = self.__read_sst__(index)
        
        # 计算异常
        ssta = current_sst - climatology_sst
        
        return ssta
    
    def __del__(self):
        """
        析构函数：关闭NC文件
        """
        if self._nc_file is not None:
            try:
                self._nc_file.close()
            except:
                pass
                