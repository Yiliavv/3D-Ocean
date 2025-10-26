# -*- coding: utf-8 -*-
import os
import arrow
import numpy as np
import netCDF4 as nc

from torch import tensor, unsqueeze, float32
from torch.utils.data import Dataset

from src.config.params import BASE_ERA5_DAILY_DATA_PATH, BASE_ERA5_MONTHLY_DATA_PATH
from src.utils.util import resource_era5_monthly_sst_data

# ERA5 海表数据集
class ERA5SSTDataset(Dataset):
    """
    ERA5 SST 数据集

    :arg  width: 序列长度宽度
    :arg  step: 时间平移步长
    :arg  offset: 时间偏移 (该偏移值是数据批次的偏移，即已经除以了时间步长的值)
    :arg  lon: 经度范围
    :arg  lat: 纬度范围
    :arg  resolution: 空间分辨率（度）
    """

    def __init__(self, width=10, step=10, offset=0, lon=None, lat=None, resolution=1, *args):
        super().__init__(*args)
                    
        if lat is None:
            lat = np.array([0, 0])
        if lon is None:
            lon = np.array([0, 0])

        self.step = step
        self.width = width
        self.offset = offset
        self.lon = np.array(lon)
        self.lat = np.array(lat)
        self.resolution = resolution
        
        self.cur = 0
        # 数据集中包含的时间范围
        self.s_time = arrow.get('2004-01-01')
        self.e_time = arrow.get('2024-12-31')
        
        # 实例级缓存
        self._cache = {}
    
    def __read_item__(self, time: arrow.Arrow):
        """
        读取单个时间点的数据，用于从数据文件中读取数据
        能够跨文件处理
        """
        year = time.year
        
        # 计算文件内偏移
        s_time = arrow.get(year, 1, 1)
        offset = (time - s_time).days
        
        # 使用实例缓存
        if year in self._cache:
            sst = self._cache[year]
        else:            
            # 读文件
            sst, _ = self.__read_sst__(year)
            
            # 刷新缓存（限制缓存大小为3年，防止内存溢出）
            if len(self._cache) >= 3:
                # 删除最早的缓存
                oldest_year = min(self._cache.keys())
                del self._cache[oldest_year]
            
            self._cache[year] = sst
            
        # 经纬度反向
        sst = np.flip(sst, axis=1)
        
        # 经纬度坐标转换为数组索引
        # ERA5: 0.25°分辨率，全球范围 [-180, 180] × [-90, 90]
        lon_indices = (np.arange(self.lon[0], self.lon[1], self.resolution)) * 4
        lat_indices = (np.arange(self.lat[0], self.lat[1], self.resolution) + 90) * 4
        
        lon_indices = lon_indices.astype(np.int32)
        lat_indices = lat_indices.astype(np.int32)
        
        lon_grid, lat_grid = np.meshgrid(lon_indices, lat_indices)
            
        return sst[offset, lat_grid, lon_grid]
    
    def __read_sst__(self, year: int):
        """
        读取文件中的温度和时间数据
        """
        file_path = f"{BASE_ERA5_DAILY_DATA_PATH}/{year}-dailymean.nc"
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        try:
            nc_file = nc.Dataset(file_path, 'r', format='NETCDF4')
            variables = nc_file.variables

            sst = variables['sst'][:]
            time = variables['valid_time'][:]
            
            nc_file.close()

            return sst, time
        except Exception as e:
            raise IOError(f"读取文件 {file_path} 时出错: {str(e)}")

    def __len__(self):
        day_len = (self.e_time - self.s_time).days
        length = (day_len - self.width) / self.step
        
        return int(length) - self.offset

    def __getitem__(self, index):
        self.cur = index
        
        offset = self.offset + self.cur
        start_index = offset * self.step
        end_index = start_index + self.width
        
        # 计算时间范围
        s_time = self.s_time.shift(days=start_index)
        e_time = self.s_time.shift(days=end_index)
        
        time_range = arrow.Arrow.span_range('day', s_time, e_time)
        
        sst_time_series = []
        
        for time, _ in time_range:
            sst = self.__read_item__(time)
            sst_time_series.append(sst)
        
        sst_time_series = np.array(sst_time_series)
        
        # 数据预处理
        sst_time_series = sst_time_series - 273.15
        sst_time_series[sst_time_series > 99] = np.nan
        
        # 数据集不需要梯度追踪
        sst_time_series = tensor(sst_time_series.copy(), dtype=float32)

        fore_ = sst_time_series[:self.width - 1, ...]
        last_ = sst_time_series[-1, ...]

        # 增加一个通道维度, 通道数为 1, 即 (seq_len, width, height) -> (seq_len, 1,  width, height)
        fore_ = unsqueeze(fore_, dim=1)
        last_ = unsqueeze(last_, dim=0)

        return fore_, last_ 
    
# ERA5 海表月平均温度数据集
class ERA5SSTMonthlyDataset(Dataset):
    """
    ERA5 SST 月平均温度数据集
    """
    
    def __init__(self, seq_len=2, offset=0, lon=None, lat=None, resolution=1):
        super().__init__()
        
        if lat is None:
            lat = np.array([0, 0])
        if lon is None:
            lon = np.array([0, 0])
        
        self.lon = np.array(lon)
        self.lat = np.array(lat)
        self.start_time = arrow.get('2004-01-01')
        self.end_time = arrow.get('2024-12-01')
        
        print(f'起始时间：{self.start_time.shift(months=offset).format("YYYY-MM-DD")}')

        self.offset = offset
        self.seq_len = seq_len
        self.resolution = resolution
        
        self.sst_data = resource_era5_monthly_sst_data(BASE_ERA5_MONTHLY_DATA_PATH)
    
    def __len__(self):
        month_len = len(self.sst_data)
        length = month_len - self.seq_len
        
        return length - self.offset
    
    def __getitem__(self, index):
        
        start_index = index + self.offset
        end_index = start_index + self.seq_len
        
        # 支持读取单个月份数据
        if  (self.seq_len == 1):
            return self.__read_sst__(start_index)
        
        # print(f"读取月份: {self.start_time.shift(months=start_index).format('YYYY-MM-DD')} - {self.start_time.shift(months=end_index).format('YYYY-MM-DD')}")
        
        sst_time_series = []
        
        for i in range(start_index, end_index):
            sst = self.__read_sst__(i)
            sst_time_series.append(sst)
        
        # 数据集不需要梯度追踪，模型会自动处理
        sst_time_series = tensor(np.array(sst_time_series), dtype=float32)
        
        fore_ = sst_time_series[:self.seq_len - 1, ...]
        last_ = sst_time_series[-1, ...]
        
        # 增加一个通道维度, 通道数为 1, 即 (seq_len, width, height) -> (seq_len, 1,  width, height)
        fore_ = unsqueeze(fore_, dim=1)
        last_ = unsqueeze(last_, dim=0)

        return fore_, last_
        
    def __read_files__(self):
        for file in os.listdir(BASE_ERA5_MONTHLY_DATA_PATH):
            if file.endswith('.nc'):
                self.files.append(f"{BASE_ERA5_MONTHLY_DATA_PATH}/{file}")
                
    def __read_sst__(self, index: int):
        sst = self.sst_data[index, :, :]
        
        sst = np.flip(sst, axis=0)
        
        sst = sst - 273.15
        
        # 经纬度坐标转换为数组索引
        lon_indices = (np.arange(self.lon[0], self.lon[1], self.resolution)) * 4
        lat_indices = (np.arange(self.lat[0], self.lat[1], self.resolution) + 90) * 4
        
        lon_indices = lon_indices.astype(np.int32)
        lat_indices = lat_indices.astype(np.int32)
        
        lon_grid, lat_grid = np.meshgrid(lon_indices, lat_indices)
        
        return sst[lat_grid, lon_grid]

    def read_ssta(self, index: int):
        """
        计算海表温度异常 (Sea Surface Temperature Anomaly)
        
        SSTA = 当前SST - 气候平均态SST
        气候平均态是指该位置在历史时期的平均温度
        
        :param index: 当前时间索引
        :return: SSTA，与SST相同的形状 [lat, lon]
        """
        sst_list = []

        for i in range(index):
            sst = self.__read_sst__(i)
            sst_list.append(sst)

        # 将列表转换为数组：shape = [时间, 纬度, 经度]
        sst_array = np.array(sst_list)
        
        # 计算气候平均态：在时间维度(axis=0)上取平均
        # 得到每个网格点的平均SST，shape = [纬度, 经度]
        climatology_sst = np.nanmean(sst_array, axis=0)
        
        # 当前时刻的SST
        current_sst = sst_list[-1]

        # 计算异常：当前SST - 气候平均态
        ssta = current_sst - climatology_sst

        return ssta
                