import sys
import arrow
import numpy as np
import netCDF4 as nc

from torch import tensor, unsqueeze
from torch.utils.data import Dataset

from src.config.params import BASE_ERA5_DAILY_DATA_PATH
from src.utils.log import Log

cache = {}

months = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
months_leap = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

def hit(year: int):
    if year in cache:
        return cache[year]
    
    return None

def refresh(year: int, sst: np.ndarray):
    cache[year] = sst
    # Log.d(f"已缓存: {cache.keys()}")

# ERA5 海表数据集
class ERA5SSTDataset(Dataset):
    """
    ERA5 SST 数据集

    :arg  width: 序列长度宽度
    :arg  step: 时间平移步长
    :arg  offset: 时间偏移 (该偏移值是数据批次的偏移，即已经除以了时间步长的值)
    :arg  lon: 经度范围
    :arg  lat: 纬度范围
    """

    def __init__(self, width=10, step=10, offset=0, lon=None, lat=None, *args):
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
        
        self.cur = 0
        self.precision = 4
        # 数据集中包含的时间范围
        self.s_time = arrow.get('2004-01-01')
        self.e_time = arrow.get('2024-12-31')
    
    """
    读取单个时间点的数据，用于从数据文件中读取数据
    能够跨文件处理
    """
    def __read_item__(self, time: arrow.Arrow):
        Log.d(f"读取时间: {time.format('YYYY-MM-DD')}")
        
        year = time.year
        
        # 计算文件内偏移
        s_time = arrow.get(year, 1, 1)
        offset = (time - s_time).days
        
        sst = hit(year)
        
        if sst is None:            
            # 读文件
            sst, time = self.__read_sst__(year)
            
            # 刷新缓存
            refresh(year, sst)
        
        # 获取纬度范围
        lat_start = int(self.lat[0] * self.precision)
        lat_end = int(self.lat[1] * self.precision)
        
        # 获取经度范围
        lon_start = int(self.lon[0] * self.precision)
        lon_end = int(self.lon[1] * self.precision)
            
        return sst[offset, lat_start:lat_end:self.precision, lon_start:lon_end:self.precision]
    
    """
    读取文件中的温度和时间数据
    """
    def __read_sst__(self, year: int):
        file_path = f"{BASE_ERA5_DAILY_DATA_PATH}/{year}-dailymean.nc"
        
        nc_file = nc.Dataset(file_path, 'r', format='NETCDF4')
        variables = nc_file.variables

        sst = variables['sst'][:]
        time = variables['valid_time'][:]

        return sst, time

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
        sst_time_series = np.flip(sst_time_series, axis=1)
        sst_time_series[sst_time_series > 99] = np.nan
        sst_time_series = tensor(sst_time_series.copy(), requires_grad=True)

        fore_ = sst_time_series[:self.width - 1, ...]
        last_ = sst_time_series[-1, ...]

        # 增加一个通道维度, 通道数为 1, 即 (seq_len, width, height) -> (seq_len, 1,  width, height)
        fore_ = unsqueeze(fore_, dim=1)
        last_ = unsqueeze(last_, dim=0)

        return fore_, last_


# ERA5 海表月平均数据
class ERA5SSTMonthlyDataset(Dataset):
    """
    ERA5 SST 月平均数据集
    
    :arg  lon: 经度范围
    :arg  lat: 纬度范围
    """
    
    def __init__(self, lon=None, lat=None, *args):
        super().__init__(*args)
        
        if lat is None:
            lat = np.array([0, 0])
        if lon is None:
            lon = np.array([0, 0])
            
        self.lon = np.array(lon)
        self.lat = 90 - np.array(lat)
        self.precision = 4
        
        # 数据集中包含的时间范围
        self.s_time = arrow.get('2004-01-01')
        self.e_time = arrow.get('2024-12-31')
    
    def __len__(self):
        years = self.e_time.year - self.s_time.year
        months = self.e_time.month - self.s_time.month
        
        month_len = years * 12 + months

        return month_len

    """
    读取文件中的温度和时间数据
    """
    def __read_sst__(self, year: int):
        file_path = f"{BASE_ERA5_DAILY_DATA_PATH}/{year}-dailymean.nc"
        
        nc_file = nc.Dataset(file_path, 'r', format='NETCDF4')
        variables = nc_file.variables

        sst = variables['sst'][:]
        time = variables['valid_time'][:]

        return sst, time
    
    """
    读取单个时间点的数据，用于从数据文件中读取数据
    能够跨文件处理
    """
    def __read_item__(self, time: arrow.Arrow):
        year = time.year
        
        # 计算文件内偏移
        s_time = arrow.get(year, 1, 1)
        month_offset = time.month - s_time.month
        
        # 是否是闰年
        is_leap = year % 4 == 0 and year % 100 != 0 or year % 400 == 0
        
        local_months = months
        
        if is_leap:
            local_months = months_leap
        
        sst = hit(year)
        
        if sst is None:            
            # 读文件
            sst, time = self.__read_sst__(year)
            
            # 刷新缓存
            refresh(year, sst)
        
        # 经纬度坐标系转换到索引坐标系
        lon_index_start = (180 + self.lon[0]) * self.precision
        lon_index_end = (180 + self.lon[1]) * self.precision
        
        lat_index_start = (90 + self.lat[0]) * self.precision
        lat_index_end = (90 + self.lat[1]) * self.precision
    
        
        e_ = np.sum(local_months[:month_offset])
        
        if month_offset == 0:
            s_ = 0
        else:
            s_ = np.sum(local_months[:month_offset - 1])
        
        month_sst = sst[s_:e_,
                        lat_index_start:lat_index_end:self.precision,
                        lon_index_start:lon_index_end:self.precision]
        
        # 计算月平均
        month_sst = np.nanmean(month_sst, axis=0)
        
        return month_sst
    
    def __getitem__(self, index):
        month = self.s_time.shift(months=index)
        
        sst = self.__read_item__(month)
        
        return sst     


