# -*- coding: utf-8 -*-
import os
import arrow
import numpy as np
import netCDF4 as nc
from datetime import datetime, timedelta

from torch import tensor, unsqueeze, float32
from torch.utils.data import Dataset

from src.config.params import BASE_OISST_DATA_PATH

# OISST 海表温度月平均数据集
class OISSTMonthlyDataset(Dataset):
    """
    OISST SST 月平均温度数据集
    
    与ERA5不同，OISST所有月份数据存储在单个NC文件中
    支持多种分辨率: 0.25°, 0.5°, 1°, 2°
    
    数据时间范围: 1981-09-01 至 2025-09-01
    
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
        self.seq_len = seq_len
        self.offset = offset
        self.resolution = resolution
        
        # OISST数据时间范围
        self.start_time = arrow.get('1981-09-01')
        self.end_time = arrow.get('2025-09-01')
        
        print(f'起始时间：{self.start_time.shift(months=offset).format("YYYY-MM-DD")}')
        
        # 根据分辨率选择对应的文件
        self.nc_file_path = self.__get_file_path__()
        
        # 懒加载数据文件
        self._nc_file = None
        self._sst_data = None
        self._lon_data = None
        self._lat_data = None
        self._time_data = None
        
        # 初始化数据
        self.__load_data__()
        
        # 缓存气候平均态，避免read_ssta重复计算
        self._climatology_cache = {}
    
    def __get_file_path__(self):
        """
        根据分辨率返回对应的文件路径
        """
        if self.resolution == 0.25:
            filename = 'sst.mon.mean.nc'
        elif self.resolution == 0.5:
            filename = 'sst.mon.mean.0.5deg.nc'
        elif self.resolution == 1:
            filename = 'sst.mon.mean.1.0deg.nc'
        elif self.resolution == 2:
            filename = 'sst.mon.mean.2.0deg.nc'
        else:
            raise ValueError(f"不支持的分辨率: {self.resolution}，支持的分辨率为 0.25, 0.5, 1, 2")
        
        file_path = os.path.join(BASE_OISST_DATA_PATH, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"OISST数据文件不存在: {file_path}")
        
        return file_path
    
    def __load_data__(self):
        """
        加载NC文件并读取所有数据到内存
        由于OISST文件相对较小（最大2.1GB），可以一次性加载
        
        注意：OISST 原始经度范围是 [0, 360]，这里转换为 [-180, 180] 以与 ERA5 保持一致
        """
        try:
            self._nc_file = nc.Dataset(self.nc_file_path, 'r', format='NETCDF4')
            
            # 读取原始数据
            sst_data = self._nc_file.variables['sst'][:]  # [time, lat, lon]
            lon_data = self._nc_file.variables['lon'][:]
            self._lat_data = self._nc_file.variables['lat'][:]
            self._time_data = self._nc_file.variables['time'][:]
            
            # 转换经度：[0, 360] -> [-180, 180]
            # 1. 将经度值转换
            lon_data = np.where(lon_data > 180, lon_data - 360, lon_data)
            
            # 2. 重新排序经度（现在经度可能是乱序的）
            lon_sort_indices = np.argsort(lon_data)
            self._lon_data = lon_data[lon_sort_indices]
            
            # 3. 相应地重新排序SST数据（只在经度维度上）
            self._sst_data = sst_data[:, :, lon_sort_indices]
            
            print(f'成功加载OISST数据: {self._sst_data.shape}')
            print(f'时间步数: {len(self._time_data)}')
            print(f'经度范围: [{self._lon_data.min():.2f}, {self._lon_data.max():.2f}] (已转换为 [-180, 180])')
            print(f'纬度范围: [{self._lat_data.min():.2f}, {self._lat_data.max():.2f}]')
            
        except Exception as e:
            raise IOError(f"读取OISST文件 {self.nc_file_path} 时出错: {str(e)}")
    
    def __len__(self):
        """
        返回数据集长度
        """
        total_months = len(self._time_data)
        length = total_months - self.seq_len
        return length - self.offset
    
    def __getitem__(self, index):
        """
        获取一个序列样本
        
        :param index: 样本索引
        :return: (fore_, last_)
                 fore_: [seq_len-1, 1, height, width] 输入序列
                 last_: [1, height, width] 预测目标
        """
        start_index = index + self.offset
        end_index = start_index + self.seq_len
        
        # 支持读取单个月份数据
        if self.seq_len == 1:
            return self.__read_sst__(start_index)
        
        # 预分配数组
        first_sst = self.__read_sst__(start_index)
        sst_time_series = np.empty((self.seq_len, *first_sst.shape), dtype=np.float32)
        sst_time_series[0] = first_sst
        
        for i in range(1, self.seq_len):
            sst_time_series[i] = self.__read_sst__(start_index + i)
        
        # 转换为tensor
        sst_time_series = tensor(sst_time_series, dtype=float32)
        
        fore_ = sst_time_series[:self.seq_len - 1, ...]
        last_ = sst_time_series[-1, ...]
        
        # 增加通道维度: (seq_len, height, width) -> (seq_len, 1, height, width)
        fore_ = unsqueeze(fore_, dim=1)
        last_ = unsqueeze(last_, dim=0)
        
        return fore_, last_
    
    def __read_sst__(self, index: int):
        """
        读取指定时间索引的SST数据
        
        :param index: 时间索引（从0开始）
        :return: SST数据 [height, width]
        """
        # 从内存中读取数据
        sst = self._sst_data[index, :, :]  # [lat, lon]
        
        # OISST数据已经是摄氏度，不需要转换
        # 但需要检查是否有异常值
        sst = sst.astype(np.float32)
        
        # 处理异常值：温度 > 99°C 或 < -10°C 视为无效
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
        OISST经度已转换为 [-180, 180]，与 ERA5 一致
        """
        lon_min = self.lon[0]
        lon_max = self.lon[1]
        
        # 找到最接近的经度索引
        lon_indices = []
        for lon in np.arange(lon_min, lon_max, self.resolution):
            # 找到最接近的索引
            idx = np.argmin(np.abs(self._lon_data - lon))
            lon_indices.append(idx)
        
        return np.array(lon_indices, dtype=np.int32)
    
    def __get_lat_indices__(self):
        """
        将纬度范围转换为数组索引
        OISST纬度: [-90, 90]
        """
        lat_min = self.lat[0]
        lat_max = self.lat[1]
        
        # 找到最接近的纬度索引
        lat_indices = []
        for lat in np.arange(lat_min, lat_max, self.resolution):
            # 找到最接近的索引
            idx = np.argmin(np.abs(self._lat_data - lat))
            lat_indices.append(idx)
        
        return np.array(lat_indices, dtype=np.int32)
    
    def read_ssta(self, index: int):
        """
        计算海表温度异常 (Sea Surface Temperature Anomaly)
        
        SSTA = 当前SST - 气候平均态SST
        气候平均态是指该位置在历史时期的平均温度
        
        :param index: 当前时间索引
        :return: SSTA，与SST相同的形状 [height, width]
        """
        # 检查缓存
        if index in self._climatology_cache:
            climatology_sst = self._climatology_cache[index]
        else:
            # 计算气候平均态
            if index == 0:
                climatology_sst = self.__read_sst__(0)
            else:
                # 批量读取并计算平均值
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

