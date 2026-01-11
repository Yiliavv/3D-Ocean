# -*- coding: utf-8 -*-
import os
import arrow
import numpy as np
import netCDF4 as nc
from glob import glob
from datetime import datetime, timedelta
from collections import defaultdict
from torch import tensor, unsqueeze, float32
from torch.utils.data import Dataset

from src.config.params import BASE_OISST_DATA_PATH, BASE_OISST_DAILY_DATA_PATH



# OISST 海表温度月平均数据集
class OISSTMonthlyDataset(Dataset):
    """
    OISST SST 月平均温度数据集
    
    与ERA5不同，OISST所有月份数据存储在单个NC文件中
    支持多种分辨率: 0.25°, 0.5°, 1°, 2°
    
    数据时间范围: 1981-09-01 至 2025-09-01
    样本数量: ~530 个月
    
    :arg seq_len: 序列长度（包含输入和输出）
    :arg offset: 时间偏移（数据批次的偏移）
    :arg lon: 经度范围 [lon_min, lon_max]
    :arg lat: 纬度范围 [lat_min, lat_max]
    :arg resolution: 空间分辨率（度），支持 0.25, 0.5, 1, 2
    """
    
    def __init__(
        self, 
        seq_len=2, 
        offset=0, 
        lon=None, 
        lat=None, 
        resolution=1,
        **kwargs  # 忽略其他参数
    ):
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
            # 使用 with 语句确保文件正确关闭，避免 Windows 多进程 pickle 问题
            with nc.Dataset(self.nc_file_path, 'r', format='NETCDF4') as nc_file:
                # 读取原始数据
                sst_data = nc_file.variables['sst'][:]  # [time, lat, lon]
                lon_data = nc_file.variables['lon'][:]
                self._lat_data = nc_file.variables['lat'][:]
                self._time_data = nc_file.variables['time'][:]
            
            # 转换经度：[0, 360] -> [-180, 180]
            # 1. 将经度值转换
            lon_data = np.where(lon_data > 180, lon_data - 360, lon_data)
            
            # 2. 重新排序经度（现在经度可能是乱序的）
            lon_sort_indices = np.argsort(lon_data)
            self._lon_data = lon_data[lon_sort_indices]
            
            # 3. 相应地重新排序SST数据（只在经度维度上）
            self._sst_data = sst_data[:, :, lon_sort_indices]
            
            # 设置 _nc_file 为 None，表示文件已关闭
            self._nc_file = None
            
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
        return total_months - self.seq_len - self.offset
    
    def __getitem__(self, index):
        """
        获取一个序列样本
        
        :param index: 样本索引
        :return: (fore_, last_)
                 fore_: [seq_len-1, height, width] 输入序列
                 last_: [height, width] 预测目标
        """
        start_index = index + self.offset
        
        # 支持读取单个月份数据
        if self.seq_len == 1:
            return self.__read_sst__(start_index)
        
        # 预分配数组
        first_sst = self.__read_sst__(start_index)
        sst_time_series = np.empty((self.seq_len, *first_sst.shape), dtype=np.float32)
        sst_time_series[0] = first_sst
        
        for i in range(1, self.seq_len):
            sst_time_series[i] = self.__read_sst__(start_index + i)
        
        # 转换为 tensor
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
        
        # 统一精度：保留到 0.001℃
        sst = np.round(sst, 3)
        
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
        析构函数：清理资源
        注意：NC文件在 __load_data__ 中已使用 with 语句关闭
        """
        pass


# 基于OISST日数据聚合的月平均数据集
class OISSTDailyMonthlyDataset(Dataset):
    """
    基于 OISST 日尺度数据聚合的月平均 SST 数据集
    
    数据来源: REMSS L4 GHRSST MW_OI 日数据
    文件夹: OISST-D (每天一个NC文件)
    
    该数据集将日数据按月聚合计算平均值，生成月平均SST
    
    数据时间范围: 2020-01-01 至 2025-12-07
    空间分辨率: 0.25° (约25km)
    
    :arg seq_len: 序列长度（包含输入和输出）
    :arg offset: 时间偏移（数据批次的偏移，以月为单位）
    :arg lon: 经度范围 [lon_min, lon_max]
    :arg lat: 纬度范围 [lat_min, lat_max]
    :arg resolution: 目标空间分辨率（度），支持 0.25, 0.5, 1, 2
    """
    
    def __init__(self, seq_len=2, offset=0, lon=None, lat=None, resolution=1):
        super().__init__()
        
        if lat is None:
            lat = np.array([-80, 80])
        if lon is None:
            lon = np.array([-180, 180])
        
        self.lon = np.array(lon)
        self.lat = np.array(lat)
        self.seq_len = seq_len
        self.offset = offset
        self.resolution = resolution
        
        # 数据路径
        self.data_path = BASE_OISST_DAILY_DATA_PATH
        
        # 获取所有日数据文件并按月份分组
        self._files_by_month = self._group_files_by_month()
        self._months = sorted(self._files_by_month.keys())
        
        # 数据时间范围
        self.start_time = arrow.get(self._months[0], 'YYYY-MM')
        self.end_time = arrow.get(self._months[-1], 'YYYY-MM')
        
        print(f'[OISSTDailyMonthlyDataset] 数据月份数: {len(self._months)}')
        print(f'[OISSTDailyMonthlyDataset] 时间范围: {self.start_time.format("YYYY-MM")} ~ {self.end_time.format("YYYY-MM")}')
        print(f'[OISSTDailyMonthlyDataset] 起始月份 (含offset): {self.start_time.shift(months=offset).format("YYYY-MM")}')
        
        # 读取经纬度信息（从第一个文件）
        self._lon_data = None
        self._lat_data = None
        self._load_coordinates()
        
        # 月平均数据缓存
        self._monthly_cache = {}
        self._max_cache_size = 24  # 缓存最近24个月的数据
        
        # 气候平均态缓存
        self._climatology_cache = {}
    
    def _group_files_by_month(self):
        """
        将所有日数据文件按月份分组
        
        :return: dict, key为'YYYY-MM', value为该月所有文件路径列表
        """
        files = sorted(glob(os.path.join(self.data_path, '*.nc')))
        
        if not files:
            raise FileNotFoundError(f"OISST日数据文件夹为空: {self.data_path}")
        
        files_by_month = defaultdict(list)
        
        for f in files:
            # 文件名格式: 20200101120000-REMSS-L4_GHRSST-SSTfnd-MW_OI-GLOB-v02.0-fv05.1.nc
            filename = os.path.basename(f)
            date_str = filename[:8]  # 提取 YYYYMMDD
            year_month = f'{date_str[:4]}-{date_str[4:6]}'  # 转为 YYYY-MM
            files_by_month[year_month].append(f)
        
        return dict(files_by_month)
    
    def _load_coordinates(self):
        """
        从第一个文件加载经纬度坐标
        """
        first_file = list(self._files_by_month.values())[0][0]
        
        with nc.Dataset(first_file, 'r') as ds:
            self._lon_data = ds.variables['lon'][:]
            self._lat_data = ds.variables['lat'][:]
        
        print(f'[OISSTDailyMonthlyDataset] 原始经度范围: [{self._lon_data.min():.2f}, {self._lon_data.max():.2f}]')
        print(f'[OISSTDailyMonthlyDataset] 原始纬度范围: [{self._lat_data.min():.2f}, {self._lat_data.max():.2f}]')
        print(f'[OISSTDailyMonthlyDataset] 原始分辨率: {abs(self._lon_data[1] - self._lon_data[0]):.4f}°')
    
    def __len__(self):
        """
        返回数据集长度（以月为单位）
        """
        total_months = len(self._months)
        length = total_months - self.seq_len
        return max(0, length - self.offset)
    
    def __getitem__(self, index):
        """
        获取一个序列样本
        
        :param index: 样本索引
        :return: (fore_, last_)
                 fore_: [seq_len-1, 1, height, width] 输入序列
                 last_: [1, height, width] 预测目标
        """
        start_index = index + self.offset
        
        # 支持读取单个月份数据
        if self.seq_len == 1:
            sst = self._read_monthly_sst(start_index)
            return tensor(sst, dtype=float32)
        
        # 预分配数组
        first_sst = self._read_monthly_sst(start_index)
        sst_time_series = np.empty((self.seq_len, *first_sst.shape), dtype=np.float32)
        sst_time_series[0] = first_sst
        
        for i in range(1, self.seq_len):
            sst_time_series[i] = self._read_monthly_sst(start_index + i)
        
        # 转换为tensor
        sst_time_series = tensor(sst_time_series, dtype=float32)
        
        fore_ = sst_time_series[:self.seq_len - 1, ...]
        last_ = sst_time_series[-1, ...]
        
        return fore_, last_
    
    def _read_monthly_sst(self, month_index: int):
        """
        读取指定月份的月平均SST数据
        
        :param month_index: 月份索引（从0开始）
        :return: 月平均SST数据 [height, width]
        """
        if month_index < 0 or month_index >= len(self._months):
            raise IndexError(f"月份索引越界: {month_index}, 有效范围: [0, {len(self._months)-1}]")
        
        month_key = self._months[month_index]
        
        # 检查缓存
        if month_key in self._monthly_cache:
            return self._monthly_cache[month_key]
        
        # 获取该月所有日数据文件
        daily_files = self._files_by_month[month_key]
        
        # 读取并计算月平均
        sst_sum = None
        valid_count = None
        
        for f in daily_files:
            with nc.Dataset(f, 'r') as ds:
                ds.set_auto_maskandscale(True)
                sst_daily = ds.variables['analysed_sst'][0, :, :]  # [lat, lon]
                
                # 转换为摄氏度
                sst_celsius = sst_daily - 273.15
                
                # 提取指定范围
                sst_region = self._extract_region(sst_celsius)
                
                if sst_sum is None:
                    sst_sum = np.zeros_like(sst_region, dtype=np.float64)
                    valid_count = np.zeros_like(sst_region, dtype=np.int32)
                
                # 累加有效值
                valid_mask = ~np.ma.getmaskarray(sst_region) & ~np.isnan(sst_region)
                sst_sum[valid_mask] += sst_region[valid_mask]
                valid_count[valid_mask] += 1
        
        # 计算月平均
        with np.errstate(divide='ignore', invalid='ignore'):
            monthly_sst = np.where(valid_count > 0, sst_sum / valid_count, np.nan)
        
        monthly_sst = monthly_sst.astype(np.float32)
        
        # 处理异常值
        monthly_sst[monthly_sst > 50] = np.nan
        monthly_sst[monthly_sst < -10] = np.nan
        
        # 统一精度：保留到 0.001℃
        monthly_sst = np.round(monthly_sst, 3)
        
        # 缓存结果
        self._update_cache(month_key, monthly_sst)
        
        return monthly_sst
    
    def _extract_region(self, sst_data):
        """
        提取指定经纬度范围的数据，并进行降采样
        
        :param sst_data: 完整SST数据 [lat, lon]
        :return: 提取后的SST数据
        """
        # 计算索引
        lon_indices = self._get_lon_indices()
        lat_indices = self._get_lat_indices()
        
        # 创建网格索引
        lon_grid, lat_grid = np.meshgrid(lon_indices, lat_indices)
        
        return sst_data[lat_grid, lon_grid]
    
    def _get_lon_indices(self):
        """
        将经度范围转换为数组索引，支持降采样
        """
        lon_min = self.lon[0]
        lon_max = self.lon[1]
        
        # 原始分辨率
        original_res = abs(self._lon_data[1] - self._lon_data[0])
        
        # 计算步长（用于降采样）
        step = max(1, int(round(self.resolution / original_res)))
        
        # 找到最接近的经度索引
        lon_indices = []
        current_lon = lon_min
        while current_lon < lon_max:
            idx = np.argmin(np.abs(self._lon_data - current_lon))
            lon_indices.append(idx)
            current_lon += self.resolution
        
        return np.array(lon_indices, dtype=np.int32)
    
    def _get_lat_indices(self):
        """
        将纬度范围转换为数组索引，支持降采样
        """
        lat_min = self.lat[0]
        lat_max = self.lat[1]
        
        # 找到最接近的纬度索引
        lat_indices = []
        current_lat = lat_min
        while current_lat < lat_max:
            idx = np.argmin(np.abs(self._lat_data - current_lat))
            lat_indices.append(idx)
            current_lat += self.resolution
        
        return np.array(lat_indices, dtype=np.int32)
    
    def _update_cache(self, month_key, data):
        """
        更新缓存，保持缓存大小在限制内
        """
        if len(self._monthly_cache) >= self._max_cache_size:
            # 删除最旧的缓存
            oldest_key = next(iter(self._monthly_cache))
            del self._monthly_cache[oldest_key]
        
        self._monthly_cache[month_key] = data.copy()
    
    def read_ssta(self, index: int):
        """
        计算海表温度异常 (Sea Surface Temperature Anomaly)
        
        SSTA = 当前SST - 气候平均态SST
        气候平均态是指该月份在历史时期的平均温度
        
        :param index: 当前月份索引
        :return: SSTA，与SST相同的形状 [height, width]
        """
        current_month = self._months[index]
        month_num = int(current_month.split('-')[1])  # 提取月份 (1-12)
        
        # 检查缓存
        if month_num in self._climatology_cache:
            climatology_sst = self._climatology_cache[month_num]
        else:
            # 计算该月份的气候平均态（所有年份该月的平均）
            same_month_indices = [
                i for i, m in enumerate(self._months) 
                if int(m.split('-')[1]) == month_num and i < index
            ]
            
            if not same_month_indices:
                # 如果没有历史数据，使用当前月份作为气候态
                climatology_sst = self._read_monthly_sst(index)
            else:
                sst_sum = None
                count = 0
                for idx in same_month_indices:
                    sst = self._read_monthly_sst(idx)
                    if sst_sum is None:
                        sst_sum = np.zeros_like(sst, dtype=np.float64)
                    valid_mask = ~np.isnan(sst)
                    sst_sum[valid_mask] += sst[valid_mask]
                    count += 1
                
                climatology_sst = (sst_sum / count).astype(np.float32)
            
            self._climatology_cache[month_num] = climatology_sst
        
        # 当前月份的SST
        current_sst = self._read_monthly_sst(index)
        
        # 计算异常
        ssta = current_sst - climatology_sst
        
        return ssta
    
    def get_month_info(self, index: int):
        """
        获取指定索引对应的月份信息
        
        :param index: 样本索引
        :return: 月份字符串 'YYYY-MM'
        """
        actual_index = index + self.offset
        if actual_index < 0 or actual_index >= len(self._months):
            raise IndexError(f"索引越界: {index}")
        return self._months[actual_index]
    
    def get_daily_file_count(self, index: int):
        """
        获取指定月份的日数据文件数量（用于验证数据完整性）
        
        :param index: 月份索引
        :return: 该月的日数据文件数量
        """
        month_key = self._months[index + self.offset]
        return len(self._files_by_month[month_key])


# OISST 日平均数据集
class OISSTDailyDataset(Dataset):
    """
    OISST 日尺度 SST 数据集
    
    数据来源: REMSS L4 GHRSST MW_OI 日数据
    文件夹: OISST-D (每天一个NC文件)
    
    数据时间范围: 2020-01-01 至 2025-12-07
    空间分辨率: 0.25° (约25km)
    
    :arg seq_len: 序列长度（包含输入和输出）
    :arg offset: 时间偏移（数据批次的偏移，以天为单位）
    :arg lon: 经度范围 [lon_min, lon_max]
    :arg lat: 纬度范围 [lat_min, lat_max]
    :arg resolution: 目标空间分辨率（度），支持 0.25, 0.5, 1, 2
    """
    
    def __init__(self, seq_len=2, offset=0, lon=None, lat=None, resolution=1):
        super().__init__()
        
        if lat is None:
            lat = np.array([-80, 80])
        if lon is None:
            lon = np.array([-180, 180])
        
        self.lon = np.array(lon)
        self.lat = np.array(lat)
        self.seq_len = seq_len
        self.offset = offset
        self.resolution = resolution
        
        # 数据路径
        self.data_path = BASE_OISST_DAILY_DATA_PATH
        
        # 获取所有日数据文件（按日期排序）
        self._files = self._load_files()
        self._dates = self._parse_dates()
        
        # 数据时间范围
        self.start_time = arrow.get(self._dates[0])
        self.end_time = arrow.get(self._dates[-1])
        
        print(f'[OISSTDailyDataset] 数据天数: {len(self._files)}')
        print(f'[OISSTDailyDataset] 时间范围: {self.start_time.format("YYYY-MM-DD")} ~ {self.end_time.format("YYYY-MM-DD")}')
        print(f'[OISSTDailyDataset] 起始日期 (含offset): {self.start_time.shift(days=offset).format("YYYY-MM-DD")}')
        
        # 读取经纬度信息（从第一个文件）
        self._lon_data = None
        self._lat_data = None
        self._load_coordinates()
        
        # 日数据缓存（LRU缓存）
        self._daily_cache = {}
        self._max_cache_size = 60  # 缓存最近60天的数据
    
    def _load_files(self):
        """
        加载所有日数据文件路径
        
        :return: 排序后的文件路径列表
        """
        files = sorted(glob(os.path.join(self.data_path, '*.nc')))
        
        if not files:
            raise FileNotFoundError(f"OISST日数据文件夹为空: {self.data_path}")
        
        return files
    
    def _parse_dates(self):
        """
        解析文件名中的日期
        
        :return: 日期字符串列表 ['YYYY-MM-DD', ...]
        """
        dates = []
        for f in self._files:
            # 文件名格式: 20200101120000-REMSS-L4_GHRSST-SSTfnd-MW_OI-GLOB-v02.0-fv05.1.nc
            filename = os.path.basename(f)
            date_str = filename[:8]  # 提取 YYYYMMDD
            formatted_date = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}'
            dates.append(formatted_date)
        return dates
    
    def _load_coordinates(self):
        """
        从第一个文件加载经纬度坐标
        """
        with nc.Dataset(self._files[0], 'r') as ds:
            self._lon_data = ds.variables['lon'][:]
            self._lat_data = ds.variables['lat'][:]
        
        print(f'[OISSTDailyDataset] 原始经度范围: [{self._lon_data.min():.2f}, {self._lon_data.max():.2f}]')
        print(f'[OISSTDailyDataset] 原始纬度范围: [{self._lat_data.min():.2f}, {self._lat_data.max():.2f}]')
        print(f'[OISSTDailyDataset] 原始分辨率: {abs(self._lon_data[1] - self._lon_data[0]):.4f}°')
        print(f'[OISSTDailyDataset] 目标分辨率: {self.resolution}°')
    
    def __len__(self):
        """
        返回数据集长度（以天为单位）
        """
        total_days = len(self._files)
        length = total_days - self.seq_len
        return max(0, length - self.offset)
    
    def __getitem__(self, index):
        """
        获取一个序列样本
        
        :param index: 样本索引
        :return: (fore_, last_)
                 fore_: [seq_len-1, 1, height, width] 输入序列
                 last_: [1, height, width] 预测目标
        """
        start_index = index + self.offset
        
        # 支持读取单天数据
        if self.seq_len == 1:
            sst = self._read_daily_sst(start_index)
            return tensor(sst, dtype=float32)
        
        # 预分配数组
        first_sst = self._read_daily_sst(start_index)
        sst_time_series = np.empty((self.seq_len, *first_sst.shape), dtype=np.float32)
        sst_time_series[0] = first_sst
        
        for i in range(1, self.seq_len):
            sst_time_series[i] = self._read_daily_sst(start_index + i)
        
        # 转换为tensor
        sst_time_series = tensor(sst_time_series, dtype=float32)
        
        fore_ = sst_time_series[:self.seq_len - 1, ...]
        last_ = sst_time_series[-1, ...]
        
        return fore_, last_
    
    def _read_daily_sst(self, day_index: int):
        """
        读取指定日期的SST数据
        
        :param day_index: 日期索引（从0开始）
        :return: SST数据 [height, width]
        """
        if day_index < 0 or day_index >= len(self._files):
            raise IndexError(f"日期索引越界: {day_index}, 有效范围: [0, {len(self._files)-1}]")
        
        date_key = self._dates[day_index]
        
        # 检查缓存
        if date_key in self._daily_cache:
            return self._daily_cache[date_key]
        
        # 读取数据
        file_path = self._files[day_index]
        
        with nc.Dataset(file_path, 'r') as ds:
            ds.set_auto_maskandscale(True)
            sst_data = ds.variables['analysed_sst'][0, :, :]  # [lat, lon]
            
            # 转换为摄氏度
            sst_celsius = sst_data - 273.15
            
            # 提取指定范围
            sst_region = self._extract_region(sst_celsius)
        
        sst_region = sst_region.astype(np.float32)
        
        # 处理异常值
        sst_region[sst_region > 50] = np.nan
        sst_region[sst_region < -10] = np.nan
        
        # 统一精度：保留到 0.001℃
        sst_region = np.round(sst_region, 3)
        
        # 缓存结果
        self._update_cache(date_key, sst_region)
        
        return sst_region
    
    def _extract_region(self, sst_data):
        """
        提取指定经纬度范围的数据，并进行降采样
        
        :param sst_data: 完整SST数据 [lat, lon]
        :return: 提取后的SST数据
        """
        lon_indices = self._get_lon_indices()
        lat_indices = self._get_lat_indices()
        
        lon_grid, lat_grid = np.meshgrid(lon_indices, lat_indices)
        
        return sst_data[lat_grid, lon_grid]
    
    def _get_lon_indices(self):
        """
        将经度范围转换为数组索引，支持降采样
        """
        lon_min = self.lon[0]
        lon_max = self.lon[1]
        
        lon_indices = []
        current_lon = lon_min
        while current_lon < lon_max:
            idx = np.argmin(np.abs(self._lon_data - current_lon))
            lon_indices.append(idx)
            current_lon += self.resolution
        
        return np.array(lon_indices, dtype=np.int32)
    
    def _get_lat_indices(self):
        """
        将纬度范围转换为数组索引，支持降采样
        """
        lat_min = self.lat[0]
        lat_max = self.lat[1]
        
        lat_indices = []
        current_lat = lat_min
        while current_lat < lat_max:
            idx = np.argmin(np.abs(self._lat_data - current_lat))
            lat_indices.append(idx)
            current_lat += self.resolution
        
        return np.array(lat_indices, dtype=np.int32)
    
    def _update_cache(self, date_key, data):
        """
        更新缓存，保持缓存大小在限制内
        """
        if len(self._daily_cache) >= self._max_cache_size:
            # 删除最旧的缓存
            oldest_key = next(iter(self._daily_cache))
            del self._daily_cache[oldest_key]
        
        self._daily_cache[date_key] = data.copy()
    
    def get_date_info(self, index: int):
        """
        获取指定索引对应的日期信息
        
        :param index: 样本索引
        :return: 日期字符串 'YYYY-MM-DD'
        """
        actual_index = index + self.offset
        if actual_index < 0 or actual_index >= len(self._dates):
            raise IndexError(f"索引越界: {index}")
        return self._dates[actual_index]
    
    def get_file_path(self, index: int):
        """
        获取指定索引对应的文件路径
        
        :param index: 样本索引
        :return: 文件路径
        """
        actual_index = index + self.offset
        if actual_index < 0 or actual_index >= len(self._files):
            raise IndexError(f"索引越界: {index}")
        return self._files[actual_index]
    
    def read_ssta(self, index: int, climatology_days: int = 30):
        """
        计算海表温度异常 (Sea Surface Temperature Anomaly)
        
        SSTA = 当前SST - 气候平均态SST
        气候平均态使用前 climatology_days 天的平均值
        
        :param index: 当前日期索引
        :param climatology_days: 用于计算气候态的天数（默认30天）
        :return: SSTA，与SST相同的形状 [height, width]
        """
        actual_index = index + self.offset
        
        # 计算气候平均态（使用前N天的平均）
        start_idx = max(0, actual_index - climatology_days)
        
        if start_idx == actual_index:
            # 没有历史数据，返回0异常
            return np.zeros_like(self._read_daily_sst(actual_index))
        
        sst_sum = None
        valid_count = None
        
        for i in range(start_idx, actual_index):
            sst = self._read_daily_sst(i)
            if sst_sum is None:
                sst_sum = np.zeros_like(sst, dtype=np.float64)
                valid_count = np.zeros_like(sst, dtype=np.int32)
            
            valid_mask = ~np.isnan(sst)
            sst_sum[valid_mask] += sst[valid_mask]
            valid_count[valid_mask] += 1
        
        with np.errstate(divide='ignore', invalid='ignore'):
            climatology_sst = np.where(valid_count > 0, sst_sum / valid_count, np.nan)
        
        # 当前日的SST
        current_sst = self._read_daily_sst(actual_index)
        
        # 计算异常
        ssta = current_sst - climatology_sst
        
        return ssta.astype(np.float32)
