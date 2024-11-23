import os
import pandas

from enum import Enum

import numpy as np
from torch.utils.data import Dataset, DataLoader

from src.config.params import BASE_BOA_ARGO_DATA_PATH, BASE_ERA5_DATA_PATH
from src.utils.log import Log
from src.utils.util import resource_argo_monthly_data, import_era5_sst


class FrameType(Enum):
    surface = 0
    mld = 1


class Argo3DTemperatureDataset(Dataset):
    """
    Argo 三维温度数据集
    """

    def __init__(self, step=1, lon=None, lat=None, depth=None, dtype=FrameType.surface, *args):
        super().__init__(*args)

        if lon is None:
            lon = np.array([0, 0])
        if lat is None:
            lat = np.array([0, 0])
        if depth is None:
            depth = np.array([0, 0])

        self.step = step
        self.lon = lon
        self.lat = lat
        self.depth = depth
        self.dtype = dtype
        self.data = resource_argo_monthly_data(BASE_BOA_ARGO_DATA_PATH)

    def __len__(self):
        return int(len(self.data) / self.step)

    def __getitem__(self, index):
        cur = index * self.step

        temp_3d = None

        match self.dtype:
            case FrameType.surface:
                temp_3d = np.array([temp['temp'] for temp in self.data[cur:cur + self.step]])
            case FrameType.mld:
                temp_3d = np.array([temp['mld'] for temp in self.data[cur:cur + self.step]])

        return temp_3d[self.lon[0]:self.lon[1], self.lat[0]:self.lat[1], self.depth[0]:self.depth[1]]


# ERA5 三维数据集
class ERA5SstDataset(Dataset):
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

        self.precision = 4

        self.width = width
        self.step = step
        self.offset = offset
        self.lon = np.array(lon) * self.precision
        self.lat = np.array(lat) * self.precision

        self.page_size = 1000
        self.page_start = offset * step
        self.cache = None

        first_file = None

        with os.scandir(BASE_ERA5_DATA_PATH) as files:
            for entry in files:
                if entry.is_file() and entry.name.endswith('.nc'):
                    first_file = entry.path
                    break
        if first_file is not None:
            self.file = first_file
            page_end = self.page_start + self.page_size
            sst, shape, times = import_era5_sst(self.file, self.page_start, page_end)
            self.shape = shape
            self.times = times
            self.cache = sst
        else:
            self.file = None

    def __len__(self):
        return int((self.shape[0] - self.width) / self.step) - self.offset

    def __getitem__(self, index):
        Log.d(f"Get data from {index + self.offset}")

        # 数据的区间
        start = (index + self.offset) * self.step
        end = start + self.width

        # 缓存的区间
        page_end = self.page_start + self.page_size

        # 如果取的数据在缓存内
        if (self.page_start <= start) and (end < page_end):
            new_start = start - self.page_start
            new_end = start + self.width
            sst = self.cache[new_start:new_end]
        else:
            # 更新数据
            self.page_start = start
            page_end = start + self.page_size
            sst, shape, times = import_era5_sst(self.file, self.page_start, page_end)
            self.cache = sst

            new_start = start - self.page_start
            new_end = start + self.width

            sst = sst[new_start:new_end]

        sst = sst[:, self.lon[0]:self.lon[1], self.lat[0]:self.lat[1]] - 273.15

        fore_ = sst[:self.width - 1, ...]
        last_ = sst[-1, ...]

        # 去掉小时
        start_time = str(pandas.to_datetime(self.times[start], unit='s'))[:-9]
        end_time = str(pandas.to_datetime(self.times[end], unit='s'))[:-9]

        Log.d(f"Time: {start_time} - {end_time}")

        return fore_, last_

    def getTime(self, index):
        start = (index + self.offset) * self.step
        end = start + self.width

        start_time = str(pandas.to_datetime(self.times[start], unit='s'))[:-9]
        end_time = str(pandas.to_datetime(self.times[end], unit='s'))[:-9]

        return start_time, end_time
