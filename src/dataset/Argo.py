from enum import Enum

import numpy as np

from torch.utils.data import Dataset

from src.config.params import BASE_BOA_ARGO_DATA_PATH
from src.utils.log import Log
from src.utils.util import resource_argo_monthly_data


class FrameType(Enum):
    surface = 0
    mld = 1


class Argo3DTemperatureDataset(Dataset):
    """
    Argo 三维温度数据集
    """

    def __init__(self, lon=None, lat=None, depth=None, dtype=FrameType.surface, *args):
        super().__init__(*args)

        if lon is None:
            lon = np.array([0, 0])
        if lat is None:
            lat = np.array([0, 0])
        if depth is None:
            depth = np.array([0, 0])

        self.current = 0
        self.lon = lon
        self.lat = lat
        self.depth = depth
        self.dtype = dtype
        self.data = resource_argo_monthly_data(BASE_BOA_ARGO_DATA_PATH)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        self.current = index

        temp_3d = None

        match self.dtype:
            case FrameType.surface:
                (sst, station), profile = self.construct(self.data[index])
                temp_3d = np.column_stack([sst, station])
            case FrameType.mld:
                temp_3d = np.array(self.data[index]['mld'])

        return temp_3d, profile

    def construct(self, one_month):
        temp = one_month['temp']
        lon = one_month['lon']
        lat = one_month['lat']
        
        lon_start = self.lon[0]
        lon_end = self.lon[1]
        lat_start = self.lat[0]
        lat_end = self.lat[1]
        
        width = lon_end - lon_start
        height = lat_end - lat_start

        # 输入
        _sst = (temp[lon_start:lon_end, lat_start:lat_end, 0]
                .reshape(width * height, -1)
                .reshape(-1))
        _station = np.array(
            [(lon[i], lat[j]) for i in range(lon_start, lon_end) for j in range(lat_start, lat_end)]
        )
        # 输出
        _profile = temp[lon_start:lon_end, lat_start:lat_end, :].reshape(width * height, -1).copy()
    
        return [_sst, _station], _profile

    def current_month(self):
        return self.data[self.current]