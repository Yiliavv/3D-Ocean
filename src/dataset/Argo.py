import arrow
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

    def __init__(self, lon=None, lat=None, depth=None, dtype=FrameType.surface, resolution=1, *args):
        super().__init__(*args)

        if lon is None:
            lon = np.array([0, 0])
        if lat is None:
            lat = np.array([0, 0])
        if depth is None:
            depth = np.array([0, 0])

        self.current = 0
        self.lon = np.array(lon)
        self.lat = np.array(lat)
        self.depth = depth
        self.dtype = dtype
        self.data = resource_argo_monthly_data(BASE_BOA_ARGO_DATA_PATH)
        self.s_time = arrow.get('2004-01-01')
        self.e_time = arrow.get('2024-12-31')
        self.resolution = resolution

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        self.current = index
        
        temp_3d = None

        match self.dtype:
            case FrameType.surface:
                (sst, station), profile = self.construct(index)
                temp_3d = np.column_stack([sst, station])
            case FrameType.mld:
                temp_3d = np.array(self.data[index]['mld'])

        return temp_3d, profile

    def construct(self, index):
        # print(f"total: {len(self.data)}")
        one_month = self.data[index]
        temp = one_month['temp']
        
        print(f"读取月份: {one_month['year']}-{one_month['month']}")
        
        # 经纬度坐标系换算到索引坐标系
        lon_indices = np.arange(self.lon[0], self.lon[1], self.resolution) % 360
        
        lat_indices = np.arange(self.lat[0], self.lat[1], self.resolution) + 80
        
        # print("Argo: ", lat_indices)
        
        lon_grid, lat_grid = np.meshgrid(lon_indices, lat_indices)
        
        width = lat_indices.shape[0]
        height = lon_indices.shape[0]
        
        temp = np.transpose(temp, (1, 0, 2))
        temp = temp[lat_grid, lon_grid, :]

        # 输入
        _sst = temp[:, :, 0].reshape(width * height, -1)
        _sst[_sst > 99] = np.ma.masked
        
        _station = np.array(
            [(i, j) for i in lat_indices for j in lon_indices]
        )
        
        # 输出
        _profile = temp.reshape(width * height, -1).copy()
    
        return [_sst, _station], _profile

    def current_month(self):
        return self.data[self.current]
    
    def get_item_at(self, time: arrow.Arrow):
        # 计算月份差作为索引
        index = (time.year - self.s_time.year) * 12 + (time.month - self.s_time.month)
        
        # 获取数据集中的数据
        return self.data[index - 1]

depthMap = np.array([
    0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
    110, 120, 130, 140, 150, 160, 170, 180, 200,
    220, 240, 260, 280, 300, 320, 340, 360, 380, 400,
    420, 440, 460, 500, 550, 600, 650, 700, 750, 800,
    850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250,
    1300, 1400, 1500, 1600, 1700, 1800, 1900, 1975
])