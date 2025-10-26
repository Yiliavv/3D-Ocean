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
    
    通用的三维温度数据集
    """

    def __init__(self,
                 lon=None, lat=None, depth=None, offset=0,
                 dtype=FrameType.surface, resolution=1, *args):
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
        self.offset = offset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        self.current = index + self.offset
        
        sst = None
        profile = None

        match self.dtype:
            case FrameType.surface:
                sst, profile = self.construct(self.current)
            case FrameType.mld:
                profile = np.array(self.data[self.current]['mld'])

        return sst, profile

    def construct(self, index):
        # print(f"total: {len(self.data)}")
        one_month = self.data[index]
        temp = one_month['temp']
        
        # print(f"读取月份: {one_month['year']}-{one_month['month']}")
        
        # 经纬度坐标转换为数组索引
        lon_indices = np.arange(self.lon[0], self.lon[1], self.resolution)
        lat_indices = np.arange(self.lat[0], self.lat[1], self.resolution) + 80
        
        lon_indices = lon_indices.astype(np.int32)
        lat_indices = lat_indices.astype(np.int32)
        
        lon_grid, lat_grid = np.meshgrid(lon_indices, lat_indices)
        
        # 将温度大于99的值替换为nan，在 Argo 数据中，9999 表示无效值
        temp[temp > 99] = np.nan
        
        temp = self.__normalize__(temp)
        
        temp = np.transpose(temp, (1, 0, 2))
        temp = temp[lat_grid, lon_grid, self.depth[0]:self.depth[1]]
        
        # print(f"=========== temp 3d ==========: {temp.shape}")
        
        sst = temp[:, :, 0]
        
        # print(f"=========== sst ==========: {sst.shape}")
    
        return sst, temp

    def current(self):
        '''
        获取索引指向的月份的数据
        '''
        return self.data[self.current]
    
    def get_item_at(self, time: arrow.Arrow):
        '''
        获取指定时间的数据
        
        参数:
            time: arrow.Arrow, 时间
        '''
        # 计算月份差作为索引
        index = (time.year - self.s_time.year) * 12 + (time.month - self.s_time.month)
        
        # 获取数据集中的数据
        return self.data[index - 1]
    
    def __normalize__(self, x):
        # 保存原始nan掩码
        x_mask = np.isnan(x)
        # 将nan值替换为0，以便模型处理
        x_processed = np.copy(x)
        x_processed[x_mask] = 0.0
        
        return x_processed


class ArgoDepthMap():
    depth_map = np.array([
        0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
        110, 120, 130, 140, 150, 160, 170, 180, 200,
        220, 240, 260, 280, 300, 320, 340, 360, 380, 400,
        420, 440, 460, 500, 550, 600, 650, 700, 750, 800,
        850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250,
        1300, 1400, 1500, 1600, 1700, 1800, 1900, 1975
    ])

    @staticmethod
    def get(r: list[int] | int):
        if isinstance(r, int):
            return ArgoDepthMap.depth_map[r]

        d_list = []

        for d_index in range(r[0], r[1]):
            d_list.append(ArgoDepthMap.depth_map[d_index])
        
        return d_list
