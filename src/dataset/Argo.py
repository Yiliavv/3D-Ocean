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
        Log.d("Index: ", index)
        cur = index * self.step

        temp_3d = None
        profiles = None

        match self.dtype:
            case FrameType.surface:
                (sst, stations), profiles = self.construct(self.data)
                temp_3d = np.column_stack((sst, stations))[cur:cur + self.step]
            case FrameType.mld:
                temp_3d = np.array([temp['mld'] for temp in self.data[cur:cur + self.step]])

        return temp_3d, profiles[cur:cur + self.step].copy()

    def construct(self, all_months):
        _all_sst = None
        _all_stations = None

        _all_profiles = None

        for one_month in all_months:
            temperature = one_month['temp']
            lon = one_month['lon']
            lat = one_month['lat']

            width = self.lon[1] - self.lon[0]
            height = self.lat[1] - self.lat[0]

            # 输入
            _sst = (temperature[self.lon[0]:self.lon[1], self.lat[0]:self.lat[1], 0]
                    .reshape(width * height, -1)
                    .reshape(-1))
            _station = np.array(
                [(lon[i], lat[j]) for i in range(self.lon[0], self.lon[1]) for j in range(self.lat[0], self.lat[1])]
            )

            # 输出
            _profile = temperature[:180, self.lat[0]:self.lat[1], :].reshape(width * height, -1).copy()

            if _all_sst is None:
                _all_sst = _sst
                _all_stations = _station

                _all_profiles = _profile
            else:
                _all_sst = np.concatenate((_all_sst, _sst))
                _all_stations = np.concatenate((_all_stations, _station))

                _all_profiles = np.concatenate((_all_profiles, _profile))

        return [_all_sst, _all_stations], _all_profiles

    def get(self, index):
        return self.__getitem__(index)