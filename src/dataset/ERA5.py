import os
import numpy as np
import netCDF4 as nc

from torch import tensor, unsqueeze
from torch.utils.data import Dataset

from pandas import to_datetime

from src.config.params import BASE_ERA5_DATA_PATH, TEMP_FILE


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

        # 初始化属性
        self.precision = 4

        self.width = width
        self.step = step
        self.offset = offset
        self.lon = np.array(lon) * self.precision
        self.lat = np.array(lat) * self.precision

        self.current = 0

        # 导入 netcdf 数据， 创建临时文件进行重新分块
        first_file = None

        with os.scandir(BASE_ERA5_DATA_PATH) as files:
            for entry in files:
                if entry.is_file() and entry.name.endswith('.nc'):
                    first_file = entry.path
                    break

        if first_file is not None:
            # 读取变量
            nc_file = nc.Dataset(first_file, 'r', format='NETCDF4')
            variables = nc_file.variables

            sst = variables['sst']
            time = variables['valid_time']
            lon = variables['longitude']
            lat = variables['latitude']

            if os.path.exists(TEMP_FILE + "/sst.nc"):
                self.temp_file = nc.Dataset(TEMP_FILE + "/sst.nc", 'r', format='NETCDF4')
            else:
                open(TEMP_FILE + "/sst.nc", 'w').close()
                # 创建临时文件
                self.temp_file = nc.Dataset(TEMP_FILE + "/sst.nc", 'w', format='NETCDF4')
                # 创建变量
                self.temp_file.createDimension('valid_time', time.shape[0])
                self.temp_file.createDimension('longitude', lon.shape[0])
                self.temp_file.createDimension('latitude', lat.shape[0])

                chunk_sst = self.temp_file.createVariable('sst', 'float32',
                                                          dimensions=sst.dimensions,
                                                          chunksizes=(1, sst.shape[1], sst.shape[2]))
                # 写入数据
                chunk_size = 1000  # Adjust this value based on your memory capacity
                num_chunks = sst.shape[0] // chunk_size

                for i in range(num_chunks):
                    start = i * chunk_size
                    end = start + chunk_size
                    print(start, end)
                    chunk_sst[start:end, :, :] = sst[start:end, :, :]

                if sst.shape[0] % chunk_size != 0:
                    start = num_chunks * chunk_size
                    chunk_sst[start::self, :, :] = sst[start:, :, :]
        else:
            raise FileNotFoundError("No NetCDF file found in the directory")

        sst = self.temp_file.variables['sst']

        # 初始化属性
        self.data = sst
        self.times = time[:]

    def __len__(self):
        return int((self.data.shape[0] - self.width) / self.step) - self.offset

    def __getitem__(self, index):
        self.current = index

        start = self.offset + index
        end = start + self.width

        sst = tensor(self.data[start:end, self.lon[0]:self.lon[1]:self.precision, self.lat[0]:self.lat[1]:self.precision] - 273.15, requires_grad=True)

        fore_ = sst[:self.width - 1, ...]
        last_ = sst[-1, ...]

        # 增加一个通道维度, 通道数为 1, 即 (seq_len, width, height) -> (seq_len, 1,  width, height)
        fore_ = unsqueeze(fore_, dim=1)
        last_ = unsqueeze(last_, dim=0)

        return fore_, last_

    def getTime(self, index):
        start = (index + self.offset) * self.step
        end = start + self.width

        start_time = str(to_datetime(self.times[start], unit='s'))[:-9]
        end_time = str(to_datetime(self.times[end], unit='s'))[:-9]

        return start_time, end_time

    def get(self, index):
        return self.__getitem__(index)

    def __del__(self):
        # 生命周期结束关闭文件
        self.temp_file.close()
