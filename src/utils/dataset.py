import sys, os
import numpy as np
from enum import Enum

sys.path.insert(0, '../')

from config.params import BASE_BOA_ARGO_DATA_PATH, BASE_ERA5_DATA_PATH
from utils.util import resource_argo_monthly_data, import_era5_sst

class FrameType(Enum):
        surface = 0
        mld = 1

class Argo3DTemperatureDataset:
    '''
    Argo 三维温度数据集
    '''
    def __init__(self):
        self.data = resource_argo_monthly_data(BASE_BOA_ARGO_DATA_PATH)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return None
    
    def shape(self, type = FrameType.surface):
        """
        Get the shape of the argo ocean dataset.
        
        Args:
            type: The type of the frame.
        """
        
        if type == FrameType.surface:
            return self.data[0]['temp'].shape
        elif type == FrameType.mld:
            return self.data[0]['mld'].shape
        else:
            return None
    
    def getFrame(self, type = FrameType.surface, time = (0, 0), lon = (0, 0), lat = (0, 0), depth = (0, 0)):
        """
        Get a frame of the argo ocean dataset.
        
        Args:
            type: The type of the frame.
            lon: The range of longitude.
            lat: The range of latitude.
            time: The range of time.
            depth: The range of depth.
        """
        
        time_slice_of_data = np.array(self.data[time[0]:time[1]])
        frame = []
        
        match type:
              case FrameType.surface:
                for i in range(time_slice_of_data.shape[0]):
                    surface_temp = time_slice_of_data[i]['temp']
                    surface_temp = surface_temp[lon[0]:lon[1], lat[0]:lat[1], depth[0]:depth[1]]
                    frame.append(surface_temp)
              case FrameType.mld:
                for i  in range(time_slice_of_data.shape[0]):
                    mld = time_slice_of_data[i]['mld']
                    mld = mld[lon[0]:lon[1], lat[0]:lat[1], depth[0]:depth[1]]
                    frame.append(mld)
        
        return np.array(frame)
    
    
# ERA5 三维数据集
class ERA5SstDataset:
    def __init__(self):
        first_file = None
        
        with os.scandir(BASE_ERA5_DATA_PATH) as files:
            for entry in files:
                if entry.is_file() and entry.name.endswith('.nc'):
                    first_file = entry.path
                    break
        if (first_file is not None):
            self.data = np.transpose(import_era5_sst(first_file), (0, 2, 1)) # 交换位置保证经度在前
            self.data.precision = 0.25
        else: self.data = np.empty(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def shape(self):
        return self.data.shape

    def getFrame(self, time = (0, 0), lon = (0, 0), lat = (0, 0)):
        lon = tuple([int(item / self.data.precision) for item in lon])
        lat = tuple([int(item / self.data.precision) for item in lat])
        return np.array(self.data[time[0]:time[1], lon[0]:lon[1], lat[0]:lat[1]])
