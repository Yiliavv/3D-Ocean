import os
import sys
import numpy as np
from enum import Enum

current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))

# 将项目根目录添加到 sys.path
sys.path.append(project_root)

from src.config.params import BASE_BOA_ARGO_DATA_PATH
from src.utils.log import Log
from src.utils.util import resource_argo_monthly_data

ArgoDataset_3Dim_Sequence = None


# Argo 三维网格数据集
class FrameType(Enum):
        surface = 0
        mld = 1

class Argo3DTemperatureDataset:
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
                    surface_temp = surface_temp[lat[0]:lat[1], lon[0]:lon[1], depth[0]:depth[1]]
                    frame.append(surface_temp)
              case FrameType.mld:
                for i  in range(time_slice_of_data.shape[0]):
                    mld = time_slice_of_data[i]['mld']
                    mld = mld[lat[0]:lat[1], lon[0]:lon[1], depth[0]:depth[1]]
                    frame.append(mld)
        
        return np.array(frame)
