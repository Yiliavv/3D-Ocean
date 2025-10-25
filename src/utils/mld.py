import arrow
import numpy as np
from numpy import sqrt, nanmean

from src.config.constants import deep
from src.dataset.Argo import Argo3DTemperatureDataset

def get_mld(month: int, lon: list, lat: list):
    base_time = arrow.get('2004-01-01')
    
    mld_time = base_time.shift(months=month)
    
    dataset = Argo3DTemperatureDataset(lon=lon, lat=lat, depth=[0, 0])
    
    one_month = dataset.get_item_at(mld_time)
    
    mld = one_month['mld']
    
    lon_indices = np.arange(lon[0], lon[1]) % 360
        
    lat_indices = np.arange(lat[0], lat[1]) + 80
    
    lon_grid, lat_grid = np.meshgrid(lon_indices, lat_indices)
    
    mld = np.transpose(mld, (1, 0))
    
    mld = mld[lat_grid, lon_grid]
    
    return mld

# 基于混合层计算误差
def compute_rmse(mld, pred, profile):
    shape = mld.shape

    x_len = shape[0]
    y_len = shape[1]
    
    # 以 20 m 之上为海表
    surface_level = 3
    
    # 混合层之上，为海表
    upper_rmses = []
    # 海表到混合层下界
    middle_rmses = []
    # 混合层之下
    lower_rmses = []
    
    for x in range(x_len):
        for y in range(y_len):
            mld_value = mld[x, y]
            pred_value = pred[x, y, :]
            profile_value = profile[x, y, :]
            
            deep_level = np.searchsorted(deep, mld_value, side='right') + 1
            
            upper_pred = pred_value[:surface_level]
            middle_pred = pred_value[surface_level:deep_level]
            lower_pred = pred_value[deep_level:]
            
            upper_profile = profile_value[:surface_level]
            middle_profile = profile_value[surface_level:deep_level]
            lower_profile = profile_value[deep_level:]
            
            upper_rmse = sqrt(nanmean((upper_pred - upper_profile) ** 2))
            middle_rmse = sqrt(nanmean((middle_pred - middle_profile) ** 2))
            lower_rmse = sqrt(nanmean((lower_pred - lower_profile) ** 2))
            
            upper_rmses.append(upper_rmse)
            middle_rmses.append(middle_rmse)
            lower_rmses.append(lower_rmse)
            
    middle_rmses = np.array(middle_rmses).reshape(x_len, y_len)
    upper_rmses = np.array(upper_rmses).reshape(x_len, y_len)
    lower_rmses = np.array(lower_rmses).reshape(x_len, y_len)
            
    return upper_rmses, middle_rmses, lower_rmses
