import joblib

from torch import load
from numpy import array, sqrt, append, nanmean, arange, column_stack, ma
from torch.utils.data import DataLoader

from src.config.area import Area
from src.dataset.ERA5 import ERA5SSTDataset
from src.dataset.Argo import Argo3DTemperatureDataset

from src.config.params import MODEL_SAVE_PATH


def load_sst_model(area: Area):
    sst_model = load(f'{MODEL_SAVE_PATH}/sst_model_{area.title}.pth')
    sst_model.eval()
    return sst_model

def load_rf_model():
    rf_model = joblib.load(f'{MODEL_SAVE_PATH}/rf_model.pkl')
    return rf_model

def predict_sst(model, area: Area, start: int, time: int):
    lon = area.lon
    lat = area.lat
    
    dataset = ERA5SSTDataset(width=2, step=1, offset=start, lon=lon, lat=lat)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    it = iter(loader)
    
    sst_queue = []
    rmses = array([])
    
    for i in range(time):
        input_, output = next(it)
        
        pred = model(input_)
        
        pred = pred.detach().numpy()[0, 0, :, :]
        output = output.detach().numpy()[0, 0, :, :]
        
        sst_queue.append(pred)
        
        # 计算 RMSE
        rmse = sqrt(nanmean((pred - output) ** 2))
        rmse = round(rmse, 2)
        rmses = append(rmses, rmse)
    
    return array(sst_queue), rmses

def predict_rf(model, area: Area, sst: array, start):
    lon = area.lon
    lat = area.lat
    
    lat_indices = arange(lat[0], lat[1]) + 80
    lon_indices = arange(lon[0], lon[1]) % 360
    
    rmses = array([])
    
    _station = array([(i, j) for i in lat_indices for j in lon_indices ])

    _sst = sst.reshape(area.width * area.height, -1).reshape(-1)

    input = column_stack((_sst, _station))
    
    pred = model.predict(input)
    
    dataset = Argo3DTemperatureDataset(lon=lon, lat=lat, depth=[0, 58])

    sst_, profile_ = dataset.__getitem__(start)

    pred = pred.reshape(area.width, area.height, 58)
    profile = profile_.reshape(area.width, area.height, 58)
    
    pred[pred > 99] = ma.masked
    profile[profile > 99] = ma.masked
    
    for i in range(58):
        o_temp = profile[:, :, i]
        p_temp = pred[:, :, i]
        rmse = sqrt(nanmean((o_temp - p_temp) ** 2))
        rmses = append(rmses, rmse)

    return pred, profile, rmses
