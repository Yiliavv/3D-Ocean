# %% 导入库
import sys

sys.path.append('B://workspace/tensorflow')

import numpy as np

from torch.utils.data import DataLoader, random_split

from src.models.RDF import RDFNetwork
from src.dataset.Argo import Argo3DTemperatureDataset
from src.config.params import Areas
from src.utils.log import Log

from src.dataset.Argo import depthMap
from src.config.constants import Alphabet

# %% 工具函数

def get_lon(lon):
    lon_s = 360 + lon[0] if lon[0] < 0 else lon[0]
    lon_e = 360 + lon[1] if lon[1] < 0 else lon[1]
    
    print(lon_s, lon_e)
    
    return np.array([lon_s, lon_e])

def get_lat(lat):
    return lat + 80

def split_dataset(area):
    dataset = Argo3DTemperatureDataset(lon=get_lon(np.array(area['lon'])), lat=get_lat(np.array(area['lat'])), depth=[0, 58])

    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])
    
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return train_loader, val_loader, test_loader

def get_input(loader):
    input, output = next(iter(loader))
    input = input.reshape(-1, input.shape[-1])
    output = output.reshape(-1, output.shape[-1])
    return input, output

def train_and_evaluate(input, output, test_input, test_output, network):
    model = network(input, output)
    score = model.score(test_input, test_output)
    score = np.around(score, 3)
    return model, score

def rmse(pred, true):
    true = true.numpy()
    rmses = []
    
    for i in range(58):
        pred_temp = pred[:, i].reshape(20, 20)
        true_temp = true[:, i].reshape(20, 20)
        mse = np.nanmean((pred_temp - true_temp) ** 2)
        rmse = np.sqrt(mse)
        rmses.append(rmse)
        
    # 计算 1000dbr 以上的 RMSE
    rmses_1000u = rmses[:45]
    # 计算 1000dbr 以下的 RMSE
    rmses_1000d = rmses[45:]
    
    return np.nanmean(rmses_1000u), np.nanmean(rmses_1000d)

def reshape(data):
    data[data > 99] = np.nan
    return data.reshape(20, 20, 58)

def profile_u(data):
    return np.transpose(data[:, 0, :35], (1, 0))

def profile_d(data):
    return np.transpose(data[0, :, :35])

def plot_sst_station(profile, ax, levels, label, x_labels = None):
    
    # 将 depth 设置为 Y 轴刻度
    ax.set_yticks(np.arange(0, len(depthMap[:35]), 10))
    ax.set_yticks(np.arange(0, len(depthMap[:35]), 5), minor=True)
    # 设置对应的深度值标签
    ax.set_yticklabels(depthMap[:35][::10])
    
    # 设置 X 轴刻度，2004年1月到2024年3月
    ax.set_xticks(np.arange(0, 20, 4))
    # 设置小刻度
    ax.set_xticks(np.arange(0, 20, 1), minor=True)
    
    x_labels = x_labels if x_labels is not None else np.arange(0, 20, 4)
    ax.set_xticklabels(x_labels)
    
    # 修改文本位置到左上角，使用相对坐标
    ax.text(0.02, 0.12, label, 
            fontsize=8, 
            color='orange',
            transform=ax.transAxes,  # 使用相对坐标系统
            verticalalignment='bottom'
    )

    ax.contourf(profile, levels=levels)
    ax.contour(profile, colors='black', alpha=0.5, linewidths=0.2, linestyles='--', levels=30)
    contour_lines = ax.contour(profile, colors='black', linewidths=0.5)
    
    ax.clabel(contour_lines, inline=True, colors='black', fontsize=5, fmt='%d', manual=False)
        
    ax.invert_yaxis()
    

# %% 主函数

def train_rf_model():
    network = RDFNetwork()

    list = []
    test_ = []

    for area in Areas:
        train_loader, val_loader, test_loader = split_dataset(area)

        input, output = get_input(train_loader)
        val_input, val_output = get_input(val_loader)
        model, score = train_and_evaluate(input, output, val_input, val_output, network)
        
        test_input, test_output = get_input(test_loader)
        print(test_input.shape, test_output.shape)
        
        result = model.predict(test_input)
        
        mse_1000u, mse_1000d = rmse(result, test_output);

        list.append([reshape(result), reshape(test_output)])
        
        print(f"Area: {area['title']}, Score: {score}, RMSE: {mse_1000u}, {mse_1000d} \n\
                origin: max = {np.nanmax(test_output)}, min = {np.nanmin(test_output)} \n\
                predict: max = {np.nanmax(result)}, min = {np.nanmin(result)}")
        
        test_.append(test_loader)

    return model, test_ 

