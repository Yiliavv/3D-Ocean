# %% 导入库
import sys

sys.path.append('B://workspace/tensorflow')

import numpy as np

from torch.utils.data import DataLoader, Subset

from src.models.RDF import RDFNetwork
from src.dataset.Argo import Argo3DTemperatureDataset
from src.config.params import Areas

from src.dataset.Argo import depthMap

# %% 工具函数

def split_dataset(dataset):

    # 计算数据集大小和划分点
    total_size = len(dataset)
    offset = int(0.5 * total_size)
    train_size = int(0.15 * total_size)
    val_size = int(0.05 * total_size)
    
    # 顺序划分数据集
    train_dataset = Subset(dataset, range(0, train_size))
    val_dataset = Subset(dataset, range(train_size + offset, train_size + offset + val_size))
    test_dataset = Subset(dataset, range(train_size + offset + val_size, total_size))
    
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return train_loader, val_loader, test_loader

def set_parts(loader):
    input, output = next(iter(loader))
    
    # print("input before: ", input.shape, "output before: ", output.shape)
    
    input = input.reshape(-1, input.shape[-1])
    output = output.reshape(-1, output.shape[-1])
    
    print("input: ", input.shape, "output: ", output.shape)
    
    return input, output

def train_and_evaluate(model, resolution=1):
    score = 0;
    
    epochs = 1
    
    train_parts = np.array([
        [-180, 180, -80, 80],
    ])
    
    for part in train_parts:
        lon = part[:2]
        lat = part[2:]
        
        dataset = Argo3DTemperatureDataset(lon=lon, lat=lat, depth=[0, 58], resolution=resolution)   
        
        train_loader, val_loader, test_loader = split_dataset(dataset)
        
        input, output = set_parts(train_loader)
        test_input, test_output = set_parts(test_loader)
        
        for epoch in range(epochs):
            print(f"Epoch {epoch} start: ")
            model = model.fit(input, output)
        
        score = model.score(test_input, test_output)
        
        del train_loader, test_loader, dataset
    
    return model, score

def rmse(pred, true):
    rmses = []
    
    for i in range(58):
        pred_temp = pred[:, i].reshape(10, 10)
        true_temp = true[:, i].reshape(10, 10)
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

# %% 单模型

def train_rf_model(resolution=1):
    network = RDFNetwork()
    model = network.get_model()
    
    _, score = train_and_evaluate(model, resolution)
        
    print(f"score: {score}")
    
    for area in Areas:
        dataset = Argo3DTemperatureDataset(lon=area.lon, lat=area.lat, depth=[0, 58], resolution=resolution)
        
        input, output = set_parts(dataset)
        result = model.predict(input)
    
        mse_1000u, mse_1000d = rmse(result, output);
    
        print(f"area: {area.title}, rmse: {mse_1000u}, {mse_1000d}")
        
    return model

