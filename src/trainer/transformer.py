# %% 导入库
import sys

from src.plot.sst import  COLOR_MAP_SST, _range

sys.path.append('B://workspace/tensorflow/')

# 训练 Conv-LSTM 模型。
# 该模型通过同一个月的前 14 天的 SST 数据预测未来 1 天的 SST 数据。

# 导入数据集
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

# 定义参数
OFFSET = 0 # 2004-01-01 
WIDTH = 15 # 2 天为一个窗口，获取前 7 天的数据，预测第 8 天的数据
STEP = 1 # 1 天为一个步长

# %% 工具函数
from numpy import array
from torch.utils.data import random_split

from src.dataset.ERA5 import ERA5SSTDataset

def split_data(area):
    lon = area['lon']
    lat = area['lat']

    dataset = ERA5SSTDataset(WIDTH, STEP, OFFSET, lon, lat)
    
    train_data_set, val_data_set, test_data_set = random_split(dataset, [0.7, 0.2, 0.1])

    train_dataloader = DataLoader(train_data_set, batch_size=15, shuffle=False)
    val_dataloader = DataLoader(val_data_set, batch_size=15, shuffle=False)
    test_dataloader = DataLoader(test_data_set, batch_size=15, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def get_sst(): 
    lon = [-180, 180]
    lat = [-90, 90]

    dataset = ERA5SSTDataset(WIDTH, STEP, OFFSET, lon, lat)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    fore_, last_ = next(iter(loader))

    return last_[0, :, :]

def plot_sst_month(sst, ax, levels, label, area):
    lon = area['lon']
    lat = area['lat']
        
    ax.set_xticks(_range([0, lon[1] - lon[0]], 5))
    ax.set_yticks(_range([0, lat[1] - lat[0]], 5))
    ax.tick_params(axis='both', which='major', labelsize=8)  # 设置刻度标签大小
    # 计算刻度标签，转换为经纬度
    x_labels = _range(lon, 5)
    y_labels = _range(lat, 5)
    
    # 转换坐标
    x_labels = [f"{x:.0f}°E" if x >= 0 else f"{abs(x):.0f}°W" for x in x_labels]
    y_labels = [f"{y:.0f}°N" if y >= 0 else f"{abs(y):.0f}°S" for y in y_labels]
    
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    
    # 修改文本位置到左上角，使用相对坐标
    ax.text(0.02, 0.12, label, 
            fontsize=8, 
            color='orange',
            transform=ax.transAxes,  # 使用相对坐标系统
            verticalalignment='bottom'
    )

    _ = ax.contourf(sst, levels=levels, cmap=COLOR_MAP_SST)
    ax.contour(sst, colors='black', alpha=0.5, linewidths=0.2, linestyles='--', levels=30)
    contour_lines = ax.contour(sst, colors='black', linewidths=0.5)
    
    ax.clabel(contour_lines, inline=True, colors='black', fontsize=5, fmt='%d', manual=False)
    
    return _

# %% 主函数

# 每个区域独立模型
def train_transformer_models():
    import torch
    from lightning import Trainer # type: ignore
    from lightning.pytorch.callbacks.early_stopping import EarlyStopping # type: ignore

    from src.config.params import Areas
    from src.models.Transformer import SSTTransformer

    models = []
    test_ = []
    el_stop = EarlyStopping(monitor='train_loss', patience=20, min_delta=0.05)

    for area in Areas:
        model = SSTTransformer()
        
        train_dataloader, val_dataloader, test_dataloader = split_data(area)

        trainer = Trainer(max_epochs=150, enable_checkpointing=False)

        trainer.fit(model, train_dataloaders=train_dataloader)

        test_.append(test_dataloader)
        models.append(model)

    return models, test_

# 所有区域共享模型
def train_transformer_model():
    from lightning import Trainer # type: ignore
    from lightning.pytorch.callbacks.early_stopping import EarlyStopping # type: ignore

    from src.config.params import Areas
    from src.models.Transformer import SSTTransformer

    model = SSTTransformer()
    test_ = []
    el_stop = EarlyStopping(monitor='train_loss', patience=50, min_delta=0.1, strict=False)

    for area, i in zip(Areas, range(len(Areas))):
        train_dataloader, val_dataloader, test_dataloader = split_data(area)
        
        # epoch 逐渐减少
        epoch = 20 - i * 5 if i < 3 else 20

        trainer = Trainer(max_epochs=epoch, auto_lr_find=True, enable_checkpointing=False)

        trainer.fit(model, train_dataloaders=train_dataloader)

        test_.append(test_dataloader)

    return model, test_
