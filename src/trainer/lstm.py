import sys
sys.path.append('B://workspace/tensorflow/')

# 训练 Conv-LSTM 模型。
# 该模型通过同一个月的前 14 天的 SST 数据预测未来 1 天的 SST 数据。

# 导入数据集
from torch.utils.data import DataLoader

# 定义参数
OFFSET = 23376 # 2004-01-01
WIDTH = 15 # 15 天为一个窗口，获取前 14 天的数据，预测第 15 天的数据
STEP = 1 # 1 天为一个步长

# %% 工具函数
from numpy import array
from torch.utils.data import random_split

from src.dataset.ERA5 import ERA5SSTDataset

def get_lon(lon):
    lon_s = 360 + lon[0] if lon[0] <= 0 else lon[0]
    lon_e = 360 + lon[1] if lon[1] <= 0 else lon[1]
    
    print(lon_s, lon_e)
    
    return [lon_s, lon_e]

def split_data(area):
    lon = array(get_lon(area['lon']))
    lat = array(area['lat']) + 90

    dataset = ERA5SSTDataset(WIDTH, STEP, OFFSET, lon, lat)
    
    train_data_set, val_data_set, test_data_set = random_split(dataset, [0.7, 0.2, 0.1])

    train_dataloader = DataLoader(train_data_set, batch_size=20, shuffle=False)
    val_dataloader = DataLoader(val_data_set, batch_size=20, shuffle=False)
    test_dataloader = DataLoader(test_data_set, batch_size=20, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader

def get_sst(): 
    lon = array([-180, 180]) + 180
    lat = array([-90, 90]) + 90

    dataset = ERA5SSTDataset(WIDTH, STEP, OFFSET, lon, lat)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print(dataset.getTime(dataset.current))

    fore_, last_ = next(iter(loader))
    print(fore_.shape)

    return last_[0, :, :]

# %% 主函数

def train_lstm_model():
    from lightning import Trainer # type: ignore
    from lightning.pytorch.callbacks.early_stopping import EarlyStopping # type: ignore

    from src.models.LSTM import ConvLSTM
    from src.config.params import Areas

    models = []
    test_ = []

    for area in Areas: 
        model = ConvLSTM(1, 10, kernel_size=(5,5), num_layers=5)
        el_stop = EarlyStopping(monitor='loss', patience=20, min_delta=0.05)
        
        train_dataloader, val_dataloader, test_dataloader = split_data(area)
        trainer = Trainer(max_epochs=30, limit_train_batches=200, enable_checkpointing=False, callbacks=[el_stop])

        trainer.fit(model, train_dataloaders=train_dataloader)
        
        models.append(model)
        test_.append(test_dataloader)
    
    return models, test_
