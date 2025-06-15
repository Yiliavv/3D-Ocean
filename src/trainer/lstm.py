# %% 导入库
import sys

sys.path.append('B://workspace/tensorflow/')

# 定义参数
STEP = 1 # 1 天为一个步长
OFFSET = 0 # 2004-01-01 

# %% 训练器
import torch
import arrow
from lightning import Trainer
from src.config.area import Area
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from src.dataset.ERA5 import ERA5SSTMonthlyDataset
from models.ConvLSTM import ConvLSTM
from src.utils.mio import ModelParams, DatasetParams, TrainOutput, write_m

class LSTMTrainer:
    """
    LSTM 训练器
    训练数据集用 ERA5 月度 SST 数据，模型用 ConvLSTM
    
    参数:
        title: str, 标题
        area: Area, 区域
        seq_len: int = 2, 序列长度
        offset: int = 0, 时间偏移
        resolution: float = 1, 分辨率
        epochs: int = 300, 训练轮数
        batch_size: int = 20, 批量大小
        model_path: str = None, 模型保存路径, 如果为 None, 则不保存模型
        pre_model: ConvLSTM = None, 预训练模型
        **kwargs: ConvLSTM 模型参数
        
    主要的模型参数:
        input_dim: 输入通道数
        hidden_dim: 隐藏层通道数
        kernel_size: 卷积核大小
        num_layers: LSTM层数
        bias: 是否使用偏置
    """
    def __init__(self,
                 title: str,
                 area: Area,
                 seq_len: int = 2,
                 offset: int = 0,
                 resolution: float = 1,
                 epochs: int = 300,
                 batch_size: int = 20,
                 model_path: str = None,
                 pre_model: ConvLSTM = None,
                 **kwargs):
        
        # 初始化
        self.title = title
        self.area = area
        self.seq_len = seq_len
        self.offset = offset
        self.resolution = resolution
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_path = model_path
        self.model_params = kwargs
        
        self.model = pre_model or ConvLSTM(**kwargs)
        
        # 训练结束后的数据
        self.val_loader = None
        self.train_loader = None
        
    def split(self, dataset):
        train_set, val_set = random_split(dataset, [0.8, 0.2])
        
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def output(self):
        model_params = ModelParams(
            model='ConvLSTM',
            m_type='LSTM',
            model_path=self.model_path,
            params=self.model_params,
        )
        
        dataset_params = DatasetParams(
            dataset='ERA5-Monthly-SST',
            range=[self.area.lon, self.area.lat],
            resolution=self.resolution,
            start_time=arrow.get(2004, 1, 1).format('YYYY-MM-DD'),
            end_time=arrow.get(2024, 12, 31).format('YYYY-MM-DD'),
        )
        
        train_output = TrainOutput(
            epoch=self.epochs,
            val_loss=self.model.val_loss,
            batch_size=self.batch_size,
            train_loss=self.model.train_loss,
            m_params=model_params,
            d_params=dataset_params,
        )
        
        write_m(train_output, self.title)
        
    def save_model(self, model_path: str):
        self.model_path = model_path
        torch.save(self.model, model_path)
        
    def train(self):
        lon = self.area.lon
        lat = self.area.lat
        
        dataset = ERA5SSTMonthlyDataset(
            width=self.seq_len,
            offset=self.offset,
            lon=lon,
            lat=lat,
            resolution=self.resolution
        )

        self.train_loader, self.val_loader = self.split(dataset)

        trainer = Trainer(
            max_epochs=self.epochs,
            enable_checkpointing=False,
            accelerator="gpu",
        )

        trainer.fit(self.model, train_dataloaders=self.train_loader, val_dataloaders=self.val_loader)

        if self.model_path:
            self.save_model(self.model_path)
            return self.model

        self.output()
        
        return self.model
