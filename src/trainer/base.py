import arrow
import uuid
from torch import save, load
from lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from src.config.params import Area
from src.utils.mio import DatasetParams, ModelParams, TrainOutput, write_m

class BaseTrainer:
    """
    训练器基类
    
    参数:
        title: str, 模型名称
        area: Area, 区域
        model_class: LightningModule, 模型类
        save_path: str, 保存路径
        pre_model: LightningModule, 预训练模型
        dataset_params: dict, 数据集参数
        trainer_params: dict, 训练参数
        model_params: dict, 模型参数
        
    dataset_params:
        seq_len: int, 序列长度
        offset: int, 偏移量
        resolution: float, 分辨率
        
    trainer_params:
        epochs: int, 训练轮数
        batch_size: int, 批量大小
        split_ratio: list, 训练集和验证集的分割比例

    """
    
    def __init__(self,
                 title: str,
                 area: Area,
                 model_class = None,
                 dataset_class = None,
                 save_path: str = None,
                 pre_model: LightningModule = None,
                 dataset_params: dict = {},
                 trainer_params: dict = {},
                 model_params: dict = {}):
        
        self.trainer_uid = str(uuid.uuid4())

        self.title = title
        self.area = area
        
        # 工厂类型
        self.model_class = model_class
        self.dataset_class = dataset_class

        # 参数
        self.dataset_params = dataset_params
        self.trainer_params = trainer_params
        self.model_params = model_params
        
        # 保存路径和预训练模型
        self.save_path = save_path
        self.pre_model = pre_model
        
        # 模型
        self.model = None
        self.trained = False
    
    def split(self, dataset):
        split_ratio = self.trainer_params.get('split_ratio', [0.8, 0.2])
        batch_size = self.trainer_params.get('batch_size', 20)
        
        # override
        train_set, val_set = random_split(dataset, split_ratio)
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
        
    def train(self):
        lon = self.area.lon
        lat = self.area.lat
        
        dataset = self.dataset_class(
            lon=lon,
            lat=lat,
            **self.dataset_params
        )
        
        train_loader, val_loader = self.split(dataset)
        
        self.model = self.model_class(
            **self.model_params
        )
        
        epochs = self.trainer_params.get('epochs', 100)

        trainer = Trainer(
            max_epochs=epochs,
            accelerator="gpu",
            enable_checkpointing=False,
        )

        trainer.fit(self.model, train_loader, val_loader)

        self.trained = True
        self.output()

        if self.save_path:
            self.save()

        return self.model

    def predict(self, offset: int) -> tuple:
        
        """
        预测
        
        :param offset: 数据偏移量
        :return: 输入和预测输出
        """

        if not self.trained:
            self.model = load(self.save_path)
            self.trained = True
            
        if not self.model:
            raise ValueError('无已训练模型')
        
        dataset_params = {
            **self.dataset_params,
            'offset': offset,
        }
        
        pred_dataset = self.dataset_class(
            lon=self.area.lon,
            lat=self.area.lat,
            **dataset_params
        )
        
        pred_loader = DataLoader(pred_dataset, batch_size=1, shuffle=False)
        
        input, output = next(iter(pred_loader))
        
        pred_output = self.model(input)
        
        input = input.detach().numpy()
        output = output.detach().numpy()
        pred_output = pred_output.detach().numpy()
        
        return (input, output, pred_output)

    def save(self):
        save(self.model, self.save_path)

    def output(self):
        
        model_params = ModelParams(
            model=self.model_class.__name__,
            m_type=self.model_class.__name__,
            model_path=self.save_path,
            params=self.model_params,
        )
        
        offset = self.dataset_params.get('offset', 0)
        
        dataset_params = DatasetParams(
            dataset=self.dataset_class.__name__,
            range=[self.area.lon, self.area.lat],
            resolution=self.dataset_params.get('resolution', 1),
            start_time=arrow.get(2004, 1, 1).shift(months=offset).format('YYYY-MM-DD'),
            end_time=arrow.get(2024, 12, 31).format('YYYY-MM-DD'),
        )
        
        train_output = TrainOutput(
            epoch=self.trainer_params.get('epochs', 100),
            val_loss=self.model.val_loss if hasattr(self.model, 'val_loss') else [],
            batch_size=self.trainer_params.get('batch_size', 20),
            train_loss=self.model.train_loss if hasattr(self.model, 'train_loss') else [],
            m_params=model_params,
            d_params=dataset_params,
        )
        
        write_m(train_output, self.title, self.trainer_uid)
