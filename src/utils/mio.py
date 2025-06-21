# 模型训练数据工具
# 
# 用特定的格式保存模型训练数据，方便后续汇总


import os
import json
import arrow

from dataclasses import dataclass, asdict

@dataclass
class ModelParams:
    # 模型名称
    model: str
    # 模型类型
    m_type: str
    # 模型保存路径
    model_path: str
    # 模型参数
    params: dict


@dataclass
class DatasetParams:
    # 数据集名称
    dataset: str
    # 经纬度范围
    range: list[list[float]]
    # 分辨率
    resolution: float
    # 开始的时间
    start_time: str
    # 结束的时间
    end_time: str


@dataclass
class TrainOutput:
    epoch: int
    batch_size: int
    val_loss: list[float]
    train_loss: list[float]
    m_params: ModelParams
    d_params: DatasetParams


BASE_DIR = "B:/workspace/tensorflow/train_output"

# 写入训练结果
def write_m(train_output: TrainOutput, model_name: str, uid: str):
    # 创建模型目录
    model_dir = os.path.join(BASE_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"file_name: {uid}")
    
    data = json.dumps(asdict(train_output), indent=4)
    
    # 保存模型参数
    with open(os.path.join(model_dir, f"{uid}.json"), "w") as f:
        f.write(data)
