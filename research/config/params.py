# Config for researches.
# 日志级别
from enum import Enum, auto


class Level(Enum):
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

    def __lt__(self, other):
        if isinstance(other, Level):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Level):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Level):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Level):
            return self.value >= other.value
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Level):
            return self.value == other.value
        return NotImplemented


LOG_LEVEL = Level.DEBUG

# 数据基础路径
BASE_CDAC_DATA_PATH = "B:/workspace/Argo_data/Argo_v3_core"
BASE_BOA_ARGO_DATA_PATH = "B:/workspace/Argo_data/ArgoDataset"

# 模型保存路径
MODEL_SAVE_PATH = "B:/workspace/tensorflow/research/output"

# 太平洋区域经纬度范围
LON_RANGE = [80, 140]
LAT_RANGE = [-40, 40]
