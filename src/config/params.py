# Config for researches.
from src.config.area import Area


# 数据基础路径
BASE_CDAC_DATA_PATH = "B:/workspace/Argo_data/Argo_v3_core"
BASE_BOA_ARGO_DATA_PATH = "B:/ArgoDataset"
BASE_ERA5_DATA_PATH = "B:/era5"
BASE_ERA5_DAILY_DATA_PATH = "B:/era5/daily_mean"
BASE_ERA5_MONTHLY_DATA_PATH = "B:/era5/monthly_mean_at"

BASE_CSV_PATH = "B:/workspace/tensorflow/output/csv"
BASE_PROFILE_PATH = "B:/workspace/tensorflow/output/profile"

# 模型保存路径
MODEL_SAVE_PATH = "B:/workspace/tensorflow/output/models"

# 误差保存路径
ERROR_SAVE_PATH = "B:/workspace/tensorflow/output/error"

# 预测保存路径
PREDICT_SAVE_PATH = "B:/workspace/tensorflow/output/sst"

# 研究区域
Areas = [
    Area(title="Area a", lon=[-150, -130], lat=[25, 45], description="北太平洋东部海域，西经150-130，北纬25-45"),
    Area(title="Area b", lon=[-140, -120], lat=[-40, -20], description="南太平洋东部海域，西经140-120，南纬40-20"),
    Area(title="Area c", lon=[70, 90], lat=[-55, -35], description="印度洋中部海域，东经70-90，南纬55-35"),
    Area(title="Area d", lon=[-40, -20], lat=[30, 50], description="北大西洋东部海域，西经40-20，北纬30-50"),
]


