# Config for researches.

WORKSPACE_PATH = "/home/morisi/Workspace"
PROJECT_PATH = f"{WORKSPACE_PATH}/3D-Ocean"

# 数据基础路径
BASE_BOA_ARGO_DATA_PATH = f"{WORKSPACE_PATH}/ArgoDataset"
BASE_ERA5_DAILY_DATA_PATH = f"{WORKSPACE_PATH}/ERA5/daily_mean"
BASE_ERA5_MONTHLY_DATA_PATH = f"{WORKSPACE_PATH}/ERA5/monthly_mean"
BASE_NINO_DATA_PATH = f"{WORKSPACE_PATH}/ERA5"

# 模型保存路径
MODEL_SAVE_PATH = f"{PROJECT_PATH}/out/models"
# 误差保存路径
ERROR_SAVE_PATH = f"{PROJECT_PATH}/out/error"
# 预测保存路径
PREDICT_SAVE_PATH = f"{PROJECT_PATH}/out/sst"

# wandb
WANDB_PROJECT = "3-D Ocean"
WANDB_ENTITY = "yiliavei-zhejiang-university"

