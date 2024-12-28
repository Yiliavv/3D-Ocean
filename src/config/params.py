# Config for researches.

# 数据基础路径
BASE_CDAC_DATA_PATH = "B:/workspace/Argo_data/Argo_v3_core"
BASE_BOA_ARGO_DATA_PATH = "B:/ArgoDataset"
BASE_ERA5_DATA_PATH = "B:/era5"

# 临时文件夹
TEMP_FILE = "B:/era5/temp"

# 模型保存路径
MODEL_SAVE_PATH = "B:/workspace/tensorflow/output/models"

# 太平洋区域经纬度范围
LON_RANGE = [80, 140]
LAT_RANGE = [-40, 40]

# 研究区域
Areas = [
    {
        'title': 'a)',
        'lon': [-160, -140],
        'lat': [20, 40]
    },
    {
        'title': 'b)',
        'lon': [-140, -120],
        'lat': [-35, -15]
    },
    {
        'title': 'c)',
        'lon': [145, 165],
        'lat': [10, 30]
    },
    {
        'title': 'd)',
        'lon': [-40, -20],
        'lat': [20, 40]
    },
    {
        'title': 'e)',
        'lon': [70, 90],
        'lat': [-20, 0]
    }
]
