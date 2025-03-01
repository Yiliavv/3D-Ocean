# Config for researches.

# 数据基础路径
BASE_CDAC_DATA_PATH = "B:/workspace/Argo_data/Argo_v3_core"
BASE_BOA_ARGO_DATA_PATH = "B:/ArgoDataset"
BASE_ERA5_DATA_PATH = "B:/era5"
BASE_ERA5_DAILY_DATA_PATH = "B:/era5/daily_mean"

# 模型保存路径
MODEL_SAVE_PATH = "B:/workspace/tensorflow/output/models"

# 研究区域
Areas = [
    {
        'title': 'a',
        'lon': [-160, -140],
        'lat': [20, 40],
        'description': '北太平洋东部海域，西经160-140，北纬20-40'
    },
    {
        'title': 'b',
        'lon': [-140, -120],
        'lat': [-35, -15],
        'description': '南太平洋东部海域，西经140-120，南纬35-15'
    },
    {
        'title': 'c',
        'lon': [70, 90],
        'lat': [-5, 15],
        'description': '印度洋中部海域，东经70-90，北纬0-20'
    },
    {
        'title': 'd',
        'lon': [-40, -20],
        'lat': [20, 40],
        'description': '北大西洋东部海域，西经40-20，北纬20-40'
    },
]
