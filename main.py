# MAIN - For all runner.
from research.config.params import BASE_CDAC_DATA_PATH, BASE_BOA_ARGO_DATA_PATH
from research.log import Log
from research.util import resource_monthly_data, import_argo_ocean_variables, import_ear5_sst

# 本文主要观测中国东海区域的Argo浮标，构建该区域的海洋三维温度场

# 读取CDAC数据

# 读取Argo数据
month_file = BASE_BOA_ARGO_DATA_PATH + "/BOA_Argo_2023_06.nc"
Log.i("开始读取Argo数据 ...")
temperatures, lon, lat, ild, mld, cmld = import_argo_ocean_variables(month_file)  # 月平均数据

# 读取EAR5数据
ear_data_file = "B:/workspace/ERA5/Download data.nc"  # 1998.1 ~ 2023.1
sst_ear = import_ear5_sst(ear_data_file)


