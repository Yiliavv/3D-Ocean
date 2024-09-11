# MAIN - For all runner.
import numpy as np

from research.config.params import BASE_BOA_ARGO_DATA_PATH
from research.log import Log
from research.model import (train_single_parameter_model_for_linear_regression,
                            train_parameter_model_for_random_forest, train_single_parameter_model_for_gpr,
                            model_error)
from research.plot import plot_temperature_profile_compare
from research.util import import_argo_ocean_variables, import_ear5_sst, construct_argo_training_set

# 本文主要观测中国东海区域的Argo浮标，构建该区域的海洋三维温度场

# 读取CDAC数据

# 读取Argo数据
month_file = BASE_BOA_ARGO_DATA_PATH + "/BOA_Argo_2023_01.nc"
Log.i("开始读取Argo数据 ...")
temperatures, lon, lat, ild, mld, cmld = import_argo_ocean_variables(month_file)  # 月平均数据

argo_temp = np.transpose(temperatures, (1, 0, 2))

# 23年1月的海表温度
argo_sst = np.transpose(temperatures[:, :, 0], (1, 0))
print("Argo SST shape: ", argo_sst.shape)

input_data, output_data = construct_argo_training_set(temperatures, lon, lat, mld)

print("Input data shape: ", len(input_data))
print("Output data shape: ", output_data.shape)

model = train_parameter_model_for_random_forest(input_data, output_data)

