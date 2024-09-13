# MAIN - For all runner.
import numpy as np

from research.config.params import BASE_BOA_ARGO_DATA_PATH
from research.log import Log
from research.model import (train_parameter_model_for_random_forest)
from research.plot import plot_temperature_profile_compare
from research.util import import_argo_ocean_variables, construct_argo_training_set

# 本文主要观测西太平洋区域的Argo浮标，构建该区域的海洋三维温度场

# 读取CDAC数据

# 读取Argo数据
month_file = BASE_BOA_ARGO_DATA_PATH + "/BOA_Argo_2023_01.nc"
Log.i("开始读取Argo数据 ...")
temperatures, lon, lat, ild, mld, cmld = import_argo_ocean_variables(month_file)  # 月平均数据

print("Temperature shape: ", temperatures.shape)

input_data, output_data = construct_argo_training_set(temperatures, lon, lat, mld)

print("Input data shape: ", len(input_data))
print("Output data shape: ", output_data.shape)

model = train_parameter_model_for_random_forest(input_data, output_data)

pre_y = model.predict(np.column_stack(([temperatures[136, 80, 0]], [[lon[136], lat[80]]], [mld[136, 80]])))

print(pre_y)

plot_temperature_profile_compare(temperatures[136, 80, :], pre_y[0])

