# MAIN - For all runner.
import numpy as np

from research.config.params import BASE_BOA_ARGO_DATA_PATH
from research.log import Log
from research.model import (train_parameter_model_for_random_forest, train_parameter_model_for_lstm)
from research.plot import plot_sst_distribution, \
    plot_compared_profile_for_predicted, plot_profile_for_predicted_in_lat, \
    plot_compared_profile_for_predicted_with_different
from research.util import import_argo_ocean_variables, construct_argo_training_set, resource_argo_monthly_data

# 本文主要观测西太平洋区域的Argo浮标，构建该区域的海洋三维温度场

# 读取CDAC数据

# --------------------------- 读取单月数据 ------------------------
month_file = BASE_BOA_ARGO_DATA_PATH + "/BOA_Argo_2024_03.nc"
Log.i("开始读取Argo数据 ...")
temperature, lon, lat, ild, mld, cmld = import_argo_ocean_variables(month_file)
Log.d("3月数据读取完成: ", temperature.shape)
# 3月海表温度
march_sst = temperature[160:180, 60:80, 0]
# 3月剖面温度
march_profile = temperature[160:180, 60:80, :]

# ---------------------------------------------------------------

# --------------------------- 读取所有数据，生成训练集 ------------------------
datas = resource_argo_monthly_data(BASE_BOA_ARGO_DATA_PATH)
Log.d("数据读取完成: ")

# 生成训练集
input_data, output_data = construct_argo_training_set(datas)
Log.d("训练集生成完成: ", len(input_data), output_data.shape)

# 训练模型
model, x_test, y_test = train_parameter_model_for_random_forest(input_data, output_data)
# model_lstm, x_test_lstm, y_test_lstm = train_parameter_model_for_lstm(input_data, output_data)

# ------------------------------------------------------------------------

# ------------------------------------ 预测海温数据 ------------------------
pre_month_file = BASE_BOA_ARGO_DATA_PATH + "/BOA_Argo_2024_03.nc"
Log.i("开始读取预测Argo 海表数据 ...")
pre_temp, pre_lon, pre_lat, pre_ild, pre_mld, pre_cmld = import_argo_ocean_variables(month_file)

# pre_sst = pre_temp[160:180, 60:80, 0].reshape(400, -1).reshape(-1)
# Log.d("预测海表: ", pre_sst)
pre_stations = np.array([tuple([i, j]) for i in np.arange(160, 180) for j in np.arange(-19, 1)])
pre_years = np.array([2024] * len(pre_stations))
pre_months = np.array([3] * len(pre_stations))
pres_input = np.column_stack((pre_stations, pre_years, pre_months))
Log.i("预测的输入：", pres_input)

pres_result = np.empty((len(pres_input), 58), dtype=float)

for i in range(len(pres_input)):
    pre_y = model.predict(pres_input[i].reshape(1, -1))
    pres_result[i] = np.array(pre_y[0])

# 去除异常数据
pres_result[pres_result > 50] = np.nan
pres_sst = pres_result[:, 10].reshape(20, 20)

# plot_sst_distribution(pres_sst, "2024 March SST Distribution")

# plot_profile_for_predicted_in_lat(pres_result)

plot_compared_profile_for_predicted(march_profile, pres_result)
# ------------------------------------------------------------------------


