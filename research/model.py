# For data handler, to data model.
import os

import gsw
import numpy as np
from onnxconverter_common import FloatTensorType
from skl2onnx import to_onnx
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from research.config.params import LAT_RANGE, LON_RANGE, MODEL_SAVE_PATH


# -------------------------- CADC 数据处理 --------------------------

def range_cdac_one_day_float_data(one_day_data):
    """
    将浮标数据约束在指定的经纬度范围内
    """
    ranged = []
    for float_data in one_day_data:
        pos = float_data['pos']
        if LAT_RANGE[0] <= pos['lat'] <= LAT_RANGE[1] and LON_RANGE[0] <= pos['lon'] <= LON_RANGE[1]:
            ranged.append(float_data)

    return ranged


# -------------------------- Argo 数据处理 --------------------------

def range_argo_mld_data(mld):
    """
    将Argo数据约束在指定的经纬度范围内
    """
    return mld[LON_RANGE[0] + 180:LON_RANGE[1] + 180, LAT_RANGE[0] + 80:LAT_RANGE[1] + 80]


# -------------------------- 数学方法 --------------------------

def linear_fit(x, y):
    """"
    线性拟合
    """
    x = np.array(x)
    y = np.array(y)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c


def calculate_seawater_density(temperature, salinity, pressure):
    """
    使用温度（C）、盐度（PSU）和压强（dbar）计算海水密度（kg/m^3）
    """

    # Calculate density using the Gibbs SeaWater (GSW) Oceanographic Toolbox
    density = gsw.density.rho(salinity, temperature, pressure)

    return density


def calculate_angle_tan(g1, g2):
    """
    计算两个梯度的角度的正切值
    :param g1:
    :param g2:
    :return:
    """
    return abs(g2 - g1) / (1 + g1 * g2)


# -------------------------- 机器学习模型 --------------------------

def train_single_parameter_model_for_linear_regression(input_set, output_set):
    """
    训练海洋单参数线性回归模型
    """

    # Convert input_set and output_set to numpy arrays
    input_set = np.array(input_set).reshape(-1, 1)  # Reshape to 2D array for sklearn
    output_set = np.array(output_set)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(input_set, output_set, test_size=0.2, random_state=42)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    # Initialize the model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    score = model.score(X_test, y_test)
    print(f"Model R^2 score: {score}")

    # Persist the model
    onx = to_onnx(model, initial_types=[('input', FloatTensorType([None, 1]))])
    with open(MODEL_SAVE_PATH + "/Linear.onnx", "wb") as f:
        f.write(onx.SerializeToString())

    return model


def train_parameter_model_for_random_forest(input_set, output_set):
    """
    训练海洋随机森林模型
    """

    # Convert input_set and output_set to numpy arrays
    input_set = np.array(input_set)  # Reshape to 2D array for sklearn
    output_set = np.array(output_set)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(input_set, output_set, test_size=0.2, random_state=42)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    score = model.score(X_test, y_test)
    print(f"Model R^2 score: {score}")

    #
    if os.path.exists(MODEL_SAVE_PATH + "/RandomForest.onnx"):
        return model

    onx = to_onnx(model, initial_types=[('input', FloatTensorType([None, 1]))])
    with open(MODEL_SAVE_PATH + "/RandomForest.onnx", "wb") as f:
        f.write(onx.SerializeToString())

    return model


def train_single_parameter_model_for_gpr(input_set, output_set):
    """
    训练海洋单参数支持向量机模型
    """
    input_set = np.array(input_set).reshape(-1, 1)  # Reshape to 2D array for sklearn
    output_set = np.array(output_set)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(input_set, output_set, test_size=0.2, random_state=42)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    # Initialize the model
    model = GaussianProcessRegressor()

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    score = model.score(X_test, y_test)
    print(f"ModelR ^2 score: {score}")

    if os.path.exists(MODEL_SAVE_PATH + "/GMM.onnx"):
        return model

    # Persist the model
    onx = to_onnx(model, initial_types=[('input', FloatTensorType([None, 1]))])
    with open(MODEL_SAVE_PATH + "/GMM.onnx", "wb") as f:
        f.write(onx.SerializeToString())

    return model


# -------------------------- 模型评估 --------------------------

def model_error(y_true, y_predi):
    """
    计算模型的误差
    """
    mbe = mean_absolute_error(y_true, y_predi)
    mse = mean_squared_error(y_true, y_predi)

    return mbe, mse
