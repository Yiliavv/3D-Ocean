# For models

import gsw
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from joblib import dump, load
from torch import nn, optim, mean, sqrt, tensor
import tensorflow as tf

from src.config.params import MODEL_SAVE_PATH
from src.utils.log import Log


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


def calculate_seawater_density(temperature):
    """
    使用温度（C）计算海水密度（kg/m^3）
    """

    # Calculate density using the Gibbs SeaWater (GSW) Oceanographic Toolbox
    density = gsw.density.rho(temperature)

    return density


def calculate_angle_tan(g1, g2):
    """
    计算两个梯度的角度的正切值
    :param g1:
    :param g2:
    :return:
    """
    return abs(g2 - g1) / (1 + g1 * g2)


# -------------------------- 机器学习回归模型（预测）  --------------------------


def train_parameter_model_for_random_forest(input_set, output_set):
    """
    训练海洋随机森林模型
    """

    # Convert input_set and output_set to numpy arrays
    input_set = np.column_stack((input_set[0], input_set[1], input_set[2]))
    print(f"input_set shape: {input_set.shape}")
    # Reshape to 2D array for sklearn
    output_set = np.array(output_set)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(input_set, output_set, test_size=0.2, random_state=42)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    if load_model(MODEL_SAVE_PATH + "/random_forest_model.joblib"):
        return load_model(MODEL_SAVE_PATH + "/random_forest_model.joblib"), X_test, y_test

    # Initialize the model
    model = RandomForestRegressor(n_estimators=300, random_state=42, max_features=5, verbose=True)

    # Train the model
    model.fit(X_train, y_train)

    # save_model(model, MODEL_SAVE_PATH + "/random_forest_model.joblib")

    # Evaluate the model
    score = model.score(X_test, y_test)
    print(f"Model R^2 score: {score}")

    return model, X_test, y_test


# -------------------------- 模型评估 --------------------------

def model_error(y_true, y_predi):
    """
    计算模型的误差
    """
    mbe = mean_absolute_error(y_true, y_predi)
    mse = mean_squared_error(y_true, y_predi)

    return mbe, mse


def profile_error(origin, predict):
    """
    计算剖面的误差
    """
    origin = np.array(origin)
    predict = np.array(predict)


def ssim_loss(y_true, y_pred):
    y_t = y_true.detach().cpu()
    y_p = y_pred.detach().cpu()
    loss = 1 - tf.reduce_mean(tf.image.ssim(y_t, y_p, max_val=1.0))
    return loss.numpy()


# -------------------------- 模型工具 --------------------------

def train(model, x_train, label_train, epoches: int = 10, loss_function=None, optimizer=None):
    """
    训练模型
    """
    if loss_function is None:
        loss_function = nn.MSELoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epoches):
        optimizer.zero_grad()
        output = model(x_train)
        loss = loss_function(output, label_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss {loss.item()}")


def save_model(model, path):
    """
    保存模型
    """
    # Save the model use joblib
    Log.i("保存模型到：", path)
    dump(model, path)


def load_model(path):
    """
    加载模型
    """
    try:
        with open(path) as f:
            Log.i("加载模型：", path)
            return load(path)
    except FileNotFoundError:
        return None