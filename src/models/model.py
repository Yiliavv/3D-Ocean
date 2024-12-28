# For models

import gsw
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from joblib import dump, load
from torch import nn, optim, mean, sqrt, tensor, Tensor

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
    input_set = np.column_stack((input_set[0], input_set[1]))
    print(f"input_set shape: {input_set.shape}")
    # Reshape to 2D array for sklearn
    output_set = np.array(output_set)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(input_set, output_set, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.33, random_state=42)

    if load_model(MODEL_SAVE_PATH + "/random_forest_model.joblib"):
        return load_model(MODEL_SAVE_PATH + "/random_forest_model.joblib"), X_test, y_test

    # Initialize the model
    model = RandomForestRegressor(n_estimators=300, random_state=42, max_features=5, verbose=True)

    # Train the model
    model.fit(X_train, y_train)

    # save_model(model, MODEL_SAVE_PATH + "/random_forest_model.joblib")

    # Evaluate the model
    score = model.score(X_val, y_val)
    print(f"Model R^2 score: {score}")

    return model, X_val, y_val


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


import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return 1 - _ssim(img1, img2, window, window_size, channel, size_average)





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