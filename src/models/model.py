# For data handler, to data model.

import gsw, sys, os
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from joblib import dump, load

sys.path.insert(0, '../')

from config.params import LAT_RANGE, LON_RANGE, MODEL_SAVE_PATH
from utils.log import Log


# -------------------------- CDAC 数据处理 --------------------------

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


def train_parameter_model_for_lstm(input_set, output_set):
    """
    训练海洋LSTM模型

    :param output_set: 输入数据集
    :param input_set: 输出数据集
    """

    Log.d("数据集大小：", len(input_set))

    # Convert input_set and output_set to numpy arrays
    input_set = np.column_stack((input_set[0], input_set[1], input_set[2]))
    print(f"input_set shape: {input_set.shape}")
    # Reshape to 2D array for sklearn
    output_set = np.array(output_set)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(input_set, output_set, test_size=0.2, random_state=42)

    Log.i("X_train: ", X_train.shape)

    window_1 = WindowGenerator(input_width=12, label_width=1, shift=1, train_data=X_train, test_data=X_test)
    Log.i("WindowGenerator 1: ", window_1)

    split_input, split_output = window_1.split_window(X_train)
    Log.d('All shapes are: (time, features)')
    Log.d(f'Window shape: {window_1}')
    Log.d(f'Split input shape: {split_input.shape}')
    Log.d(f'Split output shape: {split_output.shape}')

    return None, None, None


class WindowGenerator:
    def __init__(self, input_width, label_width, shift, train_data, test_data, columns=None):
        # 训练数据集
        self.train_data = train_data
        self.test_data = test_data
        # 单列
        self.columns = columns
        # window parameters
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slices = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slices]

        self.label_start = self.total_window_size - self.label_width
        self.label_slices = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.label_slices]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
        ])

    @property
    def train(self):
        return self.make_dataset(self.train_data)

    @property
    def test(self):
        return self.make_dataset(self.test_data)

    

    def split_window(self, features):
        inputs = features[self.input_indices, :]
        labels = features[self.label_indices, :]

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data, targets=None, sequence_length=self.total_window_size, sequence_stride=1, shuffle=True,
        )
        ds = ds.map(self.split_window)

        return ds


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


# -------------------------- 模型工具 --------------------------

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
