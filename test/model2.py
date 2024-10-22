import os
import numpy as np
import torch
import tensorflow as tf

from src.utils.log import Log
from src.utils.plot import plot_sst_distribution_compare

import matplotlib.pyplot as plt

os.environ['KERAS_BACKEND'] = 'torch'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
Log.d("CUDA_VISIBLE_DEVICES: ", torch.cuda.is_available())
torch.cuda.set_device(0)

import keras

TAG = "Model"

# config dataset
from src.utils.dataset import load_surface_Sequence

time_step = 12

ArgoDataset_surface_Sequence = np.expand_dims(load_surface_Sequence(), axis=-1)
ArgoDataset_surface_Sequence = ArgoDataset_surface_Sequence.reshape(int((ArgoDataset_surface_Sequence.shape[0] / time_step)),
                                                                    time_step, 20, 20, 1)

Log.d(TAG, "ArgoDataset_3Dim_Sequence shape: ", ArgoDataset_surface_Sequence.shape)

maxTemp = np.max(ArgoDataset_surface_Sequence)
minTemp = np.min(ArgoDataset_surface_Sequence)

Log.d("ArgoDataset_surface_Sequence max: ", maxTemp)
Log.d("ArgoDataset_surface_Sequence min: ", minTemp)

indexes = np.arange(ArgoDataset_surface_Sequence.shape[0])

train_index = indexes[: int(len(indexes) * 0.8)]
test_index = indexes[int(len(indexes) * 0.8):]

train_dataset = ArgoDataset_surface_Sequence[train_index] / 50
test_dataset = ArgoDataset_surface_Sequence[test_index] / 50


# time offset
def created_shifted(data):
    x_ = data[:, 0:data.shape[1] - 1, :, :]
    y_ = data[:, 1:data.shape[1], :, :]
    return x_, y_


x_train, y_train = created_shifted(train_dataset)
x_test, y_test = created_shifted(test_dataset)

Log.d(TAG, "train data: ", x_train.shape, y_train.shape)
Log.d(TAG, "test data: ", x_test.shape, y_test.shape)

# config Model
inp = keras.layers.Input(shape=(x_train.shape[1:]))

# We will construct 3 `ConvLSTM2D` layers with batch normalization,
# followed by a `Conv3D` layer for the spatiotemporal outputs.
x = keras.layers.ConvLSTM2D(
    filters=20,
    kernel_size=(5, 5),
    padding="same",
    return_sequences=True,
    activation="relu",
)(inp)
x = keras.layers.ConvLSTM2D(
    filters=20,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = keras.layers.ConvLSTM2D(
    filters=20,
    kernel_size=(1, 1),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = keras.layers.Conv3D(
    filters=1, kernel_size=(2, 2, 2), activation="sigmoid", padding="same"
)(x)


def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


# model = keras.models.load_model("model.keras")
model = None
if model is None:
    model = keras.models.Model(inp, x)
    model.compile(loss=ssim_loss,
                  optimizer=keras.optimizers.Adam(learning_rate=0.001))
    model.summary()

    epochs = 300
    batch_size = 2

    # train model
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

    # 绘制训练 & 验证的损失值
    plt.title('SSIM')
    plt.plot(history.history['loss'])

    model.save("model.keras", overwrite=True)

# prediction
surfaces = x_test

maxSurfaceTemp = np.nanmax(surfaces)
minSurfaceTemp = np.nanmin(surfaces)

Log.d(TAG, "maxSurfaceTemp: ", maxSurfaceTemp)
Log.d(TAG, "minSurfaceTemp: ", minSurfaceTemp)

new_surfaces = model.predict(surfaces, batch_size=3)

Log.d(TAG, "surfaces shape: ", surfaces.shape)
Log.d(TAG, "new_surfaces shape: ", new_surfaces.shape)

for i in range(2):
    prediction = new_surfaces[2, i, :, :, 0] * 50
    origin = surfaces[2, i, :, :, 0] * 50
    Log.d(TAG, "prediction sst: ", prediction)
    Log.d(TAG, "origin sst: ", origin)
    r2 = np.corrcoef(prediction.flatten(), origin.flatten())
    Log.d(TAG, "r2: ", r2)
    plot_sst_distribution_compare(prediction, origin)
