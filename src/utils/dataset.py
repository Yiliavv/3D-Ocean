import os
import numpy as np
from netCDF4 import Dataset

from src.config.params import BASE_BOA_ARGO_DATA_PATH
from src.utils.log import Log

ArgoDataset_3Dim_Sequence = None


def load_3Dim_Sequence():
    dir_name = BASE_BOA_ARGO_DATA_PATH
    TAG = "3Dim_Sequence"

    files = []

    with os.scandir(dir_name) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".nc"):
                files.append(entry.name)
    files = files[:240]
    total_size = len(files)
    Log.d(TAG, "total files: ", total_size)

    print(total_size)

    # init 3Dim_Sequence
    global ArgoDataset_3Dim_Sequence
    if ArgoDataset_3Dim_Sequence is not None: return
    ArgoDataset_3Dim_Sequence = np.empty((total_size,2, 20, 20), dtype=float)

    lat_range = [60, 80]
    lon_range = [160, 180]

    for file in files:
        month_file = dir_name + "/" + file
        nc_file = Dataset(month_file, "r")
        temperature = nc_file.variables["temp"][0][:2, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]]
        # 处理缺失值
        for i in range(2):
            deal_with_one_surface(temperature[i, :, :])
        # Log.d("temperature max: ", np.max(temperature[i, :, :]))

        ArgoDataset_3Dim_Sequence[files.index(file)] = temperature
        # Log.d(TAG, "3Dim_Sequence max: ", np.max(ArgoDataset_3Dim_Sequence))

    Log.d(TAG, "3Dim_Sequence shape: ", ArgoDataset_3Dim_Sequence.shape)
    return ArgoDataset_3Dim_Sequence


def load_surface_Sequence():
    Sequence_3Dim = load_3Dim_Sequence()

    surface_sequence = Sequence_3Dim[:, 0, :, :]

    for i in range(1):
        surface_sequence = np.concatenate((surface_sequence, Sequence_3Dim[:, i+1, :, :]), axis=0)

    Log.d("surface_Sequence shape: ", surface_sequence.shape)

    return surface_sequence


def deal_with_one_surface(surface):
    for i in range(20):
        for j in range(20):
            if surface[i, j] > 999 or np.isnan(surface[i, j]) or np.ma.is_masked(surface[i, j]):
                surface[i, j] = np.nanmean(surface)
