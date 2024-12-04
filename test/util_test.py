import unittest

from torch.utils.data import DataLoader
from src.config.params import BASE_CDAC_DATA_PATH
from src.utils.dataset import Argo3DTemperatureDataset, ERA5SstDataset


class TestUtil(unittest.TestCase):
    def test_import_era5_sst(self):
        print(BASE_CDAC_DATA_PATH)


class TestDataSet(unittest.TestCase):
    def test_era5_sst_dataset(self):
        data_set = ERA5SstDataset(width=15, step=1, lon=[60, 80], lat=[160, 180])
        print(data_set.shape)
        data_loader = DataLoader(data_set)
        loader_itr = iter(data_loader)

        print(len(data_set))

        while True:
            try:
                train_data, train_label = next(loader_itr)
                print(train_data.shape)
            except StopIteration:
                break

    def test_argo_3d_temperature_dataset(self):
        data_set = Argo3DTemperatureDataset(step=1, lat=[60, 80], lon=[160, 180], depth=[0, 10])
        data_loader = DataLoader(data_set)
        loader_itr = iter(data_loader)

        print(len(data_set))

        while True:
            try:
                train_data, train_label = next(loader_itr)
                print(train_data.shape)
            except StopIteration:
                break
