import unittest
from research.config.params import BASE_CDAC_DATA_PATH
from research.dataset import load_3Dim_Sequence
from research.util import read_data_file, resource_monthly_data
from research.config.params import LOG_LEVEL, Level


class TestUtil(unittest.TestCase):

    def test_read_single_file(self):
        filename = BASE_CDAC_DATA_PATH + "\\199707\\13857_001.dat"
        [eng, pos, data] = read_data_file(filename)
        print(data['temp'])
        print(pos)
        self.assertNotEqual(len(data), 0)

    def test_source_monthly_data(self):
        filename = BASE_CDAC_DATA_PATH + "\\199707"
        daily_data = resource_monthly_data(filename)
        self.assertNotEqual(len(daily_data), 0)

    def test_load_3Dim_Sequence(self):
        load_3Dim_Sequence()
        pass

    def test_log(self):
        self.assertGreater(Level.INFO, Level.DEBUG)
        pass