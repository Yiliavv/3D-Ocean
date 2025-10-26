import os
import re
import numpy as np
import netCDF4 as nc
from datetime import datetime
import torch
import random
from functools import lru_cache

from src.utils.log import Log

# ---------------------------- CDAC Argo 数据导入 ----------------------------

# 数据文件列表 -- list all .dat file in the directory
def list_data_files(data_dir):
    files = []

    with os.scandir(data_dir) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith('.dat'):
                files.append(entry.name)

    return files


# 读取单个文件数据 -- reused from "read_data_from_single_profile.m"
def read_data_file(filename):
    eng = {}
    pos = {}
    data = {}

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    with open(filename, 'r') as f:
        lines = f.readlines()

    pres, pres_adj, pres_qc = [], [], []
    temp, temp_adj, temp_qc = [], [], []
    psal, psal_adj, psal_qc = [], [], []

    for line in lines:
        line_length = len(line)
        if line.startswith("     PLATFORM NUMBER"):
            eng['wmo'] = re.findall(r'\S+', line[28:line_length])[0]
        elif line.startswith("     CYCLE NUMBER"):
            eng['cycle'] = int(re.findall(r'\S+', line[28:line_length])[0])
        elif line.startswith("     DATE CREATION"):
            eng['date_creation'] = re.findall(r'\S+', line[28:line_length])[0]
        elif line.startswith("     DATE UPDATE"):
            eng['date_update'] = re.findall(r'\S+', line[28:line_length])[0]
        elif line.startswith("     PROJECT NAME"):
            if line_length <= 29:
                eng['proj_name'] = None
            else:
                eng['proj_name'] = re.findall(r'\S+', line[28:line_length])[0]
        elif line.startswith("     PI NAME"):
            eng['pi_name'] = re.findall(r'\S+', line[28:line_length])[0]
        elif line.startswith("     INSTRUMENT TYPE"):
            eng['inst_type'] = re.findall(r'\S+', line[28:line_length])[0]
        elif line.startswith("     FLOAT SERIAL NO"):
            eng['float_sn'] = re.findall(r'\S+', line[28:line_length])[0]
        elif line.startswith("     FIRMWARE VERSION"):
            eng['firmware'] = re.findall(r'\S+', line[28:line_length])[0]
        elif line.startswith("     WMO INSTRUMENT TYPE"):
            eng['wmo_inst'] = re.findall(r'\S+', line[28:line_length])[0]
        elif line.startswith("     TRANSMISSION SYSTEM"):
            eng['trans_sys'] = re.findall(r'\S+', line[28:line_length])[0]
        elif line.startswith("     POSITIONING SYSTEM"):
            eng['pos_sys'] = re.findall(r'\S+', line[28:line_length])[0]
        elif line.startswith("     SAMPLE DIRECTION"):
            eng['direction'] = re.findall(r'\S+', line[28:29])[0]
        elif line.startswith("     DATA MODE"):
            eng['data_mode'] = re.findall(r'\S+', line[28:29])[0]
        elif line.startswith("     JULIAN DAY"):
            pos['juld'] = float(re.findall(r'\S+', line[28:38])[0])
        elif line.startswith("     QC FLAG FOR DATE"):
            pos['juld_qc'] = int(re.findall(r'\S+', line[28:29])[0])
        elif line.startswith("     LATITUDE"):
            pos['lat'] = float(re.findall(r'\S+', line[28:line_length])[0])
        elif line.startswith("     LONGITUDE"):
            pos['lon'] = float(re.findall(r'\S+', line[28:line_length])[0])
        elif line.startswith("     QC FLAG FOR LOCATION"):
            pos['pos_qc'] = int(re.findall(r'\S+', line[28:29])[0])
        elif line.startswith("     COLUMN"):
            continue
        elif line.startswith("=========================================================================="):
            continue
        elif len(line) > 55:
            pres.append(float(line[0:7]))
            pres_adj.append(float(line[7:14]))
            pres_qc.append(int(line[14:17]))
            temp.append(float(line[17:26]))
            temp_adj.append(float(line[26:35]))
            temp_qc.append(int(line[35:38]))
            psal.append(float(line[38:47]))
            psal_adj.append(float(line[47:56]))
            psal_qc.append(int(line[56:59]))

    pos['lat'] = np.nan if pos.get('lat', -1000) < -999 else pos['lat']
    pos['lon'] = np.nan if pos.get('lon', -1000) < -999 else pos['lon']

    pres = [np.nan if p < -999 else p for p in pres]
    pres_adj = [np.nan if p < -999 else p for p in pres_adj]
    temp = [np.nan if t < -99 else t for t in temp]
    temp_adj = [np.nan if t < -99 else t for t in temp_adj]
    psal = [np.nan if s < -99 else s for s in psal]
    psal_adj = [np.nan if s < -99 else s for s in psal_adj]

    data['pres'] = pres
    data['pres_adj'] = pres_adj
    data['pres_qc'] = pres_qc
    data['temp'] = temp
    data['temp_adj'] = temp_adj
    data['temp_qc'] = temp_qc
    data['psal'] = psal
    data['psal_adj'] = psal_adj
    data['psal_qc'] = psal_qc

    return eng, pos, data


# 读取单个浮漂数据 -- reused from "read_data_from_float.m"
def read_data_from_float(wmo_dir, wmo):
    eng = {}
    pos = {}
    data = {}

    if not os.path.isdir(wmo_dir):
        raise FileNotFoundError(f"Cannot find directory: {wmo_dir}")

    files = [f for f in os.listdir(wmo_dir) if f.startswith(wmo) and f.endswith('.dat')]
    m = len(files)
    if m < 1:
        raise FileNotFoundError(f"Cannot find any files in {wmo_dir}")

    PRES = np.full((100, m), np.nan)
    PRES_ADJ = np.full((100, m), np.nan)
    PRES_QC = np.full((100, m), np.nan)
    TEMP = np.full((100, m), np.nan)
    TEMP_ADJ = np.full((100, m), np.nan)
    TEMP_QC = np.full((100, m), np.nan)
    PSAL = np.full((100, m), np.nan)
    PSAL_ADJ = np.full((100, m), np.nan)
    PSAL_QC = np.full((100, m), np.nan)

    for i, file in enumerate(files):
        filename = os.path.join(wmo_dir, file)
        with open(filename, 'r') as f:
            lines = f.readlines()

        pres, pres_adj, pres_qc = [], [], []
        temp, temp_adj, temp_qc = [], [], []
        psal, psal_adj, psal_qc = [], [], []

        for line in lines:
            line_length = len(line)
            if line.startswith("     PLATFORM NUMBER"):
                eng['wmo'] = re.findall(r'\S+', line[28:line_length])[0]
            elif line.startswith("     CYCLE NUMBER"):
                eng.setdefault('cycle', []).append(int(re.findall(r'\S+', line[28:line_length])[0]))
            elif line.startswith("     DATE CREATION"):
                eng.setdefault('date_creation', []).append(re.findall(r'\S+', line[28:line_length])[0])
            elif line.startswith("     DATE UPDATE"):
                eng.setdefault('date_update', []).append(re.findall(r'\S+', line[28:line_length])[0])
            elif line.startswith("     PROJECT NAME"):
                eng['proj_name'] = re.findall(r'\S+', line[28:line_length])[0]
            elif line.startswith("     PI NAME"):
                eng['pi_name'] = re.findall(r'\S+', line[28:line_length])[0]
            elif line.startswith("     INSTRUMENT TYPE"):
                eng['inst_type'] = re.findall(r'\S+', line[28:line_length])[0]
            elif line.startswith("     FLOAT SERIAL NO"):
                eng['float_sn'] = re.findall(r'\S+', line[28:line_length])[0]
            elif line.startswith("     FIRMWARE VERSION"):
                eng['firmware'] = re.findall(r'\S+', line[28:line_length])[0]
            elif line.startswith("     WMO INSTRUMENT TYPE"):
                eng['wmo_inst'] = re.findall(r'\S+', line[28:line_length])[0]
            elif line.startswith("     TRANSMISSION SYSTEM"):
                eng['trans_sys'] = re.findall(r'\S+', line[28:line_length])[0]
            elif line.startswith("     POSITIONING SYSTEM"):
                eng['pos_sys'] = re.findall(r'\S+', line[28:line_length])[0]
            elif line.startswith("     SAMPLE DIRECTION"):
                eng.setdefault('direction', []).append(re.findall(r'\S+', line[28:29])[0])
            elif line.startswith("     DATA MODE"):
                eng.setdefault('data_mode', []).append(re.findall(r'\S+', line[28:29])[0])
            elif line.startswith("     JULIAN DAY"):
                pos.setdefault('juld', []).append(float(re.findall(r'\S+', line[28:38])[0]))
            elif line.startswith("     QC FLAG FOR DATE"):
                pos.setdefault('juld_qc', []).append(int(re.findall(r'\S+', line[28:29])[0]))
            elif line.startswith("     LATITUDE"):
                pos.setdefault('lat', []).append(float(re.findall(r'\S+', line[28:line_length])[0]))
            elif line.startswith("     LONGITUDE"):
                pos.setdefault('lon', []).append(float(re.findall(r'\S+', line[28:line_length])[0]))
            elif line.startswith("     QC FLAG FOR LOCATION"):
                pos.setdefault('pos_qc', []).append(int(re.findall(r'\S+', line[28:29])[0]))
            elif line.startswith("=========================================================================="):
                continue
            elif len(line) > 55:
                pres.append(float(line[0:7]))
                pres_adj.append(float(line[7:14]))
                pres_qc.append(int(line[14:17]))
                temp.append(float(line[17:26]))
                temp_adj.append(float(line[26:35]))
                temp_qc.append(int(line[35:38]))
                psal.append(float(line[38:47]))
                psal_adj.append(float(line[47:56]))
                psal_qc.append(int(line[56:59]))

        k = len(pres)
        if k > PRES.shape[0]:
            PRES = np.pad(PRES, ((0, k - PRES.shape[0]), (0, 0)), constant_values=np.nan)
            PRES_ADJ = np.pad(PRES_ADJ, ((0, k - PRES_ADJ.shape[0]), (0, 0)), constant_values=np.nan)
            PRES_QC = np.pad(PRES_QC, ((0, k - PRES_QC.shape[0]), (0, 0)), constant_values=np.nan)
            TEMP = np.pad(TEMP, ((0, k - TEMP.shape[0]), (0, 0)), constant_values=np.nan)
            TEMP_ADJ = np.pad(TEMP_ADJ, ((0, k - TEMP_ADJ.shape[0]), (0, 0)), constant_values=np.nan)
            TEMP_QC = np.pad(TEMP_QC, ((0, k - TEMP_QC.shape[0]), (0, 0)), constant_values=np.nan)
            PSAL = np.pad(PSAL, ((0, k - PSAL.shape[0]), (0, 0)), constant_values=np.nan)
            PSAL_ADJ = np.pad(PSAL_ADJ, ((0, k - PSAL_ADJ.shape[0]), (0, 0)), constant_values=np.nan)
            PSAL_QC = np.pad(PSAL_QC, ((0, k - PSAL_QC.shape[0]), (0, 0)), constant_values=np.nan)

        PRES[:k, i] = pres
        PRES_ADJ[:k, i] = pres_adj
        PRES_QC[:k, i] = pres_qc
        TEMP[:k, i] = temp
        TEMP_ADJ[:k, i] = temp_adj
        TEMP_QC[:k, i] = temp_qc
        PSAL[:k, i] = psal
        PSAL_ADJ[:k, i] = psal_adj
        PSAL_QC[:k, i] = psal_qc

    pos['lat'] = [np.nan if lat < -999 else lat for lat in pos.get('lat', [])]
    pos['lon'] = [np.nan if lon < -999 else lon for lon in pos.get('lon', [])]

    PRES[PRES < -999] = np.nan
    PRES_ADJ[PRES_ADJ < -999] = np.nan
    TEMP[TEMP < -99] = np.nan
    TEMP_ADJ[TEMP_ADJ < -99] = np.nan
    PSAL[PSAL < -99] = np.nan
    PSAL_ADJ[PSAL_ADJ < -99] = np.nan

    data['pres'] = PRES
    data['pres_adj'] = PRES_ADJ
    data['pres_qc'] = PRES_QC
    data['temp'] = TEMP
    data['temp_adj'] = TEMP_ADJ
    data['temp_qc'] = TEMP_QC
    data['psal'] = PSAL
    data['psal_adj'] = PSAL_ADJ
    data['psal_qc'] = PSAL_QC

    return eng, pos, data


# 导入每月每天数据 --
def resource_monthly_data(month_data_dir):
    # 获取年份和月份
    year = int(month_data_dir.split("/")[-1][:-2])
    month = int(month_data_dir.split("/")[-1][-2:])
    # 根据年份和月份计算当月的天数
    if month in [1, 3, 5, 7, 8, 10, 12]:
        days = 31
    elif month in [4, 6, 9, 11]:
        days = 30
    elif year % 4 == 0 and year % 100 != 0 or year % 400 == 0:
        days = 29
    else:
        days = 28

    daily = [[] for _ in range(days + 1)]

    # 遍历浮漂数据文件，建立日期索引
    with os.scandir(month_data_dir) as it:
        for entry in it:
            # 读取数据
            if entry.name.endswith('.dat'):
                Log.i(f"Reading data file: {entry.path}")
                eng, pos, data = read_data_file(entry.path)
                # 计算创建日期
                date_str = eng['date_creation']
                date = datetime.strptime(date_str, '%Y%m%d%H%M%S')
                # 保存对应日期的数据
                day = date.day
                one_day = daily[day - 1]
                one_day.append({'eng': eng, 'pos': pos, 'data': data})
                one_day = np.array(one_day, dtype=object)
    daily = np.array(daily, dtype=object)
    Log.i("daily shape: ", daily.shape)
    for i in range(len(daily)):
        Log.i("one_day shape: ", len(daily[i]))
        Log.i("one_day shape after range: ", len(daily[i]))
    return daily


# ---------------------------- BOA Argo 数据处理 ----------------------------

def import_argo_ocean_variables(nc_filename):
    """导入Argo海洋数据变量

    nc_filename -- Argo 数据集的文件路径
    """

    nc_file = nc.Dataset(nc_filename, 'r')
    variables = nc_file.variables

    temperature = variables['temp'][0].transpose((2, 1, 0))
    lon = variables['lon'][:] - 0.5
    lat = variables['lat'][:] + 0.5
    ild = variables['ILD'][:][0].transpose((1, 0))
    mld = variables['MLD'][:][0].transpose((1, 0))
    cmld = variables['CMLD'][:][0].transpose((1, 0))

    return temperature, lon, lat, ild, mld, cmld


class ArgoDataLoader:
    """
    Argo 懒加载数据加载器
    优化：按需加载数据，避免一次性加载所有3D温度场（节省~3.8GB内存）
    """
    def __init__(self, data_dir):
        self.file_info = []
        with os.scandir(data_dir) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith('.nc') and entry.name.startswith('BOA_Argo'):
                    info = {
                        'path': entry.path,
                        'year': int(entry.name.split('_')[2]),
                        'month': int(entry.name.split('_')[3].split('.')[0])
                    }
                    self.file_info.append(info)
        # 按时间排序
        self.file_info.sort(key=lambda x: (x['year'], x['month']))
    
    @lru_cache(maxsize=6)  # 缓存最近6个月的数据（约200MB）
    def _load_month(self, path):
        """加载单个月份数据（带缓存）"""
        temperature, lon, lat, ild, mld, cmld = import_argo_ocean_variables(path)
        return {
            'temp': temperature,
            'lon': lon,
            'lat': lat,
            'mld': mld
        }
    
    def __len__(self):
        return len(self.file_info)
    
    def __getitem__(self, index):
        """按需加载单个月份"""
        if isinstance(index, int):
            if index < 0 or index >= len(self.file_info):
                raise IndexError(f"Index {index} out of range")
            info = self.file_info[index]
            data = self._load_month(info['path'])
            data['year'] = info['year']
            data['month'] = info['month']
            return data
        elif isinstance(index, slice):
            # 支持切片操作
            indices = range(*index.indices(len(self)))
            return [self[i] for i in indices]
        else:
            raise TypeError(f"Index must be int or slice, not {type(index).__name__}")


def resource_argo_monthly_data(argo_data_dir):
    """
    读取Argo数据（优化版：懒加载）
    内存优化：从一次性加载3.8GB → 按需加载，缓存约200MB
    :param argo_data_dir: Argo数据目录
    :return: 懒加载数据加载器（兼容原有接口）
    """
    return ArgoDataLoader(argo_data_dir)


class ERA5DataLoader:
    """
    ERA5 懒加载数据加载器（支持年度文件，每个文件月份数可能不同）
    优化：按需加载数据，避免一次性加载所有月份数据（节省~1.5GB内存）
    
    注意：每个NetCDF文件包含若干月的数据，月份数可能不同
    """
    def __init__(self, data_dir):
        self.file_paths = []
        with os.scandir(data_dir) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith('.nc'):
                    self.file_paths.append(entry.path)
        # 按文件名排序，确保时间顺序
        self.file_paths.sort()
        
        # 构建索引映射：全局月份索引 -> (文件索引, 文件内月份索引)
        self._index_map = []
        self._file_months = []  # 每个文件的月份数
        
        if len(self.file_paths) > 0:
            for file_idx, file_path in enumerate(self.file_paths):
                months_in_file = self._get_months_in_file(file_path)
                self._file_months.append(months_in_file)
                for month_idx in range(months_in_file):
                    self._index_map.append((file_idx, month_idx))
        
        self._total_months = len(self._index_map)
    
    def _get_months_in_file(self, file_path):
        """获取单个文件中的月份数"""
        nc_file = nc.Dataset(file_path, 'r')
        time_dim = nc_file.variables['sst'].shape[0]  # 时间维度
        nc_file.close()
        return time_dim
    
    @lru_cache(maxsize=3)  # 缓存最近3个文件的数据（约300MB）
    def _load_file(self, path):
        """加载一个文件的所有数据（带缓存）"""
        nc_file = nc.Dataset(path, 'r')
        sst = nc_file.variables['sst'][:]  # [time, lat, lon]
        nc_file.close()
        return np.array(sst)
    
    def __len__(self):
        """返回总月份数"""
        return self._total_months
    
    def __getitem__(self, index):
        """按需加载单个月份"""
        if isinstance(index, int):
            if index < 0 or index >= len(self):
                raise IndexError(f"Index {index} out of range [0, {len(self)})")
            
            # 使用索引映射获取文件索引和文件内月份索引
            file_idx, month_idx = self._index_map[index]
            
            # 加载文件数据（带缓存）
            file_data = self._load_file(self.file_paths[file_idx])
            
            # 返回特定月份
            return file_data[month_idx, :, :]
            
        elif isinstance(index, slice):
            # 支持切片操作：返回numpy数组
            indices = range(*index.indices(len(self)))
            return np.array([self[i] for i in indices])
        else:
            raise TypeError(f"Index must be int or slice, not {type(index).__name__}")
    
    @property
    def shape(self):
        """返回数据形状（兼容numpy数组接口）"""
        if len(self) == 0:
            return (0,)
        first = self[0]
        return (len(self), *first.shape)


def resource_era5_monthly_sst_data(era5_data_dir):
    """
    读取ERA5月平均数据（优化版：懒加载）
    内存优化：从一次性加载1.5GB → 按需加载，缓存约100MB
    :param era5_data_dir: ERA5数据目录
    :return: 懒加载数据加载器（兼容原有接口）
    """
    return ERA5DataLoader(era5_data_dir)

def construct_argo_training_set(all_months):
    """构建Argo训练集
    """

    _all_sst = None
    _all_stations = None

    _all_profiles = None

    for one_month in all_months:
        temperature = one_month['temp']
        lon = one_month['lon']
        lat = one_month['lat']

        # 输入
        _sst = temperature[160:180, 60:80, 0].reshape(400, -1).reshape(-1)
        _station = np.array([(lon[i], lat[j]) for i in range(160, 180) for j in range(60, 80)])

        # 输出
        _profile = temperature[160:180, 60:80, :].reshape(400, -1)

        if one_month['year'] == 2023 and one_month['month'] == 9:
            Log.i("2023年9月数据: ")
            Log.i("海表温度: ", temperature[160:180, 60:80, 0].reshape(400, -1).reshape(-1))
            Log.i("经纬度: ", _station)
            Log.i("剖面温度序列: ", _profile)

        if _all_sst is None:
            _all_sst = _sst
            _all_stations = _station

            _all_profiles = _profile
        else:
            _all_sst = np.concatenate((_all_sst, _sst))
            _all_stations = np.concatenate((_all_stations, _station))

            _all_profiles = np.concatenate((_all_profiles, _profile))

    return [_all_sst, _all_stations], _all_profiles

def scope(block_size = 20):
    from src.config.area import Area
    
    blocks = []
    lon_range = range(-180, 181, block_size)
    lat_range = range(-80, 81, block_size)
    for (l, li) in zip(lon_range, range(len(lon_range))):
        if li+1 >= len(lon_range):
            continue
        lon_block = [l, lon_range[li+1]]
        for (k, ki) in zip(lat_range, range(len(lat_range))):
            if ki+1 >= len(lat_range):
                continue
            lat_block = [k, lat_range[ki+1]]
            block = {"lon":lon_block,"lat":lat_block}
            area = Area(title=f"Area_{lon_block[0]}_{lon_block[1]}_{lat_block[0]}_{lat_block[1]}.png", lon=lon_block, lat=lat_block, description=f"Area {len(blocks)+1}")
            blocks.append(area)

    return blocks

def set_seed(seed):
    """设置随机种子以确保实验的可重现性
    
    Args:
        seed (int): 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保cuDNN的确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False