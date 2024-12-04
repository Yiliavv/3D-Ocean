import os, sys
import re
import numpy as np
import netCDF4 as nc
from datetime import datetime

from memory_profiler import profile

from src.utils.log import Log
from src.config.params import BASE_CDAC_DATA_PATH
from src.models.model import calculate_seawater_density, linear_fit, calculate_angle_tan


# 数据列表
def list_data_dirs():
    dirs = []

    with os.scandir(BASE_CDAC_DATA_PATH) as it:
        for entry in it:
            if entry.is_dir():
                dirs.append(entry.name)

    return dirs


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


def max_angle_compute_mld(float_data):
    """
    最大角度法计算混合层深度
    """
    data = float_data['data']
    pressures = data['pres']  # 1dbar = 1m, 压力即深度
    temperatures = data['temp']
    salinity = data['psal']

    # 计算密度
    densities = []
    for i in range(len(pressures)):
        density = calculate_seawater_density(temperatures[i], salinity[i], pressures[i])
        densities.append(density)

    # 过滤掉 NaN 值
    densities = np.array(densities)
    pressures = np.array(pressures)
    valid_indices = ~np.isnan(densities)
    densities = densities[valid_indices]
    pressures = pressures[valid_indices]

    if len(densities) == 0:
        Log.e("所有密度值均为 NaN，无法计算混合层深度")
        return 9999

    max_tan_h = 0
    mixed_layer_depth = None

    # 密度最大值和最小值
    q_min_index = np.argmin(densities)
    q_max_index = np.argmax(densities)
    z_min = pressures[q_min_index]
    z_max = pressures[q_max_index]

    # 密度差
    delta_d = densities[q_max_index] - densities[q_min_index]

    # 密度变化范围
    q_01 = 0.1 * delta_d + densities[q_min_index]
    q_02 = 0.7 * delta_d + densities[q_min_index]

    # 计算在密度变化范围内的点个数
    n = int(np.sum([1 for q in densities if q_01 <= q <= q_02]))
    m = int(min(20, n))

    # 对每个点
    for i, pressure in enumerate(pressures):
        # 确定上方的点
        j = max(i - 1, 0)
        if i < m:
            j = i - 1
        elif i > m:
            j = m

        if i >= len(pressures) - 2: break

        # 线性拟合上方的点
        m1, c1 = linear_fit([pressures[j], pressures[i + 2]], [densities[j], densities[i + 2]])

        # 拟合下方的点
        m2, c2 = linear_fit([pressures[i + 1], pressures[i + 2]], [densities[i + 1], densities[i + 2]])

        # 计算正切
        tan_h = calculate_angle_tan(m1, m2)

        # 记录最大值
        if tan_h > max_tan_h:
            max_tan_h = tan_h
            mixed_layer_depth = pressure

    return mixed_layer_depth


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


def resource_argo_monthly_data(argo_data_dir):
    """
    读取Argo数据
    :param argo_data_dir: Argo数据目录
    :return:
    """
    argo_datas = []

    with os.scandir(argo_data_dir) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith('.nc') and entry.name.startswith('BOA_Argo'):
                one_month = {}
                temperature, lon, lat, ild, mld, cmld = import_argo_ocean_variables(entry.path)
                one_month['temp'] = temperature
                one_month['lon'] = lon
                one_month['lat'] = lat
                one_month['mld'] = mld
                one_month['year'] = int(entry.name.split('_')[2])
                one_month['month'] = int(entry.name.split('_')[3].split('.')[0])
                argo_datas.append(one_month)

    return argo_datas


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


# ---------------------------- EAR5 数据处理 ----------------------------

def import_era5_sst(nc_filename, start=0, end=0):
    """导入EAR5海洋数据变量

    nc_filename -- EAR5 数据集的文件路径
    """

    nc_file = nc.Dataset(nc_filename, 'r')
    variables = nc_file.variables

    sst = variables['sst'][start:end, :, :].copy()
    time = variables['valid_time'][:].copy()
    shape = tuple(variables['sst'].shape)
    nc_file.close()
    del variables

    return sst, shape, time
