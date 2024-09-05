# For data handler, to data model.
import gsw
import numpy as np

from research.config.params import LAT_RANGE, LON_RANGE


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
