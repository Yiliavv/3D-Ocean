# 基础画图模块和工具

import numpy as np
from matplotlib import pyplot as plt
from cartopy import crs as ccrs
from cartopy import feature as cfeat

def create_base_figure():
    """
    创建固定 dpi 为 2400 的高分辨率画布
    """

    dpi = 2400
    
    figure = plt.figure(dpi=dpi)
    
    return figure

def create_ax():
    """
    创建一个单独的子图
    
    :return: 返回一个子图对象
    """
    figure = create_base_figure()
    ax = figure.add_subplot(111)

    return ax

def create_axes(row=1, col=1, shared='all'):
    """
    创建一个共享坐标轴的子图画布，分辩率为 1200
    
    :param row: 子图的行数
    :param col: 子图的列数
    :param shared: 共享类型,可选值为 'all'(共享 x 和 y 轴),'x'(仅共享 x 轴),'y'(仅共享 y 轴)
    :return: 返回一个包含所有子图的列表
    """
    
    match shared:
        case 'all':
            sharex = "all"
            sharey = "all"
        case 'x':
            sharex = "all"
            sharey = False  
        case 'y':
            sharex = False
            sharey = "all"
        case _:
            sharex = sharey = False
            
    _, axes = plt.subplots(row, col, dpi = 2400, sharex=sharex, sharey=sharey)
    
    return axes

# cartopy 投影

def create_carto_ax():
    """
    创建一个基础投影地图，分辩率为 2400
    """
    figure = create_base_figure()
    ax = figure.add_subplot(111, projection=ccrs.PlateCarree())
    
    # 设置字体为 Liberation Serif（Times New Roman 的开源等效字体）
    plt.rcParams['font.family'] = 'Liberation Serif'
    
    ax.add_feature(cfeat.LAND)
    ax.add_feature(cfeat.COASTLINE, linewidth=0.5)
    
    return ax

def create_carto_axes(row=1, col=1):
    """
    创建多个基础投影地图，分辩率为 2400
    """
    axes = []
    for m in range(row):
        for n in range(col):
            axes.append(create_carto_ax())
            
    return axes


# 3D 绘图

def create_3d_ax():
    """
    创建一个3D图像画布，分辩率为 2400
    
    :return: 返回图像对象和3D子图对象
    """
    figure = create_base_figure()
    ax = figure.add_subplot(111, projection='3d')
    
    return ax

def create_3d_axes(row=1, col=1):
    """
    创建一个包含多个3D子图的画布，分辩率为 2400
    
    :param row: 子图的行数
    :param col: 子图的列数
    :return: 返回一个包含所有3D子图的列表
    """
    
    _, axes = plt.subplots(row, col, dpi = 2400, projection='3d')

    return axes

def create_shared_3d_axes(row=1, col=1, shared='all'):
    """
    创建一个包含多个共享坐标轴的3D子图的画布，分辩率为 2400
    
    :param row: 子图的行数
    :param col: 子图的列数
    :param shared: 共享的坐标轴,'all'表示共享所有轴,'x'表示共享x轴,'y'表示共享y轴
    :return: 返回一个包含所有3D子图的列表
    """
    figure = create_base_figure()
    axes = []
    
    # 根据shared参数设置共享轴
    match shared:
        case 'all':
            sharex = sharey = True
        case 'x':
            sharex = True
            sharey = False
        case 'y':
            sharey = True
            sharex = False
        case _:
            sharex = sharey = False
            
    for i in range(row):
        for j in range(col):
            axes.append(figure.add_subplot(row, col, i * col + j + 1, projection='3d', sharex=sharex, sharey=sharey))
    
    return axes