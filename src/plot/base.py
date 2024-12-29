# 基础画图模块和工具

from matplotlib import pyplot as plt

def create_base_figure():
    """
    创建固定 dpi 为 600 的高分辨率画布
    """

    dpi = 600
    
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

def create_axes(row=1, col=1):
    """
    创建一个 row 行 col 列的子图画布
    
    :param row: 子图的行数
    :param col: 子图的列数
    :return: 返回一个包含所有子图的列表
    """
    figure = create_base_figure()
    
    axes = []
    
    for m in range(row):
        for n in range(col):
            index = m * row + n
            axes.append(figure.add_subplot(row, col, index))
    
    return axes

def create_shared_axes(row=1, col=1, shared='all'):
    """
    创建一个共享坐标轴的子图画布
    
    :param row: 子图的行数
    :param col: 子图的列数
    :param shared: 共享类型,可选值为 'all'(共享 x 和 y 轴),'x'(仅共享 x 轴),'y'(仅共享 y 轴)
    :return: 返回一个包含所有子图的列表
    """
    figure = create_base_figure()
    
    axes = []
    
    match shared:
        case 'all':
            sharex = sharey = True
        case 'x':
            sharex = True
            sharey = False  
        case 'y':
            sharex = False
            sharey = True
        case _:
            sharex = sharey = False
            
    for m in range(row):
        for n in range(col):
            index = m * row + n
            if index == 0:
                axes.append(figure.add_subplot(row, col, index + 1))
            else:
                axes.append(figure.add_subplot(row, col, index + 1, 
                                             sharex=axes[0] if sharex else None,
                                             sharey=axes[0] if sharey else None))
    
    return axes

# 3D 绘图

def create_3d_ax():
    """
    创建一个3D图像画布
    
    :return: 返回图像对象和3D子图对象
    """
    figure = create_base_figure()
    ax = figure.add_subplot(111, projection='3d')
    
    return ax

def create_3d_axes(row=1, col=1):
    """
    创建一个包含多个3D子图的画布
    
    :param row: 子图的行数
    :param col: 子图的列数
    :return: 返回一个包含所有3D子图的列表
    """
    figure = create_base_figure()
    
    axes = []
    for m in range(row):
        for n in range(col):
            axes.append(figure.add_subplot(row, col, m * row + n + 1, projection='3d'))
    
    return axes

def create_shared_3d_axes(row=1, col=1, shared='all'):
    """
    创建一个包含多个共享坐标轴的3D子图的画布
    
    :param row: 子图的行数
    :param col: 子图的列数
    :param shared: 共享的坐标轴,'all'表示共享所有轴,'x'表示共享x轴,'y'表示共享y轴,'z'表示共享z轴
    :return: 返回一个包含所有3D子图的列表
    """
    figure = create_base_figure()
    axes = []
    
    # 根据shared参数设置共享轴
    match shared:
        case 'all':
            sharex = sharey = sharez = True
        case 'x':
            sharex = True
            sharey = sharez = False
        case 'y':
            sharey = True
            sharex = sharez = False
        case 'z':
            sharez = True
            sharex = sharey = False
        case _:
            sharex = sharey = sharez = False
            
    for m in range(row):
        for n in range(col):
            index = m * row + n
            if index == 0:
                axes.append(figure.add_subplot(row, col, index + 1, projection='3d'))
            else:
                axes.append(figure.add_subplot(row, col, index + 1, projection='3d',
                                             sharex=axes[0] if sharex else None,
                                             sharey=axes[0] if sharey else None))
                if sharez:
                    axes[-1].set_zlim(axes[0].get_zlim())
    
    return axes