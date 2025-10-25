# 海表温度掩码可视化函数

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as cm_plt
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl import ticker as tk

from src.plot.base import create_ax, create_carto_ax
from src.plot.sst import _range

def plot_mask_binary(mask_sst, lon, lat, step=1, filename='mask_binary.png', title='海表温度掩码分布'):
    """
    绘制二值化掩码图 - 黑白对比显示有效/无效区域
    
    :param mask_sst: 掩码数组，True表示有效数据，False表示无效数据
    :param lon: 经度范围 [起始经度, 结束经度]
    :param lat: 纬度范围 [起始纬度, 结束纬度]
    :param step: 网格步长
    :param filename: 保存文件名
    :param title: 图像标题
    :return: 返回图像对象
    """
    from src.config.params import PREDICT_SAVE_PATH
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    
    # 生成网格点
    lon_grid, lat_grid = np.meshgrid(_range(lon, step), _range(lat, step))
    
    # 创建现代化的二值化颜色映射
    colors = ['#34495e', '#ecf0f1']  # 深蓝灰色表示无效数据，浅灰色表示有效数据
    cmap = ListedColormap(colors)
    
    # 绘制掩码
    im = ax.contourf(lon_grid, lat_grid, mask_sst.astype(int), 
                     levels=[0, 0.5, 1], 
                     cmap=cmap)
    
    # 添加边界线
    ax.contour(lon_grid, lat_grid, mask_sst.astype(int), 
               levels=[0.5], colors='#2c3e50', linewidths=2, alpha=0.8)
    
    # 添加现代化网格线
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8, color='#bdc3c7')
    
    # 设置坐标轴标签
    ax.set_xlabel('经度 (°E)', fontsize=14, fontweight='bold', color='#2c3e50')
    ax.set_ylabel('纬度 (°N)', fontsize=14, fontweight='bold', color='#2c3e50')
    
    # 美化坐标轴
    ax.tick_params(axis='both', which='major', labelsize=11, 
                   colors='#2c3e50', width=1.5)
    
    # 设置坐标轴边框样式
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('#34495e')
    
    # 添加现代化颜色条
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', 
                       pad=0.12, shrink=0.8, aspect=30)
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(['NaN (无效数据)', 'Not NaN (有效数据)'])
    cbar.ax.tick_params(labelsize=12, colors='#2c3e50')
    
    # 美化颜色条边框
    cbar.outline.set_linewidth(1.5)
    cbar.outline.set_color('#34495e')
    
    # 设置现代化标题
    plt.title(title, fontsize=16, fontweight='bold', color='#2c3e50', pad=25)
    
    # 设置背景色
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig(f'{PREDICT_SAVE_PATH}/{filename}', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    return ax

def plot_mask_geographic(mask_sst, lon, lat, step=1, filename='mask_geographic.png', title='海表温度掩码地理分布'):
    """
    绘制带地理投影的掩码图 - 在地图投影上显示掩码分布
    
    :param mask_sst: 掩码数组，True表示有效数据，False表示无效数据
    :param lon: 经度范围 [起始经度, 结束经度]
    :param lat: 纬度范围 [起始纬度, 结束纬度]
    :param step: 网格步长
    :param filename: 保存文件名
    :param title: 图像标题
    :return: 返回图像对象
    """
    from src.config.params import PREDICT_SAVE_PATH
    
    # 创建地图投影
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    
    # 设置地图范围
    ax.set_extent([lon[0], lon[1], lat[0], lat[1]], crs=ccrs.PlateCarree())
    
    # 添加地理特征
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, alpha=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.6)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    
    # 生成网格点
    lon_grid, lat_grid = np.meshgrid(_range(lon, step), _range(lat, step))
    
    # 创建透明度掩码图
    colors = ['red', 'green']  # 红色表示无效数据，绿色表示有效数据
    cmap = ListedColormap(colors)
    
    # 绘制掩码，使用透明度
    im = ax.contourf(lon_grid, lat_grid, mask_sst.astype(int), 
                     levels=[0, 0.5, 1], 
                     cmap=cmap, alpha=0.7,
                     transform=ccrs.PlateCarree())
    
    # 添加掩码边界线
    ax.contour(lon_grid, lat_grid, mask_sst.astype(int), 
               levels=[0.5], colors='black', linewidths=2, alpha=0.9,
               transform=ccrs.PlateCarree())
    
    # 设置网格线和标签
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                 alpha=0.5, linestyle='--', linewidth=0.5)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(['NaN', 'Not NaN'])
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout()
    
    return ax

def plot_mask_statistics(mask_sst, lon, lat, filename='mask_statistics.png', title='Statistics of Mask'):
    """
    绘制掩码统计信息图 - 显示有效/无效数据的统计信息
    
    :param mask_sst: 掩码数组，True表示有效数据，False表示无效数据
    :param lon: 经度范围 [起始经度, 结束经度]
    :param lat: 纬度范围 [起始纬度, 结束纬度]
    :param filename: 保存文件名
    :param title: 图像标题
    :return: 返回图像对象
    """
    from src.config.params import PREDICT_SAVE_PATH
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12), dpi=1200)
    
    # 计算统计信息
    total_points = mask_sst.size
    valid_points = np.sum(mask_sst)
    invalid_points = total_points - valid_points
    valid_ratio = valid_points / total_points * 100
    invalid_ratio = invalid_points / total_points * 100
    
    # 3. 按纬度统计有效数据比例
    row_valid_ratio = np.mean(mask_sst, axis=1) * 100
    lat_values = np.linspace(lat[0], lat[1], len(row_valid_ratio))
    
    ax3.plot(row_valid_ratio, lat_values, 'b', linewidth=2, marker='o', 
             markersize=3, color='#4472C4')
    ax3.set_xlabel('Valid Data Ratio (%)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Latitude (°N)', fontsize=11, fontweight='bold')
    ax3.set_title('Valid Data Distribution by Latitude', fontsize=12, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax3.set_xlim(0, 100)
    
    # 4. 按经度统计有效数据比例
    col_valid_ratio = np.mean(mask_sst, axis=0) * 100
    lon_values = np.linspace(lon[0], lon[1], len(col_valid_ratio))
    
    ax4.plot(lon_values, col_valid_ratio, 'r', linewidth=2, marker='s', 
             markersize=3, color='#E15759')
    ax4.set_xlabel('Longitude (°E)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Valid Data Ratio (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Valid Data Distribution by Longitude', fontsize=12, fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax4.set_ylim(0, 100)
    
    # 设置整体标题
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.95)
    
    # 调整子图布局，为底部统计信息留出空间，并增加子图间距
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.28, hspace=0.35, wspace=0.25)
    
    # 创建现代化的统计信息卡片
    # 计算卡片位置和尺寸（宽度为图像的80%，居中显示）
    card_width = 0.8
    card_height = 0.15
    card_left = (1 - card_width) / 2
    card_bottom = 0.05
    
    # 创建统计信息面板
    stats_panel = fig.add_axes([card_left, card_bottom, card_width, card_height])
    stats_panel.set_xlim(0, 1)
    stats_panel.set_ylim(0, 1)
    stats_panel.axis('off')
    
    # 添加现代化卡片背景
    from matplotlib.patches import FancyBboxPatch
    card_bg = FancyBboxPatch((0, 0), 1, 1,
                            boxstyle="round,pad=0.02",
                            facecolor='#f8f9fa',
                            edgecolor='#dee2e6',
                            linewidth=1.5,
                            alpha=0.95)
    stats_panel.add_patch(card_bg)
    
    # 添加渐变顶部条
    top_bar = FancyBboxPatch((0, 0.85), 1, 0.15,
                            boxstyle="round,pad=0.01,rounding_size=0.02",
                            facecolor='#4472C4',
                            alpha=0.1)
    stats_panel.add_patch(top_bar)
    
    # 添加卡片标题
    stats_panel.text(0.5, 0.9, 'Data Coverage Statistics', 
                     ha='center', va='center', fontsize=12, 
                     fontweight='bold', color='#2c3e50')
    
    # 统计信息内容，分两列显示
    left_stats = [
        f'Geographic Range: {lon[0]}°E - {lon[1]}°E, {lat[0]}°N - {lat[1]}°N',
        f'Total Points: {total_points:,}',
        f'Data Completeness: {valid_ratio:.1f}%'
    ]
    
    right_stats = [
        f'Valid Points: {valid_points:,} ({valid_ratio:.1f}%)',
        f'Invalid Points: {invalid_points:,} ({invalid_ratio:.1f}%)',
        f'Coverage Area: {(lon[1]-lon[0]) * (lat[1]-lat[0]):.1f} deg²'
    ]
    
    # 左列统计信息
    y_positions = [0.65, 0.45, 0.25]
    for i, stat in enumerate(left_stats):
        stats_panel.text(0.05, y_positions[i], stat, 
                        ha='left', va='center', fontsize=10, 
                        color='#495057', family='monospace')
    
    # 右列统计信息
    for i, stat in enumerate(right_stats):
        stats_panel.text(0.52, y_positions[i], stat, 
                        ha='left', va='center', fontsize=10, 
                        color='#495057', family='monospace')
    
    # 添加分隔线
    stats_panel.plot([0.5, 0.5], [0.15, 0.75], color='#dee2e6', 
                     linewidth=1, alpha=0.7)
    
    # 统一设置所有子图的边框样式
    for ax in [ax1, ax2, ax3, ax4]:
        for spine in ax.spines.values():
            spine.set_linewidth(1)
            spine.set_color('black')
        ax.tick_params(axis='both', which='major', labelsize=9)

def plot_pred_error_statistics(pred_error, lon, lat, filename='pred_error_statistics.png', 
                              title='Prediction Error Statistics'):
    """
    绘制预测误差统计信息图 - 显示预测误差的分布和趋势
    
    :param pred_error: 预测误差数组，值范围通常为[-1.5, 1.5]
    :param lon: 经度范围 [起始经度, 结束经度]
    :param lat: 纬度范围 [起始纬度, 结束纬度]
    :param filename: 保存文件名
    :param title: 图像标题
    :return: 返回图像对象
    """
    from src.config.params import PREDICT_SAVE_PATH
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12), dpi=1200)
    
    # 计算统计信息
    valid_mask = ~np.isnan(pred_error)
    valid_errors = pred_error[valid_mask]
    
    if len(valid_errors) == 0:
        print("Warning: No valid error data found")
        return fig
    
    total_points = pred_error.size
    valid_points = np.sum(valid_mask)
    positive_errors = np.sum(valid_errors > 0)
    negative_errors = np.sum(valid_errors < 0)
    zero_errors = np.sum(valid_errors == 0)
    
    mean_error = np.nanmean(pred_error)
    std_error = np.nanstd(pred_error)
    rmse = np.sqrt(np.nanmean(pred_error**2))
    mae = np.nanmean(np.abs(pred_error))
    
    # 1. 饼图显示正负误差分布 - 学术化风格
    labels = ['Positive Error', 'Negative Error', 'Zero Error']
    sizes = [positive_errors, negative_errors, zero_errors]
    colors = ['#E15759', '#4472C4', '#70AD47']  # 红色、蓝色、绿色
    
    # 过滤掉为0的部分
    non_zero_labels = []
    non_zero_sizes = []
    non_zero_colors = []
    for i, size in enumerate(sizes):
        if size > 0:
            non_zero_labels.append(labels[i])
            non_zero_sizes.append(size)
            non_zero_colors.append(colors[i])
    
    if non_zero_sizes:
        wedges, texts, autotexts = ax1.pie(non_zero_sizes, labels=non_zero_labels, 
                                          colors=non_zero_colors, autopct='%1.1f%%', 
                                          startangle=90, textprops={'fontsize': 10})
        
        # 设置饼图文本颜色为白色以增强对比度
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    ax1.set_title('Error Sign Distribution', fontsize=12, fontweight='bold', pad=15)
    
    # 2. 柱状图显示误差统计 - 简洁风格
    categories = ['Positive', 'Negative', 'Zero']
    counts = [positive_errors, negative_errors, zero_errors]
    bar_colors = ['#E15759', '#4472C4', '#70AD47']
    
    # 设置柱子位置，使分布更均匀
    x_positions = [0.2, 1.0, 1.8]
    
    bars = ax2.bar(x_positions, counts, color=bar_colors, alpha=0.8, 
                   edgecolor='black', linewidth=0.5, width=0.4)
    
    # 设置x轴标签位置
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(categories)
    
    ax2.set_ylabel('Number of Points', fontsize=11, fontweight='bold')
    ax2.set_title('Error Count by Sign', fontsize=12, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
    
    # 在柱状图上添加数值标签
    for bar, count in zip(bars, counts):
        if count > 0:  # 只为非零值添加标签
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
    
    # 设置y轴上限，为标签留出空间
    max_count = max(counts) if max(counts) > 0 else 1
    ax2.set_ylim(0, max_count * 1.15)
    ax2.set_xlim(-0.1, 2.1)
    
    # 3. 按纬度统计平均误差和RMSE
    row_mean_error = np.nanmean(pred_error, axis=1)
    row_rmse = np.sqrt(np.nanmean(pred_error**2, axis=1))
    lat_values = np.linspace(lat[0], lat[1], len(row_mean_error))
    
    # 创建上下双x轴（用于纬度方向）
    ax3_top = ax3.twiny()
    
    # 绘制平均误差（底部x轴）
    line1 = ax3.plot(row_mean_error, lat_values, linewidth=2, marker='o', 
                     markersize=3, color='#4472C4', label='Mean Error')
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax3.set_xlabel('Mean Error', fontsize=11, fontweight='bold', color='#4472C4')
    ax3.set_ylabel('Latitude (°N)', fontsize=11, fontweight='bold')
    ax3.tick_params(axis='x', labelcolor='#4472C4')
    ax3.set_xlim(-1.5, 1.5)
    
    # 绘制RMSE（顶部x轴）
    line2 = ax3_top.plot(row_rmse, lat_values, linewidth=2, marker='s', 
                         markersize=3, color='#E15759', label='RMSE')
    ax3_top.set_xlabel('RMSE', fontsize=11, fontweight='bold', color='#E15759')
    ax3_top.tick_params(axis='x', labelcolor='#E15759')
    ax3_top.set_xlim(0, 2.0)
    
    ax3.set_title('Mean Error & RMSE by Latitude', fontsize=12, fontweight='bold', pad=25)
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 添加图例
    lines1 = line1 + line2
    labels1 = [l.get_label() for l in lines1]
    ax3.legend(lines1, labels1, loc='upper right', fontsize=9)
    
    # 4. 按经度统计平均误差和RMSE
    col_mean_error = np.nanmean(pred_error, axis=0)
    col_rmse = np.sqrt(np.nanmean(pred_error**2, axis=0))
    lon_values = np.linspace(lon[0], lon[1], len(col_mean_error))
    
    # 创建双y轴（保持经度方向的左右布局）
    ax4_twin = ax4.twinx()
    
    # 绘制平均误差
    line3 = ax4.plot(lon_values, col_mean_error, linewidth=2, marker='o', 
                     markersize=3, color='#4472C4', label='Mean Error')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax4.set_xlabel('Longitude (°E)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Mean Error', fontsize=11, fontweight='bold', color='#4472C4')
    ax4.tick_params(axis='y', labelcolor='#4472C4')
    ax4.set_ylim(-1.5, 1.5)
    
    # 绘制RMSE
    line4 = ax4_twin.plot(lon_values, col_rmse, linewidth=2, marker='s', 
                          markersize=3, color='#E15759', label='RMSE')
    ax4_twin.set_ylabel('RMSE', fontsize=11, fontweight='bold', color='#E15759')
    ax4_twin.tick_params(axis='y', labelcolor='#E15759')
    ax4_twin.set_ylim(0, 2.0)
    
    ax4.set_title('Mean Error & RMSE by Longitude', fontsize=12, fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 添加图例
    lines2 = line3 + line4
    labels2 = [l.get_label() for l in lines2]
    ax4.legend(lines2, labels2, loc='upper right', fontsize=9)
    
    # 设置整体标题
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.95)
    
    # 调整子图布局，为底部统计信息留出空间，并增加子图间距
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.28, hspace=0.35, wspace=0.3)
    
    # 创建现代化的统计信息卡片
    # 计算卡片位置和尺寸（宽度为图像的80%，居中显示）
    card_width = 0.8
    card_height = 0.15
    card_left = (1 - card_width) / 2
    card_bottom = 0.05
    
    # 创建统计信息面板
    stats_panel = fig.add_axes([card_left, card_bottom, card_width, card_height])
    stats_panel.set_xlim(0, 1)
    stats_panel.set_ylim(0, 1)
    stats_panel.axis('off')
    
    # 添加现代化卡片背景
    from matplotlib.patches import FancyBboxPatch
    card_bg = FancyBboxPatch((0, 0), 1, 1,
                            boxstyle="round,pad=0.02",
                            facecolor='#f8f9fa',
                            edgecolor='#dee2e6',
                            linewidth=1.5,
                            alpha=0.95)
    stats_panel.add_patch(card_bg)
    
    # 添加渐变顶部条
    top_bar = FancyBboxPatch((0, 0.85), 1, 0.15,
                            boxstyle="round,pad=0.01,rounding_size=0.02",
                            facecolor='#4472C4',
                            alpha=0.1)
    stats_panel.add_patch(top_bar)
    
    # 添加卡片标题
    stats_panel.text(0.5, 0.9, 'Prediction Error Statistics', 
                     ha='center', va='center', fontsize=12, 
                     fontweight='bold', color='#2c3e50')
    
    # 统计信息内容，分两列显示
    left_stats = [
        f'Geographic Range: {lon[0]}°E - {lon[1]}°E, {lat[0]}°N - {lat[1]}°N',
        f'Total Points: {total_points:,}',
        f'Mean Error: {mean_error:.4f}'
    ]
    
    right_stats = [
        f'Valid Points: {valid_points:,} ({valid_points/total_points*100:.1f}%)',
        f'RMSE: {rmse:.4f}  |  MAE: {mae:.4f}',
        f'Std Error: {std_error:.4f}'
    ]
    
    # 左列统计信息
    y_positions = [0.65, 0.45, 0.25]
    for i, stat in enumerate(left_stats):
        stats_panel.text(0.05, y_positions[i], stat, 
                        ha='left', va='center', fontsize=10, 
                        color='#495057', family='monospace')
    
    # 右列统计信息
    for i, stat in enumerate(right_stats):
        stats_panel.text(0.52, y_positions[i], stat, 
                        ha='left', va='center', fontsize=10, 
                        color='#495057', family='monospace')
    
    # 添加分隔线
    stats_panel.plot([0.5, 0.5], [0.15, 0.75], color='#dee2e6', 
                     linewidth=1, alpha=0.7)
    
    # 统一设置所有子图的边框样式
    for ax in [ax1, ax2, ax3, ax4]:
        for spine in ax.spines.values():
            spine.set_linewidth(1)
            spine.set_color('black')
        ax.tick_params(axis='both', which='major', labelsize=9)