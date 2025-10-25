import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

RMSE = {
    'LSTM': [
        1.441,
        1.430,
        1.905,
        1.188,
        1.143,
        1.195,
        1.603,
        1.642
    ],
    'ConvLSTM': [
        1.042,
        1.039,
        0.980,
        0.964,
        0.947,
        1.020,
        1.089,
        1.107
    ],
    'UNetLSTM': [
        0.641,
        0.849,
        0.626,
        0.569,
        0.693,
        0.678,
        0.873,
        0.696,
    ],
    'Transformer': [
        0.518,
        0.564,
        0.529,
        0.484,
        0.538,
        0.576,
        0.551,
        0.567
    ],
    'RATransformer': [
        0.422,
        0.361,
        0.335,
        0.356,
        0.384,
        0.470,
        0.413,
        0.510
    ]
}

R2 = {
    'LSTM': [ 
        0.551,
        0.653,
        0.672,
        0.589,
        0.689,
        0.560,
        0.630,
        0.634
    ],
    'ConvLSTM': [
        0.741,
        0.761,
        0.692,
        0.762,
        0.732,
        0.711,
        0.620,
        0.705
    ],
    'UNetLSTM': [
        0.835,
        0.939,
        0.891,
        0.900,
        0.918,
        0.919,
        0.839,
        0.878
    ],
    'Transformer': [
        0.937,
        0.944,
        0.978,
        0.981,
        0.937,
        0.929,
        0.945,
        0.923
    ],
    'RATransformer': [
        0.986,
        0.943,
        0.969,
        0.974,
        0.966,
        0.966,
        0.953,
        0.986
    ]
}

def plot_metrics():
    # Set font for English
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Liberation Serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    
    # Create time series (8 months from 2024-10 to 2025-05)
    dates = []
    start_year, start_month = 2024, 10
    for i in range(8):
        year = start_year + (start_month + i - 1) // 12
        month = (start_month + i - 1) % 12 + 1
        dates.append(datetime(year, month, 1))
    x_ticks = [d.strftime('%Y-%m') for d in dates]
    
    # Create figure and subplots
    fig, ax1 = plt.subplots(figsize=(14, 8), facecolor='white', dpi=1200)
    ax2 = ax1.twinx()
    
    # Set background color
    ax1.set_facecolor('#f8f9fa')
    
    # Set bar width and position
    width = 0.15
    x = np.arange(len(dates))
    
    # Color scheme (optimized with prominent RA model)
    colors = [
        '#7E57C2',  # LSTM
        '#66BB6A',  # ConvLSTM  
        '#42A5F5',  # UNetLSTM
        '#EF5350',  # Transformer
        '#FFA726'   # RATransformer
    ]
    
    # Add grid lines
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    
    # Draw NRMSE bars
    models = ['LSTM', 'ConvLSTM', 'UNetLSTM', 'Transformer', 'RATransformer']
    bars = []
    for i, model in enumerate(models):
        bar = ax1.bar(x + i*width, RMSE[model], width, 
                     label=f'NRMSE-{model}', 
                     color=colors[i], 
                     alpha=1,  # Opaque
                     zorder=3)
        bars.append(bar)
    
    # Draw R2 lines
    markers = ['o', 's', '^', 'D', 'v']
    lines = []
    for i, model in enumerate(models):
        line = ax2.plot(x + width*2, R2[model], 
                       marker=markers[i], 
                       label=f'R²-{model}', 
                       color=colors[i],
                       linewidth=2,
                       markersize=7,
                       zorder=4)
        lines.append(line[0])
    
    # Set title and labels
    ax1.set_xlabel('Prediction Time (month)', fontsize=16, labelpad=10)
    ax1.set_ylabel('Normalized Root Mean Square Error (NRMSE, °C)', fontsize=16, labelpad=10)
    ax2.set_ylabel('Coefficient of Determination (R²)', fontsize=16, labelpad=10)
    
    # Set x-axis ticks
    ax1.set_xticks(x + width*2)
    ax1.set_xticklabels(x_ticks, rotation=30, ha='right', fontsize=13)
    
    # Beautify axes
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    # Set y-axis range and format
    ax1.set_ylim(0, 2.0)
    ax2.set_ylim(0, 1.0)
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    
    # Legend
    legend_elements = bars + lines
    legend_labels = [bar.get_label() for bar in bars] + [line.get_label() for line in lines]
    fig.legend(legend_elements, 
              legend_labels, 
              loc='center', 
              bbox_to_anchor=(0.52, 0),
              ncol=5,
              frameon=False,
              fontsize=14)
    
    # Adjust layout
    plt.subplots_adjust(bottom=0.15, right=0.95, left=0.1)
    
    # Show plot
    plt.show()

if __name__ == '__main__':
    plot_metrics()