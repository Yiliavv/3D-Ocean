import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from scipy.signal import periodogram
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns

from src.dataset.ERA5 import ERA5SSTMonthlyDataset

class SeasonalityAnalysis:
    def __init__(self, dataset: ERA5SSTMonthlyDataset):
        self.dataset = dataset
        self.sst_monthly = None
        self.df = None
        self._prepare_data()

    def _prepare_data(self):
        """Prepare monthly mean SST data"""
        # Retrieve 20 years of monthly SST data
        all_months = []
        for i in range(240):  # 20 years * 12 months
            sst = self.dataset.__getitem__(i)
            if isinstance(sst, tuple):
                sst = sst[0]  # If a tuple is returned, use the first element
            mean_sst = np.nanmean(sst)  # Spatial mean
            all_months.append(mean_sst)
        
        # Create a time-series DataFrame
        dates = pd.date_range(start='2004-01-01', periods=240, freq='ME')
        self.df = pd.DataFrame({
            'sst': all_months,
            'month': dates.month,
            'year': dates.year
        }, index=dates)
        
    def test_seasonality(self):
        """
        Verify the existence of seasonality using three approaches:
        1. Lomb-Scargle periodogram
        2. Seasonal decomposition (STL)
        3. Monthly ANOVA
        """
        results = {}
        
        # 1. 周期性检验
        freqs, power = periodogram(self.df['sst'])
        peak_freq = freqs[np.argmax(power)]
        period = 1/peak_freq if peak_freq != 0 else float('inf')
        results['dominant_period'] = period
        
        # 2. 季节性分解
        decomposition = seasonal_decompose(self.df['sst'], period=12)
        seasonal_strength = np.var(decomposition.seasonal) / np.var(self.df['sst'])
        results['seasonal_strength'] = seasonal_strength
        
        # 3. 月度ANOVA检验
        f_stat, p_value = stats.f_oneway(*[
            self.df[self.df['month'] == month]['sst']
            for month in range(1, 13)
        ])
        results['anova_f'] = f_stat
        results['anova_p'] = p_value
        
        return results
    
    def plot_seasonal_patterns(self):
        """Plot visualizations of seasonal patterns as two separate figures"""
        # Apply a professional-looking style
        sns.set_style("whitegrid")
        # Set font to Liberation Serif for English text
        plt.rcParams['font.family'] = 'Liberation Serif'
        plt.rcParams['font.size'] = 10
        
        # Figure 1: Time-series plot
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6), dpi=1200)
        
        ax1.plot(self.df.index, self.df['sst'], label="Mean SST", color="#1f77b4", linewidth=1)
        # 12-month rolling mean for smoother trend view
        rolling_mean = self.df['sst'].rolling(window=12, center=True).mean()
        ax1.plot(self.df.index, rolling_mean, label="12-month Rolling Mean", color="#d62728", linewidth=2)
        ax1.legend(fontsize=14)
        ax1.set_xlabel('Time (month)', fontsize=14)
        ax1.set_ylabel('Temperature (°C)', fontsize=14)
        
        # 优化横坐标显示 - 按月份显示，从2004年1月开始
        ax1.set_xlim(pd.Timestamp('2004-01-01'), self.df.index[-1])  # 设置x轴范围从2004年1月开始
        
        # 创建自定义刻度位置，确保包含最后一个刻度
        major_ticks = pd.date_range(start='2004-01-01', end=self.df.index[-1], freq='24ME')  # 每24个月
        if self.df.index[-1] not in major_ticks:
            major_ticks = major_ticks.union([self.df.index[-1]])  # 添加最后一个时间点
        
        ax1.set_xticks(major_ticks)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # 格式：年-月
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        ax1.grid(True, linestyle="--", alpha=0.5)
        
        # 添加子图标签 (a)
        ax1.text(0.97, 0.97, '(a)', transform=ax1.transAxes, fontsize=16,
                ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Figure 2: Seasonal decomposition
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6), dpi=1200)
        
        decomposition = seasonal_decompose(self.df['sst'], period=12)
        ax2.plot(decomposition.seasonal, color="#ff7f0e")
        ax2.set_xlabel('Time (month)', fontsize=14)
        ax2.set_ylabel('Temperature (°C)', fontsize=14)
        
        # 优化横坐标显示 - 按月份显示，从2004年1月开始
        ax2.set_xlim(pd.Timestamp('2004-01-01'), self.df.index[-1])  # 设置x轴范围从2004年1月开始
        
        # 创建自定义刻度位置，确保包含最后一个刻度
        major_ticks = pd.date_range(start='2004-01-01', end=self.df.index[-1], freq='24ME')  # 每24个月
        if self.df.index[-1] not in major_ticks:
            major_ticks = major_ticks.union([self.df.index[-1]])  # 添加最后一个时间点
        
        ax2.set_xticks(major_ticks)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # 格式：年-月
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        ax2.grid(True, linestyle="--", alpha=0.5)
        
        # 添加子图标签 (b)
        ax2.text(0.97, 0.97, '(b)', transform=ax2.transAxes, fontsize=16,
                ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        return fig1, fig2