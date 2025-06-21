import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        """Plot visualizations of seasonal patterns"""
        # Apply a professional-looking style
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=1200)
        fig.suptitle("Seasonality Analysis of SST (2004–2023)", fontsize=16, fontweight="bold", y=0.98)
        
        # 1. Time-series plot
        axes[0,0].plot(self.df.index, self.df['sst'], label="Mean SST", color="#1f77b4", linewidth=1)
        # 12-month rolling mean for smoother trend view
        rolling_mean = self.df['sst'].rolling(window=12, center=True).mean()
        axes[0,0].plot(self.df.index, rolling_mean, label="12-month Rolling Mean", color="#d62728", linewidth=2)
        axes[0,0].legend()
        # Improve x-axis date formatting
        axes[0,0].xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=10))
        axes[0,0].set_title('SST Time Series')
        axes[0,0].set_xlabel('Time')
        axes[0,0].set_ylabel('Temperature (°C)')
        
        # 2. Monthly boxplot
        sns.boxplot(data=self.df, x='month', y='sst', hue='month', ax=axes[0,1], palette="coolwarm", legend=False)
        axes[0,1].set_title('Monthly Temperature Distribution')
        axes[0,1].set_xlabel('Month')
        axes[0,1].set_ylabel('Temperature (°C)')
        
        # 3. Seasonal decomposition
        decomposition = seasonal_decompose(self.df['sst'], period=12)
        # 手动绘制趋势和季节成分，避免 DecomposeResult.plot 不支持 axes 参数
        axes[1, 0].plot(decomposition.trend, color="#2ca02c")
        axes[1, 0].set_title('SST Trend')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Temperature (°C)')

        axes[1, 1].plot(decomposition.seasonal, color="#ff7f0e")
        axes[1, 1].set_title('SST Seasonal Component')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Temperature (°C)')
        
        # Turn on grid for all subplots and tighten layout
        for ax_row in axes:
            for ax in ax_row:
                ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

def main():
    """Main function to run the seasonality analysis"""
    # Initialize dataset
    dataset = ERA5SSTMonthlyDataset(
        width=1,
        offset=0,
        lon=np.array([100, 180]),  # Pacific region
        lat=np.array([-30, 30])    # Tropical region
    )
    
    # Create analyzer
    analyzer = SeasonalityAnalysis(dataset)
    
    # Perform seasonality test
    results = analyzer.test_seasonality()
    
    print("\nSeasonality analysis results:")
    print(f"Dominant period: {results['dominant_period']:.2f} months")
    print(f"Seasonal strength: {results['seasonal_strength']:.2%}")
    print(f"ANOVA F-statistic: {results['anova_f']:.2f}")
    print(f"ANOVA p-value: {results['anova_p']:.2e}")
    
    # Plot seasonality charts
    fig = analyzer.plot_seasonal_patterns()
    plt.show()

if __name__ == "__main__":
    main() 