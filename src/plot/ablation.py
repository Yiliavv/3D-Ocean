"""
消融实验可视化模块

生成学术级图表，支持:
- AGU/IEEE/Nature 样式
- 性能对比柱状图（带误差棒）
- 组件贡献度分析图
- 预测误差热力图
- 超参数敏感性曲线
- 精度-效率 Pareto 曲线
- LaTeX 表格生成
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import seaborn as sns

from src.analysis.ablation.config import (
    VisualizationStyle,
    STYLES,
    ABLATION_VARIANTS,
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置 matplotlib 中文支持
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.unicode_minus'] = False


class AblationVisualizer:
    """
    消融实验可视化器
    
    生成学术级的消融实验分析图表。
    """
    
    def __init__(
        self,
        style: str = 'agu',
        output_dir: str = 'out/ablation/figures',
        save_formats: List[str] = None
    ):
        """
        Args:
            style: 样式名称 ('agu', 'ieee', 'nature')
            output_dir: 输出目录
            save_formats: 保存格式列表
        """
        self.style = STYLES.get(style, STYLES['agu'])
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_formats = save_formats or self.style.save_formats
        
        # 应用样式
        self._apply_style()
        
        logger.info(f"AblationVisualizer initialized with {style} style")
    
    def _apply_style(self):
        """应用学术样式"""
        rcparams = self.style.get_matplotlib_rcparams()
        plt.rcParams.update(rcparams)
        
        # 额外设置
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 16,
            'axes.labelsize': 16,
            'axes.titlesize': 16,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 16,
            'figure.titlesize': 16,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'text.usetex': False,  # 禁用 LaTeX（兼容性）
        })

    def _clear_all_titles(self, fig: plt.Figure):
        """
        Clear figure and axes titles to keep title-free plots.
        """
        fig.suptitle('')
        for ax in fig.axes:
            ax.set_title('')
    
    def _save_figure(self, fig: plt.Figure, name: str):
        """保存图表到多种格式"""
        self._clear_all_titles(fig)
        for fmt in self.save_formats:
            output_path = self.output_dir / f"{name}.{fmt}"
            fig.savefig(
                output_path,
                format=fmt,
                dpi=self.style.dpi,
                bbox_inches='tight',
                pad_inches=0.1
            )
            logger.info(f"Figure saved: {output_path}")
    
    def plot_performance_comparison(
        self,
        results_df: pd.DataFrame,
        metric: str = 'RMSE',
        title: str = None,
        figname: str = 'performance_comparison'
    ) -> plt.Figure:
        """
        绘制性能对比柱状图（带误差棒）
        
        Args:
            results_df: 实验结果 DataFrame
            metric: 评估指标 ('RMSE', 'MAE', 'R2', etc.)
            title: 图表标题
            figname: 保存文件名
            
        Returns:
            matplotlib Figure 对象
        """
        # 按变体聚合
        metric_col = metric.lower()
        if metric_col not in results_df.columns:
            metric_col = metric
        
        grouped = results_df.groupby('config_name')[metric_col].agg(['mean', 'std']).reset_index()
        
        # 排序（baseline 在前）
        order = ['baseline'] + [v for v in grouped['config_name'] if v != 'baseline']
        grouped['order'] = grouped['config_name'].map({n: i for i, n in enumerate(order)})
        grouped = grouped.sort_values('order')
        
        # 创建图表
        fig, ax = plt.subplots(figsize=self.style.get_figure_size('double', 0.6))
        
        x = np.arange(len(grouped))
        colors = self.style.primary_colors[:len(grouped)]
        
        bars = ax.bar(
            x, 
            grouped['mean'],
            yerr=grouped['std'],
            capsize=3,
            color=colors,
            edgecolor='black',
            linewidth=0.5,
            error_kw={'linewidth': 1}
        )
        
        # 标签
        display_names = []
        for name in grouped['config_name']:
            if name in ABLATION_VARIANTS:
                display_names.append(ABLATION_VARIANTS[name].display_name)
            else:
                display_names.append(name)
        
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=30, ha='right')
        ax.set_ylabel(metric)
        
        if title:
            ax.set_title('')
        else:
            ax.set_title('')
        
        # 添加数值标签
        for bar, (_, row) in zip(bars, grouped.iterrows()):
            height = bar.get_height()
            ax.annotate(
                f'{row["mean"]:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=16
            )
        
        plt.tight_layout()
        self._save_figure(fig, figname)
        
        return fig
    
    def plot_component_contribution(
        self,
        results_df: pd.DataFrame,
        baseline_name: str = 'baseline',
        figname: str = 'component_contribution'
    ) -> plt.Figure:
        """
        绘制组件贡献度分析图
        
        显示移除每个组件后性能下降的程度。
        
        Args:
            results_df: 实验结果 DataFrame
            baseline_name: 基准变体名称
            figname: 保存文件名
            
        Returns:
            matplotlib Figure 对象
        """
        # 计算每个组件的贡献
        baseline_rmse = results_df[results_df['config_name'] == baseline_name]['rmse'].mean()
        
        contributions = []
        for name in ABLATION_VARIANTS:
            if name == baseline_name:
                continue
            
            variant_data = results_df[results_df['config_name'] == name]
            if len(variant_data) == 0:
                continue
                
            variant_rmse = variant_data['rmse'].mean()
            variant_std = variant_data['rmse'].std()
            
            # 贡献度 = 移除后的性能下降
            contribution = (variant_rmse - baseline_rmse) / baseline_rmse * 100
            
            contributions.append({
                'component': ABLATION_VARIANTS[name].display_name,
                'contribution': contribution,
                'std': variant_std / baseline_rmse * 100,
                'variant_name': name
            })
        
        if not contributions:
            logger.warning("No contribution data to plot")
            return None
        
        df = pd.DataFrame(contributions)
        df = df.sort_values('contribution', ascending=True)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=self.style.get_figure_size('single', 1.0))
        
        y = np.arange(len(df))
        colors = [self.style.primary_colors[1] if c > 0 else self.style.primary_colors[2] 
                  for c in df['contribution']]
        
        bars = ax.barh(
            y,
            df['contribution'],
            xerr=df['std'],
            color=colors,
            edgecolor='black',
            linewidth=0.5,
            capsize=3
        )
        
        ax.set_yticks(y)
        ax.set_yticklabels(df['component'])
        ax.set_xlabel('RMSE Increase (%)')
        ax.set_title('')
        
        # 添加零线
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
        
        # 添加数值标签
        for bar, (_, row) in zip(bars, df.iterrows()):
            width = bar.get_width()
            x_pos = width + 0.5 if width > 0 else width - 0.5
            ax.annotate(
                f'{row["contribution"]:.1f}%',
                xy=(x_pos, bar.get_y() + bar.get_height() / 2),
                va='center',
                ha='left' if width > 0 else 'right',
                fontsize=16
            )
        
        plt.tight_layout()
        self._save_figure(fig, figname)
        
        return fig
    
    def plot_error_heatmap(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        variant_name: str = 'baseline',
        figname: str = 'error_heatmap'
    ) -> plt.Figure:
        """
        绘制预测误差热力图
        
        Args:
            predictions: 预测值 [H, W]
            targets: 真实值 [H, W]
            variant_name: 变体名称
            figname: 保存文件名
            
        Returns:
            matplotlib Figure 对象
        """
        # 计算误差
        error = predictions - targets
        
        # 创建图表
        fig, axes = plt.subplots(1, 3, figsize=self.style.get_figure_size('double', 0.4))
        
        # 真实值
        im1 = axes[0].imshow(
            targets,
            cmap=self.style.sst_colormap,
            aspect='auto',
            origin='lower'
        )
        axes[0].set_title('')
        plt.colorbar(im1, ax=axes[0], label='SST (°C)')
        
        # 预测值
        im2 = axes[1].imshow(
            predictions,
            cmap=self.style.sst_colormap,
            aspect='auto',
            origin='lower'
        )
        axes[1].set_title('')
        plt.colorbar(im2, ax=axes[1], label='SST (°C)')
        
        # 误差图
        vmax = np.nanpercentile(np.abs(error), 95)
        im3 = axes[2].imshow(
            error,
            cmap=self.style.error_colormap,
            aspect='auto',
            origin='lower',
            vmin=-vmax,
            vmax=vmax
        )
        axes[2].set_title('')
        plt.colorbar(im3, ax=axes[2], label='Error (°C)')
        
        # 设置标签
        for ax in axes:
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        
        fig.suptitle('')
        
        plt.tight_layout()
        self._save_figure(fig, figname)
        
        return fig
    
    def plot_sensitivity_curve(
        self,
        sensitivity_df: pd.DataFrame,
        param_name: str,
        figname: str = None
    ) -> plt.Figure:
        """
        绘制超参数敏感性曲线
        
        Args:
            sensitivity_df: 敏感性分析结果 DataFrame
            param_name: 参数名称
            figname: 保存文件名
            
        Returns:
            matplotlib Figure 对象
        """
        if figname is None:
            figname = f'sensitivity_{param_name}'
        
        # 聚合
        grouped = sensitivity_df.groupby('value').agg({
            'rmse': ['mean', 'std'],
            'mae': ['mean', 'std']
        }).reset_index()
        grouped.columns = ['value', 'rmse_mean', 'rmse_std', 'mae_mean', 'mae_std']
        
        # 创建图表
        fig, ax1 = plt.subplots(figsize=self.style.get_figure_size('single', 0.8))
        
        color1 = self.style.primary_colors[0]
        color2 = self.style.primary_colors[1]
        
        # RMSE
        ax1.errorbar(
            grouped['value'],
            grouped['rmse_mean'],
            yerr=grouped['rmse_std'],
            color=color1,
            marker='o',
            linewidth=2,
            label='RMSE',
            capsize=3
        )
        ax1.set_xlabel(param_name)
        ax1.set_ylabel('RMSE', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # MAE (次坐标轴)
        ax2 = ax1.twinx()
        ax2.errorbar(
            grouped['value'],
            grouped['mae_mean'],
            yerr=grouped['mae_std'],
            color=color2,
            marker='s',
            linewidth=2,
            label='MAE',
            capsize=3
        )
        ax2.set_ylabel('MAE', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # 图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=16)
        
        ax1.set_title('')
        
        plt.tight_layout()
        self._save_figure(fig, figname)
        
        return fig
    
    def plot_sensitivity_heatmap(
        self,
        results: Dict[Tuple[str, str], float],
        param1_name: str,
        param2_name: str,
        metric: str = 'RMSE',
        figname: str = 'sensitivity_heatmap'
    ) -> plt.Figure:
        """
        绘制双参数敏感性热力图
        
        Args:
            results: {(param1_val, param2_val): metric_value} 字典
            param1_name: 参数1名称
            param2_name: 参数2名称
            metric: 评估指标
            figname: 保存文件名
            
        Returns:
            matplotlib Figure 对象
        """
        # 转换为矩阵
        param1_vals = sorted(set(k[0] for k in results.keys()))
        param2_vals = sorted(set(k[1] for k in results.keys()))
        
        matrix = np.zeros((len(param2_vals), len(param1_vals)))
        for i, p2 in enumerate(param2_vals):
            for j, p1 in enumerate(param1_vals):
                matrix[i, j] = results.get((p1, p2), np.nan)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=self.style.get_figure_size('single', 0.9))
        
        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        
        ax.set_xticks(np.arange(len(param1_vals)))
        ax.set_yticks(np.arange(len(param2_vals)))
        ax.set_xticklabels(param1_vals)
        ax.set_yticklabels(param2_vals)
        ax.set_xlabel(param1_name)
        ax.set_ylabel(param2_name)
        
        # 添加数值标注
        for i in range(len(param2_vals)):
            for j in range(len(param1_vals)):
                text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                             ha='center', va='center', color='white',
                             fontsize=16)
        
        plt.colorbar(im, ax=ax, label=metric)
        ax.set_title('')
        
        plt.tight_layout()
        self._save_figure(fig, figname)
        
        return fig
    
    def plot_pareto_curve(
        self,
        results_df: pd.DataFrame,
        efficiency_metric: str = 'train_time_seconds',
        accuracy_metric: str = 'rmse',
        figname: str = 'pareto_curve'
    ) -> plt.Figure:
        """
        绘制精度-效率 Pareto 曲线
        
        Args:
            results_df: 实验结果 DataFrame
            efficiency_metric: 效率指标列名
            accuracy_metric: 精度指标列名
            figname: 保存文件名
            
        Returns:
            matplotlib Figure 对象
        """
        # 按变体聚合
        grouped = results_df.groupby('config_name').agg({
            efficiency_metric: 'mean',
            accuracy_metric: 'mean',
        }).reset_index()
        
        # 创建图表
        fig, ax = plt.subplots(figsize=self.style.get_figure_size('single', 0.8))
        
        colors = self.style.primary_colors
        
        for i, (_, row) in enumerate(grouped.iterrows()):
            color = colors[i % len(colors)]
            display_name = ABLATION_VARIANTS.get(row['config_name'], row['config_name'])
            if hasattr(display_name, 'display_name'):
                display_name = display_name.display_name
            
            ax.scatter(
                row[efficiency_metric],
                row[accuracy_metric],
                s=100,
                color=color,
                edgecolor='black',
                linewidth=0.5,
                label=display_name,
                zorder=10
            )
        
        # 计算 Pareto 前沿
        pareto_points = self._compute_pareto_frontier(
            grouped[efficiency_metric].values,
            grouped[accuracy_metric].values
        )
        
        if len(pareto_points) > 1:
            pareto_x = [grouped[efficiency_metric].iloc[i] for i in pareto_points]
            pareto_y = [grouped[accuracy_metric].iloc[i] for i in pareto_points]
            # 排序
            sorted_idx = np.argsort(pareto_x)
            pareto_x = [pareto_x[i] for i in sorted_idx]
            pareto_y = [pareto_y[i] for i in sorted_idx]
            
            ax.plot(pareto_x, pareto_y, '--', color='gray', 
                   alpha=0.7, linewidth=1.5, label='Pareto Frontier')
        
        ax.set_xlabel(f'Training Time (s)')
        ax.set_ylabel('RMSE')
        ax.set_title('')
        ax.legend(loc='upper right', fontsize=16)
        
        plt.tight_layout()
        self._save_figure(fig, figname)
        
        return fig
    
    def _compute_pareto_frontier(
        self,
        x_vals: np.ndarray,
        y_vals: np.ndarray
    ) -> List[int]:
        """计算 Pareto 前沿点索引"""
        pareto_points = []
        
        for i in range(len(x_vals)):
            is_dominated = False
            for j in range(len(x_vals)):
                if i != j:
                    # 对于最小化问题：j 支配 i 如果 j 在两个维度都更好或相等，且至少一个严格更好
                    if x_vals[j] <= x_vals[i] and y_vals[j] <= y_vals[i]:
                        if x_vals[j] < x_vals[i] or y_vals[j] < y_vals[i]:
                            is_dominated = True
                            break
            
            if not is_dominated:
                pareto_points.append(i)
        
        return pareto_points
    
    def plot_component_breakdown(
        self,
        results_df: pd.DataFrame,
        baseline_name: str = 'baseline',
        figname: str = 'component_breakdown'
    ) -> plt.Figure:
        """
        绘制组件贡献度堆叠图
        
        Args:
            results_df: 实验结果 DataFrame
            baseline_name: 基准变体名称
            figname: 保存文件名
            
        Returns:
            matplotlib Figure 对象
        """
        # 计算每个组件的贡献
        baseline_rmse = results_df[results_df['config_name'] == baseline_name]['rmse'].mean()
        
        components = []
        for name in ['wo_convstem', 'wo_attention', 'wo_shpe', 'wo_multiscale', 'wo_gate']:
            variant_data = results_df[results_df['config_name'] == name]
            if len(variant_data) == 0:
                continue
            
            variant_rmse = variant_data['rmse'].mean()
            contribution = max(0, (variant_rmse - baseline_rmse) / baseline_rmse * 100)
            
            components.append({
                'name': ABLATION_VARIANTS[name].display_name,
                'contribution': contribution
            })
        
        if not components:
            return None
        
        # 创建饼图
        fig, ax = plt.subplots(figsize=self.style.get_figure_size('single', 0.9))
        
        sizes = [c['contribution'] for c in components]
        labels = [c['name'] for c in components]
        colors = self.style.primary_colors[:len(components)]
        
        # 如果有总贡献，添加一个 "Other" 类别
        total_contribution = sum(sizes)
        if total_contribution < 100:
            sizes.append(100 - total_contribution)
            labels.append('Other/Interaction')
            colors.append('#CCCCCC')
        
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=[0.02] * len(sizes)
        )
        
        ax.set_title('')
        
        plt.tight_layout()
        self._save_figure(fig, figname)
        
        return fig
    
    def generate_nature_figure(
        self,
        results_df: pd.DataFrame,
        figname: str = 'ablation_study_nature'
    ) -> plt.Figure:
        """
        生成 Nature 风格的整合图表
        
        包含 4 个子图:
        (a) 性能对比柱状图 (RMSE/MAE)
        (b) 组件贡献度分析 (相对变化)
        (c) 训练效率对比
        (d) 精度-效率权衡分析
        
        Args:
            results_df: 实验结果 DataFrame
            figname: 输出文件名
            
        Returns:
            matplotlib Figure 对象
        """
        # Nature 风格设置
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 16,
            'axes.labelsize': 16,
            'axes.titlesize': 16,
            'legend.fontsize': 16,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'axes.linewidth': 0.5,
            'xtick.major.width': 0.5,
            'ytick.major.width': 0.5,
            'axes.spines.top': False,
            'axes.spines.right': False,
        })
        
        # Nature 配色方案 (色盲友好)
        colors = {
            'baseline': '#2166AC',      # 深蓝
            'wo_attention': '#D6604D',  # 红
            'wo_convstem': '#4DAC26',   # 绿
            'wo_gate': '#B2ABD2',       # 淡紫
            'wo_multiscale': '#FDB863', # 橙
            'wo_shpe': '#762A83',       # 紫
        }
        
        # 创建 2x2 布局
        fig = plt.figure(figsize=(7.0, 5.5))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3, 
                             left=0.08, right=0.98, top=0.95, bottom=0.08)
        
        # 按变体聚合数据
        metrics = ['rmse', 'mae', 'r2', 'train_time_seconds', 'num_parameters']
        valid_metrics = [m for m in metrics if m in results_df.columns]
        grouped = results_df.groupby('config_name')[valid_metrics].agg(['mean', 'std']).reset_index()
        
        # 获取变体顺序
        variant_order = ['baseline', 'wo_attention', 'wo_convstem', 'wo_gate', 'wo_multiscale', 'wo_shpe']
        variant_order = [v for v in variant_order if v in grouped['config_name'].values]
        
        # 变体显示名称
        display_names = {
            'baseline': 'Full Model',
            'wo_attention': 'w/o RGA',
            'wo_convstem': 'w/o ConvStem',
            'wo_gate': 'w/o Gate',
            'wo_multiscale': 'w/o MultiScale',
            'wo_shpe': 'w/o SHPE',
        }
        
        # ===== (a) 性能对比柱状图 =====
        ax_a = fig.add_subplot(gs[0, 0])
        
        x = np.arange(len(variant_order))
        width = 0.35
        
        # RMSE 和 MAE
        rmse_means = [grouped[grouped['config_name'] == v][('rmse', 'mean')].values[0] for v in variant_order]
        rmse_stds = [grouped[grouped['config_name'] == v][('rmse', 'std')].values[0] for v in variant_order]
        mae_means = [grouped[grouped['config_name'] == v][('mae', 'mean')].values[0] for v in variant_order]
        mae_stds = [grouped[grouped['config_name'] == v][('mae', 'std')].values[0] for v in variant_order]
        
        bar_colors = [colors.get(v, '#888888') for v in variant_order]
        
        bars1 = ax_a.bar(x - width/2, rmse_means, width, yerr=rmse_stds, 
                        color=bar_colors, alpha=0.9, label='RMSE',
                        edgecolor='white', linewidth=0.5,
                        error_kw={'linewidth': 0.8, 'capsize': 2, 'capthick': 0.8})
        bars2 = ax_a.bar(x + width/2, mae_means, width, yerr=mae_stds,
                        color=bar_colors, alpha=0.5, label='MAE',
                        edgecolor='white', linewidth=0.5, hatch='///',
                        error_kw={'linewidth': 0.8, 'capsize': 2, 'capthick': 0.8})
        
        ax_a.set_ylabel('Error (°C)')
        ax_a.set_xticks(x)
        ax_a.set_xticklabels([display_names.get(v, v) for v in variant_order], rotation=45, ha='right')
        ax_a.legend(loc='upper right', frameon=False)
        ax_a.set_ylim(0, max(rmse_means) * 1.3)
        ax_a.text(-0.15, 1.05, 'a', transform=ax_a.transAxes, fontsize=16, fontweight='bold')
        
        # 添加 baseline 参考线
        baseline_rmse = rmse_means[0]
        ax_a.axhline(y=baseline_rmse, color='#2166AC', linestyle='--', linewidth=0.8, alpha=0.7)
        
        # ===== (b) 组件贡献度分析 =====
        ax_b = fig.add_subplot(gs[0, 1])
        
        # 计算相对于 baseline 的变化百分比
        baseline_rmse = grouped[grouped['config_name'] == 'baseline'][('rmse', 'mean')].values[0]
        
        components = [v for v in variant_order if v != 'baseline']
        contributions = []
        for v in components:
            v_rmse = grouped[grouped['config_name'] == v][('rmse', 'mean')].values[0]
            change = ((v_rmse - baseline_rmse) / baseline_rmse) * 100
            contributions.append(change)
        
        # 按贡献度排序（从小到大，这样最大的显示在最上面）
        sorted_data = sorted(zip(components, contributions), key=lambda x: x[1], reverse=False)
        components_sorted, contributions_sorted = zip(*sorted_data)
        
        y_pos = np.arange(len(components_sorted))
        bar_colors_contrib = [colors.get(v, '#888888') for v in components_sorted]
        
        bars = ax_b.barh(y_pos, contributions_sorted, color=bar_colors_contrib, 
                        edgecolor='white', linewidth=0.5, height=0.65)
        
        # 添加数值标签（更明显的样式）
        for i, (bar, val) in enumerate(zip(bars, contributions_sorted)):
            x_pos = bar.get_width() + 1.0
            ax_b.text(x_pos, bar.get_y() + bar.get_height()/2, 
                     f'+{val:.1f}%',
                     va='center', ha='left', fontsize=16, fontweight='bold', color='#333333')
        
        ax_b.set_yticks(y_pos)
        ax_b.set_yticklabels([display_names.get(v, v) for v in components_sorted])
        ax_b.set_xlabel('RMSE Degradation (%)', fontsize=16)
        ax_b.axvline(x=0, color='#333333', linestyle='-', linewidth=0.8)
        ax_b.set_xlim(-2, max(contributions_sorted) * 1.5)
        
        # 添加说明文字
        ax_b.text(0.98, 0.02, 'Higher = more important', 
                 transform=ax_b.transAxes, fontsize=16, ha='right', va='bottom',
                 style='italic', color='#666666')
        
        ax_b.text(-0.15, 1.05, 'b', transform=ax_b.transAxes, fontsize=16, fontweight='bold')
        
        # ===== (c) 训练效率对比 =====
        ax_c = fig.add_subplot(gs[1, 0])
        
        if 'train_time_seconds' in valid_metrics:
            time_means = [grouped[grouped['config_name'] == v][('train_time_seconds', 'mean')].values[0] for v in variant_order]
            
            bars = ax_c.bar(x, time_means, color=bar_colors, edgecolor='white', linewidth=0.5)
            
            # 添加数值标签
            for bar, val in zip(bars, time_means):
                ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                         f'{val:.0f}s', ha='center', va='bottom', fontsize=16)
            
            ax_c.set_ylabel('Training Time (s)')
            ax_c.set_xticks(x)
            ax_c.set_xticklabels([display_names.get(v, v) for v in variant_order], rotation=45, ha='right')
            ax_c.set_ylim(0, max(time_means) * 1.2)
        
        ax_c.text(-0.15, 1.05, 'c', transform=ax_c.transAxes, fontsize=16, fontweight='bold')
        
        # ===== (d) 精度-效率权衡散点图 =====
        ax_d = fig.add_subplot(gs[1, 1])
        
        if 'train_time_seconds' in valid_metrics:
            # 预定义标签偏移量避免重叠
            label_offsets = {
                'baseline': (15, -15),
                'wo_attention': (-50, 15),
                'wo_convstem': (10, 15),
                'wo_gate': (-45, -15),
                'wo_multiscale': (10, -20),
                'wo_shpe': (-55, 10),
            }
            
            scatter_data = []
            for v in variant_order:
                v_data = grouped[grouped['config_name'] == v]
                rmse_val = v_data[('rmse', 'mean')].values[0]
                time_val = v_data[('train_time_seconds', 'mean')].values[0]
                scatter_data.append((v, time_val, rmse_val))
                
                ax_d.scatter(time_val, rmse_val, 
                           c=colors.get(v, '#888888'),
                           s=120, alpha=0.9, edgecolors='white', linewidth=1,
                           label=display_names.get(v, v), zorder=3)
            
            # 添加带箭头的标签
            for v, time_val, rmse_val in scatter_data:
                offset = label_offsets.get(v, (10, 10))
                ax_d.annotate(
                    display_names.get(v, v), 
                    (time_val, rmse_val),
                    xytext=offset, 
                    textcoords='offset points',
                    fontsize=16,
                    alpha=0.9,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'),
                    arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5, lw=0.5)
                )
            
            ax_d.set_xlabel('Training Time (s)')
            ax_d.set_ylabel('RMSE (°C)')
            
            # 添加理想区域标注（左下角）
            ax_d.axhline(y=baseline_rmse, color='#2166AC', linestyle='--', linewidth=0.8, alpha=0.6)
            ax_d.axvline(x=250, color='#2166AC', linestyle=':', linewidth=0.8, alpha=0.4)
            
            # 标注理想区域
            ax_d.text(50, 1.05, 'Ideal\nregion', fontsize=16, color='#2166AC', alpha=0.7,
                     ha='center', va='center')
            ax_d.fill_between([0, 250], [1.0, 1.0], [baseline_rmse, baseline_rmse], 
                             alpha=0.08, color='#2166AC')
        
        ax_d.text(-0.15, 1.05, 'd', transform=ax_d.transAxes, fontsize=16, fontweight='bold')
        
        # 保存图表
        self._save_figure(fig, figname)
        
        return fig
    
    def generate_all_figures(
        self,
        results_df: pd.DataFrame,
        sensitivity_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, plt.Figure]:
        """
        生成消融实验图表（仅 Nature 风格整合图）
        
        Args:
            results_df: 实验结果 DataFrame
            sensitivity_df: 敏感性分析结果（可选，暂不使用）
            
        Returns:
            生成的图表字典
        """
        figures = {}
        
        # 只生成 Nature 风格整合图（包含所有必要信息的 4 子图）
        logger.info("Generating Nature-style integrated figure...")
        figures['nature_main'] = self.generate_nature_figure(results_df)
        
        logger.info(f"Generated {len(figures)} figure")
        
        return figures


class TableGenerator:
    """
    LaTeX 表格生成器
    
    生成学术论文格式的 LaTeX 表格。
    """
    
    def __init__(self, output_dir: str = 'out/ablation/tables'):
        """
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_main_results_table(
        self,
        results_df: pd.DataFrame,
        metrics: List[str] = None,
        caption: str = 'Ablation study results',
        label: str = 'tab:ablation',
        output_name: str = 'main_results.tex'
    ) -> str:
        """
        生成主结果表格（含均值/标准差）
        
        Args:
            results_df: 实验结果 DataFrame
            metrics: 指标列表
            caption: 表格标题
            label: LaTeX 标签
            output_name: 输出文件名
            
        Returns:
            LaTeX 代码字符串
        """
        if metrics is None:
            metrics = ['rmse', 'mae', 'r2', 'spatial_corr']
        
        # 按变体聚合
        valid_metrics = [m for m in metrics if m in results_df.columns]
        grouped = results_df.groupby('config_name')[valid_metrics].agg(['mean', 'std']).reset_index()
        
        # 构建 LaTeX
        n_metrics = len(valid_metrics)
        col_spec = 'l' + 'r' * n_metrics
        
        latex = []
        latex.append(r'\begin{table}[htbp]')
        latex.append(r'\centering')
        latex.append(f'\\caption{{{caption}}}')
        latex.append(f'\\label{{{label}}}')
        latex.append(f'\\begin{{tabular}}{{{col_spec}}}')
        latex.append(r'\toprule')
        
        # 表头
        headers = ['Variant'] + [m.upper() for m in valid_metrics]
        latex.append(' & '.join(headers) + r' \\')
        latex.append(r'\midrule')
        
        # 数据行
        baseline_means = {}
        
        # 提取 config_name 列（处理 MultiIndex 列）
        config_names = grouped['config_name']
        if isinstance(config_names, pd.DataFrame):
            config_names = config_names.iloc[:, 0]  # 如果是 DataFrame，取第一列
        
        for idx, row in grouped.iterrows():
            # 获取变体名称
            variant_name = str(config_names.iloc[idx] if hasattr(config_names, 'iloc') else config_names[idx])
            
            if variant_name in ABLATION_VARIANTS:
                display_name = ABLATION_VARIANTS[variant_name].display_name
            else:
                display_name = variant_name
            
            values = [display_name]
            
            for metric in valid_metrics:
                try:
                    mean = row[(metric, 'mean')]
                    std = row[(metric, 'std')]
                except (KeyError, TypeError):
                    # 处理单层索引情况
                    mean = row.get(f'{metric}_mean', 0)
                    std = row.get(f'{metric}_std', 0)
                
                # 保存 baseline 用于加粗
                if variant_name == 'baseline':
                    baseline_means[metric] = mean
                
                # 格式化
                val_str = f'{mean:.4f} ± {std:.4f}'
                
                # 如果是最佳值，加粗
                if variant_name == 'baseline':
                    val_str = f'\\textbf{{{val_str}}}'
                
                values.append(val_str)
            
            latex.append(' & '.join(values) + r' \\')
        
        latex.append(r'\bottomrule')
        latex.append(r'\end{tabular}')
        latex.append(r'\end{table}')
        
        latex_code = '\n'.join(latex)
        
        # 保存
        output_path = self.output_dir / output_name
        output_path.write_text(latex_code, encoding='utf-8')
        logger.info(f"LaTeX table saved: {output_path}")
        
        return latex_code
    
    def add_significance_markers(
        self,
        latex_table: str,
        significance_df: pd.DataFrame
    ) -> str:
        """
        添加统计显著性标记 (*, **, ***)
        
        Args:
            latex_table: 原始 LaTeX 表格
            significance_df: 显著性检验结果
            
        Returns:
            带标记的 LaTeX 代码
        """
        # 简单实现：在表格后添加脚注
        footnote = []
        footnote.append(r'\begin{tablenotes}')
        footnote.append(r'\small')
        footnote.append(r'\item[$*$] $p < 0.05$')
        footnote.append(r'\item[$**$] $p < 0.01$')
        footnote.append(r'\item[$***$] $p < 0.001$')
        footnote.append(r'\end{tablenotes}')
        
        # 在 \end{table} 前插入脚注
        latex_table = latex_table.replace(
            r'\end{table}',
            '\n'.join(footnote) + '\n' + r'\end{table}'
        )
        
        return latex_table
    
    def generate_efficiency_table(
        self,
        results_df: pd.DataFrame,
        caption: str = 'Computational efficiency comparison',
        label: str = 'tab:efficiency',
        output_name: str = 'efficiency_comparison.tex'
    ) -> str:
        """
        生成效率对比表格
        
        Args:
            results_df: 实验结果 DataFrame
            caption: 表格标题
            label: LaTeX 标签
            output_name: 输出文件名
            
        Returns:
            LaTeX 代码字符串
        """
        # 按变体聚合
        metrics = ['train_time_seconds', 'inference_time_ms', 'peak_memory_mb', 'num_parameters']
        agg_dict = {m: 'mean' for m in metrics if m in results_df.columns}
        grouped = results_df.groupby('config_name').agg(agg_dict).reset_index()
        
        # 构建 LaTeX
        latex = []
        latex.append(r'\begin{table}[htbp]')
        latex.append(r'\centering')
        latex.append(f'\\caption{{{caption}}}')
        latex.append(f'\\label{{{label}}}')
        latex.append(r'\begin{tabular}{lrrrr}')
        latex.append(r'\toprule')
        
        # 表头
        latex.append(r'Variant & Train Time (s) & Inference (ms) & Memory (MB) & Parameters \\')
        latex.append(r'\midrule')
        
        # 数据行
        for _, row in grouped.iterrows():
            variant_name = row['config_name']
            if variant_name in ABLATION_VARIANTS:
                display_name = ABLATION_VARIANTS[variant_name].display_name
            else:
                display_name = variant_name
            
            train_time = row.get('train_time_seconds', 0)
            infer_time = row.get('inference_time_ms', 0)
            memory = row.get('peak_memory_mb', 0)
            params = row.get('num_parameters', 0)
            
            latex.append(
                f'{display_name} & {train_time:.1f} & {infer_time:.2f} & '
                f'{memory:.1f} & {params:,} \\\\'
            )
        
        latex.append(r'\bottomrule')
        latex.append(r'\end{tabular}')
        latex.append(r'\end{table}')
        
        latex_code = '\n'.join(latex)
        
        # 保存
        output_path = self.output_dir / output_name
        output_path.write_text(latex_code, encoding='utf-8')
        logger.info(f"LaTeX table saved: {output_path}")
        
        return latex_code


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='消融实验可视化')
    
    parser.add_argument(
        '--results', '-r',
        type=str,
        default='out/ablation/results/ablation_results.csv',
        help='实验结果 CSV 文件路径'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='out/ablation/figures',
        help='输出目录'
    )
    
    parser.add_argument(
        '--style', '-s',
        type=str,
        default='agu',
        choices=['agu', 'ieee', 'nature'],
        help='图表样式'
    )
    
    parser.add_argument(
        '--tables',
        action='store_true',
        help='生成 LaTeX 表格'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='生成所有图表'
    )
    
    args = parser.parse_args()
    
    # 检查结果文件
    results_path = Path(args.results)
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        return 1
    
    # 加载数据
    results_df = pd.read_csv(results_path)
    logger.info(f"Loaded {len(results_df)} results from {results_path}")
    
    # 创建可视化器
    visualizer = AblationVisualizer(
        style=args.style,
        output_dir=args.output
    )
    
    if args.all:
        # 生成所有图表（目前只有 Nature 风格整合图）
        visualizer.generate_all_figures(results_df)
    else:
        # 默认只生成 Nature 风格整合图
        visualizer.generate_nature_figure(results_df)
    
    # 生成表格
    if args.tables:
        table_gen = TableGenerator(output_dir='out/ablation/tables')
        table_gen.generate_main_results_table(results_df)
        table_gen.generate_efficiency_table(results_df)
    
    logger.info("Visualization complete")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

