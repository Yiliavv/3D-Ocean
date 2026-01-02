# Quickstart: RGTransformer 消融实验

**Feature**: 003-rgtransformer-ablation  
**Date**: 2024-12-28

## 快速开始

### 1. 运行完整消融实验

```bash
# 使用默认配置运行所有消融变体
python -m src.analysis.ablation.runner

# 指定配置文件
python -m src.analysis.ablation.runner --config configs/ablation_config.yaml

# 只运行特定变体
python -m src.analysis.ablation.runner --variants baseline wo_convstem wo_shpe

# 指定运行次数
python -m src.analysis.ablation.runner --runs 5
```

### 2. 从断点恢复实验

```bash
# 自动检测并恢复未完成的实验
python -m src.analysis.ablation.runner --resume

# 强制重新运行所有实验
python -m src.analysis.ablation.runner --force
```

### 3. 生成可视化图表

```bash
# 生成所有图表
python -m src.plot.ablation --results out/ablation/results/ablation_results.csv

# 只生成性能对比图
python -m src.plot.ablation --type performance

# 指定输出格式和样式
python -m src.plot.ablation --format pdf --style agu
```

### 4. 生成 LaTeX 表格

```bash
# 生成主结果表格
python -m src.plot.ablation --tables --output out/ablation/tables/

# 包含显著性标记
python -m src.plot.ablation --tables --significance
```

---

## 命令行参数

### ablation.runner

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | str | `configs/ablation_config.yaml` | 配置文件路径 |
| `--variants` | list | all | 要运行的变体列表 |
| `--runs` | int | 3 | 每变体运行次数 |
| `--resume` | flag | - | 从断点恢复 |
| `--force` | flag | - | 强制重新运行 |
| `--output` | str | `out/ablation` | 输出目录 |
| `--seed` | int | 42 | 随机种子 |
| `--gpu` | int | 0 | GPU 设备编号 |

### plot.ablation

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--results` | str | required | 结果 CSV 文件路径 |
| `--type` | str | all | 图表类型: performance, contribution, heatmap, sensitivity, pareto |
| `--format` | list | pdf,png | 输出格式 |
| `--style` | str | agu | 可视化样式: agu, ieee, nature |
| `--output` | str | `out/ablation/figures` | 输出目录 |
| `--tables` | flag | - | 生成 LaTeX 表格 |
| `--significance` | flag | - | 包含显著性分析 |

---

## Python API

### 运行消融实验

```python
from src.analysis.ablation import AblationRunner, AblationConfig

# 创建运行器
runner = AblationRunner(
    output_dir='out/ablation',
    runs_per_variant=3,
    seed=42
)

# 运行所有变体
results = runner.run_all()

# 或只运行特定变体
results = runner.run_variants(['baseline', 'wo_convstem'])

# 获取汇总报告
report = runner.generate_report()
```

### 生成可视化

```python
from src.plot.ablation import AblationVisualizer
from src.analysis.ablation import VisualizationStyle, STYLES

# 加载结果
visualizer = AblationVisualizer(
    results_csv='out/ablation/results/ablation_results.csv',
    style=STYLES['agu']
)

# 生成所有图表
visualizer.generate_all(output_dir='out/ablation/figures')

# 或生成单个图表
visualizer.plot_performance_comparison('performance_comparison.pdf')
visualizer.plot_component_contribution('component_contribution.pdf')
visualizer.plot_error_heatmap('error_heatmap.pdf')
visualizer.plot_pareto_curve('pareto_curve.pdf')
```

### 生成 LaTeX 表格

```python
from src.plot.ablation import TableGenerator

generator = TableGenerator(
    results_csv='out/ablation/results/ablation_results.csv'
)

# 生成主结果表格
latex_table = generator.generate_main_results_table(
    include_significance=True
)

# 保存到文件
generator.save_table(latex_table, 'out/ablation/tables/main_results.tex')
```

---

## 输出目录结构

```
out/ablation/
├── results/
│   ├── ablation_results.csv        # 所有实验结果
│   ├── experiment_configs.yaml     # 实验配置备份
│   ├── statistical_tests.csv       # 统计检验结果
│   └── experiment_state.json       # 实验状态（断点续传）
│
├── figures/
│   ├── performance_comparison.pdf  # 性能对比柱状图
│   ├── performance_comparison.png
│   ├── component_contribution.pdf  # 组件贡献分析
│   ├── component_contribution.png
│   ├── error_heatmap.pdf           # 误差空间分布
│   ├── error_heatmap.png
│   ├── sensitivity_analysis.pdf    # 超参数敏感性
│   ├── sensitivity_analysis.png
│   ├── pareto_curve.pdf            # 精度-效率权衡
│   └── pareto_curve.png
│
├── tables/
│   ├── main_results.tex            # 主结果表格 (LaTeX)
│   └── efficiency_comparison.tex   # 效率对比表格
│
└── checkpoints/
    ├── baseline/
    │   ├── run_1/
    │   │   └── model.ckpt
    │   ├── run_2/
    │   └── run_3/
    ├── wo_convstem/
    ├── wo_attention/
    ├── wo_shpe/
    ├── wo_multiscale/
    └── wo_gate/
```

---

## 示例输出

### LaTeX 表格示例

```latex
\begin{table}[htbp]
\centering
\caption{Ablation Study Results on SST Prediction}
\label{tab:ablation}
\begin{tabular}{lccccc}
\toprule
Model Variant & MSE & RMSE & MAE & R² & Spatial Corr \\
\midrule
Baseline & 0.089±0.003 & 0.298±0.005 & 0.221±0.004 & 0.956±0.002 & 0.978±0.001 \\
w/o ConvStem & 0.112±0.004*** & 0.335±0.006 & 0.254±0.005 & 0.944±0.003 & 0.969±0.002 \\
w/o EfficientRGA & 0.095±0.003* & 0.308±0.005 & 0.229±0.004 & 0.952±0.002 & 0.975±0.001 \\
w/o SHPE & 0.102±0.004** & 0.319±0.006 & 0.239±0.005 & 0.949±0.003 & 0.972±0.002 \\
w/o MultiScale & 0.098±0.003** & 0.313±0.005 & 0.234±0.004 & 0.951±0.002 & 0.974±0.001 \\
w/o Gate & 0.091±0.003 & 0.302±0.005 & 0.224±0.004 & 0.955±0.002 & 0.977±0.001 \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item *p<0.05, **p<0.01, ***p<0.001 compared to Baseline
\end{tablenotes}
\end{table}
```

---

## 常见问题

### Q: 如何处理 GPU 显存不足？

```bash
# 减小 batch size
python -m src.analysis.ablation.runner --batch-size 16

# 或启用梯度累积
python -m src.analysis.ablation.runner --gradient-accumulation 2
```

### Q: 如何跳过失败的实验继续运行？

失败的实验会自动记录在 `experiment_state.json` 中。使用 `--resume` 会跳过已完成的实验，但会重试失败的实验。使用 `--skip-failed` 可以跳过失败的实验。

### Q: 如何自定义图表样式？

```python
from src.analysis.ablation import VisualizationStyle

custom_style = VisualizationStyle(
    name='custom',
    font_size=12,
    primary_colors=['#1f77b4', '#ff7f0e', '#2ca02c'],
    dpi=600
)
```

### Q: 如何添加新的消融变体？

在 `ablation_config.yaml` 中添加新变体：

```yaml
variants:
  # ... existing variants ...
  
  wo_dropout:
    display_name: "w/o Dropout"
    dropout: 0.0
```

或在代码中：

```python
from src.analysis.ablation import AblationConfig

new_variant = AblationConfig(
    name='wo_dropout',
    display_name='w/o Dropout',
    # 自定义参数
)
```

