# Research: RGTransformer 消融实验

**Feature**: 003-rgtransformer-ablation  
**Date**: 2024-12-28  
**Status**: Complete

## 1. 消融实验最佳实践

### Decision: 增量消融策略

采用"移除单一组件"策略，每次仅移除或替换一个组件，保持其他组件不变。

### Rationale

1. **可解释性**: 单变量控制使性能变化可归因于特定组件
2. **统计有效性**: 减少交互效应干扰，结果更可靠
3. **学术惯例**: 符合 NeurIPS/ICML/CVPR 等顶会论文标准

### Alternatives Considered

| 方案 | 优点 | 缺点 | 决定 |
|------|------|------|------|
| 完全析因设计 | 捕获交互效应 | 2^6=64 种组合，计算成本过高 | ❌ 拒绝 |
| 逐步添加 | 展示组件累积效果 | 无法量化单一组件贡献 | ❌ 拒绝 |
| 单一移除 | 简洁，可归因 | 忽略交互效应 | ✅ 采用 |

---

## 2. 统计显著性检验方法

### Decision: 配对 t 检验 + Wilcoxon 符号秩检验

使用配对 t 检验作为主要方法，Wilcoxon 检验作为非参数验证。

### Rationale

1. **配对设计**: 同一数据集上的不同模型，样本天然配对
2. **样本量**: 3 次重复 × 验证集样本，满足 t 检验假设
3. **稳健性**: 双重检验避免分布假设影响结论

### Implementation

```python
from scipy import stats

def significance_test(baseline_scores, variant_scores, alpha=0.05):
    """
    计算统计显著性
    
    Returns:
        p_value: 显著性水平
        significance: *, **, *** 或空字符串
    """
    # 配对 t 检验
    t_stat, p_value = stats.ttest_rel(baseline_scores, variant_scores)
    
    # 显著性标记
    if p_value < 0.001:
        return p_value, '***'
    elif p_value < 0.01:
        return p_value, '**'
    elif p_value < 0.05:
        return p_value, '*'
    return p_value, ''
```

### Alternatives Considered

| 方法 | 适用场景 | 决定 |
|------|----------|------|
| 独立样本 t 检验 | 非配对数据 | ❌ 不适用 |
| ANOVA | 多组比较 | ⚠️ 可选用于超参数敏感性 |
| Bootstrap | 小样本 | ⚠️ 备选 |
| Mann-Whitney U | 非参数 | ❌ 非配对设计 |

---

## 3. 学术图表风格规范

### Decision: 采用 AGU/JGR 期刊风格

基于目标期刊 JGR-Ocean，采用 AGU 官方风格规范。

### Style Specifications

```python
ACADEMIC_STYLE = {
    # 字体设置
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    
    # 图表尺寸 (JGR 单栏: 85mm, 双栏: 170mm)
    'figure.figsize': (6.69, 5),  # 170mm = 6.69 inches
    'figure.dpi': 300,
    
    # 线条和边框
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    
    # 颜色（色盲友好）
    'axes.prop_cycle': cycler('color', [
        '#0072B2',  # Blue
        '#D55E00',  # Vermillion
        '#009E73',  # Green
        '#CC79A7',  # Pink
        '#F0E442',  # Yellow
        '#56B4E9',  # Sky Blue
    ]),
    
    # LaTeX 渲染
    'text.usetex': False,  # 使用 mathtext 替代，避免依赖问题
    'mathtext.fontset': 'stix',
}
```

### Figure Types and Specifications

| 图表类型 | 尺寸 (英寸) | 颜色方案 | 备注 |
|----------|------------|----------|------|
| 柱状图 | 6.69 × 5 | 色盲友好6色 | 带误差棒 |
| 热力图 | 6.69 × 5 | RdBu_r (误差), jet (SST) | 对称色标 |
| 曲线图 | 3.35 × 2.5 | 同上 | 单栏宽度 |
| Pareto | 6.69 × 5 | 渐变色 | 带 Pareto 前沿线 |

### Alternatives Considered

| 风格 | 特点 | 决定 |
|------|------|------|
| IEEE | 更紧凑 | ⚠️ 备选 |
| Nature | 高度精炼 | ⚠️ 备选 |
| Matplotlib 默认 | 不适合出版 | ❌ 拒绝 |

---

## 4. 消融变体实现策略

### Decision: 工厂模式 + 配置驱动

使用工厂函数根据配置创建不同的模型变体。

### Rationale

1. **代码复用**: 避免为每个变体复制模型代码
2. **可扩展性**: 新增变体只需添加配置
3. **可测试性**: 统一接口便于测试

### Implementation Strategy

```python
@dataclass
class AblationConfig:
    """消融实验配置"""
    name: str
    use_conv_stem: bool = True
    use_efficient_attention: bool = True
    use_spherical_encoding: bool = True
    use_multiscale_decoder: bool = True
    use_gated_residual: bool = True


ABLATION_VARIANTS = {
    'baseline': AblationConfig(name='Baseline'),
    'wo_convstem': AblationConfig(name='w/o ConvStem', use_conv_stem=False),
    'wo_attention': AblationConfig(name='w/o EfficientRGA', use_efficient_attention=False),
    'wo_shpe': AblationConfig(name='w/o SHPE', use_spherical_encoding=False),
    'wo_multiscale': AblationConfig(name='w/o MultiScale', use_multiscale_decoder=False),
    'wo_gate': AblationConfig(name='w/o Gate', use_gated_residual=False),
}
```

---

## 5. 断点续传机制

### Decision: 基于状态文件的检查点机制

使用 JSON 状态文件跟踪实验进度，支持中断后恢复。

### Rationale

1. **简单可靠**: 无需复杂的数据库或分布式状态管理
2. **人类可读**: JSON 格式便于调试和手动干预
3. **原子性**: 每个实验完成后更新状态，避免部分写入

### Implementation

```python
class ExperimentState:
    """实验状态管理"""
    
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.completed = self._load_completed()
    
    def _load_completed(self) -> set:
        if self.state_file.exists():
            data = json.loads(self.state_file.read_text())
            return set(data.get('completed', []))
        return set()
    
    def mark_completed(self, experiment_id: str):
        self.completed.add(experiment_id)
        self._save()
    
    def is_completed(self, experiment_id: str) -> bool:
        return experiment_id in self.completed
```

---

## 6. 评估指标实现

### Decision: 统一指标计算接口

所有指标通过单一函数计算，确保一致性。

### Metrics Implementation

```python
def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """
    计算所有评估指标（排除 NaN 区域）
    
    Args:
        y_pred: 预测值 [N, H, W]
        y_true: 真实值 [N, H, W]
    
    Returns:
        dict: 包含 MSE, RMSE, MAE, R², 空间相关系数
    """
    # 创建有效值掩码
    valid_mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    
    y_pred_valid = y_pred[valid_mask]
    y_true_valid = y_true[valid_mask]
    
    mse = np.mean((y_pred_valid - y_true_valid) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred_valid - y_true_valid))
    
    # R²
    ss_res = np.sum((y_true_valid - y_pred_valid) ** 2)
    ss_tot = np.sum((y_true_valid - np.mean(y_true_valid)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # 空间相关系数（每个样本的平均）
    correlations = []
    for i in range(len(y_pred)):
        mask_i = valid_mask[i]
        if mask_i.sum() > 10:
            corr = np.corrcoef(y_pred[i][mask_i], y_true[i][mask_i])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
    spatial_corr = np.mean(correlations) if correlations else 0
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'SpatialCorr': spatial_corr
    }
```

---

## Summary

| 研究主题 | 决定 | 关键理由 |
|----------|------|----------|
| 消融策略 | 单一移除 | 可解释性，学术惯例 |
| 统计检验 | 配对 t + Wilcoxon | 配对设计，稳健性 |
| 图表风格 | AGU/JGR | 目标期刊要求 |
| 变体实现 | 工厂模式 | 代码复用，可扩展 |
| 断点续传 | JSON 状态文件 | 简单可靠 |
| 指标计算 | 统一接口 | NaN 处理一致性 |

所有研究决定已完成，可进入 Phase 1 设计阶段。

