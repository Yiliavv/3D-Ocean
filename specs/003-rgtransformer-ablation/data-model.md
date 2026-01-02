# Data Model: RGTransformer 消融实验

**Feature**: 003-rgtransformer-ablation  
**Date**: 2024-12-28  
**Status**: Complete

## Entity Definitions

### 1. AblationConfig

消融实验配置，定义模型变体的组件启用/禁用状态。

```python
@dataclass
class AblationConfig:
    """消融实验配置"""
    
    # 标识
    name: str                          # 变体名称，如 "baseline", "wo_convstem"
    display_name: str                  # 显示名称，如 "Baseline", "w/o ConvStem"
    
    # 组件开关
    use_conv_stem: bool = True         # 是否使用 ConvStem（否则用 PatchEmbedding）
    use_efficient_attention: bool = True  # 是否使用 EfficientRGAttention
    use_spherical_encoding: bool = True   # 是否使用球谐波编码
    use_multiscale_decoder: bool = True   # 是否使用多尺度解码器
    use_gated_residual: bool = True       # 是否使用门控残差
    
    # 超参数覆盖（可选）
    d_model: Optional[int] = None
    num_heads: Optional[int] = None
    num_attn_layers: Optional[int] = None
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AblationConfig':
        """从字典创建"""
        return cls(**data)
```

**Validation Rules**:
- `name` 必须是有效的 Python 标识符（用于文件命名）
- `display_name` 不能为空
- 至少有一个组件被修改（否则与 baseline 相同）

---

### 2. ExperimentResult

单次实验的完整结果记录。

```python
@dataclass
class ExperimentResult:
    """单次实验结果"""
    
    # 实验标识
    config_name: str                   # 配置名称
    run_id: int                        # 运行编号 (1, 2, 3)
    experiment_id: str                 # 唯一标识: f"{config_name}_run{run_id}"
    
    # 性能指标
    mse: float                         # Mean Squared Error
    rmse: float                        # Root MSE
    mae: float                         # Mean Absolute Error
    r2: float                          # R² 决定系数
    spatial_corr: float                # 空间相关系数
    
    # 效率指标
    train_time_seconds: float          # 训练总时间
    inference_time_ms: float           # 单次推理时间
    peak_memory_mb: float              # 峰值显存占用
    num_parameters: int                # 模型参数量
    
    # 元数据
    checkpoint_path: Path              # 模型权重路径
    timestamp: str                     # ISO 8601 格式时间戳
    seed: int                          # 随机种子
    
    def to_dict(self) -> dict:
        """转换为字典（用于 CSV 导出）"""
        data = asdict(self)
        data['checkpoint_path'] = str(data['checkpoint_path'])
        return data
```

**State Transitions**:
- `PENDING` → `RUNNING` → `COMPLETED` 或 `FAILED`

---

### 3. AblationReport

汇总所有实验结果的报告。

```python
@dataclass
class AblationReport:
    """消融实验汇总报告"""
    
    # 实验配置
    experiment_name: str               # 实验名称
    created_at: str                    # 创建时间
    total_variants: int                # 变体总数
    runs_per_variant: int              # 每变体运行次数
    
    # 结果汇总
    results: List[ExperimentResult]    # 所有实验结果
    
    # 统计分析
    summary_stats: Dict[str, Dict]     # 每变体的均值/标准差
    significance_tests: Dict[str, Dict]  # 显著性检验结果
    
    # 输出路径
    figures_dir: Path                  # 图表输出目录
    tables_dir: Path                   # 表格输出目录
    results_csv: Path                  # CSV 结果文件
    
    def get_variant_results(self, config_name: str) -> List[ExperimentResult]:
        """获取指定变体的所有运行结果"""
        return [r for r in self.results if r.config_name == config_name]
    
    def compute_summary_stats(self) -> Dict[str, Dict]:
        """计算每变体的统计摘要"""
        stats = {}
        for config_name in set(r.config_name for r in self.results):
            variant_results = self.get_variant_results(config_name)
            metrics = ['mse', 'rmse', 'mae', 'r2', 'spatial_corr']
            stats[config_name] = {}
            for metric in metrics:
                values = [getattr(r, metric) for r in variant_results]
                stats[config_name][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        return stats
```

---

### 4. VisualizationStyle

可视化样式配置。

```python
@dataclass
class VisualizationStyle:
    """可视化样式配置"""
    
    # 样式标识
    name: str                          # 如 "agu", "ieee", "nature"
    
    # 字体设置
    font_family: str = 'sans-serif'
    font_size: int = 10
    label_size: int = 11
    title_size: int = 12
    legend_size: int = 9
    
    # 图表尺寸
    single_column_width: float = 3.35  # 英寸
    double_column_width: float = 6.69  # 英寸
    default_height: float = 5.0
    dpi: int = 300
    
    # 颜色方案
    primary_colors: List[str] = field(default_factory=lambda: [
        '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9'
    ])
    error_colormap: str = 'RdBu_r'
    sst_colormap: str = 'jet'
    
    # 输出格式
    save_formats: List[str] = field(default_factory=lambda: ['pdf', 'png'])
    
    def get_matplotlib_rcparams(self) -> dict:
        """获取 matplotlib rc 参数"""
        return {
            'font.family': self.font_family,
            'font.size': self.font_size,
            'axes.labelsize': self.label_size,
            'axes.titlesize': self.title_size,
            'legend.fontsize': self.legend_size,
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
        }


# 预定义样式
STYLES = {
    'agu': VisualizationStyle(name='agu'),
    'ieee': VisualizationStyle(
        name='ieee',
        font_size=8,
        single_column_width=3.5,
        double_column_width=7.16
    ),
    'nature': VisualizationStyle(
        name='nature',
        font_size=7,
        single_column_width=3.5,
        double_column_width=7.0
    ),
}
```

---

### 5. ExperimentState

实验状态管理（用于断点续传）。

```python
@dataclass
class ExperimentState:
    """实验状态（用于断点续传）"""
    
    # 状态文件
    state_file: Path
    
    # 已完成的实验
    completed_experiments: Set[str] = field(default_factory=set)
    
    # 失败的实验
    failed_experiments: Dict[str, str] = field(default_factory=dict)  # id -> error message
    
    # 当前进度
    current_experiment: Optional[str] = None
    
    def save(self):
        """保存状态到文件"""
        data = {
            'completed': list(self.completed_experiments),
            'failed': self.failed_experiments,
            'current': self.current_experiment,
            'updated_at': datetime.now().isoformat()
        }
        self.state_file.write_text(json.dumps(data, indent=2))
    
    @classmethod
    def load(cls, state_file: Path) -> 'ExperimentState':
        """从文件加载状态"""
        state = cls(state_file=state_file)
        if state_file.exists():
            data = json.loads(state_file.read_text())
            state.completed_experiments = set(data.get('completed', []))
            state.failed_experiments = data.get('failed', {})
        return state
```

---

## Entity Relationships

```
┌─────────────────┐
│ AblationConfig  │
│ (消融配置)       │
└────────┬────────┘
         │ 1:N
         ▼
┌─────────────────┐
│ ExperimentResult│
│ (实验结果)       │
└────────┬────────┘
         │ N:1
         ▼
┌─────────────────┐
│ AblationReport  │◄─────┐
│ (汇总报告)       │      │
└─────────────────┘      │
         │               │
         ▼               │
┌─────────────────┐      │
│VisualizationStyle│     │
│ (可视化样式)     │──────┘
└─────────────────┘

┌─────────────────┐
│ ExperimentState │ (独立，用于状态持久化)
│ (实验状态)       │
└─────────────────┘
```

---

## Configuration Files

### YAML 配置示例 (ablation_config.yaml)

```yaml
# 消融实验配置文件
experiment:
  name: "RGTransformer_Ablation_Study"
  runs_per_variant: 3
  seed: 42
  output_dir: "out/ablation"

# 训练参数（所有变体共享）
training:
  epochs: 100
  batch_size: 24
  learning_rate: 0.001
  num_workers: 4

# 模型基础参数
model:
  d_model: 512
  num_heads: 8
  dim_feedforward: 256
  num_attn_layers: 2
  patch_size: 4

# 消融变体定义
variants:
  baseline:
    display_name: "Baseline"
    
  wo_convstem:
    display_name: "w/o ConvStem"
    use_conv_stem: false
    
  wo_attention:
    display_name: "w/o EfficientRGA"
    use_efficient_attention: false
    
  wo_shpe:
    display_name: "w/o SHPE"
    use_spherical_encoding: false
    
  wo_multiscale:
    display_name: "w/o MultiScale"
    use_multiscale_decoder: false
    
  wo_gate:
    display_name: "w/o Gate"
    use_gated_residual: false

# 可视化设置
visualization:
  style: "agu"
  save_formats: ["pdf", "png"]
```

---

## CSV Output Schema

### ablation_results.csv

| Column | Type | Description |
|--------|------|-------------|
| experiment_id | str | 唯一标识 |
| config_name | str | 配置名称 |
| display_name | str | 显示名称 |
| run_id | int | 运行编号 |
| mse | float | Mean Squared Error |
| rmse | float | Root MSE |
| mae | float | Mean Absolute Error |
| r2 | float | R² 决定系数 |
| spatial_corr | float | 空间相关系数 |
| train_time_seconds | float | 训练时间 |
| inference_time_ms | float | 推理时间 |
| peak_memory_mb | float | 峰值显存 |
| num_parameters | int | 参数量 |
| checkpoint_path | str | 权重路径 |
| timestamp | str | 时间戳 |
| seed | int | 随机种子 |

### statistical_tests.csv

| Column | Type | Description |
|--------|------|-------------|
| variant | str | 变体名称 |
| metric | str | 指标名称 |
| baseline_mean | float | 基准均值 |
| baseline_std | float | 基准标准差 |
| variant_mean | float | 变体均值 |
| variant_std | float | 变体标准差 |
| t_statistic | float | t 统计量 |
| p_value | float | p 值 |
| significance | str | 显著性标记 (*, **, ***) |
| effect_size | float | 效应量 (Cohen's d) |

