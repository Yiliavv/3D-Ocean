"""
消融实验配置模块

包含:
- AblationConfig: 消融实验配置数据类
- ExperimentResult: 单次实验结果数据类
- ExperimentState: 实验状态管理（断点续传）
- VisualizationStyle: 可视化样式配置
- HyperparameterSweep: 超参数扫描配置
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Set, Any
import yaml


@dataclass
class AblationConfig:
    """
    消融实验配置
    
    定义模型变体的组件启用/禁用状态和超参数覆盖。
    """
    # 标识
    name: str                              # 变体名称，如 "baseline", "wo_convstem"
    display_name: str = ""                 # 显示名称，如 "Baseline", "w/o ConvStem"
    description: str = ""                  # 描述
    
    # 组件开关
    use_conv_stem: bool = True             # 是否使用 ConvStem（否则用 PatchEmbedding）
    use_efficient_attention: bool = True   # 是否使用 RGAttention（门控注意力）
    use_spherical_encoding: bool = True    # 是否使用球谐波编码
    use_multiscale_decoder: bool = False   # 是否使用多尺度解码器（默认关闭，与正常训练一致）
    use_gated_residual: bool = True        # 是否使用门控残差
    
    # 超参数覆盖（可选）
    d_model: Optional[int] = None
    num_heads: Optional[int] = None
    num_attn_layers: Optional[int] = None
    patch_size: Optional[int] = None
    
    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name.replace("_", " ").title()
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AblationConfig':
        """从字典创建"""
        # 过滤掉不存在的字段
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
    
    def get_model_kwargs(self, base_params: dict) -> dict:
        """
        获取模型参数，合并基础参数和覆盖值
        
        Args:
            base_params: 基础模型参数字典
            
        Returns:
            合并后的模型参数
        """
        kwargs = base_params.copy()
        
        # 应用超参数覆盖
        if self.d_model is not None:
            kwargs['d_model'] = self.d_model
        if self.num_heads is not None:
            kwargs['num_heads'] = self.num_heads
        if self.num_attn_layers is not None:
            kwargs['num_attn_layers'] = self.num_attn_layers
        if self.patch_size is not None:
            kwargs['patch_size'] = self.patch_size
            
        # 组件开关
        kwargs['use_multiscale'] = self.use_multiscale_decoder
        
        return kwargs


@dataclass
class ExperimentResult:
    """
    单次实验结果
    
    记录实验的完整性能指标、效率指标和元数据。
    """
    # 实验标识
    config_name: str                       # 配置名称
    display_name: str                      # 显示名称
    run_id: int                            # 运行编号 (1, 2, 3)
    experiment_id: str = ""                # 唯一标识: f"{config_name}_run{run_id}"
    
    # 性能指标
    mse: float = 0.0                       # Mean Squared Error
    rmse: float = 0.0                      # Root MSE
    mae: float = 0.0                       # Mean Absolute Error
    r2: float = 0.0                        # R² 决定系数
    spatial_corr: float = 0.0             # 空间相关系数
    
    # 效率指标
    train_time_seconds: float = 0.0        # 训练总时间
    inference_time_ms: float = 0.0         # 单次推理时间
    peak_memory_mb: float = 0.0            # 峰值显存占用
    num_parameters: int = 0                # 模型参数量
    
    # 元数据
    checkpoint_path: str = ""              # 模型权重路径
    timestamp: str = ""                    # ISO 8601 格式时间戳
    seed: int = 42                         # 随机种子
    
    def __post_init__(self):
        if not self.experiment_id:
            self.experiment_id = f"{self.config_name}_run{self.run_id}"
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        """转换为字典（用于 CSV 导出）"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ExperimentResult':
        """从字典创建"""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass 
class ExperimentState:
    """
    实验状态管理（用于断点续传）
    
    使用 JSON 文件持久化实验进度，支持中断后恢复。
    """
    state_file: Path
    
    # 已完成的实验
    completed_experiments: Set[str] = field(default_factory=set)
    
    # 失败的实验
    failed_experiments: Dict[str, str] = field(default_factory=dict)  # id -> error message
    
    # 当前进度
    current_experiment: Optional[str] = None
    
    # 结果缓存
    results: List[ExperimentResult] = field(default_factory=list)
    
    def __post_init__(self):
        if isinstance(self.state_file, str):
            self.state_file = Path(self.state_file)
        # 加载已有状态
        self._load()
    
    def _load(self):
        """从文件加载状态"""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding='utf-8'))
                self.completed_experiments = set(data.get('completed', []))
                self.failed_experiments = data.get('failed', {})
                self.current_experiment = data.get('current')
                # 加载结果
                results_data = data.get('results', [])
                self.results = [ExperimentResult.from_dict(r) for r in results_data]
            except (json.JSONDecodeError, KeyError):
                pass
    
    def save(self):
        """保存状态到文件"""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'completed': list(self.completed_experiments),
            'failed': self.failed_experiments,
            'current': self.current_experiment,
            'results': [r.to_dict() for r in self.results],
            'updated_at': datetime.now().isoformat()
        }
        self.state_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    
    def mark_started(self, experiment_id: str):
        """标记实验开始"""
        self.current_experiment = experiment_id
        self.save()
    
    def mark_completed(self, experiment_id: str, result: ExperimentResult):
        """标记实验完成"""
        self.completed_experiments.add(experiment_id)
        self.results.append(result)
        self.current_experiment = None
        # 从失败列表中移除（如果有）
        self.failed_experiments.pop(experiment_id, None)
        self.save()
    
    def mark_failed(self, experiment_id: str, error: str):
        """标记实验失败"""
        self.failed_experiments[experiment_id] = error
        self.current_experiment = None
        self.save()
    
    def is_completed(self, experiment_id: str) -> bool:
        """检查实验是否已完成"""
        return experiment_id in self.completed_experiments
    
    def is_failed(self, experiment_id: str) -> bool:
        """检查实验是否失败"""
        return experiment_id in self.failed_experiments
    
    def get_pending_experiments(self, all_experiments: List[str]) -> List[str]:
        """获取待运行的实验列表"""
        return [e for e in all_experiments 
                if e not in self.completed_experiments]
    
    @classmethod
    def load(cls, state_file: Path) -> 'ExperimentState':
        """从文件加载状态"""
        return cls(state_file=state_file)


@dataclass
class VisualizationStyle:
    """
    可视化样式配置
    
    定义学术图表的字体、颜色、尺寸等样式参数。
    """
    # 样式标识
    name: str                              # 如 "agu", "ieee", "nature"
    
    # 字体设置
    font_family: str = 'serif'
    font_size: int = 10
    label_size: int = 11
    title_size: int = 12
    legend_size: int = 9
    tick_size: int = 9
    
    # 图表尺寸
    single_column_width: float = 3.35      # 英寸
    double_column_width: float = 6.69      # 英寸
    default_height: float = 5.0
    dpi: int = 300
    
    # 颜色方案（色盲友好）
    primary_colors: List[str] = field(default_factory=lambda: [
        '#0072B2',  # Blue
        '#D55E00',  # Vermillion
        '#009E73',  # Green
        '#CC79A7',  # Pink
        '#F0E442',  # Yellow
        '#56B4E9',  # Sky Blue
    ])
    error_colormap: str = 'RdBu_r'
    sst_colormap: str = 'jet'
    
    # 输出格式
    save_formats: List[str] = field(default_factory=lambda: ['pdf', 'png'])
    
    def get_matplotlib_rcparams(self) -> dict:
        """获取 matplotlib rc 参数"""
        return {
            'font.family': self.font_family,
            'font.serif': ['Times New Roman'],
            'font.size': self.font_size,
            'axes.labelsize': self.label_size,
            'axes.titlesize': self.title_size,
            'legend.fontsize': self.legend_size,
            'xtick.labelsize': self.tick_size,
            'ytick.labelsize': self.tick_size,
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'figure.figsize': (self.double_column_width, self.default_height),
        }
    
    def get_figure_size(self, width: str = 'double', aspect: float = 0.75) -> tuple:
        """
        获取图表尺寸
        
        Args:
            width: 'single' 或 'double'
            aspect: 高宽比
            
        Returns:
            (width, height) 元组
        """
        w = self.single_column_width if width == 'single' else self.double_column_width
        return (w, w * aspect)


@dataclass
class HyperparameterSweep:
    """
    超参数扫描配置
    
    用于敏感性分析的参数范围定义。
    """
    param_name: str                        # 参数名称
    values: List[Any]                      # 参数值列表
    display_name: str = ""                 # 显示名称
    
    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.param_name.replace("_", " ").title()


# ============================================================
# 预定义配置
# ============================================================

# 消融变体配置
ABLATION_VARIANTS: Dict[str, AblationConfig] = {
    'baseline': AblationConfig(
        name='baseline',
        display_name='Baseline',
        description='完整 RGTransformer 模型'
    ),
    'wo_convstem': AblationConfig(
        name='wo_convstem',
        display_name='w/o ConvStem',
        description='使用 PatchEmbedding 替代 ConvStem',
        use_conv_stem=False
    ),
    'wo_attention': AblationConfig(
        name='wo_attention',
        display_name='w/o EfficientRGA',
        description='使用标准 MultiheadAttention 替代',
        use_efficient_attention=False
    ),
    'wo_shpe': AblationConfig(
        name='wo_shpe',
        display_name='w/o SHPE',
        description='移除球谐波位置编码',
        use_spherical_encoding=False
    ),
    'wo_multiscale': AblationConfig(
        name='wo_multiscale',
        display_name='w/o MultiScale',
        description='使用单层反卷积替代多尺度解码器',
        use_multiscale_decoder=False
    ),
    'wo_gate': AblationConfig(
        name='wo_gate',
        display_name='w/o Gate',
        description='移除门控残差连接',
        use_gated_residual=False
    ),
}

# 可视化样式预设
STYLES: Dict[str, VisualizationStyle] = {
    'agu': VisualizationStyle(
        name='agu',
        font_size=10,
        label_size=11,
        title_size=12,
        single_column_width=3.35,  # 85mm
        double_column_width=6.69,  # 170mm
    ),
    'ieee': VisualizationStyle(
        name='ieee',
        font_size=8,
        label_size=9,
        title_size=10,
        single_column_width=3.5,
        double_column_width=7.16,
    ),
    'nature': VisualizationStyle(
        name='nature',
        font_size=7,
        label_size=8,
        title_size=9,
        single_column_width=3.5,
        double_column_width=7.0,
    ),
}


def load_config_from_yaml(config_path: Path) -> dict:
    """
    从 YAML 文件加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

