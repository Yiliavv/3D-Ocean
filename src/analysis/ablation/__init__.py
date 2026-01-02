"""
消融实验模块 (Ablation Study Module)

用于 RGTransformer 模型的系统性消融实验，验证各核心组件对海表温度预测性能的贡献。

主要组件:
- config: 消融实验配置类 (AblationConfig, ExperimentResult, ExperimentState)
- metrics: 评估指标计算 (compute_metrics, significance_test)
- variants: 模型变体工厂 (create_variant_model)
- runner: 实验运行器 (AblationRunner)
"""

from src.analysis.ablation.config import (
    AblationConfig,
    ExperimentResult,
    ExperimentState,
    VisualizationStyle,
    ABLATION_VARIANTS,
    STYLES,
)

from src.analysis.ablation.metrics import (
    compute_metrics,
    significance_test,
)

from src.analysis.ablation.variants import (
    create_variant_model,
)

__all__ = [
    # 配置类
    "AblationConfig",
    "ExperimentResult", 
    "ExperimentState",
    "VisualizationStyle",
    "ABLATION_VARIANTS",
    "STYLES",
    # 指标函数
    "compute_metrics",
    "significance_test",
    # 模型工厂
    "create_variant_model",
]

__version__ = "0.1.0"

