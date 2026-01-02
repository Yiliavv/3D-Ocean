# Implementation Plan: RGTransformer 消融实验与学术分析

**Branch**: `003-rgtransformer-ablation` | **Date**: 2024-12-28 | **Spec**: [spec.md](./spec.md)  
**Input**: Feature specification from `/specs/003-rgtransformer-ablation/spec.md`

## Summary

实现 RGTransformer 模型的完整消融实验框架，系统性验证各核心组件（ConvStem、EfficientRGAttention、球谐波编码、MultiScaleDecoder、门控残差）对海表温度预测性能的贡献。生成学术级可视化图表和 LaTeX 表格，支持论文写作。

## Technical Context

**Language/Version**: Python 3.13+  
**Primary Dependencies**: PyTorch 2.9+, Lightning 2.5+, matplotlib 3.10+, seaborn 0.13+, scipy 1.16+, pandas 2.3+  
**Storage**: CSV (实验结果), YAML (配置), .ckpt (模型权重)  
**Testing**: pytest (单元测试), 手动验证 (可视化)  
**Target Platform**: Linux/Windows with NVIDIA GPU (CUDA 12.8)  
**Project Type**: Single project (科研实验框架)  
**Performance Goals**: 单变体训练 ≤3 小时，完整消融实验 ≤24 小时 (RTX 3090)  
**Constraints**: 显存 ≤24GB，支持断点续传  
**Scale/Scope**: 6 个消融变体 × 3 次重复 = 18 次训练

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| **I. Modular Architecture** | ✅ PASS | 消融配置通过配置类管理，模型变体继承 LightningModule |
| **II. Data Integrity & NaN Handling** | ✅ PASS | 复用现有 `custom_mse_loss`，评估指标排除 NaN |
| **III. Reproducibility & Validation** | ✅ PASS | 使用 seed=42，保存完整配置，多次运行计算统计 |
| **IV. Consistent User Experience** | ✅ PASS | 可视化遵循 Constitution 规范（colormap, 坐标系） |
| **V. Performance & Efficiency** | ✅ PASS | 复用 DataLoader，批处理效率满足要求 |

**Gate Status**: ✅ All principles satisfied, proceed to Phase 0

## Project Structure

### Documentation (this feature)

```text
specs/003-rgtransformer-ablation/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # N/A (internal feature, no API)
├── checklists/
│   └── requirements.md  # Spec quality checklist
└── tasks.md             # Phase 2 output (speckit.tasks)
```

### Source Code (repository root)

```text
src/
├── analysis/
│   ├── ablation/                    # 新增: 消融实验模块
│   │   ├── __init__.py
│   │   ├── config.py                # 消融配置类
│   │   ├── runner.py                # 实验运行器
│   │   ├── metrics.py               # 评估指标计算
│   │   └── variants.py              # 模型变体工厂
│   └── ...
├── plot/
│   ├── ablation.py                  # 新增: 消融实验可视化
│   └── ...
├── models/SST/
│   ├── RGTransformer.py             # 现有: 基础模型
│   └── ...
└── trainer/
    └── ...

out/
└── ablation/                        # 新增: 消融实验输出
    ├── results/
    │   ├── ablation_results.csv
    │   ├── experiment_configs.yaml
    │   └── statistical_tests.csv
    ├── figures/
    │   ├── performance_comparison.pdf
    │   ├── component_contribution.pdf
    │   ├── error_heatmap.pdf
    │   ├── sensitivity_analysis.pdf
    │   └── pareto_curve.pdf
    ├── tables/
    │   ├── main_results.tex
    │   └── efficiency_comparison.tex
    └── checkpoints/
        ├── baseline/
        ├── wo_convstem/
        └── ...

tests/
├── unit/
│   └── test_ablation.py             # 新增: 消融模块单元测试
└── integration/
    └── test_ablation_runner.py      # 新增: 集成测试
```

**Structure Decision**: 使用现有单项目结构，在 `src/analysis/` 下新增 `ablation/` 子模块，可视化在 `src/plot/ablation.py`。输出到 `out/ablation/`。

## Complexity Tracking

> No Constitution violations identified. All features align with existing architecture.

## Phase 0 Output Reference

See [research.md](./research.md) for:
- 消融实验最佳实践研究
- 统计显著性检验方法选择
- 学术图表风格规范

## Phase 1 Output Reference

See [data-model.md](./data-model.md) for:
- AblationConfig 配置结构
- ExperimentResult 结果结构
- VisualizationStyle 样式配置

See [quickstart.md](./quickstart.md) for:
- 快速运行消融实验命令
- 可视化生成命令
- 输出目录说明
