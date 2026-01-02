# Feature Specification: RGTransformer 消融实验与学术分析

**Feature Branch**: `003-rgtransformer-ablation`  
**Created**: 2024-12-28  
**Status**: Draft  
**Input**: 为 RGTransformer 模型设计完整的消融实验，生成学术级的分析图像和图表数据

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 运行完整消融实验 (Priority: P1)

研究人员希望通过系统性的消融实验验证 RGTransformer 各核心组件对模型性能的贡献，为学术论文提供实验支撑。

**Why this priority**: 消融实验是学术论文的核心证据，直接决定论文的说服力和可发表性。

**Independent Test**: 可以独立运行消融实验脚本，获得所有实验配置的性能指标（MSE, RMSE, MAE）并保存为 CSV 格式。

**Acceptance Scenarios**:

1. **Given** 已配置好的数据集和基础模型，**When** 执行消融实验脚本，**Then** 系统依次训练所有消融变体并记录性能指标
2. **Given** 消融实验完成后，**When** 查看输出目录，**Then** 存在包含所有变体实验结果的 CSV 文件
3. **Given** 实验中途中断，**When** 重新运行脚本，**Then** 系统能够从断点恢复继续实验

---

### User Story 2 - 生成学术级可视化图表 (Priority: P1)

研究人员需要生成符合学术论文标准的可视化图表，包括性能对比柱状图、组件贡献分析图、误差分布图等。

**Why this priority**: 高质量的可视化图表是学术论文被接收的关键因素，与实验数据同等重要。

**Independent Test**: 给定实验结果 CSV 文件，可以独立运行可视化脚本，生成符合 IEEE/Nature/AGU 风格的高分辨率图表。

**Acceptance Scenarios**:

1. **Given** 消融实验结果 CSV 文件，**When** 运行可视化脚本，**Then** 生成性能对比柱状图（300 DPI, PDF/PNG 格式）
2. **Given** 实验结果数据，**When** 运行可视化脚本，**Then** 生成组件贡献度饼图或堆叠柱状图
3. **Given** 模型预测结果，**When** 运行可视化脚本，**Then** 生成预测误差热力图（展示空间误差分布）

---

### User Story 3 - 组件独立性能分析 (Priority: P2)

研究人员需要单独分析每个核心组件（ConvStem、EfficientRGAttention、MultiScaleDecoder、球谐波编码）的性能贡献。

**Why this priority**: 详细的组件分析能够帮助理解模型工作机制，为模型改进提供方向。

**Independent Test**: 可以选择特定组件进行独立的启用/禁用实验，并获得该组件的量化贡献度。

**Acceptance Scenarios**:

1. **Given** 选择分析 ConvStem 组件，**When** 运行组件分析，**Then** 获得 ConvStem vs PatchEmbedding 的性能对比
2. **Given** 选择分析球谐波编码，**When** 运行组件分析，**Then** 获得有无球谐波编码的性能差异
3. **Given** 选择分析多尺度解码器，**When** 运行组件分析，**Then** 获得跳跃连接对预测精度的影响量化

---

### User Story 4 - 超参数敏感性分析 (Priority: P2)

研究人员需要分析关键超参数（d_model、num_heads、num_attn_layers、patch_size）对模型性能的影响。

**Why this priority**: 超参数敏感性分析可以指导实际应用中的模型配置选择。

**Independent Test**: 可以指定超参数范围，运行敏感性分析并生成趋势图。

**Acceptance Scenarios**:

1. **Given** 指定 d_model 取值范围 [128, 256, 512]，**When** 运行敏感性分析，**Then** 生成 d_model 与性能的关系曲线图
2. **Given** 指定 num_heads 取值范围 [4, 8, 16]，**When** 运行敏感性分析，**Then** 生成注意力头数与性能的关系分析
3. **Given** 多个超参数变化，**When** 运行敏感性分析，**Then** 生成参数-性能热力图

---

### User Story 5 - 生成论文级表格 (Priority: P2)

研究人员需要生成可直接用于论文的 LaTeX 格式表格，包含所有实验结果和统计显著性分析。

**Why this priority**: 标准化的表格格式可以大幅减少论文写作时间。

**Independent Test**: 给定实验结果，可以自动生成 LaTeX 格式的结果汇总表格。

**Acceptance Scenarios**:

1. **Given** 消融实验结果，**When** 运行表格生成脚本，**Then** 输出包含 MSE、RMSE、MAE 及标准差的 LaTeX 表格
2. **Given** 多次实验结果，**When** 运行表格生成脚本，**Then** 表格包含统计显著性标记（*, **, ***）
3. **Given** 模型参数量数据，**When** 运行表格生成脚本，**Then** 表格包含各变体的参数量和计算量对比

---

### User Story 6 - 计算效率分析 (Priority: P3)

研究人员需要分析各消融变体的计算效率，包括训练时间、推理速度、显存占用。

**Why this priority**: 计算效率是模型实用性的重要指标，但优先级低于预测精度。

**Independent Test**: 可以独立运行效率基准测试，获得各变体的时间和内存指标。

**Acceptance Scenarios**:

1. **Given** 多个模型变体，**When** 运行效率基准测试，**Then** 记录每个变体的单步训练时间
2. **Given** 效率测试结果，**When** 生成分析图表，**Then** 包含精度-效率 Pareto 曲线图
3. **Given** 显存监控启用，**When** 运行效率测试，**Then** 记录峰值显存占用

---

### Edge Cases

- 当某个消融变体训练失败时，系统如何处理（跳过并记录错误，继续其他实验）？
- 当 GPU 显存不足时，如何自动调整 batch size 或使用梯度累积？
- 当实验数据量过大时，如何进行分批可视化？

## Requirements *(mandatory)*

### Functional Requirements

#### 消融实验配置

- **FR-001**: 系统 MUST 支持以下核心消融实验配置：
  - Baseline: 完整 RGTransformer 模型
  - w/o ConvStem: 使用简单 PatchEmbedding 替代 ConvStem
  - w/o EfficientRGAttention: 使用标准 MultiheadAttention 替代
  - w/o SphericalHarmonicEncoding: 移除球谐波位置编码
  - w/o MultiScaleDecoder: 使用单层反卷积替代
  - w/o GatedResidual: 移除门控残差连接

- **FR-002**: 系统 MUST 支持通过配置文件定义消融实验组合

- **FR-003**: 系统 MUST 为每个实验变体保存完整的模型权重和训练日志

#### 性能评估

- **FR-004**: 系统 MUST 计算以下评估指标：
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² (决定系数)
  - 空间相关系数

- **FR-005**: 系统 MUST 支持多次运行并计算均值和标准差

- **FR-006**: 系统 MUST 计算统计显著性（t-test 或 Wilcoxon 检验）

#### 可视化生成

- **FR-007**: 系统 MUST 生成以下学术级图表：
  - 消融实验性能对比柱状图（带误差棒）
  - 组件贡献度分析图（堆叠柱状图或分解图）
  - 预测误差空间分布热力图
  - 超参数敏感性曲线图
  - 精度-效率 Pareto 图

- **FR-008**: 系统 MUST 支持输出格式：PDF（矢量图）、PNG（300 DPI）

- **FR-009**: 系统 MUST 支持多种学术风格：IEEE、Nature、AGU

- **FR-010**: 所有图表 MUST 包含清晰的坐标轴标签、图例、标题

#### 数据导出

- **FR-011**: 系统 MUST 导出实验结果为 CSV 格式

- **FR-012**: 系统 MUST 生成 LaTeX 格式的结果表格

- **FR-013**: 系统 MUST 保存可复现的实验配置（YAML/JSON）

### Key Entities

- **AblationConfig**: 消融实验配置，包含启用/禁用的组件列表、超参数设置
- **ExperimentResult**: 单次实验结果，包含指标值、训练时间、模型权重路径
- **AblationReport**: 汇总报告，包含所有实验结果、统计分析、图表路径
- **VisualizationStyle**: 可视化风格配置，包含字体、颜色方案、图表尺寸

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 完整消融实验可在 24 小时内完成（6 个变体，各 3 次重复，RTX 3090 配置）
- **SC-002**: 生成的可视化图表符合学术期刊投稿要求（分辨率 ≥ 300 DPI，矢量格式可用）
- **SC-003**: 实验结果可重现性：相同配置下运行偏差 < 1%
- **SC-004**: 所有生成的表格可直接复制到 LaTeX 论文中使用
- **SC-005**: 统计显著性分析自动标注（p < 0.05 为 *, p < 0.01 为 **, p < 0.001 为 ***）
- **SC-006**: 支持断点续传，中断后恢复实验耗时 < 5 分钟

## Assumptions

1. **硬件环境**: 假设使用 NVIDIA RTX 3090 或同等级别 GPU，显存 ≥ 24GB
2. **数据集**: 使用已有的海表温度数据集，数据预处理已完成
3. **训练配置**: 每个消融变体使用相同的训练超参数（学习率、batch size、epochs）
4. **统计方法**: 使用配对 t 检验或 Wilcoxon 符号秩检验进行显著性分析
5. **可视化库**: 使用 matplotlib + seaborn 生成图表，支持 LaTeX 渲染
6. **实验轮次**: 默认每个配置重复 3 次以获得稳定的统计结果

## Appendix: 消融实验设计详情

### A1. 核心消融变体定义

| 变体名称 | 修改内容 | 预期影响 |
|---------|---------|---------|
| Baseline | 完整 RGTransformer | 基准性能 |
| w/o ConvStem | PatchEmbedding 替代 ConvStem | 边界效应增加，局部特征提取减弱 |
| w/o EfficientRGAttention | 标准 MHA 替代 | 参数量增加，门控机制缺失 |
| w/o SHPE | 移除球谐波编码 | 球面几何信息缺失 |
| w/o MultiScale | 单层反卷积替代 | 多尺度特征丢失 |
| w/o Gate | 移除门控残差 | 残差连接固定权重 |

### A2. 图表规格

| 图表类型 | 尺寸 (英寸) | 格式 | 用途 |
|---------|------------|------|-----|
| 柱状对比图 | 8 × 6 | PDF/PNG | 主要性能对比 |
| 热力图 | 10 × 8 | PDF/PNG | 空间误差分布 |
| 曲线图 | 6 × 4 | PDF/PNG | 敏感性分析 |
| Pareto 图 | 6 × 6 | PDF/PNG | 精度-效率权衡 |

### A3. 输出目录结构

```
out/ablation/
├── results/
│   ├── ablation_results.csv        # 汇总结果
│   ├── experiment_configs.yaml     # 实验配置
│   └── statistical_tests.csv       # 显著性检验结果
├── figures/
│   ├── performance_comparison.pdf  # 性能对比图
│   ├── component_contribution.pdf  # 组件贡献图
│   ├── error_heatmap.pdf          # 误差热力图
│   ├── sensitivity_analysis.pdf    # 敏感性分析
│   └── pareto_curve.pdf           # Pareto 曲线
├── tables/
│   ├── main_results.tex           # 主结果表格
│   └── efficiency_comparison.tex  # 效率对比表格
└── checkpoints/
    ├── baseline/
    ├── wo_convstem/
    └── ...
```
