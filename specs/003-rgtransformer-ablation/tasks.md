# Tasks: RGTransformer 消融实验与学术分析

**Input**: Design documents from `/specs/003-rgtransformer-ablation/`  
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, quickstart.md ✓

**Tests**: 包含单元测试任务（用于验证核心功能）

**Organization**: 任务按用户故事分组，支持独立实现和测试

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可并行执行（不同文件，无依赖）
- **[Story]**: 所属用户故事 (US1, US2, ...)
- 所有描述包含精确文件路径

## 项目结构

```text
src/analysis/ablation/         # 消融实验模块
├── __init__.py
├── config.py                  # AblationConfig, ExperimentState
├── runner.py                  # AblationRunner
├── metrics.py                 # 评估指标计算
└── variants.py                # 模型变体工厂

src/plot/ablation.py           # 可视化模块

out/ablation/                  # 输出目录
├── results/
├── figures/
├── tables/
└── checkpoints/
```

---

## Phase 1: Setup (共享基础设施)

**Purpose**: 创建消融实验模块目录结构和基础配置

- [x] T001 创建消融实验模块目录结构 `src/analysis/ablation/`
- [x] T002 创建输出目录结构 `out/ablation/{results,figures,tables,checkpoints}`
- [x] T003 [P] 创建消融实验配置文件 `configs/ablation_config.yaml`
- [x] T004 [P] 创建模块初始化文件 `src/analysis/ablation/__init__.py`

---

## Phase 2: Foundational (阻塞性前置条件)

**Purpose**: 核心基础设施，必须在所有用户故事之前完成

**⚠️ 重要**: 此阶段完成前，不能开始任何用户故事

- [x] T005 实现 AblationConfig 数据类 `src/analysis/ablation/config.py`
- [x] T006 实现 ExperimentResult 数据类 `src/analysis/ablation/config.py`
- [x] T007 [P] 实现 ExperimentState 状态管理类 `src/analysis/ablation/config.py`
- [x] T008 [P] 实现 VisualizationStyle 样式配置类 `src/analysis/ablation/config.py`
- [x] T009 实现评估指标计算函数 compute_metrics() `src/analysis/ablation/metrics.py`
- [x] T010 [P] 实现统计显著性检验函数 significance_test() `src/analysis/ablation/metrics.py`
- [x] T011 实现模型变体工厂 create_variant_model() `src/analysis/ablation/variants.py`
- [x] T012 实现 PatchEmbedding 替代模块（用于 w/o ConvStem）`src/analysis/ablation/variants.py`

**Checkpoint**: 基础设施就绪 - 用户故事实现可以开始

---

## Phase 3: User Story 1 - 运行完整消融实验 (Priority: P1) 🎯 MVP

**Goal**: 研究人员可以通过单一脚本运行所有消融变体实验，获得完整的性能指标

**Independent Test**: 执行 `python -m src.analysis.ablation.runner`，验证所有变体训练完成并生成 CSV 结果文件

### 测试任务

- [x] T013 [P] [US1] 单元测试：AblationConfig 配置验证 `tests/unit/test_ablation.py`
- [x] T014 [P] [US1] 单元测试：模型变体创建 `tests/unit/test_ablation.py`

### 实现任务

- [x] T015 [US1] 实现 AblationRunner 核心类 `src/analysis/ablation/runner.py`
- [x] T016 [US1] 实现单变体训练流程 run_single_experiment() `src/analysis/ablation/runner.py`
- [x] T017 [US1] 实现多变体批量运行 run_all_variants() `src/analysis/ablation/runner.py`
- [x] T018 [US1] 实现断点续传逻辑（基于 ExperimentState）`src/analysis/ablation/runner.py`
- [x] T019 [US1] 实现 CSV 结果导出 export_results_csv() `src/analysis/ablation/runner.py`
- [x] T020 [US1] 实现 YAML 配置保存 save_experiment_config() `src/analysis/ablation/runner.py`
- [x] T021 [US1] 实现命令行入口 `src/analysis/ablation/__main__.py`
- [x] T022 [US1] 添加进度条和日志输出 `src/analysis/ablation/runner.py`

**Checkpoint**: 可以独立运行消融实验并获得 CSV 结果

---

## Phase 4: User Story 2 - 生成学术级可视化图表 (Priority: P1)

**Goal**: 研究人员可以从实验结果生成符合学术标准的高质量图表

**Independent Test**: 给定 `ablation_results.csv`，运行可视化脚本生成 PDF/PNG 图表

### 测试任务

- [x] T023 [P] [US2] 单元测试：可视化样式配置 `tests/unit/test_ablation.py`

### 实现任务

- [x] T024 [US2] 创建可视化模块基础结构 `src/plot/ablation.py`
- [x] T025 [US2] 实现 AGU/IEEE/Nature 学术样式配置 `src/plot/ablation.py`
- [x] T026 [P] [US2] 实现性能对比柱状图（带误差棒）plot_performance_comparison() `src/plot/ablation.py`
- [x] T027 [P] [US2] 实现组件贡献度分析图 plot_component_contribution() `src/plot/ablation.py`
- [x] T028 [P] [US2] 实现预测误差热力图 plot_error_heatmap() `src/plot/ablation.py`
- [x] T029 [US2] 实现批量图表生成 generate_all_figures() `src/plot/ablation.py`
- [x] T030 [US2] 添加 PDF 和 PNG 双格式输出支持 `src/plot/ablation.py`
- [x] T031 [US2] 实现命令行可视化入口 `src/plot/ablation.py`

**Checkpoint**: 可以从 CSV 结果生成学术级图表

---

## Phase 5: User Story 3 - 组件独立性能分析 (Priority: P2)

**Goal**: 研究人员可以单独分析每个组件的性能贡献

**Independent Test**: 选择单个组件（如 ConvStem），运行分析并获得该组件的量化贡献

### 实现任务

- [x] T032 [US3] 实现单组件分析接口 analyze_component() `src/analysis/ablation/runner.py`
- [x] T033 [US3] 实现组件贡献度计算（baseline vs variant 差异）`src/analysis/ablation/metrics.py`
- [x] T034 [P] [US3] 实现组件贡献度堆叠图 plot_component_breakdown() `src/plot/ablation.py`
- [x] T035 [US3] 添加组件分析命令行参数 --component `src/analysis/ablation/__main__.py`

**Checkpoint**: 可以分析单个组件对整体性能的贡献

---

## Phase 6: User Story 4 - 超参数敏感性分析 (Priority: P2)

**Goal**: 研究人员可以分析关键超参数对模型性能的影响

**Independent Test**: 指定 d_model=[128, 256, 512]，运行敏感性分析并生成趋势图

### 实现任务

- [x] T036 [US4] 实现超参数扫描配置 HyperparameterSweep `src/analysis/ablation/config.py`
- [x] T037 [US4] 实现超参数敏感性实验运行器 run_sensitivity_analysis() `src/analysis/ablation/runner.py`
- [x] T038 [P] [US4] 实现单参数敏感性曲线图 plot_sensitivity_curve() `src/plot/ablation.py`
- [x] T039 [P] [US4] 实现多参数热力图 plot_sensitivity_heatmap() `src/plot/ablation.py`
- [x] T040 [US4] 添加敏感性分析命令行参数 --sensitivity `src/analysis/ablation/__main__.py`

**Checkpoint**: 可以分析超参数对性能的影响

---

## Phase 7: User Story 5 - 生成论文级表格 (Priority: P2)

**Goal**: 研究人员可以自动生成 LaTeX 格式的结果表格

**Independent Test**: 给定实验结果，生成可直接用于 LaTeX 的表格文件

### 实现任务

- [x] T041 [US5] 实现 LaTeX 表格生成器 TableGenerator 类 `src/plot/ablation.py`
- [x] T042 [US5] 实现主结果表格生成（含均值/标准差）generate_main_results_table() `src/plot/ablation.py`
- [x] T043 [US5] 实现统计显著性标记（*, **, ***）add_significance_markers() `src/plot/ablation.py`
- [x] T044 [P] [US5] 实现效率对比表格生成 generate_efficiency_table() `src/plot/ablation.py`
- [x] T045 [US5] 添加表格生成命令行参数 --tables `src/plot/ablation.py`

**Checkpoint**: 可以生成论文级 LaTeX 表格

---

## Phase 8: User Story 6 - 计算效率分析 (Priority: P3)

**Goal**: 研究人员可以分析各消融变体的计算效率

**Independent Test**: 运行效率基准测试，获得训练时间、推理速度、显存占用数据

### 实现任务

- [x] T046 [US6] 实现训练时间统计 `src/analysis/ablation/metrics.py`
- [x] T047 [US6] 实现推理速度基准测试 benchmark_inference() `src/analysis/ablation/metrics.py`
- [x] T048 [US6] 实现显存峰值监控 measure_peak_memory() `src/analysis/ablation/metrics.py`
- [x] T049 [P] [US6] 实现精度-效率 Pareto 曲线图 plot_pareto_curve() `src/plot/ablation.py`
- [x] T050 [US6] 实现效率分析汇总 generate_efficiency_report() `src/analysis/ablation/runner.py`
- [x] T051 [US6] 添加效率分析命令行参数 --efficiency `src/analysis/ablation/__main__.py`

**Checkpoint**: 可以进行完整的效率分析

---

## Phase 9: Polish & 跨切面优化

**Purpose**: 影响多个用户故事的改进

- [x] T052 [P] 集成测试：完整消融实验流程 `tests/integration/test_ablation_runner.py`
- [x] T053 [P] 错误处理：GPU 显存不足自动降低 batch size `src/analysis/ablation/runner.py`
- [x] T054 [P] 错误处理：训练失败时跳过并记录错误 `src/analysis/ablation/runner.py`
- [x] T055 代码清理和类型注解完善
- [x] T056 验证 quickstart.md 中所有命令可正常运行

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: 无依赖 - 立即开始
- **Foundational (Phase 2)**: 依赖 Setup 完成 - 阻塞所有用户故事
- **User Stories (Phase 3-8)**: 依赖 Foundational 完成
  - US1 和 US2 可以并行（但 US2 需要 US1 的输出来测试）
  - US3-US6 依赖 US1 完成（需要基础实验框架）
- **Polish (Phase 9)**: 依赖所有用户故事完成

### User Story Dependencies

```
          ┌─────────────────┐
          │ Foundational    │
          │ (Phase 2)       │
          └────────┬────────┘
                   │
       ┌───────────┼───────────┐
       │           │           │
       ▼           ▼           ▼
    ┌──────┐   ┌──────┐   ┌──────────┐
    │ US1  │   │ US2  │   │ US3-US6  │
    │ 实验 │   │ 可视化 │   │ (依赖US1) │
    └──┬───┘   └──────┘   └──────────┘
       │
       ▼
  CSV Results
       │
       ├───────► US2 (验证可视化)
       ├───────► US3 (组件分析)
       ├───────► US4 (敏感性分析)
       └───────► US5 (表格生成)
```

### 各用户故事内部顺序

- **US1**: 配置 → Runner 核心 → 单实验 → 批量运行 → 断点续传 → 导出
- **US2**: 样式配置 → 柱状图 → 贡献图 → 热力图 → 批量生成
- **US3**: 分析接口 → 贡献计算 → 可视化
- **US4**: 扫描配置 → 运行器 → 曲线图 → 热力图
- **US5**: 生成器类 → 主表格 → 显著性 → 效率表格
- **US6**: 时间统计 → 推理基准 → 显存监控 → Pareto 图

### Parallel Opportunities

```bash
# Phase 1 并行任务:
T003, T004 可并行

# Phase 2 并行任务:
T007, T008 可并行（独立数据类）
T009, T010 可并行（独立函数）
T011, T012 可串行（T12 依赖 T11 的接口）

# US1 并行任务:
T013, T014 可并行（独立测试）

# US2 并行任务:
T026, T027, T028 可并行（独立图表函数）

# US4 并行任务:
T038, T039 可并行（独立图表函数）

# US6 并行任务:
T046, T047, T048 可并行（独立指标）
```

---

## Implementation Strategy

### MVP First (仅 User Story 1)

1. 完成 Phase 1: Setup
2. 完成 Phase 2: Foundational (关键 - 阻塞所有故事)
3. 完成 Phase 3: User Story 1 (消融实验运行器)
4. **停止并验证**: 独立测试 US1
5. 如果可用，进行部署/演示

### Incremental Delivery

1. Setup + Foundational → 基础就绪
2. 添加 US1 → 独立测试 → 可运行消融实验 (MVP!)
3. 添加 US2 → 独立测试 → 可生成图表
4. 添加 US3-US5 → 独立测试 → 完整分析能力
5. 添加 US6 → 独立测试 → 效率分析
6. 每个故事都增加价值，不破坏之前的功能

### 推荐执行顺序（单人开发）

```
Day 1: T001-T012 (Setup + Foundational)
Day 2: T013-T022 (US1 - 消融实验)
Day 3: T023-T031 (US2 - 可视化)
Day 4: T032-T040 (US3 + US4 - 组件分析 + 敏感性)
Day 5: T041-T051 (US5 + US6 - 表格 + 效率)
Day 6: T052-T056 (Polish)
```

---

## Notes

- [P] 任务 = 不同文件，无依赖
- [Story] 标签映射到具体用户故事
- 每个用户故事应可独立完成和测试
- 每个任务或逻辑组完成后提交
- 在任何 Checkpoint 停止以独立验证故事
- 避免: 模糊任务、同文件冲突、破坏独立性的跨故事依赖

---

## Task Summary

| Phase | 描述 | 任务数量 |
|-------|------|----------|
| Phase 1 | Setup | 4 |
| Phase 2 | Foundational | 8 |
| Phase 3 | US1 - 消融实验 | 10 |
| Phase 4 | US2 - 可视化 | 9 |
| Phase 5 | US3 - 组件分析 | 4 |
| Phase 6 | US4 - 敏感性分析 | 5 |
| Phase 7 | US5 - 表格生成 | 5 |
| Phase 8 | US6 - 效率分析 | 6 |
| Phase 9 | Polish | 5 |
| **Total** | | **56** |

### 并行机会

- Phase 1: 2 个任务可并行
- Phase 2: 4 个任务可并行
- US1: 2 个测试任务可并行
- US2: 3 个图表任务可并行
- US4: 2 个图表任务可并行
- US6: 3 个指标任务可并行

### 用户故事独立测试标准

| Story | 测试方式 |
|-------|----------|
| US1 | `python -m src.analysis.ablation.runner` 生成 CSV |
| US2 | `python -m src.plot.ablation --results out/ablation/results/ablation_results.csv` 生成 PDF |
| US3 | `python -m src.analysis.ablation.runner --component convstem` 获取组件贡献 |
| US4 | `python -m src.analysis.ablation.runner --sensitivity d_model` 生成趋势图 |
| US5 | `python -m src.plot.ablation --tables` 生成 LaTeX 表格 |
| US6 | `python -m src.analysis.ablation.runner --efficiency` 生成效率报告 |

### MVP 范围

**推荐 MVP**: User Story 1 + User Story 2

完成后可以:
- ✅ 运行完整消融实验
- ✅ 生成学术级图表
- ✅ 导出 CSV 结果
- 足以支撑论文的核心实验部分

