# Tasks: RGTransformer 模型效率优化

**Input**: Design documents from `/specs/001-rgtransformer-efficiency/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, quickstart.md ✅

**Tests**: 包含性能基准测试（A/B 对比测试）

**Organization**: 任务按用户故事组织，支持独立实现和测试。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可并行执行（不同文件，无依赖）
- **[Story]**: 任务所属用户故事 (US1, US2, US3, US4)
- 路径相对于仓库根目录

---

## Phase 1: Setup（基础设施）

**Purpose**: 项目初始化和性能分析工具

- [x] T001 创建性能分析工具模块 in `src/utils/profiling.py`
- [x] T002 [P] 创建测试目录结构 `tests/unit/` 和 `tests/integration/`
- [x] T003 [P] 安装 einops 依赖并更新 `pyproject.toml`

---

## Phase 2: Foundational（核心组件）

**Purpose**: 构建优化后的核心模块，所有用户故事都依赖这些组件

**⚠️ CRITICAL**: 必须先完成此阶段才能进行用户故事测试

### 2.1 EfficientRGAttention 模块

- [x] T004 [P] 创建 `EfficientRGAttention` 类骨架 in `src/models/SST/Attention/EfficientRGAttention.py`
- [x] T005 实现轻量 Scalar Gate（替代原 131K 参数的门控） in `src/models/SST/Attention/EfficientRGAttention.py`
- [x] T006 移除 Global Token 机制 in `src/models/SST/Attention/EfficientRGAttention.py`
- [x] T007 实现可配置的注意力层数（替代固定递归深度） in `src/models/SST/Attention/EfficientRGAttention.py`

### 2.2 ConvStem 模块

- [x] T008 [P] 创建 `ConvStem` 类（替代 Patch Embedding） in `src/models/SST/ConvStem.py`
- [x] T009 实现多层卷积降采样（3×3 stride=2 两层 + 1×1 投影） in `src/models/SST/ConvStem.py`
- [x] T010 添加 BatchNorm 和 GELU 激活 in `src/models/SST/ConvStem.py`

### 2.3 RGTransformerV2 主模型

- [x] T011 创建 `RGTransformerV2` 类骨架（继承 LightningModule） in `src/models/SST/RGTransformerV2.py`
- [x] T012 集成 `ConvStem` 替代原 Patch Embedding in `src/models/SST/RGTransformerV2.py`
- [x] T013 集成 `EfficientRGAttention` 替代原 RGAttention in `src/models/SST/RGTransformerV2.py`
- [x] T014 使用 einops.rearrange 简化张量操作 in `src/models/SST/RGTransformerV2.py`
- [x] T015 添加 `use_compile` 参数支持 torch.compile in `src/models/SST/RGTransformerV2.py`
- [x] T016 复用原有 `_normalize_sst` 和 `custom_mse_loss` 方法 in `src/models/SST/RGTransformerV2.py`
- [x] T017 实现 training_step/validation_step/configure_optimizers in `src/models/SST/RGTransformerV2.py`

**Checkpoint**: 核心组件完成，可开始用户故事测试

---

## Phase 3: User Story 1 - 模型训练加速 (Priority: P1) 🎯 MVP

**Goal**: 训练速度提升 ≥20%

**Independent Test**: 对比 V1/V2 相同数据集单 epoch 训练时间

### 基准测试

- [x] T018 [US1] 创建训练速度基准测试脚本 in `tests/integration/test_efficiency_benchmark.py`
- [x] T019 [US1] 实现 V1 模型训练计时功能 in `tests/integration/test_efficiency_benchmark.py`
- [x] T020 [US1] 实现 V2 模型训练计时功能 in `tests/integration/test_efficiency_benchmark.py`
- [x] T021 [US1] 添加 epoch 训练时间对比报告 in `tests/integration/test_efficiency_benchmark.py`

### 优化验证

- [x] T022 [US1] 启用 torch.compile 并验证训练兼容性 in `src/models/SST/RGTransformerV2.py`
- [x] T023 [US1] 验证参数量减少（目标 -11%） in `tests/integration/test_efficiency_benchmark.py`

**Checkpoint**: 训练速度提升 ≥20% 验证通过

---

## Phase 4: User Story 2 - 保持或提升预测精度 (Priority: P1)

**Goal**: 验证集 MSE ≤ 105% V1

**Independent Test**: 对比 V1/V2 验证集 MSE

### 精度验证

- [x] T024 [US2] 创建精度对比测试脚本 in `tests/integration/test_accuracy_benchmark.py`
- [x] T025 [US2] 实现 V1 模型验证集评估 in `tests/integration/test_accuracy_benchmark.py`
- [x] T026 [US2] 实现 V2 模型验证集评估 in `tests/integration/test_accuracy_benchmark.py`
- [x] T027 [US2] 添加 MSE/RMSE/R² 对比报告 in `tests/integration/test_accuracy_benchmark.py`

### NaN 处理验证

- [x] T028 [US2] 添加沿海区域 NaN 处理测试 in `tests/unit/test_rgtransformer_v2.py`
- [x] T029 [US2] 添加可视化对比（预测 vs 真实） in `tests/integration/test_accuracy_benchmark.py`

**Checkpoint**: 精度不下降验证通过

---

## Phase 5: User Story 3 - 降低 GPU 显存占用 (Priority: P2)

**Goal**: 峰值显存减少 ≥15%

**Independent Test**: 对比 V1/V2 相同 batch size 下峰值显存

### 显存测试

- [x] T030 [US3] 添加显存监控功能 in `src/utils/profiling.py`
- [x] T031 [US3] 实现峰值显存对比测试 in `tests/integration/test_efficiency_benchmark.py`
- [x] T032 [US3] 添加最大 batch size 测试 in `tests/integration/test_efficiency_benchmark.py`

### 优化验证

- [x] T033 [US3] 验证 memory_format=channels_last 优化 in `src/models/SST/RGTransformerV2.py`

**Checkpoint**: 显存减少 ≥15% 验证通过

---

## Phase 6: User Story 4 - 推理速度提升 (Priority: P2)

**Goal**: 单次推理时间减少 ≥15%

**Independent Test**: 对比 V1/V2 100 次推理平均耗时

### 推理测试

- [x] T034 [US4] 添加推理延迟基准测试 in `tests/integration/test_efficiency_benchmark.py`
- [x] T035 [US4] 实现预热 + 100 次推理计时 in `tests/integration/test_efficiency_benchmark.py`
- [x] T036 [US4] 添加吞吐量（samples/sec）报告 in `tests/integration/test_efficiency_benchmark.py`

### 优化验证

- [x] T037 [US4] 验证 torch.compile mode="reduce-overhead" 推理优化 in `src/models/SST/RGTransformerV2.py`

**Checkpoint**: 推理速度提升 ≥15% 验证通过

---

## Phase 7: A/B 测试与替换

**Purpose**: 综合 A/B 测试并执行替换流程

### A/B 综合测试

- [ ] T038 创建综合 A/B 测试脚本 in `tests/integration/test_ab_comparison.py`
- [ ] T039 运行完整 A/B 测试并生成报告 in `tests/integration/test_ab_comparison.py`
- [ ] T040 验证所有通过标准（训练≤80%、显存≤85%、推理≤85%、MSE≤105%）

### 替换流程（测试通过后）

- [ ] T041 创建备份目录并备份 V1 文件 `src/models/SST/_backup_v1/`
- [ ] T042 重命名 V1 为 Legacy 版本
- [ ] T043 重命名 V2 为主版本
- [ ] T044 更新类名和添加兼容别名 in `src/models/SST/RGTransformer.py`
- [ ] T045 更新 `__init__.py` 导出 in `src/models/SST/__init__.py`
- [ ] T046 运行全量测试验证替换成功

**Checkpoint**: A/B 测试通过，替换完成

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: 文档更新和代码清理

- [ ] T047 [P] 更新 README.md 模型说明
- [ ] T048 [P] 更新 quickstart.md 导入路径
- [ ] T049 [P] 添加 CHANGELOG.md 版本变更记录
- [ ] T050 代码清理和注释完善
- [ ] T051 提交最终变更并创建 PR

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup) ─────────────────────────────────────────────────────┐
        │                                                            │
        ▼                                                            │
Phase 2 (Foundational) ─── BLOCKS ALL USER STORIES ────────────────►│
        │                                                            │
        ├─────────────────┬─────────────────┬─────────────────┐      │
        ▼                 ▼                 ▼                 ▼      │
   Phase 3 (US1)    Phase 4 (US2)    Phase 5 (US3)    Phase 6 (US4) │
   训练加速          精度验证          显存优化          推理加速     │
        │                 │                 │                 │      │
        └─────────────────┴─────────────────┴─────────────────┘      │
                                    │                                │
                                    ▼                                │
                          Phase 7 (A/B 测试与替换)                    │
                                    │                                │
                                    ▼                                │
                          Phase 8 (Polish)                           │
                                                                     ▼
                                                                  DONE
```

### User Story Dependencies

| User Story | 依赖 | 可并行 |
|------------|------|--------|
| US1 (训练加速) | Phase 2 | ✅ |
| US2 (精度验证) | Phase 2 | ✅ |
| US3 (显存优化) | Phase 2 | ✅ |
| US4 (推理加速) | Phase 2 | ✅ |

### Within Each Phase

- T004-T007 (EfficientRGAttention): 顺序执行
- T008-T010 (ConvStem): 顺序执行，与 T004-T007 并行
- T011-T017 (RGTransformerV2): 依赖 T007 和 T010

---

## Parallel Execution Examples

### Phase 2 并行

```bash
# 并行构建两个核心模块
Batch 1:
  - T004 EfficientRGAttention 骨架
  - T008 ConvStem 骨架

# 并行实现细节
Batch 2:
  - T005-T007 EfficientRGAttention 实现
  - T009-T010 ConvStem 实现

# 集成（顺序）
Batch 3:
  - T011-T017 RGTransformerV2 集成
```

### User Stories 并行

```bash
# Phase 2 完成后，所有 User Stories 可并行
Developer A: Phase 3 (US1 训练加速)
Developer B: Phase 4 (US2 精度验证)
Developer C: Phase 5 (US3 显存优化)
Developer D: Phase 6 (US4 推理加速)
```

---

## Implementation Strategy

### MVP First (Phase 1-3)

1. ✅ Complete Phase 1: Setup
2. ✅ Complete Phase 2: Foundational
3. ✅ Complete Phase 3: User Story 1 (训练加速)
4. **STOP and VALIDATE**: 验证训练速度提升 ≥20%
5. 如果通过，继续其他用户故事

### Incremental Delivery

1. Setup + Foundational → 核心组件就绪
2. + US1 (训练加速) → 验证效率提升 → **MVP ✅**
3. + US2 (精度验证) → 确保精度不下降
4. + US3 (显存优化) → 显存减少验证
5. + US4 (推理加速) → 推理优化验证
6. A/B 综合测试 → 替换流程 → **Production Ready ✅**

---

## Summary

| 统计 | 数量 |
|------|------|
| **总任务数** | 51 |
| Phase 1 (Setup) | 3 |
| Phase 2 (Foundational) | 14 |
| Phase 3 (US1 训练加速) | 6 |
| Phase 4 (US2 精度验证) | 6 |
| Phase 5 (US3 显存优化) | 4 |
| Phase 6 (US4 推理加速) | 4 |
| Phase 7 (A/B 测试与替换) | 9 |
| Phase 8 (Polish) | 5 |
| **可并行任务** | 8 |
| **MVP 范围** | Phase 1-3 (23 tasks) |

---

## Notes

- [P] 标记的任务可并行执行
- [US*] 标记任务属于对应用户故事
- 每个用户故事可独立测试
- 每个 Checkpoint 后验证阶段目标
- A/B 测试必须全部通过才能执行替换
- 替换后保留 Legacy 版本以支持回滚

