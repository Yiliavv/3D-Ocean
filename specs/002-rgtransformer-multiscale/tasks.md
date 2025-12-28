# Tasks: RGTransformer 多尺度特征增强

**Input**: Design documents from `/specs/002-rgtransformer-multiscale/`  
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, quickstart.md ✅

**Tests**: 本功能规范中明确提到需要单元测试验证，因此包含测试任务。

**Organization**: 任务按用户故事组织，每个故事可独立实现和测试。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: 项目初始化和基础结构准备

- [x] T001 创建 MultiScaleDecoder 模块文件 `src/models/SST/MultiScaleDecoder.py`
- [x] T002 创建多尺度组件单元测试文件 `tests/unit/test_multiscale.py`
- [x] T003 [P] 更新 SST 模块 `__init__.py` 导出新组件 `src/models/SST/__init__.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: 必须先完成的核心基础设施

**⚠️ CRITICAL**: 用户故事实现必须等待此阶段完成

- [x] T004 实现 DecoderStage 类 `src/models/SST/MultiScaleDecoder.py`
  - 包含上采样、通道对齐、特征融合和精炼卷积
  - 参考 data-model.md 第 68-122 行接口设计

- [x] T005 [P] 实现 DecoderStage 单元测试 `tests/unit/test_multiscale.py`
  - 测试尺寸变换正确性
  - 测试跳跃连接融合
  - 测试 NaN 值处理

**Checkpoint**: 基础组件就绪 - 可开始用户故事实现

---

## Phase 3: User Story 1 & 2 - 多尺度特征提取与跳跃连接 (Priority: P1) 🎯 MVP

**Goal**: 实现 MultiScaleConvStem 多尺度编码器，支持跳跃连接输出

**Independent Test**: 验证 MultiScaleConvStem 输出正确的多尺度特征图，各尺度尺寸符合设计

**Note**: US1（多尺度特征提取）和 US2（跳跃连接）在实现上紧密耦合，合并为一个阶段

### Tests for User Story 1 & 2

- [x] T006 [P] [US1] MultiScaleConvStem 单元测试 `tests/unit/test_multiscale.py`
  - 测试 3 个 Stage 输出尺寸正确
  - 测试 skip_features 列表长度和尺寸
  - 测试向后兼容模式（单尺度输出）

- [x] T007 [P] [US2] 跳跃连接融合测试 `tests/unit/test_multiscale.py`
  - 测试 skip 特征与 decoder 特征尺寸匹配
  - 测试加法融合结果正确性

### Implementation for User Story 1 & 2

- [x] T008 [US1] 实现 MultiScaleConvStem 类 `src/models/SST/ConvStem.py`
  - 3 个 Stage: 1→64, 64→128, 128→256
  - 在 Stage 1 和 Stage 2 后输出 skip 特征
  - 参考 data-model.md 第 12-65 行接口设计

- [x] T009 [US2] 实现 MultiScaleDecoder 类 `src/models/SST/MultiScaleDecoder.py`
  - 组合 2 个 DecoderStage
  - 逐层融合跳跃连接特征
  - 最终 1x1 卷积输出单通道
  - 参考 data-model.md 第 125-168 行接口设计

- [x] T010 [US1] 添加 MultiScaleConvStem 参数初始化 `src/models/SST/ConvStem.py`
  - Kaiming 初始化卷积层
  - BatchNorm 权重初始化为 1，偏置为 0

- [x] T011 [US2] 添加 MultiScaleDecoder NaN 处理逻辑 `src/models/SST/MultiScaleDecoder.py`
  - 在融合前检查 NaN 区域
  - 确保 NaN 不传播到有效区域

**Checkpoint**: MultiScaleConvStem 和 MultiScaleDecoder 独立可用

---

## Phase 4: 集成到 RGTransformer (Priority: P1)

**Goal**: 将多尺度组件集成到 RGTransformer 主模型

**Independent Test**: 增强版 RGTransformer 可正常前向传播，输入输出尺寸与基线一致

### Tests for Integration

- [x] T012 [P] 增强版 RGTransformer 集成测试 `tests/unit/test_rgtransformer_v2.py`
  - 测试 `use_multiscale=True` 前向传播
  - 测试 `use_multiscale=False` 向后兼容
  - 测试不同 `num_skip_connections` 配置

### Implementation for Integration

- [x] T013 添加 RGTransformer 新参数 `src/models/SST/RGTransformer.py`
  - `use_multiscale: bool = True`
  - `num_skip_connections: int = 2`
  - `skip_fusion: str = "add"`

- [x] T014 修改 RGTransformer.__init__ 条件初始化 `src/models/SST/RGTransformer.py`
  - 当 `use_multiscale=True` 时使用 MultiScaleConvStem
  - 当 `use_multiscale=True` 时使用 MultiScaleDecoder
  - 保留原始组件作为后备

- [x] T015 修改 RGTransformer._forward_impl 数据流 `src/models/SST/RGTransformer.py`
  - 调用 MultiScaleConvStem 获取 main_feature 和 skip_features
  - 跳跃特征需要在时序维度上聚合（取最后一帧或加权平均）
  - 传递 skip_features 到 MultiScaleDecoder

- [x] T016 更新 save_hyperparameters 保存新参数 `src/models/SST/RGTransformer.py`
  - 确保检查点包含多尺度配置

**Checkpoint**: 增强版 RGTransformer 可训练，接口向后兼容

---

## Phase 5: User Story 3 - 计算效率验证 (Priority: P2)

**Goal**: 验证增强后模型满足性能约束

**Independent Test**: 使用性能分析工具对比基线和增强版的显存/速度

### Tests for User Story 3

- [x] T017 [P] [US3] 性能基准测试 `tests/unit/test_multiscale.py`
  - 测试参数量增加 ≤50%
  - 测试单次前向传播时间增加 ≤30%

### Implementation for User Story 3

- [x] T018 [US3] 添加 get_num_parameters 方法到新模块 `src/models/SST/MultiScaleDecoder.py`
  - 统计可训练参数量
  - 提供分模块统计

- [x] T019 [US3] 添加性能对比脚本逻辑 `tests/unit/test_multiscale.py`
  - 对比基线 vs 增强版参数量
  - 对比推理时间
  - 输出性能报告

- [x] T020 [US3] 优化内存使用（如需要）`src/models/SST/MultiScaleDecoder.py`
  - 使用 inplace 操作减少中间变量
  - 考虑 checkpoint 梯度（如显存超标）

**Checkpoint**: 性能约束验证通过

---

## Phase 6: User Story 4 - 2D 空间注意力 (Priority: P3) ⚠️ 可选

**Goal**: 探索保留空间结构的注意力机制

**Independent Test**: 可视化注意力权重，验证空间局部性

**Note**: 此用户故事为可选优化，可根据 Phase 5 结果决定是否实施

### Research & Design for User Story 4

- [ ] T021 [US4] 调研 2D 空间注意力方案 `specs/002-rgtransformer-multiscale/research.md`
  - Axial Attention
  - Window Attention (Swin Transformer)
  - 局部注意力 + 全局 token

### Implementation for User Story 4 (如决定实施)

- [ ] T022 [US4] 实现 Spatial2DAttention 模块 `src/models/SST/Attention/Spatial2DAttention.py`
  - 保留 H×W 空间结构
  - 在空间维度上应用注意力

- [ ] T023 [US4] 集成 Spatial2DAttention 到 RGTransformer `src/models/SST/RGTransformer.py`
  - 添加 `use_spatial_attention` 配置选项

**Checkpoint**: 2D 空间注意力可用（可选）

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: 完善文档、清理代码、最终验证

- [ ] T024 [P] 更新模块 docstring 和类型注解 `src/models/SST/ConvStem.py`
- [ ] T025 [P] 更新模块 docstring 和类型注解 `src/models/SST/MultiScaleDecoder.py`
- [ ] T026 [P] 更新 RGTransformer 文档字符串 `src/models/SST/RGTransformer.py`
- [ ] T027 运行 quickstart.md 验证步骤 `specs/002-rgtransformer-multiscale/quickstart.md`
- [ ] T028 运行完整测试套件验证无回归 `tests/`
- [ ] T029 代码风格检查和格式化 `src/models/SST/`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - 可立即开始
- **Foundational (Phase 2)**: 依赖 Phase 1 完成 - **阻塞所有用户故事**
- **User Stories (Phase 3-6)**: 依赖 Foundational 完成
  - US1 & US2 (Phase 3): 可在 Foundational 后开始
  - Integration (Phase 4): 依赖 Phase 3 完成
  - US3 (Phase 5): 依赖 Phase 4 完成
  - US4 (Phase 6): 依赖 Phase 4 完成，可与 Phase 5 并行
- **Polish (Phase 7)**: 依赖所有功能阶段完成

### User Story Dependencies

```
Phase 1 (Setup)
    ↓
Phase 2 (Foundational) ← BLOCKS ALL
    ↓
Phase 3 (US1 & US2: MultiScale + Skip)
    ↓
Phase 4 (Integration)
    ↓
┌───────────────┬───────────────┐
↓               ↓               ↓
Phase 5 (US3)   Phase 6 (US4)   [Parallel]
└───────────────┴───────────────┘
    ↓
Phase 7 (Polish)
```

### Within Each Phase

- Tests (if included) → Implementation
- Core modules before integration
- Each checkpoint validates phase independently

### Parallel Opportunities

| Phase | Parallel Tasks |
|-------|----------------|
| Phase 1 | T001, T002, T003 全部可并行 |
| Phase 2 | T004, T005 可并行 |
| Phase 3 | T006, T007 可并行；T010, T011 可并行 |
| Phase 4 | T012 独立测试 |
| Phase 5 | T017 独立测试 |
| Phase 7 | T024, T025, T026 可并行 |

---

## Parallel Example: Phase 3

```bash
# 并行启动 US1 & US2 测试:
Task: "T006 [P] [US1] MultiScaleConvStem 单元测试"
Task: "T007 [P] [US2] 跳跃连接融合测试"

# 并行启动模块初始化:
Task: "T010 [US1] 添加 MultiScaleConvStem 参数初始化"
Task: "T011 [US2] 添加 MultiScaleDecoder NaN 处理逻辑"
```

---

## Implementation Strategy

### MVP First (Phase 1-4 Only)

1. ✅ Complete Phase 1: Setup
2. ✅ Complete Phase 2: Foundational (DecoderStage)
3. ✅ Complete Phase 3: MultiScaleConvStem + MultiScaleDecoder
4. ✅ Complete Phase 4: RGTransformer Integration
5. **STOP and VALIDATE**: 使用 quickstart.md 测试增强版模型
6. 验证 MSE 是否降低

### Incremental Delivery

| 阶段 | 交付物 | 验证方式 |
|------|--------|----------|
| Phase 2 | DecoderStage 模块 | 单元测试 |
| Phase 3 | MultiScaleConvStem + MultiScaleDecoder | 独立前向传播测试 |
| Phase 4 | 增强版 RGTransformer | 完整模型前向传播 |
| Phase 5 | 性能验证报告 | 参数量/速度对比 |
| Phase 6 | 2D 空间注意力 (可选) | 注意力可视化 |

---

## Task Summary

| Phase | Task Count | Priority | Status |
|-------|------------|----------|--------|
| Phase 1: Setup | 3 | - | Pending |
| Phase 2: Foundational | 2 | - | Pending |
| Phase 3: US1 & US2 | 6 | P1 | Pending |
| Phase 4: Integration | 5 | P1 | Pending |
| Phase 5: US3 | 4 | P2 | Pending |
| Phase 6: US4 | 3 | P3 | Optional |
| Phase 7: Polish | 6 | - | Pending |
| **Total** | **29** | - | - |

### MVP Scope (Recommended)

- **必须完成**: Phase 1-4 (16 tasks)
- **建议完成**: Phase 5 (4 tasks) - 验证性能约束
- **可选**: Phase 6 (3 tasks) - 高级优化
- **最终**: Phase 7 (6 tasks) - 代码完善

---

## Notes

- [P] tasks = 不同文件，无依赖
- [Story] label 映射到 spec.md 中的用户故事
- US1 和 US2 合并实现因为技术上紧密耦合
- US4 (P3) 标记为可选，可根据进度决定是否实施
- 每个 Checkpoint 后可独立验证功能
- 提交时按任务或逻辑组提交

