# Implementation Plan: RGTransformer 模型效率优化

**Branch**: `001-rgtransformer-efficiency` | **Date**: 2025-12-28 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-rgtransformer-efficiency/spec.md`

## Summary

优化 RGTransformer 模型的计算效率和参数效率，主要针对以下问题：
1. **固定位置 Token 化**：改进 Patch Embedding 策略，解决边界效应和平移等变性问题
2. **RGAttention 参数效率低**：简化门控机制，减少冗余计算
3. **内存访问模式**：优化张量操作，减少 view/permute/contiguous 调用

目标：训练速度提升 ≥20%，显存占用减少 ≥15%，同时保持预测精度。

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: PyTorch 2.0+, PyTorch Lightning, NumPy  
**Storage**: N/A（模型优化，不涉及数据存储变更）  
**Testing**: pytest + PyTorch profiler + 性能基准测试  
**Target Platform**: Linux/Windows, NVIDIA GPU (CUDA 11.0+)  
**Project Type**: Single（深度学习模型优化）  
**Performance Goals**:
- 单 epoch 训练时间减少 ≥20%
- 峰值 GPU 显存减少 ≥15%
- 单次推理时间减少 ≥15%

**Constraints**:
- API 兼容性：必须保持与原模型相同的输入输出接口
- 精度约束：验证集 MSE 不高于原模型的 105%
- 参数量：变化不超过 ±20%

**Scale/Scope**:
- 影响文件：3-4 个模型文件
- 输入数据规模：全球海洋温度场 (360×180 网格)
- 时序长度：2-30 个时间步

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| 原则 | 要求 | 合规状态 | 备注 |
|------|------|----------|------|
| **I. Modular Architecture** | 继承 LightningModule | ✅ 通过 | 保持现有架构，仅优化实现 |
| **II. Data Integrity & NaN Handling** | 正确处理 NaN 值 | ✅ 通过 | 不修改 NaN 处理逻辑 |
| **III. Reproducibility & Validation** | 可复现性 | ✅ 通过 | 保持随机种子和数据分割方式 |
| **IV. Consistent User Experience** | 一致的用户体验 | ✅ 通过 | API 接口保持不变 |
| **V. Performance & Efficiency** | 性能标准 | 🎯 目标 | 本次优化的核心目标 |

### Performance Standards 检查

| 标准 | 要求 | 当前状态 | 优化目标 |
|------|------|----------|----------|
| SST Prediction RMSE | ≤ 0.50°C | 待验证 | 保持或改善 |
| Inference Latency | < 1 second | 待测量 | 减少 ≥15% |
| Model Size | < 500MB | ✅ 符合 | 不显著增加 |

### Quality Gates

- [ ] 通过精度目标验证集测试
- [ ] 可视化检查物理合理性
- [ ] NaN 处理验证（沿海区域）
- [ ] 时序一致性检查
- [ ] 训练日志归档

## Project Structure

### Documentation (this feature)

```text
specs/001-rgtransformer-efficiency/
├── plan.md              # This file
├── research.md          # Phase 0: 优化策略研究
├── data-model.md        # Phase 1: 模型结构设计
├── quickstart.md        # Phase 1: 快速开始指南
└── checklists/
    └── requirements.md  # 需求检查清单
```

### Source Code (repository root)

```text
src/
├── models/
│   └── SST/
│       ├── RGTransformer.py          # 主模型（待优化）
│       ├── RGTransformerV2.py        # 优化后的新版本（新建）
│       ├── Attention/
│       │   ├── RGAttention.py        # 原注意力模块
│       │   └── EfficientRGAttention.py  # 优化后的注意力（新建）
│       └── PE/
│           └── SphericalHarmonicEncoding.py  # 位置编码（可能优化）
├── trainer/
│   └── BaseTrainer.py                # 训练器（不变）
└── utils/
    └── profiling.py                  # 性能分析工具（新建）

tests/
├── unit/
│   └── test_rgtransformer_v2.py      # 单元测试（新建）
└── integration/
    └── test_efficiency_benchmark.py  # 性能基准测试（新建）
```

**Structure Decision**: 采用 Option 1 (Single project)，新建优化版本文件而非直接修改原文件，便于 A/B 对比测试。

## A/B 测试与替换流程

### 测试通过标准

A/B 对比测试通过的条件（**全部满足**）：

| 指标 | 通过标准 | 验证方法 |
|------|----------|----------|
| 训练速度 | V2 ≤ 80% V1 时间 | 相同数据集单 epoch 计时 |
| 显存占用 | V2 ≤ 85% V1 峰值 | torch.cuda.max_memory_allocated |
| 推理延迟 | V2 ≤ 85% V1 延迟 | 100 次推理平均耗时 |
| 预测精度 | V2 MSE ≤ 105% V1 MSE | 验证集评估 |
| NaN 处理 | 沿海区域正确 | 可视化检查 |

### 替换操作步骤

**前置条件**: A/B 测试全部通过 ✅

#### Step 1: 备份原文件

```bash
# 创建备份目录
mkdir -p src/models/SST/_backup_v1

# 备份原文件
cp src/models/SST/RGTransformer.py src/models/SST/_backup_v1/RGTransformer.py
cp src/models/SST/Attention/RGAttention.py src/models/SST/_backup_v1/RGAttention.py
```

#### Step 2: 重命名文件

```bash
# 将 V1 重命名为 Legacy
mv src/models/SST/RGTransformer.py src/models/SST/RGTransformerLegacy.py
mv src/models/SST/Attention/RGAttention.py src/models/SST/Attention/RGAttentionLegacy.py

# 将 V2 重命名为主版本
mv src/models/SST/RGTransformerV2.py src/models/SST/RGTransformer.py
mv src/models/SST/Attention/EfficientRGAttention.py src/models/SST/Attention/RGAttention.py
```

#### Step 3: 更新导入路径

```python
# 在新的 RGTransformer.py 中，确保类名为 RGTransformer
# 旧代码: class RGTransformerV2(LightningModule):
# 新代码: class RGTransformer(LightningModule):

# 添加别名以保持向后兼容
RGTransformerV2 = RGTransformer  # 兼容旧代码引用
```

#### Step 4: 更新 __init__.py

```python
# src/models/SST/__init__.py
from .RGTransformer import RGTransformer
from .RGTransformerLegacy import RGTransformer as RGTransformerLegacy  # 保留旧版本访问

# src/models/SST/Attention/__init__.py
from .RGAttention import EfficientRGAttention as RGAttention
from .RGAttentionLegacy import RGAttention as RGAttentionLegacy
```

#### Step 5: 验证替换

```bash
# 运行单元测试
pytest tests/unit/test_rgtransformer_v2.py -v

# 运行集成测试
pytest tests/integration/test_efficiency_benchmark.py -v

# 运行原有测试（确保 API 兼容）
pytest tests/ -v --ignore=tests/unit/test_rgtransformer_v2.py
```

#### Step 6: 更新文档

- [ ] 更新 README.md 中的模型说明
- [ ] 更新 quickstart.md 中的导入路径
- [ ] 在 CHANGELOG.md 中记录版本变更

#### Step 7: 提交变更

```bash
git add .
git commit -m "refactor: replace RGTransformer with optimized V2 version

- 训练速度提升 XX%
- 显存占用减少 XX%
- 参数量减少 11%
- API 完全兼容，旧版本保留为 RGTransformerLegacy"
```

### 回滚方案

如果替换后发现问题，可快速回滚：

```bash
# 从备份恢复
cp src/models/SST/_backup_v1/RGTransformer.py src/models/SST/RGTransformer.py
cp src/models/SST/_backup_v1/RGAttention.py src/models/SST/Attention/RGAttention.py

# 或者使用 git 回滚
git revert HEAD
```

## Complexity Tracking

> 无 Constitution 违规需要说明

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | N/A | N/A |
