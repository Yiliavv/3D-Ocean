# Implementation Plan: RGTransformer 多尺度特征增强

**Branch**: `002-rgtransformer-multiscale` | **Date**: 2025-12-28 | **Spec**: [spec.md](./spec.md)  
**Input**: Feature specification from `/specs/002-rgtransformer-multiscale/spec.md`

## Summary

增强 RGTransformer 模型的空间特征提取能力，通过引入多尺度 ConvStem、跳跃连接和多尺度解码器，使其具备类似 UNet 的多尺度特征传递能力，同时保持现有时序注意力机制的优势。核心目标是在不显著增加计算成本的前提下提升预测精度。

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: PyTorch 2.x, PyTorch Lightning 2.x, einops  
**Storage**: N/A (模型增强，不涉及存储变更)  
**Testing**: pytest (已有 tests/unit/test_rgtransformer_v2.py)  
**Target Platform**: Linux/Windows 服务器，NVIDIA GPU  
**Project Type**: Single (深度学习模型库)  
**Performance Goals**: 
- 验证集 MSE 降低 ≥5%
- 边界区域 RMSE 降低 ≥10%
**Constraints**: 
- GPU 显存增加 ≤20%
- 单 epoch 训练时间增加 ≤30%
- 参数量增加 ≤50%
**Scale/Scope**: 单一模型增强，影响 3 个核心模块

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Pre-Design Check (Phase 0)

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Modular Architecture | ✅ PASS | 新组件（MultiScaleConvStem, MultiScaleDecoder）独立可测试，继承 nn.Module |
| II. Data Integrity & NaN Handling | ✅ PASS | 跳跃连接中 NaN 区域保持掩码处理，不传播无效值 |
| III. Reproducibility & Validation | ✅ PASS | 增强模型继承 LightningModule，保持相同的日志和检查点机制 |
| IV. Consistent User Experience | ✅ PASS | 输入输出接口不变，向后兼容 |
| V. Performance & Efficiency | ⚠️ MONITOR | 需验证性能约束：显存 ≤20%，速度 ≤30%，参数 ≤50% |

**Gate Result**: ✅ PASS - 可进入 Phase 0

### Post-Design Check (Phase 1)

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Modular Architecture | ✅ PASS | MultiScaleConvStem, DecoderStage, MultiScaleDecoder 均为独立模块 |
| II. Data Integrity & NaN Handling | ✅ PASS | data-model.md 定义了运行时 NaN 断言 |
| III. Reproducibility & Validation | ✅ PASS | 使用相同的 LightningModule 训练框架 |
| IV. Consistent User Experience | ✅ PASS | `use_multiscale=False` 提供完全向后兼容 |
| V. Performance & Efficiency | ✅ PASS | 参数增量 16% < 50%，预估显存增量 19% < 20% |

**Gate Result**: ✅ PASS - 设计完成，可进入 Phase 2 任务分解

## Project Structure

### Documentation (this feature)

```text
specs/002-rgtransformer-multiscale/
├── plan.md              # 本文件
├── research.md          # Phase 0: 技术研究
├── data-model.md        # Phase 1: 模块设计
├── quickstart.md        # Phase 1: 快速开始指南
├── checklists/
│   └── requirements.md  # 规范质量检查
└── tasks.md             # Phase 2: 任务分解 (by /speckit.tasks)
```

### Source Code (repository root)

```text
src/
├── models/
│   └── SST/
│       ├── RGTransformer.py      # 修改: 集成多尺度组件
│       ├── ConvStem.py           # 扩展: 添加 MultiScaleConvStem
│       ├── MultiScaleDecoder.py  # 新增: 多尺度解码器
│       └── Attention/
│           └── RGAttention.py    # 保持不变

tests/
├── unit/
│   ├── test_rgtransformer_v2.py  # 修改: 添加多尺度测试
│   └── test_multiscale.py        # 新增: 多尺度组件单元测试
└── integration/
    └── test_training.py          # 修改: 添加增强模型训练测试
```

**Structure Decision**: 采用 Option 1 (Single project)，在现有 `src/models/SST/` 目录中扩展模块。

## Architecture Design

### 数据流对比

**当前 RGTransformer**:
```
Input [B, T, H, W]
    ↓
ConvStem → [B*T, D, H/4, W/4]
    ↓
Flatten → [B*H'*W', T, D]
    ↓
EfficientRGAttention
    ↓
Reshape → [B, D, H/4, W/4]
    ↓
ConvTranspose2d → [B, 1, H, W]
```

**增强后 RGTransformer**:
```
Input [B, T, H, W]
    ↓
MultiScaleConvStem → {
    scale1: [B*T, D/4, H/2, W/2]   ← Skip1
    scale2: [B*T, D/2, H/4, W/4]   ← Skip2
    scale3: [B*T, D, H/4, W/4]     (main)
}
    ↓
Flatten → [B*H'*W', T, D]
    ↓
EfficientRGAttention (保持不变)
    ↓
Reshape → [B, D, H/4, W/4]
    ↓
MultiScaleDecoder ← Fuse Skip2
    ↓ [B, D/2, H/2, W/2]
MultiScaleDecoder ← Fuse Skip1
    ↓ [B, D/4, H, W]
Final Conv → [B, 1, H, W]
```

### 设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 多尺度层数 | 2 层跳跃连接 | 平衡效果与效率，3 层会超出显存约束 |
| 特征融合方式 | 加法融合 | 比 concat 更省显存，UNet++ 研究表明效果相当 |
| ConvStem 结构 | 扩展现有 ConvStemV2 | 复用残差块设计，保持代码一致性 |
| 注意力模块 | 保持不变 | EfficientRGAttention 已优化，避免引入额外复杂度 |

## Complexity Tracking

> 本方案不违反 Constitution，无需记录复杂度豁免。

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| (无) | - | - |
