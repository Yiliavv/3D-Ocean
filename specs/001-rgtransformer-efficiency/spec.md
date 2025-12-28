# Feature Specification: RGTransformer 模型效率优化

**Feature Branch**: `001-rgtransformer-efficiency`  
**Created**: 2025-12-28  
**Status**: Draft  
**Input**: User description: "分析 RGTransformer 的模型结构，我认为现在这个模型结构不够高效"

## 模型架构现状分析

### 当前架构组件

1. **RGTransformer** (`src/models/SST/RGTransformer.py`)
   - 输入: `[batch, seq_len-1, width, height]`
   - Patch Embedding: 使用 Conv2d 将空间网格分块
   - 时序处理: 每个空间位置独立处理时序
   - 全局上下文: 通过 RGAttention 的 Global Token 实现
   - 输出: `[batch, width, height]`

2. **RGAttention** (`src/models/SST/Attention/RGAttention.py`)
   - 类 ViT 的 Global Token ([CLS] Token 机制)
   - 递归泛化自注意力 (Recursion Depth = 2)
   - 门控融合机制

3. **SpatialSphericalHarmonicEncoding** (`src/models/SST/PE/SphericalHarmonicEncoding.py`)
   - 基于球谐波的空间位置编码
   - 在原始分辨率上计算，占用较大计算量

### 识别的效率问题

| 问题 | 当前实现 | 影响 |
|------|----------|------|
| **固定位置 Token 化** | `Conv2d(stride=patch_size)` 无重叠划分 | 边界效应、缺乏平移等变性、跨 patch 特征被割裂 |
| **RGAttention 参数效率低** | 门控机制 131K 参数、Global Token 被丢弃 | 参数量大但收益小，计算冗余 |
| 空间维度展平 | `[B*W'*H', S, D]` 格式处理 | 内存碎片化，批量效率低 |
| 单一 Transformer 块 | 仅一层 Attention + FFN | 表达能力受限 |
| 固定递归深度 | `recursion_depth=2` 硬编码 | 无法动态适应不同复杂度 |
| 时序聚合简单 | 加权求和所有时间步 | 丢失细粒度时序依赖 |
| 位置编码开销 | 原始分辨率球谐波计算 | Patch 级别后冗余计算 |
| 内存访问模式 | 多次 view/permute/contiguous | GPU 计算效率降低 |

#### 固定位置 Token 化问题详解

当前 Patch Embedding 使用 `stride=patch_size` 的卷积，导致：

1. **边界效应**：重要的空间特征（如温度梯度、洋流边界）可能恰好被 patch 边界切割
2. **缺乏平移等变性**：输入数据轻微平移会导致完全不同的 token 划分
3. **信息孤岛**：相邻 patch 在初始 embedding 时无法交互，完全依赖后续 Attention 弥补

**潜在改进方向**：
- 重叠 Patch（Overlapping Patches）：`stride < kernel_size`
- 移位窗口（Shifted Window）：类似 Swin Transformer
- 卷积前处理（Convolutional Stem）：多层小卷积逐步降采样
- 考虑海洋温度场的物理连续性约束

#### RGAttention 参数效率问题详解

当前 RGAttention 模块存在参数量与收益不成正比的问题：

**参数构成分析**（d_model=256）：

| 组件 | 参数量 | 占比 | 收益评估 |
|------|--------|------|----------|
| `MultiheadAttention` | ~263K | 67% | ✅ 核心组件，必要 |
| `gate_projection` | ~131K | 33% | ⚠️ 收益存疑 |
| `global_token` | 256 | <0.1% | ⚠️ 输出被丢弃 |
| `step_weights` + `temperature` | 3 | <0.01% | ⚠️ 作用有限 |

**具体问题**：

1. **门控机制开销大**：`gate_projection` 占 33% 参数（131K），但可能退化为简单残差连接
   ```python
   gate = sigmoid(gate_projection([current_state, attn_output]))
   output = gate * attn_output + (1 - gate) * current_state
   ```
   如果 gate ≈ 0.5，等价于 `0.5 * (attn_output + current_state)`

2. **递归计算重复**：`recursion_depth=2` 使 attention 计算 2 次，但共享参数，收益边际递减

3. **Global Token 被丢弃**：
   ```python
   output = accumulated_output[:, 1:, :]  # 丢弃 Global Token
   ```
   Global Token 仅作为 attention 中的"信息枢纽"，但其输出未被直接使用

4. **加权累积可能冗余**：`step_weights` 的加权求和与直接使用最后一层输出差异可能很小

**潜在改进方向**：
- 移除或简化门控机制，改用简单残差连接
- 减少递归深度或改用多层独立 Transformer 块
- 利用 Global Token 的输出作为额外监督信号
- 使用更轻量的注意力变体（如 Linear Attention）

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 模型训练加速 (Priority: P1)

研究人员在训练 RGTransformer 时，期望在相同硬件条件下获得更快的训练速度，从而能够更快地进行实验迭代。

**Why this priority**: 训练速度直接影响研究效率，是提升模型效率的核心目标。

**Independent Test**: 可通过对比优化前后相同数据集上的单 epoch 训练时间来验证，预期训练时间减少 20% 以上即为成功。

**Acceptance Scenarios**:

1. **Given** 相同的训练数据集和硬件配置，**When** 使用优化后的模型训练一个完整 epoch，**Then** 训练时间比优化前减少至少 20%
2. **Given** 固定的 GPU 显存限制，**When** 使用优化后的模型，**Then** 可以使用更大的 batch size

---

### User Story 2 - 保持或提升预测精度 (Priority: P1)

研究人员期望在优化效率的同时，模型的海表温度预测精度不会下降，最好能有所提升。

**Why this priority**: 效率优化不能以牺牲模型性能为代价，这是优化的基本约束条件。

**Independent Test**: 可通过在验证集上对比优化前后模型的 MSE、RMSE、相关系数等指标来验证。

**Acceptance Scenarios**:

1. **Given** 相同的训练配置和数据集，**When** 优化后的模型完成训练，**Then** 验证集 MSE 不高于优化前模型
2. **Given** 相同的测试数据，**When** 使用优化后模型进行预测，**Then** 预测精度（RMSE）不低于优化前模型

---

### User Story 3 - 降低 GPU 显存占用 (Priority: P2)

研究人员期望模型在推理和训练时占用更少的 GPU 显存，以便在资源受限的环境下也能运行。

**Why this priority**: 显存优化可以使模型在更多设备上可用，同时为增大 batch size 提供空间。

**Independent Test**: 可通过 PyTorch 的显存监控工具对比优化前后峰值显存占用来验证。

**Acceptance Scenarios**:

1. **Given** 相同的 batch size 和输入尺寸，**When** 使用优化后的模型进行前向传播，**Then** 峰值 GPU 显存占用减少至少 15%
2. **Given** 固定的 GPU 显存限制，**When** 使用优化后的模型，**Then** 可支持的最大 batch size 增加

---

### User Story 4 - 推理速度提升 (Priority: P2)

研究人员期望模型在推理阶段能够更快地生成预测结果，以便实时或准实时应用场景。

**Why this priority**: 推理速度影响模型的实际应用价值和用户体验。

**Independent Test**: 可通过对比优化前后单次推理的平均耗时来验证。

**Acceptance Scenarios**:

1. **Given** 相同的测试数据，**When** 使用优化后的模型进行推理，**Then** 单次推理时间减少至少 15%

---

### Edge Cases

- 当输入序列长度 `seq_len` 非常短（如 2）或非常长（如 30+）时，优化是否仍然有效？
- 当空间分辨率变化（patch_size 不同）时，优化策略是否通用？
- 当 batch size 为 1 时（推理场景），效率提升是否依然显著？
- 当 GPU 显存极度受限时，模型是否能优雅降级？

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 优化后的模型必须保持与原模型相同的输入输出接口（API 兼容性）
- **FR-002**: 优化后的模型必须支持与原模型相同的训练流程（继承 LightningModule）
- **FR-003**: 优化后的模型必须能够加载原模型的预训练权重（向后兼容）
- **FR-004**: 优化策略必须不改变模型的核心预测逻辑和理论基础
- **FR-005**: 优化后的模型必须在相同配置下达到不低于原模型的预测精度
- **FR-006**: 优化后的模型必须减少至少 20% 的训练时间或 15% 的显存占用
- **FR-007**: 优化实现必须保持代码可读性和可维护性

### Key Entities

- **RGTransformer**: 主模型类，包含 Patch Embedding、Transformer 块、时序聚合和 Patch Recovery
- **RGAttention**: 递归泛化自注意力模块，包含 Global Token 和门控融合机制
- **SpatialSphericalHarmonicEncoding**: 球谐波空间位置编码模块
- **ChannelFeedForward**: 通道混合前馈网络

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 单 epoch 训练时间减少至少 20%（在相同硬件和数据集条件下）
- **SC-002**: 验证集 MSE 不高于优化前模型的 105%（允许 5% 误差范围内波动）
- **SC-003**: 峰值 GPU 显存占用减少至少 15%（在相同 batch size 条件下）
- **SC-004**: 单次推理时间减少至少 15%（在相同输入条件下）
- **SC-005**: 优化后模型参数量变化不超过原模型的 ±20%

## Assumptions

- 用户使用 PyTorch 2.0+ 版本，支持现代编译优化特性
- 目标硬件为 NVIDIA GPU，支持 CUDA 11.0+
- 训练数据格式和预处理流程保持不变
- 模型的核心架构（Patch Embedding + Transformer + Recovery）保持不变，仅优化实现细节
