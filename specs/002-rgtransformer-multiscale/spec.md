# Feature Specification: RGTransformer 多尺度特征增强

**Feature Branch**: `002-rgtransformer-multiscale`  
**Created**: 2025-12-28  
**Status**: Draft  
**Input**: User description: "提高 RGTransformer 的能力，经过和 UNetLSTM 对比之后，发现 RGTransformer 在空间特征提取、跳跃连接、时序建模和解码器方面存在差距。希望添加跳跃连接、使用 2D 注意力保留空间结构、增加 ConvStem 深度，同时不增加 GPU 内存负担和影响训练速度。"

## 背景分析

### 当前 RGTransformer 与 UNetLSTM 对比

| 特性 | UNetLSTM | RGTransformer（当前） |
|------|----------|----------------------|
| 空间特征提取 | 多尺度编码器 (64→128→256→512) | ConvStem 单一尺度 (→256) |
| 跳跃连接 | ✅ 4层跳跃连接 | ❌ 无跳跃连接 |
| 时序建模 | ConvLSTM（保留空间结构） | Attention（空间位置展平） |
| 解码器 | 多尺度逐层上采样 | 单次 ConvTranspose |

### 设计约束

- **内存约束**: 不显著增加 GPU 显存占用
- **速度约束**: 不降低训练速度
- **架构约束**: 需与现有 RGTransformer 结构融合，而非简单堆叠

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 多尺度特征提取提升预测精度 (Priority: P1)

研究人员希望 RGTransformer 能够捕获不同空间尺度的海温特征（如大尺度涡旋、中尺度涡、小尺度温度梯度），从而提升预测精度。

**Why this priority**: 多尺度特征是 UNetLSTM 优于当前 RGTransformer 的核心原因，直接影响模型对复杂海温模式的建模能力。

**Independent Test**: 可通过对比增强前后的验证损失（MSE/RMSE）来验证，预期在复杂海域（如黑潮、湾流区域）的预测精度有明显提升。

**Acceptance Scenarios**:

1. **Given** 相同的训练数据和配置, **When** 使用增强后的 RGTransformer 训练模型, **Then** 验证集 MSE 损失应优于增强前的基线模型
2. **Given** 包含多尺度海温结构的测试样本, **When** 模型进行预测, **Then** 大尺度和中尺度特征的预测误差均应降低

---

### User Story 2 - 跳跃连接保留细节信息 (Priority: P1)

研究人员需要模型在预测时保留空间细节信息，避免因特征压缩导致的细节丢失，特别是海岸线边界和小尺度温度异常区域。

**Why this priority**: 跳跃连接是解决特征压缩信息损失的关键机制，与多尺度特征提取同等重要。

**Independent Test**: 可通过可视化预测结果，对比增强前后在边界区域和细节区域的预测效果。

**Acceptance Scenarios**:

1. **Given** 包含复杂海岸线的测试区域, **When** 模型进行预测, **Then** 海岸线边界的温度梯度预测应更清晰
2. **Given** 包含小尺度温度异常的测试样本, **When** 模型进行预测, **Then** 小尺度特征应被保留而非平滑掉

---

### User Story 3 - 保持计算效率 (Priority: P2)

研究人员需要增强后的模型在保持或提升性能的同时，不显著增加训练时间和显存占用，以便在现有硬件条件下完成训练。

**Why this priority**: 实用性约束，过度增加计算成本会影响研究迭代速度。

**Independent Test**: 可通过性能分析工具对比增强前后的 GPU 显存使用和每轮训练时间。

**Acceptance Scenarios**:

1. **Given** 相同的批量大小和硬件配置, **When** 使用增强后的模型训练, **Then** GPU 显存增加不超过 20%
2. **Given** 相同的训练配置, **When** 运行完整训练周期, **Then** 单 epoch 训练时间增加不超过 30%

---

### User Story 4 - 2D 空间注意力保留位置关系 (Priority: P3)

研究人员希望注意力机制能够感知空间位置关系，而不是将空间完全展平，以便更好地建模空间相关性。

**Why this priority**: 这是一个可选的优化方向，可以进一步提升空间建模能力，但实现复杂度较高。

**Independent Test**: 可通过可视化注意力权重分布，验证注意力是否体现空间局部性。

**Acceptance Scenarios**:

1. **Given** 训练好的模型, **When** 可视化注意力权重, **Then** 应能观察到空间局部相关性模式

---

### Edge Cases

- 当输入包含大面积陆地（NaN）区域时，多尺度特征提取是否能正确处理？
- 当不同尺度的特征存在冲突时（如小尺度噪声 vs 大尺度趋势），模型如何平衡？
- 跳跃连接是否会在陆地边界产生伪影？

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 系统必须支持多尺度空间特征提取，至少包含 2-3 个不同分辨率的特征层
- **FR-002**: 系统必须实现编码器到解码器的跳跃连接，保留不同尺度的细节信息
- **FR-003**: 系统必须保持与现有 RGTransformer 接口的向后兼容性（输入输出格式不变）
- **FR-004**: 系统必须正确处理 NaN 值（陆地区域），在多尺度特征和跳跃连接中不传播 NaN
- **FR-005**: 系统必须提供配置选项来控制多尺度层数和跳跃连接行为
- **FR-006**: 解码器必须支持多尺度逐层上采样，利用跳跃连接特征进行特征融合

### Key Entities

- **MultiScaleConvStem**: 多尺度卷积前处理模块，输出多个分辨率的特征图
- **SkipConnection**: 跳跃连接机制，存储并传递中间特征到解码器
- **MultiScaleDecoder**: 多尺度解码器，结合跳跃连接特征进行逐层上采样重建

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 增强后的模型在验证集上的 MSE 损失应比基线 RGTransformer 降低至少 5%
- **SC-002**: 增强后的模型 GPU 显存占用增加不超过 20%（相同批量大小）
- **SC-003**: 增强后的模型单 epoch 训练时间增加不超过 30%
- **SC-004**: 增强后的模型参数量增加不超过 50%（保持轻量化）
- **SC-005**: 在边界区域（海岸线附近 50km）的预测 RMSE 应比基线降低至少 10%

## Assumptions

- 现有 ConvStem 模块可以被扩展或替换为多尺度版本
- 现有 EfficientRGAttention 模块可以保持不变或进行轻微调整
- 训练环境具有足够的 GPU 显存（建议 8GB+）来支持增强后的模型
- 用户愿意接受适度的参数量增加以换取性能提升
