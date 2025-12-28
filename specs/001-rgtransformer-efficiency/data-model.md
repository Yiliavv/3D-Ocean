# Data Model: RGTransformer V2 架构设计

**Branch**: `001-rgtransformer-efficiency` | **Date**: 2025-12-28

## 概述

本文档定义优化后的 RGTransformer V2 模型架构，基于 research.md 中确定的优化策略。

---

## 模块架构

### 整体结构

```
RGTransformerV2
├── ConvStem                 # 替代原 Patch Embedding
├── SpatialSphericalHarmonicEncoding  # 保持，优化缓存
├── EfficientRGAttention     # 替代原 RGAttention
├── ChannelFeedForward       # 保持不变
├── TemporalAggregation      # 保持不变
└── PatchRecovery            # 替代原 ConvTranspose2d
```

### 数据流

```
Input: [B, S, W, H]
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Normalize SST + Position Encoding                        │
│    [B, S, W, H] + [W, H] → [B, S, W, H]                     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. ConvStem (NEW)                                           │
│    [B*S, 1, W, H] → [B*S, D, W', H']                        │
│    多层卷积逐步降采样，而非单次 patch embedding              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Reshape for Attention                                    │
│    [B*S, D, W', H'] → [B*W'*H', S, D]                       │
│    使用 einops.rearrange 简化                               │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. EfficientRGAttention (NEW)                               │
│    - 无 Global Token                                        │
│    - 轻量 Scalar Gate (256 params vs 131K)                  │
│    - 单次注意力计算（无递归）                                │
│    [B*W'*H', S, D] → [B*W'*H', S, D]                        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. ChannelFeedForward                                       │
│    [B*W'*H', S, D] → [B*W'*H', S, D]                        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Temporal Aggregation                                     │
│    [B*W'*H', S, D] → [B*W'*H', D]                           │
│    加权求和时序维度                                          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. PatchRecovery                                            │
│    [B, D, W', H'] → [B, 1, W, H]                            │
│    上采样恢复原始分辨率                                      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Output: [B, W, H]
```

---

## 模块详细设计

### 1. ConvStem

**目的**: 替代原有的单层 Patch Embedding，提供更好的局部特征提取和平移等变性。

```python
class ConvStem(nn.Module):
    """
    卷积前处理模块，替代直接的 Patch Embedding
    
    优势:
    - 多层小卷积捕获局部特征
    - 自然引入平移等变性
    - 避免 patch 边界硬切割
    """
    
    def __init__(self, in_channels: int = 1, 
                 embed_dim: int = 256, 
                 target_reduction: int = 4):
        """
        Args:
            in_channels: 输入通道数 (SST 为 1)
            embed_dim: 输出嵌入维度
            target_reduction: 总降采样倍数 (对应原 patch_size)
        """
```

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `in_channels` | int | 1 | 输入通道数 |
| `embed_dim` | int | 256 | 输出嵌入维度 |
| `target_reduction` | int | 4 | 降采样倍数 |

**内部结构**:

| 层 | 输入 → 输出 | 操作 |
|---|-------------|------|
| conv1 | [B, 1, W, H] → [B, 64, W/2, H/2] | Conv2d(3×3, stride=2) + BN + GELU |
| conv2 | [B, 64, W/2, H/2] → [B, 128, W/4, H/4] | Conv2d(3×3, stride=2) + BN + GELU |
| proj | [B, 128, W/4, H/4] → [B, 256, W/4, H/4] | Conv2d(1×1) |

**参数量**: ~33K（比原 Patch Embedding 的 ~1K 多，但提供更好的特征）

---

### 2. EfficientRGAttention

**目的**: 简化原 RGAttention，移除低效组件。

```python
class EfficientRGAttention(nn.Module):
    """
    高效版递归泛化自注意力
    
    改进:
    - 移除 Global Token（输出未使用，计算浪费）
    - 轻量 Scalar Gate（256 params vs 131K）
    - 可选递归深度（默认 1，即无递归）
    """
    
    def __init__(self, d_model: int, 
                 num_heads: int, 
                 dropout: float = 0.1,
                 num_layers: int = 1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: Dropout 比例
            num_layers: 注意力层数（替代递归深度）
        """
```

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `d_model` | int | 256 | 模型维度 |
| `num_heads` | int | 8 | 注意力头数 |
| `dropout` | float | 0.1 | Dropout 比例 |
| `num_layers` | int | 1 | 注意力层数 |

**组件对比**:

| 组件 | 原版 | V2 | 变化 |
|------|------|-----|------|
| Global Token | ✅ 有 | ❌ 移除 | -256 params |
| gate_projection | Linear(512→256) | Linear(256→1) | -130,816 params |
| recursion_depth | 2 (硬编码) | num_layers (可配置) | 更灵活 |
| step_weights | ✅ 有 | ❌ 移除 | -3 params |

**参数量对比**:
- 原版: ~394K
- V2: ~263K
- **节省**: ~131K (33%)

---

### 3. RGTransformerV2 主模型

**目的**: 整合所有优化组件。

```python
class RGTransformerV2(LightningModule):
    """
    优化版 RG-Transformer 海表温度预测模型
    
    主要优化:
    1. ConvStem 替代 Patch Embedding
    2. EfficientRGAttention 替代 RGAttention
    3. einops 简化张量操作
    4. torch.compile 兼容设计
    """
    
    def __init__(self, 
                 width: int, 
                 height: int, 
                 seq_len: int,
                 d_model: int = 256,
                 num_heads: int = 8,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 num_attn_layers: int = 1,
                 learning_rate: float = 1e-4,
                 lat_range: Optional[List[float]] = None,
                 lon_range: Optional[List[float]] = None,
                 resolution: float = 1.0,
                 patch_size: int = 4,
                 use_compile: bool = True,
                 **kwargs):
```

**新增参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_attn_layers` | int | 1 | 注意力层数（替代 recursion_depth） |
| `use_compile` | bool | True | 是否启用 torch.compile |

**兼容性**:

| 特性 | 状态 | 说明 |
|------|------|------|
| 输入接口 | ✅ 兼容 | `[B, S-1, W, H]` |
| 输出接口 | ✅ 兼容 | `[B, W, H]` |
| LightningModule | ✅ 兼容 | 继承不变 |
| 训练流程 | ✅ 兼容 | training_step/validation_step 不变 |
| 权重加载 | ⚠️ 部分兼容 | 需要权重映射函数 |

---

## 状态转换

### 模型状态

```
┌─────────────────┐
│   Initialized   │
│  (torch.compile │
│   if enabled)   │
└────────┬────────┘
         │ .train()
         ▼
┌─────────────────┐     .eval()     ┌─────────────────┐
│    Training     │ ◄─────────────► │    Inference    │
│   (grad enabled)│                 │  (grad disabled)│
└─────────────────┘                 └─────────────────┘
```

### 权重迁移映射

从 V1 加载权重到 V2:

| V1 参数路径 | V2 参数路径 | 映射策略 |
|-------------|-------------|----------|
| `patch_embed.*` | `conv_stem.*` | 需要重新训练 |
| `attention.shared_attention.*` | `attention.layers.0.attention.*` | 直接映射 |
| `attention.gate_projection.*` | `attention.layers.0.gate.*` | 截取/重新训练 |
| `attention.global_token` | N/A | 丢弃 |
| `ffn.*` | `ffn.*` | 直接映射 |
| `patch_recovery.*` | `patch_recovery.*` | 直接映射 |
| `temporal_weights` | `temporal_weights` | 直接映射 |

---

## 性能预估

### 参数量对比

| 模块 | V1 | V2 | 变化 |
|------|-----|-----|------|
| Patch Embedding / ConvStem | ~1K | ~33K | +32K |
| RGAttention / EfficientRGAttention | ~394K | ~263K | -131K |
| FFN | ~525K | ~525K | 0 |
| 其他 | ~5K | ~5K | 0 |
| **总计** | ~925K | ~826K | **-99K (-11%)** |

### FLOPs 预估（单次前向）

| 操作 | V1 | V2 | 变化 |
|------|-----|-----|------|
| Embedding | 低 | 中 | +20% |
| Attention | 高 | 中 | -50%（无递归） |
| 其他 | 相同 | 相同 | 0 |
| **总计** | 基准 | -20~30% | ⬇️ |

---

## 验证规则

### 输入验证

```python
def validate_input(x: torch.Tensor, config: ModelConfig) -> None:
    assert x.dim() == 4, f"Expected 4D input, got {x.dim()}D"
    assert x.shape[1] == config.seq_len - 1, f"Expected seq_len-1={config.seq_len-1}, got {x.shape[1]}"
    assert x.shape[2] == config.width, f"Expected width={config.width}, got {x.shape[2]}"
    assert x.shape[3] == config.height, f"Expected height={config.height}, got {x.shape[3]}"
```

### 输出验证

```python
def validate_output(y: torch.Tensor, x: torch.Tensor) -> None:
    assert y.dim() == 3, f"Expected 3D output, got {y.dim()}D"
    assert y.shape[0] == x.shape[0], "Batch size mismatch"
    assert y.shape[1] == x.shape[2], "Width mismatch"
    assert y.shape[2] == x.shape[3], "Height mismatch"
    assert not torch.isnan(y[~torch.isnan(y)]).any(), "Unexpected NaN in valid regions"
```

