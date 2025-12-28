# Data Model: RGTransformer 多尺度特征增强

**Feature**: 002-rgtransformer-multiscale  
**Date**: 2025-12-28

## 模块设计

本功能涉及 3 个核心模块的设计，以下为详细的模块规格。

---

## 1. MultiScaleConvStem

### 职责

多尺度卷积前处理模块，在降采样过程中输出多个尺度的特征图，供跳跃连接使用。

### 接口

```python
class MultiScaleConvStem(nn.Module):
    """
    多尺度卷积前处理模块
    
    Attributes:
        in_channels: 输入通道数 (默认 1，SST 数据)
        embed_dim: 最终嵌入维度 (默认 256)
        num_scales: 输出尺度数 (默认 2，不含最终输出)
        use_bn: 是否使用 BatchNorm
    
    Input:
        x: [batch, channels, height, width]
    
    Output:
        main_feature: [batch, embed_dim, H/4, W/4]
        skip_features: List[[batch, C_i, H_i, W_i]] 各尺度的跳跃特征
    """
```

### 内部结构

| Stage | 输入尺寸 | 输出尺寸 | 操作 |
|-------|----------|----------|------|
| Stage 1 | [B, 1, H, W] | [B, 64, H/2, W/2] | Conv3x3(s=2) + BN + GELU |
| Stage 2 | [B, 64, H/2, W/2] | [B, 128, H/4, W/4] | Conv3x3(s=2) + BN + GELU |
| Stage 3 | [B, 128, H/4, W/4] | [B, 256, H/4, W/4] | Conv1x1 投影 |

### 跳跃特征输出

| 跳跃连接 | 来源 | 尺寸 | 用途 |
|----------|------|------|------|
| skip_1 | Stage 1 后 | [B, 64, H/2, W/2] | 解码器第 2 层融合 |
| skip_2 | Stage 2 后 | [B, 128, H/4, W/4] | 解码器第 1 层融合 |

### 参数估算

| 组件 | 参数量 |
|------|--------|
| Stage 1 Conv | 1 × 64 × 3 × 3 = 576 |
| Stage 1 BN | 64 × 2 = 128 |
| Stage 2 Conv | 64 × 128 × 3 × 3 = 73,728 |
| Stage 2 BN | 128 × 2 = 256 |
| Stage 3 Conv | 128 × 256 × 1 × 1 = 32,768 |
| **总计** | **~107K** |

---

## 2. DecoderStage

### 职责

解码器单层模块，负责上采样和跳跃连接融合。

### 接口

```python
class DecoderStage(nn.Module):
    """
    解码器单层
    
    Attributes:
        in_channels: 输入通道数
        out_channels: 输出通道数
        skip_channels: 跳跃连接通道数
        scale_factor: 上采样倍数 (默认 2)
    
    Input:
        x: [batch, in_channels, H, W] 解码器特征
        skip: [batch, skip_channels, H*scale, W*scale] 跳跃连接特征
    
    Output:
        out: [batch, out_channels, H*scale, W*scale]
    """
```

### 内部结构

```
x [B, C_in, H, W]
    ↓
Upsample (bilinear, 2x)
    ↓ [B, C_in, H*2, W*2]
    +
skip [B, C_skip, H*2, W*2] → ChannelAlign (1x1 Conv if needed)
    ↓
Fused [B, C_out, H*2, W*2]
    ↓
RefineConv (3x3 Conv + BN + GELU)
    ↓
out [B, C_out, H*2, W*2]
```

### 参数估算 (per stage)

| 组件 | 参数量 |
|------|--------|
| ChannelAlign Conv | C_skip × C_out × 1 × 1 |
| RefineConv | C_out × C_out × 3 × 3 |
| BN | C_out × 2 |
| **Stage 1 (128→64)** | **~78K** |
| **Stage 2 (64→32)** | **~20K** |

---

## 3. MultiScaleDecoder

### 职责

多尺度解码器，组合多个 DecoderStage，逐层上采样并融合跳跃连接。

### 接口

```python
class MultiScaleDecoder(nn.Module):
    """
    多尺度解码器
    
    Attributes:
        in_channels: 输入通道数 (来自 Attention 输出)
        out_channels: 输出通道数 (1 for SST)
        num_stages: 解码器层数 (对应跳跃连接数)
    
    Input:
        x: [batch, in_channels, H, W] Attention 输出
        skip_features: List[[batch, C_i, H_i, W_i]] 跳跃连接特征列表
    
    Output:
        out: [batch, out_channels, H_final, W_final]
    """
```

### 层级配置

| Stage | 输入 | 跳跃连接 | 输出 |
|-------|------|----------|------|
| Stage 1 | [B, 256, H/4, W/4] | skip_2 [B, 128, H/4, W/4] | [B, 128, H/2, W/2] |
| Stage 2 | [B, 128, H/2, W/2] | skip_1 [B, 64, H/2, W/2] | [B, 64, H, W] |
| Final | [B, 64, H, W] | - | [B, 1, H, W] |

### 总参数估算

| 组件 | 参数量 |
|------|--------|
| DecoderStage 1 | ~78K |
| DecoderStage 2 | ~20K |
| Final Conv | 64 × 1 × 1 × 1 = 64 |
| **总计** | **~98K** |

---

## 4. 增强版 RGTransformer 接口

### 新增参数

```python
class RGTransformer(LightningModule):
    def __init__(
        self,
        # ... 现有参数 ...
        
        # 新增参数
        use_multiscale: bool = True,       # 是否启用多尺度
        num_skip_connections: int = 2,     # 跳跃连接数量
        skip_fusion: str = "add",          # 融合方式: "add" | "concat"
    ):
```

### 向后兼容性

| 配置 | 行为 |
|------|------|
| `use_multiscale=False` | 使用原始 ConvStem + 单次 ConvTranspose |
| `use_multiscale=True` | 使用 MultiScaleConvStem + MultiScaleDecoder |

---

## 5. 数据流示意

```
Input: [B, T, H, W]
    ↓
NaN 处理 + 位置编码
    ↓
MultiScaleConvStem
    ├─ skip_1: [B*T, 64, H/2, W/2]
    ├─ skip_2: [B*T, 128, H/4, W/4]
    └─ main: [B*T, 256, H/4, W/4]
    ↓
Reshape: [B*H'*W', T, D]
    ↓
EfficientRGAttention (unchanged)
    ↓
Reshape: [B, D, H', W']
    ↓
时序聚合: [B, D, H', W']
    ↓
MultiScaleDecoder
    ├─ Stage 1 + skip_2 → [B, 128, H/2, W/2]
    ├─ Stage 2 + skip_1 → [B, 64, H, W]
    └─ Final Conv → [B, 1, H, W]
    ↓
Output: [B, H, W]
```

---

## 6. 验证规则

### 输入验证

- `height` 和 `width` 必须能被 4 整除
- `num_skip_connections` 必须 ≤ 2（当前设计限制）
- `skip_fusion` 必须是 `"add"` 或 `"concat"`

### 运行时断言

```python
# 尺寸一致性检查
assert skip.shape[2:] == x.shape[2:], \
    f"Skip shape {skip.shape[2:]} != decoder shape {x.shape[2:]}"

# NaN 传播检查
assert not torch.isnan(fused).any() or torch.isnan(skip).any(), \
    "NaN should only exist where input had NaN"
```

---

## 总参数量对比

| 模块 | 基线 | 增强后 | 增量 |
|------|------|--------|------|
| ConvStem | 67K | 107K | +40K |
| Attention | 263K | 263K | 0 |
| Decoder | 66K | 164K | +98K |
| 其他 | 450K | 450K | 0 |
| **总计** | **~846K** | **~984K** | **+138K (+16%)** |

✅ 参数增量 16% < 50% 约束

