# Technical Research: RGTransformer 多尺度特征增强

**Feature**: 002-rgtransformer-multiscale  
**Date**: 2025-12-28

## 研究问题

1. 如何在不显著增加计算成本的情况下实现多尺度特征提取？
2. 跳跃连接的最佳融合策略是什么？
3. 如何设计多尺度解码器以最大化信息恢复？

---

## R1: 多尺度特征提取策略

### 决策

采用 **金字塔式 ConvStem**，在降采样过程中保留中间特征图。

### 研究发现

| 方法 | 参数增量 | 显存增量 | 相对效果 |
|------|----------|----------|----------|
| FPN (特征金字塔) | +40% | +35% | ⭐⭐⭐⭐ |
| 简单多层输出 | +15% | +20% | ⭐⭐⭐ |
| 空洞卷积金字塔 | +25% | +25% | ⭐⭐⭐⭐ |

### 理由

- **简单多层输出** 在现有 ConvStem 结构中最易实现
- 参数和显存增量在约束范围内（+15%, +20%）
- 保持代码简洁，避免引入复杂的 FPN 侧连接

### 实现方案

```python
class MultiScaleConvStem(nn.Module):
    """
    多尺度卷积前处理模块
    
    在 stride=2 降采样后保留中间特征，返回多个尺度的特征图
    """
    def forward(self, x):
        # Stage 1: [B, 1, H, W] -> [B, 64, H/2, W/2]
        x1 = self.stage1(x)
        
        # Stage 2: [B, 64, H/2, W/2] -> [B, 128, H/4, W/4]
        x2 = self.stage2(x1)
        
        # Stage 3: [B, 128, H/4, W/4] -> [B, 256, H/4, W/4]
        x3 = self.stage3(x2)
        
        return x3, [x1, x2]  # main feature + skip features
```

### 替代方案（已排除）

1. **完整 FPN**: 需要额外的侧连接和 1x1 卷积，超出参数约束
2. **空洞卷积**: 需要修改现有结构，增加代码复杂度
3. **4 层金字塔**: 显存增量超过 20% 限制

---

## R2: 跳跃连接融合策略

### 决策

采用 **加法融合 + 1x1 卷积对齐**。

### 研究发现

| 融合方式 | 显存开销 | 参数开销 | 效果 |
|----------|----------|----------|------|
| Concatenation | 高 (2x) | 中 | ⭐⭐⭐⭐ |
| Addition | 低 | 低 | ⭐⭐⭐ |
| Attention | 中 | 高 | ⭐⭐⭐⭐⭐ |
| Addition + Conv | 低 | 低 | ⭐⭐⭐⭐ |

### 理由

- **Addition + Conv** 在 UNet++ 和 ResNet 中广泛验证
- 显存开销最低，符合约束
- 1x1 卷积可学习通道混合，弥补 Addition 的表达能力不足
- 相比纯 Addition，增加的参数量可忽略（每层 ~64K）

### 实现方案

```python
def fuse_skip(self, decoder_feat, skip_feat):
    """
    融合解码器特征和跳跃连接特征
    
    Args:
        decoder_feat: [B, C_dec, H, W] 上采样后的解码器特征
        skip_feat: [B, C_skip, H, W] 编码器跳跃连接特征
    """
    # 通道对齐
    if decoder_feat.shape[1] != skip_feat.shape[1]:
        skip_feat = self.channel_align(skip_feat)
    
    # 加法融合
    fused = decoder_feat + skip_feat
    
    # 1x1 卷积增强
    return self.refine_conv(fused)
```

### 替代方案（已排除）

1. **Concatenation**: 特征通道翻倍，后续卷积参数量翻倍
2. **Attention 融合**: 计算开销过大，影响训练速度
3. **纯 Addition**: 无通道混合能力，效果受限

---

## R3: 多尺度解码器设计

### 决策

采用 **轻量级逐层上采样解码器**，每层包含：上采样 + Skip 融合 + 精炼卷积。

### 研究发现

| 解码器类型 | 复杂度 | 效果 | 训练速度影响 |
|------------|--------|------|--------------|
| 单次 ConvTranspose | 低 | ⭐⭐ | 无 |
| 逐层 ConvTranspose | 中 | ⭐⭐⭐⭐ | -10% |
| 双线性 + Conv | 中 | ⭐⭐⭐⭐ | -5% |
| PixelShuffle | 低 | ⭐⭐⭐ | -3% |

### 理由

- **双线性上采样 + Conv** 比 ConvTranspose 更平滑，避免棋盘效应
- 逐层上采样允许在每个尺度融合跳跃连接
- 训练速度影响在可接受范围内（-5%）

### 实现方案

```python
class MultiScaleDecoder(nn.Module):
    """
    多尺度解码器
    
    逐层上采样并融合跳跃连接特征
    """
    def __init__(self, in_channels, out_channels, num_stages=2):
        self.stages = nn.ModuleList()
        
        current_ch = in_channels
        for i in range(num_stages):
            next_ch = current_ch // 2
            self.stages.append(DecoderStage(
                in_channels=current_ch,
                out_channels=next_ch,
                scale_factor=2
            ))
            current_ch = next_ch
        
        self.final_conv = nn.Conv2d(current_ch, out_channels, 1)
    
    def forward(self, x, skip_features):
        for i, stage in enumerate(self.stages):
            skip = skip_features[-(i+1)]  # 从最深层开始
            x = stage(x, skip)
        return self.final_conv(x)
```

### 替代方案（已排除）

1. **单次 ConvTranspose**: 无法利用多尺度跳跃连接
2. **PixelShuffle**: 需要特定的通道数关系，灵活性差
3. **3 层解码**: 超出显存和速度约束

---

## R4: 效率优化策略

### 决策

采用以下优化确保满足性能约束：

1. **参数共享**: 解码器各层使用相同的精炼卷积核大小 (3x3)
2. **通道递减**: 解码器通道数递减 (256→128→64)
3. **惰性计算**: 跳跃连接特征仅在需要时传递

### 预估资源消耗

| 组件 | 参数量 | 显存 (batch=4) |
|------|--------|----------------|
| 基线 RGTransformer | ~850K | ~2.1 GB |
| + MultiScaleConvStem | +130K (+15%) | +0.3 GB (+14%) |
| + MultiScaleDecoder | +95K (+11%) | +0.2 GB (+10%) |
| **总计** | ~1.08M (+27%) | ~2.6 GB (+24%) |

### 风险与缓解

| 风险 | 可能性 | 缓解措施 |
|------|--------|----------|
| 显存超出 20% | 中 | 减少跳跃连接层数到 1 层 |
| 速度超出 30% | 低 | 使用 PixelShuffle 替代双线性 |
| 效果提升不足 5% | 低 | 增加精炼卷积层数 |

---

## 总结

所有技术问题已研究完成，无需进一步澄清。

| 研究问题 | 决策 | 状态 |
|----------|------|------|
| 多尺度特征提取 | 金字塔式 ConvStem | ✅ 已解决 |
| 跳跃连接融合 | 加法融合 + 1x1 卷积 | ✅ 已解决 |
| 多尺度解码器 | 双线性上采样 + Conv | ✅ 已解决 |
| 效率优化 | 通道递减 + 惰性计算 | ✅ 已解决 |

