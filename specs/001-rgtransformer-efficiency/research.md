# Research: RGTransformer 优化策略研究

**Branch**: `001-rgtransformer-efficiency` | **Date**: 2025-12-28

## 研究目标

针对 spec.md 中识别的 8 个效率问题，研究最佳优化方案。

---

## 1. 固定位置 Token 化优化

### 问题回顾
当前使用 `Conv2d(stride=patch_size)` 进行无重叠划分，导致边界效应和缺乏平移等变性。

### 方案对比

| 方案 | 优点 | 缺点 | 复杂度 |
|------|------|------|--------|
| **A. Overlapping Patches** | 简单，保留边界信息 | token 数量增加，计算量增大 | 低 |
| **B. Convolutional Stem** | 平滑特征提取，业界验证 | 额外参数，多层卷积 | 中 |
| **C. Swin-style Shifted Window** | 打破边界限制 | 实现复杂，需要窗口注意力 | 高 |
| **D. Hybrid: Conv Stem + Pooling** | 平衡性能和效率 | 需要调参 | 中 |

### 决策: **B. Convolutional Stem**

**Rationale**:
1. 在 ViT 后续研究（如 LeViT, ConvNeXt）中被广泛验证有效
2. 多层小卷积比单次大卷积更能捕获局部特征
3. 不增加 token 数量，避免计算量膨胀
4. 自然引入平移等变性

**实现方案**:
```python
class ConvStem(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, patch_size=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels//4),
            nn.GELU(),
            nn.Conv2d(out_channels//4, out_channels//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels//2),
            nn.GELU(),
            nn.Conv2d(out_channels//2, out_channels, 1),  # 1x1 调整通道
        )
```

**Alternatives Considered**:
- Overlapping Patches: 增加 30-50% token 数量，与效率优化目标冲突
- Swin Transformer: 实现复杂度过高，需要重构整个注意力机制

---

## 2. RGAttention 参数效率优化

### 问题回顾
门控机制占 33% 参数（~131K），递归深度导致重复计算，Global Token 输出被丢弃。

### 方案对比

| 方案 | 参数节省 | 性能影响 | 风险 |
|------|----------|----------|------|
| **A. 移除门控，改用简单残差** | ~131K | 可能略降 | 低 |
| **B. 轻量门控（scalar gate）** | ~130K | 保持 | 极低 |
| **C. 移除递归，改用多层独立块** | 0 | 可能提升 | 中 |
| **D. 组合 A+C** | ~131K | 取决于层数 | 中 |

### 决策: **B. 轻量门控 + C. 多层独立块（可选）**

**Rationale**:
1. Scalar gate 保留门控机制的自适应能力，但参数量从 131K 降到 512
2. 用 `nn.Linear(d_model, 1)` 替代 `nn.Linear(d_model*2, d_model)`
3. 多层独立块比递归更易于优化（PyTorch 可以更好地并行化）

**实现方案**:
```python
class EfficientRGAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        # 轻量门控: 从 d_model*2 -> d_model 简化为 d_model -> 1
        self.gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        gate = self.gate(attn_out)  # [B, S, 1]
        return gate * attn_out + (1 - gate) * x
```

**参数对比**:
- 原 gate_projection: `d_model * 2 * d_model` = 131,072
- 新 gate: `d_model * 1` = 256
- **节省**: 130,816 参数 (99.8%)

**Alternatives Considered**:
- 完全移除门控: 可能影响模型表达能力，保守起见保留轻量版本

---

## 3. Global Token 优化

### 问题回顾
Global Token 输出被丢弃 `output[:, 1:, :]`，仅作为信息枢纽。

### 方案对比

| 方案 | 优点 | 缺点 |
|------|------|------|
| **A. 移除 Global Token** | 简化，减少计算 | 可能损失全局信息聚合 |
| **B. 利用 Global Token 输出** | 提取有价值信息 | 需要额外处理逻辑 |
| **C. 保持现状** | 无风险 | 浪费计算 |

### 决策: **A. 移除 Global Token**

**Rationale**:
1. 当前实现中 Global Token 的输出未被使用，计算浪费
2. 全局信息可以通过时序聚合阶段的 attention 机制获取
3. 移除后序列长度减少 1，attention 计算量略微减少

**实现**: 在 `EfficientRGAttention` 中设置 `use_global_token=False`

**Alternatives Considered**:
- 利用 Global Token 输出作为额外特征: 增加模型复杂度，与效率优化目标冲突

---

## 4. 空间维度展平优化

### 问题回顾
当前使用 `[B*W'*H', S, D]` 格式，导致内存碎片化。

### 方案对比

| 方案 | 内存效率 | 实现难度 |
|------|----------|----------|
| **A. 保持展平，使用 contiguous** | 中 | 低 |
| **B. 批量空间注意力** | 高 | 中 |
| **C. 使用 torch.compile** | 高 | 低 |

### 决策: **C. 使用 torch.compile + A. 优化 contiguous 调用**

**Rationale**:
1. PyTorch 2.0 的 `torch.compile` 可以自动优化内存访问模式
2. 减少不必要的 `.contiguous()` 调用
3. 使用 `memory_format=torch.channels_last` 优化 CNN 部分

**实现**:
```python
# 在模型初始化后
model = torch.compile(model, mode="reduce-overhead")
```

---

## 5. 内存访问模式优化

### 决策: 合并张量操作

**实现**:
1. 减少 `view` -> `permute` -> `contiguous` 链式调用
2. 使用 `einops.rearrange` 简化维度变换
3. 预分配输出张量避免动态分配

```python
# Before
x = x.view(B, S, D, W, H)
x = x.permute(0, 3, 4, 1, 2).contiguous()
x = x.view(-1, S, D)

# After (using einops)
from einops import rearrange
x = rearrange(x, 'b s d w h -> (b w h) s d')
```

---

## 6. 时序聚合优化

### 问题回顾
当前使用简单的加权求和，可能丢失细粒度时序依赖。

### 决策: 保持现状（低优先级）

**Rationale**:
1. 时序聚合的改进属于模型能力增强，而非效率优化
2. 当前方案已经足够简洁高效
3. 如需改进，可以在后续迭代中考虑使用 Temporal Attention

---

## 7. 位置编码优化

### 问题回顾
原始分辨率球谐波计算在 Patch 级别后冗余。

### 决策: 延迟计算 + 缓存

**实现**:
1. 将位置编码从原始分辨率改为 Patch 级别计算
2. 使用 `@torch.jit.script` 加速球谐波计算
3. 缓存预计算结果到 buffer

```python
# 在 Patch 级别计算位置编码
self.register_buffer('patch_pos_encoding', 
    self._compute_patch_level_encoding())
```

---

## 研究总结

### 优化策略优先级

| 优先级 | 优化项 | 预期收益 | 风险 |
|--------|--------|----------|------|
| P0 | torch.compile 启用 | 10-30% 加速 | 极低 |
| P1 | RGAttention 轻量门控 | 33% 参数减少 | 低 |
| P1 | 移除 Global Token | 简化计算 | 低 |
| P1 | Convolutional Stem | 更好的特征提取 | 低 |
| P2 | einops 张量操作 | 代码简化 | 极低 |
| P2 | 位置编码缓存 | 小幅加速 | 极低 |
| P3 | 时序聚合改进 | 可选 | 中 |

### 预期效果

基于上述优化组合：
- **训练速度**: 预计提升 25-35%（torch.compile + 参数减少 + 简化计算）
- **显存占用**: 预计减少 15-25%（参数减少 + 内存优化）
- **推理速度**: 预计提升 20-30%（同上）
- **精度影响**: 预计持平或略有提升（Conv Stem 可能改善特征提取）

### 参考文献

1. **LeViT** (ICCV 2021): Conv Stem 的有效性
2. **ConvNeXt** (CVPR 2022): 现代卷积网络设计
3. **PyTorch 2.0 Compile**: 官方性能优化指南
4. **einops**: 张量操作最佳实践

