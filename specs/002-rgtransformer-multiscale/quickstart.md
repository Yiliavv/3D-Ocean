# Quick Start: RGTransformer 多尺度特征增强

**Feature**: 002-rgtransformer-multiscale  
**Date**: 2025-12-28

## 快速开始

### 1. 使用增强版 RGTransformer

```python
from src.models.SST.RGTransformer import RGTransformer

# 创建增强版模型（默认启用多尺度）
model = RGTransformer(
    width=128,
    height=128,
    seq_len=12,
    d_model=256,
    num_heads=8,
    
    # 新增参数
    use_multiscale=True,       # 启用多尺度
    num_skip_connections=2,     # 2 层跳跃连接
)

# 训练和推理与之前完全相同
x = torch.randn(4, 11, 128, 128)  # [batch, seq_len-1, H, W]
y_pred = model(x)  # [batch, H, W]
```

### 2. 向后兼容模式

```python
# 禁用多尺度，使用原始架构
model = RGTransformer(
    width=128,
    height=128,
    seq_len=12,
    use_multiscale=False,  # 回退到 V2 行为
)
```

### 3. 训练配置

```python
from lightning import Trainer
from src.trainer.base import BaseTrainer

# 配置训练器
trainer = Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
)

# 训练
trainer.fit(model, train_dataloader, val_dataloader)
```

---

## 性能对比

### 基准测试命令

```bash
# 运行性能对比测试
python -m pytest tests/unit/test_multiscale.py -v --benchmark
```

### 预期结果

| 指标 | 基线 V2 | 增强版 | 变化 |
|------|---------|--------|------|
| 参数量 | 846K | 984K | +16% |
| 显存 (batch=4) | 2.1 GB | ~2.5 GB | +19% |
| 训练速度 | 1.0x | ~0.85x | -15% |
| 验证 MSE | 0.45 | ~0.42 | -7% |

---

## 关键代码位置

| 模块 | 文件路径 | 描述 |
|------|----------|------|
| MultiScaleConvStem | `src/models/SST/ConvStem.py` | 多尺度编码器 |
| MultiScaleDecoder | `src/models/SST/MultiScaleDecoder.py` | 多尺度解码器 |
| RGTransformer | `src/models/SST/RGTransformer.py` | 集成入口 |

---

## 常见问题

### Q1: 如何调整跳跃连接数量？

```python
# 使用 1 层跳跃连接（更轻量）
model = RGTransformer(
    ...,
    num_skip_connections=1,
)
```

### Q2: 显存不足怎么办？

```python
# 选项 1: 减少跳跃连接
model = RGTransformer(..., num_skip_connections=1)

# 选项 2: 减小批量大小
dataloader = DataLoader(..., batch_size=2)

# 选项 3: 禁用多尺度
model = RGTransformer(..., use_multiscale=False)
```

### Q3: 如何验证多尺度是否生效？

```python
# 检查模型结构
print(model)

# 预期输出应包含:
# - MultiScaleConvStem
# - MultiScaleDecoder
# - DecoderStage (2 个)
```

---

## 测试验证

### 单元测试

```bash
# 运行多尺度组件测试
python -m pytest tests/unit/test_multiscale.py -v

# 运行完整模型测试
python -m pytest tests/unit/test_rgtransformer_v2.py -v
```

### 集成测试

```bash
# 运行短期训练测试
python -m pytest tests/integration/test_training.py -v -k "multiscale"
```

---

## 下一步

1. **验证性能**: 使用完整数据集训练，对比基线
2. **可视化分析**: 检查跳跃连接特征和预测结果
3. **超参数调优**: 调整 `num_skip_connections` 和 `d_model`

