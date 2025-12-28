# Quickstart: RGTransformer V2

**Branch**: `001-rgtransformer-efficiency` | **Date**: 2025-12-28

## 概述

本指南帮助你快速开始使用优化后的 RGTransformerV2 模型。

---

## 安装依赖

```bash
# 确保 PyTorch 2.0+ 已安装
pip install torch>=2.0.0 torchvision torchaudio

# 安装额外依赖
pip install einops lightning
```

---

## 快速使用

### 1. 创建模型

```python
from src.models.SST.RGTransformerV2 import RGTransformerV2

# 创建优化后的模型
model = RGTransformerV2(
    width=360,              # 经度方向网格数
    height=180,             # 纬度方向网格数
    seq_len=8,              # 输入序列长度 + 1
    d_model=256,            # 模型维度
    num_heads=8,            # 注意力头数
    dim_feedforward=1024,   # FFN 维度
    dropout=0.1,
    num_attn_layers=1,      # 注意力层数（新参数）
    learning_rate=1e-4,
    lat_range=[-90, 90],    # 纬度范围
    lon_range=[0, 360],     # 经度范围
    resolution=1.0,         # 空间分辨率（度）
    patch_size=4,           # Patch 大小
    use_compile=True        # 启用 torch.compile（新参数）
)
```

### 2. 训练模型

```python
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# 创建训练器
trainer = L.Trainer(
    max_epochs=100,
    accelerator='gpu',
    devices=1,
    callbacks=[
        ModelCheckpoint(
            monitor='val_loss',
            save_top_k=3,
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'
        )
    ]
)

# 开始训练
trainer.fit(model, train_dataloader, val_dataloader)
```

### 3. 推理预测

```python
import torch

# 加载模型
model = RGTransformerV2.load_from_checkpoint('path/to/checkpoint.ckpt')
model.eval()

# 准备输入数据
# x: [batch, seq_len-1, width, height]
x = torch.randn(1, 7, 360, 180)

# 预测
with torch.no_grad():
    y_pred = model(x)  # [batch, width, height]
```

---

## 从 V1 迁移

### 代码更改

```python
# Before (V1)
from src.models.SST.RGTransformer import RGTransformer

model = RGTransformer(
    width=360,
    height=180,
    seq_len=8,
    recursion_depth=2,  # V1 参数
    # ...
)

# After (V2)
from src.models.SST.RGTransformerV2 import RGTransformerV2

model = RGTransformerV2(
    width=360,
    height=180,
    seq_len=8,
    num_attn_layers=1,  # V2 参数（替代 recursion_depth）
    use_compile=True,   # V2 新参数
    # ...
)
```

### 权重迁移

由于架构变化，V1 权重**不能**直接加载到 V2。推荐重新训练：

```python
# 如果需要从 V1 权重初始化部分参数
from src.utils.weight_migration import migrate_v1_to_v2

v1_checkpoint = torch.load('v1_model.ckpt')
v2_model = RGTransformerV2(...)

# 迁移兼容的权重
migrate_v1_to_v2(v2_model, v1_checkpoint['state_dict'])

# 继续训练
trainer.fit(v2_model, train_dataloader, val_dataloader)
```

---

## 性能对比

### 基准测试脚本

```python
import torch
import time

def benchmark_model(model, input_shape, num_iterations=100):
    """对模型进行性能基准测试"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # 预热
    x = torch.randn(*input_shape, device=device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
    
    # 同步 CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 计时
    start = time.perf_counter()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    avg_time = elapsed / num_iterations * 1000  # ms
    
    # 显存占用
    if device.type == 'cuda':
        memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    else:
        memory = 0
    
    return {
        'avg_inference_time_ms': avg_time,
        'peak_memory_mb': memory,
        'throughput_per_sec': 1000 / avg_time
    }

# 运行基准测试
results_v2 = benchmark_model(
    RGTransformerV2(...), 
    input_shape=(1, 7, 360, 180)
)
print(f"V2 推理时间: {results_v2['avg_inference_time_ms']:.2f} ms")
print(f"V2 显存占用: {results_v2['peak_memory_mb']:.2f} MB")
```

### 预期性能提升

| 指标 | V1 基准 | V2 目标 | 预期提升 |
|------|---------|---------|----------|
| 训练时间/epoch | 100% | ≤80% | ≥20% |
| 峰值显存 | 100% | ≤85% | ≥15% |
| 推理延迟 | 100% | ≤85% | ≥15% |
| 验证 MSE | 基准 | ≤105% | 保持 |

---

## 常见问题

### Q: torch.compile 报错怎么办？

```python
# 禁用 compile
model = RGTransformerV2(..., use_compile=False)

# 或者使用不同的 compile 模式
model = RGTransformerV2(..., use_compile=True)
model = torch.compile(model, mode="default")  # 而非 "reduce-overhead"
```

### Q: 如何查看模型参数量？

```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"V2 参数量: {count_parameters(model):,}")
# 预期: ~826K（比 V1 的 ~925K 减少约 11%）
```

### Q: 如何验证 NaN 处理正确？

```python
# 创建包含 NaN 的测试输入（模拟陆地区域）
x_test = torch.randn(1, 7, 360, 180)
x_test[0, :, 50:100, 50:100] = float('nan')  # 模拟陆地

y_pred = model(x_test)

# 验证 NaN 区域被正确处理
assert torch.isnan(y_pred[0, 50:100, 50:100]).all(), "陆地区域应保持 NaN"
assert not torch.isnan(y_pred[0, 0:50, 0:50]).any(), "海洋区域不应有 NaN"
```

---

## 下一步

1. **运行基准测试**: 验证性能提升是否达到目标
2. **训练完整模型**: 在完整数据集上训练 V2
3. **对比精度**: 确保预测精度不下降
4. **可视化验证**: 检查预测结果的物理合理性

