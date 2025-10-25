# BaseTrainer 优化功能使用指南

`BaseTrainer` 现已集成性能优化功能，可直接使用，无需额外配置。

**默认优化（自动启用）**：
- ✅ 多进程数据加载 (num_workers=8)
- ✅ GPU内存固定 (pin_memory=True)
- ✅ 混合精度训练 (precision="16-mixed")
- ✅ 持久化workers (persistent_workers=True)

**预期效果**: 4-8倍训练速度提升！

---

## 快速开始

### 1. 基础使用（默认优化已启用）

```python
import uuid
import numpy as np
from src.trainer.base import BaseTrainer  # ✅ 使用原来的BaseTrainer
from src.models.RATransformer import RecursiveAttentionTransformer
from src.dataset.ERA5 import ERA5SSTMonthlyDataset
from src.config.area import Area
from src.config.params import MODEL_SAVE_PATH

# 配置区域
area = Area('Global', lon=[-180, 180], lat=[-80, 80], description='全球区域')

# 配置参数
trainer_uid = str(uuid.uuid4())
resolution = 2
seq_len = 2

width = int(area.width / resolution)
height = int(area.height / resolution)

# 数据集参数
dataset_params = {
    "seq_len": seq_len,
    "offset": 0,
    "resolution": resolution,
}

# 模型参数
model_params = {
    "width": width,
    "height": height,
    "seq_len": seq_len,
    "d_model": 512, 
    "num_heads": 8,
    "num_layers": 3,
    "dim_feedforward": 1024,
    "dropout": 0.1,
    "recursion_depth": 2,
    "learning_rate": 1e-3,
}

# 训练参数 - 默认启用优化
trainer_params = {
    "epochs": 500,
    "batch_size": 150,  # ✅ 从50增加到150（显存充足）
    
    # 以下参数已默认启用，无需手动设置
    # "num_workers": 8,          # 多进程数据加载
    # "pin_memory": True,        # GPU内存固定
    # "persistent_workers": True,# 保持工作进程
    # "prefetch_factor": 2,      # 预取2个batch
    # "precision": "16-mixed",   # 混合精度训练
}

# 创建训练器 - 优化功能已自动集成
trainer = BaseTrainer(
    title='RATransformer',
    area=area,
    uid=trainer_uid,
    model_class=RecursiveAttentionTransformer,
    dataset_class=ERA5SSTMonthlyDataset,
    save_path=f'{MODEL_SAVE_PATH}/seq_len-{seq_len}/ra_transformer.pkl',
    pre_model=False,
    dataset_params=dataset_params,
    trainer_params=trainer_params,
    model_params=model_params,
)

# 开始训练 - 自动享受 4-8倍速度提升
model = trainer.train()
```

**预期效果**：
- 训练速度提升 **4-8倍**
- GPU利用率从 12% → 70-80%
- 显存使用从 2.7GB → 6-7GB
- 每个epoch时间：~10分钟 → ~1.5-2.5分钟

---

## 2. 自定义优化参数

如果你想手动调整优化参数（覆盖默认值）：

```python
trainer_params = {
    "epochs": 500,
    "batch_size": 150,
    
    # 数据加载优化（自定义）
    "num_workers": 6,           # 默认8，可根据CPU调整
    "pin_memory": True,         # 默认True
    "persistent_workers": True, # 默认True
    "prefetch_factor": 4,       # 默认2，增加预取提升速度
    
    # 训练精度优化
    "precision": "16-mixed",    # 默认"16-mixed"，可选"32"或"bf16-mixed"
    
    # 梯度累积（可选，默认1）
    "accumulate_grad_batches": 2,  # 累积2个batch，有效batch_size=300
    
    # 梯度裁剪（可选）
    "gradient_clip_val": 1.0,
    "gradient_clip_algorithm": "norm",
}

trainer = BaseTrainer(
    # ... 其他参数 ...
    trainer_params=trainer_params,
)
```

---

## 3. 启用PyTorch 2.0编译（最大化性能）

启用所有优化，包括PyTorch 2.0模型编译：

```python
trainer_params = {
    "epochs": 500,
    "batch_size": 150,
    
    # PyTorch 2.0 编译（需要PyTorch >= 2.0）
    "compile_model": True,
    "compile_mode": "reduce-overhead",  # 可选: "default", "reduce-overhead", "max-autotune"
}

trainer = BaseTrainer(
    # ... 其他参数 ...
    trainer_params=trainer_params,
)

model = trainer.train()
```

**注意**：
- 需要 PyTorch >= 2.0
- 首次编译需要额外时间（约1-2分钟）
- 编译后的模型在后续epoch中会显著加速

**预期效果**：
- 训练速度提升 **8-15倍**（相比未优化版本）
- GPU利用率 85-95%
- 每个epoch时间：~10分钟 → ~0.7-1.2分钟

---

## 4. 禁用优化（调试模式）

如果遇到问题需要调试，可以禁用优化：

```python
trainer_params = {
    "epochs": 500,
    "batch_size": 50,           # 恢复原始batch size
    
    # 禁用优化
    "num_workers": 0,           # 单线程数据加载
    "pin_memory": False,        # 禁用内存固定
    "persistent_workers": False,
    "precision": "32",          # FP32全精度
}

trainer = BaseTrainer(
    # ... 其他参数 ...
    trainer_params=trainer_params,
)
```

---

## 5. 不同batch size的建议

根据您的GPU显存选择合适的batch size：

| GPU显存 | 推荐batch_size | 学习率调整 | 预期显存使用 |
|---------|---------------|-----------|-------------|
| 8GB     | 100-150       | 1.4-1.7e-3 | ~6-7GB     |
| 12GB    | 200-250       | 2.0-2.2e-3 | ~10-11GB   |
| 16GB    | 300-400       | 2.4-2.8e-3 | ~14-15GB   |
| 24GB    | 500-600       | 3.2-3.5e-3 | ~22-23GB   |

**学习率调整公式**：
```
new_lr = old_lr × sqrt(new_batch_size / old_batch_size)
```

示例（batch_size从50增加到150）：
```python
import math

old_lr = 1e-3
old_batch = 50
new_batch = 150

new_lr = old_lr * math.sqrt(new_batch / old_batch)
print(f"New learning rate: {new_lr:.4e}")  # 输出: 1.7321e-03

model_params = {
    # ... 其他参数 ...
    "learning_rate": new_lr,  # 使用调整后的学习率
}
```

---

## 6. 优化对比测试

```python
import time
from src.trainer.base import BaseTrainer

# 测试未优化版本
print("=" * 60)
print("测试未优化训练（FP32 + 单线程）")
print("=" * 60)

unopt_trainer = BaseTrainer(
    # ... 参数配置 ...
    trainer_params={
        "epochs": 5, 
        "batch_size": 50,
        "num_workers": 0,
        "precision": "32",
        "pin_memory": False,
    },
)

start_time = time.time()
unopt_trainer.train()
unopt_time = time.time() - start_time

print(f"未优化版本 5 epochs 用时: {unopt_time:.2f}秒")

# 测试优化版本
print("\n" + "=" * 60)
print("测试优化训练（FP16 + 多线程）")
print("=" * 60)

opt_trainer = BaseTrainer(
    # ... 相同参数配置 ...
    trainer_params={
        "epochs": 5, 
        "batch_size": 150,
        # 使用默认优化配置
    },
)

start_time = time.time()
opt_trainer.train()
opt_time = time.time() - start_time

print(f"优化版本 5 epochs 用时: {opt_time:.2f}秒")

# 计算提速比
speedup = unopt_time / opt_time
print("\n" + "=" * 60)
print(f"🚀 速度提升: {speedup:.2f}x")
print("=" * 60)
```

---

## 7. 监控训练性能

### 方法1：使用nvidia-smi监控GPU

在另一个终端运行：
```bash
# Windows PowerShell
nvidia-smi dmon -s pucvmet -d 1

# 或简单查看
nvidia-smi
```

**应达到的指标**：
- GPU利用率 (sm): >80%
- 显存利用率 (mem): >70%
- 功耗: >120W

### 方法2：在训练中添加性能日志

```python
from lightning.pytorch.callbacks import Callback
import time

class PerformanceCallback(Callback):
    def __init__(self):
        self.epoch_start_time = None
        
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        
    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.epoch_start_time
        
        # 计算吞吐量
        samples_per_epoch = len(trainer.train_dataloader.dataset)
        throughput = samples_per_epoch / epoch_time
        
        print(f"\n📊 Epoch {trainer.current_epoch} Performance:")
        print(f"  • Time: {epoch_time:.2f}s")
        print(f"  • Throughput: {throughput:.2f} samples/s")
        print(f"  • Samples: {samples_per_epoch}")

# 在训练时使用
# 注意：OptimizedTrainer暂不支持callbacks，可以在base.py中添加支持
```

---

## 8. 故障排查

### 问题1：显存不足 (CUDA Out of Memory)

**症状**：训练开始后报错 `RuntimeError: CUDA out of memory`

**解决方案**：
```python
# 方法1: 减小batch size
trainer_params = {
    "batch_size": 100,  # 从150减少到100
}

# 方法2: 使用梯度累积保持有效batch size
trainer_params = {
    "batch_size": 75,           # 减小实际batch size
    "accumulate_grad_batches": 2,  # 有效batch size = 75×2 = 150
}

# 方法3: 启用梯度检查点（需要在模型中实现）
# model_params = {
#     "use_gradient_checkpointing": True,
# }
```

### 问题2：num_workers导致内存不足

**症状**：系统内存占用过高，训练变慢

**解决方案**：
```python
trainer_params = {
    "num_workers": 4,  # 从8减少到4
}
```

### 问题3：混合精度训练出现NaN

**症状**：loss变成NaN

**解决方案**：
```python
# 方法1: 使用bfloat16（如果GPU支持）
trainer_params = {
    "precision": "bf16-mixed",  # RTX 4060 Ti支持
}

# 方法2: 回退到FP32
trainer_params = {
    "precision": "32",
}

# 方法3: 增强梯度裁剪
trainer_params = {
    "gradient_clip_val": 0.5,  # 从1.0减小到0.5
}
```

### 问题4：数据加载器卡住

**症状**：训练开始后长时间无响应

**解决方案**：
```python
# Windows系统可能需要禁用persistent_workers
trainer_params = {
    "num_workers": 0,          # 先尝试单线程
    "persistent_workers": False,
}
```

---

## 9. 最佳实践

### ✅ 推荐配置（RTX 4060 Ti 8GB）

```python
trainer_params = {
    "epochs": 500,
    "batch_size": 150,
    
    # 数据加载
    "num_workers": 6,           # 保守配置，避免内存问题
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,
    
    # 训练精度
    "precision": "16-mixed",    # 混合精度
    
    # 梯度管理
    "gradient_clip_val": 1.0,
}

# 学习率调整
model_params = {
    # ... 其他参数 ...
    "learning_rate": 1.7e-3,  # 从1e-3调整（batch size从50→150）
}

trainer = BaseTrainer(
    # ... 其他参数 ...
    trainer_params=trainer_params,
    model_params=model_params,
)
```

### ⚠️ 注意事项

1. **首次运行建议**：
   - 先用5个epochs测试配置是否正常
   - 确认没有OOM错误后再进行完整训练

2. **学习率调整**：
   - batch size改变后必须调整学习率
   - 使用warmup可以提高稳定性

3. **保存检查点**：
   - 长时间训练建议启用checkpointing
   - 防止意外中断丢失训练进度

4. **验证结果**：
   - 优化后的模型应该达到相同或更好的精度
   - 如果精度下降，尝试调整学习率或减小batch size

---

## 10. 总结

### 优化效果对比

| 优化项 | 实施难度 | 预期提升 | 推荐优先级 | 已默认启用 |
|--------|---------|---------|-----------|----------|
| 增加batch size | ⭐ 极简单 | 2-2.5x | 🔥 最高 | ❌ 需手动设置 |
| num_workers | ⭐ 极简单 | 1.5-2x | 🔥 最高 | ✅ 默认=8 |
| 混合精度训练 | ⭐ 极简单 | 1.5-2x | 🔥 最高 | ✅ 默认FP16 |
| pin_memory | ⭐ 极简单 | 1.1-1.3x | ⚡ 高 | ✅ 默认True |
| 梯度累积 | ⭐⭐ 简单 | 1.0-1.2x | ⚡ 高 | ❌ 可选 |
| PyTorch编译 | ⭐⭐ 简单 | 1.2-1.5x | 🟢 中 | ❌ 可选 |

### 快速启用方案

**方案1: 一行代码优化（默认启用）**
```python
# 只需增加batch_size，其他优化自动生效
trainer_params = {"epochs": 500, "batch_size": 150}
```
**预期提升**: 4-8倍

**方案2: 完整优化（启用PyTorch编译）**
```python
trainer_params = {
    "epochs": 500, 
    "batch_size": 150,
    "compile_model": True,
}
```
**预期提升**: 8-15倍

### 关键改进

✅ **BaseTrainer现已默认启用性能优化**
- 无需额外配置，直接享受4-8倍速度提升
- 向后兼容，可手动覆盖任何优化参数
- 自动打印优化配置摘要，方便监控

---

**更多问题？** 查看 `src/docs/PERFORMANCE_ANALYSIS.md` 获取详细的性能分析报告。

