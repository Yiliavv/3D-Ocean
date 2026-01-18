# 训练配置（轻量级，不导入 PyTorch）
# 可被 notebook 和训练脚本共享
# 
# V3.1 优化版本 - 8G 显存优化，目标 RMSE < 0.5°C
# 主要改进：增加模型参数 + 数据增强

from src.config.area import Area

# ============================================================
# 区域配置
# ============================================================
area = Area('Global', lon=[-180, 180], lat=[-80, 80], description='全球区域')

# ============================================================
# 基础参数
# ============================================================
resolution = 1
seq_len = 2

# 计算空间尺寸
# 注意: Area 类中 width=lat_range, height=lon_range（命名颠倒）
# 这里修正为: width=lon_dim, height=lat_dim
width = int(area.height / resolution)   # lon dimension (360)
height = int(area.width / resolution)   # lat dimension (160)

# ============================================================
# 数据集参数
# ============================================================
dataset_params = {
    "seq_len": seq_len,
    "offset": 0,
    "resolution": resolution,
}

# ============================================================
# 模型参数（V3.2 - 小样本优化版）
# ============================================================
# 
# 🎯 针对月平均数据（~530样本）的策略：
#    数据量有限时，应该简化模型 + 加强正则化
#    而不是盲目增大模型（会过拟合）
#
model_params = {
    "width": width,
    "height": height,
    "resolution": resolution,
    "lat_range": area.lat,
    "lon_range": area.lon,
    "seq_len": seq_len,
    
    # === 适中的模型容量（避免过拟合）===
    "d_model": 256,                    # 模型维度（适中）
    "num_heads": 8,                    # 注意力头数
    "dim_feedforward": 512,            # FFN 维度 = d_model * 2
    
    # === 强正则化（防止过拟合）===
    "dropout": 0.2,                    # Dropout 增加到 0.1
    
    # === 非线性增强（这个有用）===
    "ffn_activation": "swiglu",        # FFN 激活函数
    "use_se_attention": True,          # SE 通道注意力
    
    # === 显存配置 ===
    "use_gradient_checkpointing": False,
    
    # === 损失函数 ===
    "loss_type": "huber",              # Huber 损失（对异常值鲁棒）
    "huber_delta": 1.0,
    
    # === 优化器配置（小数据需要更谨慎）===
    "learning_rate": 1e-4,             # 学习率（降低，更稳定）
    "use_lr_scheduler": True,          # 余弦退火
    "warmup_epochs": 10,               # 预热轮数
    "min_lr": 1e-6,                    # 最小学习率
    "weight_decay": 0.05,              # 权重衰减增加（重要！）
}

# ============================================================
# 训练器参数（小样本优化版）
# ============================================================
# 
# 小数据策略：
# - 较小的 batch_size（更好的泛化）
# - 更长的训练（给模型足够时间收敛）
# - 严格的早停（防止过拟合）
#
trainer_params = {
    # === 基础训练参数 ===
    "epochs": 100,                    # 小数据需要更多轮次
    "batch_size": 8,                  # 适中批量
    
    # === 梯度累积（可选）===
    "accumulate_grad_batches": 1,      # 数据少时不需要累积
    
    # === 数据加载 ===
    "num_workers": 8,
    "pin_memory": True,
    
    # === 混合精度训练 ===
    "precision": "16-mixed",
    "matmul_precision": "high",
    
    # === Checkpoint 配置 ===
    "use_wandb": True,
    "use_checkpoint": True,
    "save_top_k": 3,
    "monitor": "val_loss",
    "mode": "min",
    
    # === 早停策略（重要！防止过拟合）===
    "early_stopping_patience": 10,     # 30轮无改善则停止
    
    # === 梯度裁剪 ===
    "gradient_clip_val": 1.0,
}

# ============================================================
# 配置说明
# ============================================================
# 
# 🎯 目标: RMSE < 0.25°C (接近SOTA水平)
# 
# V4.0 更新 - 对标SOTA方法优化:
# 
# 1. 【序列长度优化】⭐ 最重要的改动
#    - seq_len: 2 -> 10
#    - 使用过去9天历史预测下一天
#    - 对标: U-Transformer(10天), CAAD-Transformer(30天)
#    - 预期RMSE改进: 10-15%
# 
# 2. 【后续优化方向】(待实施)
#    - 添加坐标注意力 (CAAD-Transformer思路)
#    - 添加轻量谱卷积 (FNO思路)
#    - 添加物理约束损失 (SSTODE思路)
# 
# 显存估算 (seq_len=10):
#   模型参数: ~1M * 4 bytes * 2 (optimizer) ≈ 8MB
#   激活值: ~3GB (batch_size=16, seq_len=10)
#   梯度: ~1GB
#   总计: ~4-5GB，显存充足

