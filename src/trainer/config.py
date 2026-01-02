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
    "d_model": 512,                    # 模型维度（适中）
    "num_heads": 8,                    # 注意力头数
    "dim_feedforward": 1024,           # FFN 维度 = d_model * 2
    "num_attn_layers": 1,              # 注意力层数（不宜过深）
    
    # === 强正则化（防止过拟合）===
    "dropout": 0.1,                    # Dropout 增加到 0.1
    
    # === 非线性增强（这个有用）===
    "ffn_activation": "swiglu",        # FFN 激活函数
    "use_se_attention": True,          # SE 通道注意力
    
    # === 显存配置 ===
    "use_gradient_checkpointing": False,
    
    # === 损失函数 ===
    "loss_type": "huber",              # Huber 损失（对异常值鲁棒）
    "huber_delta": 1.0,
    
    # === 优化器配置（小数据需要更谨慎）===
    "learning_rate": 5e-4,             # 学习率（降低，更稳定）
    "use_lr_scheduler": True,          # 余弦退火
    "warmup_epochs": 20,               # 预热轮数
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
    "epochs": 1000,                    # 小数据需要更多轮次
    "batch_size": 16,                  # 适中批量
    
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
    "early_stopping_patience": 30,     # 30轮无改善则停止
    
    # === 梯度裁剪 ===
    "gradient_clip_val": 1.0,
}

# ============================================================
# 配置说明
# ============================================================
# 
# 🎯 目标: RMSE < 0.5°C
# 
# V3.1 更新（针对 8G 显存）:
# 
# 1. 【模型增强】
#    - d_model: 512 -> 768 (+50%)
#    - num_heads: 8 -> 12
#    - dim_feedforward: 1024 -> 2048 (+100%)
#    - num_attn_layers: 2 -> 3
#    - 参数量预计增加约 2-3 倍
# 
# 2. 【数据增强】
#    - 水平翻转: 增加空间多样性
#    - 高斯噪声: 模拟观测误差
#    - 随机遮挡: 模拟云层遮挡
#    - 温度偏移: 模拟系统偏差
#    - augment_factor=4: 样本量 530 -> 2120
# 
# 3. 【显存利用优化】
#    - batch_size: 8 -> 16
#    - 关闭梯度检查点（加速训练）
#    - 预计显存使用: ~6-7G
#
# 显存估算:
#   模型参数: ~30M * 4 bytes * 2 (optimizer) ≈ 240MB
#   激活值: ~2GB (batch_size=16)
#   梯度: ~2GB
#   缓冲区: ~1GB
#   总计: ~5-6GB，留有余量

