# 训练配置（轻量级，不导入 PyTorch）
# 可被 notebook 和训练脚本共享

from src.config.area import Area

# ============================================================
# 区域配置
# ============================================================
area = Area('Global', lon=[-180, 180], lat=[-90, 90], description='全球区域')

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
# 模型参数
# ============================================================
model_params = {
    "width": width,
    "height": height,
    "resolution": resolution,
    "lat_range": area.lat,
    "lon_range": area.lon,
    "seq_len": seq_len,
    "d_model": 512,
    "num_heads": 8,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "num_attn_layers": 2,  # V2: 替代 recursion_depth
    "learning_rate": 1e-3,
}

# ============================================================
# 训练器参数
# ============================================================
trainer_params = {
    "epochs": 300,
    "batch_size": 24,
    "num_workers": 4,
    "use_wandb": True,
    "use_checkpoint": True,
    "save_top_k": 1,
    "monitor": "val_loss",
    "mode": "min",
}

