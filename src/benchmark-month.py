"""
SST 模型对比实验脚本（OISST 月平均数据集）

运行 20 组实验：5 模型 x 2 epochs x 2 序列长度
记录：训练时间、参数量、最佳 train_loss、val_loss、val_rmse

数据源: OISST 海表月平均温度 (1981-2025)

用法: python src/benchmark-month.py
"""

import os
import sys
import csv
import time
import torch
import tempfile
import platform
import numpy as np

sys.path.append('X:/Workspace/3D-Ocean')

from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader, TensorDataset


class EpochLogger(Callback):
    """每个 epoch 结束时覆盖式刷新一行摘要"""
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        t_loss = metrics.get("train_loss_epoch", metrics.get("train_loss", 0))
        v_loss = metrics.get("val_loss", 0)
        v_rmse = metrics.get("val_rmse", 0)
        line = f"\r  Epoch {epoch:>3d}/{trainer.max_epochs} | train_loss: {t_loss:.4f} | val_loss: {v_loss:.4f} | val_rmse: {v_rmse:.4f}    "
        sys.stdout.write(line)
        sys.stdout.flush()

    def on_train_end(self, trainer, pl_module):
        print()

from src.config.area import Area
from src.config.params import PROJECT_PATH
from src.dataset.OISST import OISSTMonthlyDataset

from src.models.SST.LSTM import LSTM
from src.models.SST.ConvLSTM import ConvLSTM
from src.models.SST.UNetLSTM import UNetLSTM
from src.models.SST.Transformer import SSTTransformer
from src.models.SST.RGTransformer import RGTransformer

import torch.nn.functional as F


# ============================================================
# UNetLSTM 适配：输入 [B, S-1, W, H] -> UNetLSTM 期望 [B, S, C, H, W]
# ============================================================

class WrappedUNetLSTM(UNetLSTM):
    def __init__(self, width, height, seq_len, **kwargs):
        super().__init__(
            input_channels=1, output_channels=1,
            features=[16, 32, 64, 128], lstm_hidden_channels=64,
            learning_rate=kwargs.get('learning_rate', 1e-4),
        )
        self.width = width
        self.height = height
        self.seq_len = seq_len

    def forward(self, x):
        x = x.unsqueeze(2).permute(0, 1, 2, 4, 3)  # [B,S-1,W,H] -> [B,S-1,1,H,W]
        out = super().forward(x)  # [B, 1, H, W]
        return out.squeeze(1).permute(0, 2, 1)  # [B, W, H]

    def _compute_loss(self, y_pred, y_true):
        mask = ~(torch.isnan(y_true) | torch.isnan(y_pred))
        if mask.sum() == 0:
            return y_pred.sum() * 0.0
        return F.mse_loss(y_pred[mask], y_true[mask])

    def training_step(self, batch, batch_idx=None):
        x, y = batch
        loss = self._compute_loss(self(x), y)
        self.train_loss.append(loss.detach().cpu().item())
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx=None):
        x, y = batch
        loss = self._compute_loss(self(x), y)
        self.val_loss.append(loss.detach().cpu().item())
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_rmse', torch.sqrt(loss), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def get_num_parameters(self, trainable_only=True):
        return sum(p.numel() for p in self.parameters() if (not trainable_only or p.requires_grad))


# ============================================================
# Transformer 适配：输出 [B, 1, W, H] -> [B, W, H]，补齐接口
# ============================================================

class WrappedTransformer(SSTTransformer):
    def __init__(self, width, height, seq_len, d_model=256, learning_rate=1e-4, **kwargs):
        super().__init__(
            width=width, height=height, seq_len=seq_len,
            d_model=d_model, learning_rate=learning_rate,
            nhead=8, num_encoder_layers=2, num_decoder_layers=2,
            dim_feedforward=512, dropout=0.1,
        )

    def forward(self, x):
        out = super().forward(x)  # [B, 1, W, H] (W=lon, H=lat)
        return out.squeeze(1).permute(0, 2, 1)  # [B, lat, lon] 与 target 一致

    def _compute_loss(self, y_pred, y_true):
        mask = ~(torch.isnan(y_true) | torch.isnan(y_pred))
        if mask.sum() == 0:
            return y_pred.sum() * 0.0
        return F.mse_loss(y_pred[mask], y_true[mask])

    def training_step(self, batch, batch_idx=None):
        x, y = batch
        loss = self._compute_loss(self(x), y)
        self.train_loss.append(loss.detach().cpu().item())
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx=None):
        x, y = batch
        loss = self._compute_loss(self(x), y)
        self.val_loss.append(loss.detach().cpu().item())
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_rmse', torch.sqrt(loss), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def get_num_parameters(self, trainable_only=True):
        return sum(p.numel() for p in self.parameters() if (not trainable_only or p.requires_grad))


# ============================================================
# 实验配置
# ============================================================

AREA = Area('Global', lon=[-180, 180], lat=[-80, 80], description='全球区域')
RESOLUTION = 2
WIDTH = int(AREA.height / RESOLUTION)   # 180
HEIGHT = int(AREA.width / RESOLUTION)   # 80

EXPERIMENTS = [
    # LSTM
    {"model": "LSTM", "epochs": 100, "seq_len": 2},
    {"model": "LSTM", "epochs": 100, "seq_len": 7},
    {"model": "LSTM", "epochs": 300, "seq_len": 2},
    {"model": "LSTM", "epochs": 300, "seq_len": 7},
    # ConvLSTM
    {"model": "ConvLSTM", "epochs": 100, "seq_len": 2},
    {"model": "ConvLSTM", "epochs": 100, "seq_len": 7},
    {"model": "ConvLSTM", "epochs": 300, "seq_len": 2},
    {"model": "ConvLSTM", "epochs": 300, "seq_len": 7},
    # UNetLSTM
    {"model": "UNetLSTM", "epochs": 100, "seq_len": 2},
    {"model": "UNetLSTM", "epochs": 100, "seq_len": 7},
    {"model": "UNetLSTM", "epochs": 300, "seq_len": 2},
    {"model": "UNetLSTM", "epochs": 300, "seq_len": 7},
    # Transformer
    {"model": "Transformer", "epochs": 100, "seq_len": 2},
    {"model": "Transformer", "epochs": 100, "seq_len": 7},
    {"model": "Transformer", "epochs": 300, "seq_len": 2},
    {"model": "Transformer", "epochs": 300, "seq_len": 7},
    # RGTransformer
    {"model": "RGTransformer", "epochs": 100, "seq_len": 2},
    {"model": "RGTransformer", "epochs": 100, "seq_len": 7},
    {"model": "RGTransformer", "epochs": 300, "seq_len": 2},
    {"model": "RGTransformer", "epochs": 300, "seq_len": 7},
]

MODEL_CLASSES = {
    "LSTM": LSTM,
    "ConvLSTM": ConvLSTM,
    "UNetLSTM": WrappedUNetLSTM,
    "Transformer": WrappedTransformer,
    "RGTransformer": RGTransformer,
}

BASE_MODEL_PARAMS = {
    "width": WIDTH,
    "height": HEIGHT,
    "resolution": RESOLUTION,
    "lat_range": AREA.lat,
    "lon_range": AREA.lon,
    "learning_rate": 1e-3,
    "dropout": 0.1,
    "weight_decay": 0.01,
}

TRAINER_PARAMS = {
    "batch_size": 32,
    "precision": "16-mixed",
    "matmul_precision": "high",
    "gradient_clip_val": 1.0,
}

# RTX 4060 Ti 8GB 显存限制：按模型调整 batch_size
BATCH_SIZE_OVERRIDE = {
    "LSTM": 64,
    "ConvLSTM": 16,
    "UNetLSTM": 32,
    "Transformer": 32,
    "RGTransformer": 32,
}

# 缓存
_raw_data_cache = {}
_dataloader_cache = {}


def create_model(model_name: str, seq_len: int):
    """创建模型实例"""
    model_class = MODEL_CLASSES[model_name]
    params = {**BASE_MODEL_PARAMS, "seq_len": seq_len}

    if model_name == "RGTransformer":
        params.update({
            "d_model": 256, "num_heads": 8, "dim_feedforward": 512,
            "ffn_activation": "swiglu", "use_se_attention": True,
            "loss_type": "huber", "huber_delta": 1.0,
            "use_lr_scheduler": True, "warmup_epochs": 10, "min_lr": 1e-6,
        })
    elif model_name == "Transformer":
        params.update({"d_model": 256})
    elif model_name == "LSTM":
        params.update({"hidden_dim": 256, "num_layers": 2})
    elif model_name == "ConvLSTM":
        params.update({"hidden_dim": 32, "kernel_size": (3, 3), "num_layers": 1})

    return model_class(**params)


def preload_dataset(seq_len: int):
    """一次性将所有数据从磁盘加载到内存"""
    print(f"[Preload] Loading OISST monthly data into memory (seq_len={seq_len})...")
    load_start = time.time()

    dataset = OISSTMonthlyDataset(
        lon=AREA.lon, lat=AREA.lat,
        seq_len=seq_len, offset=0, resolution=RESOLUTION,
    )

    total_size = len(dataset)
    x0, y0 = dataset[0]
    all_x = torch.empty(total_size, *x0.shape, dtype=torch.float32)
    all_y = torch.empty(total_size, *y0.shape, dtype=torch.float32)

    for i in range(total_size):
        x, y = dataset[i]
        all_x[i] = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
        all_y[i] = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)

    load_time = time.time() - load_start
    mem_mb = (all_x.nbytes + all_y.nbytes) / 1024**2
    print(f"[Preload] Done: {total_size} samples, {mem_mb:.1f} MB, {load_time:.1f}s")
    return all_x, all_y


def create_dataloaders(seq_len: int, batch_size: int):
    """创建纯内存 DataLoader"""
    cache_key = (seq_len, batch_size)
    if cache_key in _dataloader_cache:
        print(f"[Data] Reusing cached dataloaders (seq_len={seq_len}, bs={batch_size})")
        return _dataloader_cache[cache_key]

    if seq_len not in _raw_data_cache:
        _raw_data_cache[seq_len] = preload_dataset(seq_len)

    all_x, all_y = _raw_data_cache[seq_len]
    total = len(all_x)
    split = int(total * 0.9)

    train_loader = DataLoader(
        TensorDataset(all_x[:split], all_y[:split]),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(all_x[split:], all_y[split:]),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )

    _dataloader_cache[cache_key] = (train_loader, val_loader)
    return train_loader, val_loader


def run_experiment(exp_config: dict, exp_id: int, total: int) -> dict:
    """运行单个实验"""
    model_name = exp_config["model"]
    epochs = exp_config["epochs"]
    seq_len = exp_config["seq_len"]

    print("\n" + "=" * 70)
    print(f"[Experiment {exp_id}/{total}] {model_name} | epochs={epochs} | seq_len={seq_len}")
    print("=" * 70)

    model = create_model(model_name, seq_len)
    num_params = model.get_num_parameters()
    print(f"[Model] {model_name} | Parameters: {num_params:,}")

    batch_size = BATCH_SIZE_OVERRIDE.get(model_name, TRAINER_PARAMS["batch_size"])
    train_loader, val_loader = create_dataloaders(seq_len, batch_size)
    print(f"[Data] Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | bs: {batch_size}")

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.set_float32_matmul_precision(TRAINER_PARAMS["matmul_precision"])
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    trainer = Trainer(
        max_epochs=epochs,
        accelerator='gpu' if use_gpu else 'cpu',
        devices=1,
        precision=TRAINER_PARAMS["precision"] if use_gpu else "32",
        gradient_clip_val=TRAINER_PARAMS["gradient_clip_val"],
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=tempfile.gettempdir(),
        callbacks=[EpochLogger()],
    )

    start_time = time.time()
    trainer.fit(model, train_loader, val_loader)
    train_time = time.time() - start_time

    best_train_loss = min(model.train_loss) if model.train_loss else float('inf')
    best_val_loss = min(model.val_loss) if model.val_loss else float('inf')
    best_val_rmse = best_val_loss ** 0.5 if best_val_loss != float('inf') else float('inf')

    result = {
        "model": model_name,
        "epochs": epochs,
        "seq_len": seq_len,
        "params": num_params,
        "train_time_sec": round(train_time, 2),
        "best_train_loss": round(best_train_loss, 6),
        "best_val_loss": round(best_val_loss, 6),
        "best_val_rmse": round(best_val_rmse, 6),
    }

    print(f"\n[Result] Time: {train_time:.1f}s | TrainLoss: {best_train_loss:.6f} | ValLoss: {best_val_loss:.6f} | ValRMSE: {best_val_rmse:.4f}")

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def save_results(results: list, output_path: str):
    """保存结果到 CSV"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fields = ["model", "epochs", "seq_len", "params", "train_time_sec", "best_train_loss", "best_val_loss", "best_val_rmse"]
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[Saved] {output_path}")


def print_summary(results: list):
    """打印结果汇总"""
    print("\n" + "=" * 120)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 120)
    print(f"{'Model':<16} {'Epochs':>7} {'SeqLen':>7} {'Params':>12} {'Time(s)':>10} {'TrainLoss':>12} {'ValLoss':>12} {'ValRMSE':>10}")
    print("-" * 120)
    for r in results:
        print(f"{r['model']:<16} {r['epochs']:>7} {r['seq_len']:>7} {r['params']:>12,} {r['train_time_sec']:>10.1f} {r['best_train_loss']:>12.6f} {r['best_val_loss']:>12.6f} {r['best_val_rmse']:>10.4f}")
    print("=" * 120)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SST Model Benchmark")
    parser.add_argument("--start-from", type=int, default=1,
                        help="从第 N 个实验开始跑（1-based），跳过之前的实验")
    parser.add_argument("--skip-models", type=str, default="",
                        help="跳过的模型名，逗号分隔，如 'LSTM,ConvLSTM'")
    args = parser.parse_args()

    skip_models = set(m.strip() for m in args.skip_models.split(",") if m.strip())

    print("=" * 70)
    print("SST Model Benchmark - OISST Monthly (5 models x 2 epochs x 2 seq_len = 20 runs)")
    print("=" * 70)
    print(f"[System] OS: {platform.system()} | PyTorch: {torch.__version__}")
    print(f"[System] CUDA: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")

    print(f"\n[Config] Resolution: {RESOLUTION}deg | Spatial: {WIDTH}x{HEIGHT}")
    print(f"[Config] Total Experiments: {len(EXPERIMENTS)}")
    if args.start_from > 1:
        print(f"[Config] Starting from experiment #{args.start_from}")
    if skip_models:
        print(f"[Config] Skipping models: {', '.join(skip_models)}")

    results = []
    total = len(EXPERIMENTS)

    for i, exp in enumerate(EXPERIMENTS, 1):
        if i < args.start_from:
            print(f"\n[Skip] Experiment {i}/{total} (--start-from={args.start_from})")
            continue
        if exp["model"] in skip_models:
            print(f"\n[Skip] Experiment {i}/{total} {exp['model']} (--skip-models)")
            continue

        result = run_experiment(exp, i, total)
        results.append(result)

        save_results(results, f"{PROJECT_PATH}/out/benchmark_oisst_monthly.csv")

    print_summary(results)
    print("\n[OK] All experiments completed!")


if __name__ == "__main__":
    main()
