import os
import json
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

# ---------------- Configuration ---------------- #
# Training output directory (update the path if the folder name changes)
OUTPUT_DIR = 'B://workspace/tensorflow/train_output/Transformer 训练器'
# Default batch_size used when the log does not explicitly provide one
DEFAULT_BATCH_SIZE = 20

# Matplotlib settings: allow negative sign and configure common CJK fonts
plt.rcParams["axes.unicode_minus"] = False  # Properly display minus sign

# ---------------- Utility Functions ---------------- #

def _group_loss_by_step(loss_seq: List[float], batch_size: int) -> List[float]:
    """Reshape the flattened loss sequence according to batch_size and
    calculate the mean loss for every training step."""
    steps = len(loss_seq) // batch_size
    return [
        float(np.mean(loss_seq[i * batch_size : (i + 1) * batch_size]))
        for i in range(steps)
    ] if steps > 0 else []


def _parse_log_file(path: str) -> Dict:
    """Parse a single JSON log file and return a cleaned-up information
    dictionary ready for analysis."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"[Warning] Skip unreadable file {path}: {e}")
        return {}

    fname = os.path.basename(path)
    try:
        timestamp = datetime.strptime(fname.split(".")[0], "%Y-%m-%d-%H-%M")
    except ValueError:
        timestamp = None

    batch_size = data.get("batch_size", DEFAULT_BATCH_SIZE)
    loss_seq = data.get("val_loss", [])
    step_loss = _group_loss_by_step(loss_seq, batch_size)

    return {
        "file": fname,
        "timestamp": timestamp,
        "epoch": data.get("epoch"),
        "batch_size": batch_size,
        "raw_loss": loss_seq,
        "step_loss": step_loss,
        "n_steps": len(step_loss),
        "final_loss": step_loss[-1] if step_loss else None,
    }


def collect_logs(directory: str) -> pd.DataFrame:
    """遍历目录读取所有 JSON 日志，并合并为 DataFrame。"""
    records: List[Dict] = []
    for fname in os.listdir(directory):
        if not fname.endswith(".json"):
            continue
        record = _parse_log_file(os.path.join(directory, fname))
        if record:
            records.append(record)
    if not records:
        raise RuntimeError("指定目录下未找到有效的日志文件！")
    return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)

# ---------- Y-axis configuration ---------- #

def _configure_yaxis(ax):
    """Use a symlog scale to magnify the 0–2 region and add fine ticks there."""
    # Piece-wise scale: linear between -linthresh and +linthresh, logarithmic outside
    ax.set_yscale("symlog", linthresh=2, linscale=2)

    # Add minor ticks every 0.2 within 0–2 range for better resolution
    ymin, ymax = ax.get_ylim()
    lower = max(0.0, ymin)
    upper = min(2.0, ymax)
    if lower < upper:
        minor_ticks = np.arange(lower, upper + 1e-6, 0.2)
        ax.set_yticks(minor_ticks, minor=True)
        ax.tick_params(axis="y", which="minor", length=3)

# ---------------- Plotting Functions ---------------- #


def plot_overview(df: pd.DataFrame):
    """Plot aggregated statistical charts based on all log files."""
    # 1. Final loss over time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["timestamp"], df["final_loss"], "o-", color="#d62728")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Final Validation Loss", fontsize=12)
    ax.set_title("Final Validation Loss over Time", fontsize=14)
    ax.grid(alpha=0.3, linestyle="--")
    fig.autofmt_xdate()
    fig.tight_layout()
    _configure_yaxis(ax)

    # 2. Histogram of training-step counts
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["n_steps"], bins=20, color="#2ca02c", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("File Count", fontsize=12)
    ax.set_title("Distribution of Training Steps", fontsize=14)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    _configure_yaxis(ax)

    # 3. Scatter: initial vs. final loss
    df_non_empty = df[df["step_loss"].apply(len) > 0].copy()
    df_non_empty["initial_loss"] = df_non_empty["step_loss"].apply(lambda x: x[0])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df_non_empty["initial_loss"], df_non_empty["final_loss"], color="#9467bd", alpha=0.75)
    ax.set_xlabel("Initial Loss", fontsize=12)
    ax.set_ylabel("Final Loss", fontsize=12)
    ax.set_title("Initial vs. Final Loss", fontsize=14)
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    _configure_yaxis(ax)

    # 4. Comparison of different batch_sizes under the same epoch
    # Find epochs that have multiple batch sizes available
    epoch_multi_bs = (
        df.groupby("epoch")["batch_size"].nunique()
        .loc[lambda s: s > 1]
        .index.tolist()
    )

    if epoch_multi_bs:
        n_epochs = len(epoch_multi_bs)
        ncols = 2 if n_epochs > 1 else 1
        nrows = int(np.ceil(n_epochs / ncols))
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(8 * ncols, 4 * nrows),
            squeeze=False,
        )

        # Color cycle for different batch sizes
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for idx, epoch_val in enumerate(epoch_multi_bs):
            row_idx, col_idx = divmod(idx, ncols)
            ax = axes[row_idx][col_idx]

            epoch_df = df[df["epoch"] == epoch_val]
            # Sort by batch_size for consistent legend order
            epoch_df = epoch_df.sort_values("batch_size")

            for i, (_, rec) in enumerate(epoch_df.iterrows()):
                if not rec["step_loss"]:
                    continue
                ax.plot(
                    rec["step_loss"],
                    label=f"batch_size={rec['batch_size']}",
                    color=color_cycle[i % len(color_cycle)],
                )

            ax.set_title(f"Epoch {epoch_val}", fontsize=12)
            ax.set_xlabel("Training Steps", fontsize=10)
            ax.set_ylabel("Loss", fontsize=10)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=0.3, linestyle="--")
            ax.legend(fontsize=9)
            _configure_yaxis(ax)

        # Hide unused subplots if any
        total_axes = nrows * ncols
        for j in range(len(epoch_multi_bs), total_axes):
            axes.flat[j].set_visible(False)

        fig.tight_layout()
        _configure_yaxis(ax)

# ---------------- Main ---------------- #

def main():
    df = collect_logs(OUTPUT_DIR)

    # 绘制整体统计图
    plot_overview(df)


if __name__ == "__main__":
    main() 