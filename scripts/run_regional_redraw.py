"""
Run redraw tasks for publication figures.

Usage:
    uv run python scripts/run_regional_redraw.py
"""

import sys
sys.path.insert(0, 'X:/Workspace/3D-Ocean')

import os
import numpy as np
import torch

from src.models.SST.RGTransformer import RGTransformer
from src.dataset.OISST import OISSTDailyDataset
from src.config.area import Area
from src.config.params import CHECKPOINT_SAVE_PATH

RUN_ID = '2026-02-04-14-58'
TEST_OFFSET = 520
RUN_UNET_ANALYSIS_ONLY = True
UNET_CHECKPOINT_PATH = 'X:/WorkSpace/3D-Ocean/src/out/checkpoints/2026-05-22-21-48/UNet3D.ckpt'
UNET_N_DEPTH = 58
UNET_BASE_CHANNELS = 160

area = Area('Global', lon=[-180, 180], lat=[-80, 80], description='Global')
resolution = 1
seq_len = 2
width = int(area.height / resolution)   # lon dim (360)
height = int(area.width / resolution)   # lat dim (160)

model_params = {
    'width': width,
    'height': height,
    'resolution': resolution,
    'lat_range': area.lat,
    'lon_range': area.lon,
    'seq_len': seq_len,
    'd_model': 256,
    'num_heads': 8,
    'dim_feedforward': 512,
    'dropout': 0.2,
    'ffn_activation': 'swiglu',
    'use_se_attention': True,
    'use_gradient_checkpointing': False,
    'loss_type': 'huber',
    'huber_delta': 1.0,
    'learning_rate': 1e-4,
    'use_lr_scheduler': True,
    'warmup_epochs': 10,
    'min_lr': 1e-6,
    'weight_decay': 0.05,
}


def redraw_unet_analysis_figures():
    """
    Redraw UNet-3D process figures using the same OISST sample source.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    from src.analysis.profile import UNet3DNatureAnalyzer
    from src.models.Profile.UNet3D import UNet3D

    if not UNET_CHECKPOINT_PATH or not os.path.exists(UNET_CHECKPOINT_PATH):
        raise FileNotFoundError(
            'Set UNET_CHECKPOINT_PATH to your trained UNet3D.ckpt before redrawing.'
        )

    model = UNet3D.load_from_checkpoint(
        UNET_CHECKPOINT_PATH,
        n_channels=1,
        n_depth=UNET_N_DEPTH,
        base_channels=UNET_BASE_CHANNELS,
    )
    model.eval()

    # Use the same spatial domain as the training dataset for feature visualization.
    dataset = OISSTDailyDataset(
        lon=area.lon,
        lat=area.lat,
        seq_len=seq_len,
        offset=TEST_OFFSET,
        resolution=resolution,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    _, output_tensor = next(iter(loader))
    sample_data = output_tensor.detach().cpu().numpy()[0, :, :]

    analyzer = UNet3DNatureAnalyzer(model, output_dir='out')
    fig2 = analyzer.generate_fig2_feature_maps(sample_data, verbose=False)
    fig3 = analyzer.generate_fig3_skip_connections(sample_data, verbose=False)
    fig4 = analyzer.generate_fig4_nan_handling(sample_data, verbose=False)
    plt.close('all')

    print('[OK] UNet analysis figures saved:')
    print(f'- {fig2["save_path"]}')
    print(f'- {fig3["save_path"]}')
    print(f'- {fig4["save_path"]}')


if RUN_UNET_ANALYSIS_ONLY:
    redraw_unet_analysis_figures()
    raise SystemExit(0)

# Load model
ckpt_path = f'{CHECKPOINT_SAVE_PATH}/{RUN_ID}/RGTransformer.ckpt'
print(f'[Load] {ckpt_path}')
assert os.path.exists(ckpt_path), f'Checkpoint not found: {ckpt_path}'

model = RGTransformer.load_from_checkpoint(ckpt_path, strict=False, **model_params)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()
print(f'[OK] Model loaded on {device}')

# Load dataset sample
dataset = OISSTDailyDataset(
    lon=area.lon, lat=area.lat,
    seq_len=seq_len, offset=TEST_OFFSET, resolution=resolution,
)

from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=1, shuffle=False)
input_tensor, output_tensor = next(iter(loader))

input_tensor = input_tensor.to(device)
output_tensor = output_tensor.to(device)

with torch.no_grad():
    pred_tensor = model(input_tensor)

output_np = output_tensor.detach().cpu().numpy()[0, :, :]
pred_np = pred_tensor.detach().cpu().numpy()[0, :, :]

masked = np.isnan(output_np)
pred_np[masked] = np.nan

diff = pred_np - output_np
rmse = float(np.sqrt(np.nanmean(diff ** 2)))
r2 = float(1 - np.nanmean(diff ** 2) / np.nanmean((output_np - np.nanmean(output_np)) ** 2))

print(f'Global RMSE: {rmse:.4f} C')
print(f'Global R2:   {r2:.4f}')
print(f'pred shape:  {pred_np.shape}')
print(f'true shape:  {output_np.shape}')

# Diagnose per-region stats
from src.analysis.regional import OCEAN_REGIONS

lon_array = np.linspace(-180, 180, 360, endpoint=False)
lat_array = np.linspace(-80, 80, 160)
error = pred_np - output_np

print('\n--- Region diagnostics ---')
for rk, rinfo in OCEAN_REGIONS.items():
    rlon, rlat = rinfo['lon'], rinfo['lat']
    if rlon[0] == -180 and rlon[1] == 180:
        lm = np.ones(360, dtype=bool)
    else:
        lm = (lon_array >= rlon[0]) & (lon_array <= rlon[1])
    lat_m = (lat_array >= rlat[0]) & (lat_array <= rlat[1])
    mask = np.outer(lat_m, lm)

    true_r = output_np[mask]
    true_r = true_r[~np.isnan(true_r)]
    err_r = error[mask]
    err_r = err_r[~np.isnan(err_r)]

    if true_r.size == 0:
        continue

    ss_res = float(np.sum(err_r ** 2))
    ss_tot = float(np.sum((true_r - np.mean(true_r)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse_r = float(np.sqrt(np.mean(err_r ** 2)))
    bias_r = float(np.mean(err_r))

    print(f'{rinfo["name"]:35s}  pixels={true_r.size:5d}  '
          f'true_mean={np.mean(true_r):6.2f}  true_std={np.std(true_r):5.3f}  '
          f'RMSE={rmse_r:.4f}  Bias={bias_r:+.4f}  '
          f'ss_res={ss_res:.2f}  ss_tot={ss_tot:.2f}  R2={r2:.4f}')
print()

# Generate plot
from src.plot.redraw import plot_regional_prediction_performance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, save_path = plot_regional_prediction_performance(
    pred_output=pred_np,
    true_output=output_np,
    lon=area.lon,
    lat=area.lat,
    output_dir='out',
)

print(f'[OK] Plot saved to: {save_path}')
plt.close(fig)

# Generate region detail figures for selected regions
from src.plot.redraw import plot_region_detail

SELECTED_REGIONS = {
    'southern_ocean': {
        'name': 'Southern Ocean',
        'lon': [80, 180],
        'lat': [-60, -40],
    },
    'west_pacific_equatorial': {
        'name': 'Western Pacific Equatorial',
        'lon': [120, 180],
        'lat': [-4.5, 4.5],
    },
    'northwest_pacific': {
        'name': 'Northwest Pacific',
        'lon': [125, 170],
        'lat': [25, 40],
    },
    'central_east_pacific_equatorial': {
        'name': 'Central-East Pacific Equatorial',
        'lon': [-180, -120],
        'lat': [-4.5, 4.5],
    },
}

for rk, rinfo in SELECTED_REGIONS.items():
    fig_r, path_r = plot_region_detail(
        pred_output=pred_np,
        true_output=output_np,
        lon=area.lon,
        lat=area.lat,
        region_key=rk,
        region_info=rinfo,
        output_dir='out',
    )
    print(f'[OK] {rk}: {path_r}')
    plt.close(fig_r)
print(f'[DONE] {len(SELECTED_REGIONS)} region detail figures generated.')
