import argparse
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

from src.plot.base import add_panel_label, apply_plot_style

# -----------------------------------------------------------------------------
# Plotting Standards (User-Confirmed)
# -----------------------------------------------------------------------------
# 1) Global typography
#    - Font family: Times New Roman
#    - Global font size: 16
#    - Axis/tick font size: 14
#    - Keep all visible figure text in English
#
# 2) Axes and spacing
#    - Increase distance between axis labels and ticks
#    - Prefer shared x-axis behavior for multi-row map panels
#    - Keep map panel margins visually balanced (avoid uneven blank areas)
#
# 3) Titles and panel labels
#    - No plot titles
#    - Panel labels must use unified style via add_panel_label(), e.g. (a), (b)
#    - White rounded label box with no border
#
# 4) Map-specific conventions used in redraw figures
#    - Raw vs processed panels must preserve their intended data source semantics
#    - If data orientation is flipped, coordinate orientation must be handled consistently
#    - Use dashed horizontal lines to indicate cropped latitude band when applicable
#
# 5) File execution methods
#    - Run as module (recommended):
#      uv run python -m src.plot.redraw --task all
#    - Run by file path:
#      uv run python src/plot/redraw.py --task all
#    - Common task values:
#      --task map | season | timeline | multi_source | multi_source_diff | all
# -----------------------------------------------------------------------------


def _sort_lon_to_0_360(lon, field_2d):
    """
    Convert longitude from [-180, 180] to [0, 360] and sort accordingly.
    """
    lon_360 = np.where(lon < 0, lon + 360, lon)
    sort_idx = np.argsort(lon_360)
    return lon_360[sort_idx], field_2d[:, sort_idx]


def _create_output_path(filename, output_dir=None):
    """
    Build and ensure output path for figure export.
    """
    export_dir = output_dir or 'out'
    os.makedirs(export_dir, exist_ok=True)
    return os.path.join(export_dir, filename)


def _setup_geo_axes(ax, show_xlabel=False, show_ylabel=False, reverse_lat_axis=False):
    """
    Set full-domain axes and clipping markers consistently.
    """
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())

    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f'{int(round((x + 360) % 360))}°E')
    )
    def _format_lat(y_val):
        y_int = int(round(y_val))
        if y_int > 0:
            return f'{y_int}°N'
        if y_int < 0:
            return f'{abs(y_int)}°S'
        return '0°'

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: _format_lat(y)))
    if reverse_lat_axis:
        # Reverse entire map geometry (data + coastline + land) for this panel.
        ax.set_ylim(90, -90)

    ax.tick_params(labelsize=14, pad=8)
    ax.set_xlabel('Longitude (°E)' if show_xlabel else '', fontsize=14, labelpad=12)
    ax.set_ylabel('Latitude (°N)' if show_ylabel else '', fontsize=14, labelpad=12)

    # Dashed lines indicate cropped latitude band [-80, 80].
    ax.plot([-180, 180], [80, 80], '--', color='black', linewidth=1.0, alpha=0.7, transform=ccrs.PlateCarree())
    ax.plot([-180, 180], [-80, -80], '--', color='black', linewidth=1.0, alpha=0.7, transform=ccrs.PlateCarree())


def _wrap_lon_to_180(lon, field_2d):
    """
    Convert longitude from [0, 360) to [-180, 180) for stable Cartopy plotting.
    """
    lon_wrapped = np.where(lon > 180, lon - 360, lon)
    sort_idx = np.argsort(lon_wrapped)
    return lon_wrapped[sort_idx], field_2d[:, sort_idx]


def _month_index_from_start(start_year, start_month, target_year, target_month):
    """
    Convert (year, month) to zero-based index from dataset start month.
    """
    return (target_year - start_year) * 12 + (target_month - start_month)


def _nan_rmse_bias(field):
    """
    Compute RMSE and bias while ignoring NaN.
    """
    valid_mask = ~np.isnan(field)
    if not np.any(valid_mask):
        return np.nan, np.nan
    field_valid = field[valid_mask]
    rmse = float(np.sqrt(np.mean(field_valid ** 2)))
    bias = float(np.mean(field_valid))
    return rmse, bias


def _build_multi_source_context(resolution, target_year=2024, target_month=1):
    """
    Build shared datasets, domains, and month indices for multi-source plots.
    """
    from src.dataset.Argo import Argo3DTemperatureDataset
    from src.dataset.ERA5 import ERA5SSTMonthlyDataset
    from src.dataset.OISST import OISSTDailyMonthlyDataset, OISSTMonthlyDataset

    target_lon_180 = [-180, 180]
    target_lon_360 = [0, 360]
    target_lat = [-80, 80]

    era5_ds = ERA5SSTMonthlyDataset(seq_len=1, lon=target_lon_180, lat=target_lat, resolution=resolution)
    oisst_ds = OISSTMonthlyDataset(seq_len=1, lon=target_lon_180, lat=target_lat, resolution=resolution)
    oisst_d_ds = OISSTDailyMonthlyDataset(seq_len=1, lon=target_lon_180, lat=target_lat, resolution=resolution)
    argo_ds = Argo3DTemperatureDataset(lon=target_lon_360, lat=target_lat, depth=[0, 1], resolution=resolution)

    era5_idx = _month_index_from_start(era5_ds._start_year, 1, target_year, target_month)
    oisst_idx = _month_index_from_start(1981, 9, target_year, target_month)
    oisst_d_month_key = f'{target_year:04d}-{target_month:02d}'
    if oisst_d_month_key not in oisst_d_ds._months:
        raise ValueError(f'OISST-D monthly data does not contain {oisst_d_month_key}.')
    oisst_d_idx = oisst_d_ds._months.index(oisst_d_month_key)
    argo_idx = _month_index_from_start(2004, 1, target_year, target_month)

    return {
        'resolution': resolution,
        'target_lon_180': target_lon_180,
        'target_lon_360': target_lon_360,
        'target_lat': target_lat,
        'era5_ds': era5_ds,
        'oisst_ds': oisst_ds,
        'oisst_d_ds': oisst_d_ds,
        'argo_ds': argo_ds,
        'era5_idx': era5_idx,
        'oisst_idx': oisst_idx,
        'oisst_d_idx': oisst_d_idx,
        'argo_idx': argo_idx,
    }


def _load_multi_source_sst_fields(context):
    """
    Load and harmonize fields used by multi-source redraw figures.
    """
    resolution = context['resolution']
    target_lon_180 = context['target_lon_180']
    target_lon_360 = context['target_lon_360']
    target_lat = context['target_lat']

    era5_sst = context['era5_ds'].__read_sst__(context['era5_idx'])
    oisst_sst = context['oisst_ds'].__read_sst__(context['oisst_idx'])
    oisst_d_sst = context['oisst_d_ds']._read_monthly_sst(context['oisst_d_idx'])
    argo_sst, _ = context['argo_ds'][context['argo_idx']]
    argo_sst = argo_sst[::-1, :]

    lon_180 = np.arange(target_lon_180[0], target_lon_180[1], resolution)
    lon_360 = np.arange(target_lon_360[0], target_lon_360[1], resolution)
    lat_180 = np.arange(target_lat[0], target_lat[1], resolution)
    argo_lon_180, argo_sst_180 = _wrap_lon_to_180(lon_360, argo_sst)

    return {
        'oisst_d_sst': oisst_d_sst,
        'oisst_sst': oisst_sst,
        'era5_sst': era5_sst,
        'argo_sst': argo_sst,
        'argo_sst_180': argo_sst_180,
        'lon_180': lon_180,
        'lon_360': lon_360,
        'lat_180': lat_180,
        'argo_lon_180': argo_lon_180,
    }


def _configure_global_map_axis(
    ax,
    lon_formatter,
    lat_formatter,
    *,
    land_color,
    coastline_width,
    tick_labelsize,
    tick_pad,
    show_bottom_labels=True,
    show_left_labels=True,
):
    """
    Apply shared Cartopy map-axis settings for global SST panels.
    """
    ax.set_extent([-180, 180, -80, 80], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor=land_color, zorder=2)
    ax.coastlines(linewidth=coastline_width)

    ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))
    ax.tick_params(labelsize=tick_labelsize, pad=tick_pad)

    if not show_bottom_labels:
        ax.tick_params(labelbottom=False)
    if not show_left_labels:
        ax.tick_params(labelleft=False)


def plot_era5_oisst_argo_raw_processed(
    filename='era5_oisst_argo_raw_processed.png',
    output_dir=None,
    month_index=0,
    resolution=1,
):
    """
    Plot a 3x2 comparison panel: ERA5/OISST/Argo raw vs processed SST.
    """
    from src.dataset.Argo import Argo3DTemperatureDataset
    from src.dataset.ERA5 import ERA5SSTMonthlyDataset
    from src.dataset.OISST import OISSTMonthlyDataset

    apply_plot_style()

    target_lon = [-180, 180]
    target_lat = [-80, 80]

    era5_ds = ERA5SSTMonthlyDataset(seq_len=1, lon=target_lon, lat=target_lat, resolution=resolution)
    oisst_ds = OISSTMonthlyDataset(seq_len=1, lon=target_lon, lat=target_lat, resolution=resolution)
    argo_ds = Argo3DTemperatureDataset(lon=[0, 360], lat=target_lat, depth=[0, 1], resolution=resolution)

    # Right column: dataset output.
    era5_processed = era5_ds.__read_sst__(month_index)
    era5_lon_proc, era5_processed = _sort_lon_to_0_360(
        np.arange(target_lon[0], target_lon[1], resolution),
        era5_processed,
    )
    era5_lat_proc = np.arange(target_lat[0], target_lat[1], resolution)

    oisst_processed = oisst_ds.__read_sst__(month_index)
    oisst_lon_proc, oisst_processed = _sort_lon_to_0_360(
        np.arange(target_lon[0], target_lon[1], resolution),
        oisst_processed,
    )
    oisst_lat_proc = np.arange(target_lat[0], target_lat[1], resolution)

    argo_processed, _ = argo_ds[month_index]
    # Dataset output is latitude-flipped internally; flip back for this comparison figure.
    argo_processed = argo_processed[::-1, :]
    argo_lon_proc = np.arange(0, 360, resolution)
    argo_lat_proc = np.arange(-80, 80, resolution)

    # Left column: raw source grids.
    era5_raw = era5_ds._sst_data[month_index, :, :] - 273.15
    era5_raw = np.round(era5_raw.astype(np.float32), 3)
    era5_raw[(era5_raw > 99) | (era5_raw < -10)] = np.nan
    era5_lon_raw = np.where(era5_ds._lon_data < 0, era5_ds._lon_data + 360, era5_ds._lon_data)
    era5_lon_raw, era5_raw = _sort_lon_to_0_360(era5_lon_raw, era5_raw)
    # ERA5 raw latitude orientation: (90, -90)
    era5_lat_raw = era5_ds._lat_data[::-1].copy()
    era5_raw = era5_raw[::-1, :]

    oisst_raw = np.round(oisst_ds._sst_data[month_index, :, :].astype(np.float32), 3)
    oisst_raw[(oisst_raw > 99) | (oisst_raw < -10)] = np.nan
    oisst_lon_raw = np.where(oisst_ds._lon_data < 0, oisst_ds._lon_data + 360, oisst_ds._lon_data)
    oisst_lon_raw, oisst_raw = _sort_lon_to_0_360(oisst_lon_raw, oisst_raw)
    oisst_lat_raw = oisst_ds._lat_data.copy()

    argo_temp = argo_ds.data[month_index]['temp'].copy()
    argo_temp[argo_temp > 99] = np.nan
    # Keep Argo raw data in its original latitude orientation for this comparison.
    argo_raw = np.transpose(argo_temp, (1, 0, 2))
    argo_raw = np.round(argo_raw[:, :, 0].astype(np.float32), 3)
    argo_raw[(argo_raw > 99) | (argo_raw < -10)] = np.nan
    # User-requested visual inversion for the left (raw) Argo panel.
    argo_raw = argo_raw[::-1, :]
    argo_lon_raw = np.arange(0, 360, resolution)
    # Pair with reversed latitude coordinates so map orientation remains unchanged.
    argo_lat_raw = np.arange(-80, 80, resolution)[::-1]

    panel_data = [
        ('ERA5', era5_raw, era5_lon_raw, era5_lat_raw, era5_processed, era5_lon_proc, era5_lat_proc),
        ('OISST', oisst_raw, oisst_lon_raw, oisst_lat_raw, oisst_processed, oisst_lon_proc, oisst_lat_proc),
        ('Argo', argo_raw, argo_lon_raw, argo_lat_raw, argo_processed, argo_lon_proc, argo_lat_proc),
    ]

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(12, 10),
        sharex=True,
        subplot_kw={'projection': ccrs.PlateCarree()},
    )
    # Keep subplot area symmetric and reserve a fixed strip for colorbar.
    fig.subplots_adjust(left=0.12, right=0.84, bottom=0.08, top=0.96, hspace=0.12, wspace=0.25)

    subpanel_labels = ['a', 'b', 'c', 'd', 'e', 'f']
    color_mesh = None
    label_idx = 0

    for row_idx, (dataset_name, raw_data, raw_lon, raw_lat, proc_data, proc_lon, proc_lat) in enumerate(panel_data):
        for col_idx in range(2):
            ax = axes[row_idx, col_idx]
            ax.add_feature(cfeature.LAND, facecolor='0.9', zorder=2)
            ax.coastlines(linewidth=0.4)

            if col_idx == 0:
                data = raw_data
                lon = raw_lon
                lat = raw_lat
            else:
                data = proc_data
                lon = proc_lon
                lat = proc_lat

            lon_plot, data_plot = _wrap_lon_to_180(lon, data)

            color_mesh = ax.pcolormesh(
                lon_plot,
                lat,
                data_plot,
                cmap='RdYlBu_r',
                vmin=0,
                vmax=32,
                transform=ccrs.PlateCarree(),
                shading='auto',
            )

            _setup_geo_axes(
                ax,
                show_xlabel=(row_idx == 2),
                show_ylabel=(col_idx == 0),
                reverse_lat_axis=False,
            )
            # Enforce visual x-axis sharing: only bottom row shows x tick labels.
            if row_idx < 2:
                ax.tick_params(labelbottom=False)
            if col_idx == 0:
                ax.text(
                    -0.28,
                    0.5,
                    dataset_name,
                    transform=ax.transAxes,
                    rotation=90,
                    ha='center',
                    va='center',
                    fontsize=14,
                    fontweight='bold',
                )
            ax.set_title('')
            add_panel_label(ax, f"({subpanel_labels[label_idx]})")
            label_idx += 1

    cax = fig.add_axes([0.87, 0.20, 0.025, 0.60])
    cbar = fig.colorbar(color_mesh, cax=cax, orientation='vertical')
    cbar.set_label('SST (°C)', fontsize=14, labelpad=12)
    cbar.ax.tick_params(labelsize=14, pad=8)

    save_path = _create_output_path(filename, output_dir)
    fig.savefig(save_path, dpi=300)

    return fig, save_path


def plot_multi_source_sst_global_202401(
    filename='sst_multi_source_global_202401.png',
    output_dir=None,
    resolution=1,
):
    """
    Plot 2x2 global SST comparison for 2024-01:
    OISST-D / OISST / ERA5 / Argo.
    """
    apply_plot_style()
    context = _build_multi_source_context(resolution=resolution)
    fields = _load_multi_source_sst_fields(context)

    panel_specs = [
        ('OISST-D (REMSS MW OI)', fields['oisst_d_sst'], fields['lon_180'], fields['lat_180']),
        ('OISST (NOAA v2.1)', fields['oisst_sst'], fields['lon_180'], fields['lat_180']),
        ('ERA5 (ECMWF)', fields['era5_sst'], fields['lon_180'], fields['lat_180']),
        ('Argo (BOA)', fields['argo_sst'], fields['lon_360'], fields['lat_180']),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.08, top=0.92, hspace=0.18, wspace=0.12)

    def _fmt_lon(x_val, _):
        x_int = int(round(x_val))
        if x_int < 0:
            return f'{abs(x_int)}°W'
        if x_int > 0:
            return f'{x_int}°E'
        return '0°'

    def _fmt_lat(y_val, _):
        y_int = int(round(y_val))
        if y_int < 0:
            return f'{abs(y_int)}°S'
        if y_int > 0:
            return f'{y_int}°N'
        return '0°'

    panel_labels = ['a', 'b', 'c', 'd']
    color_mesh = None
    for idx, (title, field, lon, lat) in enumerate(panel_specs):
        row_idx, col_idx = divmod(idx, 2)
        ax = axes[row_idx, col_idx]
        _configure_global_map_axis(
            ax,
            _fmt_lon,
            _fmt_lat,
            land_color='0.92',
            coastline_width=0.4,
            tick_labelsize=10,
            tick_pad=5,
            show_bottom_labels=(row_idx != 0),
            show_left_labels=(col_idx == 0),
        )
        if idx in (2, 3):
            # User-requested longitude label style for panels (c) and (d):
            # left 180 uses west notation, right 180 keeps east notation.
            ax.set_xticklabels(['180°W', '120°W', '60°W', '0°', '60°E', '120°E', '180°E'])

        lon_plot, field_plot = _wrap_lon_to_180(lon, field)
        color_mesh = ax.pcolormesh(
            lon_plot,
            lat,
            field_plot,
            cmap='RdYlBu_r',
            vmin=0,
            vmax=32,
            transform=ccrs.PlateCarree(),
            shading='auto',
        )

        ax.set_title(title, fontsize=11, fontweight='bold', pad=4)
        add_panel_label(ax, f"({panel_labels[idx]})")

    cax = fig.add_axes([0.92, 0.18, 0.015, 0.66])
    cbar = fig.colorbar(color_mesh, cax=cax, orientation='vertical')
    cbar.set_label('SST (°C)', fontsize=12, labelpad=8)
    cbar.ax.tick_params(labelsize=10, pad=4)

    save_path = _create_output_path(filename, output_dir)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, save_path


def plot_multi_source_sst_difference_202401(
    filename='sst_multi_source_diff_202401.png',
    output_dir=None,
    resolution=1,
):
    """
    Plot 1x3 global SST differences for 2024-01 using OISST-D as reference:
    OISST-D minus OISST / ERA5 / Argo.
    """
    apply_plot_style()
    context = _build_multi_source_context(resolution=resolution)
    fields = _load_multi_source_sst_fields(context)

    diff_oisst = fields['oisst_d_sst'] - fields['oisst_sst']
    diff_era5 = fields['oisst_d_sst'] - fields['era5_sst']
    diff_argo = fields['oisst_d_sst'] - fields['argo_sst_180']

    panel_specs = [
        ('OISST-D - OISST', diff_oisst, fields['lon_180'], fields['lat_180'], 'a'),
        ('OISST-D - ERA5', diff_era5, fields['lon_180'], fields['lat_180'], 'b'),
        ('OISST-D - Argo', diff_argo, fields['argo_lon_180'], fields['lat_180'], 'c'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.3), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.16, top=0.88, wspace=0.10)

    def _fmt_lon(x_val, _):
        return f'{int(round((x_val + 360) % 360))}°E'

    def _fmt_lat(y_val, _):
        y_int = int(round(y_val))
        if y_int > 0:
            return f'{y_int}°N'
        if y_int < 0:
            return f'{abs(y_int)}°S'
        return '0°'

    color_mesh = None
    for idx, (title, field, lon, lat, panel_label) in enumerate(panel_specs):
        ax = axes[idx]
        _configure_global_map_axis(
            ax,
            _fmt_lon,
            _fmt_lat,
            land_color='0.94',
            coastline_width=0.35,
            tick_labelsize=9.5,
            tick_pad=4,
            show_bottom_labels=True,
            show_left_labels=(idx == 0),
        )

        color_mesh = ax.pcolormesh(
            lon,
            lat,
            field,
            cmap='RdBu_r',
            vmin=-2.2,
            vmax=2.2,
            transform=ccrs.PlateCarree(),
            shading='auto',
        )

        ax.text(
            0.5,
            1.02,
            title,
            transform=ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold',
        )

        rmse, bias = _nan_rmse_bias(field)
        metric_text = f'RMSE={rmse:.2f}°C   Bias={bias:+.2f}°C'
        ax.text(
            0.5,
            0.02,
            metric_text,
            transform=ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=9.5,
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8},
        )

        add_panel_label(ax, f'({panel_label})', x=0.97, y=0.95, fontsize=12)

    cax = fig.add_axes([0.93, 0.22, 0.012, 0.56])
    cbar = fig.colorbar(color_mesh, cax=cax, orientation='vertical')
    cbar.set_label('Diff (°C)', fontsize=12, labelpad=8)
    cbar.ax.tick_params(labelsize=10, pad=4)

    save_path = _create_output_path(filename, output_dir)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, save_path


def plot_multi_source_sst_consistency_202401(
    filename='sst_multi_source_consistency_202401.png',
    output_dir=None,
    resolution=1,
):
    """
    Plot 1x3 consistency/correlation scatter comparison for 2024-01 using
    OISST-D as reference: OISST-D vs OISST / ERA5 / Argo.
    """
    apply_plot_style()
    context = _build_multi_source_context(resolution=resolution)
    fields = _load_multi_source_sst_fields(context)

    def _aligned_reference_pair(reference_field, target_field):
        """
        Align field shapes by shared lat/lon size and return paired arrays.
        """
        n_lat = min(reference_field.shape[0], target_field.shape[0])
        n_lon = min(reference_field.shape[1], target_field.shape[1])
        return reference_field[:n_lat, :n_lon], target_field[:n_lat, :n_lon]

    panel_specs = [
        ('OISST', fields['oisst_sst'], 'a'),
        ('ERA5', fields['era5_sst'], 'b'),
        ('Argo', fields['argo_sst_180'], 'c'),
    ]

    paired_arrays = []
    all_values = []
    for source_name, source_field, panel_label in panel_specs:
        ref_field, src_field = _aligned_reference_pair(fields['oisst_d_sst'], source_field)
        valid_mask = (
            ~np.isnan(ref_field)
            & ~np.isnan(src_field)
            & (ref_field >= -2.5)
            & (ref_field <= 40.0)
            & (src_field >= -2.5)
            & (src_field <= 40.0)
        )
        ref_valid = ref_field[valid_mask]
        src_valid = src_field[valid_mask]
        paired_arrays.append((source_name, src_valid, ref_valid, panel_label))
        all_values.append(src_valid)
        all_values.append(ref_valid)

    if all_values and any(values.size > 0 for values in all_values):
        merged_values = np.concatenate([values for values in all_values if values.size > 0])
        axis_min = float(np.floor(np.nanmin(merged_values)))
        axis_max = float(np.ceil(np.nanmax(merged_values)))
    else:
        axis_min, axis_max = 0.0, 32.0
    axis_padding = 0.5
    axis_min -= axis_padding
    axis_max += axis_padding

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.3))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.20, top=0.92, wspace=0.18)

    for idx, (source_name, source_values, reference_values, panel_label) in enumerate(paired_arrays):
        ax = axes[idx]

        ax.scatter(
            source_values,
            reference_values,
            s=6,
            color='#1f77b4',
            alpha=0.35,
            edgecolors='none',
            rasterized=True,
        )
        ax.plot(
            [axis_min, axis_max],
            [axis_min, axis_max],
            linestyle='--',
            color='red',
            linewidth=1.8,
        )

        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.22, linewidth=0.5)
        ax.tick_params(labelsize=11, pad=5)
        ax.set_xlabel(f'{source_name} (°C)', fontsize=13, labelpad=8)
        if idx == 0:
            ax.set_ylabel('OISST-D (°C)', fontsize=13, labelpad=10)
        else:
            ax.set_ylabel('')

        if source_values.size == 0:
            rmse = np.nan
            corr = np.nan
        else:
            rmse = float(np.sqrt(np.mean((reference_values - source_values) ** 2)))
            corr = float(np.corrcoef(source_values, reference_values)[0, 1]) if source_values.size > 1 else np.nan

        metric_text = f'RMSE = {rmse:.2f}°C\nR = {corr:.4f}'
        ax.text(
            0.97,
            0.03,
            metric_text,
            transform=ax.transAxes,
            ha='right',
            va='bottom',
            fontsize=11,
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'edgecolor': '0.75', 'alpha': 0.95},
        )

        ax.text(
            0.5,
            1.02,
            f'OISST-D vs {source_name}',
            transform=ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold',
        )
        add_panel_label(ax, f'({panel_label})', x=0.97, y=0.95, fontsize=12)

    save_path = _create_output_path(filename, output_dir)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, save_path


def plot_seasonal_analysis_figures(
    filename_a='season_pattern_a.png',
    filename_b='season_pattern_b.png',
    output_dir=None,
    resolution=1,
):
    """
    Generate and export the two seasonal-analysis figures in one call.
    """
    from src.analysis.season import SeasonalityAnalysis
    from src.dataset.ERA5 import ERA5SSTMonthlyDataset

    apply_plot_style()

    dataset = ERA5SSTMonthlyDataset(
        seq_len=1,
        lon=[-180, 180],
        lat=[-80, 80],
        resolution=resolution,
    )
    analyzer = SeasonalityAnalysis(dataset)
    fig_a, fig_b = analyzer.plot_seasonal_patterns()

    save_path_a = _create_output_path(filename_a, output_dir)
    save_path_b = _create_output_path(filename_b, output_dir)
    fig_a.savefig(save_path_a, dpi=300, bbox_inches='tight')
    fig_b.savefig(save_path_b, dpi=300, bbox_inches='tight')

    return (fig_a, fig_b), (save_path_a, save_path_b)


def plot_temporal_alignment_timeline(
    filename='temporal_alignment_timeline.png',
    output_dir=None,
):
    """
    Plot before/after temporal alignment timeline for ERA5/OISST/Argo.
    """
    apply_plot_style()

    def _ym_start(year, month):
        return year + (month - 1) / 12

    def _ym_next(year, month):
        return year + month / 12

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.10, top=0.95, hspace=0.34)

    colors = {
        'ERA5': '#0072B2',
        'OISST': '#009E73',
        'Argo': '#D55E00',
    }
    split_colors = {
        'train': '#1F4E79',
        'validation': '#5D7290',
        'test': '#6C757D',
    }

    # (a) Before alignment
    add_panel_label(ax1, '(a)', x=0.06, y=0.95)
    before_ranges = [
        ('ERA5 (Reanalysis)', _ym_start(1940, 1), _ym_next(2024, 12), colors['ERA5'], '1940-01 to 2024-12'),
        ('OISST (Satellite)', _ym_start(1981, 9), _ym_next(2025, 9), colors['OISST'], '1981-09 to 2025-09'),
        ('Argo (In-situ)', _ym_start(2004, 1), _ym_next(2024, 12), colors['Argo'], '2004-01 to 2024-12'),
    ]
    y_pos_before = [2, 1, 0]
    for (label, start, end, color, _), y in zip(before_ranges, y_pos_before):
        ax1.barh(y, end - start, left=start, color=color, height=0.48, alpha=0.95)

    unified_start = _ym_start(2004, 1)
    unified_end = _ym_next(2025, 12)
    ax1.axvline(unified_start, color='0.45', linestyle='--', linewidth=1.2)
    ax1.axvline(unified_end, color='0.45', linestyle='--', linewidth=1.2)
    ax1.text(unified_start - 0.2, 2.5, 'Start', ha='right', va='bottom', fontsize=11, fontweight='bold')
    ax1.text(unified_end - 0.2, 2.5, 'End', ha='right', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_xlim(1938, 2026.5)
    ax1.set_ylim(-0.7, 2.8)
    ax1.set_yticks(y_pos_before)
    ax1.set_yticklabels([item[0] for item in before_ranges], fontsize=13, rotation=22, ha='right')
    ax1.set_ylabel('Dataset', fontsize=14, labelpad=12)
    ax1.yaxis.set_label_coords(-0.16, 0.5)
    ax1.set_xticks(np.arange(1940, 2027, 10))
    ax1.grid(True, axis='x', alpha=0.22)
    ax1.spines[['top', 'right']].set_visible(False)

    # (b) After alignment + split
    add_panel_label(ax2, '(b)', x=0.06, y=0.95)
    train_start, train_end = _ym_start(2004, 1), _ym_start(2018, 10)
    val_start, val_end = _ym_start(2018, 10), _ym_next(2024, 12)
    test_start, test_end = _ym_start(2025, 6), _ym_next(2025, 12)

    ax2.barh(2.1, train_end - train_start, left=train_start, color=split_colors['train'], height=0.62, alpha=0.98)
    ax2.barh(2.1, val_end - val_start, left=val_start, color=split_colors['validation'], height=0.62, alpha=0.98)
    ax2.barh(2.1, test_end - test_start, left=test_start, color=split_colors['test'], height=0.62, alpha=0.98)
    ax2.text((train_start + train_end) / 2, 2.72, 'Training (70%)', color='black', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax2.text((val_start + val_end) / 2, 2.72, 'Validation (30%)', color='black', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax2.text((test_start + test_end) / 2, 2.72, 'Test', color='black', ha='center', va='bottom', fontsize=12, fontweight='bold')

    availability = [
        ('ERA5', _ym_start(2004, 1), _ym_next(2024, 12), colors['ERA5'], 1.15),
        ('OISST', _ym_start(2004, 1), _ym_next(2025, 9), colors['OISST'], 0.80),
        ('Argo', _ym_start(2004, 1), _ym_next(2024, 12), colors['Argo'], 0.45),
    ]
    for source_name, start, end, color, y in availability:
        ax2.plot([start, end], [y, y], color=color, linewidth=4.5, solid_capstyle='round')

    ax2.set_xlim(2003.5, 2026.2)
    ax2.set_ylim(0.1, 3.1)
    ax2.set_yticks([2.1, 1.15, 0.80, 0.45])
    ax2.set_yticklabels(
        ['Split', 'ERA5 (Reanalysis)', 'OISST (Satellite)', 'Argo (In-situ)'],
        fontsize=13,
        rotation=22,
        ha='right',
    )
    ax2.set_ylabel('Dataset', fontsize=14, labelpad=12)
    ax2.yaxis.set_label_coords(-0.16, 0.5)
    ax2.set_xlabel('Year', fontsize=14, labelpad=10)
    ax2.set_xticks(np.arange(2004, 2027, 2))
    ax2.grid(True, axis='x', alpha=0.22)
    ax2.spines[['top', 'right']].set_visible(False)

    save_path = _create_output_path(filename, output_dir)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, save_path


def plot_regional_prediction_performance(
    pred_output,
    true_output,
    lon=(-180, 180),
    lat=(-80, 80),
    filename='regional_prediction_performance.png',
    output_dir=None,
):
    """
    Plot two-panel regional prediction performance figure:
    (a) global error map with annotated ocean regions;
    (b) lollipop RMSE ranking with R² and bias annotations.
    """
    from src.analysis.regional import OCEAN_REGIONS

    apply_plot_style()

    error = pred_output - true_output
    height, width = pred_output.shape
    lon_array = np.linspace(lon[0], lon[1], width, endpoint=False)
    lat_array = np.linspace(lat[0], lat[1], height)

    def _region_mask(region_key):
        info = OCEAN_REGIONS[region_key]
        rlon, rlat = info['lon'], info['lat']
        if rlon[0] == -180 and rlon[1] == 180:
            lm = np.ones(width, dtype=bool)
        elif rlon[0] > rlon[1]:
            lm = (lon_array >= rlon[0]) | (lon_array <= rlon[1])
        else:
            lm = (lon_array >= rlon[0]) & (lon_array <= rlon[1])
        return np.outer(
            (lat_array >= rlat[0]) & (lat_array <= rlat[1]),
            lm,
        )

    def _extract(data, region_key):
        vals = data[_region_mask(region_key)]
        return vals[~np.isnan(vals)]

    valid_stats = {}
    for rk, rinfo in OCEAN_REGIONS.items():
        pred_d = _extract(pred_output, rk)
        true_d = _extract(true_output, rk)
        error_d = _extract(error, rk)
        if pred_d.size == 0:
            continue
        corr = float(np.corrcoef(pred_d, true_d)[0, 1]) if pred_d.size > 1 else np.nan
        valid_stats[rk] = {
            'name': rinfo['name'],
            'rmse': float(np.sqrt(np.mean(error_d ** 2))),
            'bias': float(np.mean(error_d)),
            'corr': corr,
        }

    region_keys = list(valid_stats.keys())
    names = [valid_stats[k]['name'] for k in region_keys]
    rmse_arr = np.array([valid_stats[k]['rmse'] for k in region_keys])
    corr_arr = np.array([valid_stats[k]['corr'] for k in region_keys])
    bias_arr = np.array([valid_stats[k]['bias'] for k in region_keys])
    color_list = [OCEAN_REGIONS[k]['color'] for k in region_keys]
    sort_idx = np.argsort(rmse_arr)

    short_labels = {
        'nino34': 'N3.4', 'nino3': 'N3', 'warm_pool': 'WP',
        'gulf_stream': 'GS', 'kuroshio': 'KS', 'acc': 'ACC',
        'north_indian': 'NIO', 'north_atlantic_subpolar': 'NASP',
    }

    fig = plt.figure(figsize=(12, 10), facecolor='white')
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.3, 1], hspace=0.35)

    # ---- (a) Global error map with region annotations ----
    ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    ax_map.set_extent([lon[0], lon[1], lat[0], lat[1]], crs=ccrs.PlateCarree())
    ax_map.coastlines(resolution='110m', linewidth=0.4, color='#404040')
    ax_map.add_feature(cfeature.LAND, facecolor='#f0f0f0', edgecolor='none')

    gl = ax_map.gridlines(
        draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle='-',
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}

    lon_grid, lat_grid = np.meshgrid(lon_array, lat_array)
    abs_max = min(1.2, max(abs(np.nanmin(error)), abs(np.nanmax(error))))
    levels = np.linspace(-abs_max, abs_max, 25)
    error_cmap = LinearSegmentedColormap.from_list(
        'error',
        ['#2166ac', '#67a9cf', '#d1e5f0', '#f7f7f7', '#fddbc7', '#ef8a62', '#b2182b'],
        N=256,
    )

    cf = ax_map.contourf(
        lon_grid, lat_grid, error, levels=levels,
        cmap=error_cmap, extend='both', transform=ccrs.PlateCarree(), alpha=0.85,
    )

    for rk in region_keys:
        info = OCEAN_REGIONS[rk]
        rlon, rlat, color = info['lon'], info['lat'], info['color']
        if rlon[0] == -180 and rlon[1] == 180:
            bw, ls = 358, -179
        else:
            bw, ls = rlon[1] - rlon[0], rlon[0]
        bh = rlat[1] - rlat[0]

        ax_map.add_patch(mpatches.Rectangle(
            (ls, rlat[0]), bw, bh,
            linewidth=1.5, edgecolor=color, facecolor='none',
            transform=ccrs.PlateCarree(), zorder=10,
        ))
        cx, cy = ls + bw / 2, rlat[0] + bh / 2
        ax_map.text(
            cx, cy, short_labels.get(rk, ''),
            ha='center', va='center', fontsize=11, fontweight='bold',
            color='white', transform=ccrs.PlateCarree(), zorder=12,
            bbox={
                'boxstyle': 'round,pad=0.3',
                'facecolor': color,
                'edgecolor': 'white',
                'linewidth': 1.2,
                'alpha': 0.9,
            },
        )

    region_legend_handles = [
        mpatches.Patch(
            facecolor=OCEAN_REGIONS[rk]['color'],
            edgecolor=OCEAN_REGIONS[rk]['color'],
            label=f'{short_labels.get(rk, rk)}: {OCEAN_REGIONS[rk]["name"]}',
        )
        for rk in region_keys
    ]
    ax_map.legend(
        handles=region_legend_handles,
        loc='upper left',
        bbox_to_anchor=(0.0, -0.08),
        ncol=2,
        fontsize=8.5,
        framealpha=0.88,
        edgecolor='0.8',
        columnspacing=0.9,
        handlelength=1.1,
        handletextpad=0.4,
    )

    add_panel_label(ax_map, '(a)')

    # ---- (b) RMSE lollipop ranking ----
    ax_stats = fig.add_subplot(gs[1])

    sorted_names = [names[i] for i in sort_idx]
    sorted_rmse = rmse_arr[sort_idx]
    sorted_corr = corr_arr[sort_idx]
    sorted_bias = bias_arr[sort_idx]
    sorted_colors = [color_list[i] for i in sort_idx]

    n_regions = len(region_keys)
    global_rmse = float(np.sqrt(np.nanmean(error ** 2)))
    y_max = max(sorted_rmse) * 1.35

    for i, (rv, cv, bv, c) in enumerate(
        zip(sorted_rmse, sorted_corr, sorted_bias, sorted_colors),
    ):
        ax_stats.vlines(x=i, ymin=0, ymax=rv, color=c, linewidth=3, alpha=0.85)
        ax_stats.scatter(i, rv, s=220, color=c, edgecolor='white', linewidth=1.8, zorder=5)
        ax_stats.text(
            i, rv + 0.04, f'r={cv:.2f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='#333333',
        )

    ax_stats.axhline(
        y=global_rmse, color='#D32F2F', linestyle='--', linewidth=2.0, alpha=0.8, zorder=1,
    )
    ax_stats.text(
        0.98, global_rmse + 0.03, f'Global: {global_rmse:.3f}°C',
        ha='right', va='bottom', fontsize=12, color='#D32F2F', fontweight='bold',
        transform=ax_stats.get_yaxis_transform(),
    )

    ax_stats.set_xticks(np.arange(n_regions))
    ax_stats.set_xticklabels(sorted_names, rotation=35, ha='right')
    ax_stats.set_ylabel('RMSE (°C)', fontsize=14, labelpad=12)
    ax_stats.set_xlim(-0.6, n_regions - 0.4)
    ax_stats.set_ylim(-0.12, y_max)
    ax_stats.tick_params(labelsize=14, pad=8)

    ax_stats.spines['top'].set_visible(False)
    ax_stats.spines['right'].set_visible(False)
    ax_stats.spines['left'].set_linewidth(0.5)
    ax_stats.spines['bottom'].set_linewidth(0.5)
    ax_stats.yaxis.grid(True, linestyle='-', alpha=0.15, zorder=0)
    ax_stats.set_axisbelow(True)

    for i, bv in enumerate(sorted_bias):
        bc = '#C62828' if bv > 0 else '#1565C0'
        ax_stats.text(
            i, -0.06, f'{bv:+.2f}',
            ha='center', va='top', fontsize=12, color=bc, fontweight='medium',
        )

    legend_handles = [
        mpatches.Patch(facecolor='#C62828', label='Bias > 0 (warm)'),
        mpatches.Patch(facecolor='#1565C0', label='Bias < 0 (cold)'),
    ]
    ax_stats.legend(
        handles=legend_handles, loc='upper left',
        fontsize=11, framealpha=0.9, edgecolor='0.8',
    )

    add_panel_label(ax_stats, '(b)')

    fig.subplots_adjust(left=0.08, right=0.86, top=0.95, bottom=0.15, hspace=0.35)

    map_pos = ax_map.get_position()
    cbar_height = map_pos.height * 0.7
    cbar_bottom = map_pos.y0 + (map_pos.height - cbar_height) / 2
    err_cax = fig.add_axes([0.88, cbar_bottom, 0.015, cbar_height])
    err_cbar = fig.colorbar(cf, cax=err_cax, orientation='vertical')
    err_cbar.ax.tick_params(labelsize=14, pad=8)

    fig.text(
        0.02, (map_pos.y0 + map_pos.y1) / 2,
        'Prediction Error (°C)', fontsize=14,
        ha='center', va='center', rotation=90,
    )

    save_path = _create_output_path(filename, output_dir)
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig, save_path


_REGION_MAP_TARGET_ASPECT = 4.0


def _compute_padded_extent(sub_lon, sub_lat):
    """
    Pad the geographic extent so all region maps share the same display
    aspect ratio (_REGION_MAP_TARGET_ASPECT).  The pcolormesh data keeps its
    original extent; only the Cartopy view is enlarged with context area.
    """
    lon_span = float(sub_lon[-1] - sub_lon[0]) if len(sub_lon) > 1 else 1.0
    lat_span = float(sub_lat[-1] - sub_lat[0]) if len(sub_lat) > 1 else 1.0
    geo_aspect = lon_span / lat_span

    if geo_aspect > _REGION_MAP_TARGET_ASPECT:
        needed_lat = lon_span / _REGION_MAP_TARGET_ASPECT
        pad = (needed_lat - lat_span) / 2
        return [float(sub_lon[0]), float(sub_lon[-1]),
                max(-90.0, float(sub_lat[0]) - pad),
                min(90.0, float(sub_lat[-1]) + pad)]

    if geo_aspect < _REGION_MAP_TARGET_ASPECT:
        needed_lon = lat_span * _REGION_MAP_TARGET_ASPECT
        pad = (needed_lon - lon_span) / 2
        return [float(sub_lon[0]) - pad, float(sub_lon[-1]) + pad,
                float(sub_lat[0]), float(sub_lat[-1])]

    return [float(sub_lon[0]), float(sub_lon[-1]),
            float(sub_lat[0]), float(sub_lat[-1])]


def _setup_region_map_panel(ax, lon_grid, lat_grid, data, cmap, vmin, vmax,
                            show_left_labels=True, extent=None):
    """
    Configure a single Cartopy map panel for regional detail figures.
    """
    if extent is None:
        extent = [float(lon_grid.min()), float(lon_grid.max()),
                  float(lat_grid.min()), float(lat_grid.max())]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(resolution='110m', linewidth=0.4, color='#404040')
    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', edgecolor='none')

    im = ax.pcolormesh(
        lon_grid, lat_grid, data,
        cmap=cmap, vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree(), shading='auto',
    )

    gl = ax.gridlines(
        draw_labels=True, linewidth=0.25, color='gray', alpha=0.5, linestyle='-',
    )
    gl.top_labels = False
    gl.right_labels = False
    if not show_left_labels:
        gl.left_labels = False
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}

    return im


def plot_region_detail(
    pred_output,
    true_output,
    lon=(-180, 180),
    lat=(-80, 80),
    region_key='acc',
    region_info=None,
    filename=None,
    output_dir=None,
):
    """
    Plot 6-panel regional detail figure:
    top row  — (a) Observed SST, (b) Predicted SST, (c) Error map;
    bottom row — (d) scatter density, (e) error histogram + KDE, (f) metrics table.

    Accepts either *region_key* (looked up from OCEAN_REGIONS) or a custom
    *region_info* dict with at least ``{'name': str, 'lon': [lo, hi], 'lat': [lo, hi]}``.
    """
    from src.analysis.regional import OCEAN_REGIONS
    from scipy import stats as scipy_stats
    from matplotlib.colors import LogNorm

    apply_plot_style()

    if region_info is None:
        if region_key not in OCEAN_REGIONS:
            raise ValueError(f'Unknown region key: {region_key}')
        region_info = OCEAN_REGIONS[region_key]

    rlon, rlat = region_info['lon'], region_info['lat']

    h, w = pred_output.shape
    lon_array = np.linspace(lon[0], lon[1], w, endpoint=False)
    lat_array = np.linspace(lat[0], lat[1], h)
    error = pred_output - true_output

    # ---- Region mask ----
    if rlon[0] == -180 and rlon[1] == 180:
        lon_mask = np.ones(w, dtype=bool)
    elif rlon[0] > rlon[1]:
        lon_mask = (lon_array >= rlon[0]) | (lon_array <= rlon[1])
    else:
        lon_mask = (lon_array >= rlon[0]) & (lon_array <= rlon[1])
    lat_mask = (lat_array >= rlat[0]) & (lat_array <= rlat[1])
    mask_2d = np.outer(lat_mask, lon_mask)

    def _valid(data_arr):
        vals = data_arr[mask_2d]
        return vals[~np.isnan(vals)]

    pred_flat = _valid(pred_output)
    true_flat = _valid(true_output)
    error_flat = _valid(error)

    if pred_flat.size == 0:
        return None, None

    # ---- Statistics ----
    rmse_val = float(np.sqrt(np.mean(error_flat ** 2)))
    mae_val = float(np.mean(np.abs(error_flat)))
    bias_val = float(np.mean(error_flat))
    ss_res = float(np.sum(error_flat ** 2))
    ss_tot = float(np.sum((true_flat - np.mean(true_flat)) ** 2))
    r2_val = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # ---- 2-D sub-region extraction ----
    lon_idx = np.where(lon_mask)[0]
    lat_idx = np.where(lat_mask)[0]
    sub_pred = pred_output[lat_idx[0]:lat_idx[-1] + 1][:, lon_idx]
    sub_true = true_output[lat_idx[0]:lat_idx[-1] + 1][:, lon_idx]
    sub_error = error[lat_idx[0]:lat_idx[-1] + 1][:, lon_idx]

    sub_lon = lon_array[lon_idx]
    sub_lat = lat_array[lat_idx]
    lon_grid, lat_grid = np.meshgrid(sub_lon, sub_lat)

    vmin_sst = float(min(np.nanmin(sub_true), np.nanmin(sub_pred)))
    vmax_sst = float(max(np.nanmax(sub_true), np.nanmax(sub_pred)))
    err_lim = max(0.5, min(1.5, float(np.nanpercentile(np.abs(sub_error), 99))))

    error_cmap = LinearSegmentedColormap.from_list(
        'error_div',
        ['#2166ac', '#67a9cf', '#d1e5f0', '#f7f7f7', '#fddbc7', '#ef8a62', '#b2182b'],
        N=256,
    )

    # ---- Uniform map extent ----
    map_extent = _compute_padded_extent(sub_lon, sub_lat)

    # ---- Figure ----
    fig = plt.figure(figsize=(12, 14), facecolor='white')
    gs = GridSpec(
        3, 2, figure=fig, height_ratios=[0.8, 1.0, 1.0],
        hspace=0.42, wspace=0.28,
        left=0.08, right=0.95, top=0.96, bottom=0.06,
    )

    # (a) Observed SST
    ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    im_sst = _setup_region_map_panel(
        ax_a, lon_grid, lat_grid, sub_true,
        'RdYlBu_r', vmin_sst, vmax_sst, show_left_labels=True,
        extent=map_extent,
    )
    ax_a.text(
        0.5, 1.02, 'Observed', transform=ax_a.transAxes,
        ha='center', va='bottom', fontsize=14, fontweight='bold',
    )
    add_panel_label(ax_a, '(a)')

    # (b) Predicted SST
    ax_b = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    _setup_region_map_panel(
        ax_b, lon_grid, lat_grid, sub_pred,
        'RdYlBu_r', vmin_sst, vmax_sst, show_left_labels=False,
        extent=map_extent,
    )
    ax_b.text(
        0.5, 1.02, 'Predicted', transform=ax_b.transAxes,
        ha='center', va='bottom', fontsize=14, fontweight='bold',
    )
    add_panel_label(ax_b, '(b)')

    # (c) Error map
    ax_c = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    im_err = _setup_region_map_panel(
        ax_c, lon_grid, lat_grid, sub_error,
        error_cmap, -err_lim, err_lim, show_left_labels=True,
        extent=map_extent,
    )
    ax_c.text(
        0.5, 1.02, f'Error (RMSE={rmse_val:.3f}\u00b0C)', transform=ax_c.transAxes,
        ha='center', va='bottom', fontsize=14, fontweight='bold',
    )
    add_panel_label(ax_c, '(c)')

    # (d) Scatter density
    ax_d = fig.add_subplot(gs[1, 1])
    hist_result = ax_d.hist2d(
        true_flat, pred_flat, bins=50, cmap='YlGnBu',
        norm=LogNorm(), cmin=1,
    )
    val_lims = [
        min(float(true_flat.min()), float(pred_flat.min())),
        max(float(true_flat.max()), float(pred_flat.max())),
    ]
    ax_d.plot(val_lims, val_lims, 'k--', linewidth=1.0, label='1:1')
    coef = np.polyfit(true_flat, pred_flat, 1)
    ax_d.plot(
        val_lims, np.poly1d(coef)(val_lims),
        color='#E74C3C', linewidth=1.0,
        label=f'Fit (y={coef[0]:.2f}x{coef[1]:+.2f})',
    )
    ax_d.set_xlabel('Observed (\u00b0C)', fontsize=14, labelpad=12)
    ax_d.set_ylabel('Predicted (\u00b0C)', fontsize=14, labelpad=12)
    ax_d.set_aspect('equal', adjustable='box')
    ax_d.legend(loc='lower right', fontsize=11, framealpha=0.9, handlelength=1.5)
    ax_d.text(
        0.03, 0.97, f'R\u00b2={r2_val:.3f}\nN={pred_flat.size:,}',
        transform=ax_d.transAxes, fontsize=12, va='top',
        bbox={'boxstyle': 'round,pad=0.2', 'facecolor': 'white',
              'alpha': 0.8, 'linewidth': 0.3},
    )
    add_panel_label(ax_d, '(d)')

    # (e) Error histogram + KDE
    ax_e = fig.add_subplot(gs[2, 0])
    ax_e.hist(
        error_flat, bins=50, density=True,
        color='#3498DB', alpha=0.6, edgecolor='white', linewidth=0.3,
    )
    kde = scipy_stats.gaussian_kde(error_flat)
    x_kde = np.linspace(float(error_flat.min()), float(error_flat.max()), 200)
    ax_e.plot(x_kde, kde(x_kde), color='#2C3E50', linewidth=1.0, label='KDE')
    ax_e.axvline(0, color='#E74C3C', linestyle='-', linewidth=0.75, label='Zero')
    ax_e.axvline(
        bias_val, color='#27AE60', linestyle='--', linewidth=0.75,
        label=f'Bias={bias_val:+.3f}',
    )
    ax_e.set_xlabel('Error (\u00b0C)', fontsize=14, labelpad=12)
    ax_e.set_ylabel('Density', fontsize=14, labelpad=12)
    ax_e.legend(loc='upper right', fontsize=11, framealpha=0.9, handlelength=1.2)
    add_panel_label(ax_e, '(e)')

    # (f) Metrics table
    ax_f = fig.add_subplot(gs[2, 1])
    ax_f.axis('off')
    metrics = [
        ('RMSE', f'{rmse_val:.4f}', '\u00b0C'),
        ('MAE', f'{mae_val:.4f}', '\u00b0C'),
        ('Bias', f'{bias_val:+.4f}', '\u00b0C'),
        ('R\u00b2', f'{r2_val:.4f}', ''),
        ('\u03c3_obs', f'{np.nanstd(true_flat):.3f}', '\u00b0C'),
        ('\u03c3_pred', f'{np.nanstd(pred_flat):.3f}', '\u00b0C'),
    ]
    y_start = 0.85
    for i, (name, val, unit) in enumerate(metrics):
        y_pos = y_start - i * 0.13
        ax_f.text(
            0.05, y_pos, name, fontsize=16, fontweight='bold',
            transform=ax_f.transAxes, va='center',
        )
        ax_f.text(
            0.50, y_pos, val, fontsize=16, transform=ax_f.transAxes,
            va='center', ha='right', family='monospace',
        )
        ax_f.text(
            0.53, y_pos, unit, fontsize=14, transform=ax_f.transAxes,
            va='center', color='#666666',
        )

    ax_f.plot(
        [0.02, 0.58], [0.92, 0.92], color='#CCCCCC', linewidth=0.5,
        transform=ax_f.transAxes, clip_on=False,
    )
    ax_f.plot(
        [0.02, 0.58], [0.10, 0.10], color='#CCCCCC', linewidth=0.5,
        transform=ax_f.transAxes, clip_on=False,
    )
    ax_f.text(
        0.05, 0.04, region_info['name'], fontsize=14, fontweight='bold',
        transform=ax_f.transAxes,
    )
    lon_label = f'Lon {rlon[0]}\u00b0 ~ {rlon[1]}\u00b0'
    lat_label = f'Lat {rlat[0]}\u00b0 ~ {rlat[1]}\u00b0'
    ax_f.text(
        0.05, -0.04, f'{lon_label},  {lat_label}',
        fontsize=12, color='#666666', transform=ax_f.transAxes,
    )
    add_panel_label(ax_f, '(f)')

    # ---- Colorbars ----
    fig.canvas.draw()

    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    pos_c = ax_c.get_position()

    cax_sst = fig.add_axes([pos_a.x0, pos_a.y0 - 0.035, pos_b.x1 - pos_a.x0, 0.010])
    cb_sst = fig.colorbar(im_sst, cax=cax_sst, orientation='horizontal')
    cb_sst.set_label('SST (\u00b0C)', fontsize=14, labelpad=8)
    cb_sst.ax.tick_params(labelsize=14, pad=4)

    cax_err = fig.add_axes([pos_c.x0, pos_c.y0 - 0.035, pos_c.width, 0.010])
    cb_err = fig.colorbar(im_err, cax=cax_err, orientation='horizontal', extend='both')
    cb_err.set_label('Error (\u00b0C)', fontsize=14, labelpad=8)
    cb_err.ax.tick_params(labelsize=14, pad=4)

    pos_d = ax_d.get_position()
    cax_den = fig.add_axes([
        pos_d.x1 + 0.006, pos_d.y0 + 0.02,
        0.008, pos_d.height * 0.8,
    ])
    cb_den = fig.colorbar(hist_result[3], cax=cax_den, orientation='vertical')
    cb_den.set_label('Density', fontsize=12, labelpad=8)
    cb_den.ax.tick_params(labelsize=11, pad=4)

    # ---- Save ----
    if filename is None:
        filename = f'region_detail_{region_key}.png'
    save_path = _create_output_path(filename, output_dir)
    fig.savefig(save_path, dpi=300, facecolor='white')

    return fig, save_path


def plot_all_region_details(
    pred_output,
    true_output,
    lon=(-180, 180),
    lat=(-80, 80),
    output_dir=None,
):
    """
    Generate region detail figures for every ocean region in OCEAN_REGIONS.
    Returns a list of (region_key, save_path) tuples.
    """
    from src.analysis.regional import OCEAN_REGIONS

    results = []
    for region_key in OCEAN_REGIONS:
        try:
            fig, save_path = plot_region_detail(
                pred_output, true_output, lon, lat,
                region_key=region_key, output_dir=output_dir,
            )
            if fig is not None:
                results.append((region_key, save_path))
                plt.close(fig)
        except Exception as exc:
            print(f'[WARN] Region {region_key} skipped: {exc}')
    return results


def run_redraw_tasks(task='all', output_dir='out', month_index=0, resolution=1):
    """
    Execute redraw tasks programmatically and return exported file paths.
    """
    exported_paths = []

    if task in ('all', 'map'):
        fig_map, path_map = plot_era5_oisst_argo_raw_processed(
            output_dir=output_dir,
            month_index=month_index,
            resolution=resolution,
        )
        exported_paths.append(path_map)
        plt.close(fig_map)

    if task in ('all', 'season'):
        (fig_a, fig_b), (path_a, path_b) = plot_seasonal_analysis_figures(
            output_dir=output_dir,
            resolution=resolution,
        )
        exported_paths.extend([path_a, path_b])
        plt.close(fig_a)
        plt.close(fig_b)

    if task in ('all', 'timeline'):
        fig_timeline, path_timeline = plot_temporal_alignment_timeline(
            output_dir=output_dir,
        )
        exported_paths.append(path_timeline)
        plt.close(fig_timeline)

    if task in ('all', 'multi_source'):
        fig_multi_source, path_multi_source = plot_multi_source_sst_global_202401(
            output_dir=output_dir,
            resolution=resolution,
        )
        exported_paths.append(path_multi_source)
        plt.close(fig_multi_source)

    if task in ('all', 'multi_source_diff'):
        fig_multi_source_diff, path_multi_source_diff = plot_multi_source_sst_difference_202401(
            output_dir=output_dir,
            resolution=resolution,
        )
        exported_paths.append(path_multi_source_diff)
        plt.close(fig_multi_source_diff)

    if task in ('multi_source_consistency',):
        fig_multi_source_consistency, path_multi_source_consistency = plot_multi_source_sst_consistency_202401(
            output_dir=output_dir,
            resolution=resolution,
        )
        exported_paths.append(path_multi_source_consistency)
        plt.close(fig_multi_source_consistency)

    return exported_paths


def main():
    parser = argparse.ArgumentParser(description='Unified redraw entrypoint for plot regeneration.')
    parser.add_argument(
        '--task',
        choices=['all', 'map', 'season', 'timeline', 'multi_source', 'multi_source_diff', 'multi_source_consistency'],
        default='all',
        help='Select which redraw task to run.',
    )
    parser.add_argument(
        '--output-dir',
        default='out',
        help='Output directory for exported figures.',
    )
    parser.add_argument(
        '--month-index',
        type=int,
        default=0,
        help='Month index used by the map-comparison plot.',
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=1,
        help='Spatial resolution (degree) for data sampling.',
    )
    args = parser.parse_args()

    exported_paths = run_redraw_tasks(
        task=args.task,
        output_dir=args.output_dir,
        month_index=args.month_index,
        resolution=args.resolution,
    )

    print('Redraw completed. Exported files:')
    for file_path in exported_paths:
        print(f'- {file_path}')


if __name__ == '__main__':
    main()
