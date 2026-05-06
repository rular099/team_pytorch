#!/usr/bin/env python3
"""Generate figures and tables for the PGA academic report.

The script is intentionally file-based: it can run on a login node or a compute
node after experiments have produced eval_results.txt and, preferably,
eval_results.npz. Text outputs are sufficient for summary tables and station
count buckets; NPZ outputs are used for scatter/residual/binning figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:
    ccrs = None
    cfeature = None


MODEL_LABELS = {
    "b0_baseline_noamp": "Baseline",
    "b1_single_pga08_noamp": "PGA-weighted pretrain",
    "b2_single_pga_only_noamp": "PGA-only pretrain",
    "b3_pga_norm_noamp": "Target normalization",
    "b4_mse_noamp": "MSE loss",
    "b5_huber3_noamp": "Huber delta=3",
    "b6_station_decor_1em4_noamp": "Station decor.",
    "b7_relative_geometry_noamp": "Relative geometry",
}

DEFAULT_PROJECT_LABEL = "Proposed model"

BUCKET_ORDER = ["1", "2-3", "4-5", "6-10", "11-15", "16+"]
DEFAULT_POS_OFFSET = (37.0, 140.0)

JAPAN_REFERENCE_COASTLINES = [
    # Very lightweight lon/lat reference outlines for report maps. Cartopy and
    # geopandas are not guaranteed on the HPC nodes used for this project.
    [(141.0, 45.5), (143.2, 44.7), (145.4, 43.3), (144.2, 42.1), (141.8, 41.4), (140.2, 42.2), (140.5, 44.2), (141.0, 45.5)],
    [(140.8, 41.2), (141.5, 39.5), (141.3, 38.2), (140.8, 37.2), (140.1, 36.1), (139.7, 35.5), (138.8, 34.9), (137.0, 35.2), (136.0, 35.6), (135.0, 35.4), (134.2, 34.6), (133.0, 34.4), (132.0, 34.2), (131.1, 34.4), (130.9, 35.0), (132.5, 35.8), (134.0, 36.2), (135.5, 36.9), (137.0, 37.5), (138.5, 38.6), (139.7, 40.0), (140.8, 41.2)],
    [(133.0, 34.4), (134.3, 34.3), (135.0, 33.9), (134.1, 33.3), (132.8, 33.4), (132.4, 33.9), (133.0, 34.4)],
    [(130.0, 33.9), (131.2, 33.6), (131.9, 32.7), (131.2, 31.5), (130.2, 31.1), (129.6, 32.1), (129.7, 33.0), (130.0, 33.9)],
    [(129.5, 28.5), (128.2, 27.4), (127.7, 26.4), (126.8, 26.0), (125.5, 24.8)],
    [(126.0, 34.8), (127.8, 35.2), (129.3, 35.8), (129.6, 34.8), (128.2, 34.2), (126.0, 34.8)],
]


def cartopy_maps_enabled() -> bool:
    return ccrs is not None and cfeature is not None and os.environ.get("PGA_REPORT_DISABLE_CARTOPY", "0") != "1"


def cartopy_features_enabled() -> bool:
    return cartopy_maps_enabled() and os.environ.get("PGA_REPORT_CARTOPY_FEATURES", "0") == "1"


def map_subplot_kw() -> dict:
    if cartopy_maps_enabled():
        return {"projection": ccrs.PlateCarree()}
    return {}


def is_cartopy_axis(ax) -> bool:
    return cartopy_maps_enabled() and hasattr(ax, "projection") and hasattr(ax, "set_extent")


def replace_with_map_axis(fig, ax):
    if not cartopy_maps_enabled():
        return ax
    spec = ax.get_subplotspec()
    ax.remove()
    return fig.add_subplot(spec, projection=ccrs.PlateCarree())


@dataclass
class EvalFile:
    result_dir: Path
    txt_path: Path
    npz_path: Path | None
    model_id: str
    checkpoint_tag: str


def model_id_from_dir(path: Path) -> str:
    name = path.name
    for prefix in (
        "weights_japan_overfit_pga15_stage1_",
        "weights_japan_full_pga15_",
        "weights_",
    ):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def display_model_label(model_id: str, compact: bool = True) -> str:
    if model_id in MODEL_LABELS:
        return MODEL_LABELS[model_id]
    if compact:
        return DEFAULT_PROJECT_LABEL
    return model_id


def checkpoint_tag_from_txt(txt: str, fallback: str) -> str:
    m = re.search(r"Loaded .*?/(full_model_[^/\s]+\.pth)", txt)
    if not m:
        return fallback
    stem = Path(m.group(1)).stem
    return stem.replace("full_model_", "")


def discover_eval_files(results_root: Path, pattern: str) -> list[EvalFile]:
    evals: list[EvalFile] = []
    for result_dir in sorted(results_root.glob(pattern)):
        if not result_dir.is_dir():
            continue
        model_id = model_id_from_dir(result_dir)
        for txt_path in sorted(result_dir.glob("eval_results*.txt")):
            txt = txt_path.read_text(errors="replace")
            suffix = txt_path.stem.replace("eval_results", "").strip("_")
            checkpoint_tag = checkpoint_tag_from_txt(txt, suffix or "last")
            npz_candidate = result_dir / f"{txt_path.stem}.npz"
            if not npz_candidate.exists() and txt_path.name == "eval_results.txt":
                npz_candidate = result_dir / "eval_results.npz"
            evals.append(
                EvalFile(
                    result_dir=result_dir,
                    txt_path=txt_path,
                    npz_path=npz_candidate if npz_candidate.exists() else None,
                    model_id=model_id,
                    checkpoint_tag=checkpoint_tag,
                )
            )
    return evals


def parse_eval_txt(eval_file: EvalFile) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    text = eval_file.txt_path.read_text(errors="replace")
    if "Traceback" in text:
        raise RuntimeError(f"Traceback found in {eval_file.txt_path}")

    rows: list[dict] = []
    bucket_rows: list[dict] = []
    single_rows: list[dict] = []
    diag_rows: list[dict] = []

    cos = re.search(r"Cosine similarity \(off-diag\): min=[^,]+, max=[^,]+, mean=([0-9.]+)", text)
    if cos:
        diag_rows.append({
            "model_id": eval_file.model_id,
            "checkpoint": eval_file.checkpoint_tag,
            "metric": "station_feature_cosine_mean",
            "value": float(cos.group(1)),
        })

    parts = re.split(r"(?:^|\n)={60}\n\s+(TRAIN|VAL) set:", text)
    for i in range(1, len(parts), 2):
        split = parts[i].lower()
        section = parts[i + 1]
        pga = re.search(
            r"--- pga ---.*?"
            r"MAE=([0-9.]+), RMSE=([0-9.]+).*?"
            r"Correlation: ([0-9.nan-]+).*?"
            r"R\^2: ([0-9.-]+).*?"
            r"Linear fit: pred = ([0-9.-]+) \* label \+ ([0-9.-]+)",
            section,
            re.S,
        )
        if pga:
            mae, rmse, corr, r2, slope, intercept = pga.groups()
            rows.append({
                "model_id": eval_file.model_id,
                "model": display_model_label(eval_file.model_id),
                "checkpoint": eval_file.checkpoint_tag,
                "split": split,
                "mae": float(mae),
                "rmse": float(rmse),
                "corr": float(corr) if corr != "nan" else np.nan,
                "r2": float(r2),
                "slope": float(slope),
                "intercept": float(intercept),
            })

            pga_section = section.split("--- pga ---", 1)[1]
            if "SINGLE-STATION" in pga_section:
                pga_section = pga_section.split("SINGLE-STATION", 1)[0]
            for line in pga_section.splitlines():
                m = re.search(
                    r"n=([^:]+): events=([0-9]+), targets=([0-9]+), "
                    r"MAE=([0-9.]+), RMSE=([0-9.]+), corr=([0-9.nan-]+)",
                    line,
                )
                if m:
                    bucket, events, targets, b_mae, b_rmse, b_corr = m.groups()
                    bucket_rows.append({
                        "model_id": eval_file.model_id,
                        "model": display_model_label(eval_file.model_id),
                        "checkpoint": eval_file.checkpoint_tag,
                        "split": split,
                        "bucket": bucket,
                        "events": int(events),
                        "targets": int(targets),
                        "mae": float(b_mae),
                        "rmse": float(b_rmse),
                        "corr": float(b_corr) if b_corr != "nan" else np.nan,
                    })

    for split_name in ("TRAIN", "VAL"):
        marker = f"SINGLE-STATION {split_name} set:"
        if marker not in text:
            continue
        section = text.split(marker, 1)[1]
        if split_name == "TRAIN" and "VAL set:" in section:
            section = section.split("VAL set:", 1)[0]
        m = re.search(
            r"--- single/pga ---.*?"
            r"MAE=([0-9.]+), RMSE=([0-9.]+).*?"
            r"Correlation: ([0-9.]+).*?"
            r"norm mean=([0-9.]+), std=([0-9.]+)",
            section,
            re.S,
        )
        if m:
            mae, rmse, corr, emb_mean, emb_std = m.groups()
            single_rows.append({
                "model_id": eval_file.model_id,
                "model": display_model_label(eval_file.model_id),
                "checkpoint": eval_file.checkpoint_tag,
                "split": split_name.lower(),
                "single_pga_mae": float(mae),
                "single_pga_rmse": float(rmse),
                "single_pga_corr": float(corr),
                "embedding_norm_mean": float(emb_mean),
                "embedding_norm_std": float(emb_std),
            })

    return rows, bucket_rows, single_rows, diag_rows


def npz_stack(values: np.ndarray) -> np.ndarray:
    if values.dtype == object:
        return np.stack([np.asarray(v) for v in values])
    return np.asarray(values)


def npz_optional_stack(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data:
        return None
    return npz_stack(data[key])


def get_pga_arrays(npz_path: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    data = np.load(npz_path, allow_pickle=True)
    labels = npz_stack(data[f"{split}_pga_label"])
    preds = npz_stack(data[f"{split}_pga_mu_best"])
    valid = npz_stack(data[f"{split}_pga_target_valid"]).astype(bool)
    labels = labels.reshape(labels.shape[0], -1)
    preds = preds.reshape(preds.shape[0], -1)
    valid = valid.reshape(valid.shape[0], -1)
    counts = None
    key = f"{split}_station_valid_count"
    if key in data:
        counts = np.asarray(data[key], dtype=np.int64).reshape(-1)
    return labels, preds, valid, counts


def flatten_valid(labels: np.ndarray, preds: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return labels[valid].astype(float), preds[valid].astype(float)


def horizontal_distances_km(coords: np.ndarray, event_coord: np.ndarray) -> np.ndarray:
    """Approximate epicentral distance from horizontal coordinate differences."""
    coords = np.asarray(coords, dtype=float)
    event_coord = np.asarray(event_coord, dtype=float).reshape(-1)
    if coords.ndim != 2 or coords.shape[-1] < 2 or event_coord.size < 2:
        return np.arange(coords.shape[0], dtype=float)
    horizontal_delta = coords[:, :2] - event_coord[None, :2]
    # Current eval coordinates are unscaled degree-like coordinates. Distances
    # should be horizontal epicentral distances, not 3-D hypocentral distances.
    return np.linalg.norm(horizontal_delta, axis=-1) * 111.19


def set_map_limits(ax, point_arrays: list[np.ndarray], pad_fraction: float = 0.08) -> None:
    points = []
    for arr in point_arrays:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1 and arr.size >= 2:
            points.append(arr[None, :2])
        elif arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] > 0:
            points.append(arr[:, :2])
    if not points:
        return
    all_points = np.concatenate(points, axis=0)
    mins = np.nanmin(all_points, axis=0)
    maxs = np.nanmax(all_points, axis=0)
    span = np.maximum(maxs - mins, 1e-3)
    pad = np.maximum(span.max() * pad_fraction, 0.02)
    ax.set_xlim(mins[0] - pad, maxs[0] + pad)
    ax.set_ylim(mins[1] - pad, maxs[1] + pad)
    ax.set_aspect("equal", adjustable="box")


def coords_to_lonlat(coords: np.ndarray, pos_offset: tuple[float, float] = DEFAULT_POS_OFFSET) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords2 = coords.reshape(1, -1)
        squeeze = True
    else:
        coords2 = coords
        squeeze = False
    if coords2.shape[-1] < 2:
        out = coords2
    else:
        lat = coords2[:, 0] + pos_offset[0]
        lon = coords2[:, 1] + pos_offset[1]
        out = np.column_stack([lon, lat])
    return out[0] if squeeze else out


def map_extent_from_lonlat_arrays(lonlat_arrays: list[np.ndarray]) -> list[float]:
    points = []
    for arr in lonlat_arrays:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1 and arr.size >= 2:
            points.append(arr[None, :2])
        elif arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] > 0:
            points.append(arr[:, :2])
    if points:
        all_points = np.concatenate(points, axis=0)
        mins = np.nanmin(all_points, axis=0)
        maxs = np.nanmax(all_points, axis=0)
        span = np.maximum(maxs - mins, 0.5)
        pad = np.maximum(span.max() * 0.25, 1.0)
        extent = [
            max(120.0, mins[0] - pad),
            min(148.0, maxs[0] + pad),
            max(23.0, mins[1] - pad),
            min(47.0, maxs[1] + pad),
        ]
    else:
        extent = [122.0, 147.0, 24.0, 46.0]
    return extent


def draw_reference_map(ax, lonlat_arrays: list[np.ndarray]) -> list[float]:
    extent = map_extent_from_lonlat_arrays(lonlat_arrays)
    if is_cartopy_axis(ax):
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_facecolor("#eaf4fb")
        if cartopy_features_enabled():
            # Requires Natural Earth data to be available in Cartopy's data dir.
            # Keep disabled by default because many HPC nodes cannot download
            # these assets at render time.
            ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#eaf4fb", zorder=0)
            ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f2efe6", edgecolor="none", zorder=0)
            ax.coastlines(resolution="50m", color="#555555", linewidth=1.2, zorder=1)
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#777777", linewidth=0.8, zorder=1)
        else:
            for coastline in JAPAN_REFERENCE_COASTLINES:
                poly = np.asarray(coastline, dtype=float)
                ax.fill(
                    poly[:, 0],
                    poly[:, 1],
                    color="#f2efe6",
                    alpha=0.85,
                    zorder=0,
                    transform=ccrs.PlateCarree(),
                )
                ax.plot(
                    poly[:, 0],
                    poly[:, 1],
                    color="#6f6f6f",
                    linewidth=1.2,
                    zorder=1,
                    transform=ccrs.PlateCarree(),
                )
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.7,
            color="white",
            alpha=0.65,
            linestyle="-",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 12}
        gl.ylabel_style = {"size": 12}
        return extent

    ax.set_facecolor("#eaf4fb")
    for coastline in JAPAN_REFERENCE_COASTLINES:
        poly = np.asarray(coastline, dtype=float)
        ax.fill(poly[:, 0], poly[:, 1], color="#f2efe6", alpha=0.85, zorder=0)
        ax.plot(poly[:, 0], poly[:, 1], color="#6f6f6f", linewidth=1.2, zorder=1)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    return extent


def draw_japan_inset(ax, focus_extent: list[float]) -> None:
    # Anchor the context map to the upper-right corner so its top/right borders
    # coincide with the zoomed map frame instead of floating over the data.
    inset_bounds = [0.68, 0.68, 0.32, 0.32]
    if is_cartopy_axis(ax):
        inset = ax.inset_axes(inset_bounds, projection=ccrs.PlateCarree())
    else:
        inset = ax.inset_axes(inset_bounds)
    inset.set_facecolor("#eaf4fb")
    if is_cartopy_axis(inset) and cartopy_features_enabled():
        inset.set_extent([123, 147, 24, 46], crs=ccrs.PlateCarree())
        inset.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#eaf4fb", zorder=0)
        inset.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f2efe6", edgecolor="none", zorder=0)
        inset.coastlines(resolution="50m", color="#555555", linewidth=0.9, zorder=1)
        inset.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#777777", linewidth=0.55, zorder=1)
        transform = ccrs.PlateCarree()
    else:
        for coastline in JAPAN_REFERENCE_COASTLINES:
            poly = np.asarray(coastline, dtype=float)
            plot_kwargs = {"transform": ccrs.PlateCarree()} if is_cartopy_axis(inset) else {}
            inset.fill(poly[:, 0], poly[:, 1], color="#f2efe6", alpha=0.9, zorder=0, **plot_kwargs)
            inset.plot(poly[:, 0], poly[:, 1], color="#6f6f6f", linewidth=0.8, zorder=1, **plot_kwargs)
        if is_cartopy_axis(inset):
            inset.set_extent([123, 147, 24, 46], crs=ccrs.PlateCarree())
        else:
            inset.set_xlim(123, 147)
            inset.set_ylim(24, 46)
        transform = ccrs.PlateCarree() if is_cartopy_axis(inset) else inset.transData
    rect = Rectangle(
        (focus_extent[0], focus_extent[2]),
        focus_extent[1] - focus_extent[0],
        focus_extent[3] - focus_extent[2],
        fill=False,
        edgecolor="#d62728",
        linewidth=2.2,
        zorder=5,
        transform=transform,
    )
    inset.add_patch(rect)
    inset.set_aspect("auto")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor("#444444")


def draw_spatial_residual_map(
    ax,
    target_xy: np.ndarray,
    residual: np.ndarray,
    event_coord: np.ndarray,
    station_xy: np.ndarray | None = None,
    title: str = "Spatial Residual Map",
    vlim: float | None = None,
):
    target_lonlat = coords_to_lonlat(target_xy)
    residual = np.asarray(residual, dtype=float)
    event_lonlat = coords_to_lonlat(np.asarray(event_coord, dtype=float).reshape(-1))
    if vlim is None:
        vlim = max(float(np.nanmax(np.abs(residual))) if residual.size else 0.1, 0.1)
    station_lonlat = None if station_xy is None else coords_to_lonlat(station_xy)

    focus_extent = draw_reference_map(
        ax,
        [target_lonlat, event_lonlat, station_lonlat if station_lonlat is not None else np.empty((0, 2))],
    )
    draw_japan_inset(ax, focus_extent)
    geo_kwargs = {"transform": ccrs.PlateCarree()} if is_cartopy_axis(ax) else {}
    sc = ax.scatter(
        target_lonlat[:, 0],
        target_lonlat[:, 1],
        c=residual,
        s=90,
        cmap="coolwarm",
        vmin=-vlim,
        vmax=vlim,
        edgecolor="black",
        linewidth=0.4,
        label="PGA targets",
        zorder=3,
        **geo_kwargs,
    )
    ax.scatter([event_lonlat[0]], [event_lonlat[1]], marker="*", s=240, color="gold", edgecolor="black", label="Event", zorder=5, **geo_kwargs)
    if station_lonlat is not None and station_lonlat.ndim == 2 and station_lonlat.shape[0] > 0:
        ax.scatter(
            station_lonlat[:, 0],
            station_lonlat[:, 1],
            marker="^",
            s=155,
            color="#4daf4a",
            edgecolor="black",
            linewidth=1.1,
            label="Input stations",
            zorder=7,
            **geo_kwargs,
        )
    ax.set_xlabel("Longitude", fontsize=17)
    ax.set_ylabel("Latitude", fontsize=17)
    ax.set_title(title, fontsize=20, weight="bold")
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.25, color="white", linewidth=1.0)
    return sc


def get_distance_arrays(npz_path: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    data = np.load(npz_path, allow_pickle=True)
    required = [f"{split}_pga_target_abs", f"{split}_loc_label_abs", f"{split}_pga_label", f"{split}_pga_mu_best", f"{split}_pga_target_valid"]
    if any(key not in data for key in required):
        return None
    target_coords = npz_stack(data[f"{split}_pga_target_abs"]).astype(float)
    event_coords = npz_stack(data[f"{split}_loc_label_abs"]).astype(float)
    labels = npz_stack(data[f"{split}_pga_label"]).reshape(target_coords.shape[0], -1).astype(float)
    preds = npz_stack(data[f"{split}_pga_mu_best"]).reshape(target_coords.shape[0], -1).astype(float)
    valid = npz_stack(data[f"{split}_pga_target_valid"]).reshape(target_coords.shape[0], -1).astype(bool)
    if target_coords.ndim != 3 or target_coords.shape[-1] < 2:
        return None
    if event_coords.ndim == 3:
        event_coords = event_coords.reshape(event_coords.shape[0], -1, event_coords.shape[-1])[:, 0, :]
    event_coords = event_coords[:, : target_coords.shape[-1]]
    distances = np.stack([
        horizontal_distances_km(target_coords[i], event_coords[i])
        for i in range(target_coords.shape[0])
    ])
    return distances, labels, preds, valid


def parse_distance_bins(spec: str) -> list[float]:
    bins: list[float] = []
    for item in spec.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item in {"inf", "infinity"}:
            bins.append(float("inf"))
        else:
            bins.append(float(item))
    if len(bins) < 2:
        raise ValueError("--distance-bins must contain at least two edges")
    return bins


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_df(df: pd.DataFrame, path_stem: Path) -> None:
    df.to_csv(path_stem.with_suffix(".csv"), index=False)
    path_stem.with_suffix(".md").write_text(df.to_markdown(index=False) + "\n")


def readable_model_order(metrics: pd.DataFrame) -> list[str]:
    val = metrics[metrics["split"] == "val"].copy()
    if val.empty:
        return sorted(metrics["model_id"].unique())
    val = val.sort_values(["mae", "model_id"])
    return list(dict.fromkeys(val["model_id"].tolist()))


def preferred_checkpoint_rows(df: pd.DataFrame, preferred: str = "best") -> pd.DataFrame:
    """Keep one checkpoint per model/split for presentation plots."""
    if df.empty or "checkpoint" not in df.columns:
        return df
    group_cols = [col for col in ("model_id", "split") if col in df.columns]
    if not group_cols:
        return df
    frames = []
    for _, group in df.groupby(group_cols, sort=False):
        preferred_rows = group[group["checkpoint"] == preferred]
        frames.append(preferred_rows if not preferred_rows.empty else group.iloc[[0]])
    return pd.concat(frames, ignore_index=True)


def table_png(df: pd.DataFrame, out: Path, title: str, font_size: int = 16) -> None:
    fig_w = max(12, 1.9 * len(df.columns))
    fig_h = max(3.5, 0.8 * len(df) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=font_size + 4, weight="bold", pad=16)
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        else:
            display[col] = display[col].map(
                lambda x: "\n".join(textwrap.wrap(str(x), width=28)) if pd.notna(x) else ""
            )
    table = ax.table(
        cellText=display.values,
        colLabels=["\n".join(textwrap.wrap(str(col), width=14)) for col in display.columns],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.9)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#dfe8f3")
            cell.set_height(cell.get_height() * 1.7)
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f6f7f9")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def split_label(value: str) -> str:
    mapping = {"train": "Train", "dev": "Validation", "val": "Validation", "test": "Test"}
    return mapping.get(str(value).lower(), str(value))


def split_order_values(values: Iterable[str]) -> list[str]:
    order = ["train", "dev", "val", "test"]
    present = list(dict.fromkeys(str(v) for v in values))
    return [v for v in order if v in present] + sorted(v for v in present if v not in order)


def plot_split_data_distributions(split_dir: Path, out_dir: Path) -> None:
    """Plot event/station distributions from split_events.csv and split_stations.csv."""
    events_path = split_dir / "split_events.csv"
    stations_path = split_dir / "split_stations.csv"
    if not events_path.exists() or not stations_path.exists():
        print(f"[WARN] split_events.csv/split_stations.csv not found under {split_dir}; skip data distribution plots.", file=sys.stderr)
        return

    events = pd.read_csv(events_path)
    stations = pd.read_csv(stations_path)
    if "split" not in events.columns or "split" not in stations.columns:
        print(f"[WARN] split CSVs under {split_dir} do not contain split column; skip data distribution plots.", file=sys.stderr)
        return

    event_splits = split_order_values(events["split"].dropna().astype(str))
    station_splits = split_order_values(stations["split"].dropna().astype(str))
    colors = {
        "train": "#4daf4a",
        "dev": "#377eb8",
        "val": "#377eb8",
        "test": "#984ea3",
    }

    pga_col = "pga_norm_resampled_mps2" if "pga_norm_resampled_mps2" in stations.columns else "pga_norm_native_mps2"
    summary_rows = []
    for split in station_splits:
        ev = events[events["split"].astype(str) == split]
        st = stations[stations["split"].astype(str) == split]
        summary_rows.append({
            "Split": split_label(split),
            "Events": int(len(ev)),
            "Station records": int(len(st)),
            "Magnitude median": float(ev["Magnitude"].median()) if "Magnitude" in ev else np.nan,
            "Magnitude max": float(ev["Magnitude"].max()) if "Magnitude" in ev else np.nan,
            "Epicentral dist. median km": float(st["epicentral_distance_km"].median()) if "epicentral_distance_km" in st else np.nan,
            "PGA median m/s2": float(st[pga_col].median()) if pga_col in st else np.nan,
            "PGA max m/s2": float(st[pga_col].max()) if pga_col in st else np.nan,
        })
    summary = pd.DataFrame(summary_rows)
    save_df(summary, out_dir / "data_split_summary")
    table_png(summary, out_dir / "data_split_summary.png", "Training Data Summary", font_size=15)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    ax = axes[0, 0]
    counts = events["split"].astype(str).value_counts().reindex(event_splits).fillna(0)
    ax.bar([split_label(x) for x in counts.index], counts.values, color=[colors.get(x, "#888888") for x in counts.index])
    ax.set_title("Event Count by Split", fontsize=22, weight="bold")
    ax.set_ylabel("Events", fontsize=18)
    ax.tick_params(labelsize=15)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    for split in event_splits:
        vals = events.loc[events["split"].astype(str) == split, "Magnitude"].dropna()
        if not vals.empty:
            ax.hist(vals, bins=np.arange(2.8, max(8.1, vals.max() + 0.3), 0.3), alpha=0.55, label=split_label(split), color=colors.get(split))
    ax.set_title("Magnitude Distribution", fontsize=22, weight="bold")
    ax.set_xlabel("Magnitude", fontsize=18)
    ax.set_ylabel("Events", fontsize=18)
    ax.tick_params(labelsize=15)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=15)

    ax = axes[1, 0]
    if "epicentral_distance_km" in stations:
        bins = np.linspace(0, min(1000, max(100, stations["epicentral_distance_km"].quantile(0.995))), 32)
        for split in station_splits:
            vals = stations.loc[stations["split"].astype(str) == split, "epicentral_distance_km"].dropna()
            if not vals.empty:
                ax.hist(vals, bins=bins, alpha=0.5, label=split_label(split), color=colors.get(split))
    ax.set_title("Station Epicentral Distance", fontsize=22, weight="bold")
    ax.set_xlabel("Epicentral distance (km)", fontsize=18)
    ax.set_ylabel("Station records", fontsize=18)
    ax.tick_params(labelsize=15)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=15)

    ax = axes[1, 1]
    if pga_col in stations:
        pga = stations[pga_col].where(stations[pga_col] > 0)
        lo = np.nanpercentile(np.log10(pga.dropna()), 0.5)
        hi = np.nanpercentile(np.log10(pga.dropna()), 99.5)
        bins = np.linspace(lo, hi, 36)
        for split in station_splits:
            vals = stations.loc[stations["split"].astype(str) == split, pga_col].dropna()
            vals = np.log10(vals[vals > 0])
            if not vals.empty:
                ax.hist(vals, bins=bins, alpha=0.5, label=split_label(split), color=colors.get(split))
    ax.set_title("PGA Distribution", fontsize=22, weight="bold")
    ax.set_xlabel("log10(PGA m/s2)", fontsize=18)
    ax.set_ylabel("Station records", fontsize=18)
    ax.tick_params(labelsize=15)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=15)

    fig.tight_layout()
    fig.savefig(out_dir / "data_distribution_overview.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    if "n_station_rows" in events:
        ax = axes[0]
        for split in event_splits:
            vals = events.loc[events["split"].astype(str) == split, "n_station_rows"].dropna()
            if not vals.empty:
                ax.hist(np.log10(vals.clip(lower=1)), bins=28, alpha=0.5, label=split_label(split), color=colors.get(split))
        ax.set_title("Station Records per Event", fontsize=22, weight="bold")
        ax.set_xlabel("log10(records per event)", fontsize=18)
        ax.set_ylabel("Events", fontsize=18)
        ax.tick_params(labelsize=15)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=15)
    if "DEPTH" in events:
        ax = axes[1]
        for split in event_splits:
            vals = events.loc[events["split"].astype(str) == split, "DEPTH"].dropna()
            if not vals.empty:
                ax.hist(vals, bins=28, alpha=0.5, label=split_label(split), color=colors.get(split))
        ax.set_title("Event Depth Distribution", fontsize=22, weight="bold")
        ax.set_xlabel("Depth (km)", fontsize=18)
        ax.set_ylabel("Events", fontsize=18)
        ax.tick_params(labelsize=15)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=15)
    fig.tight_layout()
    fig.savefig(out_dir / "data_event_station_depth.png", dpi=220)
    plt.close(fig)

    plot_event_station_distribution_map(events, stations, out_dir)


def plot_event_station_distribution_map(events: pd.DataFrame, stations: pd.DataFrame, out_dir: Path) -> None:
    if not {"Latitude", "Longitude", "Magnitude"}.issubset(events.columns):
        return
    if not {"station_lat", "station_lon"}.issubset(stations.columns):
        return
    event_points = events[["Longitude", "Latitude", "Magnitude", "split"]].dropna()
    station_points = stations[["station_lon", "station_lat", "split"]].dropna()
    if event_points.empty or station_points.empty:
        return

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(1, 1, 1, **map_subplot_kw())
    event_lonlat = event_points[["Longitude", "Latitude"]].to_numpy(dtype=float)
    station_lonlat = station_points[["station_lon", "station_lat"]].drop_duplicates().to_numpy(dtype=float)
    draw_reference_map(ax, [event_lonlat, station_lonlat])
    geo_kwargs = {"transform": ccrs.PlateCarree()} if is_cartopy_axis(ax) else {}

    ax.scatter(
        station_lonlat[:, 0],
        station_lonlat[:, 1],
        marker="^",
        s=22,
        color="#4d4d4d",
        alpha=0.42,
        linewidth=0,
        label=f"Stations ({len(station_lonlat)})",
        zorder=2,
        **geo_kwargs,
    )
    sc = ax.scatter(
        event_points["Longitude"],
        event_points["Latitude"],
        c=event_points["Magnitude"],
        s=24 + 16 * np.maximum(event_points["Magnitude"].to_numpy(dtype=float) - 3.0, 0.0),
        cmap="plasma",
        alpha=0.82,
        edgecolor="black",
        linewidth=0.25,
        label=f"Events ({len(event_points)})",
        zorder=3,
        **geo_kwargs,
    )
    ax.set_title("Event and Station Distribution", fontsize=24, weight="bold")
    ax.set_xlabel("Longitude", fontsize=18)
    ax.set_ylabel("Latitude", fontsize=18)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14, loc="lower left")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Magnitude", fontsize=16)
    cbar.ax.tick_params(labelsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "data_event_station_map.png", dpi=220)
    plt.close(fig)


def plot_summary_bars(metrics: pd.DataFrame, out_dir: Path) -> None:
    val = metrics[metrics["split"] == "val"].copy()
    if val.empty:
        return
    order = readable_model_order(metrics)
    if len(order) <= 1:
        table = val[["model", "checkpoint", "mae", "rmse", "corr", "r2", "slope"]].copy()
        table_png(table, out_dir / "validation_metric_summary.png", "Validation Metrics", font_size=14)
        return
    val["model_label"] = val["model_id"].map(lambda x: display_model_label(x))
    val["model_label"] = pd.Categorical(val["model_label"], [display_model_label(x) for x in order], ordered=True)
    val = val.sort_values("model_label")

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    for ax, metric, ylabel in zip(axes, ["mae", "r2", "slope"], ["MAE", "R2", "Linear-fit slope"]):
        ax.bar(val["model_label"].astype(str), val[metric], color="#386cb0")
        ax.set_ylabel(ylabel, fontsize=18)
        ax.tick_params(axis="x", rotation=35, labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title("Validation Error", fontsize=20, weight="bold")
    axes[1].set_title("Explained Variance", fontsize=20, weight="bold")
    axes[2].set_title("Dynamic Range", fontsize=20, weight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "validation_metric_bars.png", dpi=220)
    plt.close(fig)


def plot_train_val_metric(metrics: pd.DataFrame, out_dir: Path) -> None:
    if metrics.empty:
        return
    metrics = preferred_checkpoint_rows(metrics)
    order = readable_model_order(metrics)
    fig, axes = plt.subplots(1, 3, figsize=(23, 7))
    for ax, metric, ylabel in zip(axes, ["mae", "r2", "slope"], ["MAE", "R2", "Linear-fit slope"]):
        x = np.arange(len(order))
        width = 0.38
        for offset, split, color in [(-width / 2, "train", "#4daf4a"), (width / 2, "val", "#377eb8")]:
            vals = []
            for model_id in order:
                row = metrics[(metrics.model_id == model_id) & (metrics.split == split)]
                vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + offset, vals, width=width, label=split, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels([display_model_label(m) for m in order], rotation=0, ha="center", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.tick_params(axis="y", labelsize=14)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=14)
    axes[0].set_title("Train vs Validation Error", fontsize=20, weight="bold")
    axes[1].set_title("Train vs Validation R2", fontsize=20, weight="bold")
    axes[2].set_title("Train vs Validation Slope", fontsize=20, weight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "train_val_metric_bars.png", dpi=220)
    plt.close(fig)


def plot_station_buckets(bucket_df: pd.DataFrame, out_dir: Path, selected: list[str]) -> None:
    val = bucket_df[bucket_df["split"] == "val"].copy()
    if val.empty:
        return
    val = preferred_checkpoint_rows(val)
    if selected:
        val = val[val["model_id"].isin(selected)]
    if val.empty:
        return
    val["bucket"] = pd.Categorical(val["bucket"], BUCKET_ORDER, ordered=True)
    fig, ax = plt.subplots(figsize=(14, 8))
    for model_id, group in val.groupby("model_id", sort=False):
        group = group.sort_values("bucket")
        ax.plot(
            group["bucket"].astype(str),
            group["mae"],
            marker="o",
            linewidth=3,
            markersize=8,
            label=display_model_label(model_id),
        )
    ax.set_xlabel("Number of input stations", fontsize=18)
    ax.set_ylabel("Validation MAE", fontsize=18)
    ax.set_title("PGA Error by Input Station Count", fontsize=22, weight="bold")
    ax.tick_params(labelsize=15)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "val_mae_by_station_count.png", dpi=220)
    plt.close(fig)


def distance_bucket_table(eval_file: EvalFile, split: str, bin_edges: list[float]) -> pd.DataFrame | None:
    if eval_file.npz_path is None:
        return None
    arrays = get_distance_arrays(eval_file.npz_path, split)
    if arrays is None:
        return None
    distances, labels, preds, valid = arrays
    d = distances[valid]
    y = labels[valid]
    p = preds[valid]
    if len(d) == 0:
        return None
    rows = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if math.isinf(hi):
            mask = d >= lo
            name = f"{lo:g}+"
        else:
            mask = (d >= lo) & (d < hi)
            name = f"{lo:g}-{hi:g}"
        if not mask.any():
            continue
        res = p[mask] - y[mask]
        rows.append({
            "model_id": eval_file.model_id,
            "model": display_model_label(eval_file.model_id),
            "checkpoint": eval_file.checkpoint_tag,
            "split": split,
            "distance_bin": name,
            "targets": int(mask.sum()),
            "distance_mean": float(d[mask].mean()),
            "mae": float(np.mean(np.abs(res))),
            "rmse": float(np.sqrt(np.mean(res ** 2))),
            "bias": float(np.mean(res)),
            "corr": float(np.corrcoef(y[mask], p[mask])[0, 1]) if mask.sum() > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def plot_distance_buckets(evals: list[EvalFile], out_dir: Path, selected: list[str], bin_edges: list[float]) -> pd.DataFrame:
    frames = []
    for eval_file in evals:
        if selected and eval_file.model_id not in selected:
            continue
        df = distance_bucket_table(eval_file, "val", bin_edges)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    all_bins = pd.concat(frames, ignore_index=True)
    save_df(all_bins, out_dir / "epicentral_distance_buckets")
    plot_bins = preferred_checkpoint_rows(all_bins)

    fig, ax = plt.subplots(figsize=(14, 8))
    for model_id, group in plot_bins.groupby("model_id", sort=False):
        ax.plot(
            group["distance_bin"],
            group["mae"],
            marker="o",
            linewidth=3,
            markersize=8,
            label=display_model_label(model_id),
        )
    ax.set_xlabel("Epicentral distance to PGA target (km)", fontsize=18)
    ax.set_ylabel("Validation MAE", fontsize=18)
    ax.set_title("PGA Error by Epicentral Distance", fontsize=22, weight="bold")
    ax.tick_params(labelsize=15)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "val_mae_by_epicentral_distance.png", dpi=220)
    plt.close(fig)
    return all_bins


def plot_scatter_and_residual(eval_file: EvalFile, out_dir: Path, max_points: int, rng: np.random.Generator) -> None:
    if eval_file.npz_path is None:
        return
    for split in ("train", "val"):
        try:
            labels, preds, valid, _ = get_pga_arrays(eval_file.npz_path, split)
        except KeyError:
            continue
        y, p = flatten_valid(labels, preds, valid)
        if len(y) == 0:
            continue
        idx = np.arange(len(y))
        if len(idx) > max_points:
            idx = rng.choice(idx, size=max_points, replace=False)
        ys = y[idx]
        ps = p[idx]
        residual = ps - ys
        lo = min(ys.min(), ps.min())
        hi = max(ys.max(), ps.max())
        slope, intercept = np.polyfit(y, p, 1)
        corr = np.corrcoef(y, p)[0, 1]
        mae = np.mean(np.abs(p - y))
        r2 = 1.0 - np.sum((p - y) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)

        fig, ax = plt.subplots(figsize=(9, 8))
        ax.scatter(ys, ps, s=18, alpha=0.35, color="#377eb8", edgecolors="none")
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=2.5, linestyle="--", label="y = x")
        ax.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept], color="#e41a1c", linewidth=3, label="fit")
        ax.set_xlabel("True log PGA", fontsize=18)
        ax.set_ylabel("Predicted log PGA", fontsize=18)
        split_title = "Training" if split == "train" else "Validation"
        ax.set_title(f"Predicted vs True PGA ({split_title})", fontsize=22, weight="bold")
        ax.text(
            0.04,
            0.96,
            f"MAE={mae:.3f}\nR2={r2:.3f}\nCorr={corr:.3f}\nSlope={slope:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=15,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="#cccccc"),
        )
        ax.tick_params(labelsize=15)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=14, loc="lower right")
        fig.tight_layout()
        fig.savefig(out_dir / f"scatter_{eval_file.model_id}_{eval_file.checkpoint_tag}_{split}.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(ys, residual, s=18, alpha=0.35, color="#984ea3", edgecolors="none")
        ax.axhline(0, color="black", linewidth=2, linestyle="--")
        ax.set_xlabel("True log PGA", fontsize=18)
        ax.set_ylabel("Prediction residual", fontsize=18)
        ax.set_title(f"Residual vs True PGA ({split_title})", fontsize=22, weight="bold")
        ax.tick_params(labelsize=15)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"residual_vs_true_{eval_file.model_id}_{eval_file.checkpoint_tag}_{split}.png", dpi=220)
        plt.close(fig)


def pga_bin_table(eval_file: EvalFile, split: str, n_bins: int) -> pd.DataFrame | None:
    if eval_file.npz_path is None:
        return None
    try:
        labels, preds, valid, _ = get_pga_arrays(eval_file.npz_path, split)
    except KeyError:
        return None
    y, p = flatten_valid(labels, preds, valid)
    if len(y) < n_bins:
        return None
    qs = np.quantile(y, np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs)
    if len(qs) <= 2:
        return None
    rows = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        if hi == qs[-1]:
            mask = (y >= lo) & (y <= hi)
        else:
            mask = (y >= lo) & (y < hi)
        if not mask.any():
            continue
        res = p[mask] - y[mask]
        rows.append({
            "bin": f"[{lo:.2f}, {hi:.2f}]",
            "n": int(mask.sum()),
            "label_mean": float(y[mask].mean()),
            "mae": float(np.mean(np.abs(res))),
            "bias": float(np.mean(res)),
            "rmse": float(np.sqrt(np.mean(res ** 2))),
        })
    return pd.DataFrame(rows)


def plot_pga_bins(eval_file: EvalFile, out_dir: Path, split: str, n_bins: int) -> None:
    df = pga_bin_table(eval_file, split, n_bins)
    if df is None or df.empty:
        return
    save_df(df, out_dir / f"pga_strength_bins_{eval_file.model_id}_{eval_file.checkpoint_tag}_{split}")
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.bar(x - 0.18, df["mae"], width=0.36, label="MAE", color="#377eb8")
    ax.bar(x + 0.18, df["bias"], width=0.36, label="Bias", color="#ff7f00")
    ax.axhline(0, color="black", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["bin"], rotation=25, ha="right", fontsize=13)
    ax.set_ylabel("Error", fontsize=18)
    ax.set_title("Error by True PGA Strength", fontsize=22, weight="bold")
    ax.text(
        0.01,
        0.98,
        "MAE = mean |pred - true|\nBias = mean(pred - true)",
        transform=ax.transAxes,
        va="top",
        fontsize=14,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=15)
    fig.tight_layout()
    fig.savefig(out_dir / f"pga_strength_bins_{eval_file.model_id}_{eval_file.checkpoint_tag}_{split}.png", dpi=220)
    plt.close(fig)


def station_count_bucket(count: int) -> str:
    if count <= 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 5:
        return "4-5"
    if count <= 10:
        return "6-10"
    if count <= 15:
        return "11-15"
    return "16+"


def event_metadata_from_split_table(result_dir: Path, split: str, event_idx: int) -> dict:
    """Return event metadata for a split-local event index when split CSV exists."""
    path = result_dir / "split_events.csv"
    if not path.exists():
        return {}
    try:
        events = pd.read_csv(path)
    except Exception:
        return {}
    split_alias = {"val": "dev", "validation": "dev"}.get(split, split)
    subset = events[events["split"].astype(str) == split_alias].reset_index(drop=True)
    if event_idx < 0 or event_idx >= len(subset):
        return {}
    row = subset.iloc[int(event_idx)]
    out = {}
    if "Magnitude" in row:
        out["magnitude"] = float(row["Magnitude"])
    if "EVENT" in row:
        out["event_id"] = str(row["EVENT"])
    return out


def plot_case_studies(eval_file: EvalFile, out_dir: Path, split: str, max_cases: int) -> None:
    if eval_file.npz_path is None:
        return
    data = np.load(eval_file.npz_path, allow_pickle=True)
    labels, preds, valid, counts = get_pga_arrays(eval_file.npz_path, split)
    target_coords = npz_optional_stack(data, f"{split}_pga_target_abs")
    event_coords = npz_optional_stack(data, f"{split}_loc_label_abs")
    station_coords = npz_optional_stack(data, f"{split}_station_coords_abs")
    station_valid = npz_optional_stack(data, f"{split}_station_valid")

    preferred_events: list[int] = []
    preferred_path = out_dir / f"selected_case_events_{eval_file.model_id}_{eval_file.checkpoint_tag}_{split}.json"
    if preferred_path.exists():
        try:
            preferred_events = [int(x) for x in json.loads(preferred_path.read_text()).get("event_indices", [])]
        except Exception:
            preferred_events = []

    event_mae = []
    flat_labels = labels.reshape(labels.shape[0], -1)
    flat_preds = preds.reshape(preds.shape[0], -1)
    for i in range(labels.shape[0]):
        mask = valid[i]
        if not mask.any():
            continue
        res = flat_preds[i][mask] - flat_labels[i][mask]
        event_mae.append((float(np.mean(np.abs(res))), i))
    if not event_mae:
        return
    event_mae.sort()
    case_candidates = []
    for mae, idx in event_mae:
        mask = valid[idx].astype(bool)
        valid_count = int(mask.sum())
        overlap = 0
        distance_span = 0.0
        if (
            target_coords is not None
            and station_coords is not None
            and station_valid is not None
            and event_coords is not None
            and np.asarray(target_coords[idx]).ndim == 2
        ):
            targets = np.asarray(target_coords[idx], dtype=float)[mask]
            stations = np.asarray(station_coords[idx], dtype=float)
            station_mask = np.asarray(station_valid[idx]).astype(bool)
            stations = stations[station_mask] if stations.ndim == 2 and station_mask.ndim == 1 else np.empty((0, 3))
            for target in targets:
                if stations.size and np.any(np.linalg.norm(stations[:, :2] - target[:2], axis=1) < 1e-5):
                    overlap += 1
            if targets.shape[0] >= 2:
                ev = np.asarray(event_coords[idx], dtype=float).reshape(-1)
                distances = horizontal_distances_km(targets, ev)
                distance_span = float(distances.max() - distances.min())
        overlap_ratio = overlap / max(valid_count, 1)
        case_candidates.append((valid_count, overlap_ratio, distance_span, mae, idx))

    report_ready = [
        item for item in case_candidates
        if item[0] >= 5 and item[1] < 0.8
    ]
    report_ready.sort(key=lambda item: (item[3], -item[0], -item[2], item[1]))
    fallback = [
        item for item in case_candidates
        if item[0] >= 2 and item[1] < 1.0
    ]
    fallback.sort(key=lambda item: (item[3], -item[0], -item[2], item[1]))
    selected = []
    preferred_items = [item for event in preferred_events for item in case_candidates if item[-1] == event]
    for item in preferred_items + report_ready + fallback + case_candidates:
        idx = item[-1]
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= max_cases:
            break

    for rank, event_idx in enumerate(selected, start=1):
        mask = valid[event_idx].astype(bool)
        y = flat_labels[event_idx][mask].astype(float)
        p = flat_preds[event_idx][mask].astype(float)
        res = p - y
        station_count = int(counts[event_idx]) if counts is not None else -1
        station_bucket = station_count_bucket(station_count) if station_count >= 0 else "unknown"

        have_coords = (
            target_coords is not None
            and event_coords is not None
            and np.asarray(target_coords[event_idx]).ndim == 2
        )
        fig, axes = plt.subplots(1, 3 if have_coords else 2, figsize=(22 if have_coords else 15, 7), constrained_layout=True)
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])
        if have_coords:
            axes = axes.astype(object)
            axes[2] = replace_with_map_axis(fig, axes[2])

        ax0 = axes[0]
        target_idx = np.arange(len(y))
        distance_order = np.arange(len(y))
        x_values = target_idx
        x_label = "Valid PGA target index"
        if have_coords:
            coords_for_x = np.asarray(target_coords[event_idx], dtype=float)[mask]
            ev_for_x = np.asarray(event_coords[event_idx], dtype=float).reshape(-1)
            x_values = horizontal_distances_km(coords_for_x, ev_for_x)
            distance_order = np.argsort(x_values)
            x_values = x_values[distance_order]
        ax0.plot(x_values, y[distance_order], marker="o", linewidth=2.5, label="True", color="#377eb8")
        ax0.plot(x_values, p[distance_order], marker="s", linewidth=2.5, label="Pred", color="#e41a1c")
        if have_coords:
            x_label = "Epicentral distance to target (km)"
        ax0.set_xlabel(x_label, fontsize=17)
        ax0.set_ylabel("log PGA", fontsize=17)
        ax0.set_title("PGA by Epicentral Distance" if have_coords else "Target-wise PGA", fontsize=20, weight="bold")
        ax0.tick_params(labelsize=14)
        ax0.grid(alpha=0.3)
        ax0.legend(fontsize=14)

        ax1 = axes[1]
        if have_coords:
            coords = np.asarray(target_coords[event_idx], dtype=float)[mask]
            ev = np.asarray(event_coords[event_idx], dtype=float).reshape(-1)[: coords.shape[-1]]
            distances = horizontal_distances_km(coords, ev)
            ax1.scatter(distances, res, s=70, c=res, cmap="coolwarm", edgecolor="black", linewidth=0.4)
            ax1.set_xlabel("Epicentral distance to target (km)", fontsize=17)
        else:
            distances = target_idx
            ax1.scatter(target_idx, res, s=70, c=res, cmap="coolwarm", edgecolor="black", linewidth=0.4)
            ax1.set_xlabel("Valid PGA target index", fontsize=17)
        ax1.axhline(0, color="black", linestyle="--", linewidth=2)
        ax1.set_ylabel("Residual", fontsize=17)
        ax1.set_title("Residual Diagnostics", fontsize=20, weight="bold")
        ax1.tick_params(labelsize=14)
        ax1.grid(alpha=0.3)

        if have_coords:
            ax2 = axes[2]
            coords = np.asarray(target_coords[event_idx], dtype=float)[mask]
            ev = np.asarray(event_coords[event_idx], dtype=float).reshape(-1)
            st = None
            if station_coords is not None and station_valid is not None:
                st = np.asarray(station_coords[event_idx], dtype=float)
                sv = np.asarray(station_valid[event_idx]).astype(bool)
                if st.ndim == 2 and sv.ndim == 1:
                    st = st[sv]
            sc = draw_spatial_residual_map(ax2, coords, res, ev, station_xy=st)
            ax2.legend(fontsize=13)
            cbar = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
            cbar.set_label("Residual", fontsize=15)
            cbar.ax.tick_params(labelsize=13)

        mae = float(np.mean(np.abs(res)))
        bias = float(np.mean(res))
        meta = event_metadata_from_split_table(eval_file.result_dir, split, int(event_idx))
        mag_text = f", M={meta['magnitude']:.1f}" if "magnitude" in meta else ""
        fig.suptitle(
            f"Case {rank}: {display_model_label(eval_file.model_id)} "
            f"({split}, event {event_idx}{mag_text}, stations={station_count}, bucket={station_bucket}, "
            f"MAE={mae:.3f}, bias={bias:.3f})",
            fontsize=22,
            weight="bold",
        )
        fig.savefig(out_dir / f"case_study_{rank}_{eval_file.model_id}_{eval_file.checkpoint_tag}_{split}.png", dpi=220)
        plt.close(fig)


def plot_case_station_sweeps(eval_file: EvalFile, out_dir: Path, split: str, max_cases: int) -> None:
    if eval_file.npz_path is None or max_cases <= 0:
        return
    data = np.load(eval_file.npz_path, allow_pickle=True)
    prefix = f"case_sweep_{split}_"
    required = [
        f"{prefix}event_index",
        f"{prefix}requested_station_count",
        f"{prefix}actual_station_count",
        f"{prefix}pga_label",
        f"{prefix}pga_mu_best",
        f"{prefix}pga_target_valid",
    ]
    if any(key not in data for key in required):
        return

    event_indices = np.asarray(data[f"{prefix}event_index"], dtype=np.int64).reshape(-1)
    requested = np.asarray(data[f"{prefix}requested_station_count"], dtype=np.int64).reshape(-1)
    actual = np.asarray(data[f"{prefix}actual_station_count"], dtype=np.int64).reshape(-1)
    labels = npz_stack(data[f"{prefix}pga_label"]).reshape(event_indices.shape[0], -1).astype(float)
    preds = npz_stack(data[f"{prefix}pga_mu_best"]).reshape(event_indices.shape[0], -1).astype(float)
    valid = npz_stack(data[f"{prefix}pga_target_valid"]).reshape(event_indices.shape[0], -1).astype(bool)
    target_coords = npz_optional_stack(data, f"{prefix}pga_target_abs")
    event_coords = npz_optional_stack(data, f"{prefix}loc_label_abs")
    station_coords = npz_optional_stack(data, f"{prefix}station_coords_abs")
    station_valid = npz_optional_stack(data, f"{prefix}station_valid")

    event_candidates = []
    seen_events = []
    for idx in event_indices:
        if int(idx) not in seen_events:
            seen_events.append(int(idx))
    for idx in seen_events:
        rows = np.where(event_indices == idx)[0]
        max_valid = 0
        max_distance_span = 0.0
        best_mae = float("inf")
        best_actual = -1
        best_overlap_ratio = 1.0
        for row in rows:
            mask = valid[row]
            max_valid = max(max_valid, int(mask.sum()))
            if mask.any():
                residual = preds[row][mask] - labels[row][mask]
                mae = float(np.mean(np.abs(residual)))
                actual_count = int(actual[row])
                if actual_count > best_actual or (actual_count == best_actual and mae < best_mae):
                    best_actual = actual_count
                    best_mae = mae
                    best_overlap_ratio = 1.0
                    if (
                        target_coords is not None
                        and station_coords is not None
                        and station_valid is not None
                        and np.asarray(target_coords[row]).ndim == 2
                    ):
                        targets = np.asarray(target_coords[row], dtype=float)[mask]
                        st_all = np.asarray(station_coords[row], dtype=float)
                        sv = np.asarray(station_valid[row]).astype(bool)
                        stations = st_all[sv] if st_all.ndim == 2 and sv.ndim == 1 else np.empty((0, 3))
                        if targets.size and stations.size:
                            overlaps = sum(
                                1 for target in targets
                                if np.any(np.linalg.norm(stations[:, :2] - target[:2], axis=1) < 1e-5)
                            )
                            best_overlap_ratio = overlaps / max(len(targets), 1)
            if (
                target_coords is not None
                and event_coords is not None
                and np.asarray(target_coords[row]).ndim == 2
                and mask.sum() >= 2
            ):
                coords = np.asarray(target_coords[row], dtype=float)[mask]
                ev = np.asarray(event_coords[row], dtype=float).reshape(-1)
                distances = horizontal_distances_km(coords, ev)
                max_distance_span = max(max_distance_span, float(distances.max() - distances.min()))
        event_candidates.append((max_valid, best_actual, best_mae, max_distance_span, best_overlap_ratio, int(idx)))

    report_ready = [item for item in event_candidates if item[0] >= 8 and item[1] >= 8 and item[3] >= 20 and item[4] < 0.4]
    report_ready.sort(key=lambda item: (item[2], -item[0], -item[1], -item[3], item[4]))
    fallback = [item for item in event_candidates if item not in report_ready]
    fallback.sort(key=lambda item: (item[4], item[2], -item[0], -item[1], -item[3]))
    unique_events = [item[5] for item in (report_ready + fallback)[:max_cases]]
    (out_dir / f"selected_case_events_{eval_file.model_id}_{eval_file.checkpoint_tag}_{split}.json").write_text(
        json.dumps({"event_indices": unique_events, "selection": "low_mae_at_largest_actual_station_count"}, indent=2)
    )

    for rank, event_idx in enumerate(unique_events, start=1):
        rows = np.where(event_indices == event_idx)[0]
        if rows.size == 0:
            continue
        order = np.argsort(requested[rows])
        rows = rows[order]

        per_count = []
        for row in rows:
            mask = valid[row]
            if not mask.any():
                continue
            residual = preds[row][mask] - labels[row][mask]
            per_count.append((
                int(requested[row]),
                int(actual[row]),
                float(np.mean(np.abs(residual))),
                float(np.sqrt(np.mean(residual ** 2))),
                float(np.mean(residual)),
                row,
            ))
        if not per_count:
            continue

        count_values = np.asarray([x[0] for x in per_count], dtype=int)
        actual_values = np.asarray([x[1] for x in per_count], dtype=int)
        maes = np.asarray([x[2] for x in per_count], dtype=float)
        biases = np.asarray([x[4] for x in per_count], dtype=float)
        map_row = per_count[-1][5]
        map_mask = valid[map_row]
        map_res = preds[map_row][map_mask] - labels[map_row][map_mask]

        have_coords = (
            target_coords is not None
            and event_coords is not None
            and np.asarray(target_coords[map_row]).ndim == 2
            and map_mask.any()
        )
        fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])

        ax0 = axes[0]
        ax0.plot(count_values, maes, marker="o", linewidth=3, color="#377eb8", label="MAE")
        ax0.plot(count_values, np.abs(biases), marker="s", linewidth=3, color="#e41a1c", label="|Bias|")
        for x, y, a in zip(count_values, maes, actual_values):
            ax0.text(x, y, f" actual {a}", fontsize=12, ha="center", va="bottom")
        ax0.set_xlabel("Requested input stations", fontsize=17)
        ax0.set_ylabel("Error", fontsize=17)
        ax0.set_title("Error vs Station Count", fontsize=20, weight="bold")
        ax0.tick_params(labelsize=14)
        ax0.grid(alpha=0.3)
        ax0.legend(fontsize=14)

        ax1 = axes[1]
        if have_coords:
            coords = np.asarray(target_coords[map_row], dtype=float)[map_mask]
            ev = np.asarray(event_coords[map_row], dtype=float).reshape(-1)[: coords.shape[-1]]
            distances = horizontal_distances_km(coords, ev)
            ax1.scatter(distances, map_res, s=70, c=map_res, cmap="coolwarm", edgecolor="black", linewidth=0.4)
            ax1.set_xlabel("Epicentral distance to target (km)", fontsize=17)
            ax1.set_title("Residual vs Distance", fontsize=20, weight="bold")
        else:
            target_idx = np.arange(int(map_mask.sum()))
            ax1.scatter(target_idx, map_res, s=70, c=map_res, cmap="coolwarm", edgecolor="black", linewidth=0.4)
            ax1.set_xlabel("Valid PGA target index", fontsize=17)
            ax1.set_title("Residual Diagnostics", fontsize=20, weight="bold")
        ax1.axhline(0, color="black", linestyle="--", linewidth=2)
        ax1.set_ylabel("Residual", fontsize=17)
        ax1.tick_params(labelsize=14)
        ax1.grid(alpha=0.3)

        meta = event_metadata_from_split_table(eval_file.result_dir, split, int(event_idx))
        mag_text = f", M={meta['magnitude']:.1f}" if "magnitude" in meta else ""
        fig.suptitle(
            f"Case Study: Station Count Sweep ({split}, event {event_idx}{mag_text})",
            fontsize=22,
            weight="bold",
        )
        fig.savefig(out_dir / f"case_station_sweep_{rank}_{eval_file.model_id}_{eval_file.checkpoint_tag}_{split}.png", dpi=220)
        plt.close(fig)

        if have_coords:
            map_entries = []
            for requested_count, actual_count, _, _, _, row in per_count:
                mask = valid[row]
                if not mask.any():
                    continue
                coords = np.asarray(target_coords[row], dtype=float)[mask]
                ev = np.asarray(event_coords[row], dtype=float).reshape(-1)
                residual = preds[row][mask] - labels[row][mask]
                st = None
                if station_coords is not None and station_valid is not None:
                    st_all = np.asarray(station_coords[row], dtype=float)
                    sv = np.asarray(station_valid[row]).astype(bool)
                    if st_all.ndim == 2 and sv.ndim == 1:
                        st = st_all[sv]
                map_entries.append((requested_count, actual_count, coords, ev, residual, st))

            if map_entries:
                ncols = 3
                nrows = int(math.ceil(len(map_entries) / ncols))
                fig, axes = plt.subplots(
                    nrows,
                    ncols,
                    figsize=(7.8 * ncols, 7.2 * nrows),
                    squeeze=False,
                    constrained_layout=True,
                    subplot_kw=map_subplot_kw(),
                )
                vlim = max(max(float(np.nanmax(np.abs(entry[4]))), 0.1) for entry in map_entries)
                last_sc = None
                for panel_idx, (ax, entry) in enumerate(zip(axes.ravel(), map_entries)):
                    requested_count, actual_count, coords, ev, residual, st = entry
                    last_sc = draw_spatial_residual_map(
                        ax,
                        coords,
                        residual,
                        ev,
                        station_xy=st,
                        title=f"Requested {requested_count}, actual {actual_count}",
                        vlim=vlim,
                    )
                    row_idx = panel_idx // ncols
                    col_idx = panel_idx % ncols
                    if row_idx < nrows - 1:
                        ax.set_xlabel("")
                        ax.tick_params(labelbottom=False)
                    if col_idx > 0:
                        ax.set_ylabel("")
                for ax in axes.ravel()[len(map_entries):]:
                    ax.axis("off")
                axes.ravel()[0].legend(loc="upper left", fontsize=12, frameon=True)
                if last_sc is not None:
                    cbar = fig.colorbar(last_sc, ax=axes.ravel().tolist(), location="right", shrink=0.86, pad=0.02)
                    cbar.set_label("Residual", fontsize=16)
                    cbar.ax.tick_params(labelsize=13)
                fig.suptitle(
                    f"Spatial Residual Maps by Input Station Count ({split}, event {event_idx}{mag_text})",
                    fontsize=24,
                    weight="bold",
                )
                fig.savefig(out_dir / f"case_station_maps_{rank}_{eval_file.model_id}_{eval_file.checkpoint_tag}_{split}.png", dpi=220)
                plt.close(fig)


def plot_loss_curves(eval_file: EvalFile, out_dir: Path) -> None:
    train_path = eval_file.result_dir / "train_epoch_loss.csv"
    val_path = eval_file.result_dir / "val_epoch_loss.csv"
    if not train_path.exists() or not val_path.exists():
        return
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    if train.empty or val.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(train["step"], train["value"], label="train", linewidth=3, color="#4daf4a")
    ax.plot(val["step"], val["value"], label="val", linewidth=3, color="#377eb8")
    best_idx = int(val["value"].idxmin())
    ax.scatter([val.loc[best_idx, "step"]], [val.loc[best_idx, "value"]], s=110, color="#e41a1c", zorder=5)
    ax.text(
        val.loc[best_idx, "step"],
        val.loc[best_idx, "value"],
        f" best val @ {int(val.loc[best_idx, 'step'])}",
        fontsize=14,
        va="bottom",
    )
    ax.set_xlabel("Epoch", fontsize=18)
    ax.set_ylabel("Loss", fontsize=18)
    ax.set_title("Training Curves", fontsize=22, weight="bold")
    ax.tick_params(labelsize=15)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=15)
    fig.tight_layout()
    fig.savefig(out_dir / f"loss_curve_{eval_file.model_id}.png", dpi=220)
    plt.close(fig)


def positioning_cards_png(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(len(df), 1, figsize=(16, 4.2 * len(df)))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    colors = ["#e8f1fb", "#edf7ed", "#fff2df"]
    for idx, (ax, row) in enumerate(zip(axes, df.to_dict("records"))):
        ax.axis("off")
        ax.set_facecolor(colors[idx % len(colors)])
        ax.text(0.02, 0.88, row["Work"], fontsize=22, weight="bold", va="top")
        body = (
            f"Task: {row['Task']}\n"
            f"Inputs: {row['Inputs']}\n"
            f"Spatial modeling: {row['Spatial modeling']}\n"
            f"Relation: {row['Relation to this project']}"
        )
        wrapped = "\n".join(
            "\n".join(textwrap.wrap(line, width=120)) for line in body.splitlines()
        )
        ax.text(0.02, 0.68, wrapped, fontsize=17, va="top", linespacing=1.35)
        ax.add_patch(
            plt.Rectangle((0.0, 0.0), 1.0, 1.0, transform=ax.transAxes, fill=False, edgecolor="#b7c0cc", linewidth=1.5)
        )
    fig.suptitle("Literature Positioning", fontsize=26, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=220)
    plt.close(fig)


def write_literature_tables(out_dir: Path) -> None:
    rows = [
        {
            "Work": "TEAM (Muenchmeyer et al., 2021)",
            "Task": "Real-time PGA distribution / warning thresholds",
            "Inputs": "Raw strong-motion waveforms; arbitrary station set and target locations",
            "Spatial modeling": "Transformer with station and target coordinates",
            "Relation to this project": "Base architecture; this project replaces waveform frontend with DiTing and uses target cross-attention for PGA readout",
        },
        {
            "Work": "QuakeFormer (Feng et al., 2024 preprint)",
            "Task": "Unified ground-motion forecasting, interpolation, and early warning",
            "Inputs": "Multi-station observations with flexible masking",
            "Spatial modeling": "Masked Transformer with absolute and relative coordinate embeddings",
            "Relation to this project": "Conceptually related in using masked/partial observations and relative geometry; this project focuses on PGA targets for Japan data",
        },
        {
            "Work": "This project",
            "Task": "Amplitude-free target-station PGA estimation",
            "Inputs": "Normalized waveforms, station coordinates, PGA target coordinates",
            "Spatial modeling": "DiTing station encoder + TEAM transformer + target cross-attention with relative geometry bias",
            "Relation to this project": "Final model to evaluate on the Japan full-data experiment",
        },
    ]
    df = pd.DataFrame(rows)
    save_df(df, out_dir / "literature_model_comparison")
    positioning_cards_png(df, out_dir / "literature_model_comparison.png")

    refs = [
        {
            "Key": "TEAM",
            "Reference": "Muenchmeyer, J., Bindi, D., Leser, U., & Tilmann, F. (2021). The transformer earthquake alerting model: a new versatile approach to earthquake early warning. Geophysical Journal International, 225(1), 646-656. https://doi.org/10.1093/gji/ggaa609",
        },
        {
            "Key": "QuakeFormer",
            "Reference": "Feng, Y., Zhu, W., & Lu, X. (2024). QuakeFormer: A Uniform Approach to Earthquake Ground Motion Prediction Using Masked Transformers. arXiv:2412.00815.",
        },
        {
            "Key": "GMPE",
            "Reference": "Boore, D. M., & Atkinson, G. M. (2008). Ground-motion prediction equations for the average horizontal component of PGA, PGV, and 5%-damped PSA. Earthquake Spectra, 24(1), 99-138. https://doi.org/10.1193/1.2830434",
        },
        {
            "Key": "Deep PGA",
            "Reference": "Liu, Y., Zhao, Q., & Wang, Y. (2024). Peak ground acceleration prediction for on-site earthquake early warning with deep learning. Scientific Reports, 14, 5485. https://doi.org/10.1038/s41598-024-56004-6",
        },
    ]
    ref_df = pd.DataFrame(refs)
    save_df(ref_df, out_dir / "references")


def write_literature_metric_comparison(metrics: pd.DataFrame, csv_path: Path | None, out_dir: Path) -> None:
    template_cols = ["model", "split", "mae", "rmse", "corr", "r2", "slope", "dataset", "note"]
    template_path = out_dir / "literature_metric_template.csv"
    if not template_path.exists():
        pd.DataFrame([
            {
                "model": "TEAM (reported or reproduced)",
                "split": "test",
                "mae": "",
                "rmse": "",
                "corr": "",
                "r2": "",
                "slope": "",
                "dataset": "Fill with comparable dataset/split",
                "note": "Only compare numerically when target definition and split are comparable.",
            },
            {
                "model": "QuakeFormer (reported or reproduced)",
                "split": "test",
                "mae": "",
                "rmse": "",
                "corr": "",
                "r2": "",
                "slope": "",
                "dataset": "Fill with comparable dataset/split",
                "note": "Use reproduced result if possible; paper tasks may differ.",
            },
        ], columns=template_cols).to_csv(template_path, index=False)

    preferred = metrics.copy()
    if "val" in set(preferred["split"]):
        best_model = preferred[preferred["split"] == "val"].sort_values(["mae", "model_id"]).iloc[0]["model_id"]
    else:
        best_model = preferred.sort_values(["mae", "model_id"]).iloc[0]["model_id"]
    project = preferred[preferred["model_id"] == best_model].copy()
    split_order = {"train": 0, "val": 1, "test": 2}
    project["_split_order"] = project["split"].map(split_order).fillna(99)
    project = project.sort_values(["_split_order", "split"]).drop(columns=["_split_order"])
    project = pd.DataFrame({
        "model": project["model"].astype(str) + " (this project)",
        "split": project["split"],
        "mae": project["mae"],
        "rmse": project["rmse"],
        "corr": project["corr"],
        "r2": project["r2"],
        "slope": project["slope"],
        "dataset": "Project evaluation split",
        "note": "Generated from eval_results.txt",
    })

    placeholder = pd.DataFrame([
        {
            "model": "TEAM",
            "split": "reported/reproduced",
            "mae": "fill",
            "rmse": "fill",
            "corr": "fill",
            "r2": "fill",
            "slope": "fill",
            "dataset": "Use same data/split when available",
            "note": "Fill from reproduced run or comparable paper setting.",
        },
        {
            "model": "QuakeFormer",
            "split": "reported/reproduced",
            "mae": "fill",
            "rmse": "fill",
            "corr": "fill",
            "r2": "fill",
            "slope": "fill",
            "dataset": "Use same data/split when available",
            "note": "Fill from reproduced run or comparable paper setting.",
        },
    ], columns=template_cols)

    frames = [placeholder, project]
    if csv_path is not None and csv_path.exists():
        external = pd.read_csv(csv_path)
        for col in template_cols:
            if col not in external.columns:
                external[col] = ""
        frames = [external[template_cols], project]

    comparison = pd.concat(frames, ignore_index=True)
    save_df(comparison, out_dir / "literature_metric_comparison")
    table_png(comparison, out_dir / "literature_metric_comparison.png", "Numerical Comparison Template", font_size=12)


def plot_attention_exports(search_dirs: Iterable[Path], out_dir: Path, max_cases: int = 3) -> list[Path]:
    """Plot saved PGA target cross-attention maps from eval_attention_*.npz."""
    outputs: list[Path] = []
    attention_paths = []
    for directory in search_dirs:
        if directory.exists():
            attention_paths.extend(sorted(directory.glob("eval_attention_*.npz")))
            attention_paths.extend(sorted(directory.glob("eval_attention.npz")))
    seen = set()
    attention_paths = [p for p in attention_paths if not (p in seen or seen.add(p))]

    for attention_path in attention_paths:
        tag = attention_path.stem.replace("eval_attention_", "")
        if tag == "eval_attention":
            tag = "attention"
        data = np.load(attention_path, allow_pickle=True)
        split_names = []
        for key in data.files:
            if key.endswith("_pga_attention"):
                split_names.append(key[: -len("_pga_attention")])
        for split in split_names:
            attn = np.asarray(data[f"{split}_pga_attention"], dtype=object)
            if attn.size == 0:
                continue
            n_cases = min(max_cases, len(attn))
            for case_idx in range(n_cases):
                station_valid = np.asarray(data[f"{split}_station_valid"][case_idx], dtype=bool)
                station_coords = np.asarray(data[f"{split}_station_coords_abs"][case_idx], dtype=float)
                target_coords = np.asarray(data[f"{split}_pga_target_abs"][case_idx], dtype=float)
                target_valid = np.asarray(data[f"{split}_pga_target_valid"][case_idx], dtype=bool).reshape(-1)
                residual = np.asarray(data[f"{split}_pga_residual"][case_idx], dtype=float).reshape(-1)
                event_coord = np.asarray(data[f"{split}_loc_label_abs"][case_idx], dtype=float).reshape(-1)
                attn_case = np.asarray(attn[case_idx], dtype=float)
                if attn_case.ndim != 2 or station_coords.ndim != 2 or target_coords.ndim != 2:
                    continue

                valid_station_coords = station_coords[station_valid]
                valid_attn = attn_case[target_valid][:, station_valid] if target_valid.any() else attn_case[:, station_valid]
                station_weight = valid_attn.mean(axis=0) if valid_attn.size else np.zeros(valid_station_coords.shape[0])
                station_weight = station_weight / max(float(station_weight.max()), 1e-12)
                valid_targets = target_coords[target_valid]
                valid_residual = residual[target_valid]
                if valid_targets.size == 0 or valid_station_coords.size == 0:
                    continue

                fig = plt.figure(figsize=(12, 9))
                ax = fig.add_subplot(1, 1, 1, **map_subplot_kw())
                target_lonlat = coords_to_lonlat(valid_targets)
                station_lonlat = coords_to_lonlat(valid_station_coords)
                event_lonlat = coords_to_lonlat(event_coord)
                focus_extent = draw_reference_map(ax, [target_lonlat, station_lonlat, event_lonlat])
                draw_japan_inset(ax, focus_extent)
                geo_kwargs = {"transform": ccrs.PlateCarree()} if is_cartopy_axis(ax) else {}
                vlim = max(float(np.nanmax(np.abs(valid_residual))), 0.1)
                sc = ax.scatter(
                    target_lonlat[:, 0],
                    target_lonlat[:, 1],
                    c=valid_residual,
                    s=90,
                    cmap="coolwarm",
                    vmin=-vlim,
                    vmax=vlim,
                    edgecolor="black",
                    linewidth=0.4,
                    label="PGA targets",
                    zorder=3,
                    **geo_kwargs,
                )
                ax.scatter(
                    station_lonlat[:, 0],
                    station_lonlat[:, 1],
                    s=70 + 360 * station_weight,
                    c=station_weight,
                    cmap="viridis",
                    marker="^",
                    edgecolor="black",
                    linewidth=0.5,
                    label="Input stations (mean attention)",
                    zorder=4,
                    **geo_kwargs,
                )
                ax.scatter([event_lonlat[0]], [event_lonlat[1]], marker="*", s=260, color="gold", edgecolor="black", label="Event", zorder=5, **geo_kwargs)
                event_idx = data[f"{split}_event_index"][case_idx] if f"{split}_event_index" in data else case_idx
                req = data[f"{split}_requested_station_count"][case_idx] if f"{split}_requested_station_count" in data else "config"
                ax.set_title(f"PGA Attention Map ({split}, {tag}, event {event_idx}, requested {req})", fontsize=22, weight="bold")
                ax.set_xlabel("Longitude", fontsize=18)
                ax.set_ylabel("Latitude", fontsize=18)
                ax.tick_params(labelsize=14)
                ax.legend(fontsize=13, loc="upper left")
                cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
                cbar.set_label("Residual", fontsize=15)
                cbar.ax.tick_params(labelsize=13)
                fig.tight_layout()
                out = out_dir / f"attention_map_{case_idx + 1}_{tag}_{split}.png"
                fig.savefig(out, dpi=220)
                plt.close(fig)
                outputs.append(out)

                valid_target_indices = np.where(target_valid)[0]
                if valid_target_indices.size == 0:
                    continue
                valid_station_attn = attn_case[:, station_valid]
                row_sums = np.clip(valid_station_attn.sum(axis=1, keepdims=True), 1e-12, None)
                normalized_attn = valid_station_attn / row_sums
                target_distances = horizontal_distances_km(target_coords, event_coord)
                valid_distances = target_distances[valid_target_indices]
                if valid_distances.size == 0:
                    continue
                quantiles = [0.2, 0.5, 0.85, 0.35, 0.7]
                desired_distance = float(np.quantile(valid_distances, quantiles[case_idx % len(quantiles)]))
                target_slot = int(valid_target_indices[np.argmin(np.abs(valid_distances - desired_distance))])
                target_weight = normalized_attn[target_slot]
                target_weight_scaled = target_weight / max(float(target_weight.max()), 1e-12)
                target_lonlat = coords_to_lonlat(target_coords)
                station_lonlat = coords_to_lonlat(valid_station_coords)
                event_lonlat = coords_to_lonlat(event_coord)

                fig = plt.figure(figsize=(9.5, 9))
                ax = fig.add_subplot(1, 1, 1, **map_subplot_kw())
                focus_extent = draw_reference_map(ax, [target_lonlat, station_lonlat, event_lonlat])
                draw_japan_inset(ax, focus_extent)
                geo_kwargs = {"transform": ccrs.PlateCarree()} if is_cartopy_axis(ax) else {}
                ax.scatter(
                    target_lonlat[target_valid, 0],
                    target_lonlat[target_valid, 1],
                    s=58,
                    color="#bdbdbd",
                    edgecolor="black",
                    linewidth=0.25,
                    alpha=0.7,
                    label="Other PGA targets",
                    zorder=2,
                    **geo_kwargs,
                )
                ax.scatter(
                    [target_lonlat[target_slot, 0]],
                    [target_lonlat[target_slot, 1]],
                    s=210,
                    facecolor="none",
                    edgecolor="#e41a1c",
                    linewidth=3.0,
                    label="Queried target station",
                    zorder=6,
                    **geo_kwargs,
                )
                st = ax.scatter(
                    station_lonlat[:, 0],
                    station_lonlat[:, 1],
                    s=90 + 520 * target_weight_scaled,
                    c=target_weight,
                    cmap="viridis",
                    vmin=0,
                    vmax=max(float(target_weight.max()), 1e-12),
                    marker="^",
                    edgecolor="black",
                    linewidth=0.6,
                    label="Input stations",
                    zorder=4,
                    **geo_kwargs,
                )
                ax.scatter([event_lonlat[0]], [event_lonlat[1]], marker="*", s=260, color="gold", edgecolor="black", label="Event", zorder=5, **geo_kwargs)
                top_order = np.argsort(-target_weight)[: min(3, len(target_weight))]
                for station_idx in top_order:
                    ax.plot(
                        [target_lonlat[target_slot, 0], station_lonlat[station_idx, 0]],
                        [target_lonlat[target_slot, 1], station_lonlat[station_idx, 1]],
                        color="#e41a1c",
                        linewidth=1.1 + 3.0 * float(target_weight_scaled[station_idx]),
                        alpha=0.55,
                        zorder=3,
                        **geo_kwargs,
                    )
                y_true = float(np.asarray(data[f"{split}_pga_label"][case_idx]).reshape(-1)[target_slot])
                y_pred = float(np.asarray(data[f"{split}_pga_mu_best"][case_idx]).reshape(-1)[target_slot])
                y_res = y_pred - y_true
                event_idx = data[f"{split}_event_index"][case_idx] if f"{split}_event_index" in data else case_idx
                req = data[f"{split}_requested_station_count"][case_idx] if f"{split}_requested_station_count" in data else "config"
                target_distance = float(target_distances[target_slot])
                ax.set_title("Target-station Attention", fontsize=22, weight="bold")
                ax.text(
                    0.98,
                    0.03,
                    f"{split}, {tag}, event {event_idx}, requested {req}\n"
                    f"target {target_slot}, dist {target_distance:.0f} km: true {y_true:.3f}, pred {y_pred:.3f}, res {y_res:.3f}",
                    transform=ax.transAxes,
                    fontsize=12,
                    ha="right",
                    va="bottom",
                    bbox=dict(facecolor="white", alpha=0.88, edgecolor="#cccccc"),
                )
                ax.set_xlabel("Longitude", fontsize=18)
                ax.set_ylabel("Latitude", fontsize=18)
                ax.tick_params(labelsize=14)
                ax.legend(fontsize=12, loc="upper left")
                cbar = fig.colorbar(st, ax=ax, fraction=0.045, pad=0.03)
                cbar.set_label("Attention weight for queried target", fontsize=15)
                cbar.ax.tick_params(labelsize=13)
                fig.tight_layout(rect=[0, 0, 0.98, 0.98])
                out = out_dir / f"attention_target_{case_idx + 1}_{tag}_{split}.png"
                fig.savefig(out, dpi=220)
                plt.close(fig)
                outputs.append(out)
    return outputs


def run_eval_if_missing(evals: list[EvalFile], args: argparse.Namespace) -> None:
    if not args.run_eval_missing:
        return
    for eval_file in evals:
        if eval_file.npz_path is not None:
            continue
        config = eval_file.result_dir / "config.json"
        checkpoint = eval_file.result_dir / "full_model_best.pth"
        tag = "best"
        if not checkpoint.exists():
            checkpoint = eval_file.result_dir / "full_model_last.pth"
            tag = "last"
        if not config.exists() or not checkpoint.exists():
            print(f"[WARN] Cannot run eval for {eval_file.result_dir}: missing config/checkpoint", file=sys.stderr)
            continue
        output = eval_file.result_dir / f"eval_results_{tag}.npz"
        txt = eval_file.result_dir / f"eval_results_{tag}.txt"
        cmd = [
            sys.executable,
            str(args.repo_root / "eval_checkpoint.py"),
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--device",
            args.device,
        ]
        if args.diting_config:
            cmd += ["--diting_config", args.diting_config]
        if args.diting_pretrained:
            cmd += ["--diting_pretrained", args.diting_pretrained]
        if args.overfit_n:
            cmd += ["--overfit_n", str(args.overfit_n)]
        print("[INFO] Running eval:", " ".join(cmd))
        with txt.open("w") as f:
            subprocess.run(cmd, cwd=args.repo_root, stdout=f, stderr=subprocess.STDOUT, check=True)


def choose_primary(evals: list[EvalFile], metrics: pd.DataFrame, primary: str | None) -> EvalFile | None:
    if primary:
        for eval_file in evals:
            if primary in eval_file.result_dir.name or primary == eval_file.model_id:
                return eval_file
    val = metrics[metrics["split"] == "val"].copy()
    if val.empty:
        return evals[0] if evals else None
    best = val.sort_values("mae").iloc[0]
    for eval_file in evals:
        if eval_file.model_id == best["model_id"] and eval_file.checkpoint_tag == best["checkpoint"]:
            return eval_file
    return evals[0] if evals else None


def write_manifest(out_dir: Path, evals: list[EvalFile], primary: EvalFile | None) -> None:
    lines = [
        "# PGA Report Assets",
        "",
        f"Generated assets directory: `{out_dir}`",
        "",
        "## Inputs",
        "",
    ]
    for eval_file in evals:
        npz = eval_file.npz_path if eval_file.npz_path else "<missing>"
        lines.append(f"- `{eval_file.txt_path}`; npz: `{npz}`")
    if primary:
        lines += [
            "",
            "## Primary Model",
            "",
            f"`{primary.result_dir.name}` checkpoint `{primary.checkpoint_tag}`",
        ]
    lines += [
        "",
        "## Main Figures",
        "",
        "- `main_pga_metrics.png`: table of train/validation PGA metrics.",
        "- `validation_metric_bars.png`: validation MAE/R2/slope comparison when multiple models are provided.",
        "- `validation_metric_summary.png`: compact validation table when only one model is provided.",
        "- `train_val_metric_bars.png`: train vs validation metrics.",
        "- `val_mae_by_station_count.png`: validation MAE by input station count.",
        "- `val_mae_by_epicentral_distance.png`: validation MAE by target epicentral distance when coordinate arrays are available.",
        "- `scatter_*_val.png`: predicted vs true PGA for the primary model.",
        "- `residual_vs_true_*_val.png`: residual diagnostics for the primary model.",
        "- `pga_strength_bins_*_val.png`: performance by true PGA strength.",
        "- `case_study_*.png`: event-level target, distance, and spatial residual diagnostics when coordinate arrays are available.",
        "- `case_station_sweep_*.png`: event-level error change across requested input-station counts when eval npz contains `case_sweep_*` arrays.",
        "- `loss_curve_*.png`: training and validation loss curve for the primary model.",
        "- `literature_model_comparison.png`: positioning relative to TEAM and QuakeFormer.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("chaosuan_res"))
    parser.add_argument("--pattern", default="weights_japan*_pga15*")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/pga_academic_report_assets"))
    parser.add_argument("--primary", default=None, help="Model id or result-dir substring for scatter/bin plots")
    parser.add_argument("--selected-models", default="", help="Comma-separated model ids for bucket comparison")
    parser.add_argument("--literature-metrics-csv", type=Path, default=None,
                        help="Optional CSV with TEAM/QuakeFormer reproduced or reported metrics.")
    parser.add_argument("--split-data-dir", type=Path, default=None,
                        help="Directory containing split_events.csv and split_stations.csv for data distribution plots.")
    parser.add_argument("--max-scatter-points", type=int, default=20000)
    parser.add_argument("--pga-bins", type=int, default=5)
    parser.add_argument("--distance-bins", default="0,50,100,200,400,800,inf",
                        help="Comma-separated epicentral-distance bin edges for NPZ coordinate outputs.")
    parser.add_argument("--case-studies", type=int, default=3,
                        help="Number of event-level case-study figures to generate for the primary model.")
    parser.add_argument("--run-eval-missing", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--diting-config", default=None)
    parser.add_argument("--diting-pretrained", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overfit-n", type=int, default=0)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    evals = discover_eval_files(args.results_root, args.pattern)
    if args.run_eval_missing:
        run_eval_if_missing(evals, args)
        evals = discover_eval_files(args.results_root, args.pattern)
    if not evals:
        raise SystemExit(f"No eval_results*.txt found under {args.results_root} with pattern {args.pattern!r}")

    metric_rows: list[dict] = []
    bucket_rows: list[dict] = []
    single_rows: list[dict] = []
    diag_rows: list[dict] = []
    for eval_file in evals:
        rows, buckets, singles, diags = parse_eval_txt(eval_file)
        metric_rows.extend(rows)
        bucket_rows.extend(buckets)
        single_rows.extend(singles)
        diag_rows.extend(diags)

    metrics = pd.DataFrame(metric_rows)
    buckets = pd.DataFrame(bucket_rows)
    singles = pd.DataFrame(single_rows)
    diagnostics = pd.DataFrame(diag_rows)

    if metrics.empty:
        raise SystemExit("No PGA metrics parsed from eval txt files.")

    split_data_dir = args.split_data_dir
    if split_data_dir is None and evals:
        for candidate in [evals[0].result_dir, args.results_root / args.pattern]:
            if (candidate / "split_events.csv").exists() and (candidate / "split_stations.csv").exists():
                split_data_dir = candidate
                break
    if split_data_dir is not None:
        plot_split_data_distributions(split_data_dir, args.output_dir)

    save_df(metrics, args.output_dir / "main_pga_metrics")
    save_df(buckets, args.output_dir / "station_count_buckets")
    if not singles.empty:
        save_df(singles, args.output_dir / "single_station_metrics")
    if not diagnostics.empty:
        save_df(diagnostics, args.output_dir / "diagnostics")

    table_cols = ["model", "checkpoint", "split", "mae", "rmse", "corr", "r2", "slope"]
    table = metrics[table_cols].sort_values(["split", "mae"])
    table_png(table, args.output_dir / "main_pga_metrics.png", "PGA Prediction Metrics", font_size=13)

    plot_summary_bars(metrics, args.output_dir)
    plot_train_val_metric(metrics, args.output_dir)
    selected = [s.strip() for s in args.selected_models.split(",") if s.strip()]
    if not selected:
        selected = readable_model_order(metrics)[:5]
    plot_station_buckets(buckets, args.output_dir, selected)
    distance_bins = parse_distance_bins(args.distance_bins)
    distance_bucket_df = plot_distance_buckets(evals, args.output_dir, selected, distance_bins)
    if distance_bucket_df.empty:
        print(
            "[WARN] Epicentral-distance bucket plot was skipped. "
            "Re-run eval_checkpoint.py with the updated code so eval_results.npz contains "
            "*_pga_target_abs and *_station_coords_abs.",
            file=sys.stderr,
        )

    rng = np.random.default_rng(42)
    primary = choose_primary(evals, metrics, args.primary)
    if primary is not None:
        plot_scatter_and_residual(primary, args.output_dir, args.max_scatter_points, rng)
        plot_pga_bins(primary, args.output_dir, "val", args.pga_bins)
        plot_pga_bins(primary, args.output_dir, "train", args.pga_bins)
        plot_loss_curves(primary, args.output_dir)
        plot_case_station_sweeps(primary, args.output_dir, "train", args.case_studies)
        plot_case_station_sweeps(primary, args.output_dir, "val", args.case_studies)
        plot_case_studies(primary, args.output_dir, "val", args.case_studies)
        plot_attention_exports([primary.result_dir], args.output_dir, max_cases=max(args.case_studies, 5))

    write_literature_tables(args.output_dir)
    write_literature_metric_comparison(metrics, args.literature_metrics_csv, args.output_dir)
    write_manifest(args.output_dir, evals, primary)
    print(f"[INFO] Wrote report assets to {args.output_dir}")


if __name__ == "__main__":
    main()
