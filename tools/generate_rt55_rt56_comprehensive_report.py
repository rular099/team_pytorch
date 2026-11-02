#!/usr/bin/env python3
"""Generate separate, reproducible RT55 and RT56 comprehensive result reports.

The reports intentionally keep RT55 standalone.  RT56 may use RT55 as a
reference, but strict comparisons are limited to identical evaluation
protocols.  Every plotted quantitative panel writes source data as CSV.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rt55-rt56-report")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Mandatory publication/export settings.  SVG labels remain editable.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update(
    {
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "lines.linewidth": 1.5,
        "savefig.facecolor": "white",
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.10,
        "figure.constrained_layout.w_pad": 0.08,
        "figure.constrained_layout.hspace": 0.08,
        "figure.constrained_layout.wspace": 0.05,
    }
)

PALETTE = {
    "rt55_dark": "#0F4D92",
    "rt55_mid": "#3775BA",
    "rt55_soft": "#B4CBE5",
    "zero": "#6B668F",
    "rt56": "#C65D3B",
    "rt56_soft": "#E9A58F",
    "gain": "#2E8B57",
    "drop": "#C23B3B",
    "neutral": "#767676",
    "light": "#D6D6D6",
    "dark": "#272727",
    "triggered": "#4C78A8",
    "untriggered": "#F28E2B",
    "weak": "#72B7B2",
    "strong": "#E45756",
}

SCATTER_DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "pga_density",
    ["#F7F9FB", "#DCE8F1", "#A9C9DE", "#65A0C8", "#286EA8", "#0F426F"],
)

TARGET_TYPE_NAMES = {-1: "invalid", 0: "input", 1: "triggered_noninput", 2: "untriggered"}
MAG_EDGES = [2.99, 3.5, 4.0, 4.5, 5.0, 6.0, np.inf]
MAG_LABELS = ["3.0–3.4", "3.5–3.9", "4.0–4.4", "4.5–4.9", "5.0–5.9", "≥6.0"]
STATION_ORDER = ["1", "2–3", "4–5", "6–10", "11–15", "≥16"]
LEAD_ORDER = ["≤0", "0–5", "5–20", ">20", "unknown"]
TIME_ORDER = [1, 3, 5, 10, 20, 40, 90]
THRESHOLD = -1.2
POS_OFFSET = (37.0, 140.0)

JAPAN_OUTLINES = [
    [(141.0, 45.5), (143.2, 44.7), (145.4, 43.3), (144.2, 42.1), (141.8, 41.4), (140.2, 42.2), (140.5, 44.2), (141.0, 45.5)],
    [(140.8, 41.2), (141.5, 39.5), (141.3, 38.2), (140.8, 37.2), (140.1, 36.1), (139.7, 35.5), (138.8, 34.9), (137.0, 35.2), (136.0, 35.6), (135.0, 35.4), (134.2, 34.6), (133.0, 34.4), (132.0, 34.2), (131.1, 34.4), (130.9, 35.0), (132.5, 35.8), (134.0, 36.2), (135.5, 36.9), (137.0, 37.5), (138.5, 38.6), (139.7, 40.0), (140.8, 41.2)],
    [(133.0, 34.4), (134.3, 34.3), (135.0, 33.9), (134.1, 33.3), (132.8, 33.4), (132.4, 33.9), (133.0, 34.4)],
    [(130.0, 33.9), (131.2, 33.6), (131.9, 32.7), (131.2, 31.5), (130.2, 31.1), (129.6, 32.1), (129.7, 33.0), (130.0, 33.9)],
    [(129.5, 28.5), (128.2, 27.4), (127.7, 26.4), (126.8, 26.0), (125.5, 24.8)],
]


@dataclass(frozen=True)
class EvalSpec:
    key: str
    label: str
    npz: Path
    metrics: Path
    split: str
    protocol: str
    color: str
    line_style: str = "-"


@dataclass
class EvalData:
    spec: EvalSpec
    raw: dict[str, np.ndarray]
    targets: pd.DataFrame
    formal: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("../chaosuan_res"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/rt55_rt56_current_results"),
    )
    parser.add_argument("--station-cache", type=Path, default=Path("resources/vs30/japan_vs30_jshis_station_cache.csv"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scatter", type=int, default=60000)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stack_object(value: np.ndarray, dtype=None) -> np.ndarray:
    if value.dtype == object:
        value = np.stack([np.asarray(item) for item in value])
    value = np.asarray(value)
    return value.astype(dtype) if dtype is not None else value


def normalize_event_ids(value: np.ndarray) -> np.ndarray:
    out = []
    for item in value.reshape(-1):
        arr = np.asarray(item).reshape(-1)
        out.append(str(arr[0]) if len(arr) else "unknown")
    return np.asarray(out, dtype=str)


def optional(raw_npz, key: str, dtype=None):
    if key not in raw_npz:
        return None
    return stack_object(raw_npz[key], dtype=dtype)


def requested_time(raw_npz, prefix: str) -> np.ndarray:
    key = f"{prefix}_realtime_requested_elapsed_time"
    if key in raw_npz:
        return stack_object(raw_npz[key], float).reshape(-1)
    return stack_object(raw_npz[f"{prefix}_realtime_elapsed_time"], float).reshape(-1)


def station_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
    values = np.asarray(values, dtype=float)
    labels = np.select(
        [values == 1, (values >= 2) & (values <= 3), (values >= 4) & (values <= 5),
         (values >= 6) & (values <= 10), (values >= 11) & (values <= 15), values >= 16],
        STATION_ORDER,
        default="unknown",
    )
    return pd.Categorical(labels, categories=STATION_ORDER, ordered=True)


def lead_bucket(values: np.ndarray) -> pd.Categorical:
    values = np.asarray(values, dtype=float)
    labels = np.full(values.shape, "unknown", dtype=object)
    finite = np.isfinite(values)
    labels[finite & (values <= 0)] = "≤0"
    labels[finite & (values > 0) & (values <= 5)] = "0–5"
    labels[finite & (values > 5) & (values <= 20)] = "5–20"
    labels[finite & (values > 20)] = ">20"
    return pd.Categorical(labels, categories=LEAD_ORDER, ordered=True)


def magnitude_bucket(values: np.ndarray) -> pd.Categorical:
    return pd.cut(values, MAG_EDGES, labels=MAG_LABELS, right=False, ordered=True)


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.asarray, (lat1, lon1, lat2, lon2))
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = p2 - p1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def mean_over_valid_stations(values: np.ndarray | None, station_valid: np.ndarray | None) -> np.ndarray:
    if values is None or station_valid is None:
        return np.full(0, np.nan)
    values = np.asarray(values, dtype=float)
    valid = np.asarray(station_valid, dtype=bool)
    masked = np.where(valid, values, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(masked, axis=1)


def flatten_eval(spec: EvalSpec) -> EvalData:
    with np.load(spec.npz, allow_pickle=True) as z:
        prefix = spec.split
        labels = stack_object(z[f"{prefix}_pga_label"], float).reshape(len(z[f"{prefix}_pga_label"]), -1)
        pred = stack_object(z[f"{prefix}_pga_mu_best"], float).reshape(labels.shape)
        sigma = stack_object(z[f"{prefix}_pga_sigma"], float).reshape(labels.shape)
        valid = stack_object(z[f"{prefix}_pga_target_valid"], bool).reshape(labels.shape)
        nll = optional(z, f"{prefix}_pga_nll_log10_mps2", float)
        prob = optional(z, f"{prefix}_pga_prob_ge_threshold", float)
        target_type = optional(z, f"{prefix}_realtime_target_type", int)
        lead = optional(z, f"{prefix}_realtime_target_lead_time", float)
        target_abs = optional(z, f"{prefix}_pga_target_abs", float)
        loc_abs = optional(z, f"{prefix}_loc_label_abs", float)
        station_valid = optional(z, f"{prefix}_station_valid", bool)
        station_coords = optional(z, f"{prefix}_station_coords_abs", float)
        waveform_seconds = optional(z, f"{prefix}_waveform_valid_seconds", float)
        post_p_seconds = optional(z, f"{prefix}_waveform_post_p_valid_seconds", float)
        sample_count, target_slots = valid.shape
        event_ids = normalize_event_ids(z[f"{prefix}_event_id"])
        event_index = stack_object(z[f"{prefix}_event_index"]).reshape(-1).astype(int)
        magnitude = stack_object(z[f"{prefix}_mag_label"], float).reshape(-1)
        station_count = stack_object(z[f"{prefix}_station_valid_count"], int).reshape(-1)
        current_time = requested_time(z, prefix)
        requested_k = optional(z, f"{prefix}_causal_random_requested_station_count", int)
        selected_k = optional(z, f"{prefix}_causal_random_selected_station_count", int)
        available_k = optional(z, f"{prefix}_causal_random_available_station_count", int)
        retained_suffixes = (
            "_station_coords_abs", "_station_valid", "_realtime_requested_elapsed_time",
            "_causal_random_requested_station_count", "_causal_random_selected_station_count",
            "_causal_random_available_station_count",
        )
        raw = {key: stack_object(z[key]) for key in z.files if key.endswith(retained_suffixes)}

    rows, slots = np.where(valid)
    df = pd.DataFrame(
        {
            "run": spec.key,
            "sample_index": rows,
            "target_slot": slots,
            "event_id": event_ids[rows],
            "event_index": event_index[rows],
            "magnitude": magnitude[rows],
            "true_pga": labels[rows, slots],
            "pred_pga": pred[rows, slots],
            "sigma": sigma[rows, slots],
            "station_count": station_count[rows],
            "current_time_s": current_time[rows],
        }
    )
    df["error"] = df["pred_pga"] - df["true_pga"]
    df["abs_error"] = df["error"].abs()
    df["squared_error"] = df["error"] ** 2
    df["strong"] = df["true_pga"] >= THRESHOLD
    df["pga_strength"] = np.where(df["strong"], "strong", "weak")
    df["station_bucket"] = station_bucket(df["station_count"])
    df["magnitude_bucket"] = magnitude_bucket(df["magnitude"])
    if nll is not None:
        df["nll"] = nll.reshape(valid.shape)[rows, slots]
    if prob is not None:
        df["prob_strong"] = prob.reshape(valid.shape)[rows, slots]
    if target_type is not None:
        codes = target_type.reshape(valid.shape)[rows, slots]
        df["target_type_code"] = codes
        df["target_type"] = [TARGET_TYPE_NAMES.get(int(code), f"code_{code}") for code in codes]
    else:
        df["target_type_code"] = -1
        df["target_type"] = "unknown"
    if lead is not None:
        lead_values = lead.reshape(valid.shape)[rows, slots]
        df["lead_time_s"] = lead_values
        df["lead_bucket"] = lead_bucket(lead_values)
    if requested_k is not None:
        df["requested_station_count"] = requested_k.reshape(-1)[rows]
    if selected_k is not None:
        df["selected_station_count"] = selected_k.reshape(-1)[rows]
    if available_k is not None:
        df["available_station_count"] = available_k.reshape(-1)[rows]
    if loc_abs is not None and target_abs is not None:
        event = loc_abs[rows]
        target = target_abs[rows, slots]
        ev_lat, ev_lon = event[:, 0] + POS_OFFSET[0], event[:, 1] + POS_OFFSET[1]
        ta_lat, ta_lon = target[:, 0] + POS_OFFSET[0], target[:, 1] + POS_OFFSET[1]
        df["event_lat"] = ev_lat
        df["event_lon"] = ev_lon
        df["event_depth_km"] = event[:, 2]
        df["target_lat"] = ta_lat
        df["target_lon"] = ta_lon
        df["epicentral_distance_km"] = haversine_km(ev_lat, ev_lon, ta_lat, ta_lon)
        if station_coords is not None and station_valid is not None:
            nearest_km = np.empty(len(rows), dtype=float)
            nearest_deg = np.empty(len(rows), dtype=float)
            chunk = 20000
            for start in range(0, len(rows), chunk):
                stop = min(start + chunk, len(rows))
                rr = rows[start:stop]
                ss = slots[start:stop]
                target_xy = target_abs[rr, ss, :2]
                stations_xy = station_coords[rr, :, :2]
                station_mask = station_valid[rr]
                delta = np.sqrt(np.sum((stations_xy - target_xy[:, None, :]) ** 2, axis=2))
                delta[~station_mask] = np.inf
                nearest_deg[start:stop] = delta.min(axis=1)
                nearest_km[start:stop] = np.min(
                    np.where(
                        station_mask,
                        haversine_km(
                            target_xy[:, None, 0] + POS_OFFSET[0],
                            target_xy[:, None, 1] + POS_OFFSET[1],
                            stations_xy[:, :, 0] + POS_OFFSET[0],
                            stations_xy[:, :, 1] + POS_OFFSET[1],
                        ),
                        np.inf,
                    ),
                    axis=1,
                )
            df["nearest_input_coord_delta_deg"] = nearest_deg
            df["nearest_input_distance_km"] = nearest_km
    wave_mean = mean_over_valid_stations(waveform_seconds, station_valid)
    post_mean = mean_over_valid_stations(post_p_seconds, station_valid)
    if len(wave_mean):
        df["waveform_valid_seconds_mean"] = wave_mean[rows]
    if len(post_mean):
        df["post_p_valid_seconds_mean"] = post_mean[rows]
    formal = json.loads(spec.metrics.read_text())
    return EvalData(spec=spec, raw=raw, targets=df, formal=formal)


def subset_targets(df: pd.DataFrame, group: str) -> pd.DataFrame:
    if group == "all":
        return df
    if group == "noninput":
        return df[df["target_type_code"].isin([1, 2])]
    if group == "input":
        return df[df["target_type_code"] == 0]
    if group == "triggered_noninput":
        return df[df["target_type_code"] == 1]
    if group == "untriggered":
        return df[df["target_type_code"] == 2]
    if group == "weak":
        return df[~df["strong"]]
    if group == "strong":
        return df[df["strong"]]
    raise ValueError(group)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = stats.rankdata(scores, method="average")
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    if labels.sum() == 0:
        return np.nan
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    y = labels[order]
    precision = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float(precision[y].mean())


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = len(probs)
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if mask.any():
            value += mask.mean() * abs(probs[mask].mean() - labels[mask].mean())
    return float(value) if total else np.nan


def compute_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    if df.empty:
        return {"targets": 0, "events": 0}
    y = df["true_pga"].to_numpy(float)
    p = df["pred_pga"].to_numpy(float)
    err = p - y
    abs_err = np.abs(err)
    sse = float(np.sum(err ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    if len(y) > 1 and np.std(y) > 0 and np.std(p) > 0:
        corr = float(np.corrcoef(y, p)[0, 1])
        slope, intercept = map(float, np.polyfit(y, p, 1))
    else:
        corr = slope = intercept = np.nan
    out: dict[str, float | int] = {
        "targets": int(len(df)),
        "events": int(df["event_id"].nunique()),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - sse / sst) if sst > 0 else np.nan,
        "correlation": corr,
        "slope": slope,
        "intercept": intercept,
        "bias": float(err.mean()),
        "abs_error_p50": float(np.quantile(abs_err, 0.50)),
        "abs_error_p90": float(np.quantile(abs_err, 0.90)),
        "abs_error_p95": float(np.quantile(abs_err, 0.95)),
        "abs_error_gt_0p3": float(np.mean(abs_err > 0.3)),
        "abs_error_gt_0p5": float(np.mean(abs_err > 0.5)),
        "predictive_sigma_mean": float(df["sigma"].mean()),
        "predictive_sigma_median": float(df["sigma"].median()),
        "coverage_1sigma": float(np.mean(abs_err <= df["sigma"].to_numpy(float))),
        "coverage_2sigma": float(np.mean(abs_err <= 2 * df["sigma"].to_numpy(float))),
    }
    if "nll" in df:
        out["nll"] = float(df["nll"].mean())
    if "prob_strong" in df:
        labels = df["strong"].to_numpy(bool)
        probs = df["prob_strong"].to_numpy(float)
        positive_rate = float(labels.mean())
        brier = float(np.mean((probs - labels) ** 2))
        climatology = positive_rate * (1 - positive_rate)
        out.update(
            {
                "brier": brier,
                "brier_skill": float(1 - brier / climatology) if climatology > 0 else np.nan,
                "positive_rate": positive_rate,
                "probability_mean": float(probs.mean()),
                "roc_auc": roc_auc(labels, probs),
                "pr_auc": average_precision(labels, probs),
                "ece_10bin": expected_calibration_error(labels, probs, bins=10),
            }
        )
    per_event = df.groupby("event_id", observed=True).agg(
        mae=("abs_error", "mean"), mse=("squared_error", "mean"), bias=("error", "mean")
    )
    out["event_balanced_mae"] = float(per_event["mae"].mean())
    out["event_balanced_rmse"] = float(np.mean(np.sqrt(per_event["mse"])))
    out["event_balanced_bias"] = float(per_event["bias"].mean())
    return out


def formal_pga_metrics(eval_data: EvalData) -> dict:
    return eval_data.formal["metrics"][eval_data.spec.split]["pga"]


def export_table(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def save_figure(fig: plt.Figure, stem: Path, source: pd.DataFrame | None = None) -> list[Path]:
    ensure_dir(stem.parent)
    fig.align_labels()
    saved = []
    for ext, kwargs in [
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 300}),
    ]:
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04, **kwargs)
        saved.append(path)
    plt.close(fig)
    if source is not None:
        export_table(source, stem.with_suffix(".csv"))
    return saved


def panel_label(ax, label: str) -> None:
    ax.text(-0.10, 1.03, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def draw_japan_outline(ax) -> None:
    for outline in JAPAN_OUTLINES:
        xy = np.asarray(outline)
        ax.plot(xy[:, 0], xy[:, 1], color="#555555", linewidth=0.55, zorder=1)
    ax.set_xlim(124, 147)
    ax.set_ylim(24, 46)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")


def specs(results_root: Path) -> tuple[list[EvalSpec], Path]:
    rt55 = results_root / "weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42"
    rt56 = results_root / "weights_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42"
    out = [
        EvalSpec("rt55_ep20_val", "RT55 ep20 validation", rt55 / "eval_validation_best_normal.npz", rt55 / "eval_validation_best_normal.metrics.json", "val", "normal", PALETTE["rt55_soft"], "--"),
        EvalSpec("rt55_ep20_test", "RT55 ep20 test", rt55 / "formal_test_ep32_retry_20260824/eval_test_best_normal.npz", rt55 / "formal_test_ep32_retry_20260824/eval_test_best_normal.metrics.json", "test", "normal", PALETTE["rt55_dark"]),
        EvalSpec("rt55_ep32_val", "RT55 ep32 validation", rt55 / "eval_ep32/eval_validation_best_normal.npz", rt55 / "eval_ep32/eval_validation_best_normal.metrics.json", "val", "normal", PALETTE["rt55_mid"]),
        EvalSpec("rt55_ep32_roll", "RT55 ep32 waveform roll", rt55 / "eval_ep32/eval_validation_best_waveform_station_roll.npz", rt55 / "eval_ep32/eval_validation_best_waveform_station_roll.metrics.json", "val", "roll", PALETTE["neutral"], ":"),
        EvalSpec("rt56_zero_random", "RT55 ep32 zero-shot random", rt56 / "zero_shot_ep32/eval_validation_ep32_zero_shot_random_mask.npz", rt56 / "zero_shot_ep32/eval_validation_ep32_zero_shot_random_mask.metrics.json", "val", "random", PALETTE["zero"], "--"),
        EvalSpec("rt56_fine_random", "RT56 ep6 fine-tuned random", rt56 / "finetuned_best_random_geometry/eval_validation_finetuned_best_random_geometry.npz", rt56 / "finetuned_best_random_geometry/eval_validation_finetuned_best_random_geometry.metrics.json", "val", "random", PALETTE["rt56"]),
        EvalSpec("rt56_fine_normal", "RT56 ep6 normal", rt56 / "finetuned_best_rt55_normal/eval_validation_best_normal.npz", rt56 / "finetuned_best_rt55_normal/eval_validation_best_normal.metrics.json", "val", "normal", PALETTE["rt56_soft"], "-."),
    ]
    for spec in out:
        for path in (spec.npz, spec.metrics):
            if not path.is_file():
                raise FileNotFoundError(path)
    return out, rt55


def load_split_metadata(rt55_root: Path, station_cache_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(rt55_root / "split_events.csv")
    stations = pd.read_csv(rt55_root / "split_stations.csv", low_memory=False)
    cache = pd.read_csv(station_cache_path)
    events["year"] = pd.to_datetime(events["Origin_Time(JST)"], errors="coerce").dt.year
    stations["year"] = pd.to_datetime(stations["Origin_Time(JST)"], errors="coerce").dt.year
    events["split"] = pd.Categorical(events["split"], ["train", "dev", "test"], ordered=True)
    stations["split"] = pd.Categorical(stations["split"], ["train", "dev", "test"], ordered=True)
    return events, stations, cache


def aggregate_metrics(evals: dict[str, EvalData]) -> pd.DataFrame:
    rows = []
    for key, data in evals.items():
        for group in ["all", "noninput", "input", "triggered_noninput", "untriggered", "weak", "strong"]:
            values = compute_metrics(subset_targets(data.targets, group))
            rows.append({"run": key, "label": data.spec.label, "protocol": data.spec.protocol, "group": group, **values})
    return pd.DataFrame(rows)


def stratified_metrics(
    evals: dict[str, EvalData],
    keys: Sequence[str],
    column: str,
    order: Sequence | None = None,
    target_group: str = "all",
) -> pd.DataFrame:
    rows = []
    for key in keys:
        df = subset_targets(evals[key].targets, target_group)
        if column not in df:
            continue
        for value, part in df.groupby(column, observed=False, dropna=False):
            if len(part) == 0:
                continue
            label = "unknown" if pd.isna(value) else str(value)
            rows.append({"run": key, "label": evals[key].spec.label, "group": target_group, "stratum": label, **compute_metrics(part)})
    out = pd.DataFrame(rows)
    if not out.empty and order is not None:
        rank = {str(value): index for index, value in enumerate(order)}
        out["stratum_order"] = out["stratum"].map(rank).fillna(len(rank)).astype(int)
        out = out.sort_values(["run", "stratum_order"]).reset_index(drop=True)
    return out


def add_distance_buckets(evals: dict[str, EvalData]) -> None:
    distance_edges = [-np.inf, 25, 50, 100, 200, 400, np.inf]
    distance_labels = ["<25", "25–50", "50–100", "100–200", "200–400", "≥400"]
    nearest_edges = [-np.inf, 25, 50, 100, 200, np.inf]
    nearest_labels = ["<25", "25–50", "50–100", "100–200", "≥200"]
    depth_edges = [-np.inf, 10, 30, 60, 100, np.inf]
    depth_labels = ["<10", "10–30", "30–60", "60–100", "≥100"]
    for data in evals.values():
        df = data.targets
        df["time_bucket"] = pd.Categorical(
            df["current_time_s"].round().astype(int).astype(str),
            categories=[str(value) for value in TIME_ORDER],
            ordered=True,
        )
        if "epicentral_distance_km" in df:
            df["distance_bucket"] = pd.cut(df["epicentral_distance_km"], distance_edges, labels=distance_labels, right=False, ordered=True)
        if "nearest_input_distance_km" in df:
            df["nearest_input_bucket"] = pd.cut(df["nearest_input_distance_km"], nearest_edges, labels=nearest_labels, right=False, ordered=True)
        if "event_depth_km" in df:
            df["depth_bucket"] = pd.cut(df["event_depth_km"], depth_edges, labels=depth_labels, right=False, ordered=True)
        if "waveform_valid_seconds_mean" in df:
            df["waveform_bucket"] = pd.cut(
                df["waveform_valid_seconds_mean"],
                [-np.inf, 1, 3, 5, 10, 20, np.inf],
                labels=["<1", "1–3", "3–5", "5–10", "10–20", "≥20"],
                right=False,
                ordered=True,
            )


def audit_formal_metrics(evals: dict[str, EvalData]) -> pd.DataFrame:
    mapping = {
        "mae": "mae", "rmse": "rmse", "r2": "r2", "nll": "nll_log10_mps2",
        "brier": "brier", "bias": "bias", "correlation": "correlation", "slope": "slope",
        "intercept": "intercept", "predictive_sigma_mean": "predictive_sigma_mean",
        "predictive_sigma_median": "predictive_sigma_median", "coverage_1sigma": "coverage_1sigma",
        "coverage_2sigma": "coverage_2sigma", "targets": "targets", "events": "events",
    }
    rows = []
    for key, data in evals.items():
        calculated = compute_metrics(data.targets)
        formal = formal_pga_metrics(data)
        for ours, theirs in mapping.items():
            if ours not in calculated or theirs not in formal:
                continue
            difference = float(calculated[ours]) - float(formal[theirs])
            rows.append({"run": key, "metric": ours, "recomputed": calculated[ours], "formal": formal[theirs], "difference": difference, "pass": abs(difference) <= 2e-6})
    audit = pd.DataFrame(rows)
    if not bool(audit["pass"].all()):
        failed = audit.loc[~audit["pass"]]
        raise AssertionError(f"Formal metric parity failed:\n{failed.to_string(index=False)}")
    return audit


def checkpoint_provenance(evals: dict[str, EvalData]) -> pd.DataFrame:
    rows = []
    for key, data in evals.items():
        meta = data.formal.get("checkpoint_metadata", {})
        rows.append(
            {
                "run": key,
                "label": data.spec.label,
                "split": data.spec.split,
                "protocol": data.spec.protocol,
                "checkpoint": data.formal.get("checkpoint", ""),
                "epoch": meta.get("epoch", np.nan),
                "checkpoint_loss": meta.get("loss", np.nan),
                "npz": str(data.spec.npz),
                "metrics_json": str(data.spec.metrics),
            }
        )
    return pd.DataFrame(rows)


def coordinate_exclusion_audit(evals: dict[str, EvalData]) -> pd.DataFrame:
    rows = []
    for key in ["rt56_zero_random", "rt56_fine_random"]:
        df = evals[key].targets
        delta = df["nearest_input_coord_delta_deg"].to_numpy(float)
        distance = df["nearest_input_distance_km"].to_numpy(float)
        rows.append(
            {
                "run": key,
                "targets": len(df),
                "overlap_targets_tol_1e-6": int(np.sum(delta <= 1e-6)),
                "overlap_fraction_tol_1e-6": float(np.mean(delta <= 1e-6)),
                "minimum_coord_delta_deg": float(np.min(delta)),
                "p01_coord_delta_deg": float(np.quantile(delta, 0.01)),
                "minimum_geodesic_distance_km": float(np.min(distance)),
                "p01_geodesic_distance_km": float(np.quantile(distance, 0.01)),
                "audit_basis": "target and valid selected-input coordinates",
            }
        )
    return pd.DataFrame(rows)


def paired_target_frame(before: EvalData, after: EvalData) -> pd.DataFrame:
    keys = ["sample_index", "target_slot", "event_id", "true_pga", "magnitude", "current_time_s", "target_type", "lead_bucket", "strong", "station_count"]
    left = before.targets[keys + ["pred_pga", "abs_error", "error"]].copy()
    right = after.targets[["sample_index", "target_slot", "true_pga", "pred_pga", "abs_error", "error"]].copy()
    paired = left.merge(right, on=["sample_index", "target_slot"], suffixes=("_before", "_after"), validate="one_to_one")
    if not np.allclose(paired["true_pga_before"], paired["true_pga_after"], atol=1e-7):
        raise AssertionError("Paired evaluation targets differ")
    paired["true_pga"] = paired.pop("true_pga_before")
    paired = paired.drop(columns="true_pga_after")
    paired["delta_abs_error"] = paired["abs_error_after"] - paired["abs_error_before"]
    paired["improvement_abs_error"] = -paired["delta_abs_error"]
    return paired


def event_bootstrap_delta(paired: pd.DataFrame, bootstrap: int, seed: int) -> pd.DataFrame:
    by_event = paired.groupby("event_id", observed=True).agg(
        before_mae=("abs_error_before", "mean"),
        after_mae=("abs_error_after", "mean"),
        delta_mae=("delta_abs_error", "mean"),
        targets=("target_slot", "size"),
    ).reset_index()
    rng = np.random.default_rng(seed)
    deltas = by_event["delta_mae"].to_numpy(float)
    draws = rng.choice(deltas, size=(bootstrap, len(deltas)), replace=True).mean(axis=1)
    summary = pd.DataFrame(
        [{
            "events": len(by_event),
            "targets": len(paired),
            "event_balanced_before_mae": by_event["before_mae"].mean(),
            "event_balanced_after_mae": by_event["after_mae"].mean(),
            "event_balanced_delta_mae": deltas.mean(),
            "bootstrap_ci95_low": np.quantile(draws, 0.025),
            "bootstrap_ci95_high": np.quantile(draws, 0.975),
            "bootstrap_replicates": bootstrap,
            "bootstrap_seed": seed,
        }]
    )
    return pd.concat([summary, pd.DataFrame([{"events": np.nan}])], ignore_index=True).iloc[:1], by_event


def distribution_summary(events: pd.DataFrame, stations: pd.DataFrame, cache: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_rows = []
    for split, part in events.groupby("split", observed=True):
        event_rows.append(
            {
                "split": str(split), "events": len(part), "magnitude_min": part["Magnitude"].min(),
                "magnitude_median": part["Magnitude"].median(), "magnitude_max": part["Magnitude"].max(),
                "depth_km_median": part["DEPTH"].median(), "stations_per_event_median": part["n_station_rows"].median(),
                "year_min": part["year"].min(), "year_max": part["year"].max(),
            }
        )
    station_rows = []
    cache_codes = set(cache["station_code"].astype(str))
    for split, part in stations.groupby("split", observed=True):
        codes = part["station_code"].astype(str)
        station_rows.append(
            {
                "split": str(split), "station_rows": len(part), "unique_stations": codes.nunique(),
                "coordinate_cache_rows": int(codes.isin(cache_codes).sum()),
                "coordinate_cache_coverage": float(codes.isin(cache_codes).mean()),
            }
        )
    mag = events.copy()
    mag["magnitude_bucket"] = magnitude_bucket(mag["Magnitude"])
    mag_table = mag.groupby(["split", "magnitude_bucket"], observed=False).size().rename("events").reset_index()
    return pd.DataFrame(event_rows), pd.DataFrame(station_rows), mag_table


def metric_lookup(metrics: pd.DataFrame, run: str, group: str, name: str) -> float:
    return float(metrics.loc[(metrics["run"] == run) & (metrics["group"] == group), name].iloc[0])


def percent_change(before: float, after: float) -> float:
    return 100.0 * (after - before) / before


def compact_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed)


def markdown_table(df: pd.DataFrame, columns: Sequence[str] | None = None, float_digits: int = 4) -> str:
    view = df.loc[:, list(columns)] if columns is not None else df.copy()
    formatted = []
    for _, row in view.iterrows():
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("—" if not np.isfinite(value) else f"{value:.{float_digits}f}")
            else:
                values.append(str(value))
        formatted.append(values)
    headers = [str(value) for value in view.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(values) + " |" for values in formatted)
    return "\n".join(lines)


def plot_training_curve(root: Path, stem: Path, title: str, marker_epochs: Sequence[int] = ()) -> None:
    train = pd.read_csv(root / "train_epoch_loss.csv").rename(columns={"step": "epoch", "value": "loss"})
    val = pd.read_csv(root / "val_epoch_loss.csv").rename(columns={"step": "epoch", "value": "loss"})
    train["series"] = "train"
    val["series"] = "validation"
    source = pd.concat([train, val], ignore_index=True)
    fig, ax = plt.subplots(figsize=(3.55, 2.45))
    for name, part, color in [("Train", train, PALETTE["rt55_mid"]), ("Validation", val, PALETTE["rt56"])]:
        ax.plot(part["epoch"], part["loss"], marker="o", markersize=3, label=name, color=color)
    for epoch in marker_epochs:
        ax.axvline(epoch, color=PALETTE["neutral"], linewidth=0.8, linestyle=":" )
        ax.text(epoch, ax.get_ylim()[1], f"ep{epoch}", ha="center", va="bottom", fontsize=6.5)
    ax.set(xlabel="Epoch", ylabel="Objective loss", title=title)
    ax.legend(ncol=2)
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.6)
    save_figure(fig, stem, source)


def plot_rt55_data_distribution(events: pd.DataFrame, stations: pd.DataFrame, cache: pd.DataFrame, stem: Path) -> None:
    colors = {"train": PALETTE["rt55_dark"], "dev": PALETTE["rt56"], "test": PALETTE["gain"]}
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.7))
    source_rows = []
    counts = events.groupby("split", observed=True).size()
    axes[0, 0].bar(counts.index.astype(str), counts.values, color=[colors[str(x)] for x in counts.index])
    for x, y in enumerate(counts.values):
        axes[0, 0].text(x, y, f"{y:,}", ha="center", va="bottom", fontsize=6.5)
    axes[0, 0].set(ylabel="Events", title="Event split size")
    for split, part in events.groupby("split", observed=True):
        label = str(split)
        axes[0, 1].hist(part["Magnitude"], bins=np.arange(3, 9.25, 0.25), density=True, histtype="step", linewidth=1.4, color=colors[label], label=label)
        values = np.sort(part["DEPTH"].dropna().to_numpy())
        axes[0, 2].plot(values, np.arange(1, len(values) + 1) / len(values), color=colors[label], label=label)
        for _, row in part[["EVENT", "Magnitude", "DEPTH", "n_station_rows", "year", "Latitude", "Longitude"]].iterrows():
            source_rows.append({"panel": "event_metadata", "split": label, **row.to_dict()})
    axes[0, 1].set(xlabel="Magnitude", ylabel="Density", title="Magnitude distribution")
    axes[0, 2].set(xlabel="Depth (km)", ylabel="Empirical CDF", title="Depth distribution", xlim=(0, 300))
    axes[0, 1].legend()
    station_event = events[["split", "n_station_rows"]].copy()
    sns.boxplot(data=station_event, x="split", y="n_station_rows", hue="split", palette=colors, legend=False, showfliers=False, ax=axes[1, 0])
    axes[1, 0].set(ylabel="Station rows per event", xlabel="", title="Network observations per event", ylim=(0, np.quantile(events["n_station_rows"], 0.99)))
    yearly = events.groupby(["year", "split"], observed=True).size().rename("events").reset_index()
    for split, part in yearly.groupby("split", observed=True):
        axes[1, 1].plot(part["year"], part["events"], color=colors[str(split)], label=str(split))
    axes[1, 1].set(xlabel="Year", ylabel="Events", title="Temporal coverage")
    coord_cache = cache.dropna(subset=["station_lat", "station_lon"])
    axes[1, 2].scatter(coord_cache["station_lon"], coord_cache["station_lat"], s=1.5, color="#BDBDBD", alpha=0.5, rasterized=True, label="station cache")
    for split in ["dev", "test"]:
        part = events[events["split"].astype(str) == split]
        axes[1, 2].scatter(part["Longitude"], part["Latitude"], s=5, color=colors[split], alpha=0.55, rasterized=True, label=f"{split} events")
    draw_japan_outline(axes[1, 2])
    axes[1, 2].set_title("Event and station coverage")
    axes[1, 2].legend(loc="lower right", fontsize=5.8)
    for label, ax in zip("ABCDEF", axes.flat):
        panel_label(ax, label)
    source = pd.concat(
        [pd.DataFrame(source_rows), yearly.assign(panel="yearly"), coord_cache[["station_code", "station_lat", "station_lon"]].assign(panel="station_cache")],
        ignore_index=True,
        sort=False,
    )
    save_figure(fig, stem, source)


def plot_target_pga_distribution(evals: dict[str, EvalData], keys: Sequence[str], stem: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55), sharex=True)
    source = []
    for ax, group in zip(axes, ["all", "noninput"]):
        for key in keys:
            data = evals[key]
            part = subset_targets(data.targets, group)
            values = part["true_pga"].to_numpy(float)
            hist, edges = np.histogram(values, bins=np.linspace(-4.0, 1.0, 61), density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            ax.plot(centers, hist, label=data.spec.label, color=data.spec.color, linestyle=data.spec.line_style)
            source.append(pd.DataFrame({"run": key, "group": group, "bin_center": centers, "density": hist, "targets": len(part)}))
        ax.axvline(THRESHOLD, color=PALETTE["strong"], linestyle=":", linewidth=1)
        ax.set(xlabel="True PGA [log10(m/s²)]", ylabel="Density", title="All targets" if group == "all" else "Non-input targets only")
    axes[0].legend(fontsize=5.7)
    fig.suptitle(title, y=1.01, fontsize=9)
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    save_figure(fig, stem, pd.concat(source, ignore_index=True))


def plot_metric_dashboard(metrics: pd.DataFrame, keys: Sequence[str], groups: Sequence[str], labels: dict[str, str], colors: dict[str, str], stem: Path, title: str) -> None:
    names = [("mae", "MAE (dex)"), ("rmse", "RMSE (dex)"), ("nll", "NLL"), ("brier", "Brier")]
    view = metrics[metrics["run"].isin(keys) & metrics["group"].isin(groups)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.5))
    width = 0.34
    x = np.arange(len(keys))
    for ax, (metric, ylabel), letter in zip(axes.flat, names, "ABCD"):
        for offset, group in enumerate(groups):
            values = [float(view.loc[(view["run"] == key) & (view["group"] == group), metric].iloc[0]) for key in keys]
            xpos = x + (offset - (len(groups) - 1) / 2) * width
            fill = [colors[key] for key in keys]
            bars = ax.bar(xpos, values, width=width, color=fill, alpha=1.0 if group == "all" else 0.48, hatch=None if group == "all" else "///", label="All targets" if group == "all" else "Non-input only")
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=5.8)
        ax.set_xticks(x, [labels[key] for key in keys], rotation=12, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.6)
        panel_label(ax, letter)
    axes[0, 0].legend(ncol=2, fontsize=6)
    fig.suptitle(title, y=1.01, fontsize=9)
    save_figure(fig, stem, view)


def plot_scatter_grid(evals: dict[str, EvalData], panels: Sequence[tuple[str, str, str]], stem: Path, seed: int, max_scatter: int, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 6.5), sharex=True, sharey=True)
    fig.get_layout_engine().set(rect=(0, 0.035, 1, 0.93))
    source = []
    hexbins = []
    for ax, (key, group, panel_title), letter in zip(axes.flat, panels, "ABCD"):
        df = compact_sample(subset_targets(evals[key].targets, group), max_scatter, seed)
        source.append(df[["run", "event_id", "sample_index", "target_slot", "true_pga", "pred_pga", "target_type", "magnitude"]].assign(group=group, panel=letter))
        # Preserve every plotted observation, but keep isolated points visually
        # subordinate to the dense relationship.  Filled hexagons require at
        # least three observations and use a near-white low-density colour.
        ax.scatter(
            df["true_pga"],
            df["pred_pga"],
            s=2.0,
            color="#6F7478",
            alpha=0.025,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        hb = ax.hexbin(
            df["true_pga"],
            df["pred_pga"],
            gridsize=55,
            mincnt=3,
            norm=LogNorm(vmin=3),
            cmap=SCATTER_DENSITY_CMAP,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
        hexbins.append(hb)
        lo = min(df["true_pga"].min(), df["pred_pga"].min())
        hi = max(df["true_pga"].max(), df["pred_pga"].max())
        ax.plot([lo, hi], [lo, hi], color="white", linewidth=2.3, zorder=3)
        ax.plot([lo, hi], [lo, hi], color=PALETTE["dark"], linestyle="--", linewidth=0.9, zorder=4)
        met = compute_metrics(subset_targets(evals[key].targets, group))
        ax.text(0.03, 0.96, f"MAE={met['mae']:.3f}\nR²={met['r2']:.3f}\nn={met['targets']:,}", transform=ax.transAxes, ha="left", va="top", fontsize=6.3, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2})
        ax.set(xlabel="True PGA [log10(m/s²)]", ylabel="Predicted PGA [log10(m/s²)]", title=panel_title)
        panel_label(ax, letter)
    max_count = max(float(np.nanmax(hb.get_array())) for hb in hexbins if len(hb.get_array()))
    shared_norm = LogNorm(vmin=3, vmax=max(3.0, max_count))
    for hb in hexbins:
        hb.set_norm(shared_norm)
    colorbar = fig.colorbar(hexbins[0], ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    colorbar.set_label("Targets per hexagon (log scale)")
    fig.text(
        0.5,
        0.005,
        "Sparse observations are retained as faint gray points; colored hexagons contain ≥3 targets.",
        ha="center",
        va="bottom",
        fontsize=6.3,
        color=PALETTE["neutral"],
    )
    fig.suptitle(title, y=0.985, fontsize=9)
    save_figure(fig, stem, pd.concat(source, ignore_index=True))


def reliability_table(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    values = []
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (df["prob_strong"] >= lo) & (df["prob_strong"] < hi if hi < 1 else df["prob_strong"] <= hi)
        part = df.loc[mask]
        if len(part):
            values.append({"bin": index, "probability": part["prob_strong"].mean(), "observed": part["strong"].mean(), "targets": len(part)})
    return pd.DataFrame(values)


def uncertainty_table(df: pd.DataFrame) -> pd.DataFrame:
    k_values = np.linspace(0, 3, 31)
    z = df["abs_error"].to_numpy(float) / np.maximum(df["sigma"].to_numpy(float), 1e-9)
    return pd.DataFrame({"sigma_multiple": k_values, "empirical_coverage": [np.mean(z <= k) for k in k_values], "gaussian_reference": [2 * stats.norm.cdf(k) - 1 for k in k_values]})


def residual_binned(df: pd.DataFrame) -> pd.DataFrame:
    quantile = pd.qcut(df["true_pga"], q=16, duplicates="drop")
    return df.assign(_bin=quantile).groupby("_bin", observed=True).agg(true_pga=("true_pga", "mean"), bias=("error", "mean"), mae=("abs_error", "mean"), targets=("error", "size")).reset_index(drop=True)


def plot_calibration(evals: dict[str, EvalData], keys: Sequence[str], labels: dict[str, str], stem: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
    source = []
    for key in keys:
        df = evals[key].targets
        rel = reliability_table(df).assign(run=key, panel="reliability")
        unc = uncertainty_table(df).assign(run=key, panel="uncertainty")
        res = residual_binned(df).assign(run=key, panel="residual")
        source.extend([rel, unc, res])
        axes[0].plot(rel["probability"], rel["observed"], marker="o", markersize=3, label=labels[key], color=evals[key].spec.color, linestyle=evals[key].spec.line_style)
        axes[1].plot(unc["sigma_multiple"], unc["empirical_coverage"], label=labels[key], color=evals[key].spec.color, linestyle=evals[key].spec.line_style)
        axes[2].plot(res["true_pga"], res["bias"], marker="o", markersize=2.5, label=labels[key], color=evals[key].spec.color, linestyle=evals[key].spec.line_style)
    axes[0].plot([0, 1], [0, 1], color=PALETTE["neutral"], linestyle=":")
    axes[0].set(xlabel="Predicted P(strong)", ylabel="Observed frequency", title="Strong-motion reliability", xlim=(0, 1), ylim=(0, 1))
    ref = uncertainty_table(evals[keys[0]].targets)
    axes[1].plot(ref["sigma_multiple"], ref["gaussian_reference"], color=PALETTE["neutral"], linestyle=":", label="Gaussian reference")
    axes[1].set(xlabel="Predictive sigma multiple", ylabel="Empirical coverage", title="Uncertainty coverage", xlim=(0, 3), ylim=(0, 1))
    axes[2].axhline(0, color=PALETTE["neutral"], linestyle=":")
    axes[2].set(xlabel="True PGA [log10(m/s²)]", ylabel="Mean residual (pred−true)", title="Conditional bias")
    axes[0].legend(fontsize=5.6)
    axes[1].legend(fontsize=5.6)
    for letter, ax in zip("ABC", axes):
        panel_label(ax, letter)
    fig.suptitle(title, y=1.02, fontsize=9)
    save_figure(fig, stem, pd.concat(source, ignore_index=True, sort=False))


def plot_stratified_grid(tables: Sequence[tuple[pd.DataFrame, str, str]], keys: Sequence[str], labels: dict[str, str], colors: dict[str, str], styles: dict[str, str], stem: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.25))
    sources = []
    for ax, (table, panel_title, xlabel), letter in zip(axes.flat, tables, "ABCDEF"):
        for key in keys:
            part = table[table["run"] == key].sort_values("stratum_order" if "stratum_order" in table else "stratum")
            if part.empty:
                continue
            x = np.arange(len(part))
            ax.plot(x, part["mae"], marker="o", markersize=3, label=labels[key], color=colors[key], linestyle=styles[key])
            ax.set_xticks(x, part["stratum"], rotation=35, ha="right")
            for xx, yy, count in zip(x, part["mae"], part["targets"]):
                if count < 1000:
                    ax.text(xx, yy, f"{int(count)}", fontsize=5, ha="center", va="bottom")
        ax.set(xlabel=xlabel, ylabel="MAE (dex)", title=panel_title)
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.6)
        panel_label(ax, letter)
        sources.append(table.assign(panel=letter, panel_title=panel_title))
    axes[0, 0].legend(fontsize=5.7)
    fig.suptitle(title, y=1.005, fontsize=9)
    save_figure(fig, stem, pd.concat(sources, ignore_index=True, sort=False))


def plot_target_type_control(metrics: pd.DataFrame, runs: Sequence[str], labels: dict[str, str], colors: dict[str, str], stem: Path, title: str) -> None:
    target_groups = ["input", "triggered_noninput", "untriggered", "noninput"]
    group_labels = ["Input", "Triggered non-input", "Untriggered", "All non-input"]
    view = metrics[metrics["run"].isin(runs) & metrics["group"].isin(target_groups)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65))
    for ax, metric, ylabel, letter in [(axes[0], "mae", "MAE (dex)", "A"), (axes[1], "bias", "Bias (dex)", "B")]:
        x = np.arange(len(target_groups))
        for index, run in enumerate(runs):
            values = [metric_lookup(view, run, group, metric) for group in target_groups]
            ax.plot(x, values, marker="o", markersize=3.5, color=colors[run], label=labels[run], linestyle="-" if index == 0 else "--")
        ax.axhline(0, color=PALETTE["neutral"], linewidth=0.7, linestyle=":" ) if metric == "bias" else None
        ax.set_xticks(x, group_labels, rotation=20, ha="right")
        ax.set(ylabel=ylabel, title="Target-type error" if metric == "mae" else "Target-type signed bias")
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.6)
        panel_label(ax, letter)
    axes[0].legend(fontsize=5.8)
    fig.suptitle(title, y=1.02, fontsize=9)
    save_figure(fig, stem, view)


def sample_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["sample_index", "event_id", "magnitude", "current_time_s"], observed=True).agg(
        targets=("target_slot", "size"),
        noninput_targets=("target_type_code", lambda x: int(np.isin(x, [1, 2]).sum())),
        strong_targets=("strong", "sum"),
        mae=("abs_error", "mean"),
        bias=("error", "mean"),
        true_min=("true_pga", "min"),
        true_max=("true_pga", "max"),
        pred_min=("pred_pga", "min"),
        pred_max=("pred_pga", "max"),
        station_count=("station_count", "first"),
    ).reset_index()


def add_sample_ranges(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["true_range"] = summary["true_max"] - summary["true_min"]
    summary["pred_range"] = summary["pred_max"] - summary["pred_min"]
    return summary


def select_rt55_cases(data: EvalData) -> pd.DataFrame:
    summary = add_sample_ranges(sample_level_summary(data.targets))
    candidates = summary[summary["noninput_targets"] >= 3].copy()
    median_mae = candidates["mae"].median()
    selected = []
    rules = [
        ("typical", (candidates["mae"] - median_mae).abs().idxmin(), "closest to median sample MAE"),
        ("flat_low_mae", candidates.loc[candidates["mae"] <= candidates["mae"].quantile(0.20), "noninput_targets"].idxmax(), "flat spatial output selected by low aggregate MAE; retained as a cautionary example"),
        ("flat_failure", candidates["mae"].idxmax(), "largest sample MAE among samples with >=3 non-input targets; flat-output failure"),
    ]
    strong_pool = candidates[(candidates["strong_targets"] >= 2) & (candidates["magnitude"] >= 5.0)]
    if strong_pool.empty:
        strong_pool = candidates[candidates["strong_targets"] >= 2]
    strong_index = (strong_pool["mae"] - strong_pool["mae"].median()).abs().idxmin()
    rules.insert(1, ("strong_motion", strong_index, "median-MAE high-magnitude sample with >=2 strong targets"))
    for name, index, rule in rules:
        row = summary.loc[index].to_dict()
        row.update({"case": name, "selection_rule": rule})
        selected.append(row)
    return pd.DataFrame(selected)


def flat_output_audit(data: EvalData, min_targets: int = 5, flat_range: float = 0.03, true_range: float = 0.30) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = add_sample_ranges(sample_level_summary(data.targets))
    detail = detail[detail["targets"] >= min_targets].copy()
    detail["flat_output"] = detail["pred_range"] <= flat_range
    detail["diverse_truth"] = detail["true_range"] >= true_range
    detail["flat_diverse"] = detail["flat_output"] & detail["diverse_truth"]
    detail["flat_diverse_mae_ge_0p3"] = detail["flat_diverse"] & (detail["mae"] >= 0.3)
    detail["flat_diverse_mae_ge_0p5"] = detail["flat_diverse"] & (detail["mae"] >= 0.5)
    one_station = detail["station_count"] == 1
    total_targets = int(detail["targets"].sum())
    flat_targets = int(detail.loc[detail["flat_diverse"], "targets"].sum())
    summary = pd.DataFrame(
        [{
            "minimum_valid_targets": min_targets,
            "flat_prediction_range_threshold_dex": flat_range,
            "diverse_truth_range_threshold_dex": true_range,
            "eligible_realtime_samples": len(detail),
            "eligible_targets": total_targets,
            "flat_output_samples": int(detail["flat_output"].sum()),
            "flat_output_sample_fraction": float(detail["flat_output"].mean()),
            "flat_diverse_samples": int(detail["flat_diverse"].sum()),
            "flat_diverse_sample_fraction": float(detail["flat_diverse"].mean()),
            "flat_diverse_target_fraction": flat_targets / total_targets,
            "flat_diverse_mae_ge_0p3_samples": int(detail["flat_diverse_mae_ge_0p3"].sum()),
            "flat_diverse_mae_ge_0p3_fraction": float(detail["flat_diverse_mae_ge_0p3"].mean()),
            "flat_diverse_mae_ge_0p5_samples": int(detail["flat_diverse_mae_ge_0p5"].sum()),
            "flat_diverse_mae_ge_0p5_fraction": float(detail["flat_diverse_mae_ge_0p5"].mean()),
            "one_station_eligible_samples": int(one_station.sum()),
            "one_station_flat_samples": int((one_station & detail["flat_output"]).sum()),
            "one_station_flat_fraction": float(detail.loc[one_station, "flat_output"].mean()),
        }]
    )
    return summary, detail


def rt56_sample_pair_summary(before: EvalData, after: EvalData) -> pd.DataFrame:
    b = sample_level_summary(before.targets).rename(columns={"mae": "mae_before", "bias": "bias_before"})
    a = sample_level_summary(after.targets).rename(columns={"mae": "mae_after", "bias": "bias_after"})
    keys = ["sample_index", "event_id", "magnitude", "current_time_s", "targets", "noninput_targets", "strong_targets", "station_count"]
    pair = b.merge(a, on=keys, validate="one_to_one")
    pair["delta_mae"] = pair["mae_after"] - pair["mae_before"]
    pair["improvement"] = -pair["delta_mae"]
    return pair


def select_rt56_cases(before: EvalData, after: EvalData) -> pd.DataFrame:
    pair = rt56_sample_pair_summary(before, after)
    pair = pair[pair["targets"] >= 5].copy()
    median = pair["improvement"].median()
    selected = []
    used: set[int] = set()

    def choose(pool: pd.DataFrame, score: pd.Series, largest: bool = True) -> int:
        available = pool.loc[~pool["sample_index"].isin(used)]
        if available.empty:
            available = pool
        aligned = score.loc[available.index]
        index = int(aligned.idxmax() if largest else aligned.idxmin())
        used.add(int(pair.loc[index, "sample_index"]))
        return index

    definitions = []
    definitions.append(("largest_gain", choose(pair, pair["improvement"]), "largest reduction in sample MAE among samples with >=5 targets"))
    definitions.append(("typical_gain", choose(pair, -(pair["improvement"] - median).abs()), "closest to median sample-level change; distinct from other cases"))
    definitions.append(("largest_degradation", choose(pair, pair["improvement"], largest=False), "largest increase in sample MAE among samples with >=5 targets"))
    strong = pair[pair["strong_targets"] >= 2]
    definitions.append(("strong_recovery", choose(strong, strong["improvement"]), "largest distinct gain with >=2 strong targets"))
    before_df = before.targets
    unknown_ids = set(before_df.loc[before_df["lead_bucket"].astype(str) == "unknown", "sample_index"])
    positive_ids = set(before_df.loc[before_df["lead_bucket"].astype(str).isin(["0–5", "5–20", ">20"]), "sample_index"])
    unknown = pair[pair["sample_index"].isin(unknown_ids)]
    positive = pair[pair["sample_index"].isin(positive_ids)]
    definitions.append(("unknown_lead_gain", choose(unknown, unknown["improvement"]), "largest distinct gain among samples containing unknown lead-time targets"))
    definitions.append(("positive_lead_degradation", choose(positive, positive["improvement"], largest=False), "largest distinct degradation among samples containing positive lead-time targets"))
    for name, index, rule in definitions:
        row = pair.loc[index].to_dict()
        row.update({"case": name, "selection_rule": rule})
        selected.append(row)
    return pd.DataFrame(selected)


def case_data(data: EvalData, sample_index: int) -> pd.DataFrame:
    return data.targets[data.targets["sample_index"] == sample_index].copy().sort_values("epicentral_distance_km")


def plot_rt55_case_profiles(data: EvalData, cases: pd.DataFrame, stem: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8))
    sources = []
    names = {"typical": "Typical", "strong_motion": "Strong motion", "flat_low_mae": "Flat output, low MAE", "flat_failure": "Flat-output failure"}
    for ax, (_, case), letter in zip(axes.flat, cases.iterrows(), "ABCD"):
        part = case_data(data, int(case["sample_index"]))
        x = np.arange(len(part))
        ax.plot(x, part["true_pga"], color=PALETTE["dark"], marker="o", markersize=3, label="Truth")
        ax.plot(x, part["pred_pga"], color=PALETTE["rt55_dark"], marker="s", markersize=3, label="Prediction")
        ax.fill_between(x, part["pred_pga"] - part["sigma"], part["pred_pga"] + part["sigma"], color=PALETTE["rt55_soft"], alpha=0.35, label="±1σ")
        noninput = part["target_type_code"].isin([1, 2]).to_numpy()
        ax.scatter(x[noninput], part.loc[noninput, "true_pga"], facecolor="none", edgecolor=PALETTE["rt56"], s=30, linewidth=0.9, label="Non-input truth")
        ax.axhline(THRESHOLD, color=PALETTE["strong"], linestyle=":", linewidth=0.8)
        ax.set(xlabel="Targets sorted by epicentral distance", ylabel="PGA [log10(m/s²)]", title=f"{names[str(case['case'])]} | M{case['magnitude']:.1f}, t={case['current_time_s']:.0f}s, MAE={case['mae']:.3f}")
        panel_label(ax, letter)
        sources.append(part.assign(case=case["case"], selection_rule=case["selection_rule"], plot_order=x))
    axes[0, 0].legend(fontsize=5.5, ncol=2)
    fig.suptitle("RT55 deterministic example profiles", y=1.005, fontsize=9)
    save_figure(fig, stem, pd.concat(sources, ignore_index=True))


def plot_rt55_case_maps(data: EvalData, cases: pd.DataFrame, stem: Path) -> None:
    projection = ccrs.PlateCarree()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), subplot_kw={"projection": projection})
    sources = []
    names = {"typical": "Typical", "strong_motion": "Strong motion", "flat_low_mae": "Flat output, low MAE", "flat_failure": "Flat-output failure"}
    for ax, (_, case), letter in zip(axes.flat, cases.iterrows(), "ABCD"):
        sample = int(case["sample_index"])
        part = case_data(data, sample)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#EAF2F7", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#F3F1EB", edgecolor="none", zorder=0)
        ax.coastlines(resolution="50m", color="#666666", linewidth=0.55, zorder=1)
        sc = ax.scatter(part["target_lon"], part["target_lat"], c=part["error"], cmap="RdBu_r", vmin=-0.6, vmax=0.6, s=25 + 18 * part["strong"].astype(int), edgecolor=np.where(part["target_type_code"].isin([1, 2]), PALETTE["dark"], "none"), linewidth=0.5, zorder=3, transform=projection)
        station = data.raw[f"{data.spec.split}_station_coords_abs"][sample]
        valid = data.raw[f"{data.spec.split}_station_valid"][sample].astype(bool)
        ax.scatter(station[valid, 1] + POS_OFFSET[1], station[valid, 0] + POS_OFFSET[0], marker="^", s=20, color=PALETTE["gain"], edgecolor="white", linewidth=0.45, label="Input", zorder=4, transform=projection)
        ax.scatter(part["event_lon"].iloc[0], part["event_lat"].iloc[0], marker="*", s=75, color="#F2C14E", edgecolor=PALETTE["dark"], linewidth=0.5, label="Epicentre", zorder=4, transform=projection)
        margin = 0.8
        all_lon = np.r_[part["target_lon"], station[valid, 1] + POS_OFFSET[1], part["event_lon"].iloc[0]]
        all_lat = np.r_[part["target_lat"], station[valid, 0] + POS_OFFSET[0], part["event_lat"].iloc[0]]
        ax.set_extent([all_lon.min() - margin, all_lon.max() + margin, all_lat.min() - margin, all_lat.max() + margin], crs=projection)
        grid = ax.gridlines(crs=projection, draw_labels=True, linewidth=0.35, color="#A0A0A0", alpha=0.55, linestyle=":", x_inline=False, y_inline=False)
        grid.top_labels = False
        grid.right_labels = False
        grid.xlabel_style = {"size": 6.2}
        grid.ylabel_style = {"size": 6.2}
        ax.set_title(names[str(case["case"])], fontsize=7.6, pad=4)
        panel_label(ax, letter)
        sources.append(part.assign(case=case["case"], selection_rule=case["selection_rule"]))
    axes[0, 0].legend(fontsize=5.5)
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.75, pad=0.02)
    cbar.set_label("Residual [log10(m/s²)]")
    fig.suptitle("RT55 spatial residual examples", y=0.995, fontsize=9)
    save_figure(fig, stem, pd.concat(sources, ignore_index=True))


def plot_rt56_geometry(data: EvalData, audit: pd.DataFrame, stem: Path) -> None:
    prefix = data.spec.split
    raw = data.raw
    sample = pd.DataFrame(
        {
            "requested": raw[f"{prefix}_causal_random_requested_station_count"].reshape(-1).astype(int),
            "selected": raw[f"{prefix}_causal_random_selected_station_count"].reshape(-1).astype(int),
            "available": raw[f"{prefix}_causal_random_available_station_count"].reshape(-1).astype(int),
            "time_s": raw[f"{prefix}_realtime_requested_elapsed_time"].reshape(-1).astype(float),
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    requested = sample["requested"].value_counts().sort_index()
    axes[0, 0].bar(requested.index.astype(str), requested.values, color=PALETTE["rt56"])
    axes[0, 0].set(xlabel="Requested station count", ylabel="Samples", title="Requested random geometry")
    selected = sample["selected"].value_counts().sort_index()
    axes[0, 1].bar(selected.index.astype(str), selected.values, color=PALETTE["gain"])
    axes[0, 1].set(xlabel="Selected station count", ylabel="Samples", title="Realized input count")
    axes[0, 2].hist(sample["available"], bins=np.arange(0.5, sample["available"].max() + 1.5), color=PALETTE["rt55_soft"], edgecolor="none")
    axes[0, 2].set(xlabel="Causally available stations", ylabel="Samples", title="Available station pool")
    time = sample["time_s"].value_counts().sort_index()
    axes[1, 0].bar(time.index.astype(int).astype(str), time.values, color=PALETTE["rt55_mid"])
    axes[1, 0].set(xlabel="Requested elapsed time (s)", ylabel="Samples", title="Realtime cut-off distribution")
    target_counts = data.targets["target_type"].value_counts().reindex(["triggered_noninput", "untriggered"]).dropna()
    axes[1, 1].bar(["Triggered\nnon-input", "Untriggered"], target_counts.values, color=[PALETTE["triggered"], PALETTE["untriggered"]])
    axes[1, 1].set(ylabel="Targets", title="Prediction target types")
    nearest = data.targets["nearest_input_distance_km"]
    axes[1, 2].hist(nearest.clip(upper=np.quantile(nearest, 0.99)), bins=45, color=PALETTE["rt56_soft"], edgecolor="none")
    aud = audit[audit["run"] == data.spec.key].iloc[0]
    axes[1, 2].text(0.98, 0.95, f"exact overlap: {int(aud['overlap_targets_tol_1e-6'])}\nmin: {aud['minimum_geodesic_distance_km']:.3f} km", transform=axes[1, 2].transAxes, ha="right", va="top", fontsize=6.3)
    axes[1, 2].set(xlabel="Nearest selected input (km)\n(99th percentile clipped)", ylabel="Targets", title="Target–input separation")
    for letter, ax in zip("ABCDEF", axes.flat):
        panel_label(ax, letter)
    fig.suptitle("RT56 random-geometry evaluation distribution and exclusion audit", y=1.005, fontsize=9)
    source = pd.concat(
        [sample.assign(panel="sample_geometry"), data.targets[["event_id", "sample_index", "target_slot", "target_type", "nearest_input_distance_km", "nearest_input_coord_delta_deg"]].assign(panel="target_geometry")],
        ignore_index=True,
        sort=False,
    )
    save_figure(fig, stem, source)


def plot_paired_delta(paired: pd.DataFrame, event_delta: pd.DataFrame, stem: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
    clipped = paired["delta_abs_error"].clip(-0.8, 0.8)
    axes[0].hist(clipped, bins=70, color=PALETTE["rt56"], alpha=0.85)
    axes[0].axvline(0, color=PALETTE["neutral"], linestyle=":")
    axes[0].set(xlabel="Fine-tuned − zero-shot absolute error (dex)\n(clipped to ±0.8)", ylabel="Targets", title="Target-paired change")
    ordered = event_delta.sort_values("delta_mae").reset_index(drop=True)
    axes[1].plot(np.linspace(0, 1, len(ordered)), ordered["delta_mae"], color=PALETTE["rt56"])
    axes[1].axhline(0, color=PALETTE["neutral"], linestyle=":")
    axes[1].set(xlabel="Event quantile", ylabel="ΔMAE (dex)", title="Event-level heterogeneity")
    by_lead = paired.groupby("lead_bucket", observed=False).agg(delta_mae=("delta_abs_error", "mean"), targets=("target_slot", "size")).reset_index()
    by_lead = by_lead[by_lead["targets"] > 0]
    colors = np.where(by_lead["delta_mae"] <= 0, PALETTE["gain"], PALETTE["drop"])
    axes[2].bar(by_lead["lead_bucket"].astype(str), by_lead["delta_mae"], color=colors)
    axes[2].axhline(0, color=PALETTE["neutral"], linestyle=":")
    axes[2].set_xticks(np.arange(len(by_lead)), by_lead["lead_bucket"].astype(str))
    axes[2].set(xlabel="Target lead-time bucket", ylabel="ΔMAE (dex)", title="Where adaptation helps")
    axes[2].tick_params(axis="x", rotation=30)
    for letter, ax in zip("ABC", axes):
        panel_label(ax, letter)
    fig.suptitle("Strictly paired random-geometry change (negative is better)", y=1.02, fontsize=9)
    source = pd.concat(
        [paired[["event_id", "sample_index", "target_slot", "lead_bucket", "delta_abs_error", "improvement_abs_error"]].assign(panel="targets"), event_delta.assign(panel="events"), by_lead.assign(panel="lead")],
        ignore_index=True,
        sort=False,
    )
    save_figure(fig, stem, source)


def plot_rt56_case_profiles(before: EvalData, after: EvalData, cases: pd.DataFrame, stem: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.25))
    sources = []
    names = {
        "largest_gain": "Largest gain", "typical_gain": "Typical change", "largest_degradation": "Largest degradation",
        "strong_recovery": "Strong-motion recovery", "unknown_lead_gain": "Unknown-lead gain", "positive_lead_degradation": "Positive-lead degradation",
    }
    for ax, (_, case), letter in zip(axes.flat, cases.iterrows(), "ABCDEF"):
        sample = int(case["sample_index"])
        b = case_data(before, sample)
        a = case_data(after, sample)[["target_slot", "pred_pga", "sigma", "abs_error"]].rename(columns={"pred_pga": "pred_after", "sigma": "sigma_after", "abs_error": "abs_error_after"})
        part = b.merge(a, on="target_slot", validate="one_to_one")
        part = part.sort_values("epicentral_distance_km").reset_index(drop=True)
        x = np.arange(len(part))
        ax.plot(x, part["true_pga"], color=PALETTE["dark"], marker="o", markersize=2.5, label="Truth")
        ax.plot(x, part["pred_pga"], color=PALETTE["zero"], marker="^", markersize=2.5, linestyle="--", label="Zero-shot")
        ax.plot(x, part["pred_after"], color=PALETTE["rt56"], marker="s", markersize=2.5, label="Fine-tuned")
        ax.axhline(THRESHOLD, color=PALETTE["strong"], linestyle=":", linewidth=0.7)
        ax.set(xlabel="Targets sorted by distance", ylabel="PGA [log10(m/s²)]", title=f"{names[str(case['case'])]}\nM{case['magnitude']:.1f}, t={case['current_time_s']:.0f}s, ΔMAE={case['delta_mae']:+.3f}")
        panel_label(ax, letter)
        sources.append(part.assign(case=case["case"], selection_rule=case["selection_rule"], plot_order=x))
    axes[0, 0].legend(fontsize=5.3, ncol=2)
    fig.suptitle("RT56 deterministic paired example profiles", y=1.005, fontsize=9)
    save_figure(fig, stem, pd.concat(sources, ignore_index=True))


def plot_rt56_case_maps(before: EvalData, after: EvalData, cases: pd.DataFrame, stem: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.65))
    sources = []
    names = {
        "largest_gain": "Largest gain", "typical_gain": "Typical change", "largest_degradation": "Largest degradation",
        "strong_recovery": "Strong recovery", "unknown_lead_gain": "Unknown-lead gain", "positive_lead_degradation": "Positive-lead degradation",
    }
    for ax, (_, case), letter in zip(axes.flat, cases.iterrows(), "ABCDEF"):
        sample = int(case["sample_index"])
        b = case_data(before, sample)
        a = case_data(after, sample)[["target_slot", "abs_error", "pred_pga"]].rename(columns={"abs_error": "abs_error_after", "pred_pga": "pred_after"})
        part = b.merge(a, on="target_slot", validate="one_to_one")
        part["delta_abs_error"] = part["abs_error_after"] - part["abs_error"]
        sc = ax.scatter(part["target_lon"], part["target_lat"], c=part["delta_abs_error"], cmap="PiYG_r", vmin=-0.5, vmax=0.5, s=30, edgecolor=PALETTE["dark"], linewidth=0.35, zorder=3)
        station = after.raw[f"{after.spec.split}_station_coords_abs"][sample]
        valid = after.raw[f"{after.spec.split}_station_valid"][sample].astype(bool)
        ax.scatter(station[valid, 1] + POS_OFFSET[1], station[valid, 0] + POS_OFFSET[0], marker="^", s=13, color=PALETTE["gain"], edgecolor="white", linewidth=0.25)
        ax.scatter(part["event_lon"].iloc[0], part["event_lat"].iloc[0], marker="*", s=60, color="#F2C14E", edgecolor=PALETTE["dark"], linewidth=0.4)
        all_lon = np.r_[part["target_lon"], station[valid, 1] + POS_OFFSET[1], part["event_lon"].iloc[0]]
        all_lat = np.r_[part["target_lat"], station[valid, 0] + POS_OFFSET[0], part["event_lat"].iloc[0]]
        ax.set(xlim=(all_lon.min() - 0.7, all_lon.max() + 0.7), ylim=(all_lat.min() - 0.7, all_lat.max() + 0.7), xlabel="Longitude (°E)", ylabel="Latitude (°N)")
        ax.set_title(names[str(case["case"])], fontsize=7.3, pad=4)
        ax.set_aspect("equal", adjustable="box")
        panel_label(ax, letter)
        sources.append(part.assign(case=case["case"], selection_rule=case["selection_rule"]))
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.75, pad=0.02)
    cbar.set_label("Fine-tuned − zero-shot absolute error (dex)")
    fig.suptitle("RT56 spatial adaptation examples (green improves, magenta degrades)", y=0.995, fontsize=9)
    save_figure(fig, stem, pd.concat(sources, ignore_index=True))


def write_rt55_report(
    path: Path,
    metrics: pd.DataFrame,
    event_summary: pd.DataFrame,
    station_summary: pd.DataFrame,
    mag_table: pd.DataFrame,
    cases: pd.DataFrame,
    strat_tables: dict[str, pd.DataFrame],
    flat_summary: pd.DataFrame,
) -> None:
    runs = ["rt55_ep20_val", "rt55_ep20_test", "rt55_ep32_val", "rt55_ep32_roll"]
    labels = {
        "rt55_ep20_val": "ep20 validation", "rt55_ep20_test": "ep20 held-out test",
        "rt55_ep32_val": "ep32 validation", "rt55_ep32_roll": "ep32 waveform-roll control",
    }
    headline = metrics[(metrics["run"].isin(runs[:3])) & (metrics["group"].isin(["all", "noninput"]))].copy()
    headline["run"] = headline["run"].map(labels)
    headline["group"] = headline["group"].map({"all": "all targets", "noninput": "non-input only"})
    target_breakdown = metrics[(metrics["run"].isin(["rt55_ep20_test", "rt55_ep32_val"])) & (metrics["group"].isin(["input", "triggered_noninput", "untriggered", "noninput"]))].copy()
    target_breakdown["run"] = target_breakdown["run"].map(labels)
    target_breakdown["group"] = target_breakdown["group"].map({"input": "input", "triggered_noninput": "triggered non-input", "untriggered": "untriggered", "noninput": "all non-input"})
    ep20_test_all = metrics[(metrics["run"] == "rt55_ep20_test") & (metrics["group"] == "all")].iloc[0]
    ep20_test_non = metrics[(metrics["run"] == "rt55_ep20_test") & (metrics["group"] == "noninput")].iloc[0]
    ep32_all = metrics[(metrics["run"] == "rt55_ep32_val") & (metrics["group"] == "all")].iloc[0]
    ep32_non = metrics[(metrics["run"] == "rt55_ep32_val") & (metrics["group"] == "noninput")].iloc[0]
    roll = metrics[(metrics["run"] == "rt55_ep32_roll") & (metrics["group"] == "all")].iloc[0]
    report = f"""# RT55 当前结果全面报告

生成日期：2026-08-29
范围：RT55 独立报告；正文不使用其他模型作比较。PGA 坐标均为 log10(m/s²)。

## 1. 结论摘要

- 当前唯一完整的独立测试结果是 **epoch 20 held-out test**。其全目标 MAE={ep20_test_all.mae:.4f}、RMSE={ep20_test_all.rmse:.4f}、R²={ep20_test_all.r2:.4f}、NLL={ep20_test_all.nll:.4f}、Brier={ep20_test_all.brier:.4f}；约等于典型绝对误差尺度 10^{ep20_test_all.mae:.3f}≈{10 ** ep20_test_all.mae:.2f} 倍。
- epoch 20 测试与 epoch 32 验证的全目标 MAE 分别为 {ep20_test_all.mae:.4f} 和 {ep32_all.mae:.4f}，总体量级一致；这支持常规验证口径下的稳定性，但不能替代缺失的 epoch 32 test。
- 排除所有输入台站后，epoch 20 测试 MAE 升至 {ep20_test_non.mae:.4f}，epoch 32 验证 MAE 升至 {ep32_non.mae:.4f}。因此全目标汇总确实被相对容易的输入台站目标显著拉低，**非输入台站应作为主要运行指标**。
- epoch 32 waveform-roll 控制的全目标 MAE={roll.mae:.4f}。它用于检验台站波形顺序扰动敏感性，不能解释成随机几何任务。
- 概率输出整体可用，但要同时看 NLL、Brier 和覆盖率。覆盖率理想参考约为 68.3%（1σ）和 95.4%（2σ）；偏离说明不确定度偏窄或偏宽。

## 2. 证据边界与命名

| 结果 | 数据 | checkpoint | 用途 |
| --- | --- | --- | --- |
| epoch 20 validation | validation | epoch 20 | 与同 checkpoint 的 test 对照 |
| epoch 20 held-out test | test | epoch 20 | 当前正式独立测试结论 |
| epoch 32 validation | validation | epoch 32 | 后期 checkpoint 的验证表现 |
| epoch 32 waveform-roll | validation | epoch 32 | 波形—台站顺序扰动控制 |

`formal_test_ep32_retry_20260824/` 是运行目录名，其中正式 metrics 元数据明确记录 checkpoint epoch=20；本报告始终标为 epoch 20 test。当前没有可用的 epoch 32 test 数值。

## 3. 训练、验证与测试数据分布

事件级概览：

{markdown_table(event_summary)}

台站行级概览：

{markdown_table(station_summary)}

按震级分桶的事件数：

{markdown_table(mag_table)}

![数据分布](figures/rt55_data_distribution.png)

解读：三个划分均覆盖 2000–2024 年，拆分单位是事件；不是按年份外推测试。震级分布以中小地震为主，M≥6 样本较少，故高震级分桶应结合事件数阅读。台站坐标缓存对各划分的台站行覆盖接近完整，地图主要用于覆盖审计，不用于声称均匀空间采样。

训练元数据不包含逐台站训练目标 PGA，因此本报告不虚构训练 PGA 分布。下图仅对有完整预测导出的 validation/test 目标真值作比较。

![PGA 真值分布](figures/rt55_target_pga_distribution.png)

## 4. 训练过程

![训练曲线](figures/rt55_training_curve.png)

曲线是优化目标而非最终 PGA MAE。最佳 checkpoint 必须以保存元数据和独立评估共同确认，不能仅凭某一个 epoch 的训练 loss 下结论。

## 5. 总体指标与指标解读

{markdown_table(headline, ["run", "group", "events", "targets", "mae", "rmse", "r2", "correlation", "slope", "bias", "nll", "brier", "coverage_1sigma", "coverage_2sigma", "event_balanced_mae"])}

![总体指标](figures/rt55_metric_dashboard.png)

- MAE/RMSE 越低越好；RMSE 对少数大误差更敏感。
- R² 和相关系数越高越好，但相关高不等于校准好；回归斜率低于 1 表示振幅动态范围被压缩。
- bias=预测−真值；正值为整体高估，负值为整体低估。
- NLL 越低越好，同时评价均值与预测不确定度；不能跨不同坐标定义比较。
- Brier 越低越好，评价超过强震动阈值 −1.2 的概率预测。
- event-balanced MAE 先在每个事件内平均、再对事件平均，避免台站/时刻多的事件完全主导结果。

## 6. 非输入台站主分析

{markdown_table(target_breakdown, ["run", "group", "events", "targets", "mae", "rmse", "bias", "abs_error_p90", "abs_error_gt_0p3"])}

![目标类型](figures/rt55_target_type_breakdown.png)

输入台站目标、已经触发但未输入的台站、尚未触发台站存在明确难度梯度。实际预警应用更应关注后两类，尤其是 untriggered 和正 lead-time 目标。报告同时导出每类的 P90/P95 绝对误差与大于 0.3/0.5 dex 的尾部比例。

## 7. 预测—真值、残差和概率校准

![散点密度](figures/rt55_scatter_truth_prediction.png)

![校准与残差](figures/rt55_calibration_residual.png)

散点图使用 hexbin，避免海量重叠点造成视觉误判。条件残差曲线揭示强弱震动端的系统性偏差；概率可靠性图比较预测强震动概率和真实发生率；σ 覆盖曲线评估预测区间是否过窄或过宽。

散点视觉上比 MAE 更“散”并不矛盾：MAE 是所有点到对角线的平均绝对垂距，而图覆盖约 3 dex 的 PGA 动态范围并用对数颜色保留低密度离群点。epoch 20 test 的高密度主体仍贴近对角线，但斜率 0.778 表明明显的回归到均值/动态范围压缩。排除输入台站后 MAE 0.195、R² 0.403，散布确实显著增大，说明全目标 MAE 被容易的输入目标拉低。

## 8. 按震级、台站数、时刻、lead time、距离和波形长度分桶

![分桶结果](figures/rt55_stratified_mae.png)

所有折线旁的 CSV 均包含 targets 和 events。台站数曲线是**观测分层**，不是固定同一事件、只改变输入台站数的严格干预实验；因而不能直接解释为增加台站的因果收益。

关键分桶表位于 `tables/rt55_stratified_*.csv`。M≥6、长正 lead-time 和远距离桶尤其需要结合样本量；少量样本的波动不能当成模型结构性规律。

## 9. 波形—台站顺序扰动控制

正常验证与 waveform-roll 使用相同 epoch 32 checkpoint 和 validation 集。若 roll 明显恶化，说明波形与台站几何配对信息确实被模型使用；它不是“随机位置输入预测随机位置目标”的替代实验。

## 10. 具体抽样示例

案例由固定规则自动选择，不是手工挑图：

{markdown_table(cases, ["case", "event_id", "sample_index", "magnitude", "current_time_s", "station_count", "targets", "noninput_targets", "strong_targets", "true_range", "pred_range", "mae", "bias", "selection_rule"])}

![案例曲线](figures/rt55_case_profiles.png)

![空间残差案例](figures/rt55_case_spatial_residuals.png)

空心强调的目标为非输入台站，地图三角形为实际输入台站，星号为震中。残差为预测−真值。底图使用 Natural Earth 50 m 海岸线，仅提供地理上下文。案例用于解释总体统计，不替代总体统计。

C/D 不是两个普通的随机坏例，而是同一个结构性模式：两者均只有一个有效输入台站，15 个目标的预测范围分别只有约 0.003 和 0.006 dex。单 key cross-attention 的注意力权重必然为 1，目标间差异只能依赖较弱的坐标/query 残差路径，因而容易退化为近常数输出。C 的 MAE 偶然较低，是因为真值大多集中在模型输出附近；它不能称为空间预测成功。

近常数输出审计（至少 5 个有效目标；预测范围≤0.03 dex；“真值多样”定义为真值范围≥0.30 dex）：

{markdown_table(flat_summary)}

该现象集中在单输入台站样本：所有 518 个满足目标数条件的单台站样本都落入近常数输出；但真正达到 MAE≥0.5 的灾难性样本仅 15 个，占合格 realtime 样本约 0.11%。因此它不是总体指标的主体，却是明确的早期预警/稀疏输入失效模式。

## 11. 局限与下一步

1. 只有 seed 42；事件 bootstrap 只反映事件抽样不确定性，不代表随机种子方差。
2. 当前没有 epoch 32 test，不能把 epoch 32 validation 重新命名为 test。
3. 训练 split CSV 没有逐目标 PGA，完整 train/dev/test PGA 分布需额外离线导出。
4. 严格台站数 sweep 需要锁定事件、时刻、目标集合和台站排序后重新推理；当前文件只能做观测分层。
5. 当前 NPZ 不含 attention 权重，不能生成真实 attention map。若需该图，必须在推理阶段显式导出权重。
6. 单次结果不能覆盖跨地区、跨网络或跨年份分布偏移。
7. 单输入台站条件下的近常数退化需要单独修复；平均 MAE 会低估其空间辨识失败。

## 12. 可复现文件

- `tables/rt55_aggregate_metrics.csv`：全指标和目标分组。
- `tables/rt55_stratified_*.csv`：所有分桶结果。
- `tables/rt55_flat_output_audit.csv`：单台站近常数输出的发生比例与严重度。
- `../tables/checkpoint_provenance.csv`：checkpoint 与评估文件来源。
- 每张图同名 `.csv`：实际绘图数据；`.svg` 保留可编辑文本，`.pdf` 为矢量输出，`.png` 为 300 dpi 预览。
- `../qa/`：数值一致性和图形产物审计。
"""
    path.write_text(report, encoding="utf-8")


def write_readme(output: Path) -> None:
    text = """# RT55 / RT56 current comprehensive reports

This directory is generated by `tools/generate_rt55_rt56_comprehensive_report.py`.

- `rt55/REPORT_CN.md`: standalone RT55 report; it deliberately contains no RT56 comparison.
- `rt56/REPORT_CN.md`: RT56 random-geometry adaptation report with protocol-matched RT55 references.
- `tables/`: shared provenance and parity audits.
- `qa/`: artifact manifest and automated checks.

Each quantitative figure is exported as editable SVG, vector PDF, 300 dpi PNG,
and an adjacent CSV containing the plotted data.

The spatial map panels require Cartopy and are generated with:

```bash
MPLCONFIGDIR=/tmp/mpl-rt-report \
  python \
  tools/generate_rt55_rt56_comprehensive_report.py
```
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def artifact_qa(output: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".svg", ".pdf", ".png", ".csv", ".md", ".json"}:
            continue
        row = {"path": str(path.relative_to(output)), "suffix": suffix, "bytes": path.stat().st_size, "nonempty": path.stat().st_size > 100}
        if suffix == ".svg":
            content = path.read_text(encoding="utf-8", errors="ignore")
            row["editable_text"] = "<text" in content
            row["contains_raster_image"] = "<image" in content
        rows.append(row)
    qa = pd.DataFrame(rows)
    if not qa["nonempty"].all():
        raise AssertionError("One or more generated artifacts are unexpectedly empty")
    svg = qa[qa["suffix"] == ".svg"]
    if not svg["editable_text"].eq(True).all():
        raise AssertionError("SVG export lacks editable text")
    return qa


def write_rt56_report(
    path: Path,
    metrics: pd.DataFrame,
    audit: pd.DataFrame,
    bootstrap: pd.DataFrame,
    cases: pd.DataFrame,
    event_summary: pd.DataFrame,
    station_summary: pd.DataFrame,
    mag_table: pd.DataFrame,
) -> None:
    zero = metrics[(metrics["run"] == "rt56_zero_random") & (metrics["group"] == "all")].iloc[0]
    fine = metrics[(metrics["run"] == "rt56_fine_random") & (metrics["group"] == "all")].iloc[0]
    base_normal = metrics[(metrics["run"] == "rt55_ep32_val") & (metrics["group"] == "all")].iloc[0]
    fine_normal = metrics[(metrics["run"] == "rt56_fine_normal") & (metrics["group"] == "all")].iloc[0]
    base_non = metrics[(metrics["run"] == "rt55_ep32_val") & (metrics["group"] == "noninput")].iloc[0]
    fine_non = metrics[(metrics["run"] == "rt56_fine_normal") & (metrics["group"] == "noninput")].iloc[0]
    comparison = metrics[(metrics["run"].isin(["rt56_zero_random", "rt56_fine_random", "rt55_ep32_val", "rt56_fine_normal"])) & (metrics["group"].isin(["all", "noninput"]))].copy()
    name_map = {
        "rt56_zero_random": "RT55 ep32 zero-shot random", "rt56_fine_random": "RT56 ep6 fine-tuned random",
        "rt55_ep32_val": "RT55 ep32 normal", "rt56_fine_normal": "RT56 ep6 normal",
    }
    comparison["run"] = comparison["run"].map(name_map)
    comparison["group"] = comparison["group"].map({"all": "all targets", "noninput": "non-input only"})
    audit_view = audit.copy()
    audit_view["run"] = audit_view["run"].map(name_map)
    report = f"""# RT56 随机几何微调当前结果全面报告

生成日期：2026-08-29
范围：RT56 mixed-random fine-tuning 的 validation 结果；允许与 RT55 同 checkpoint 起点和同协议 normal validation 对照。当前没有 RT56 held-out test。

## 1. 结论摘要

- 严格配对的随机几何 validation 中，MAE 从 {zero.mae:.4f} 降至 {fine.mae:.4f}（{percent_change(zero.mae, fine.mae):+.1f}%），RMSE 从 {zero.rmse:.4f} 降至 {fine.rmse:.4f}（{percent_change(zero.rmse, fine.rmse):+.1f}%），R² 从 {zero.r2:.4f} 升至 {fine.r2:.4f}。
- 不确定性明显修复：NLL 从 {zero.nll:.4f} 降至 {fine.nll:.4f}，Brier 从 {zero.brier:.4f} 降至 {fine.brier:.4f}；1σ/2σ 覆盖从 {zero.coverage_1sigma:.3f}/{zero.coverage_2sigma:.3f} 变为 {fine.coverage_1sigma:.3f}/{fine.coverage_2sigma:.3f}，更接近约 0.683/0.954 的参考覆盖率。
- 收益并不均匀：新增因果台站数较多时改善更明显；强震动目标显著改善，但弱震动段出现更强的正偏差；较长正 lead-time 目标仍是主要难点。
- normal validation 出现可测遗忘：全目标 MAE {base_normal.mae:.4f}→{fine_normal.mae:.4f}（{percent_change(base_normal.mae, fine_normal.mae):+.1f}%）；非输入目标 MAE {base_non.mae:.4f}→{fine_non.mae:.4f}（{percent_change(base_non.mae, fine_non.mae):+.1f}%）。这说明 mixed-random 的收益需要与原任务保持性共同优化。
- 随机几何导出的所有目标都属于 triggered non-input 或 untriggered；坐标审计在 1e-6 度容差下发现 0 个目标与选中输入台站重合。

## 2. 实验设计与可比性

严格 adaptation 比较是：

`RT55 epoch 32 zero-shot random geometry` → `RT56 epoch 6 fine-tuned random geometry`

两者使用相同 validation 样本、时刻和目标，目标级一一配对。normal 保持性比较是：

`RT55 epoch 32 normal validation` → `RT56 epoch 6 normal validation`

random 和 normal 的目标构造不同，二者的绝对指标不作同任务排名。

## 3. 底层训练、验证与保留测试数据分布

RT56 配置记录的 K-NET 过滤后事件划分与当前 RT55 数据清单一致：train 9,627 个事件、dev 1,383 个事件、test 2,769 个事件。test 仍是保留划分，但当前报告没有可用的 RT56 test 推理结果。

{markdown_table(event_summary)}

{markdown_table(station_summary)}

按震级分桶事件数：

{markdown_table(mag_table)}

![底层数据分布](figures/rt56_base_data_distribution.png)

三个划分均覆盖 2000–2024 年，M≥6 事件明显较少。训练清单没有逐目标 PGA 真值导出，因此不将 normalization 统计伪装成训练 PGA 直方图。

## 4. 随机几何输入、时刻和目标分布

![随机几何分布](figures/rt56_random_geometry_distribution.png)

requested K、selected K 和 causally available K 被分别展示。selected K 小于 requested K 时说明截止时刻可用台站不足，不应把它当成数据错误。所有随机目标都排除输入台站，因此这是“随机位置波形预测其他随机位置 PGA”的直接验证。

坐标排除审计：

{markdown_table(audit_view)}

审计使用有效目标坐标与有效选中输入台站坐标，而不是直接比较两个未确认共享命名空间的整数 index。最小距离大于零支持排除成立；index 数值碰巧相同不能据此判重。

## 5. 微调过程

![训练曲线](figures/rt56_training_curve.png)

validation loss 是混合训练目标；最终判断仍以同协议 PGA validation 为准。当前最佳 checkpoint 元数据为 epoch 6。

## 6. 总体指标

{markdown_table(comparison, ["run", "group", "events", "targets", "mae", "rmse", "r2", "correlation", "slope", "bias", "nll", "brier", "predictive_sigma_mean", "coverage_1sigma", "coverage_2sigma", "event_balanced_mae"])}

![随机几何总体指标](figures/rt56_random_metric_dashboard.png)

![normal 保持性](figures/rt56_normal_retention_dashboard.png)

指标阅读规则与 RT55 报告相同。尤其注意：NLL 的大幅改善部分来自 σ 从过窄恢复到更合理范围，不能只看点预测 MAE；normal 任务 σ 同时变宽并产生过覆盖，需要与遗忘一起解读。

## 7. 严格配对改善与事件 bootstrap

{markdown_table(bootstrap)}

![配对变化](figures/rt56_paired_delta.png)

bootstrap 以事件为簇，对每个事件的平均 ΔMAE 重采样；负值表示 fine-tuning 后更好。它反映 validation 事件抽样不确定性，不包含不同随机种子训练方差。

## 8. 预测—真值、概率校准与系统偏差

![散点密度](figures/rt56_scatter_truth_prediction.png)

![校准与残差](figures/rt56_calibration_residual.png)

zero-shot 的动态范围压缩和强震动低估在 fine-tuning 后缓解，但没有完全消失。强震动概率可靠性、Brier、覆盖率和条件 bias 应联合阅读，避免把单个阈值指标误当成完整校准结论。

## 9. 按台站数、时刻、震级、目标类型、lead time 和强弱 PGA 分桶

![分桶结果](figures/rt56_stratified_mae.png)

台站数分桶显示 fine-tuned 模型在较多选中台站时继续受益，这是支持随机几何微调有效的关键证据。不过它仍是 validation 上的观测分层；严格 station-count sweep 需要固定同一事件、时刻、目标和台站排序重新推理。

lead time 结果说明：模型对尚未来得及观测到目标台站信号的任务仍明显更难；unknown 必须单列，不能混入 ≤0 目标。强/弱 PGA 分桶则揭示 fine-tuning 将一部分零样本时的强震动低估，转化为弱震动段的正偏差。

## 10. normal 任务保持性与非输入目标

normal 对照必须同时报告 all targets 和 non-input only。全目标中的输入台站相对容易，会掩盖远端预测性能下降。当前结果表明微调不是“无代价适配”：随机几何能力提升的同时，normal MAE、RMSE、Brier 和预测 σ 均发生退化或变宽。

下一轮训练应考虑：提高 normal batch 权重、加入明确的保持性正则/蒸馏，或以 random MAE 与 normal non-input MAE 的 Pareto 前沿选 checkpoint，而不是只按混合 validation loss 选点。

## 11. 具体抽样示例

案例由固定规则自动选择：

{markdown_table(cases, ["case", "event_id", "sample_index", "magnitude", "current_time_s", "targets", "strong_targets", "mae_before", "mae_after", "delta_mae", "selection_rule"])}

![案例曲线](figures/rt56_case_profiles.png)

![空间改善图](figures/rt56_case_spatial_delta.png)

空间图颜色为 fine-tuned − zero-shot 的绝对误差：绿色为改善、洋红为退化。案例覆盖最大改善、中位变化、最大退化、强震动恢复、unknown-lead 改善和正 lead-time 退化。

## 12. 局限与必须补做的实验

1. 当前只有 validation，没有 RT56 held-out test。
2. 只有一个训练 seed；事件 bootstrap 不能替代多 seed。
3. 需要 controlled station-count sweep 才能严谨回答“多一个台站带来多少收益”。
4. 需要在 inference 中导出 attention 权重才能绘制真实 attention map；当前不制造替代图。
5. 应补 normal/random 多目标 checkpoint 选择，验证能否减少 normal forgetting。
6. 应增加按地区、网络、目标—输入最邻近距离和高震级事件的外部/压力测试。

## 13. 可复现文件

- `tables/rt56_aggregate_metrics.csv`、`tables/rt56_stratified_*.csv`。
- `tables/rt56_coordinate_exclusion_audit.csv`：目标排除输入台站审计。
- `tables/rt56_paired_event_bootstrap.csv` 和事件级明细。
- 每张图同名 CSV + SVG/PDF/PNG；SVG/PDF 文本可编辑，PNG 为 300 dpi。
- `../qa/`：正式 metrics 复算一致性与产物审计。
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = ensure_dir(args.output_dir)
    rt55_dir = ensure_dir(output / "rt55")
    rt56_dir = ensure_dir(output / "rt56")
    rt55_fig = ensure_dir(rt55_dir / "figures")
    rt56_fig = ensure_dir(rt56_dir / "figures")
    rt55_tab = ensure_dir(rt55_dir / "tables")
    rt56_tab = ensure_dir(rt56_dir / "tables")
    shared_tab = ensure_dir(output / "tables")
    qa_dir = ensure_dir(output / "qa")

    spec_list, rt55_root = specs(args.results_root)
    evals = {spec.key: flatten_eval(spec) for spec in spec_list}
    add_distance_buckets(evals)
    colors = {key: data.spec.color for key, data in evals.items()}
    styles = {key: data.spec.line_style for key, data in evals.items()}

    events, stations, cache = load_split_metadata(rt55_root, args.station_cache)
    event_summary, station_summary, mag_table = distribution_summary(events, stations, cache)
    export_table(event_summary, rt55_tab / "rt55_event_split_summary.csv")
    export_table(station_summary, rt55_tab / "rt55_station_split_summary.csv")
    export_table(mag_table, rt55_tab / "rt55_magnitude_split_counts.csv")
    export_table(event_summary, rt56_tab / "rt56_event_split_summary.csv")
    export_table(station_summary, rt56_tab / "rt56_station_split_summary.csv")
    export_table(mag_table, rt56_tab / "rt56_magnitude_split_counts.csv")

    metrics = aggregate_metrics(evals)
    parity = audit_formal_metrics(evals)
    provenance = checkpoint_provenance(evals)
    audit = coordinate_exclusion_audit(evals)
    export_table(parity, qa_dir / "formal_metric_recompute_parity.csv")
    export_table(provenance, shared_tab / "checkpoint_provenance.csv")
    export_table(metrics[metrics["run"].str.startswith("rt55")], rt55_tab / "rt55_aggregate_metrics.csv")
    export_table(metrics[metrics["run"].isin(["rt56_zero_random", "rt56_fine_random", "rt56_fine_normal", "rt55_ep32_val"])], rt56_tab / "rt56_aggregate_metrics.csv")
    export_table(audit, rt56_tab / "rt56_coordinate_exclusion_audit.csv")

    plot_rt55_data_distribution(events, stations, cache, rt55_fig / "rt55_data_distribution")
    plot_target_pga_distribution(
        evals,
        ["rt55_ep20_val", "rt55_ep20_test", "rt55_ep32_val"],
        rt55_fig / "rt55_target_pga_distribution",
        "RT55 target PGA distributions (available prediction exports)",
    )
    plot_training_curve(rt55_root, rt55_fig / "rt55_training_curve", "RT55 optimization trajectory", marker_epochs=[20, 32])
    plot_metric_dashboard(
        metrics,
        ["rt55_ep20_val", "rt55_ep20_test", "rt55_ep32_val"],
        ["all", "noninput"],
        {"rt55_ep20_val": "ep20 val", "rt55_ep20_test": "ep20 test", "rt55_ep32_val": "ep32 val"},
        colors,
        rt55_fig / "rt55_metric_dashboard",
        "RT55 normal-protocol performance",
    )
    plot_scatter_grid(
        evals,
        [
            ("rt55_ep20_test", "all", "ep20 test: all"),
            ("rt55_ep20_test", "noninput", "ep20 test: non-input"),
            ("rt55_ep32_val", "all", "ep32 val: all"),
            ("rt55_ep32_val", "noninput", "ep32 val: non-input"),
        ],
        rt55_fig / "rt55_scatter_truth_prediction",
        args.seed,
        args.max_scatter,
        "RT55 prediction versus truth",
    )
    plot_calibration(
        evals,
        ["rt55_ep20_test", "rt55_ep32_val"],
        {"rt55_ep20_test": "ep20 test", "rt55_ep32_val": "ep32 val"},
        rt55_fig / "rt55_calibration_residual",
        "RT55 probabilistic calibration and conditional bias",
    )
    plot_target_type_control(
        metrics,
        ["rt55_ep20_test", "rt55_ep32_val", "rt55_ep32_roll"],
        {"rt55_ep20_test": "ep20 test", "rt55_ep32_val": "ep32 normal", "rt55_ep32_roll": "ep32 roll"},
        colors,
        rt55_fig / "rt55_target_type_breakdown",
        "RT55 target-type difficulty and roll control",
    )

    rt55_keys = ["rt55_ep20_test", "rt55_ep32_val"]
    rt55_strat = {
        "magnitude": stratified_metrics(evals, rt55_keys, "magnitude_bucket", MAG_LABELS, "noninput"),
        "station": stratified_metrics(evals, rt55_keys, "station_bucket", STATION_ORDER, "noninput"),
        "time": stratified_metrics(evals, rt55_keys, "time_bucket", [str(v) for v in TIME_ORDER], "noninput"),
        "lead": stratified_metrics(evals, rt55_keys, "lead_bucket", LEAD_ORDER, "noninput"),
        "distance": stratified_metrics(evals, rt55_keys, "distance_bucket", ["<25", "25–50", "50–100", "100–200", "200–400", "≥400"], "noninput"),
        "waveform": stratified_metrics(evals, rt55_keys, "waveform_bucket", ["<1", "1–3", "3–5", "5–10", "10–20", "≥20"], "noninput"),
    }
    for name, table in rt55_strat.items():
        export_table(table, rt55_tab / f"rt55_stratified_{name}.csv")
    plot_stratified_grid(
        [
            (rt55_strat["magnitude"], "Magnitude", "Magnitude bin"),
            (rt55_strat["station"], "Observed input count", "Station-count bin"),
            (rt55_strat["time"], "Realtime cut-off", "Elapsed time (s)"),
            (rt55_strat["lead"], "Target lead time", "Lead-time bin (s)"),
            (rt55_strat["distance"], "Epicentral distance", "Distance bin (km)"),
            (rt55_strat["waveform"], "Mean valid waveform", "Seconds bin"),
        ],
        rt55_keys,
        {"rt55_ep20_test": "ep20 test", "rt55_ep32_val": "ep32 val"},
        colors,
        styles,
        rt55_fig / "rt55_stratified_mae",
        "RT55 non-input target MAE by operating regime",
    )
    flat_summary, flat_detail = flat_output_audit(evals["rt55_ep20_test"])
    export_table(flat_summary, rt55_tab / "rt55_flat_output_audit.csv")
    export_table(flat_detail, rt55_tab / "rt55_flat_output_sample_details.csv")
    rt55_cases = select_rt55_cases(evals["rt55_ep20_test"])
    export_table(rt55_cases, rt55_tab / "rt55_case_selection.csv")
    plot_rt55_case_profiles(evals["rt55_ep20_test"], rt55_cases, rt55_fig / "rt55_case_profiles")
    plot_rt55_case_maps(evals["rt55_ep20_test"], rt55_cases, rt55_fig / "rt55_case_spatial_residuals")

    rt56_root = args.results_root / "weights_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42"
    plot_rt55_data_distribution(events, stations, cache, rt56_fig / "rt56_base_data_distribution")
    plot_training_curve(rt56_root, rt56_fig / "rt56_training_curve", "RT56 mixed-random fine-tuning trajectory", marker_epochs=[6])
    plot_rt56_geometry(evals["rt56_fine_random"], audit, rt56_fig / "rt56_random_geometry_distribution")
    plot_metric_dashboard(
        metrics,
        ["rt56_zero_random", "rt56_fine_random"],
        ["all"],
        {"rt56_zero_random": "zero-shot", "rt56_fine_random": "fine-tuned"},
        colors,
        rt56_fig / "rt56_random_metric_dashboard",
        "Random-geometry adaptation",
    )
    plot_metric_dashboard(
        metrics,
        ["rt55_ep32_val", "rt56_fine_normal"],
        ["all", "noninput"],
        {"rt55_ep32_val": "RT55 ep32", "rt56_fine_normal": "RT56 ep6"},
        colors,
        rt56_fig / "rt56_normal_retention_dashboard",
        "Normal-protocol retention after mixed-random fine-tuning",
    )
    plot_scatter_grid(
        evals,
        [
            ("rt56_zero_random", "all", "Zero-shot: all random targets"),
            ("rt56_fine_random", "all", "Fine-tuned: all random targets"),
            ("rt56_zero_random", "strong", "Zero-shot: strong targets"),
            ("rt56_fine_random", "strong", "Fine-tuned: strong targets"),
        ],
        rt56_fig / "rt56_scatter_truth_prediction",
        args.seed,
        args.max_scatter,
        "RT56 random-geometry prediction versus truth",
    )
    plot_calibration(
        evals,
        ["rt56_zero_random", "rt56_fine_random"],
        {"rt56_zero_random": "zero-shot", "rt56_fine_random": "fine-tuned"},
        rt56_fig / "rt56_calibration_residual",
        "RT56 random-geometry calibration and conditional bias",
    )

    rt56_keys = ["rt56_zero_random", "rt56_fine_random"]
    rt56_strat = {
        "station": stratified_metrics(evals, rt56_keys, "station_bucket", STATION_ORDER),
        "time": stratified_metrics(evals, rt56_keys, "time_bucket", [str(v) for v in TIME_ORDER]),
        "magnitude": stratified_metrics(evals, rt56_keys, "magnitude_bucket", MAG_LABELS),
        "target_type": stratified_metrics(evals, rt56_keys, "target_type", ["triggered_noninput", "untriggered"]),
        "lead": stratified_metrics(evals, rt56_keys, "lead_bucket", LEAD_ORDER),
        "strength": stratified_metrics(evals, rt56_keys, "pga_strength", ["weak", "strong"]),
        "nearest_input": stratified_metrics(evals, rt56_keys, "nearest_input_bucket", ["<25", "25–50", "50–100", "100–200", "≥200"]),
    }
    for name, table in rt56_strat.items():
        export_table(table, rt56_tab / f"rt56_stratified_{name}.csv")
    plot_stratified_grid(
        [
            (rt56_strat["station"], "Selected input count", "Station-count bin"),
            (rt56_strat["time"], "Realtime cut-off", "Elapsed time (s)"),
            (rt56_strat["magnitude"], "Magnitude", "Magnitude bin"),
            (rt56_strat["target_type"], "Target type", "Target type"),
            (rt56_strat["lead"], "Target lead time", "Lead-time bin (s)"),
            (rt56_strat["strength"], "PGA strength", "Truth class"),
        ],
        rt56_keys,
        {"rt56_zero_random": "zero-shot", "rt56_fine_random": "fine-tuned"},
        colors,
        styles,
        rt56_fig / "rt56_stratified_mae",
        "RT56 random-geometry MAE by operating regime",
    )

    paired = paired_target_frame(evals["rt56_zero_random"], evals["rt56_fine_random"])
    bootstrap, event_delta = event_bootstrap_delta(paired, args.bootstrap, args.seed)
    export_table(bootstrap, rt56_tab / "rt56_paired_event_bootstrap.csv")
    export_table(event_delta, rt56_tab / "rt56_paired_event_metrics.csv")
    plot_paired_delta(paired, event_delta, rt56_fig / "rt56_paired_delta")
    rt56_cases = select_rt56_cases(evals["rt56_zero_random"], evals["rt56_fine_random"])
    export_table(rt56_cases, rt56_tab / "rt56_case_selection.csv")
    plot_rt56_case_profiles(evals["rt56_zero_random"], evals["rt56_fine_random"], rt56_cases, rt56_fig / "rt56_case_profiles")
    plot_rt56_case_maps(evals["rt56_zero_random"], evals["rt56_fine_random"], rt56_cases, rt56_fig / "rt56_case_spatial_delta")

    write_rt55_report(rt55_dir / "REPORT_CN.md", metrics, event_summary, station_summary, mag_table, rt55_cases, rt55_strat, flat_summary)
    write_rt56_report(rt56_dir / "REPORT_CN.md", metrics, audit, bootstrap, rt56_cases, event_summary, station_summary, mag_table)
    write_readme(output)
    provenance_json = {
        "generated_at": "2026-08-29 Asia/Shanghai",
        "generator": str(Path(__file__)),
        "python_backend": "NumPy/Pandas/SciPy/Matplotlib/Seaborn",
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "strong_motion_threshold_log10_mps2": THRESHOLD,
        "notes": ["RT55 formal test is epoch 20", "RT56 has validation only", "random target exclusion audited by coordinates"],
    }
    (shared_tab / "report_provenance.json").write_text(json.dumps(provenance_json, indent=2), encoding="utf-8")
    qa = artifact_qa(output)
    export_table(qa, qa_dir / "artifact_manifest.csv")
    print(f"Generated reports in {output}")
    print(f"Artifacts audited: {len(qa)}")


if __name__ == "__main__":
    main()
