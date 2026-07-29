#!/usr/bin/env python3
"""Diagnose station-pool attention relative to each station's visible P-wave window.

This is an eval-only diagnostic.  It loads an existing TEAM checkpoint, runs
the configured train and/or validation generators, and records the temporal
attention used by the DiTing station adapter for every *valid* input station.

The realtime generator right-aligns the visible waveform in the 10000-sample
model input.  For each station, the script partitions that final input into:

    left padding | visible pre-P waveform | visible post-P waveform

The partition is mapped to each feature-token resolution with fractional token
coverage.  This avoids interpreting an absolute attention effective-token
count without accounting for the station-specific P arrival and visible
waveform duration.

No model source changes are required.  The script temporarily wraps selected
AttentionPool1d.forward methods so that every per-station invocation is
captured before the adapter's ``_last_attention`` field is overwritten by the
next station.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "diting"))
sys.path.insert(0, os.path.join(REPO_ROOT, "..", "ditingbench"))

from eval_checkpoint import build_datasets, build_model_and_load  # noqa: E402
from train_light import build_diting_args  # noqa: E402


POOL_ATTRIBUTE_BY_NAME = {
    "base_x": "base_pool",
    "f2": "pool_f2",
    "f3": "pool_f3",
    "f4": "pool_f4",
    "x": "pool_x",
}

IDENTIFIER_FIELDS = {
    "model",
    "split",
    "sample_index",
    "event_id",
    "station_slot",
    "station_index",
    "pool",
    "post_p_bin",
    "station_token_weight_mode",
}

SUMMARY_METRICS = (
    "visible_seconds",
    "pre_p_seconds",
    "post_p_seconds",
    "visible_token_equiv",
    "post_p_token_equiv",
    "attention_effective_tokens_mean",
    "attention_effective_over_visible_mean",
    "attention_effective_over_post_p_mean",
    "attention_mass_padding_mean",
    "attention_mass_visible_mean",
    "attention_mass_pre_p_mean",
    "attention_mass_post_p_mean",
    "attention_visible_effective_tokens_mean",
    "attention_visible_effective_over_visible_mean",
    "attention_visible_entropy_norm_mean",
    "attention_post_p_effective_tokens_mean",
    "attention_post_p_effective_over_post_p_mean",
    "attention_post_p_entropy_norm_mean",
    "prior_effective_tokens",
    "prior_effective_over_visible",
    "prior_effective_over_post_p",
    "prior_mass_padding",
    "prior_mass_visible",
    "prior_mass_pre_p",
    "prior_mass_post_p",
    "prior_visible_effective_tokens",
    "prior_visible_effective_over_visible",
    "prior_visible_entropy_norm",
    "prior_post_p_effective_tokens",
    "prior_post_p_effective_over_post_p",
    "prior_post_p_entropy_norm",
)


def _parse_name_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_float_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    values = []
    for item in str(value).replace(";", ",").split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar(value, default=None):
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return value.detach().cpu().reshape(-1)[0].item()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return _scalar(value[0], default=default)
    return value


def _finite_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _format_edge(value):
    if abs(float(value) - round(float(value))) < 1e-8:
        return str(int(round(float(value))))
    return f"{float(value):g}"


def _post_p_bin(seconds, edges):
    seconds = _finite_float(seconds)
    if seconds is None:
        return "unknown"
    if seconds < 0:
        return "untriggered"
    edges = sorted(set(float(edge) for edge in edges))
    if not edges:
        return "all"
    if seconds < edges[0]:
        return f"<{_format_edge(edges[0])}"
    for low, high in zip(edges[:-1], edges[1:]):
        if low <= seconds < high:
            return f"[{_format_edge(low)},{_format_edge(high)})"
    return f">={_format_edge(edges[-1])}"


def _summary(values):
    clean = []
    for value in values:
        value = _finite_float(value)
        if value is not None:
            clean.append(value)
    if not clean:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "q25": None,
            "q75": None,
            "min": None,
            "max": None,
        }
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        with open(path, "w", newline="") as handle:
            handle.write("")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _find_station_adapter(raw_model):
    waveform_model = getattr(raw_model, "waveform_model", None)
    if isinstance(waveform_model, torch.nn.Sequential) and len(waveform_model) >= 2:
        adapter = waveform_model[1]
    else:
        adapter = None
    if adapter is None:
        raise ValueError(
            "Expected waveform_model to be nn.Sequential(encoder, station_adapter)."
        )
    missing = [
        attr
        for attr in ("base_pool", "pool_x")
        if not hasattr(adapter, attr)
    ]
    if missing:
        raise ValueError(
            "Station adapter does not expose the required attention pools: "
            + ", ".join(missing)
        )
    return adapter


class StationPoolCallRecorder:
    """Capture every selected station-pool invocation in model-forward order."""

    def __init__(self, adapter, pool_names):
        self.adapter = adapter
        self.pool_names = list(pool_names)
        self.calls = defaultdict(list)
        self._original_forwards = {}

    def clear(self):
        self.calls = defaultdict(list)

    def install(self):
        for pool_name in self.pool_names:
            if pool_name not in POOL_ATTRIBUTE_BY_NAME:
                raise ValueError(
                    f"Unknown pool {pool_name!r}; choose from "
                    f"{sorted(POOL_ATTRIBUTE_BY_NAME)}"
                )
            attr = POOL_ATTRIBUTE_BY_NAME[pool_name]
            pool = getattr(self.adapter, attr, None)
            if pool is None:
                raise ValueError(f"Station adapter has no {attr!r} for pool {pool_name!r}.")
            original_forward = pool.forward
            self._original_forwards[pool_name] = original_forward

            def wrapped(x, *args, _pool=pool, _name=pool_name,
                        _forward=original_forward, **kwargs):
                output = _forward(x, *args, **kwargs)
                attention = getattr(_pool, "_last_attention", None)
                if attention is None:
                    raise RuntimeError(f"Pool {_name!r} did not expose _last_attention.")
                token_weight = kwargs.get("token_weight")
                # AttentionPool1d.forward(x, query_bias, token_weight, ...)
                if token_weight is None and len(args) >= 2:
                    token_weight = args[1]
                self.calls[_name].append({
                    "attention": attention.detach().float().cpu().clone(),
                    "prior": (
                        token_weight.detach().float().cpu().clone()
                        if isinstance(token_weight, torch.Tensor)
                        else None
                    ),
                    "token_weight_floor": float(
                        kwargs.get("token_weight_floor", 1e-6)
                    ),
                    "token_weight_scale": float(
                        kwargs.get("token_weight_scale", 1.0)
                    ),
                })
                return output

            pool.forward = wrapped
        return self

    def remove(self):
        for pool_name, original_forward in self._original_forwards.items():
            attr = POOL_ATTRIBUTE_BY_NAME[pool_name]
            getattr(self.adapter, attr).forward = original_forward
        self._original_forwards = {}

    def __enter__(self):
        return self.install()

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove()


def _phase_token_fractions(trace_length, token_length, visible_start, p_pick_shifted):
    """Return fractional token coverage for padding, visible pre-P, and post-P."""
    trace_length = int(trace_length)
    token_length = int(token_length)
    visible_start = float(np.clip(float(visible_start), 0.0, float(trace_length)))
    p_pick_shifted = float(
        np.clip(float(p_pick_shifted), visible_start, float(trace_length))
    )

    sample_centers = torch.arange(trace_length, dtype=torch.float32) + 0.5
    padding = (sample_centers < visible_start).float()
    pre_p = (
        (sample_centers >= visible_start)
        & (sample_centers < p_pick_shifted)
    ).float()
    post_p = (sample_centers >= p_pick_shifted).float()
    sample_parts = torch.stack([padding, pre_p, post_p], dim=0).unsqueeze(0)
    token_parts = F.adaptive_avg_pool1d(sample_parts, token_length).squeeze(0)
    denom = token_parts.sum(dim=0, keepdim=True).clamp_min(1e-8)
    token_parts = token_parts / denom
    return {
        "padding": token_parts[0].numpy(),
        "pre_p": token_parts[1].numpy(),
        "post_p": token_parts[2].numpy(),
        "visible": (token_parts[1] + token_parts[2]).numpy(),
    }


def _entropy_effective_count(probabilities, axis=0):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    denom = probabilities.sum(axis=axis, keepdims=True)
    normalized = probabilities / np.clip(denom, 1e-12, None)
    entropy = -(
        normalized * np.log(np.clip(normalized, 1e-12, None))
    ).sum(axis=axis)
    return entropy, np.exp(entropy), normalized


def _distribution_metrics(distribution, fractions):
    """Metrics for a T x Q distribution; returns one value per query."""
    distribution = np.asarray(distribution, dtype=np.float64)
    if distribution.ndim == 1:
        distribution = distribution[:, None]
    if distribution.ndim != 2:
        raise ValueError(f"Expected distribution shape (T,Q), got {distribution.shape}")
    token_length = distribution.shape[0]
    for name, values in fractions.items():
        if np.asarray(values).shape != (token_length,):
            raise ValueError(
                f"Phase fraction {name!r} has shape {np.asarray(values).shape}; "
                f"expected {(token_length,)}"
            )

    entropy, effective, prob = _entropy_effective_count(distribution, axis=0)
    del entropy
    visible_equiv = float(np.asarray(fractions["visible"]).sum())
    post_p_equiv = float(np.asarray(fractions["post_p"]).sum())

    padding_mass = (
        prob * np.asarray(fractions["padding"])[:, None]
    ).sum(axis=0)
    visible_weight = prob * np.asarray(fractions["visible"])[:, None]
    visible_mass = visible_weight.sum(axis=0)
    pre_p_mass = (
        prob * np.asarray(fractions["pre_p"])[:, None]
    ).sum(axis=0)
    post_p_weight = prob * np.asarray(fractions["post_p"])[:, None]
    post_p_mass = post_p_weight.sum(axis=0)

    visible_entropy, visible_effective, _ = _entropy_effective_count(
        visible_weight,
        axis=0,
    )
    visible_effective = np.where(
        visible_mass > 1e-12,
        visible_effective,
        np.nan,
    )
    if visible_equiv > 1.0:
        visible_entropy_norm = visible_entropy / math.log(visible_equiv)
        visible_entropy_norm = np.where(
            visible_mass > 1e-12,
            visible_entropy_norm,
            np.nan,
        )
    else:
        visible_entropy_norm = np.where(
            visible_mass > 1e-12,
            0.0,
            np.nan,
        )

    post_entropy, post_effective, _ = _entropy_effective_count(
        post_p_weight,
        axis=0,
    )
    post_effective = np.where(post_p_mass > 1e-12, post_effective, np.nan)
    if post_p_equiv > 1.0:
        post_entropy_norm = post_entropy / math.log(post_p_equiv)
        post_entropy_norm = np.where(post_p_mass > 1e-12, post_entropy_norm, np.nan)
    else:
        post_entropy_norm = np.where(post_p_mass > 1e-12, 0.0, np.nan)

    return {
        "effective_tokens": effective,
        "effective_over_visible": effective / max(visible_equiv, 1e-8),
        "effective_over_post_p": effective / max(post_p_equiv, 1e-8),
        "mass_padding": padding_mass,
        "mass_visible": visible_mass,
        "mass_pre_p": pre_p_mass,
        "mass_post_p": post_p_mass,
        "visible_effective_tokens": visible_effective,
        "visible_effective_over_visible": (
            visible_effective / max(visible_equiv, 1e-8)
        ),
        "visible_entropy_norm": visible_entropy_norm,
        "post_p_effective_tokens": post_effective,
        "post_p_effective_over_post_p": (
            post_effective / max(post_p_equiv, 1e-8)
        ),
        "post_p_entropy_norm": post_entropy_norm,
        "probability_sum_error": np.abs(prob.sum(axis=0) - 1.0),
    }


def _add_query_aggregate(row, prefix, metrics):
    for metric_name, values in metrics.items():
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        row[f"{prefix}_{metric_name}_mean"] = float(finite.mean())
        row[f"{prefix}_{metric_name}_min"] = float(finite.min())
        row[f"{prefix}_{metric_name}_max"] = float(finite.max())


def _add_prior_metrics(row, prior, fractions):
    if prior is None:
        return
    prior = np.asarray(prior, dtype=np.float64)
    if prior.ndim == 2 and prior.shape[0] == 1:
        prior = prior[0]
    prior = prior.reshape(-1)
    metrics = _distribution_metrics(prior[:, None], fractions)
    for metric_name, values in metrics.items():
        value = np.asarray(values, dtype=np.float64).reshape(-1)[0]
        if np.isfinite(value):
            row[f"prior_{metric_name}"] = float(value)


def _dataset_sampling_rate(dataset, p_picks, fallback):
    elapsed = _finite_float(
        _scalar(
            p_picks.get("realtime_elapsed_time")
            if isinstance(p_picks, dict)
            else None
        )
    )
    current = _finite_float(
        _scalar(
            p_picks.get("realtime_current_sample")
            if isinstance(p_picks, dict)
            else None
        )
    )
    first_pick = _finite_float(
        _scalar(
            p_picks.get("realtime_first_p_pick_sample")
            if isinstance(p_picks, dict)
            else None
        )
    )
    if (
        elapsed is not None
        and elapsed > 1e-8
        and current is not None
        and first_pick is not None
        and current >= first_pick
    ):
        inferred = (current - first_pick) / elapsed
        if np.isfinite(inferred) and inferred > 0:
            return float(inferred)

    visited = set()

    def visit(obj):
        if obj is None or id(obj) in visited:
            return None
        visited.add(id(obj))
        value = _finite_float(getattr(obj, "sampling_rate", None))
        if value is not None and value > 0:
            return value
        for attr in ("generators", "datasets"):
            children = getattr(obj, attr, None)
            if children is not None:
                for child in children:
                    value = visit(child)
                    if value is not None:
                        return value
        return visit(getattr(obj, "generator", None))

    value = visit(dataset)
    return float(value if value is not None else fallback)


def _p_pick_array(p_picks, key, length, fill=np.nan):
    if not isinstance(p_picks, dict) or key not in p_picks:
        return np.full(int(length), fill, dtype=np.float64)
    values = _to_numpy(p_picks[key]).reshape(-1).astype(np.float64, copy=False)
    if values.size < int(length):
        values = np.pad(
            values,
            (0, int(length) - values.size),
            mode="constant",
            constant_values=fill,
        )
    return values[:int(length)]


def _station_index_array(p_picks, length):
    for key in (
        "selected_original_input_indices",
        "original_station_indices",
        "selected_input_indices",
    ):
        if isinstance(p_picks, dict) and key in p_picks:
            values = _to_numpy(p_picks[key]).reshape(-1).astype(np.int64, copy=False)
            if values.size < int(length):
                values = np.pad(
                    values,
                    (0, int(length) - values.size),
                    mode="constant",
                    constant_values=-1,
                )
            return values[:int(length)]
    return np.arange(int(length), dtype=np.int64)


def _event_id(p_picks, sample_index):
    if isinstance(p_picks, dict):
        for key in ("event_id", "event_name", "source_id"):
            if key in p_picks:
                return str(_scalar(p_picks[key], default=p_picks[key]))
    return str(sample_index)


def _move_inputs_to_device(inputs, device):
    return [
        value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
        for value in inputs
    ]


def _select_indices(dataset, max_samples, start_index, stride):
    indices = list(range(int(start_index), len(dataset), max(1, int(stride))))
    if int(max_samples) >= 0:
        indices = indices[:int(max_samples)]
    return indices


@torch.no_grad()
def diagnose_split(model, recorder, dataset, split, config, args):
    raw_model = model.module if hasattr(model, "module") else model
    rows = []
    indices = _select_indices(
        dataset,
        args.max_samples,
        args.start_index,
        args.stride,
    )
    print(
        f"[INFO] split={split}: dataset_size={len(dataset)}, "
        f"selected_samples={len(indices)}"
    )

    for order, sample_index in enumerate(indices):
        inputs, _labels, p_picks = dataset[sample_index]
        waveform = inputs[0]
        station_valid = _to_numpy(inputs[2]).astype(bool).reshape(-1)
        station_slots = int(station_valid.size)
        trace_length = int(waveform.shape[-1])

        shifted_p_picks = _p_pick_array(
            p_picks,
            "shifted",
            station_slots,
        )
        raw_p_picks = _p_pick_array(
            p_picks,
            "raw",
            station_slots,
        )
        station_indices = _station_index_array(p_picks, station_slots)
        shift = _finite_float(
            _scalar(p_picks.get("shift") if isinstance(p_picks, dict) else None),
            default=0.0,
        )
        current_sample = _finite_float(
            _scalar(
                p_picks.get("realtime_current_sample")
                if isinstance(p_picks, dict)
                else None
            ),
            default=float(trace_length - shift - 1),
        )
        requested_elapsed = _finite_float(
            _scalar(
                p_picks.get("realtime_requested_elapsed_time")
                if isinstance(p_picks, dict)
                else None
            )
        )
        elapsed = _finite_float(
            _scalar(
                p_picks.get("realtime_elapsed_time")
                if isinstance(p_picks, dict)
                else None
            )
        )
        sampling_rate = _dataset_sampling_rate(
            dataset,
            p_picks,
            fallback=args.sampling_rate,
        )
        event_id = _event_id(p_picks, sample_index)

        recorder.clear()
        _ = model(*_move_inputs_to_device(inputs, args.device))

        for pool_name in args.pool_names:
            calls = recorder.calls.get(pool_name, [])
            if len(calls) != station_slots:
                raise RuntimeError(
                    f"split={split} sample={sample_index} pool={pool_name}: "
                    f"captured {len(calls)} calls for {station_slots} station slots. "
                    "The model station-loop or adapter layout has changed."
                )

        for station_slot in np.flatnonzero(station_valid):
            raw_pick = _finite_float(raw_p_picks[station_slot])
            shifted_pick = _finite_float(shifted_p_picks[station_slot])
            if shifted_pick is None:
                continue
            if raw_pick is not None and current_sample is not None:
                post_p_seconds = (current_sample - raw_pick) / sampling_rate
            else:
                post_p_seconds = (
                    (trace_length - 1) - shifted_pick
                ) / sampling_rate
            post_p_seconds = max(0.0, float(post_p_seconds))
            visible_seconds = max(0.0, (trace_length - shift) / sampling_rate)
            pre_p_seconds = max(0.0, visible_seconds - post_p_seconds)
            post_p_bin = _post_p_bin(post_p_seconds, args.post_p_bin_edges)

            for pool_name in args.pool_names:
                call = recorder.calls[pool_name][int(station_slot)]
                attention = _to_numpy(call["attention"])
                if attention.ndim == 3 and attention.shape[0] == 1:
                    attention = attention[0]
                if attention.ndim != 2:
                    raise RuntimeError(
                        f"Expected attention (T,Q), got {attention.shape} "
                        f"for pool={pool_name}"
                    )
                token_length = int(attention.shape[0])
                fractions = _phase_token_fractions(
                    trace_length,
                    token_length,
                    shift,
                    shifted_pick,
                )
                attention_metrics = _distribution_metrics(attention, fractions)
                row = {
                    "model": args.model_name,
                    "split": split,
                    "sample_index": int(sample_index),
                    "sample_order": int(order),
                    "event_id": event_id,
                    "station_slot": int(station_slot),
                    "station_index": int(station_indices[station_slot]),
                    "pool": pool_name,
                    "post_p_bin": post_p_bin,
                    "sampling_rate": float(sampling_rate),
                    "trace_length": trace_length,
                    "token_length": token_length,
                    "pool_query_count": int(attention.shape[1]),
                    "station_valid_count": int(station_valid.sum()),
                    "visible_start_sample": float(shift),
                    "p_pick_shifted_sample": float(shifted_pick),
                    "p_pick_raw_sample": raw_pick,
                    "realtime_current_sample": current_sample,
                    "realtime_requested_elapsed_time": requested_elapsed,
                    "realtime_elapsed_time": elapsed,
                    "visible_seconds": float(visible_seconds),
                    "pre_p_seconds": float(pre_p_seconds),
                    "post_p_seconds": float(post_p_seconds),
                    "visible_token_equiv": float(
                        np.asarray(fractions["visible"]).sum()
                    ),
                    "pre_p_token_equiv": float(
                        np.asarray(fractions["pre_p"]).sum()
                    ),
                    "post_p_token_equiv": float(
                        np.asarray(fractions["post_p"]).sum()
                    ),
                    "padding_token_equiv": float(
                        np.asarray(fractions["padding"]).sum()
                    ),
                    "station_token_weight_mode": str(
                        getattr(raw_model, "station_token_weight_mode", "unknown")
                    ),
                    "token_weight_floor": float(call["token_weight_floor"]),
                    "token_weight_scale": float(call["token_weight_scale"]),
                }
                _add_query_aggregate(row, "attention", attention_metrics)
                prior = call.get("prior")
                if prior is not None:
                    _add_prior_metrics(row, _to_numpy(prior), fractions)
                rows.append(row)

        if (order + 1) % max(1, int(args.log_every)) == 0:
            print(
                f"[INFO] split={split}: processed {order + 1}/{len(indices)} "
                f"samples, station-pool rows={len(rows)}"
            )

    return rows, len(indices)


def _event_level_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["model"],
            row["split"],
            row["event_id"],
            row["pool"],
            row["post_p_bin"],
        )
        grouped[key].append(row)

    event_rows = []
    for key, group in grouped.items():
        model, split, event_id, pool, post_p_bin = key
        event_row = {
            "model": model,
            "split": split,
            "event_id": event_id,
            "pool": pool,
            "post_p_bin": post_p_bin,
            "station_record_count": len(group),
            "sample_count": len(set(row["sample_index"] for row in group)),
        }
        for metric in SUMMARY_METRICS:
            values = [row.get(metric) for row in group]
            stats = _summary(values)
            if stats["count"]:
                event_row[metric] = stats["mean"]
        event_rows.append(event_row)
    return event_rows


def _tidy_bin_summary(event_rows):
    grouped = defaultdict(list)
    station_counts = defaultdict(int)
    for row in event_rows:
        key = (row["model"], row["split"], row["pool"], row["post_p_bin"])
        grouped[key].append(row)
        station_counts[key] += int(row.get("station_record_count", 0))

    tidy = []
    for key in sorted(grouped):
        model, split, pool, post_p_bin = key
        group = grouped[key]
        for metric in SUMMARY_METRICS:
            stats = _summary([row.get(metric) for row in group])
            if not stats["count"]:
                continue
            tidy.append({
                "model": model,
                "split": split,
                "pool": pool,
                "post_p_bin": post_p_bin,
                "metric": metric,
                "event_count": stats["count"],
                "station_record_count": station_counts[key],
                "mean": stats["mean"],
                "std": stats["std"],
                "median": stats["median"],
                "q25": stats["q25"],
                "q75": stats["q75"],
                "min": stats["min"],
                "max": stats["max"],
            })
    return tidy


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Eval-only station-pool attention diagnostics normalized by each "
            "station's visible and post-P waveform duration."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model_name",
        default="",
        help="Short label written to outputs; defaults to checkpoint directory name.",
    )
    parser.add_argument("--diting_config", default="./diting/config/conf_reg.yml")
    parser.add_argument("--diting_pretrained", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="both", choices=("train", "val", "both"))
    parser.add_argument("--overfit_n", type=int, default=0)
    parser.add_argument(
        "--input_station_selection",
        default="config",
        choices=("config", "default", "random", "p_pick", "epidist"),
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--pool_names", default="base_x,x")
    parser.add_argument(
        "--post_p_bin_edges",
        default="0,1,3,5,10,20,40,90",
    )
    parser.add_argument("--sampling_rate", type=float, default=100.0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()
    args.pool_names = _parse_name_list(args.pool_names)
    args.post_p_bin_edges = _parse_float_list(args.post_p_bin_edges)
    if not args.pool_names:
        raise ValueError("--pool_names must contain at least one pool.")
    for name in args.pool_names:
        if name not in POOL_ATTRIBUTE_BY_NAME:
            raise ValueError(
                f"Unknown pool {name!r}; choose from "
                f"{sorted(POOL_ATTRIBUTE_BY_NAME)}"
            )
    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.config) as handle:
        config = json.load(handle)

    args.overfit_n = args.overfit_n or int(
        config.get("training_params", {}).get("overfit_n", 0)
    )
    args.device = torch.device(args.device)
    args.model_name = args.model_name or os.path.basename(
        os.path.dirname(os.path.abspath(args.checkpoint))
    )

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    diting_args = build_diting_args(
        args.diting_config,
        device=str(args.device),
        pretrained_override=args.diting_pretrained,
    )
    print("[INFO] building model")
    model = build_model_and_load(
        config,
        diting_args,
        args.checkpoint,
        args.device,
    )
    raw_model = model.module if hasattr(model, "module") else model
    adapter = _find_station_adapter(raw_model)

    print("[INFO] building train/validation datasets")
    datasets = build_datasets(
        config,
        overfit_n=args.overfit_n,
        input_station_selection=args.input_station_selection,
    )

    selected_splits = ("train", "val") if args.split == "both" else (args.split,)
    run_info = {
        "model": args.model_name,
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(args.checkpoint),
        "diting_config": os.path.abspath(args.diting_config),
        "diting_pretrained": args.diting_pretrained,
        "split": args.split,
        "overfit_n": args.overfit_n,
        "pool_names": args.pool_names,
        "post_p_bin_edges": args.post_p_bin_edges,
        "max_samples": args.max_samples,
        "start_index": args.start_index,
        "stride": args.stride,
        "station_token_weight_mode": str(
            getattr(raw_model, "station_token_weight_mode", "unknown")
        ),
        "token_weight_floor": float(
            getattr(raw_model, "token_weight_floor", float("nan"))
        ),
        "token_weight_scale": float(
            getattr(raw_model, "token_weight_scale", float("nan"))
        ),
    }
    split_info = {}

    with StationPoolCallRecorder(adapter, args.pool_names) as recorder:
        for split in selected_splits:
            rows, selected_sample_count = diagnose_split(
                model,
                recorder,
                datasets[split],
                split,
                config,
                args,
            )
            event_rows = _event_level_rows(rows)
            tidy_summary = _tidy_bin_summary(event_rows)

            station_path = os.path.join(
                args.output_dir,
                f"per_station_pool_{split}.csv",
            )
            event_path = os.path.join(
                args.output_dir,
                f"per_event_bin_{split}.csv",
            )
            summary_path = os.path.join(
                args.output_dir,
                f"summary_by_post_p_bin_{split}.csv",
            )
            _write_csv(station_path, rows)
            _write_csv(event_path, event_rows)
            _write_csv(summary_path, tidy_summary)
            split_info[split] = {
                "dataset_size": len(datasets[split]),
                "selected_sample_count": selected_sample_count,
                "station_pool_row_count": len(rows),
                "event_bin_row_count": len(event_rows),
                "summary_row_count": len(tidy_summary),
                "per_station_pool_csv": os.path.abspath(station_path),
                "per_event_bin_csv": os.path.abspath(event_path),
                "summary_by_post_p_bin_csv": os.path.abspath(summary_path),
            }
            print(
                f"[INFO] split={split}: wrote {len(rows)} station-pool rows, "
                f"{len(event_rows)} event-bin rows"
            )

    summary_json = os.path.join(args.output_dir, "summary.json")
    with open(summary_json, "w") as handle:
        json.dump(
            {
                "run_info": run_info,
                "splits": split_info,
            },
            handle,
            indent=2,
        )
    print(f"[INFO] wrote {summary_json}")


if __name__ == "__main__":
    main()
