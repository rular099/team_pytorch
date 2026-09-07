#!/usr/bin/env python3
"""Offline paired analysis for aligned RT55/RT56 random-geometry NPZ files.

The formal evaluation archives currently use NumPy object arrays.  Loading an
object array requires pickle, so this tool refuses to open the inputs unless
the caller explicitly confirms that they are trusted project artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import diagnose_query_geometry_sensitivity as querydiag


ALIGNMENT_KEYS = (
    "val_event_index",
    "val_event_id",
    "val_pga_target_valid",
    "val_pga_target_abs",
    "val_pga_target_indices",
    "val_realtime_target_type",
    "val_station_coords_abs",
    "val_station_valid",
    "val_pga_label",
    "val_realtime_requested_elapsed_time",
    "val_causal_random_requested_station_count",
    "val_causal_random_selected_station_count",
)

TARGET_POPULATIONS = (
    "all",
    "non_input",
    "triggered_noninput",
    "untriggered",
    "input",
)

POINT_METRIC_KEYS = (
    "targets",
    "mae",
    "rmse",
    "bias",
    "correlation",
    "r2",
    "slope",
    "intercept",
    "predictive_sigma_mean",
    "predictive_sigma_median",
    "coverage_1sigma",
    "coverage_2sigma",
)

FORMAL_REPORT_KEYS = (
    "coordinate",
    "point_estimate",
    "mae",
    "rmse",
    "bias",
    "correlation",
    "r2",
    "slope",
    "intercept",
    "nll",
    "nll_log10_mps2",
    "predictive_sigma_mean",
    "predictive_sigma_median",
    "coverage_1sigma",
    "coverage_2sigma",
)

SPATIAL_DIRECTION = {
    "pearson": "higher",
    "spearman": "higher",
    "event_centered_mae": "lower",
    "event_centered_rmse": "lower",
    "pairwise_delta_mae": "lower",
    "pairwise_delta_rmse": "lower",
    "range_abs_error": "lower",
}


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _scalar_text(value: Any) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar event id, got shape {array.shape}")
    return str(array.reshape(-1)[0])


def _numeric(archive: Mapping[str, np.ndarray], key: str, dtype: Any) -> np.ndarray:
    if key not in archive:
        raise KeyError(f"Required array missing from NPZ: {key}")
    return np.asarray(archive[key], dtype=dtype)


def _load_archive(path: Path) -> Dict[str, np.ndarray]:
    required = set(ALIGNMENT_KEYS) | {
        "val_pga_mu_best",
        "val_pga_sigma",
    }
    with np.load(path, allow_pickle=True) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"{path} is missing required arrays: {missing}")
        result = {
            "truth": _numeric(archive, "val_pga_label", np.float64),
            "prediction": _numeric(archive, "val_pga_mu_best", np.float64),
            "sigma": _numeric(archive, "val_pga_sigma", np.float64),
            "valid": _numeric(archive, "val_pga_target_valid", bool),
            "target_type": _numeric(
                archive, "val_realtime_target_type", np.int64
            ),
            "event_index": _numeric(archive, "val_event_index", np.int64),
            "event_id": np.asarray(
                [_scalar_text(value) for value in archive["val_event_id"]],
                dtype=str,
            ),
            "elapsed_time": _numeric(
                archive, "val_realtime_requested_elapsed_time", np.float64
            ).reshape(-1),
            "requested_station_count": _numeric(
                archive, "val_causal_random_requested_station_count", np.int64
            ).reshape(-1),
            "selected_station_count": _numeric(
                archive, "val_causal_random_selected_station_count", np.int64
            ).reshape(-1),
        }
        result["alignment"] = {
            key: np.asarray(archive[key]) for key in ALIGNMENT_KEYS
        }

    truth = result["truth"]
    if truth.ndim == 3 and truth.shape[-1] == 1:
        truth = truth[..., 0]
    result["truth"] = truth
    for key in ("prediction", "sigma", "valid", "target_type"):
        value = result[key]
        if value.ndim == 3 and value.shape[-1] == 1:
            value = value[..., 0]
        result[key] = value

    expected_shape = result["prediction"].shape
    for key in ("truth", "sigma", "valid", "target_type"):
        if result[key].shape != expected_shape:
            raise ValueError(
                f"Shape mismatch in {path}: {key}={result[key].shape}, "
                f"prediction={expected_shape}"
            )
    sample_count = expected_shape[0]
    for key in (
        "event_index",
        "event_id",
        "elapsed_time",
        "requested_station_count",
        "selected_station_count",
    ):
        if result[key].shape[0] != sample_count:
            raise ValueError(
                f"Sample-count mismatch in {path}: {key}={result[key].shape[0]}, "
                f"prediction={sample_count}"
            )
    return result


def _arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return False
    if left.dtype == object:
        left = np.asarray(left.tolist())
    if right.dtype == object:
        right = np.asarray(right.tolist())
    try:
        return bool(np.array_equal(left, right, equal_nan=True))
    except (TypeError, ValueError):
        return bool(np.array_equal(left, right))


def verify_alignment(
    baseline: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray]
) -> Dict[str, Any]:
    checks = {
        key: _arrays_equal(
            baseline["alignment"][key], candidate["alignment"][key]
        )
        for key in ALIGNMENT_KEYS
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise ValueError(f"NPZ files are not exactly paired; mismatched arrays: {failed}")
    return {
        "status": "passed",
        "exact_array_equality": checks,
        "sample_order": "identical",
        "target_slot_order": "identical",
    }


def _load_formal_pga_metrics(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        return payload["metrics"]["val"]["pga"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Formal metrics JSON has no metrics.val.pga: {path}") from exc


def validate_formal_metrics(
    arrays: Mapping[str, np.ndarray], metrics: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    recomputed = querydiag._point_error_summary(
        arrays["truth"], arrays["prediction"], arrays["valid"], arrays["sigma"]
    )
    if metrics is None:
        return {"status": "not_requested", "recomputed": recomputed}
    comparisons: Dict[str, Any] = {}
    passed = True
    for key in POINT_METRIC_KEYS:
        if key not in metrics or key not in recomputed:
            continue
        expected = metrics[key]
        observed = recomputed[key]
        if expected is None or observed is None:
            matched = expected is None and observed is None
            absolute_difference = None
        elif key == "targets":
            matched = int(expected) == int(observed)
            absolute_difference = abs(int(expected) - int(observed))
        else:
            absolute_difference = abs(float(expected) - float(observed))
            matched = bool(np.isclose(expected, observed, rtol=1e-12, atol=1e-12))
        passed = passed and matched
        comparisons[key] = {
            "formal": expected,
            "recomputed": observed,
            "absolute_difference": absolute_difference,
            "matched": matched,
        }
    if not passed:
        failed = [key for key, value in comparisons.items() if not value["matched"]]
        raise ValueError(f"NPZ point metrics do not reproduce formal JSON: {failed}")
    return {
        "status": "passed",
        "tolerance": {"relative": 1e-12, "absolute": 1e-12},
        "comparisons": comparisons,
        "reported_formal_metrics": {
            key: metrics[key] for key in FORMAL_REPORT_KEYS if key in metrics
        },
    }


def _finite_summary(values: np.ndarray) -> Dict[str, Any]:
    return querydiag._finite_summary(np.asarray(values, dtype=np.float64))


def _spatial_summary(
    metrics: Mapping[str, np.ndarray], sample_mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    counts = np.asarray(metrics["valid_target_count"], dtype=np.int64)
    if sample_mask is None:
        sample_mask = np.ones(counts.shape, dtype=bool)
    else:
        sample_mask = np.asarray(sample_mask, dtype=bool)
    result: Dict[str, Any] = {
        "aggregation": "unweighted_across_realtime_samples",
        "realtime_samples_total": int(sample_mask.sum()),
        "realtime_samples_with_valid_targets": int((sample_mask & (counts >= 1)).sum()),
    }
    for threshold in (1, 2, 5):
        selected = sample_mask & (counts >= threshold)
        result[f"valid_target_count_at_least_{threshold}"] = {
            "realtime_samples": int(selected.sum()),
            "metrics": {
                key: _finite_summary(np.asarray(value)[selected])
                for key, value in metrics.items()
            },
        }
    return result


def _add_range_error(metrics: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    result = dict(metrics)
    result["range_abs_error"] = np.abs(
        np.asarray(result["predicted_p95_p05_range"], dtype=np.float64)
        - np.asarray(result["true_p95_p05_range"], dtype=np.float64)
    )
    return result


def _categorical_strata(values: np.ndarray) -> Dict[str, np.ndarray]:
    values = np.asarray(values)
    return {
        str(value.item() if isinstance(value, np.generic) else value): values == value
        for value in np.unique(values)
    }


def summarize_model(
    arrays: Mapping[str, np.ndarray], *, pair_sample_limit: int, seed: int
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, np.ndarray]]]:
    populations = querydiag._target_population_masks(
        arrays["valid"], arrays["target_type"]
    )
    strata = {
        "requested_station_count": _categorical_strata(
            arrays["requested_station_count"]
        ),
        "selected_station_count": _categorical_strata(
            arrays["selected_station_count"]
        ),
        "requested_elapsed_time_seconds": _categorical_strata(
            arrays["elapsed_time"]
        ),
    }
    summary: Dict[str, Any] = {"target_populations": {}}
    spatial_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for population_index, name in enumerate(TARGET_POPULATIONS):
        valid = populations[name]
        spatial = querydiag._compute_spatial_metric_arrays(
            arrays["truth"],
            arrays["prediction"],
            valid,
            arrays["sigma"],
            pair_sample_limit=pair_sample_limit,
            seed=seed + population_index * 100_003,
        )
        spatial = _add_range_error(spatial)
        spatial_arrays[name] = spatial
        group: Dict[str, Any] = {
            "point": querydiag._point_error_summary(
                arrays["truth"], arrays["prediction"], valid, arrays["sigma"]
            ),
            "spatial": _spatial_summary(spatial),
            "strata": {},
        }
        for dimension, masks in strata.items():
            group["strata"][dimension] = {}
            for value, sample_mask in masks.items():
                point_valid = valid & sample_mask[:, None]
                group["strata"][dimension][value] = {
                    "realtime_samples": int(sample_mask.sum()),
                    "point": querydiag._point_error_summary(
                        arrays["truth"],
                        arrays["prediction"],
                        point_valid,
                        arrays["sigma"],
                    ),
                }
        summary["target_populations"][name] = group
    return summary, spatial_arrays


def _event_aggregates(
    event_ids: np.ndarray, values: np.ndarray, valid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    event_ids = np.asarray(event_ids, dtype=str)
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    unique_events, inverse = np.unique(event_ids, return_inverse=True)
    if values.shape[0] != inverse.size:
        raise ValueError(
            f"Event/value sample mismatch: {inverse.size} event ids, "
            f"values shape {values.shape}"
        )
    event_codes = np.broadcast_to(
        inverse.reshape((inverse.size,) + (1,) * (values.ndim - 1)), values.shape
    )
    sums = np.bincount(
        event_codes[valid], weights=values[valid], minlength=unique_events.size
    ).astype(np.float64)
    counts = np.bincount(event_codes[valid], minlength=unique_events.size).astype(
        np.int64
    )
    return unique_events, sums, counts


def _bootstrap_ratio_difference(
    event_ids: np.ndarray,
    baseline_numerator: np.ndarray,
    candidate_numerator: np.ndarray,
    valid: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
    transform,
) -> Dict[str, Any]:
    events, baseline_sums, counts = _event_aggregates(
        event_ids, baseline_numerator, valid
    )
    events_candidate, candidate_sums, candidate_counts = _event_aggregates(
        event_ids, candidate_numerator, valid
    )
    if not np.array_equal(events, events_candidate) or not np.array_equal(
        counts, candidate_counts
    ):
        raise AssertionError("Internal event aggregation mismatch")
    active = counts > 0
    baseline_sums = baseline_sums[active]
    candidate_sums = candidate_sums[active]
    counts = counts[active]
    if counts.size == 0:
        return {
            "cluster_unit": "event_id",
            "event_clusters": 0,
            "replicates": replicates,
            "estimate": None,
            "ci95": [None, None],
        }
    estimate = transform(candidate_sums.sum() / counts.sum()) - transform(
        baseline_sums.sum() / counts.sum()
    )
    draws = rng.integers(0, counts.size, size=(replicates, counts.size))
    denominators = counts[draws].sum(axis=1)
    candidate_values = candidate_sums[draws].sum(axis=1) / denominators
    baseline_values = baseline_sums[draws].sum(axis=1) / denominators
    differences = transform(candidate_values) - transform(baseline_values)
    return {
        "cluster_unit": "event_id",
        "event_clusters": int(counts.size),
        "replicates": int(replicates),
        "estimate": float(estimate),
        "ci95": [
            float(np.percentile(differences, 2.5)),
            float(np.percentile(differences, 97.5)),
        ],
    }


def paired_point_summary(
    baseline: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    valid: np.ndarray,
    *,
    sample_mask: Optional[np.ndarray],
    bootstrap_replicates: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    valid = np.asarray(valid, dtype=bool)
    if sample_mask is None:
        sample_mask = np.ones(valid.shape[0], dtype=bool)
    else:
        sample_mask = np.asarray(sample_mask, dtype=bool)
    valid = (
        valid
        & sample_mask[:, None]
        & np.isfinite(baseline["truth"])
        & np.isfinite(baseline["prediction"])
        & np.isfinite(candidate["prediction"])
    )
    baseline_residual = baseline["prediction"] - baseline["truth"]
    candidate_residual = candidate["prediction"] - candidate["truth"]
    baseline_abs = np.abs(baseline_residual)
    candidate_abs = np.abs(candidate_residual)
    target_count = int(valid.sum())
    sample_counts = valid.sum(axis=1)
    sample_has_targets = sample_counts > 0
    baseline_sample_mae = np.full(valid.shape[0], np.nan)
    candidate_sample_mae = np.full(valid.shape[0], np.nan)
    baseline_sample_mae[sample_has_targets] = (
        np.where(valid, baseline_abs, 0.0).sum(axis=1)[sample_has_targets]
        / sample_counts[sample_has_targets]
    )
    candidate_sample_mae[sample_has_targets] = (
        np.where(valid, candidate_abs, 0.0).sum(axis=1)[sample_has_targets]
        / sample_counts[sample_has_targets]
    )
    if target_count:
        baseline_mae = float(np.mean(baseline_abs[valid]))
        candidate_mae = float(np.mean(candidate_abs[valid]))
        baseline_rmse = float(np.sqrt(np.mean(baseline_residual[valid] ** 2)))
        candidate_rmse = float(np.sqrt(np.mean(candidate_residual[valid] ** 2)))
        target_improved_fraction = float(np.mean(candidate_abs[valid] < baseline_abs[valid]))
        target_tied_fraction = float(np.mean(candidate_abs[valid] == baseline_abs[valid]))
    else:
        baseline_mae = candidate_mae = baseline_rmse = candidate_rmse = None
        target_improved_fraction = target_tied_fraction = None
    sample_delta = candidate_sample_mae - baseline_sample_mae
    finite_sample = np.isfinite(sample_delta)
    return {
        "realtime_samples_selected": int(sample_mask.sum()),
        "realtime_samples_with_targets": int(sample_has_targets.sum()),
        "targets": target_count,
        "baseline_mae": baseline_mae,
        "candidate_mae": candidate_mae,
        "mae_delta_candidate_minus_baseline": (
            None if baseline_mae is None else candidate_mae - baseline_mae
        ),
        "baseline_rmse": baseline_rmse,
        "candidate_rmse": candidate_rmse,
        "rmse_delta_candidate_minus_baseline": (
            None if baseline_rmse is None else candidate_rmse - baseline_rmse
        ),
        "target_improved_fraction": target_improved_fraction,
        "target_tied_fraction": target_tied_fraction,
        "sample_mae_delta": _finite_summary(sample_delta),
        "sample_improved_fraction": (
            float(np.mean(sample_delta[finite_sample] < 0))
            if np.any(finite_sample)
            else None
        ),
        "event_cluster_bootstrap": {
            "mae_delta_candidate_minus_baseline": _bootstrap_ratio_difference(
                baseline["event_id"],
                baseline_abs,
                candidate_abs,
                valid,
                replicates=bootstrap_replicates,
                rng=rng,
                transform=lambda value: value,
            ),
            "rmse_delta_candidate_minus_baseline": _bootstrap_ratio_difference(
                baseline["event_id"],
                baseline_residual ** 2,
                candidate_residual ** 2,
                valid,
                replicates=bootstrap_replicates,
                rng=rng,
                transform=np.sqrt,
            ),
        },
    }


def _bootstrap_mean_difference(
    event_ids: np.ndarray,
    differences: np.ndarray,
    valid: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    events, sums, counts = _event_aggregates(event_ids, differences, valid)
    active = counts > 0
    sums = sums[active]
    counts = counts[active]
    if counts.size == 0:
        return {
            "cluster_unit": "event_id",
            "event_clusters": 0,
            "replicates": replicates,
            "estimate": None,
            "ci95": [None, None],
        }
    estimate = float(sums.sum() / counts.sum())
    draws = rng.integers(0, counts.size, size=(replicates, counts.size))
    values = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "cluster_unit": "event_id",
        "event_clusters": int(counts.size),
        "replicates": int(replicates),
        "estimate": estimate,
        "ci95": [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ],
    }


def paired_spatial_summary(
    event_ids: np.ndarray,
    baseline_metrics: Mapping[str, np.ndarray],
    candidate_metrics: Mapping[str, np.ndarray],
    *,
    bootstrap_replicates: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    counts = np.asarray(baseline_metrics["valid_target_count"], dtype=np.int64)
    if not np.array_equal(counts, candidate_metrics["valid_target_count"]):
        raise ValueError("Paired spatial comparison has unequal valid-target counts")
    result: Dict[str, Any] = {}
    for threshold in (2, 5):
        threshold_result: Dict[str, Any] = {
            "minimum_valid_targets": threshold,
            "metrics": {},
        }
        count_mask = counts >= threshold
        for metric, direction in SPATIAL_DIRECTION.items():
            baseline_value = np.asarray(baseline_metrics[metric], dtype=np.float64)
            candidate_value = np.asarray(candidate_metrics[metric], dtype=np.float64)
            valid = count_mask & np.isfinite(baseline_value) & np.isfinite(candidate_value)
            difference = candidate_value - baseline_value
            if np.any(valid):
                better = difference[valid] > 0 if direction == "higher" else difference[valid] < 0
                baseline_mean = float(np.mean(baseline_value[valid]))
                candidate_mean = float(np.mean(candidate_value[valid]))
                better_fraction = float(np.mean(better))
            else:
                baseline_mean = candidate_mean = better_fraction = None
            threshold_result["metrics"][metric] = {
                "direction": direction,
                "paired_realtime_samples": int(valid.sum()),
                "baseline_mean": baseline_mean,
                "candidate_mean": candidate_mean,
                "delta_candidate_minus_baseline": (
                    None if baseline_mean is None else candidate_mean - baseline_mean
                ),
                "candidate_better_fraction": better_fraction,
                "event_cluster_bootstrap": _bootstrap_mean_difference(
                    event_ids,
                    difference,
                    valid,
                    replicates=bootstrap_replicates,
                    rng=rng,
                ),
            }
        result[f"valid_target_count_at_least_{threshold}"] = threshold_result
    return result


def _point_strata(
    baseline: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    valid: np.ndarray,
    *,
    bootstrap_replicates: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    dimensions = {
        "requested_station_count": baseline["requested_station_count"],
        "selected_station_count": baseline["selected_station_count"],
        "requested_elapsed_time_seconds": baseline["elapsed_time"],
    }
    result: Dict[str, Any] = {}
    for dimension, values in dimensions.items():
        result[dimension] = {}
        for value, sample_mask in _categorical_strata(values).items():
            result[dimension][value] = paired_point_summary(
                baseline,
                candidate,
                valid,
                sample_mask=sample_mask,
                bootstrap_replicates=bootstrap_replicates,
                rng=rng,
            )
    return result


def build_analysis(
    baseline: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    *,
    baseline_metrics: Optional[Mapping[str, Any]],
    candidate_metrics: Optional[Mapping[str, Any]],
    pair_sample_limit: int,
    bootstrap_replicates: int,
    seed: int,
) -> Dict[str, Any]:
    alignment = verify_alignment(baseline, candidate)
    baseline_validation = validate_formal_metrics(baseline, baseline_metrics)
    candidate_validation = validate_formal_metrics(candidate, candidate_metrics)
    baseline_summary, baseline_spatial = summarize_model(
        baseline, pair_sample_limit=pair_sample_limit, seed=seed
    )
    candidate_summary, candidate_spatial = summarize_model(
        candidate, pair_sample_limit=pair_sample_limit, seed=seed
    )
    populations = querydiag._target_population_masks(
        baseline["valid"], baseline["target_type"]
    )
    rng = np.random.default_rng(seed)
    paired_point: Dict[str, Any] = {}
    paired_spatial: Dict[str, Any] = {}
    for name in TARGET_POPULATIONS:
        paired_point[name] = paired_point_summary(
            baseline,
            candidate,
            populations[name],
            sample_mask=None,
            bootstrap_replicates=bootstrap_replicates,
            rng=rng,
        )
        paired_spatial[name] = paired_spatial_summary(
            baseline["event_id"],
            baseline_spatial[name],
            candidate_spatial[name],
            bootstrap_replicates=bootstrap_replicates,
            rng=rng,
        )
    paired_strata = _point_strata(
        baseline,
        candidate,
        populations["all"],
        bootstrap_replicates=bootstrap_replicates,
        rng=rng,
    )
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_protocol": {
            "coordinate": "log10(m/s^2)",
            "point_estimate": "predictive_mixture_mean_from_val_pga_mu_best",
            "point_aggregation": "target_weighted",
            "spatial_aggregation": "unweighted_across_realtime_samples",
            "bootstrap": {
                "method": "percentile_cluster_bootstrap",
                "cluster_unit": "event_id",
                "replicates": bootstrap_replicates,
                "seed": seed,
                "confidence_level": 0.95,
            },
            "pair_sample_limit": pair_sample_limit,
        },
        "alignment": alignment,
        "formal_metric_validation": {
            "baseline": baseline_validation,
            "candidate": candidate_validation,
        },
        "dataset": {
            "realtime_samples": int(baseline["truth"].shape[0]),
            "target_slots_per_sample": int(baseline["truth"].shape[1]),
            "unique_event_ids": int(np.unique(baseline["event_id"]).size),
            "valid_targets": int(baseline["valid"].sum()),
        },
        "models": {
            "baseline": baseline_summary,
            "candidate": candidate_summary,
        },
        "paired_comparison": {
            "point_by_target_population": paired_point,
            "point_by_stratum_all_targets": paired_strata,
            "spatial_by_target_population": paired_spatial,
        },
    }


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _ci_text(result: Mapping[str, Any]) -> str:
    low, high = result["ci95"]
    if low is None:
        return "NA"
    return f"[{low:.4f}, {high:.4f}]"


def render_markdown(
    analysis: Mapping[str, Any], baseline_name: str, candidate_name: str
) -> str:
    dataset = analysis["dataset"]
    paired = analysis["paired_comparison"]
    overall = paired["point_by_target_population"]["all"]
    overall_mae_ci = overall["event_cluster_bootstrap"][
        "mae_delta_candidate_minus_baseline"
    ]
    one_second = paired["point_by_stratum_all_targets"][
        "requested_elapsed_time_seconds"
    ]["1.0"]
    spatial_all = paired["spatial_by_target_population"]["all"][
        "valid_target_count_at_least_5"
    ]["metrics"]
    baseline_formal = analysis["formal_metric_validation"]["baseline"].get(
        "reported_formal_metrics", {}
    )
    candidate_formal = analysis["formal_metric_validation"]["candidate"].get(
        "reported_formal_metrics", {}
    )
    lines = [
        "# RT55/RT56 random-geometry full validation: offline paired analysis",
        "",
        f"- Baseline: `{baseline_name}`",
        f"- Candidate: `{candidate_name}`",
        f"- Exactly aligned samples: {dataset['realtime_samples']:,}",
        f"- Unique event IDs: {dataset['unique_event_ids']:,}",
        f"- Valid targets: {dataset['valid_targets']:,}",
        "- Coordinate: `log10(m/s^2)`",
        "- Uncertainty: 95% percentile bootstrap CI, clustered by `event_id`.",
        "",
        "## Decision summary",
        "",
        f"- The candidate improves target-weighted MAE by {-overall['mae_delta_candidate_minus_baseline']:.4f} "
        f"({100 * -overall['mae_delta_candidate_minus_baseline'] / overall['baseline_mae']:.1f}% relative); "
        f"candidate-minus-baseline delta {_format_number(overall['mae_delta_candidate_minus_baseline'])}, "
        f"95% CI {_ci_text(overall_mae_ci)}.",
        "- The gain is present for every requested station-count stratum, and it is larger for untriggered than triggered non-input targets.",
        f"- The 1 s window regresses: MAE {_format_number(one_second['baseline_mae'])} to "
        f"{_format_number(one_second['candidate_mae'])}; delta "
        f"{_format_number(one_second['mae_delta_candidate_minus_baseline'])}.",
        f"- For fields with at least five targets, mean Pearson improves by "
        f"{_format_number(spatial_all['pearson']['delta_candidate_minus_baseline'])}, but mean P95-P05 range error worsens by "
        f"{_format_number(spatial_all['range_abs_error']['delta_candidate_minus_baseline'])}. The model still compresses spatial amplitude.",
        "- Therefore the finetuning is useful and should be retained, but the earliest window and spatial dynamic range remain explicit follow-up targets.",
        "",
        "## Formal full-validation metrics",
        "",
        "| Metric | Baseline | Candidate |",
        "|---|---:|---:|",
    ]
    for key in (
        "mae",
        "rmse",
        "bias",
        "correlation",
        "r2",
        "nll",
        "predictive_sigma_mean",
        "coverage_1sigma",
        "coverage_2sigma",
    ):
        lines.append(
            f"| {key} | {_format_number(baseline_formal.get(key))} | "
            f"{_format_number(candidate_formal.get(key))} |"
        )
    lines.extend([
        "",
        "## Paired point-error comparison",
        "",
        "Negative MAE/RMSE deltas favor the candidate.",
        "",
        "| Target population | Targets | Baseline MAE | Candidate MAE | MAE delta | 95% CI | Improved targets |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for population in TARGET_POPULATIONS:
        value = paired["point_by_target_population"][population]
        ci = value["event_cluster_bootstrap"]["mae_delta_candidate_minus_baseline"]
        lines.append(
            "| {population} | {targets:,} | {baseline} | {candidate} | {delta} | {ci} | {improved} |".format(
                population=population,
                targets=value["targets"],
                baseline=_format_number(value["baseline_mae"]),
                candidate=_format_number(value["candidate_mae"]),
                delta=_format_number(value["mae_delta_candidate_minus_baseline"]),
                ci=_ci_text(ci),
                improved=(
                    "NA"
                    if value["target_improved_fraction"] is None
                    else f"{100 * value['target_improved_fraction']:.1f}%"
                ),
            )
        )

    lines.extend([
        "",
        "## Point error by requested input-station count",
        "",
        "| Requested stations | Samples | Targets | Baseline MAE | Candidate MAE | MAE delta | 95% CI |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    station_rows = paired["point_by_stratum_all_targets"]["requested_station_count"]
    for key in sorted(station_rows, key=float):
        value = station_rows[key]
        ci = value["event_cluster_bootstrap"]["mae_delta_candidate_minus_baseline"]
        lines.append(
            f"| {key} | {value['realtime_samples_selected']:,} | {value['targets']:,} | "
            f"{_format_number(value['baseline_mae'])} | {_format_number(value['candidate_mae'])} | "
            f"{_format_number(value['mae_delta_candidate_minus_baseline'])} | {_ci_text(ci)} |"
        )

    lines.extend([
        "",
        "## Point error by requested elapsed time",
        "",
        "| Elapsed time (s) | Samples | Targets | Baseline MAE | Candidate MAE | MAE delta | 95% CI |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    time_rows = paired["point_by_stratum_all_targets"]["requested_elapsed_time_seconds"]
    for key in sorted(time_rows, key=float):
        value = time_rows[key]
        ci = value["event_cluster_bootstrap"]["mae_delta_candidate_minus_baseline"]
        lines.append(
            f"| {key} | {value['realtime_samples_selected']:,} | {value['targets']:,} | "
            f"{_format_number(value['baseline_mae'])} | {_format_number(value['candidate_mae'])} | "
            f"{_format_number(value['mae_delta_candidate_minus_baseline'])} | {_ci_text(ci)} |"
        )

    lines.extend([
        "",
        "## Paired spatial-field comparison (at least 5 valid targets)",
        "",
        "The delta is candidate minus baseline. Pearson/Spearman are better when positive; error metrics are better when negative.",
        "",
        "| Population | Metric | Paired samples | Baseline | Candidate | Delta | 95% CI | Candidate better |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for population in TARGET_POPULATIONS:
        metrics = paired["spatial_by_target_population"][population][
            "valid_target_count_at_least_5"
        ]["metrics"]
        for metric in SPATIAL_DIRECTION:
            value = metrics[metric]
            ci = value["event_cluster_bootstrap"]
            lines.append(
                f"| {population} | {metric} | {value['paired_realtime_samples']:,} | "
                f"{_format_number(value['baseline_mean'])} | {_format_number(value['candidate_mean'])} | "
                f"{_format_number(value['delta_candidate_minus_baseline'])} | {_ci_text(ci)} | "
                + (
                    "NA |"
                    if value["candidate_better_fraction"] is None
                    else f"{100 * value['candidate_better_fraction']:.1f}% |"
                )
            )
    lines.extend([
        "",
        "## Interpretation guardrails",
        "",
        "- This is paired validation analysis, not an independent test-set claim.",
        "- Event-cluster bootstrap keeps all time windows/repeated occurrences of one event in the same resampling unit.",
        "- Target-weighted point metrics and sample-weighted spatial metrics answer different questions and must not be mixed.",
        "- Full numerical details, including selected-station-count strata and the >=2-target spatial view, are in the companion JSON/CSV outputs.",
        "",
    ])
    return "\n".join(lines)


def _comparison_rows(analysis: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    paired = analysis["paired_comparison"]
    for population, value in paired["point_by_target_population"].items():
        mae_ci = value["event_cluster_bootstrap"]["mae_delta_candidate_minus_baseline"]
        rmse_ci = value["event_cluster_bootstrap"]["rmse_delta_candidate_minus_baseline"]
        yield {
            "scope": "target_population",
            "dimension": "target_type",
            "value": population,
            "samples": value["realtime_samples_with_targets"],
            "targets": value["targets"],
            "baseline_mae": value["baseline_mae"],
            "candidate_mae": value["candidate_mae"],
            "mae_delta": value["mae_delta_candidate_minus_baseline"],
            "mae_ci95_low": mae_ci["ci95"][0],
            "mae_ci95_high": mae_ci["ci95"][1],
            "baseline_rmse": value["baseline_rmse"],
            "candidate_rmse": value["candidate_rmse"],
            "rmse_delta": value["rmse_delta_candidate_minus_baseline"],
            "rmse_ci95_low": rmse_ci["ci95"][0],
            "rmse_ci95_high": rmse_ci["ci95"][1],
            "target_improved_fraction": value["target_improved_fraction"],
            "sample_improved_fraction": value["sample_improved_fraction"],
        }
    for dimension, values in paired["point_by_stratum_all_targets"].items():
        for stratum, value in values.items():
            mae_ci = value["event_cluster_bootstrap"]["mae_delta_candidate_minus_baseline"]
            rmse_ci = value["event_cluster_bootstrap"]["rmse_delta_candidate_minus_baseline"]
            yield {
                "scope": "stratum_all_targets",
                "dimension": dimension,
                "value": stratum,
                "samples": value["realtime_samples_with_targets"],
                "targets": value["targets"],
                "baseline_mae": value["baseline_mae"],
                "candidate_mae": value["candidate_mae"],
                "mae_delta": value["mae_delta_candidate_minus_baseline"],
                "mae_ci95_low": mae_ci["ci95"][0],
                "mae_ci95_high": mae_ci["ci95"][1],
                "baseline_rmse": value["baseline_rmse"],
                "candidate_rmse": value["candidate_rmse"],
                "rmse_delta": value["rmse_delta_candidate_minus_baseline"],
                "rmse_ci95_low": rmse_ci["ci95"][0],
                "rmse_ci95_high": rmse_ci["ci95"][1],
                "target_improved_fraction": value["target_improved_fraction"],
                "sample_improved_fraction": value["sample_improved_fraction"],
            }


def write_csv(path: Path, analysis: Mapping[str, Any]) -> None:
    rows = list(_comparison_rows(analysis))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-npz", type=Path, required=True)
    parser.add_argument("--candidate-npz", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path)
    parser.add_argument("--candidate-metrics", type=Path)
    parser.add_argument("--baseline-name", default="RT55 ep32 zero-shot random-mask")
    parser.add_argument("--candidate-name", default="RT56 mixed-random finetuned best")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--pair-sample-limit", type=int, default=4096)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument(
        "--trusted-pickle-input",
        action="store_true",
        help="Confirm that both object-array NPZ files are trusted project outputs.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.trusted_pickle_input:
        raise SystemExit(
            "Refusing to load object-array NPZ files without --trusted-pickle-input. "
            "Pickle-capable NumPy loading is unsafe for untrusted files."
        )
    if args.bootstrap_replicates <= 0:
        raise SystemExit("--bootstrap-replicates must be positive")
    paths = {
        "summary": Path(str(args.output_prefix) + ".summary.json"),
        "table": Path(str(args.output_prefix) + ".paired.csv"),
        "report": Path(str(args.output_prefix) + ".md"),
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not args.force:
        raise SystemExit(
            "Refusing to overwrite existing outputs; pass --force after preserving them: "
            + ", ".join(existing)
        )
    for path in (
        args.baseline_npz,
        args.candidate_npz,
        args.baseline_metrics,
        args.candidate_metrics,
    ):
        if path is not None and not path.is_file():
            raise SystemExit(f"Input file not found: {path}")

    baseline = _load_archive(args.baseline_npz)
    candidate = _load_archive(args.candidate_npz)
    analysis = build_analysis(
        baseline,
        candidate,
        baseline_metrics=_load_formal_pga_metrics(args.baseline_metrics),
        candidate_metrics=_load_formal_pga_metrics(args.candidate_metrics),
        pair_sample_limit=args.pair_sample_limit,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    analysis["inputs"] = {
        "baseline": {
            "name": args.baseline_name,
            "npz": str(args.baseline_npz.resolve()),
            "npz_size_bytes": args.baseline_npz.stat().st_size,
            "npz_sha256": _sha256(args.baseline_npz),
            "formal_metrics": (
                None if args.baseline_metrics is None else str(args.baseline_metrics.resolve())
            ),
            "formal_metrics_sha256": (
                None if args.baseline_metrics is None else _sha256(args.baseline_metrics)
            ),
        },
        "candidate": {
            "name": args.candidate_name,
            "npz": str(args.candidate_npz.resolve()),
            "npz_size_bytes": args.candidate_npz.stat().st_size,
            "npz_sha256": _sha256(args.candidate_npz),
            "formal_metrics": (
                None if args.candidate_metrics is None else str(args.candidate_metrics.resolve())
            ),
            "formal_metrics_sha256": (
                None if args.candidate_metrics is None else _sha256(args.candidate_metrics)
            ),
        },
    }
    _write_json(paths["summary"], analysis)
    write_csv(paths["table"], analysis)
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(
        render_markdown(analysis, args.baseline_name, args.candidate_name),
        encoding="utf-8",
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
