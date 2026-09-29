#!/usr/bin/env python3
"""Aggregate V01 validation NPZ files and paired event-cluster intervals."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


NAME_RE = re.compile(
    r"(?P<checkpoint>vfull|vmissing|apair)__(?P<view>vfull|vmissing|apair)__(?P<protocol>normal|random)\.npz$"
)


def _stack(data, key):
    values = data[key]
    try:
        return np.asarray(values.tolist()) if values.dtype == object else np.asarray(values)
    except (ValueError, TypeError):
        return np.stack([np.asarray(value) for value in values])


def load_rows(path: Path) -> pd.DataFrame:
    match = NAME_RE.search(path.name)
    if not match:
        raise ValueError(f"Unexpected V01 NPZ name: {path.name}")
    with np.load(path, allow_pickle=True) as data:
        prefix = "val_"
        valid = _stack(data, prefix + "pga_target_valid").astype(bool)
        label = _stack(data, prefix + "pga_label").astype(float).squeeze(-1)
        if prefix + "pga_mu_best" in data:
            pred = _stack(data, prefix + "pga_mu_best").astype(float)
        else:
            raise KeyError(f"{path} lacks val_pga_mu_best")
        elapsed = _stack(data, prefix + "realtime_requested_elapsed_time").astype(float).reshape(-1)
        target_index = _stack(data, prefix + "pga_target_indices").astype(int)
        event_key = prefix + "event_id" if prefix + "event_id" in data else prefix + "event_index"
        events = _stack(data, event_key).reshape(-1).astype(str)
    row_index, target_slot = np.where(valid & np.isfinite(label) & np.isfinite(pred))
    frame = pd.DataFrame({
        "event_id": events[row_index],
        "decision_time_s": elapsed[row_index],
        "target_index": target_index[row_index, target_slot],
        "truth": label[row_index, target_slot],
        "prediction": pred[row_index, target_slot],
    })
    for key, value in match.groupdict().items():
        frame[key] = value
    frame["abs_error"] = np.abs(frame["prediction"] - frame["truth"])
    return frame


def metrics(frame: pd.DataFrame) -> dict[str, object]:
    error = frame["prediction"].to_numpy() - frame["truth"].to_numpy()
    return {
        "checkpoint": frame["checkpoint"].iloc[0],
        "view": frame["view"].iloc[0],
        "protocol": frame["protocol"].iloc[0],
        "targets": int(len(frame)),
        "events": int(frame["event_id"].nunique()),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)),
        "p90_abs_error": float(np.quantile(np.abs(error), 0.90)),
        "p95_abs_error": float(np.quantile(np.abs(error), 0.95)),
        "within_0p1": float(np.mean(np.abs(error) <= 0.1)),
        "within_0p2": float(np.mean(np.abs(error) <= 0.2)),
    }


def paired_cluster_ci(left: pd.DataFrame, right: pd.DataFrame, draws: int, seed: int) -> dict[str, object]:
    keys = ["event_id", "decision_time_s", "target_index", "protocol"]
    paired = left[keys + ["abs_error"]].merge(
        right[keys + ["abs_error"]], on=keys, suffixes=("_left", "_right")
    )
    paired["delta"] = paired["abs_error_right"] - paired["abs_error_left"]
    event_values = {
        event: group["delta"].to_numpy()
        for event, group in paired.groupby("event_id")
    }
    events = np.asarray(sorted(event_values))
    if events.size == 0:
        return {"paired_targets": 0, "paired_events": 0, "delta_mae": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled = rng.choice(events, size=events.size, replace=True)
        values = np.concatenate([event_values[event] for event in sampled])
        estimates[draw] = values.mean()
    return {
        "paired_targets": int(len(paired)),
        "paired_events": int(events.size),
        "delta_mae": float(paired["delta"].mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    args = parser.parse_args()
    files = sorted(args.eval_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No V01 NPZ files in {args.eval_dir}")
    frames = [load_rows(path) for path in files]
    all_rows = pd.concat(frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_frame = pd.DataFrame([metrics(frame) for frame in frames])
    metric_frame.to_csv(args.output_dir / "cross_eval_metrics.csv", index=False)

    comparisons = []
    for protocol in ("normal", "random"):
        for times in ((1.0,), (3.0,), (5.0,), (1.0, 3.0, 5.0)):
            subsets = {}
            for checkpoint, view in (("vfull", "vfull"), ("vfull", "vmissing"), ("vmissing", "vmissing")):
                frame = all_rows[
                    (all_rows["checkpoint"] == checkpoint)
                    & (all_rows["view"] == view)
                    & (all_rows["protocol"] == protocol)
                    & all_rows["decision_time_s"].round(6).isin(times)
                ]
                subsets[(checkpoint, view)] = frame
            for name, left_key, right_key in (
                ("R_MM_minus_R_FF", ("vfull", "vfull"), ("vmissing", "vmissing")),
                ("R_FM_minus_R_FF", ("vfull", "vfull"), ("vfull", "vmissing")),
            ):
                result = paired_cluster_ci(
                    subsets[left_key], subsets[right_key],
                    args.bootstrap_draws, args.bootstrap_seed,
                )
                result.update({
                    "comparison": name,
                    "protocol": protocol,
                    "decision_times_s": ",".join(str(int(value)) for value in times),
                    "bootstrap_draws": args.bootstrap_draws,
                    "bootstrap_seed": args.bootstrap_seed,
                })
                comparisons.append(result)
    paired = pd.DataFrame(comparisons)
    paired.to_csv(args.output_dir / "paired_ci.csv", index=False)
    summary = {
        "status": "analysis_complete",
        "eval_files": [str(path) for path in files],
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.bootstrap_seed,
        "metrics_rows": len(metric_frame),
        "paired_rows": len(paired),
        "limitations": [
            "Validation only; test is not read.",
            "Intervals cluster by event within one seed and do not quantify training-seed uncertainty.",
            "A_pair versus velocity includes instrument, site, response and input-domain differences.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "RESULT_REVIEW.md").write_text(
        "# V01 result review\n\n"
        "This package was generated from fixed epoch-8 validation outputs. "
        "See `cross_eval_metrics.csv`, `paired_ci.csv`, and `summary.json`. "
        "Positive paired delta means the right-hand condition has larger MAE.\n"
    )


if __name__ == "__main__":
    main()
