#!/usr/bin/env python3
"""Create paired rt48-minus-rt46 summaries from phase-attention diagnostics."""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


KEY_FIELDS = ("split", "event_id", "pool", "post_p_bin")
NON_METRIC_FIELDS = {
    "model",
    "split",
    "event_id",
    "pool",
    "post_p_bin",
    "station_record_count",
    "sample_count",
}


def _read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _event_files(root):
    files = []
    for split in ("train", "val"):
        path = os.path.join(root, f"per_event_bin_{split}.csv")
        if os.path.isfile(path):
            files.append(path)
    if not files:
        raise FileNotFoundError(
            f"No per_event_bin_train.csv or per_event_bin_val.csv under {root}"
        )
    return files


def _load_event_rows(root):
    rows = []
    for path in _event_files(root):
        rows.extend(_read_csv(path))
    return rows


def _index(rows):
    indexed = {}
    for row in rows:
        key = tuple(row[field] for field in KEY_FIELDS)
        if key in indexed:
            raise ValueError(f"Duplicate event-bin key: {key}")
        indexed[key] = row
    return indexed


def _ci95(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None, None
    mean = float(values.mean())
    if values.size == 1:
        return mean, mean
    half = 1.96 * float(values.std(ddof=1)) / np.sqrt(values.size)
    return mean - half, mean + half


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        with open(path, "w", newline="") as handle:
            handle.write("")
        return
    fields = list(rows[0])
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compare(control_rows, treatment_rows):
    control = _index(control_rows)
    treatment = _index(treatment_rows)
    common_keys = sorted(set(control) & set(treatment))
    grouped = defaultdict(list)

    for key in common_keys:
        control_row = control[key]
        treatment_row = treatment[key]
        metrics = (
            set(control_row)
            & set(treatment_row)
            - NON_METRIC_FIELDS
        )
        for metric in metrics:
            control_value = _float(control_row.get(metric))
            treatment_value = _float(treatment_row.get(metric))
            if control_value is None or treatment_value is None:
                continue
            split, _event_id, pool, post_p_bin = key
            grouped[(split, pool, post_p_bin, metric)].append(
                (control_value, treatment_value)
            )

    rows = []
    for key in sorted(grouped):
        split, pool, post_p_bin, metric = key
        pairs = np.asarray(grouped[key], dtype=np.float64)
        delta = pairs[:, 1] - pairs[:, 0]
        ci_low, ci_high = _ci95(delta)
        rows.append({
            "split": split,
            "pool": pool,
            "post_p_bin": post_p_bin,
            "metric": metric,
            "paired_event_count": int(delta.size),
            "rt46_mean": float(pairs[:, 0].mean()),
            "rt48_mean": float(pairs[:, 1].mean()),
            "rt48_minus_rt46_mean": float(delta.mean()),
            "rt48_minus_rt46_std": float(delta.std(ddof=0)),
            "rt48_minus_rt46_median": float(np.median(delta)),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
        })
    return rows, len(common_keys)


def main():
    parser = argparse.ArgumentParser(
        description="Pair event-level rt46 and rt48 station-pool phase diagnostics."
    )
    parser.add_argument("--rt46_dir", required=True)
    parser.add_argument("--rt48_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    control_rows = _load_event_rows(args.rt46_dir)
    treatment_rows = _load_event_rows(args.rt48_dir)
    rows, common_event_bin_keys = compare(control_rows, treatment_rows)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(
        args.output_dir,
        "paired_rt48_minus_rt46.csv",
    )
    json_path = os.path.join(args.output_dir, "comparison_summary.json")
    _write_csv(csv_path, rows)
    with open(json_path, "w") as handle:
        json.dump(
            {
                "rt46_dir": os.path.abspath(args.rt46_dir),
                "rt48_dir": os.path.abspath(args.rt48_dir),
                "common_event_bin_key_count": common_event_bin_keys,
                "comparison_row_count": len(rows),
                "paired_csv": os.path.abspath(csv_path),
            },
            handle,
            indent=2,
        )
    print(f"[INFO] wrote {csv_path}")
    print(f"[INFO] wrote {json_path}")


if __name__ == "__main__":
    main()
