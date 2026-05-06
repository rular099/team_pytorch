#!/usr/bin/env python
"""Merge sharded eval_checkpoint.py npz outputs and write a compact summary."""

import argparse
from pathlib import Path

import numpy as np


def npz_stack(values):
    arr = np.asarray(values)
    if arr.dtype == object:
        return np.stack([np.asarray(v) for v in arr])
    return arr


def merge_npz(paths):
    loaded = [np.load(path, allow_pickle=True) for path in paths]
    keys = sorted(set().union(*(set(data.files) for data in loaded)))
    merged = {}
    for key in keys:
        arrays = [np.asarray(data[key], dtype=object) for data in loaded if key in data]
        if not arrays:
            continue
        merged[key] = np.concatenate(arrays, axis=0)

    for prefix in ("train_", "val_", "single_train_", "single_val_", "case_sweep_train_", "case_sweep_val_"):
        index_key = f"{prefix}event_index"
        if index_key not in merged:
            continue
        event_index = np.asarray(merged[index_key], dtype=np.int64).reshape(-1)
        order = np.argsort(event_index, kind="stable")
        n = len(event_index)
        for key, value in list(merged.items()):
            if key.startswith(prefix) and len(value) == n:
                merged[key] = value[order]
    return merged


def pga_metrics(merged, split):
    required = [f"{split}_pga_label", f"{split}_pga_mu_best", f"{split}_pga_target_valid"]
    if any(key not in merged for key in required):
        return None
    labels = npz_stack(merged[f"{split}_pga_label"]).reshape(len(merged[f"{split}_pga_label"]), -1).astype(float)
    preds = npz_stack(merged[f"{split}_pga_mu_best"]).reshape(labels.shape[0], -1).astype(float)
    valid = npz_stack(merged[f"{split}_pga_target_valid"]).reshape(labels.shape[0], -1).astype(bool)
    if not valid.any():
        return None
    y = labels[valid]
    p = preds[valid]
    residual = p - y
    out = {
        "n_events": labels.shape[0],
        "n_valid": int(valid.sum()),
        "n_total": int(valid.size),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "corr": np.nan,
        "r2": np.nan,
        "slope": np.nan,
        "intercept": np.nan,
        "buckets": [],
    }
    if len(y) > 1:
        out["corr"] = float(np.corrcoef(y, p)[0, 1])
        ss_res = float(np.sum((p - y) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot > 0:
            out["r2"] = 1.0 - ss_res / ss_tot
        out["slope"], out["intercept"] = [float(v) for v in np.polyfit(y, p, 1)]

    count_key = f"{split}_station_valid_count"
    if count_key in merged:
        counts = np.asarray(merged[count_key], dtype=np.int64).reshape(-1)
        bucket_defs = [
            ("1", counts == 1),
            ("2-3", (counts >= 2) & (counts <= 3)),
            ("4-5", (counts >= 4) & (counts <= 5)),
            ("6-10", (counts >= 6) & (counts <= 10)),
            ("11-15", (counts >= 11) & (counts <= 15)),
            ("16+", counts >= 16),
        ]
        for name, rows in bucket_defs:
            if not rows.any():
                continue
            mask = valid[rows]
            if not mask.any():
                continue
            by = labels[rows][mask]
            bp = preds[rows][mask]
            bres = bp - by
            bcorr = float(np.corrcoef(by, bp)[0, 1]) if len(by) > 1 else np.nan
            out["buckets"].append({
                "name": name,
                "events": int(rows.sum()),
                "targets": int(mask.sum()),
                "mae": float(np.mean(np.abs(bres))),
                "rmse": float(np.sqrt(np.mean(bres ** 2))),
                "corr": bcorr,
            })
    return out


def fmt(value):
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.4f}"


def write_summary(merged, path):
    lines = []
    for split in ("train", "val"):
        metrics = pga_metrics(merged, split)
        if metrics is None:
            continue
        lines.append("=" * 60)
        lines.append(f"  {split.upper()} set: {metrics['n_events']} samples")
        lines.append("=" * 60)
        lines.append("")
        lines.append("--- pga ---")
        lines.append(
            f"  MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f} "
            f"({metrics['n_valid']}/{metrics['n_total']} valid targets)"
        )
        lines.append(f"  Correlation: {fmt(metrics['corr'])}")
        lines.append(f"  R^2: {fmt(metrics['r2'])}")
        lines.append(f"  Linear fit: pred = {fmt(metrics['slope'])} * label + {fmt(metrics['intercept'])}")
        if metrics["buckets"]:
            lines.append("  By input station count:")
            for bucket in metrics["buckets"]:
                lines.append(
                    f"    n={bucket['name']}: events={bucket['events']}, "
                    f"targets={bucket['targets']}, MAE={bucket['mae']:.4f}, "
                    f"RMSE={bucket['rmse']:.4f}, corr={fmt(bucket['corr'])}"
                )
        lines.append("")
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Merge eval_checkpoint.py sharded npz outputs.")
    parser.add_argument("shards", nargs="+", help="Shard npz files, e.g. eval_results_last_shard*.npz")
    parser.add_argument("--output", required=True, help="Merged npz path.")
    parser.add_argument("--summary-output", default=None, help="Optional compact eval_results txt path.")
    args = parser.parse_args()

    paths = sorted(Path(item) for item in args.shards)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing shard files: " + ", ".join(missing))

    merged = merge_npz(paths)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **merged)
    print(f"[INFO] merged {len(paths)} shards -> {output}")

    if args.summary_output:
        summary = Path(args.summary_output)
        summary.parent.mkdir(parents=True, exist_ok=True)
        write_summary(merged, summary)
        print(f"[INFO] wrote summary -> {summary}")


if __name__ == "__main__":
    main()
