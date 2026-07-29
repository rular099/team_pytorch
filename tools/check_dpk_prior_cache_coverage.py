#!/usr/bin/env python
"""Check whether a precomputed DPK prior cache covers a realtime config split."""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import gemini_util_light as util
from tools.precompute_dpk_priors import load_split_generators


def load_config(path):
    with open(path, "r") as f:
        if path.endswith((".yml", ".yaml")):
            return yaml.safe_load(f)
        return json.load(f)


def infer_cache_path(config, split, config_path=None):
    training_params = config.get("training_params", {})
    cache_cfg = training_params.get("dpk_prior_cache", {}) or {}
    paths = cache_cfg.get("paths", {}) or {}

    split_aliases = [split]
    if split == "dev":
        split_aliases.append("val")
    elif split == "test":
        split_aliases.append("eval")

    for key in split_aliases:
        path = paths.get(key)
        if path:
            return path

    for key in split_aliases:
        path = cache_cfg.get(f"{key}_path")
        if path:
            return path

    if config_path is not None:
        config_path = Path(config_path).resolve()
        config_stem = config_path.stem
        candidates = []
        if config_path.parent.name == "pga_configs":
            repo_root = config_path.parent.parent
            candidates.append(repo_root / "dpk_prior_cache" / f"{config_stem}_{split}" / "dpk_priors.h5")
        candidates.append(Path.cwd() / "dpk_prior_cache" / f"{config_stem}_{split}" / "dpk_priors.h5")
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

    raise ValueError(
        "No DPK prior cache path was provided and none could be inferred from "
        f"training_params.dpk_prior_cache or dpk_prior_cache/<config>_{split}/dpk_priors.h5 "
        f"for split={split!r}."
    )


def scalar(info, key):
    value = info.get(key)
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return value.detach().cpu().reshape(-1)[0].item()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", default=None, help="HDF5 cache path; inferred from config when omitted")
    parser.add_argument("--split", default="train", choices=("train", "dev", "test"))
    parser.add_argument("--source", default="dpk_finetuned")
    parser.add_argument("--mode", default="event")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_misses", type=int, default=20)
    parser.add_argument("--stop_after_misses", type=int, default=0)
    parser.add_argument(
        "--no_align_realtime",
        action="store_true",
        help="Disable nearest-current-sample alignment. By default the check matches training.",
    )
    parser.add_argument(
        "--no_filter_missing_stations",
        action="store_true",
        help="Disable station filtering by cache availability. By default the check matches training.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    cache_path = args.cache or infer_cache_path(config, args.split, args.config)
    cache = util.DPKPriorCache(
        cache_path,
        source=args.source,
        mode=args.mode,
        missing_policy="error",
    )
    key_set = cache.row_by_event_station_time_key
    if not key_set:
        raise RuntimeError(f"Cache has no index/event_station_time_key: {cache_path}")

    generators = load_split_generators(config, args.split, limit=None, rank=0)
    checked_samples = 0
    checked_station_rows = 0
    missing_station_rows = 0
    raw_station_rows = 0
    filtered_station_rows = 0
    empty_after_filter_samples = 0
    aligned_samples = 0
    alignment_abs_deltas = []
    failed_samples = 0
    miss_preview = []
    miss_events = Counter()
    miss_times = Counter()

    for split_name, dataset_id, dataset in generators:
        total = len(dataset)
        if args.max_samples is not None:
            total = min(total, args.max_samples)
        for sample_index in range(total):
            try:
                inputs, _, info = dataset[sample_index]
            except Exception as exc:
                failed_samples += 1
                if len(miss_preview) < args.max_misses:
                    miss_preview.append(
                        f"sample_error split={split_name} dataset={dataset_id} "
                        f"idx={sample_index}: {exc!r}"
                    )
                continue

            station_valid = inputs[2].detach().cpu().numpy().astype(bool)
            original_station_indices = None
            for index_key in (
                "original_station_indices",
                "selected_original_input_indices",
                "selected_input_indices",
            ):
                if index_key in info:
                    original_station_indices = info[index_key]
                    break
            if original_station_indices is None:
                raise RuntimeError(
                    "Generator did not return original station indices; cannot "
                    "validate event_station_time_key cache coverage."
                )
            if torch.is_tensor(original_station_indices):
                original_station_indices = original_station_indices.detach().cpu().numpy()
            original_station_indices = np.asarray(original_station_indices)
            event_id = str(info.get("event_id", ""))
            current_sample = scalar(info, "realtime_current_sample")
            if current_sample is None:
                raise RuntimeError("Generator did not return realtime_current_sample.")
            current_sample = int(round(float(current_sample)))
            requested_current_sample = current_sample

            if not args.no_align_realtime:
                aligned_current_sample = cache.nearest_current_sample(
                    split_name,
                    int(dataset_id),
                    event_id,
                    current_sample,
                )
                if aligned_current_sample is not None:
                    aligned_current_sample = int(aligned_current_sample)
                    if aligned_current_sample != current_sample:
                        aligned_samples += 1
                        alignment_abs_deltas.append(abs(aligned_current_sample - current_sample))
                    current_sample = aligned_current_sample

            raw_station_rows += int(station_valid.sum())
            if not args.no_filter_missing_stations:
                cache_available = cache.station_available_mask(
                    split_name,
                    int(dataset_id),
                    event_id,
                    current_sample,
                    original_station_indices,
                    station_valid,
                )
                filtered_station_rows += int((station_valid & ~cache_available).sum())
                station_valid = station_valid & cache_available
                if not station_valid.any():
                    empty_after_filter_samples += 1
                    checked_samples += 1
                    continue

            for station_slot, valid in enumerate(station_valid):
                if not valid:
                    continue
                checked_station_rows += 1
                if station_slot >= len(original_station_indices):
                    original_station = -1
                else:
                    original_station = int(round(float(original_station_indices[station_slot])))
                key = (
                    f"{split_name}|{int(dataset_id)}|{event_id}|"
                    f"{current_sample}|{original_station}"
                )
                if key not in key_set:
                    missing_station_rows += 1
                    miss_events[event_id] += 1
                    miss_times[current_sample] += 1
                    if len(miss_preview) < args.max_misses:
                        miss_preview.append(
                            f"{key} requested_current_sample={requested_current_sample}"
                        )
                    if args.stop_after_misses and missing_station_rows >= args.stop_after_misses:
                        break
            checked_samples += 1
            if args.stop_after_misses and missing_station_rows >= args.stop_after_misses:
                break

        if args.stop_after_misses and missing_station_rows >= args.stop_after_misses:
            break

    coverage = 1.0
    if checked_station_rows:
        coverage = 1.0 - missing_station_rows / float(checked_station_rows)
    print(f"cache={cache_path}")
    print(f"config={args.config}")
    print(f"split={args.split}")
    print(f"checked_samples={checked_samples}")
    print(f"raw_station_rows={raw_station_rows}")
    print(f"filtered_station_rows={filtered_station_rows}")
    print(f"empty_after_filter_samples={empty_after_filter_samples}")
    print(f"checked_station_rows={checked_station_rows}")
    print(f"missing_station_rows={missing_station_rows}")
    print(f"failed_samples={failed_samples}")
    print(f"coverage={coverage:.8f}")
    if raw_station_rows:
        station_retention = checked_station_rows / float(raw_station_rows)
        print(f"station_retention_after_cache_filter={station_retention:.8f}")
    print(f"aligned_samples={aligned_samples}")
    if alignment_abs_deltas:
        deltas = np.asarray(alignment_abs_deltas, dtype=np.float64)
        print(f"alignment_abs_delta_mean={float(deltas.mean()):.3f}")
        print(f"alignment_abs_delta_max={int(deltas.max())}")
    if miss_preview:
        print("miss_preview:")
        for item in miss_preview:
            print(f"  {item}")
    if miss_events:
        print("top_missing_events:")
        for event_id, count in miss_events.most_common(10):
            print(f"  {event_id}: {count}")
    if miss_times:
        print("top_missing_current_samples:")
        for current_sample, count in miss_times.most_common(10):
            print(f"  {current_sample}: {count}")

    if missing_station_rows or failed_samples:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
