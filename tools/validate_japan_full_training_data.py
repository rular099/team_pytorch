#!/usr/bin/env python3
"""Validate and optionally cache the 2000-2024 Japan training shards."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REQUIRED_METADATA = {
    "event_metadata",
    "station_metadata",
    "sampling_rate",
}
REQUIRED_EVENT_DATASETS = {
    "waveforms",
    "coords",
    "p_picks",
    "pga",
    "record_start_sample",
    "valid_n_samples",
    "source_network",
    "sensor_class",
    "vs30",
    "vs30_valid",
}
DEFAULT_CONFIG = (
    ROOT
    / "pga_configs"
    / "transformer_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_chaosuan.json"
)


def parse_years(value: str) -> list[int]:
    value = value.strip()
    if "-" in value and "," not in value:
        start, end = (int(part) for part in value.split("-", 1))
        if start > end:
            raise ValueError(f"Invalid year range: {value}")
        return list(range(start, end + 1))
    years = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not years:
        raise ValueError("No years selected.")
    return years


def decode_array(dataset) -> np.ndarray:
    values = dataset[()]
    return np.asarray([
        item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
        for item in values
    ])


def table_lengths(group: h5py.Group) -> set[int]:
    return {
        int(dataset.shape[0])
        for dataset in group.values()
        if isinstance(dataset, h5py.Dataset) and dataset.ndim > 0
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_event_ids(event_ids: np.ndarray) -> dict[str, np.ndarray]:
    test_start = int(0.8 * len(event_ids))
    train_end = int(0.7 / 0.8 * test_start)
    return {
        "train": event_ids[:train_end],
        "dev": event_ids[train_end:test_start],
        "test": event_ids[test_start:],
    }


def selected_event_names(event_names: list[str], scan: str) -> list[str]:
    if scan == "all" or len(event_names) <= 3:
        return event_names
    return [event_names[0], event_names[len(event_names) // 2], event_names[-1]]


def validate_event_group(event_id: str, group: h5py.Group, errors: list[str]) -> dict[str, int]:
    prefix = f"event {event_id}"
    missing = REQUIRED_EVENT_DATASETS - set(group.keys())
    if missing:
        errors.append(f"{prefix}: missing datasets {sorted(missing)}")
        return {"stations": 0, "positive_record_start": 0}

    waveforms = group["waveforms"]
    if waveforms.ndim != 3 or waveforms.shape[-1] != 3:
        errors.append(f"{prefix}: waveforms shape must be (station,time,3), got {waveforms.shape}")
        return {"stations": 0, "positive_record_start": 0}
    if waveforms.dtype.kind not in "fc":
        errors.append(f"{prefix}: waveform dtype must be floating point, got {waveforms.dtype}")

    n_stations, n_samples, _ = waveforms.shape
    for key in REQUIRED_EVENT_DATASETS - {"waveforms"}:
        dataset = group[key]
        if dataset.ndim == 0 or dataset.shape[0] != n_stations:
            errors.append(
                f"{prefix}: {key} first dimension {dataset.shape} does not match {n_stations} stations"
            )

    starts = np.asarray(group["record_start_sample"][()], dtype=np.int64)
    lengths = np.asarray(group["valid_n_samples"][()], dtype=np.int64)
    if np.any(starts < 0) or np.any(lengths <= 0):
        errors.append(f"{prefix}: negative record start or non-positive valid length")
    if np.any(starts + lengths > n_samples):
        worst = int(np.max(starts + lengths - n_samples))
        errors.append(f"{prefix}: record_start_sample + valid_n_samples exceeds waveform by {worst}")

    p_picks = np.asarray(group["p_picks"][()], dtype=np.int64)
    if np.any((p_picks < 0) | (p_picks >= n_samples)):
        errors.append(f"{prefix}: p_picks contains values outside [0, {n_samples})")
    pga = np.asarray(group["pga"][()], dtype=np.float64)
    if not np.all(np.isfinite(pga)):
        errors.append(f"{prefix}: pga contains NaN/Inf")
    coords = np.asarray(group["coords"][()], dtype=np.float64)
    if not np.all(np.isfinite(coords)):
        errors.append(f"{prefix}: coords contains NaN/Inf")

    networks = set(decode_array(group["source_network"]))
    if not networks <= {"knt", "kik"}:
        errors.append(f"{prefix}: unexpected source_network values {sorted(networks)}")
    return {
        "stations": int(n_stations),
        "positive_record_start": int(np.sum(starts > 0)),
    }


def validate_shard(path: Path, year: int, scan: str, with_sha256: bool) -> tuple[dict, list[str], np.ndarray]:
    errors: list[str] = []
    info: dict[str, object] = {
        "year": year,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
    }
    if with_sha256:
        info["sha256"] = sha256_file(path)

    with h5py.File(path, "r") as handle:
        if set(handle.keys()) != {"data", "metadata"}:
            errors.append(f"{path}: root keys are {sorted(handle.keys())}, expected data/metadata")
        missing_metadata = REQUIRED_METADATA - set(handle["metadata"].keys())
        if missing_metadata:
            errors.append(f"{path}: missing metadata entries {sorted(missing_metadata)}")
            return info, errors, np.empty(0, dtype=np.float64)

        sampling_rate = float(np.asarray(handle["metadata/sampling_rate"][()]).reshape(-1)[0])
        info["sampling_rate_hz"] = sampling_rate
        if sampling_rate != 100.0:
            errors.append(f"{path}: expected 100 Hz, got {sampling_rate}")

        event_table = handle["metadata/event_metadata"]
        station_table = handle["metadata/station_metadata"]
        event_lengths = table_lengths(event_table)
        station_lengths = table_lengths(station_table)
        if len(event_lengths) != 1:
            errors.append(f"{path}: inconsistent event_metadata column lengths {sorted(event_lengths)}")
        if len(station_lengths) != 1:
            errors.append(f"{path}: inconsistent station_metadata column lengths {sorted(station_lengths)}")

        event_ids = decode_array(event_table["EVENT"])
        station_event_ids = decode_array(station_table["EVENT"])
        networks = decode_array(station_table["source_network"])
        if any(not event_id.startswith(str(year)) for event_id in event_ids):
            errors.append(f"{path}: at least one event ID does not start with {year}")
        data_event_names = list(handle["data"].keys())
        if set(event_ids) != set(data_event_names):
            errors.append(
                f"{path}: event table/data groups differ "
                f"({len(event_ids)} table vs {len(data_event_names)} groups)"
            )

        split_ids = split_event_ids(event_ids)
        stalta = np.asarray(station_table["stalta_ratio_at_pick"][()], dtype=np.float64)
        valid_knet = (networks == "knt") & np.isfinite(stalta) & (stalta >= 0.1)
        pga_mps2 = np.asarray(station_table["pga_norm_resampled_mps2"][()], dtype=np.float64)
        valid_pga = valid_knet & np.isfinite(pga_mps2) & (pga_mps2 > 0)
        train_mask = valid_pga & np.isin(station_event_ids, split_ids["train"])
        train_log_pga = np.log10(pga_mps2[train_mask])

        info.update({
            "events": int(len(event_ids)),
            "station_rows": int(len(station_event_ids)),
            "knet_station_rows": int(np.sum(networks == "knt")),
            "kik_station_rows": int(np.sum(networks == "kik")),
            "knet_stalta_rows": int(np.sum(valid_knet)),
            "knet_vs30_valid_rows": int(np.sum(
                valid_knet & np.asarray(station_table["vs30_valid"][()], dtype=bool)
            )),
            "split_events_before_station_filter": {
                name: int(len(ids)) for name, ids in split_ids.items()
            },
            "split_events_after_knet_filter": {
                name: int(sum(event_id in set(station_event_ids[valid_knet]) for event_id in ids))
                for name, ids in split_ids.items()
            },
            "origin_correction_status": dict(Counter(decode_array(
                event_table["Origin_Time_Correction_Status"]
            ))),
            "alignment_mode": decode_array(handle["metadata/alignment_mode"])[0],
            "waveform_dtype": str(handle["data"][data_event_names[0]]["waveforms"].dtype),
        })

        scanned_stations = 0
        positive_record_start = 0
        for event_id in selected_event_names(data_event_names, scan):
            event_info = validate_event_group(event_id, handle["data"][event_id], errors)
            scanned_stations += event_info["stations"]
            positive_record_start += event_info["positive_record_start"]
        info["scanned_events"] = len(selected_event_names(data_event_names, scan))
        info["scanned_station_rows"] = scanned_stations
        info["scanned_positive_record_start_rows"] = positive_record_start

    return info, errors, train_log_pga


def validate_config(config_path: Path, data_root: Path, years: list[int], mean: float, std: float) -> dict:
    os.environ["JAPAN_FULL_DATA_ROOT"] = str(data_root)
    os.environ.setdefault("JAPAN_FULL_WEIGHT_PATH", "weights_japan_full_validation_only")
    from train_light import load_config_file, expand_partitioned_generator_params

    config = load_config_file(str(config_path))
    model = config["model_params"]
    training = config["training_params"]
    generators = expand_partitioned_generator_params(training)
    expected_paths = [data_root / str(year) / f"japan_{year}.hdf5" for year in years]
    actual_paths = [Path(path) for path in training["data_path"]]
    if actual_paths != expected_paths:
        raise ValueError("Resolved config data_path does not exactly match the selected annual shards.")
    if len(generators) != len(expected_paths):
        raise ValueError("Generator template did not broadcast to every annual shard.")
    if any(not generator.get("emit_waveform_padding_mask", False) for generator in generators):
        raise ValueError("Every generator must emit the explicit waveform padding mask.")
    if training.get("station_filter") != "knet":
        raise ValueError("Full-data config must use station_filter=knet.")
    if training.get("dpk_prior_cache"):
        raise ValueError("Full-data config unexpectedly enables dpk_prior_cache.")
    if model.get("station_token_weight_mode") != "none":
        raise ValueError("station_token_weight_mode must be none.")
    if model.get("temporal_token_weight_mode") != "none":
        raise ValueError("temporal_token_weight_mode must be none.")
    if model.get("dpk_checkpoint_path") not in (None, ""):
        raise ValueError("dpk_checkpoint_path must be null when the DPK path is disabled.")
    if model.get("use_pga_temporal_residual", False):
        raise ValueError("PGA temporal residual must be disabled.")
    norm = training.get("pga_target_normalization") or {}
    if abs(float(norm["mean"]) - mean) > 1e-10 or abs(float(norm["std"]) - std) > 1e-10:
        raise ValueError(
            "Configured PGA normalization does not match the current train split: "
            f"config=({norm['mean']}, {norm['std']}), data=({mean}, {std})"
        )
    return config


def validate_runtime_record_start_mask(path: Path) -> None:
    """Exercise the actual generator mask reader on a positive-offset trace."""
    from gemini_util_light import PreloadedEventGenerator

    generator = PreloadedEventGenerator.__new__(PreloadedEventGenerator)
    generator.decimate = 1
    generator.waveform_padding_mask_eps = 1e-8
    with h5py.File(path, "r") as handle:
        for event_id, group in handle["data"].items():
            starts = np.asarray(group["record_start_sample"][()], dtype=np.int64)
            candidates = np.flatnonzero(starts > 0)
            if candidates.size == 0:
                continue
            rows = candidates[: min(3, candidates.size)]
            waveforms = group["waveforms"][rows, :, :]
            mask = generator._waveform_storage_mask(group, rows, waveforms)
            lengths = np.asarray(group["valid_n_samples"][rows], dtype=np.int64)
            for local_idx, (start, length) in enumerate(zip(starts[rows], lengths)):
                expected = np.zeros(waveforms.shape[1], dtype=bool)
                expected[int(start):int(start + length)] = True
                if not np.array_equal(mask[local_idx], expected):
                    raise ValueError(
                        f"Runtime mask mismatch for {event_id} station row {int(rows[local_idx])}."
                    )
            print(
                f"[MASK] runtime record_start_sample check passed: "
                f"event={event_id}, rows={rows.tolist()}"
            )
            return
    raise ValueError(f"No positive record_start_sample found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=os.environ.get("JAPAN_FULL_DATA_ROOT"),
        help="Directory containing YEAR/japan_YEAR.hdf5.",
    )
    parser.add_argument("--years", default="2000-2024")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scan", choices=["sampled", "all"], default="all")
    parser.add_argument("--sha256", action="store_true", help="Slow: hash every full HDF5 file.")
    parser.add_argument("--prepare-cache", action="store_true")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "cache" / "japan_full_2000_2024_station_metadata",
    )
    parser.add_argument("--manifest-output", type=Path, default=None)
    args = parser.parse_args()

    if args.data_root is None:
        parser.error("Set --data-root or JAPAN_FULL_DATA_ROOT.")
    data_root = args.data_root.expanduser().resolve()
    years = parse_years(args.years)
    paths = [data_root / str(year) / f"japan_{year}.hdf5" for year in years]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing annual shards:\n" + "\n".join(map(str, missing)))

    manifest = {
        "data_root": str(data_root),
        "years": years,
        "scan": args.scan,
        "files": [],
    }
    all_errors: list[str] = []
    all_event_ids: list[str] = []
    all_train_log_pga: list[np.ndarray] = []
    totals = Counter()
    split_before = Counter()
    split_after = Counter()
    origin_status = Counter()

    for year, path in zip(years, paths):
        info, errors, train_log_pga = validate_shard(path, year, args.scan, args.sha256)
        manifest["files"].append(info)
        all_errors.extend(errors)
        all_train_log_pga.append(train_log_pga)
        with h5py.File(path, "r") as handle:
            all_event_ids.extend(decode_array(handle["metadata/event_metadata/EVENT"]).tolist())
        for key in ("events", "station_rows", "knet_station_rows", "kik_station_rows"):
            totals[key] += int(info.get(key, 0))
        split_before.update(info.get("split_events_before_station_filter", {}))
        split_after.update(info.get("split_events_after_knet_filter", {}))
        origin_status.update(info.get("origin_correction_status", {}))
        print(
            f"[OK] {year}: events={info.get('events')} stations={info.get('station_rows')} "
            f"knet={info.get('knet_station_rows')} scanned={info.get('scanned_events')}"
        )

    duplicate_count = len(all_event_ids) - len(set(all_event_ids))
    if duplicate_count:
        all_errors.append(f"Found {duplicate_count} duplicate event IDs across annual shards.")
    train_log_pga = np.concatenate(all_train_log_pga)
    pga_mean = float(train_log_pga.mean())
    pga_std = float(train_log_pga.std())
    manifest.update({
        "totals": dict(totals),
        "global_unique_event_ids": len(set(all_event_ids)),
        "split_events_before_station_filter": dict(split_before),
        "split_events_after_knet_filter": dict(split_after),
        "origin_correction_status": dict(origin_status),
        "train_knet_log10_pga_mps2": {
            "count": int(train_log_pga.size),
            "mean": pga_mean,
            "std_population": pga_std,
            "min": float(train_log_pga.min()),
            "max": float(train_log_pga.max()),
        },
    })

    if all_errors:
        print("\n[FAIL] compatibility errors:", file=sys.stderr)
        for error in all_errors:
            print(f"  - {error}", file=sys.stderr)
        raise SystemExit(1)

    config = validate_config(args.config.resolve(), data_root, years, pga_mean, pga_std)
    validate_runtime_record_start_mask(paths[-1])
    if args.prepare_cache:
        import loader_light

        cache_dir = args.cache_dir.expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        stub = cache_dir / "event_metadata.csv"
        training = config["training_params"]
        for path in paths:
            cache_path = loader_light.ensure_event_metadata_cache(
                str(path),
                event_metadata_path=str(stub),
                overwrite_sampling_rate=training.get("overwrite_sampling_rate"),
                min_stalta_ratio_at_pick=training.get("min_stalta_ratio_at_pick", 0.1),
                cache_columns=training.get("metadata_cache_columns"),
            )
            print(f"[CACHE] {cache_path}")
        manifest["metadata_cache_dir"] = str(cache_dir)

    output = args.manifest_output
    if output is None and args.prepare_cache:
        output = args.cache_dir / "japan_full_2000_2024_manifest.json"
    if output is not None:
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        print(f"[MANIFEST] {output}")

    print(
        "[PASS] full-data compatibility: "
        f"events={totals['events']}, station_rows={totals['station_rows']}, "
        f"knet_rows={totals['knet_station_rows']}, "
        f"split_after_knet={dict(split_after)}, "
        f"origin_status={dict(origin_status)}, "
        f"train_log10_pga_mean={pga_mean:.12f}, std={pga_std:.12f}"
    )


if __name__ == "__main__":
    main()
