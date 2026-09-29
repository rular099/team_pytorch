#!/usr/bin/env python3
"""Audit and materialize the V01 source/query derived HDF5 cache.

Only train/dev event IDs from the frozen RT55 split are opened.  The cache is a
reproducible transport format for the unchanged legacy generator: physical
source rows contain waveforms and NaN PGA, while independent acceleration query
rows contain PGA/coordinates and an invalid waveform mask.  V_missing remains a
runtime mask view of the velocity cache, so no third waveform copy is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loader_light import _read_metadata_table_h5
from tools.prep_padding_protocol import (
    PROTOCOL_VERSION,
    assign_retained_prep_samples,
    stable_index,
)
from tools.velocity_waveform_backend import (
    BACKEND_VERSION,
    AnnualHinetArchiveReader,
    assemble_event_windows,
    discover_archives,
    sha256_file,
)


CACHE_SCHEMA = "v01-derived-source-query-hdf5-v1"
TRACE_SAMPLES = 10000
SAMPLING_RATE = 100.0
PRE_P_SECONDS = 5.0


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _timestamp(value: Any) -> float:
    return float(pd.Timestamp(value).timestamp())


def _write_table(group: h5py.Group, frame: pd.DataFrame) -> None:
    for column in frame.columns:
        values = frame[column].to_numpy()
        if values.dtype.kind in "OUS":
            text = ["" if pd.isna(value) else str(value) for value in values]
            width = max(1, max((len(value.encode("utf-8")) for value in text), default=1))
            group.create_dataset(column, data=np.asarray(text, dtype=f"S{width}"))
        else:
            group.create_dataset(column, data=values)


def _split_map(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, dtype={"EVENT": str})
    required = {"EVENT", "split"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Frozen split lacks {sorted(required - set(frame.columns))}")
    frame["EVENT"] = frame["EVENT"].astype(str)
    per_event = frame[["EVENT", "split"]].drop_duplicates()
    conflicts = per_event.groupby("EVENT")["split"].nunique()
    if (conflicts > 1).any():
        raise ValueError("Frozen split assigns at least one event to multiple splits")
    return dict(per_event.itertuples(index=False, name=None))


def _acc_paths(root: Path) -> dict[int, Path]:
    result = {}
    for year in range(2000, 2025):
        path = root / str(year) / f"japan_{year}.hdf5"
        if not path.is_file():
            raise FileNotFoundError(f"Missing acceleration shard: {path}")
        result[year] = path
    return result


def build_train_templates(acc_paths: dict[int, Path], split: dict[str, str]) -> np.ndarray:
    templates: list[int] = []
    for path in acc_paths.values():
        frame = _read_metadata_table_h5(str(path), "station_metadata")
        frame["EVENT"] = frame["EVENT"].astype(str)
        frame = frame[frame["EVENT"].map(split).eq("train")]
        frame = frame[frame["source_network"].astype(str).str.lower().eq("knt")]
        picks = pd.to_numeric(frame["p_pick_aligned"], errors="coerce")
        starts = pd.to_numeric(frame["record_start_sample"], errors="coerce")
        values = (picks - starts).dropna().astype(np.int64)
        templates.extend(np.maximum(values.to_numpy(), 0).tolist())
    if not templates:
        raise RuntimeError("No train-only KNET prefix templates were found")
    return np.asarray(templates, dtype=np.int64)


def audit_intersection(
    acc_paths: dict[int, Path],
    archive_by_year: dict[int, Path],
    split: dict[str, str],
    selected_years: set[int],
) -> pd.DataFrame:
    """Metadata-only cohort audit; never opens an acceleration waveform dataset."""
    rows = []
    for year in sorted(selected_years & set(archive_by_year) & set(acc_paths)):
        station_frame = _read_metadata_table_h5(str(acc_paths[year]), "station_metadata")
        station_frame["EVENT"] = station_frame["EVENT"].astype(str)
        station_groups = station_frame.groupby("EVENT", sort=False)
        counts = {
            "year": year,
            "archive_path": str(archive_by_year[year]),
            "train_events": 0,
            "dev_events": 0,
            "source_sensors": 0,
            "query_targets": 0,
            "knet_compatible_sources": 0,
            "kik_hinet_bridge_sources": 0,
            "other_pair_sources": 0,
            "excluded_test_events": 0,
            "excluded_no_acc_event": 0,
            "excluded_no_source_pair": 0,
        }
        with AnnualHinetArchiveReader(archive_by_year[year]) as archive:
            counts["archive_events"] = len(archive.event_ids())
            for event_id in archive.event_ids():
                event_split = split.get(str(event_id))
                if event_split == "test":
                    counts["excluded_test_events"] += 1
                    continue
                if event_split not in {"train", "dev"} or event_id not in station_groups.groups:
                    counts["excluded_no_acc_event"] += 1
                    continue
                original = station_groups.get_group(event_id)
                sources = _source_rows(archive.manifest(event_id))
                networks = []
                for source in sources.itertuples(index=False):
                    matches = original[
                        original["station_code"].astype(str).eq(str(source.knet_station))
                    ]
                    if matches.empty:
                        continue
                    networks.append(str(matches.iloc[0]["source_network"]).lower())
                query_count = int(
                    original["source_network"].astype(str).str.lower().eq("knt").sum()
                )
                if not networks or query_count == 0:
                    counts["excluded_no_source_pair"] += 1
                    continue
                counts[f"{event_split}_events"] += 1
                counts["source_sensors"] += len(networks)
                counts["query_targets"] += query_count
                counts["knet_compatible_sources"] += sum(value == "knt" for value in networks)
                counts["kik_hinet_bridge_sources"] += sum(value == "kik" for value in networks)
                counts["other_pair_sources"] += sum(value not in {"knt", "kik"} for value in networks)
        rows.append(counts)
    return pd.DataFrame(rows)


def _source_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    usable = manifest.copy()
    required = {"raw_status", "response_status", "hinet_station", "ppick_timestamp", "match_distance_km", "knet_station"}
    if usable.empty or not required.issubset(usable.columns):
        return pd.DataFrame(columns=sorted(required))
    usable = usable[usable["raw_status"].astype(str).str.startswith("downloaded")]
    usable = usable[usable["response_status"].astype(str).eq("raw_channel_table_archived")]
    usable = usable.dropna(subset=["hinet_station", "ppick_timestamp"])
    # One physical velocity sensor is one input, even if it maps to multiple
    # surface/borehole acceleration rows.
    usable = usable.sort_values(["hinet_station", "match_distance_km", "knet_station"])
    usable = usable.drop_duplicates("hinet_station", keep="first")
    # A_pair must not turn the same acceleration sensor into multiple logical
    # inputs through two nearby Hi-net mappings.
    return usable.drop_duplicates("knet_station", keep="first").reset_index(drop=True)


def _acc_station_index(station_codes: np.ndarray, code: str) -> int | None:
    normalized = np.asarray([
        value.decode() if isinstance(value, bytes) else str(value)
        for value in station_codes
    ])
    matches = np.where(normalized == str(code))[0]
    return int(matches[0]) if matches.size else None


def _copy_acc_to_grid(
    group: h5py.Group,
    row: int,
    start_timestamp: float,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(group["waveforms"][row], dtype=np.float32)
    record_start = _timestamp(group["record_start_time_jst"][row].decode())
    insertion = int(group["record_start_sample"][row])
    source_axis_start = record_start - insertion / SAMPLING_RATE
    destination_start = int(round((source_axis_start - start_timestamp) * SAMPLING_RATE))
    destination = np.zeros((TRACE_SAMPLES, 3), dtype=np.float32)
    valid = np.zeros(TRACE_SAMPLES, dtype=bool)
    source_valid_start = insertion
    source_valid_end = insertion + int(group["valid_n_samples"][row])
    left = max(0, destination_start)
    right = min(TRACE_SAMPLES, destination_start + source.shape[0])
    if right > left:
        src_left = left - destination_start
        src_right = right - destination_start
        destination[left:right] = source[src_left:src_right]
        support_left = max(left, destination_start + source_valid_start)
        support_right = min(right, destination_start + source_valid_end)
        if support_right > support_left:
            valid[support_left:support_right] = True
    destination[~valid] = 0.0
    return destination, valid


def _event_rows(
    original: pd.DataFrame,
    sources: pd.DataFrame,
    query_indices: list[int],
    source_acc_indices: list[int],
    event_id: str,
    arm: str,
) -> pd.DataFrame:
    rows = []
    for source_index, source in enumerate(sources.itertuples(index=False)):
        paired_candidates = original[
            pd.to_numeric(original["wave_idx"], errors="coerce").eq(
                source_acc_indices[source_index]
            )
        ]
        if paired_candidates.empty:
            raise ValueError(
                f"Missing station metadata for wave_idx={source_acc_indices[source_index]}"
            )
        paired = paired_candidates.iloc[0].copy()
        if arm == "vfull":
            paired["source_network"] = "hnt"
            paired["sensor_class"] = "borehole_velocity"
            paired["station_code"] = str(source.hinet_station)
        paired["stalta_ratio_at_pick"] = max(1.0, float(paired.get("stalta_ratio_at_pick", 1.0)))
        paired["wave_idx"] = len(rows)
        rows.append(paired)
    for query_index in query_indices:
        query = original.iloc[query_index].copy()
        query["source_network"] = "query_knt"
        query["sensor_class"] = "pga_query"
        query["stalta_ratio_at_pick"] = max(1.0, float(query.get("stalta_ratio_at_pick", 1.0)))
        query["wave_idx"] = len(rows)
        rows.append(query)
    if not rows:
        return pd.DataFrame(columns=original.columns)
    result = pd.DataFrame(rows).reset_index(drop=True)
    result["EVENT"] = str(event_id)
    return result


def materialize_year(
    year: int,
    acc_path: Path,
    archive_path: Path,
    output_root: Path,
    split: dict[str, str],
    templates: np.ndarray,
    *,
    seed: int,
    overwrite: bool,
    max_events: int | None = None,
) -> dict[str, Any]:
    year_root = output_root / str(year)
    year_root.mkdir(parents=True, exist_ok=True)
    velocity_path = year_root / f"japan_{year}_v01_velocity.hdf5"
    apair_path = year_root / f"japan_{year}_v01_acc_pair.hdf5"
    for path in (velocity_path, apair_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite V01 cache: {path}")

    station_frame = _read_metadata_table_h5(str(acc_path), "station_metadata")
    station_frame["EVENT"] = station_frame["EVENT"].astype(str)
    station_groups = station_frame.groupby("EVENT", sort=False)
    velocity_station_rows: list[pd.DataFrame] = []
    apair_station_rows: list[pd.DataFrame] = []
    counters = {
        "year": year,
        "archive": str(archive_path),
        "events_in_archive": 0,
        "train_events": 0,
        "dev_events": 0,
        "source_sensors": 0,
        "query_targets": 0,
        "excluded_test_events": 0,
        "excluded_no_acc_event": 0,
        "excluded_no_source_pair": 0,
        "excluded_bad_source": 0,
    }

    tmp_velocity = velocity_path.with_suffix(".hdf5.tmp")
    tmp_apair = apair_path.with_suffix(".hdf5.tmp")
    for path in (tmp_velocity, tmp_apair):
        if path.exists():
            path.unlink()
    with AnnualHinetArchiveReader(archive_path) as archive, \
            h5py.File(acc_path, "r") as acc, \
            h5py.File(tmp_velocity, "w") as vel_out, \
            h5py.File(tmp_apair, "w") as acc_out:
        counters["events_in_archive"] = len(archive.event_ids())
        archive_sha256 = sha256_file(archive_path)
        acceleration_sha256 = sha256_file(acc_path)
        for out, arm in ((vel_out, "vfull"), (acc_out, "apair")):
            out.attrs.update({
                "schema_version": CACHE_SCHEMA,
                "arm": arm,
                "backend_version": BACKEND_VERSION,
                "intervention_protocol": PROTOCOL_VERSION,
                "source_archive": str(archive_path),
                "source_archive_sha256": archive_sha256,
                "source_acceleration_hdf5": str(acc_path),
                "source_acceleration_hdf5_sha256": acceleration_sha256,
                "response_correction": "sensitivity_only" if arm == "vfull" else "existing_acceleration_processing",
                "waveform_units": "m/s" if arm == "vfull" else "m/s^2",
                "component_order": "E,N,U",
            })
            out.create_group("data")

        for event_id in archive.event_ids():
            if max_events is not None and (counters["train_events"] + counters["dev_events"]) >= max_events:
                break
            event_split = split.get(str(event_id))
            if event_split == "test":
                counters["excluded_test_events"] += 1
                continue
            if event_split not in {"train", "dev"}:
                counters["excluded_no_acc_event"] += 1
                continue
            if event_id not in station_groups.groups or event_id not in acc["data"]:
                counters["excluded_no_acc_event"] += 1
                continue
            manifest = _source_rows(archive.manifest(event_id))
            if manifest.empty:
                counters["excluded_no_source_pair"] += 1
                continue
            original = station_groups.get_group(event_id).reset_index(drop=True)
            group = acc["data"][event_id]
            station_codes = group["station_codes"][()]
            valid_sources = []
            source_acc_indices = []
            for _, source in manifest.iterrows():
                paired_index = _acc_station_index(station_codes, source["knet_station"])
                if paired_index is None:
                    continue
                valid_sources.append(source)
                source_acc_indices.append(paired_index)
            if not valid_sources:
                counters["excluded_no_source_pair"] += 1
                continue
            sources = pd.DataFrame(valid_sources).reset_index(drop=True)
            channel_table = archive.channel_table(event_id)
            component_sets = channel_table.groupby(
                channel_table["hinet_station"].astype(str).str.upper()
            )["component"].agg(lambda values: {str(value).upper() for value in values})
            good_stations = {
                station for station, components in component_sets.items()
                if {"E", "N", "U"}.issubset(components)
            }
            component_ok = sources["hinet_station"].astype(str).str.upper().isin(good_stations)
            counters["excluded_bad_source"] += int((~component_ok).sum())
            if not component_ok.any():
                counters["excluded_no_source_pair"] += 1
                continue
            kept_positions = np.where(component_ok.to_numpy())[0]
            sources = sources.loc[component_ok].reset_index(drop=True)
            source_acc_indices = [source_acc_indices[position] for position in kept_positions]
            start_timestamp = float(sources["ppick_timestamp"].min()) - PRE_P_SECONDS
            source_picks = np.rint(
                (sources["ppick_timestamp"].astype(float).to_numpy() - start_timestamp)
                * SAMPLING_RATE
            ).astype(np.int64)

            query_indices = original.index[
                original["source_network"].astype(str).str.lower().eq("knt")
            ].tolist()
            if not query_indices:
                counters["excluded_no_acc_event"] += 1
                continue
            n_source = len(sources)
            n_query = len(query_indices)
            n_total = n_source + n_query
            vel_wave = np.zeros((n_total, TRACE_SAMPLES, 3), dtype=np.float32)
            acc_wave = np.zeros_like(vel_wave)
            vel_valid = np.zeros((n_total, TRACE_SAMPLES), dtype=bool)
            acc_valid = np.zeros_like(vel_valid)
            velocity_coords = np.zeros((n_total, 3), dtype=np.float64)
            apair_coords = np.zeros((n_total, 3), dtype=np.float64)
            velocity_picks = np.full(n_total, -1, dtype=np.int64)
            apair_picks = np.full(n_total, -1, dtype=np.int64)
            pga = np.full(n_total, np.nan, dtype=np.float64)
            roles = np.zeros(n_total, dtype=np.uint8)
            retained = np.zeros(n_total, dtype=np.int64)
            template_ids = np.full(n_total, -1, dtype=np.int64)
            source_sensor_ids = [""] * n_total
            paired_acc_sensor_ids = [""] * n_total
            match_distance_km = np.full(n_total, np.nan, dtype=np.float32)
            velocity_sensor_ids: list[str] = []
            apair_sensor_ids: list[str] = []

            try:
                velocity_windows = assemble_event_windows(
                    archive,
                    event_id,
                    sources["hinet_station"].astype(str).tolist(),
                    start_timestamp,
                    n_samples=TRACE_SAMPLES,
                    sampling_rate=SAMPLING_RATE,
                )
            except (ValueError, KeyError):
                counters["excluded_bad_source"] += 1
                continue
            for source_index, source in sources.iterrows():
                wave, support, _ = velocity_windows[str(source["hinet_station"])]
                paired_wave, paired_support = _copy_acc_to_grid(
                    group, source_acc_indices[source_index], start_timestamp
                )
                vel_wave[source_index] = wave
                vel_valid[source_index] = support
                acc_wave[source_index] = paired_wave
                acc_valid[source_index] = paired_support
                velocity_coords[source_index] = (
                    float(source["hinet_lat"]),
                    float(source["hinet_lon"]),
                    float(source["hinet_elevation_m"]) / 1000.0,
                )
                acc_row = source_acc_indices[source_index]
                apair_coords[source_index] = group["coords"][acc_row]
                velocity_picks[source_index] = source_picks[source_index]
                acc_record_start = _timestamp(group["record_start_time_jst"][acc_row].decode())
                acc_axis_start = acc_record_start - int(group["record_start_sample"][acc_row]) / SAMPLING_RATE
                acc_pick_abs = acc_axis_start + float(group["p_picks"][acc_row]) / SAMPLING_RATE
                apair_picks[source_index] = int(round((acc_pick_abs - start_timestamp) * SAMPLING_RATE))
                roles[source_index] = 1
                sensor_id = str(source["hinet_station"])
                velocity_sensor_ids.append(sensor_id)
                acc_code = group["station_codes"][acc_row]
                acc_sensor_id = acc_code.decode() if isinstance(acc_code, bytes) else str(acc_code)
                apair_sensor_ids.append(acc_sensor_id)
                source_sensor_ids[source_index] = sensor_id
                paired_acc_sensor_ids[source_index] = acc_sensor_id
                match_distance_km[source_index] = float(source["match_distance_km"])
                retained[source_index] = assign_retained_prep_samples(
                    templates, event_id, sensor_id, seed=seed
                )
                template_ids[source_index] = stable_index(
                    (event_id, sensor_id), templates.size, seed
                )
            original_coords = group["coords"][()]
            original_pga = group["pga"][()]
            original_picks = group["p_picks"][()]
            original_starts = group["record_start_time_jst"][()]
            original_insertions = group["record_start_sample"][()]
            original_codes = group["station_codes"][()]
            for offset, query_index in enumerate(query_indices, start=n_source):
                row_index = int(original.iloc[query_index]["wave_idx"])
                velocity_coords[offset] = original_coords[row_index]
                apair_coords[offset] = original_coords[row_index]
                pga[offset] = original_pga[row_index]
                record_start = _timestamp(original_starts[row_index].decode())
                axis_start = record_start - int(original_insertions[row_index]) / SAMPLING_RATE
                pick_abs = axis_start + float(original_picks[row_index]) / SAMPLING_RATE
                query_pick = int(round((pick_abs - start_timestamp) * SAMPLING_RATE))
                velocity_picks[offset] = query_pick
                apair_picks[offset] = query_pick
                code = original_codes[row_index]
                query_code = code.decode() if isinstance(code, bytes) else str(code)
                velocity_sensor_ids.append(query_code)
                apair_sensor_ids.append(query_code)

            velocity_station_rows.append(_event_rows(
                original, sources, query_indices, source_acc_indices, event_id, "vfull"
            ))
            apair_station_rows.append(_event_rows(
                original, sources, query_indices, source_acc_indices, event_id, "apair"
            ))
            for out, waves, support, event_coords, event_picks, sensor_ids in (
                (vel_out, vel_wave, vel_valid, velocity_coords, velocity_picks, velocity_sensor_ids),
                (acc_out, acc_wave, acc_valid, apair_coords, apair_picks, apair_sensor_ids),
            ):
                event_group = out["data"].create_group(event_id)
                event_group.create_dataset("waveforms", data=waves, compression="gzip", compression_opts=1)
                event_group.create_dataset("waveform_valid_mask", data=support, compression="gzip", compression_opts=1)
                event_group.create_dataset("coords", data=event_coords)
                event_group.create_dataset("p_picks", data=event_picks)
                event_group.create_dataset("pga", data=pga)
                event_group.create_dataset("v01_source_role", data=roles)
                event_group.create_dataset("v01_retained_prep_samples", data=retained)
                event_group.create_dataset("v01_template_id", data=template_ids)
                event_group.create_dataset("v01_match_distance_km", data=match_distance_km)
                event_group.create_dataset(
                    "v01_reference_p_pick",
                    data=np.asarray(int(source_picks.min()), dtype=np.int64),
                )
                width = max(1, max(len(value.encode()) for value in sensor_ids))
                event_group.create_dataset("station_codes", data=np.asarray(sensor_ids, dtype=f"S{width}"))
                source_width = max(1, max(len(value.encode()) for value in source_sensor_ids))
                paired_width = max(1, max(len(value.encode()) for value in paired_acc_sensor_ids))
                event_group.create_dataset(
                    "v01_source_sensor_id",
                    data=np.asarray(source_sensor_ids, dtype=f"S{source_width}"),
                )
                event_group.create_dataset(
                    "v01_paired_acc_sensor_id",
                    data=np.asarray(paired_acc_sensor_ids, dtype=f"S{paired_width}"),
                )
                event_group.create_dataset("record_start_sample", data=np.zeros(n_total, dtype=np.int64))
                event_group.create_dataset("valid_n_samples", data=support.sum(axis=1).astype(np.int64))
                event_group.attrs["split"] = event_split
                event_group.attrs["absolute_window_start_timestamp"] = start_timestamp
                event_group.attrs["plan_hash"] = _json_hash({
                    "event_id": event_id,
                    "source_ids": sensor_ids[:n_source],
                    "query_ids": sensor_ids[n_source:],
                    "start": start_timestamp,
                    "picks": event_picks.tolist(),
                })
            counters[f"{event_split}_events"] += 1
            counters["source_sensors"] += n_source
            counters["query_targets"] += n_query

        combined_velocity = pd.concat(velocity_station_rows, ignore_index=True) if velocity_station_rows else station_frame.iloc[:0].copy()
        combined_apair = pd.concat(apair_station_rows, ignore_index=True) if apair_station_rows else station_frame.iloc[:0].copy()
        for out, combined in ((vel_out, combined_velocity), (acc_out, combined_apair)):
            meta = out.create_group("metadata")
            meta.create_dataset("sampling_rate", data=np.asarray(SAMPLING_RATE))
            meta.create_dataset("pretrigger_seconds", data=np.asarray(PRE_P_SECONDS))
            _write_table(meta.create_group("station_metadata"), combined)
            _write_table(
                meta.create_group("event_metadata"),
                combined.drop_duplicates("EVENT", keep="first"),
            )
    os.replace(tmp_velocity, velocity_path)
    os.replace(tmp_apair, apair_path)
    counters["velocity_cache"] = str(velocity_path)
    counters["acc_pair_cache"] = str(apair_path)
    counters["velocity_cache_sha256"] = sha256_file(velocity_path)
    counters["acc_pair_cache_sha256"] = sha256_file(apair_path)
    return counters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acc-root", required=True, type=Path)
    parser.add_argument("--velocity-root", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-partial-snapshot", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--years", default="2004-2024", help="Inclusive range or comma-separated years")
    parser.add_argument("--max-events-per-year", type=int)
    args = parser.parse_args()

    split = _split_map(args.split_manifest)
    acc_paths = _acc_paths(args.acc_root.resolve())
    archives = discover_archives(
        args.velocity_root.resolve(), allow_partial=args.allow_partial_snapshot
    )
    archive_by_year = {
        int(path.name.split("_")[-1].split(".")[0]): path for path in archives
    }
    if "-" in args.years and "," not in args.years:
        start, end = (int(value) for value in args.years.split("-", 1))
        selected_years = set(range(start, end + 1))
    else:
        selected_years = {int(value) for value in args.years.split(",") if value.strip()}
    templates = build_train_templates(acc_paths, split)
    args.output_root.mkdir(parents=True, exist_ok=True)
    cohort_audit = audit_intersection(acc_paths, archive_by_year, split, selected_years)
    cohort_audit.to_csv(args.output_root / "cohort_audit.csv", index=False)
    protocol_lock = {
        "schema_version": CACHE_SCHEMA,
        "backend_version": BACKEND_VERSION,
        "intervention_protocol": PROTOCOL_VERSION,
        "seed": args.seed,
        "sampling_rate_hz": SAMPLING_RATE,
        "trace_samples": TRACE_SAMPLES,
        "component_order": ["E", "N", "U"],
        "velocity_units": "m/s",
        "velocity_response_correction": "sensitivity_only",
        "pga_target_units": "log10(m/s^2)",
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "template_count": int(templates.size),
        "template_sha256": hashlib.sha256(templates.tobytes()).hexdigest(),
        "template_retained_prep_seconds": {
            "p01": float(np.quantile(templates, 0.01) / SAMPLING_RATE),
            "p05": float(np.quantile(templates, 0.05) / SAMPLING_RATE),
            "p50": float(np.quantile(templates, 0.50) / SAMPLING_RATE),
            "p95": float(np.quantile(templates, 0.95) / SAMPLING_RATE),
            "fraction_below_1s": float(np.mean(templates < SAMPLING_RATE)),
            "fraction_below_3s": float(np.mean(templates < 3 * SAMPLING_RATE)),
            "fraction_below_5s": float(np.mean(templates < 5 * SAMPLING_RATE)),
        },
        "test_waveforms_read": False,
        "archive_paths": [str(path) for path in archives],
        "cohort_audit": {
            column: int(cohort_audit[column].sum())
            for column in (
                "train_events", "dev_events", "source_sensors", "query_targets",
                "knet_compatible_sources", "kik_hinet_bridge_sources",
                "other_pair_sources", "excluded_test_events",
                "excluded_no_acc_event", "excluded_no_source_pair",
            )
        },
    }
    protocol_lock["protocol_hash"] = _json_hash(protocol_lock)
    (args.output_root / "protocol_lock.json").write_text(
        json.dumps(protocol_lock, indent=2, sort_keys=True) + "\n"
    )
    if args.audit_only:
        print(json.dumps(protocol_lock, indent=2, sort_keys=True))
        return

    results = []
    for year, archive_path in sorted(archive_by_year.items()):
        if year not in acc_paths or year not in selected_years:
            continue
        result = materialize_year(
            year,
            acc_paths[year],
            archive_path,
            args.output_root,
            split,
            templates,
            seed=args.seed,
            overwrite=args.overwrite,
            max_events=args.max_events_per_year,
        )
        results.append(result)
        print(json.dumps(result, sort_keys=True))
    frame = pd.DataFrame(results)
    frame.to_csv(args.output_root / "cohort_counts_by_year.csv", index=False)
    summary = {
        "protocol_lock": protocol_lock,
        "years_materialized": len(results),
        "train_events": int(frame.get("train_events", pd.Series(dtype=int)).sum()),
        "dev_events": int(frame.get("dev_events", pd.Series(dtype=int)).sum()),
        "source_sensors": int(frame.get("source_sensors", pd.Series(dtype=int)).sum()),
        "query_targets": int(frame.get("query_targets", pd.Series(dtype=int)).sum()),
        "note": "Only train/dev intersection was materialized; test waveforms were not opened.",
    }
    (args.output_root / "preflight_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
