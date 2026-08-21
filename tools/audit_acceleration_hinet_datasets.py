#!/usr/bin/env python3
"""Audit the raw K-NET/KiK-net acceleration download and Hi-net archive.

The audit is intentionally read-only with respect to the source datasets.  It
writes reproducible CSV summaries and two publication-sized QC figures to a
separate report directory.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import math
import re
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Keep SVG text editable and use the same font family in every export.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.download_hinet_velocity import load_event_table, load_station_table  # noqa: E402
from tools.hinet_raw_archive import AnnualHinetArchiveReader  # noqa: E402
from tools.plot_hinet_accel_velocity_qc import (  # noqa: E402
    load_archive_velocity,
    load_training_station,
    waveform_qc_metrics,
)


PALETTE = {
    "raw": "#777777",
    "processed": "#356AA0",
    "velocity": "#D9822B",
    "nomatch": "#B9B9B9",
    "missing": "#B64342",
    "vertical": "#2F7F75",
    "three": "#725A9D",
    "grid": "#D9D9D9",
    "text": "#333333",
    "pass": "#2F7F5F",
    "fail": "#B64342",
}


REPRESENTATIVE_CASES = (
    {
        "year": 2004,
        "event_id": "20040401054100",
        "knet_station": "IWTH04",
        "hinet_station": "N.SMTH",
        "knet_height_m": 620.0,
        "title": "Earliest available (2004)",
    },
    {
        "year": 2011,
        "event_id": "20110311144600",
        "knet_station": "IWTH05",
        "hinet_station": "N.FSWH",
        "knet_height_m": 120.0,
        "title": "M9.0 stress case (2011)",
    },
    {
        "year": 2012,
        "event_id": "20121231071700",
        "knet_station": "MYGH13",
        "hinet_station": "N.MSRH",
        "knet_height_m": 90.0,
        "title": "Missing-channel control (2012)",
    },
    {
        "year": 2024,
        "event_id": "20240101160600",
        "knet_station": "GNMH08",
        "hinet_station": "N.TUMH",
        "knet_height_m": 1040.0,
        "title": "Recent record (2024)",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--acceleration-root",
        type=Path,
        default=Path("/run/media/zhangb/My Passport/knet_data"),
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path(
            "/run/media/zhangb/My Passport/knet_converted/"
            "origin_corrected_diting_vel_acc_vs30"
        ),
    )
    parser.add_argument(
        "--velocity-root",
        type=Path,
        default=Path("/run/media/zhangb/My Passport/hinet_data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports" / "hinet_dataset_audit_20260811",
    )
    parser.add_argument(
        "--validate-all-tars",
        action="store_true",
        help="Open every acceleration outer tar. This takes about 15 minutes on the external disk.",
    )
    parser.add_argument("--sample-events-per-year", type=int, default=3)
    return parser.parse_args()


def event_id_from_key(value: object) -> str:
    return str(value).strip().split(",", 1)[0]


def numeric_token(values: pd.Series, pattern: str) -> pd.Series:
    return pd.to_numeric(values.astype(str).str.extract(pattern)[0], errors="coerce")


def load_acceleration_catalog(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted((root / "catalogs").glob("*.csv")):
        frame = pd.read_csv(path, dtype=str)
        frame["catalog_file"] = path.name
        match = re.match(r"(?P<year>\d{4})_(?P<month>\d{2})_", path.name)
        frame["catalog_year"] = int(match.group("year")) if match else -1
        frame["catalog_month"] = int(match.group("month")) if match else -1
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No catalog CSV files under {root / 'catalogs'}")
    catalog = pd.concat(frames, ignore_index=True)
    catalog["event_id"] = catalog["keys"].map(event_id_from_key)
    catalog["year"] = pd.to_numeric(catalog["event_id"].str[:4], errors="coerce")
    catalog["magnitude"] = numeric_token(catalog["mag"], r"M?([+-]?[0-9.]+)")
    catalog["depth_km"] = numeric_token(catalog["dis"], r"([+-]?[0-9.]+)")
    catalog["latitude"] = numeric_token(catalog["lat"], r"([+-]?[0-9.]+)")
    catalog["longitude"] = numeric_token(catalog["lon"], r"([+-]?[0-9.]+)")
    catalog["advertised_site_count"] = numeric_token(catalog["n_site"], r"(\d+)")
    return catalog


def acceleration_tar_inventory(
    root: Path,
    catalog: pd.DataFrame,
    *,
    validate_all: bool,
    sample_events_per_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tar_paths = sorted((root / "waveforms").glob("*/*.tar"))
    catalog_ids = set(catalog["event_id"].astype(str))
    rows: list[dict[str, object]] = []
    started = time.time()
    for index, path in enumerate(tar_paths, start=1):
        names: list[str] = []
        error = ""
        if validate_all:
            try:
                with tarfile.open(path, "r:*") as archive:
                    names = [member.name for member in archive.getmembers() if member.isfile()]
            except Exception as exc:  # pragma: no cover - depends on source corruption
                error = repr(exc)
        else:
            names = []
        has_knet = any(".knt." in name for name in names)
        has_kiknet = any(".kik." in name for name in names)
        has_image = any(".img." in name for name in names)
        if validate_all:
            product_class = (
                "both"
                if has_knet and has_kiknet
                else "knet_only"
                if has_knet
                else "kiknet_only"
                if has_kiknet
                else "neither"
            )
        else:
            product_class = "not_checked"
        rows.append(
            {
                "year": int(path.parent.name),
                "event_id": path.stem,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "catalog_event": int(path.stem in catalog_ids),
                "outer_tar_valid": int(not error) if validate_all else np.nan,
                "outer_member_count": len(names) if validate_all else np.nan,
                "has_knet_product": int(has_knet) if validate_all else np.nan,
                "has_kiknet_product": int(has_kiknet) if validate_all else np.nan,
                "has_image_product": int(has_image) if validate_all else np.nan,
                "product_class": product_class,
                "error": error,
            }
        )
        if validate_all and index % 1000 == 0:
            print(
                f"[acceleration tar] {index}/{len(tar_paths)} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    tar_df = pd.DataFrame(rows)

    catalog_unique = catalog.drop_duplicates("event_id", keep="first").sort_values("event_id")
    sample_rows: list[dict[str, object]] = []
    for year, year_frame in catalog_unique.groupby("year"):
        year_frame = year_frame.reset_index(drop=True)
        if year_frame.empty:
            continue
        count = max(1, int(sample_events_per_year))
        positions = np.linspace(0, len(year_frame) - 1, min(count, len(year_frame)), dtype=int)
        for position in sorted(set(map(int, positions))):
            row = year_frame.iloc[position]
            event_id = str(row["event_id"])
            path = root / "waveforms" / str(int(year)) / f"{event_id}.tar"
            network_counts = {"knt": 0, "kik": 0}
            error = ""
            try:
                with tarfile.open(path, "r:*") as outer:
                    for member in outer.getmembers():
                        if not member.isfile() or ".kwin.tar" not in member.name:
                            continue
                        network = (
                            "knt"
                            if ".knt." in member.name
                            else "kik"
                            if ".kik." in member.name
                            else ""
                        )
                        if not network:
                            continue
                        payload = outer.extractfile(member).read()
                        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as inner:
                            network_counts[network] = sum(
                                item.isfile()
                                and item.name.endswith(".kwin")
                                and item.size > 0
                                for item in inner.getmembers()
                            )
            except Exception as exc:  # pragma: no cover - depends on source corruption
                error = repr(exc)
            actual = network_counts["knt"] + network_counts["kik"]
            advertised = int(row["advertised_site_count"])
            sample_rows.append(
                {
                    "year": int(year),
                    "event_id": event_id,
                    "advertised_site_count": advertised,
                    "inner_kwin_count": actual,
                    "knet_kwin_count": network_counts["knt"],
                    "kiknet_kwin_count": network_counts["kik"],
                    "site_count_matches": int(actual == advertised and not error),
                    "error": error,
                }
            )
    return tar_df, pd.DataFrame(sample_rows)


def load_processed_inventory(
    processed_root: Path,
    catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, set[str]]]:
    catalog_ids_by_year = {
        int(year): set(frame["event_id"].astype(str))
        for year, frame in catalog[catalog["year"] <= 2024].groupby("year")
    }
    event_frames: list[pd.DataFrame] = []
    station_frames: list[pd.DataFrame] = []
    year_rows: list[dict[str, object]] = []
    event_ids_by_year: dict[int, set[str]] = {}
    for year in range(2000, 2025):
        path = processed_root / str(year) / f"japan_{year}.hdf5"
        events = load_event_table(path)
        stations = load_station_table(path)
        event_key = "EVENT" if "EVENT" in events else "KiK_File"
        ids = set(events[event_key].astype(str))
        event_ids_by_year[year] = ids
        events = events.copy()
        stations = stations.copy()
        events["year"] = year
        stations["year"] = year
        event_frames.append(events)
        station_frames.append(stations)
        source_network = stations["source_network"].astype(str).str.lower()
        year_rows.append(
            {
                "year": year,
                "raw_catalog_events": len(catalog_ids_by_year.get(year, set())),
                "processed_events": len(ids),
                "raw_not_processed": len(catalog_ids_by_year.get(year, set()) - ids),
                "processed_not_raw": len(ids - catalog_ids_by_year.get(year, set())),
                "processed_station_rows": len(stations),
                "knet_station_rows": int(source_network.eq("knt").sum()),
                "kiknet_station_rows": int(source_network.eq("kik").sum()),
                "unique_station_codes": stations["station_code"].astype(str).nunique(),
            }
        )
        print(f"[processed HDF5] year={year} events={len(ids)} rows={len(stations)}", flush=True)
    return (
        pd.DataFrame(year_rows),
        pd.concat(event_frames, ignore_index=True),
        pd.concat(station_frames, ignore_index=True),
        event_ids_by_year,
    )


def failure_category(error: object) -> str:
    text = str(error)
    if "Data not available in the time period" in text:
        return "provider_data_unavailable"
    if "RemoteDisconnected" in text or "ConnectionError" in text:
        return "connection_interrupted"
    if "time coverage is incomplete" in text:
        return "incomplete_time_coverage"
    if "NoneType" in text and "not iterable" in text:
        return "hinetpy_empty_response"
    if "FileNotFoundError" in text and "channel_table=" in text:
        return "channel_table_or_merge_file_missing"
    if "missing cnt=" in text:
        return "download_returned_no_files"
    return "other"


def archive_path_from_summary(velocity_root: Path, year: int) -> tuple[Path, dict[str, object]]:
    summary_path = velocity_root / "catalog" / f"hinet_archive_{year}.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return Path(str(summary["archive"])), summary


def components_by_station(channel_table: pd.DataFrame) -> dict[str, set[str]]:
    output: dict[str, set[str]] = collections.defaultdict(set)
    for _, row in channel_table.iterrows():
        output[str(row["hinet_station"]).upper()].add(str(row["component"]).upper())
    return output


def verify_event_payload(reader: AnnualHinetArchiveReader, event_id: str) -> None:
    reader.cnt_items(event_id, verify=True)
    reader.channel_table_item(event_id, verify=True)


def robust_rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    centered = values - np.median(values)
    return float(np.sqrt(np.mean(centered * centered)))


def energy_ratio(times: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    pre = values[(times >= -20.0) & (times <= -1.0)]
    post = values[(times >= 0.0) & (times <= 30.0)]
    pre_rms = robust_rms(pre)
    post_rms = robust_rms(post)
    ratio = post_rms / pre_rms if np.isfinite(pre_rms) and pre_rms > 0 else float("nan")
    return pre_rms, post_rms, ratio


def velocity_series_quality(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values)
    if values.size == 0:
        return {
            "velocity_zero_fraction": np.nan,
            "velocity_unique_fraction": np.nan,
            "velocity_extreme_repeat_fraction": np.nan,
            "velocity_min_counts": np.nan,
            "velocity_max_counts": np.nan,
        }
    minimum = np.min(values)
    maximum = np.max(values)
    extreme_fraction = max(float(np.mean(values == minimum)), float(np.mean(values == maximum)))
    return {
        "velocity_zero_fraction": float(np.mean(values == 0)),
        "velocity_unique_fraction": float(np.unique(values).size / values.size),
        "velocity_extreme_repeat_fraction": extreme_fraction,
        "velocity_min_counts": float(minimum),
        "velocity_max_counts": float(maximum),
    }


def select_sample_rows(events: pd.DataFrame, reader: AnnualHinetArchiveReader) -> list[pd.Series]:
    events = events[events["raw_status"].astype(str).str.startswith("downloaded")]
    events = events.sort_values("event_id").reset_index(drop=True)
    if events.empty:
        return []
    positions = sorted(set([0, len(events) // 2, len(events) - 1]))
    rows: list[pd.Series] = []
    for sample_index, position in enumerate(positions):
        event_id = str(events.iloc[position]["event_id"])
        manifest = reader.manifest(event_id).sort_values(
            ["match_distance_km", "knet_station", "knet_height_m"]
        ).reset_index(drop=True)
        if manifest.empty:
            continue
        station_positions = [0, len(manifest) // 2, len(manifest) - 1]
        row = manifest.iloc[station_positions[min(sample_index, 2)]].copy()
        row["sample_role"] = (
            "earliest_closest"
            if sample_index == 0
            else "midyear_median"
            if sample_index == 1
            else "latest_farthest"
        )
        rows.append(row)
    return rows


def run_waveform_sample_qc(
    *,
    year: int,
    archive_path: Path,
    hdf5_path: Path,
    velocity_root: Path,
    reader: AnnualHinetArchiveReader,
    event_table: pd.DataFrame,
) -> list[dict[str, object]]:
    args = SimpleNamespace(
        hdf5=hdf5_path,
        download_root=velocity_root,
        wave_idx=None,
        component="vertical",
        pre_seconds=120.0,
        post_seconds=120.0,
        max_match_distance_km=0.5,
        large_offset_seconds=20.0,
    )
    results: list[dict[str, object]] = []
    sample_rows = select_sample_rows(event_table, reader)
    with h5py.File(hdf5_path, "r") as h5:
        for row in sample_rows:
            row = row.copy()
            event_id = str(row["event_id"])
            row["archive_path"] = str(archive_path)
            row["archive_event_id"] = event_id
            base: dict[str, object] = {
                "year": year,
                "event_id": event_id,
                "sample_role": row["sample_role"],
                "knet_station": row["knet_station"],
                "hinet_station": row["hinet_station"],
                "knet_height_m": row.get("knet_height_m", np.nan),
                "match_distance_km": row.get("match_distance_km", np.nan),
                "event_magnitude": row.get("event_magnitude", np.nan),
                "epicentral_distance_km": row.get("epicentral_distance_km", np.nan),
            }
            try:
                verify_event_payload(reader, event_id)
                training = load_training_station(h5, row, args)
                velocity = load_archive_velocity(row, args)
                qc = waveform_qc_metrics(training, velocity, row, args)
                acc_pre, acc_post, acc_ratio = energy_ratio(
                    training.acceleration_t_rel, training.acceleration
                )
                vel_pre, vel_post, vel_ratio = energy_ratio(velocity.t_rel, velocity.values)
                base.update(qc)
                base.update(training.summary)
                base.update(
                    {
                        "archive_checksum_ok": 1,
                        "velocity_source": velocity.source,
                        "velocity_status": velocity.status,
                        "acceleration_pre_p_rms": acc_pre,
                        "acceleration_post_p_rms": acc_post,
                        "acceleration_post_pre_ratio": acc_ratio,
                        "velocity_pre_p_rms_counts": vel_pre,
                        "velocity_post_p_rms_counts": vel_post,
                        "velocity_post_pre_ratio": vel_ratio,
                        "final_pick_minus_theoretical_p_s": training.pick_rel_seconds.get(
                            "p_picks", np.nan
                        ),
                    }
                )
                base.update(velocity_series_quality(velocity.values))
            except Exception as exc:  # pragma: no cover - depends on source corruption
                base.update(
                    {
                        "archive_checksum_ok": 0,
                        "qc_status": "ERROR",
                        "qc_fail_reasons": repr(exc),
                    }
                )
            results.append(base)
    return results


def audit_velocity(
    processed_root: Path,
    velocity_root: Path,
    event_ids_by_year: dict[int, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    year_rows: list[dict[str, object]] = []
    missing_event_rows: list[dict[str, object]] = []
    channel_gap_rows: list[dict[str, object]] = []
    sample_qc_rows: list[dict[str, object]] = []
    for year in range(2000, 2025):
        archive_path, archive_summary = archive_path_from_summary(velocity_root, year)
        event_catalog_path = velocity_root / "catalog" / f"hinet_events_{year}.csv"
        attempt_catalog_path = velocity_root / "catalog" / f"hinet_attempts_{year}.csv"
        events = pd.read_csv(event_catalog_path, dtype={"event_id": str})
        attempts = pd.read_csv(attempt_catalog_path, dtype={"event_id": str})
        committed_ids = set(events["event_id"].astype(str))
        missing_ids = sorted(event_ids_by_year[year] - committed_ids)
        last_attempt = attempts.drop_duplicates("event_id", keep="last").set_index("event_id")
        for event_id in missing_ids:
            error = last_attempt.loc[event_id, "error"] if event_id in last_attempt.index else ""
            missing_event_rows.append(
                {
                    "year": year,
                    "event_id": event_id,
                    "failure_category": failure_category(error),
                    "last_error": error,
                }
            )

        status = events["raw_status"].astype(str)
        downloaded = events[status.str.startswith("downloaded")]
        counts = collections.Counter()
        match_distances: list[float] = []
        with AnnualHinetArchiveReader(archive_path) as reader:
            reader_events = reader.events_dataframe()
            for _, event_row in reader_events.iterrows():
                if not str(event_row["raw_status"]).startswith("downloaded"):
                    continue
                event_id = str(event_row["event_id"])
                manifest = reader.manifest(event_id)
                channel_table = reader.channel_table(event_id)
                component_map = components_by_station(channel_table)
                counts["downloaded_events"] += 1
                event_missing_vertical = False
                for _, manifest_row in manifest.iterrows():
                    station = str(manifest_row["hinet_station"]).upper()
                    components = component_map.get(station, set())
                    vertical_ok = bool(components & {"U", "Z"})
                    three_ok = (
                        vertical_ok
                        and bool(components & {"N", "1"})
                        and bool(components & {"E", "2"})
                    )
                    counts["manifest_rows"] += 1
                    counts["vertical_rows_ok" if vertical_ok else "vertical_rows_missing"] += 1
                    counts["three_component_rows_ok" if three_ok else "three_component_rows_missing"] += 1
                    distance = float(manifest_row.get("match_distance_km", np.nan))
                    if np.isfinite(distance):
                        match_distances.append(distance)
                    if not vertical_ok or not three_ok:
                        channel_gap_rows.append(
                            {
                                "year": year,
                                "event_id": event_id,
                                "knet_station": manifest_row.get("knet_station", ""),
                                "knet_height_m": manifest_row.get("knet_height_m", np.nan),
                                "hinet_station": manifest_row.get("hinet_station", ""),
                                "match_distance_km": distance,
                                "available_components": ";".join(sorted(components)),
                                "vertical_available": int(vertical_ok),
                                "three_components_available": int(three_ok),
                            }
                        )
                    if not vertical_ok:
                        event_missing_vertical = True
                if event_missing_vertical:
                    counts["events_with_missing_vertical"] += 1

            hdf5_path = processed_root / str(year) / f"japan_{year}.hdf5"
            sample_qc_rows.extend(
                run_waveform_sample_qc(
                    year=year,
                    archive_path=archive_path,
                    hdf5_path=hdf5_path,
                    velocity_root=velocity_root,
                    reader=reader,
                    event_table=reader_events,
                )
            )

        failure_counts = collections.Counter(
            row["failure_category"] for row in missing_event_rows if row["year"] == year
        )
        match_array = np.asarray(match_distances, dtype=np.float64)
        year_rows.append(
            {
                "year": year,
                "source_events": len(event_ids_by_year[year]),
                "archive_complete": int(bool(archive_summary.get("complete"))),
                "archive_filename": archive_path.name,
                "archive_size_bytes": archive_path.stat().st_size,
                "committed_events": len(events),
                "downloaded_events": len(downloaded),
                "no_matched_station_events": int(status.eq("no_matched_stations").sum()),
                "missing_events": len(missing_ids),
                "requested_station_rows": int(counts["manifest_rows"]),
                "vertical_rows_available": int(counts["vertical_rows_ok"]),
                "vertical_rows_missing": int(counts["vertical_rows_missing"]),
                "three_component_rows_available": int(counts["three_component_rows_ok"]),
                "three_component_rows_missing": int(counts["three_component_rows_missing"]),
                "events_with_missing_vertical": int(counts["events_with_missing_vertical"]),
                "match_distance_median_km": (
                    float(np.median(match_array)) if match_array.size else np.nan
                ),
                "match_distance_p95_km": (
                    float(np.percentile(match_array, 95)) if match_array.size else np.nan
                ),
                "provider_data_unavailable": failure_counts["provider_data_unavailable"],
                "channel_table_or_merge_file_missing": failure_counts[
                    "channel_table_or_merge_file_missing"
                ],
                "hinetpy_empty_response": failure_counts["hinetpy_empty_response"],
                "connection_interrupted": failure_counts["connection_interrupted"],
                "incomplete_time_coverage": failure_counts["incomplete_time_coverage"],
                "other_missing_reason": failure_counts["other"],
            }
        )
        print(
            f"[velocity] year={year} downloaded={len(downloaded)} missing={len(missing_ids)} "
            f"channel_gaps={counts['vertical_rows_missing']}",
            flush=True,
        )
    return (
        pd.DataFrame(year_rows),
        pd.DataFrame(missing_event_rows),
        pd.DataFrame(channel_gap_rows),
        pd.DataFrame(sample_qc_rows),
    )


def pick_manifest_case(
    reader: AnnualHinetArchiveReader,
    case: dict[str, object],
) -> pd.Series:
    manifest = reader.manifest(str(case["event_id"]))
    selected = manifest[
        manifest["knet_station"].astype(str).eq(str(case["knet_station"]))
        & manifest["hinet_station"].astype(str).eq(str(case["hinet_station"]))
    ].copy()
    if selected.empty:
        raise KeyError(f"Representative case not found: {case}")
    target_height = float(case["knet_height_m"])
    heights = pd.to_numeric(selected["knet_height_m"], errors="coerce")
    return selected.iloc[int(np.nanargmin(np.abs(heights.to_numpy() - target_height)))].copy()


def downsample_xy(x: np.ndarray, y: np.ndarray, maximum: int = 3500) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size <= maximum:
        return x, y
    stride = int(math.ceil(x.size / maximum))
    return x[::stride], y[::stride]


def load_representative_cases(
    processed_root: Path,
    velocity_root: Path,
) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    cases: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for case in REPRESENTATIVE_CASES:
        year = int(case["year"])
        archive_path, _ = archive_path_from_summary(velocity_root, year)
        hdf5_path = processed_root / str(year) / f"japan_{year}.hdf5"
        args = SimpleNamespace(
            hdf5=hdf5_path,
            download_root=velocity_root,
            wave_idx=None,
            component="vertical",
            pre_seconds=120.0,
            post_seconds=120.0,
            max_match_distance_km=0.5,
            large_offset_seconds=20.0,
        )
        with AnnualHinetArchiveReader(archive_path) as reader, h5py.File(hdf5_path, "r") as h5:
            row = pick_manifest_case(reader, case)
            row["archive_path"] = str(archive_path)
            row["archive_event_id"] = str(case["event_id"])
            training = load_training_station(h5, row, args)
            velocity = load_archive_velocity(row, args)
            qc = waveform_qc_metrics(training, velocity, row, args)
            item = {
                **case,
                "row": row,
                "training": training,
                "velocity": velocity,
                "qc": qc,
            }
            cases.append(item)
            acc_pre, acc_post, acc_ratio = energy_ratio(
                training.acceleration_t_rel, training.acceleration
            )
            vel_pre, vel_post, vel_ratio = energy_ratio(velocity.t_rel, velocity.values)
            velocity_quality = velocity_series_quality(velocity.values)
            absolute_maximum = (
                float(np.max(np.abs(velocity.values))) if velocity.values.size else np.nan
            )
            summary_rows.append(
                {
                    "year": year,
                    "case_title": case["title"],
                    "event_id": case["event_id"],
                    "event_magnitude": row.get("event_magnitude", np.nan),
                    "knet_station": case["knet_station"],
                    "knet_height_m": case["knet_height_m"],
                    "hinet_station": case["hinet_station"],
                    "match_distance_km": row.get("match_distance_km", np.nan),
                    "qc_status": qc.get("qc_status", ""),
                    "qc_fail_reasons": qc.get("qc_fail_reasons", ""),
                    "velocity_status": velocity.status,
                    "velocity_source": velocity.source,
                    "acceleration_pre_p_rms": acc_pre,
                    "acceleration_post_p_rms": acc_post,
                    "acceleration_post_pre_ratio": acc_ratio,
                    "velocity_pre_p_rms_counts": vel_pre,
                    "velocity_post_p_rms_counts": vel_post,
                    "velocity_post_pre_ratio": vel_ratio,
                    "velocity_abs_max_counts": absolute_maximum,
                    "velocity_hits_abs_2pow26": int(
                        np.isfinite(absolute_maximum) and absolute_maximum == float(2**26)
                    ),
                    **velocity_quality,
                }
            )
            acc_x, acc_y = downsample_xy(training.acceleration_t_rel, training.acceleration)
            for x_value, y_value in zip(acc_x, acc_y):
                source_rows.append(
                    {
                        "year": year,
                        "event_id": case["event_id"],
                        "knet_station": case["knet_station"],
                        "hinet_station": case["hinet_station"],
                        "series": "acceleration_vertical_mps2",
                        "seconds_relative_to_theoretical_p": x_value,
                        "value": y_value,
                    }
                )
            vel_x, vel_y = downsample_xy(velocity.t_rel, velocity.values)
            for x_value, y_value in zip(vel_x, vel_y):
                source_rows.append(
                    {
                        "year": year,
                        "event_id": case["event_id"],
                        "knet_station": case["knet_station"],
                        "hinet_station": case["hinet_station"],
                        "series": "hinet_vertical_raw_counts",
                        "seconds_relative_to_theoretical_p": x_value,
                        "value": y_value,
                    }
                )
    return cases, pd.DataFrame(source_rows), pd.DataFrame(summary_rows)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
    )


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_overview(
    acceleration_yearly: pd.DataFrame,
    velocity_yearly: pd.DataFrame,
    missing_events: pd.DataFrame,
    output_dir: Path,
) -> None:
    merged = acceleration_yearly.merge(velocity_yearly, on="year", how="left")
    merged = merged[merged["year"] <= 2024].copy()
    years = merged["year"].to_numpy(dtype=int)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(years, merged["raw_catalog_events"], "o-", color=PALETTE["raw"], ms=2.5, lw=1.0, label="Raw acceleration catalog")
    ax.plot(years, merged["processed_events"], "o-", color=PALETTE["processed"], ms=2.5, lw=1.1, label="Processed acceleration HDF5")
    ax.plot(years, merged["downloaded_events"], "o-", color=PALETTE["velocity"], ms=2.5, lw=1.1, label="Hi-net raw waveform archived")
    ax.set_ylabel("Events")
    ax.set_title("Event retention across data stages", loc="left", fontweight="bold")
    ax.legend(fontsize=6.2)
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.5)
    panel_label(ax, "a")

    ax = axes[0, 1]
    ax.bar(years, merged["downloaded_events"], color=PALETTE["velocity"], width=0.78, label="Downloaded")
    ax.bar(years, merged["no_matched_station_events"], bottom=merged["downloaded_events"], color=PALETTE["nomatch"], width=0.78, label="No matched station")
    ax.bar(years, merged["missing_events"], bottom=merged["downloaded_events"] + merged["no_matched_station_events"], color=PALETTE["missing"], width=0.78, label="Not archived")
    ax.set_ylabel("Processed-source events")
    ax.set_title("Current Hi-net archive outcome", loc="left", fontweight="bold")
    ax.legend(fontsize=6.2, ncol=1)
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.5)
    panel_label(ax, "b")

    ax = axes[1, 0]
    denom = merged["requested_station_rows"].replace(0, np.nan)
    vertical_pct = 100.0 * merged["vertical_rows_available"] / denom
    three_pct = 100.0 * merged["three_component_rows_available"] / denom
    ax.plot(years, vertical_pct, "o-", color=PALETTE["vertical"], ms=2.5, lw=1.1, label="Vertical available")
    ax.plot(years, three_pct, "s-", color=PALETTE["three"], ms=2.2, lw=1.0, label="All 3 components")
    finite = np.concatenate([vertical_pct.dropna().to_numpy(), three_pct.dropna().to_numpy()])
    lower = max(97.5, math.floor(float(np.min(finite)) * 10.0) / 10.0 - 0.1) if finite.size else 97.5
    ax.set_ylim(lower, 100.05)
    ax.set_ylabel("Requested station rows available (%)")
    ax.set_title("Per-station channel-table completeness", loc="left", fontweight="bold")
    ax.legend(fontsize=6.2)
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.5)
    panel_label(ax, "c")

    ax = axes[1, 1]
    reason_order = [
        "provider_data_unavailable",
        "channel_table_or_merge_file_missing",
        "hinetpy_empty_response",
        "connection_interrupted",
        "incomplete_time_coverage",
        "download_returned_no_files",
        "other",
    ]
    reason_labels = {
        "provider_data_unavailable": "Provider: no data for period",
        "channel_table_or_merge_file_missing": "Channel/merge file missing",
        "hinetpy_empty_response": "HinetPy empty response",
        "connection_interrupted": "Connection interrupted",
        "incomplete_time_coverage": "Incomplete time coverage",
        "download_returned_no_files": "No files returned",
        "other": "Other",
    }
    counts = missing_events["failure_category"].value_counts()
    present = [reason for reason in reason_order if counts.get(reason, 0) > 0]
    values = np.asarray([counts[reason] for reason in present], dtype=float)
    y_pos = np.arange(len(present))[::-1]
    colors = [PALETTE["missing"] if reason == "provider_data_unavailable" else PALETTE["raw"] for reason in present]
    ax.barh(y_pos, values, color=colors, height=0.62)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([reason_labels[reason] for reason in present], fontsize=6.2)
    ax.set_xscale("log")
    ax.set_xlabel("Missing events (log scale)")
    ax.set_title("Final reason for unarchived events", loc="left", fontweight="bold")
    for y_value, value in zip(y_pos, values):
        ax.text(value * 1.08, y_value, f"{int(value):,}", va="center", fontsize=6.2)
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.5)
    panel_label(ax, "d")

    for ax in axes.flat:
        ax.tick_params(labelsize=6.3, width=0.7, length=2.5)
    for ax in axes[:, 0]:
        ax.set_xticks(np.arange(2000, 2025, 4))
    axes[0, 1].set_xticks(np.arange(2000, 2025, 4))
    save_figure(fig, output_dir / "figures" / "dataset_coverage_overview")


def plot_representative_waveforms(cases: list[dict[str, object]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, len(cases), figsize=(7.2, 4.15), sharex=True, constrained_layout=True)
    letters = "abcd"
    for column, case in enumerate(cases):
        training = case["training"]
        velocity = case["velocity"]
        qc = case["qc"]
        top = axes[0, column]
        bottom = axes[1, column]
        top.plot(training.acceleration_t_rel, training.acceleration, color=PALETTE["processed"], lw=0.55)
        bottom.axvline(0.0, color=PALETTE["pass"], lw=0.9, ls="--")
        top.axvline(0.0, color=PALETTE["pass"], lw=0.9, ls="--")
        if velocity.t_rel.size:
            bottom.plot(velocity.t_rel, velocity.values, color=PALETTE["velocity"], lw=0.55)
        else:
            bottom.text(
                0.5,
                0.54,
                "Requested station absent\nfrom archived channel table",
                transform=bottom.transAxes,
                ha="center",
                va="center",
                fontsize=6.3,
                color=PALETTE["fail"],
            )
        status_color = PALETTE["pass"] if qc["qc_status"] == "PASS" else PALETTE["fail"]
        top.set_title(
            f"{case['title']}\n{case['event_id']} | M{float(case['row']['event_magnitude']):.1f}",
            fontsize=6.7,
            loc="left",
        )
        top.text(
            0.99,
            0.96,
            str(qc["qc_status"]),
            transform=top.transAxes,
            ha="right",
            va="top",
            fontsize=6.3,
            fontweight="bold",
            color=status_color,
        )
        top.text(-0.16, 1.08, letters[column], transform=top.transAxes, fontweight="bold", fontsize=8)
        bottom.text(
            0.02,
            0.96,
            f"{case['knet_station']} ↔ {case['hinet_station']}\n"
            f"separation {float(case['row']['match_distance_km']):.3f} km",
            transform=bottom.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            color=PALETTE["text"],
        )
        if velocity.values.size and np.max(np.abs(velocity.values)) == float(2**26):
            bottom.text(
                0.98,
                0.04,
                "Touches 2^26 count boundary",
                transform=bottom.transAxes,
                ha="right",
                va="bottom",
                fontsize=5.8,
                color=PALETTE["fail"],
            )
        top.set_xlim(-20, 60)
        bottom.set_xlim(-20, 60)
        top.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
        bottom.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
        top.grid(axis="x", color=PALETTE["grid"], lw=0.45)
        bottom.grid(axis="x", color=PALETTE["grid"], lw=0.45)
        bottom.set_xlabel("Seconds from theoretical P")
        top.tick_params(labelsize=5.8, width=0.7, length=2.3)
        bottom.tick_params(labelsize=5.8, width=0.7, length=2.3)
    axes[0, 0].set_ylabel("Acceleration (m s^-2)")
    axes[1, 0].set_ylabel("Hi-net raw counts")
    fig.text(
        0.5,
        1.01,
        "Representative acceleration and Hi-net raw-count QC (vertical component)",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )
    save_figure(fig, output_dir / "figures" / "representative_waveform_qc")


def build_acceleration_yearly(
    catalog: pd.DataFrame,
    tar_inventory: pd.DataFrame,
    processed_yearly: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year, year_catalog in catalog.groupby("year"):
        year = int(year)
        year_tars = tar_inventory[tar_inventory["year"] == year]
        classes = year_tars["product_class"].value_counts()
        rows.append(
            {
                "year": year,
                "catalog_rows": len(year_catalog),
                "raw_catalog_events": year_catalog["event_id"].nunique(),
                "advertised_station_records": int(year_catalog["advertised_site_count"].sum()),
                "tar_files": len(year_tars),
                "tar_size_bytes": int(year_tars["size_bytes"].sum()),
                "outer_tar_valid": int(year_tars["outer_tar_valid"].fillna(0).sum()),
                "both_network_products": int(classes.get("both", 0)),
                "knet_only_products": int(classes.get("knet_only", 0)),
                "kiknet_only_products": int(classes.get("kiknet_only", 0)),
                "invalid_or_empty_products": int(classes.get("neither", 0)),
            }
        )
    out = pd.DataFrame(rows)
    return out.merge(processed_yearly, on=["year", "raw_catalog_events"], how="left")


def summary_table(
    catalog: pd.DataFrame,
    acceleration_yearly: pd.DataFrame,
    tar_sample: pd.DataFrame,
    processed_events: pd.DataFrame,
    processed_stations: pd.DataFrame,
    velocity_yearly: pd.DataFrame,
    velocity_missing: pd.DataFrame,
    channel_gaps: pd.DataFrame,
    sample_qc: pd.DataFrame,
) -> pd.DataFrame:
    raw_2000_2024 = catalog[catalog["year"] <= 2024]
    status = sample_qc.get("qc_status", pd.Series(dtype=str)).astype(str)
    velocity_loaded = sample_qc.get("velocity_status", pd.Series(dtype=str)).astype(str).eq("loaded")
    vertical_gap_count = int((channel_gaps.get("vertical_available", pd.Series(dtype=int)) == 0).sum())
    three_gap_count = int((channel_gaps.get("three_components_available", pd.Series(dtype=int)) == 0).sum())
    values = [
        ("acceleration_catalog_files", catalog["catalog_file"].nunique()),
        ("acceleration_catalog_rows", len(catalog)),
        ("acceleration_unique_events_2000_2025_01", catalog["event_id"].nunique()),
        ("acceleration_unique_events_2000_2024", raw_2000_2024["event_id"].nunique()),
        ("acceleration_tar_files", int(acceleration_yearly["tar_files"].sum())),
        ("acceleration_tar_bytes", int(acceleration_yearly["tar_size_bytes"].sum())),
        ("acceleration_outer_tar_valid", int(acceleration_yearly["outer_tar_valid"].sum())),
        ("acceleration_nested_tar_samples", len(tar_sample)),
        ("acceleration_nested_site_count_matches", int(tar_sample["site_count_matches"].sum())),
        ("processed_acceleration_events", len(processed_events)),
        ("processed_acceleration_station_rows", len(processed_stations)),
        ("processed_knet_rows", int(processed_stations["source_network"].astype(str).eq("knt").sum())),
        ("processed_kiknet_rows", int(processed_stations["source_network"].astype(str).eq("kik").sum())),
        ("velocity_archive_bytes", int(velocity_yearly["archive_size_bytes"].sum())),
        ("velocity_complete_years", int(velocity_yearly["archive_complete"].sum())),
        ("velocity_committed_events", int(velocity_yearly["committed_events"].sum())),
        ("velocity_downloaded_events", int(velocity_yearly["downloaded_events"].sum())),
        ("velocity_no_match_events", int(velocity_yearly["no_matched_station_events"].sum())),
        ("velocity_missing_events", len(velocity_missing)),
        ("velocity_requested_station_rows", int(velocity_yearly["requested_station_rows"].sum())),
        ("velocity_vertical_channel_missing_rows", vertical_gap_count),
        ("velocity_three_component_missing_rows", three_gap_count),
        ("velocity_sample_qc_rows", len(sample_qc)),
        ("velocity_sample_qc_pass", int(status.eq("PASS").sum())),
        ("velocity_sample_decoded", int(velocity_loaded.sum())),
        ("velocity_sample_checksum_pass", int(pd.to_numeric(sample_qc.get("archive_checksum_ok"), errors="coerce").fillna(0).sum())),
    ]
    return pd.DataFrame(values, columns=["metric", "value"])


def main() -> int:
    args = parse_args()
    args.acceleration_root = args.acceleration_root.expanduser().resolve()
    args.processed_root = args.processed_root.expanduser().resolve()
    args.velocity_root = args.velocity_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    source_dir = args.output_dir / "source_data"
    figure_dir = args.output_dir / "figures"
    source_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_acceleration_catalog(args.acceleration_root)
    tar_inventory, tar_sample = acceleration_tar_inventory(
        args.acceleration_root,
        catalog,
        validate_all=args.validate_all_tars,
        sample_events_per_year=args.sample_events_per_year,
    )
    processed_yearly, processed_events, processed_stations, event_ids_by_year = load_processed_inventory(
        args.processed_root,
        catalog,
    )
    velocity_yearly, velocity_missing, channel_gaps, sample_qc = audit_velocity(
        args.processed_root,
        args.velocity_root,
        event_ids_by_year,
    )
    acceleration_yearly = build_acceleration_yearly(catalog, tar_inventory, processed_yearly)
    representative_cases, representative_source, representative_summary = load_representative_cases(
        args.processed_root,
        args.velocity_root,
    )

    catalog.drop(columns=[], errors="ignore").to_csv(source_dir / "acceleration_catalog_combined.csv", index=False)
    tar_inventory.to_csv(source_dir / "acceleration_tar_outer_qc.csv", index=False)
    tar_sample.to_csv(source_dir / "acceleration_tar_nested_sample_qc.csv", index=False)
    acceleration_yearly.to_csv(source_dir / "acceleration_yearly_summary.csv", index=False)
    velocity_yearly.to_csv(source_dir / "velocity_yearly_summary.csv", index=False)
    velocity_missing.to_csv(source_dir / "velocity_missing_events.csv", index=False)
    channel_gaps.to_csv(source_dir / "velocity_channel_gaps.csv", index=False)
    sample_qc.to_csv(source_dir / "velocity_qc_samples.csv", index=False)
    representative_source.to_csv(source_dir / "representative_waveform_source.csv", index=False)
    representative_summary.to_csv(source_dir / "representative_case_summary.csv", index=False)

    summary = summary_table(
        catalog,
        acceleration_yearly,
        tar_sample,
        processed_events,
        processed_stations,
        velocity_yearly,
        velocity_missing,
        channel_gaps,
        sample_qc,
    )
    summary.to_csv(source_dir / "dataset_summary.csv", index=False)

    plot_coverage_overview(acceleration_yearly, velocity_yearly, velocity_missing, args.output_dir)
    plot_representative_waveforms(representative_cases, args.output_dir)

    provenance = {
        "acceleration_root": str(args.acceleration_root),
        "processed_root": str(args.processed_root),
        "velocity_root": str(args.velocity_root),
        "validate_all_tars": bool(args.validate_all_tars),
        "sample_events_per_year": int(args.sample_events_per_year),
        "figure_backend": "Python/matplotlib",
        "waveform_qc_window_seconds": [-120, 120],
        "representative_plot_window_seconds": [-20, 60],
    }
    (args.output_dir / "audit_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"[OK] Audit outputs written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
