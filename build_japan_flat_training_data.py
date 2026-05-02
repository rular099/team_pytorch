from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, detrend, resample_poly, sosfiltfilt
from tqdm import tqdm

from japan_dataset_builder import (
    JST,
    PRETRIGGER_SECONDS,
    _compute_pga_stats,
    _write_string_or_numeric_dataset,
    format_jst_iso,
    hypocentral_distance_km,
    parse_jst_timestamp,
    refine_event_p_picks_from_arrays,
    write_dataframe_group,
)


DIR_SENSOR_MAP = {
    0: ("knt", "single_surface", "0"),
    1: ("kik", "borehole", "1"),
    2: ("kik", "surface", "2"),
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build TEAM-style Japan training HDF5 + event/station CSV metadata "
            "from flat Japandata {year}.csv/{year}.h5 files."
        )
    )
    parser.add_argument("--japan_dir", required=True, help="Directory containing {year}.csv and {year}.h5.")
    parser.add_argument("--output_dir", required=True, help="Directory to write japan_{year}.hdf5 and CSV metadata.")
    parser.add_argument("--year", type=int, required=True, help="Year to process, e.g. 2024.")
    parser.add_argument("--min_stations", type=int, default=3, help="Minimum valid stations required to keep an event.")
    parser.add_argument("--target_sampling_rate", type=float, default=100.0, help="Target sampling rate in Hz.")
    parser.add_argument("--limit_events", type=int, default=None, help="Optional cap on number of events after filtering.")
    parser.add_argument("--sensor", choices=["surface", "borehole", "knet", "surface+knet", "all"], default="surface")
    parser.add_argument("--input_unit", choices=["gal", "mps2", "raw"], default="gal", help="Unit of HDF5 waveform values.")
    parser.add_argument("--p_velocity_km_s", type=float, default=6.0, help="P-wave velocity used for travel-time coarse picks.")
    parser.add_argument("--travel_time_intercept_s", type=float, default=0.0, help="Constant offset added to distance / velocity.")
    parser.add_argument("--pretrigger_seconds", type=float, default=PRETRIGGER_SECONDS, help="Seconds before recordtime in raw traces.")
    parser.add_argument("--compression_level", type=int, default=4, help="gzip compression level for output HDF5.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")

    parser.add_argument("--run_stalta", action="store_true", help="Compute short/long STA/LTA picks around travel-time picks.")
    parser.add_argument("--stalta_feature", choices=["vertical", "norm"], default="vertical")
    parser.add_argument("--stalta_search_pre", type=float, default=10.0)
    parser.add_argument("--stalta_search_post", type=float, default=10.0)
    parser.add_argument("--stalta_threshold", type=float, default=2.5)
    parser.add_argument("--stalta_short_sta", type=float, default=0.2)
    parser.add_argument("--stalta_short_lta", type=float, default=1.0)
    parser.add_argument("--stalta_long_sta", type=float, default=0.5)
    parser.add_argument("--stalta_long_lta", type=float, default=4.0)

    parser.add_argument("--run_diting", action="store_true", help="Run DiTing pickers around travel-time picks.")
    parser.add_argument("--ditingbench_root", default=None, help="Path to ditingbench root. Added to sys.path if set.")
    parser.add_argument("--diting_model_name", default="diting1200m")
    parser.add_argument("--diting_weights", default=None, help="Checkpoint passed to dtbench.models.get_model.")
    parser.add_argument("--diting_device", default="cuda:0")
    parser.add_argument("--diting_batch_size", type=int, default=100)
    parser.add_argument("--diting_p_th", type=float, default=0.1)
    parser.add_argument("--diting_s_th", type=float, default=0.1)
    parser.add_argument("--diting_d_th", type=float, default=0.3)
    parser.add_argument("--diting_window_seconds", type=float, default=100.0)
    parser.add_argument("--diting_target_half_window_seconds", type=float, default=10.0)
    parser.add_argument("--velocity_highpass_hz", type=float, default=0.05)

    parser.add_argument(
        "--final_pick",
        choices=["travel_time", "stalta_short", "stalta_long", "diting_acc", "diting_vel"],
        default="travel_time",
        help="Pick source stored in p_picks for training. Candidate picks are always preserved when computed.",
    )
    parser.add_argument("--diagnostics_dir", default=None, help="Directory for summary CSVs and plots. Defaults to output_dir/diagnostics_{year}.")
    parser.add_argument("--n_diagnostic_plots", type=int, default=24)
    return parser.parse_args()


def parse_absolute_time(time_str: str) -> float:
    dt = datetime.strptime(str(time_str), "%Y/%m/%d %H:%M:%S").replace(tzinfo=JST)
    return dt.timestamp()


def event_id_from_row(row: pd.Series) -> str:
    origin = str(row["origintime"]).replace("/", "").replace(":", "").replace(" ", "")
    lat = f"{float(row['lat']):.4f}"
    lon = f"{float(row['long']):.4f}"
    dep = f"{float(row['depth']):.2f}"
    mag = f"{float(row['mag']):.2f}"
    return f"{origin}_{lat}_{lon}_{dep}_{mag}"


def select_sensor_rows(df: pd.DataFrame, sensor: str) -> pd.DataFrame:
    if sensor == "surface":
        return df[df["dir"] == 2].copy()
    if sensor == "borehole":
        return df[df["dir"] == 1].copy()
    if sensor == "knet":
        return df[df["dir"] == 0].copy()
    if sensor == "surface+knet":
        return df[df["dir"].isin([0, 2])].copy()
    return df.copy()


def convert_waveform_unit(waveform: np.ndarray, input_unit: str) -> np.ndarray:
    data = np.asarray(waveform, dtype=np.float64)
    if input_unit == "gal":
        data = data * 0.01
    return data


def resample_waveform(waveform: np.ndarray, raw_fs: float, target_fs: float) -> np.ndarray:
    if math.isclose(raw_fs, target_fs):
        return waveform.copy()
    ratio = target_fs / raw_fs
    up = int(round(target_fs * 1000))
    down = int(round(raw_fs * 1000))
    gcd = math.gcd(up, down)
    up //= gcd
    down //= gcd
    resampled = resample_poly(waveform, up=up, down=down, axis=0)
    expected_len = int(round(waveform.shape[0] * ratio))
    if resampled.shape[0] > expected_len:
        resampled = resampled[:expected_len]
    elif resampled.shape[0] < expected_len:
        pad = np.zeros((expected_len - resampled.shape[0], waveform.shape[1]), dtype=resampled.dtype)
        resampled = np.concatenate([resampled, pad], axis=0)
    return resampled


def source_info_from_dir(dir_value: Any) -> tuple[str, str, str]:
    try:
        key = int(dir_value)
    except Exception:
        key = -1
    return DIR_SENSOR_MAP.get(key, ("unknown", "unknown", str(dir_value)))


def load_csv(csv_path: Path, sensor: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = select_sensor_rows(df, sensor=sensor)
    df = df.copy()
    df["EVENT"] = df.apply(event_id_from_row, axis=1)
    df["samplingfreq(hz)"] = pd.to_numeric(df["samplingfreq(hz)"], errors="coerce")
    df["durationtimes"] = pd.to_numeric(df["durationtimes"], errors="coerce")
    return df


def build_event_from_group(
    event_name: str,
    group: pd.DataFrame,
    japan_h5: h5py.File,
    target_sampling_rate_hz: float,
    input_unit: str,
    pretrigger_seconds: float,
    p_velocity_km_s: float,
    travel_time_intercept_s: float,
) -> tuple[dict[str, np.ndarray], dict[str, object], pd.DataFrame] | None:
    row0 = group.iloc[0]
    _, origin_ts = parse_jst_timestamp(str(row0["origintime"]))

    station_entries: list[dict[str, Any]] = []
    for _, row in group.iterrows():
        key = str(row["key"])
        if key not in japan_h5:
            continue

        raw = japan_h5[key][()]
        if raw.ndim != 2 or raw.shape[0] != 3:
            continue

        raw_fs = float(row["samplingfreq(hz)"])
        if not np.isfinite(raw_fs) or raw_fs <= 0:
            continue

        waveform_native = convert_waveform_unit(raw.T, input_unit=input_unit)
        waveform_native = waveform_native - np.nanmean(waveform_native, axis=0, keepdims=True)
        waveform_native = np.nan_to_num(waveform_native, copy=False)
        waveform_resampled = resample_waveform(waveform_native, raw_fs=raw_fs, target_fs=target_sampling_rate_hz)
        waveform_resampled = waveform_resampled - np.mean(waveform_resampled, axis=0, keepdims=True)

        record_time_ts = parse_absolute_time(str(row["recordtime"]))
        record_start_ts = record_time_ts - float(pretrigger_seconds)
        duration_s = float(row["durationtimes"]) if np.isfinite(row["durationtimes"]) else waveform_native.shape[0] / raw_fs

        distance_km = hypocentral_distance_km(
            event_lat=float(row["lat"]),
            event_lon=float(row["long"]),
            event_depth_km=float(row["depth"]),
            station_lat=float(row["stationlat"]),
            station_lon=float(row["stationlong"]),
            station_height_m=float(row["stationheight(m)"]),
        )
        predicted_tp_s = float(travel_time_intercept_s) + distance_km / float(p_velocity_km_s)
        travel_pick_ts = origin_ts + predicted_tp_s

        p_pick_trigger_raw = int(round((record_time_ts - record_start_ts) * raw_fs))
        p_pick_travel_raw_unclamped = int(round((travel_pick_ts - record_start_ts) * raw_fs))
        p_pick_travel_raw = int(np.clip(p_pick_travel_raw_unclamped, 0, max(0, waveform_native.shape[0] - 1)))
        p_pick_travel_resampled_unclamped = int(round((travel_pick_ts - record_start_ts) * target_sampling_rate_hz))
        p_pick_travel_resampled = int(np.clip(p_pick_travel_resampled_unclamped, 0, max(0, waveform_resampled.shape[0] - 1)))

        pga_vector_native, pga_norm_native, pga_norm_native_loc = _compute_pga_stats(waveform_native)
        pga_vector_resampled, pga_norm_resampled, pga_norm_resampled_loc = _compute_pga_stats(waveform_resampled)
        source_network, sensor_class, sensor_suffix = source_info_from_dir(row["dir"])

        station_entries.append(
            {
                "row": row,
                "key": key,
                "waveform_native": waveform_native,
                "waveform_resampled": waveform_resampled,
                "record_time_ts": record_time_ts,
                "record_start_ts": record_start_ts,
                "duration_s": duration_s,
                "raw_fs": raw_fs,
                "distance_km": distance_km,
                "predicted_tp_s": predicted_tp_s,
                "travel_pick_ts": travel_pick_ts,
                "p_pick_trigger_raw": p_pick_trigger_raw,
                "p_pick_travel_raw": p_pick_travel_raw,
                "p_pick_travel_raw_unclamped": p_pick_travel_raw_unclamped,
                "p_pick_travel_resampled": p_pick_travel_resampled,
                "p_pick_travel_resampled_unclamped": p_pick_travel_resampled_unclamped,
                "travel_pick_clipped": int(p_pick_travel_raw != p_pick_travel_raw_unclamped),
                "source_network": source_network,
                "sensor_class": sensor_class,
                "sensor_suffix": sensor_suffix,
                "pga_vector_native": pga_vector_native,
                "pga_norm_native": pga_norm_native,
                "pga_norm_native_loc": pga_norm_native_loc,
                "pga_vector_resampled": pga_vector_resampled,
                "pga_norm_resampled": pga_norm_resampled,
                "pga_norm_resampled_loc": pga_norm_resampled_loc,
            }
        )

    if not station_entries:
        return None

    global_start = min(st["record_start_ts"] for st in station_entries)
    offsets = [int(round((st["record_start_ts"] - global_start) * target_sampling_rate_hz)) for st in station_entries]
    event_length = max(offset + st["waveform_resampled"].shape[0] for st, offset in zip(station_entries, offsets))
    n_stations = len(station_entries)

    aligned = np.zeros((n_stations, event_length, 3), dtype=np.float64)
    coords = np.zeros((n_stations, 3), dtype=np.float64)
    p_picks = np.zeros((n_stations,), dtype=np.int64)
    pga = np.zeros((n_stations,), dtype=np.float64)
    pga_native = np.zeros((n_stations,), dtype=np.float64)
    pga_vector_native = np.zeros((n_stations, 3), dtype=np.float64)
    pga_vector_resampled = np.zeros((n_stations, 3), dtype=np.float64)
    max_acc_header = np.full((n_stations, 3), np.nan, dtype=np.float64)
    record_start_sample = np.zeros((n_stations,), dtype=np.int64)
    valid_n_samples = np.zeros((n_stations,), dtype=np.int64)
    pga_norm_native_loc_raw = np.zeros((n_stations,), dtype=np.int64)
    pga_norm_resampled_loc = np.zeros((n_stations,), dtype=np.int64)
    pga_norm_aligned_loc = np.zeros((n_stations,), dtype=np.int64)
    trigger_sample_raw = np.zeros((n_stations,), dtype=np.int64)
    trigger_sample_resampled = np.zeros((n_stations,), dtype=np.int64)
    p_pick_travel_raw = np.zeros((n_stations,), dtype=np.int64)
    p_pick_travel_raw_unclamped = np.zeros((n_stations,), dtype=np.int64)
    p_pick_travel_resampled = np.zeros((n_stations,), dtype=np.int64)
    p_pick_travel_resampled_unclamped = np.zeros((n_stations,), dtype=np.int64)
    p_pick_travel_aligned = np.zeros((n_stations,), dtype=np.int64)
    p_pick_travel_aligned_unclamped = np.zeros((n_stations,), dtype=np.int64)
    p_pick_trigger_aligned = np.zeros((n_stations,), dtype=np.int64)
    hypocentral_distance = np.zeros((n_stations,), dtype=np.float64)
    travel_time_seconds = np.zeros((n_stations,), dtype=np.float64)
    travel_pick_clipped = np.zeros((n_stations,), dtype=np.int8)
    sampling_rate_raw_hz = np.zeros((n_stations,), dtype=np.float64)
    duration_header_s = np.zeros((n_stations,), dtype=np.float64)

    station_codes: list[str] = []
    source_networks: list[str] = []
    sensor_classes: list[str] = []
    sensor_suffixes: list[str] = []
    keys: list[str] = []
    record_time_raws: list[str] = []
    record_start_times: list[str] = []
    trigger_times: list[str] = []
    travel_pick_times: list[str] = []

    station_rows: list[dict[str, object]] = []
    for i, (st, offset) in enumerate(zip(station_entries, offsets)):
        row = st["row"]
        waveform = st["waveform_resampled"]
        n = waveform.shape[0]
        aligned[i, offset:offset + n, :] = waveform
        coords[i] = [float(row["stationlat"]), float(row["stationlong"]), float(row["stationheight(m)"]) / 1000.0]
        record_start_sample[i] = offset
        valid_n_samples[i] = n
        p_pick_travel_raw[i] = st["p_pick_travel_raw"]
        p_pick_travel_raw_unclamped[i] = st["p_pick_travel_raw_unclamped"]
        p_pick_travel_resampled[i] = st["p_pick_travel_resampled"]
        p_pick_travel_resampled_unclamped[i] = st["p_pick_travel_resampled_unclamped"]
        p_pick_travel_aligned[i] = offset + st["p_pick_travel_resampled"]
        p_pick_travel_aligned_unclamped[i] = offset + st["p_pick_travel_resampled_unclamped"]
        p_pick_trigger_aligned[i] = int(round((st["record_time_ts"] - global_start) * target_sampling_rate_hz))
        p_picks[i] = p_pick_travel_aligned[i]
        pga[i] = st["pga_norm_resampled"]
        pga_native[i] = st["pga_norm_native"]
        pga_vector_native[i] = st["pga_vector_native"]
        pga_vector_resampled[i] = st["pga_vector_resampled"]
        pga_norm_native_loc_raw[i] = st["pga_norm_native_loc"]
        pga_norm_resampled_loc[i] = st["pga_norm_resampled_loc"]
        pga_norm_aligned_loc[i] = offset + st["pga_norm_resampled_loc"]
        trigger_sample_raw[i] = st["p_pick_trigger_raw"]
        trigger_sample_resampled[i] = int(round(pretrigger_seconds * target_sampling_rate_hz))
        hypocentral_distance[i] = st["distance_km"]
        travel_time_seconds[i] = st["predicted_tp_s"]
        travel_pick_clipped[i] = st["travel_pick_clipped"]
        sampling_rate_raw_hz[i] = st["raw_fs"]
        duration_header_s[i] = st["duration_s"]

        station_code = str(row["stationcode"])
        station_codes.append(station_code)
        source_networks.append(st["source_network"])
        sensor_classes.append(st["sensor_class"])
        sensor_suffixes.append(st["sensor_suffix"])
        keys.append(st["key"])
        record_time_raws.append(str(row["recordtime"]))
        record_start_times.append(format_jst_iso(st["record_start_ts"]))
        trigger_times.append(format_jst_iso(st["record_time_ts"]))
        travel_pick_times.append(format_jst_iso(st["travel_pick_ts"]))

        station_rows.append(
            {
                "EVENT": event_name,
                "wave_idx": i,
                "Origin_Time(JST)": row["origintime"],
                "Latitude": float(row["lat"]),
                "Longitude": float(row["long"]),
                "DEPTH": float(row["depth"]),
                "Magnitude": float(row["mag"]),
                "station_code": station_code,
                "station_lat": float(row["stationlat"]),
                "station_lon": float(row["stationlong"]),
                "station_height_m": float(row["stationheight(m)"]),
                "source_network": st["source_network"],
                "sensor_class": st["sensor_class"],
                "sensor_suffix": st["sensor_suffix"],
                "raw_dir_ns": str(row["dir"]),
                "raw_dir_ew": str(row["dir"]),
                "raw_dir_ud": str(row["dir"]),
                "scale_factor_ns": str(row.get("scalefactor", "")),
                "scale_factor_ew": str(row.get("scalefactor", "")),
                "scale_factor_ud": str(row.get("scalefactor", "")),
                "sampling_rate_raw_hz": st["raw_fs"],
                "sampling_rate_hz": target_sampling_rate_hz,
                "duration_header_s": st["duration_s"],
                "raw_n_samples": st["waveform_native"].shape[0],
                "resampled_n_samples": n,
                "record_time_raw": row["recordtime"],
                "record_time_jst": format_jst_iso(st["record_time_ts"]),
                "record_start_time_jst": format_jst_iso(st["record_start_ts"]),
                "trigger_time_jst": format_jst_iso(st["record_time_ts"]),
                "travel_time_pick_jst": format_jst_iso(st["travel_pick_ts"]),
                "record_start_sample": offset,
                "valid_n_samples": n,
                "p_pick_raw": st["p_pick_travel_raw"],
                "p_pick_raw_unclamped": st["p_pick_travel_raw_unclamped"],
                "p_pick_resampled": st["p_pick_travel_resampled"],
                "p_pick_resampled_unclamped": st["p_pick_travel_resampled_unclamped"],
                "p_pick_aligned": int(p_pick_travel_aligned[i]),
                "p_pick_travel_time_aligned": int(p_pick_travel_aligned[i]),
                "p_pick_travel_time_aligned_unclamped": int(p_pick_travel_aligned_unclamped[i]),
                "p_pick_trigger_aligned": int(p_pick_trigger_aligned[i]),
                "p_pick_repaired_aligned": int(p_pick_travel_aligned[i]),
                "p_pick_refined_aligned": int(p_pick_travel_aligned[i]),
                "p_pick_repaired_source": "travel_time",
                "p_pick_refined_source": "travel_time",
                "p_pick_refine_method": "travel_time",
                "trigger_is_pick": 0,
                "travel_pick_clipped": int(st["travel_pick_clipped"]),
                "hypocentral_distance_km": st["distance_km"],
                "p_pick_predicted_seconds_after_origin": st["predicted_tp_s"],
                "p_pick_repaired_seconds_after_origin": st["predicted_tp_s"],
                "pga_norm_native_mps2": st["pga_norm_native"],
                "pga_norm_resampled_mps2": st["pga_norm_resampled"],
                "pga_norm_native_loc_raw": st["pga_norm_native_loc"],
                "pga_norm_resampled_loc": st["pga_norm_resampled_loc"],
                "pga_norm_aligned_loc": int(pga_norm_aligned_loc[i]),
                "pga_csv": row.get("PGA", np.nan),
                "pga_csv_loc": row.get("PGAloc", np.nan),
                "pga_vector_native_ns_mps2": st["pga_vector_native"][0],
                "pga_vector_native_ew_mps2": st["pga_vector_native"][1],
                "pga_vector_native_ud_mps2": st["pga_vector_native"][2],
                "pga_vector_resampled_ns_mps2": st["pga_vector_resampled"][0],
                "pga_vector_resampled_ew_mps2": st["pga_vector_resampled"][1],
                "pga_vector_resampled_ud_mps2": st["pga_vector_resampled"][2],
                "max_acc_header_ns_gal": np.nan,
                "max_acc_header_ew_gal": np.nan,
                "max_acc_header_ud_gal": np.nan,
                "archive_relpath": str(row["key"]),
                "inner_archive": str(row["key"]),
                "component_base": str(row["key"]),
                "flat_h5_key": str(row["key"]),
                "event_length_samples": int(event_length),
                "event_length_seconds": float(event_length / target_sampling_rate_hz),
                "source_mix": "",
            }
        )

    source_mix = ",".join(sorted(set(source_networks)))
    for row in station_rows:
        row["source_mix"] = source_mix

    event_meta = {
        "EVENT": event_name,
        "Origin_Time(JST)": row0["origintime"],
        "Latitude": float(row0["lat"]),
        "Longitude": float(row0["long"]),
        "DEPTH": float(row0["depth"]),
        "Magnitude": float(row0["mag"]),
        "N_Stations": n_stations,
        "Source_Mix": source_mix,
        "Archive_Path": str(row0["key"]),
        "Event_Record_Start(JST)": format_jst_iso(global_start),
        "Event_Record_End(JST)": format_jst_iso(global_start + event_length / target_sampling_rate_hz),
        "Event_Length_Samples": int(event_length),
        "Event_Length_Seconds": float(event_length / target_sampling_rate_hz),
        "Sampling_Rate_Hz": float(target_sampling_rate_hz),
        "P_Pick_Source": "travel_time",
        "P_Pick_Velocity_Km_S": float(p_velocity_km_s),
        "P_Pick_Travel_Time_Intercept_S": float(travel_time_intercept_s),
        "P_Pick_Clipped_Count": int(np.sum(travel_pick_clipped)),
    }

    datasets = {
        "waveforms": aligned,
        "coords": coords,
        "p_picks": p_picks,
        "pga": np.log10(pga + 1e-10),
        "pga_norm_native_mps2": pga_native,
        "pga_norm_resampled_mps2": pga,
        "pga_vector_native_mps2": pga_vector_native,
        "pga_vector_resampled_mps2": pga_vector_resampled,
        "max_acc_header_gal": max_acc_header,
        "record_start_sample": record_start_sample,
        "valid_n_samples": valid_n_samples,
        "pga_norm_native_loc_raw": pga_norm_native_loc_raw,
        "pga_norm_resampled_loc": pga_norm_resampled_loc,
        "pga_norm_aligned_loc": pga_norm_aligned_loc,
        "trigger_sample_raw": trigger_sample_raw,
        "trigger_sample_resampled": trigger_sample_resampled,
        "p_pick_trigger_aligned": p_pick_trigger_aligned,
        "p_pick_travel_time_raw": p_pick_travel_raw,
        "p_pick_travel_time_raw_unclamped": p_pick_travel_raw_unclamped,
        "p_pick_travel_time_resampled": p_pick_travel_resampled,
        "p_pick_travel_time_resampled_unclamped": p_pick_travel_resampled_unclamped,
        "p_pick_travel_time_aligned": p_pick_travel_aligned,
        "p_pick_travel_time_aligned_unclamped": p_pick_travel_aligned_unclamped,
        "p_pick_repaired_aligned": p_pick_travel_aligned,
        "p_pick_refined_aligned": p_pick_travel_aligned,
        "trigger_is_pick": np.zeros((n_stations,), dtype=np.int8),
        "travel_pick_clipped": travel_pick_clipped,
        "hypocentral_distance_km": hypocentral_distance,
        "travel_time_seconds": travel_time_seconds,
        "sampling_rate_raw_hz": sampling_rate_raw_hz,
        "duration_header_s": duration_header_s,
        "station_codes": np.array(station_codes, dtype=object),
        "source_network": np.array(source_networks, dtype=object),
        "sensor_class": np.array(sensor_classes, dtype=object),
        "raw_dir_ns": np.array([str(st["row"]["dir"]) for st in station_entries], dtype=object),
        "raw_dir_ew": np.array([str(st["row"]["dir"]) for st in station_entries], dtype=object),
        "raw_dir_ud": np.array([str(st["row"]["dir"]) for st in station_entries], dtype=object),
        "scale_factor_ns": np.array([str(st["row"].get("scalefactor", "")) for st in station_entries], dtype=object),
        "scale_factor_ew": np.array([str(st["row"].get("scalefactor", "")) for st in station_entries], dtype=object),
        "scale_factor_ud": np.array([str(st["row"].get("scalefactor", "")) for st in station_entries], dtype=object),
        "archive_relpath": np.array(keys, dtype=object),
        "inner_archive": np.array(keys, dtype=object),
        "component_base": np.array(keys, dtype=object),
        "sensor_suffix": np.array(sensor_suffixes, dtype=object),
        "record_time_raw": np.array(record_time_raws, dtype=object),
        "record_start_time_jst": np.array(record_start_times, dtype=object),
        "trigger_time_jst": np.array(trigger_times, dtype=object),
        "travel_time_pick_jst": np.array(travel_pick_times, dtype=object),
    }
    return datasets, event_meta, pd.DataFrame(station_rows)


def add_stalta_candidates(
    datasets: dict[str, np.ndarray],
    station_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    waveforms = np.asarray(datasets["waveforms"])
    record_start_sample = np.asarray(datasets["record_start_sample"])
    valid_n_samples = np.asarray(datasets["valid_n_samples"])

    short_df = refine_event_p_picks_from_arrays(
        event_df=station_df,
        waveforms=waveforms,
        record_start_sample=record_start_sample,
        valid_n_samples=valid_n_samples,
        coarse_pick_col="p_pick_travel_time_aligned",
        pre_seconds=args.stalta_search_pre,
        post_seconds=args.stalta_search_post,
        sta_seconds=args.stalta_short_sta,
        lta_seconds=args.stalta_short_lta,
        threshold_ratio=args.stalta_threshold,
        feature=args.stalta_feature,
    )
    long_df = refine_event_p_picks_from_arrays(
        event_df=station_df,
        waveforms=waveforms,
        record_start_sample=record_start_sample,
        valid_n_samples=valid_n_samples,
        coarse_pick_col="p_pick_travel_time_aligned",
        pre_seconds=args.stalta_search_pre,
        post_seconds=args.stalta_search_post,
        sta_seconds=args.stalta_long_sta,
        lta_seconds=args.stalta_long_lta,
        threshold_ratio=args.stalta_threshold,
        feature=args.stalta_feature,
    )

    out = station_df.copy()
    for prefix, cand in (("stalta_short", short_df), ("stalta_long", long_df)):
        out[f"p_pick_{prefix}_aligned"] = cand["stalta_refined_pick_aligned"].astype(np.int64)
        out[f"{prefix}_method"] = cand["stalta_method"].astype(str)
        out[f"{prefix}_ratio_peak"] = cand["stalta_ratio_peak"].astype(float)
        out[f"{prefix}_ratio_at_pick"] = cand["stalta_ratio_at_pick"].astype(float)
        out[f"{prefix}_search_left_aligned"] = cand["stalta_search_left_aligned"].astype(np.int64)
        out[f"{prefix}_search_right_aligned"] = cand["stalta_search_right_aligned"].astype(np.int64)

        datasets[f"p_pick_{prefix}_aligned"] = out[f"p_pick_{prefix}_aligned"].to_numpy(dtype=np.int64)
        datasets[f"{prefix}_method"] = out[f"{prefix}_method"].to_numpy(dtype=object)
        datasets[f"{prefix}_ratio_peak"] = out[f"{prefix}_ratio_peak"].to_numpy(dtype=np.float64)
        datasets[f"{prefix}_ratio_at_pick"] = out[f"{prefix}_ratio_at_pick"].to_numpy(dtype=np.float64)
        datasets[f"{prefix}_search_left_aligned"] = out[f"{prefix}_search_left_aligned"].to_numpy(dtype=np.int64)
        datasets[f"{prefix}_search_right_aligned"] = out[f"{prefix}_search_right_aligned"].to_numpy(dtype=np.int64)

    return datasets, out


def prepare_diting_input_window(
    waveform_mps2: np.ndarray,
    coarse_pick_raw: int,
    sampling_rate_hz: float,
    target_half_window_s: float,
    model_window_s: float,
    mode: str,
    velocity_highpass_hz: float,
) -> tuple[np.ndarray, int, int, int]:
    target_fs = 100.0
    if not math.isclose(sampling_rate_hz, target_fs):
        waveform = resample_waveform(waveform_mps2, raw_fs=sampling_rate_hz, target_fs=target_fs)
        coarse_pick = int(round(coarse_pick_raw * target_fs / sampling_rate_hz))
    else:
        waveform = waveform_mps2.copy()
        coarse_pick = int(coarse_pick_raw)

    half = int(round(target_half_window_s * target_fs))
    crop_start = max(0, coarse_pick - half)
    crop_end = min(waveform.shape[0], coarse_pick + half)
    crop = waveform[crop_start:crop_end].astype(np.float64, copy=False)
    crop = detrend(crop, axis=0, type="linear")
    crop = crop - np.mean(crop, axis=0, keepdims=True)

    if mode == "velocity":
        sos = butter(4, velocity_highpass_hz, btype="highpass", fs=target_fs, output="sos")
        filtered = sosfiltfilt(sos, crop, axis=0)
        crop = cumulative_trapezoid(filtered, dx=1.0 / target_fs, axis=0, initial=0.0)
        crop = detrend(crop, axis=0, type="linear")
    elif mode != "acceleration":
        raise ValueError(f"Unsupported DiTing mode: {mode}")

    model_n = int(round(model_window_s * target_fs))
    window = np.zeros((model_n, 3), dtype=np.float32)
    tail_start = max(0, model_n - crop.shape[0])
    window[tail_start:tail_start + crop.shape[0], :] = crop[:model_n - tail_start]
    return window, crop_start, crop_end, tail_start


def run_diting_single_station(
    model: Any,
    waveform_100s: np.ndarray,
    station_code: str,
    device: str,
    batch_size: int,
    p_th: float,
    s_th: float,
    d_th: float,
) -> tuple[float, float]:
    try:
        from obspy import Stream, Trace, UTCDateTime
        from dtbench.inference.dt_infer import diting_dpk_inference
    except Exception as exc:
        raise RuntimeError("DiTing inference requires obspy and ditingbench/dtbench imports") from exc

    stream = Stream()
    for comp, channel in enumerate(("BHN", "BHE", "BHZ")):
        tr = Trace(data=waveform_100s[:, comp].astype(np.float32, copy=False))
        tr.stats.network = "JP"
        tr.stats.station = str(station_code)
        tr.stats.location = ""
        tr.stats.channel = channel
        tr.stats.sampling_rate = 100.0
        tr.stats.starttime = UTCDateTime(0)
        stream.append(tr)

    infer_res = diting_dpk_inference(
        data=stream,
        data_name=str(station_code),
        data_type="stream",
        model=model,
        model_output_type="diting",
        device=device,
        batch_size=batch_size,
        preprocess="default",
        postprocess="nms",
        output_format="default",
        p_th=p_th,
        s_th=s_th,
        d_th=d_th,
        window_length=10000,
        step_size=2000,
        radius_A=300,
        radius_B=300,
        pair_score=0.5,
        joint_metric="p_union",
        nms_pos_reduce="earliest",
        nms_score_reduce="p_union",
        process_len=10,
    )

    best_idx = np.nan
    best_score = np.nan
    for value in infer_res.values():
        for pred in value.get("pred", []):
            if len(pred) < 2 or not pred[1]:
                continue
            p_pick = pred[1][0]
            idx = float(p_pick[0])
            score = float(p_pick[1]) if len(p_pick) > 1 else np.nan
            if np.isnan(best_idx) or idx < best_idx:
                best_idx = idx
                best_score = score
    return best_idx, best_score


def add_diting_candidates(
    datasets: dict[str, np.ndarray],
    station_df: pd.DataFrame,
    japan_h5: h5py.File,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    if args.ditingbench_root:
        sys.path.insert(0, str(Path(args.ditingbench_root).resolve()))

    if args.diting_weights is None:
        raise ValueError("--diting_weights is required when --run_diting is set")

    model = getattr(args, "_diting_model", None)
    if model is None:
        from dtbench.models import get_model

        model = get_model(name=args.diting_model_name, pretrain_weights=args.diting_weights, device=args.diting_device)
        setattr(args, "_diting_model", model)

    out = station_df.copy()
    for col in (
        "p_pick_diting_acc_aligned",
        "p_pick_diting_vel_aligned",
        "p_pick_diting_acc_score",
        "p_pick_diting_vel_score",
    ):
        out[col] = np.nan

    for idx, row in tqdm(out.iterrows(), total=len(out), desc="diting", unit="station"):
        key = str(row["flat_h5_key"])
        if key not in japan_h5:
            continue
        raw = japan_h5[key][()]
        raw_fs = float(row["sampling_rate_raw_hz"])
        waveform = convert_waveform_unit(raw.T, args.input_unit)
        waveform = waveform - np.mean(waveform, axis=0, keepdims=True)
        coarse_raw = int(row["p_pick_raw"])
        for mode, prefix in (("acceleration", "acc"), ("velocity", "vel")):
            try:
                window, crop_start, _, tail_start = prepare_diting_input_window(
                    waveform_mps2=waveform,
                    coarse_pick_raw=coarse_raw,
                    sampling_rate_hz=raw_fs,
                    target_half_window_s=args.diting_target_half_window_seconds,
                    model_window_s=args.diting_window_seconds,
                    mode=mode,
                    velocity_highpass_hz=args.velocity_highpass_hz,
                )
                pred_idx, score = run_diting_single_station(
                    model=model,
                    waveform_100s=window,
                    station_code=str(row["station_code"]),
                    device=args.diting_device,
                    batch_size=args.diting_batch_size,
                    p_th=args.diting_p_th,
                    s_th=args.diting_s_th,
                    d_th=args.diting_d_th,
                )
                if not np.isfinite(pred_idx):
                    continue
                raw_pick_100hz = crop_start + int(round(pred_idx - tail_start))
                aligned_pick = int(row["record_start_sample"]) + raw_pick_100hz
                out.loc[idx, f"p_pick_diting_{prefix}_aligned"] = aligned_pick
                out.loc[idx, f"p_pick_diting_{prefix}_score"] = score
            except Exception as exc:
                out.loc[idx, f"p_pick_diting_{prefix}_error"] = str(exc)

    for prefix in ("acc", "vel"):
        aligned = out[f"p_pick_diting_{prefix}_aligned"].fillna(out["p_pick_travel_time_aligned"]).to_numpy(dtype=np.int64)
        score = out[f"p_pick_diting_{prefix}_score"].to_numpy(dtype=np.float64)
        datasets[f"p_pick_diting_{prefix}_aligned"] = aligned
        datasets[f"diting_{prefix}_score"] = score
    return datasets, out


def apply_final_pick(
    datasets: dict[str, np.ndarray],
    station_df: pd.DataFrame,
    final_pick: str,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    col_map = {
        "travel_time": "p_pick_travel_time_aligned",
        "stalta_short": "p_pick_stalta_short_aligned",
        "stalta_long": "p_pick_stalta_long_aligned",
        "diting_acc": "p_pick_diting_acc_aligned",
        "diting_vel": "p_pick_diting_vel_aligned",
    }
    col = col_map[final_pick]
    if col not in station_df.columns:
        raise ValueError(f"Final pick {final_pick!r} requires missing column {col!r}")

    out = station_df.copy()
    final = pd.to_numeric(out[col], errors="coerce").fillna(out["p_pick_travel_time_aligned"]).astype(np.int64)
    out["p_pick_aligned"] = final
    out["p_pick_refined_aligned"] = final
    out["p_pick_refined_source"] = final_pick
    out["p_pick_refine_method"] = final_pick
    datasets["p_picks"] = final.to_numpy(dtype=np.int64)
    datasets["p_pick_refined_aligned"] = datasets["p_picks"]
    datasets["p_pick_refined_source"] = np.asarray([final_pick] * len(out), dtype=object)
    datasets["p_pick_refine_method"] = np.asarray([final_pick] * len(out), dtype=object)
    return datasets, out


def write_diagnostics(station_df: pd.DataFrame, hdf5_path: Path, diagnostics_dir: Path, n_plots: int) -> None:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    pick_cols = [
        col for col in (
            "p_pick_trigger_aligned",
            "p_pick_travel_time_aligned",
            "p_pick_stalta_short_aligned",
            "p_pick_stalta_long_aligned",
            "p_pick_diting_acc_aligned",
            "p_pick_diting_vel_aligned",
            "p_pick_refined_aligned",
        )
        if col in station_df.columns
    ]
    sr = pd.to_numeric(station_df["sampling_rate_hz"], errors="coerce").replace(0, np.nan)
    rows = []
    base = "p_pick_travel_time_aligned"
    for col in pick_cols:
        diff = (pd.to_numeric(station_df[col], errors="coerce") - station_df[base]) / sr
        rows.append(
            {
                "pick": col,
                "count": int(diff.notna().sum()),
                "mean_diff_s": float(diff.mean()),
                "std_diff_s": float(diff.std()),
                "median_diff_s": float(diff.median()),
                "p05_diff_s": float(diff.quantile(0.05)),
                "p95_diff_s": float(diff.quantile(0.95)),
                "min_diff_s": float(diff.min()),
                "max_diff_s": float(diff.max()),
            }
        )
        station_df[f"{col}_minus_travel_time_s"] = diff
    pd.DataFrame(rows).to_csv(diagnostics_dir / "pick_difference_summary.csv", index=False)
    station_df.to_csv(diagnostics_dir / "station_pick_differences.csv", index=False)

    try:
        os.environ.setdefault("MPLCONFIGDIR", str(diagnostics_dir / ".mplconfig"))
        (diagnostics_dir / ".mplconfig").mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt
    except Exception:
        return

    for col in pick_cols:
        if col == base:
            continue
        diff = station_df[f"{col}_minus_travel_time_s"].dropna()
        if diff.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(diff, bins=80)
        ax.set_title(f"{col} - travel_time")
        ax.set_xlabel("seconds")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(diagnostics_dir / f"{col}_minus_travel_time_hist.png", dpi=160)
        plt.close(fig)

    sample_df = station_df.dropna(subset=["Magnitude"]).copy()
    if sample_df.empty or n_plots <= 0:
        return
    sample_df["mag_bin"] = pd.qcut(sample_df["Magnitude"], q=min(4, sample_df["Magnitude"].nunique()), duplicates="drop")
    sampled_parts = []
    n_bins = max(1, sample_df["mag_bin"].nunique())
    per_bin = max(1, n_plots // n_bins)
    for _, part in sample_df.groupby("mag_bin", observed=True):
        sampled_parts.append(part.sample(min(len(part), per_bin), random_state=2024))
    sampled = pd.concat(sampled_parts, ignore_index=True).head(n_plots) if sampled_parts else sample_df.head(0)
    sampled.to_csv(diagnostics_dir / "diagnostic_plot_samples.csv", index=False)

    colors = {
        "p_pick_trigger_aligned": "tab:gray",
        "p_pick_travel_time_aligned": "tab:blue",
        "p_pick_stalta_short_aligned": "tab:orange",
        "p_pick_stalta_long_aligned": "tab:green",
        "p_pick_diting_acc_aligned": "tab:red",
        "p_pick_diting_vel_aligned": "tab:purple",
        "p_pick_refined_aligned": "black",
    }
    with h5py.File(hdf5_path, "r") as f:
        for plot_idx, row in enumerate(sampled.itertuples(index=False), start=1):
            event = str(getattr(row, "EVENT"))
            wave_idx = int(getattr(row, "wave_idx"))
            if event not in f["data"]:
                continue
            grp = f["data"][event]
            waveform = grp["waveforms"][wave_idx]
            sr_hz = float(getattr(row, "sampling_rate_hz"))
            refined = int(getattr(row, "p_pick_refined_aligned"))
            left = max(0, refined - int(round(15.0 * sr_hz)))
            right = min(waveform.shape[0], refined + int(round(25.0 * sr_hz)))
            t = (np.arange(left, right) - refined) / sr_hz
            vertical = waveform[left:right, 2]
            norm = np.linalg.norm(waveform[left:right], axis=1)

            fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
            axes[0].plot(t, vertical, color="0.2", linewidth=0.8)
            axes[0].set_ylabel("UD m/s^2")
            axes[1].plot(t, norm, color="0.2", linewidth=0.8)
            axes[1].set_ylabel("norm m/s^2")
            axes[1].set_xlabel("seconds relative to final pick")
            for col, color in colors.items():
                if col not in sampled.columns or not hasattr(row, col):
                    continue
                pick = getattr(row, col)
                if not np.isfinite(pick):
                    continue
                x = (float(pick) - refined) / sr_hz
                for ax in axes:
                    ax.axvline(x, color=color, linewidth=1.0, alpha=0.85, label=col.replace("p_pick_", "").replace("_aligned", ""))
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                by_label = dict(zip(labels, handles))
                axes[0].legend(by_label.values(), by_label.keys(), fontsize=7, ncol=3)
            title = f"{event} wave_idx={wave_idx} M={float(getattr(row, 'Magnitude')):.1f} {str(getattr(row, 'station_code'))}"
            axes[0].set_title(title)
            fig.tight_layout()
            fig.savefig(diagnostics_dir / f"waveform_pick_check_{plot_idx:03d}.png", dpi=160)
            plt.close(fig)


def write_event_group(
    data_grp: h5py.Group,
    event_name: str,
    datasets: dict[str, np.ndarray],
    compression_level: int,
) -> None:
    compression = "gzip"
    event_grp = data_grp.create_group(event_name)
    for key, values in datasets.items():
        if key == "waveforms":
            chunks = (1, min(values.shape[1], 4096), values.shape[2])
            event_grp.create_dataset(
                key,
                data=values,
                compression=compression,
                compression_opts=compression_level,
                chunks=chunks,
            )
        elif isinstance(values, np.ndarray) and values.dtype.kind not in {"S", "O", "U"} and values.ndim > 0:
            event_grp.create_dataset(
                key,
                data=values,
                compression=compression,
                compression_opts=compression_level,
            )
        else:
            _write_string_or_numeric_dataset(
                event_grp,
                key,
                values,
                compression=compression,
                compression_opts=compression_level,
            )


def main() -> None:
    args = parse_args()
    japan_dir = Path(args.japan_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = japan_dir / f"{args.year}.csv"
    h5_path = japan_dir / f"{args.year}.h5"
    output_hdf5 = output_dir / f"japan_{args.year}.hdf5"
    output_events_csv = output_dir / f"japan_{args.year}_events.csv"
    output_stations_csv = output_dir / f"japan_{args.year}_stations.csv"

    for path in (output_hdf5, output_events_csv, output_stations_csv):
        if path.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing file without --overwrite: {path}")
        if path.exists() and args.overwrite:
            path.unlink()

    df = load_csv(csv_path, sensor=args.sensor)
    grouped = list(df.groupby("EVENT", sort=True))
    if args.limit_events is not None:
        grouped = grouped[: args.limit_events]

    stats = Counter()
    event_rows: list[dict[str, object]] = []
    station_rows: list[pd.DataFrame] = []

    with h5py.File(h5_path, "r") as japan_h5, h5py.File(output_hdf5, "w") as out_h5:
        meta_grp = out_h5.create_group("metadata")
        data_grp = out_h5.create_group("data")
        meta_grp.create_dataset("sampling_rate", data=float(args.target_sampling_rate))
        meta_grp.create_dataset("pretrigger_seconds", data=float(args.pretrigger_seconds))
        meta_grp.create_dataset("year", data=int(args.year))
        _write_string_or_numeric_dataset(meta_grp, "alignment_mode", np.array(["earliest_record_start"], dtype=object))
        _write_string_or_numeric_dataset(meta_grp, "waveform_root", np.array([str(japan_dir)], dtype=object))
        _write_string_or_numeric_dataset(meta_grp, "source_format", np.array(["flat_csv_h5"], dtype=object))
        _write_string_or_numeric_dataset(meta_grp, "final_pick", np.array([args.final_pick], dtype=object))
        meta_grp.create_dataset("p_velocity_km_s", data=float(args.p_velocity_km_s))
        meta_grp.create_dataset("travel_time_intercept_s", data=float(args.travel_time_intercept_s))

        for event_name, group in tqdm(grouped, desc=f"year {args.year}", unit="event"):
            stats["events_seen"] += 1
            if len(group) < args.min_stations:
                stats["events_below_min_stations"] += 1
                stats["stations_dropped_below_min"] += len(group)
                continue

            built = build_event_from_group(
                event_name=event_name,
                group=group,
                japan_h5=japan_h5,
                target_sampling_rate_hz=args.target_sampling_rate,
                input_unit=args.input_unit,
                pretrigger_seconds=args.pretrigger_seconds,
                p_velocity_km_s=args.p_velocity_km_s,
                travel_time_intercept_s=args.travel_time_intercept_s,
            )
            if built is None:
                stats["events_empty"] += 1
                continue
            datasets, event_meta, station_df = built
            if len(station_df) < args.min_stations:
                stats["events_below_min_stations_after_h5"] += 1
                stats["stations_dropped_below_min_after_h5"] += len(station_df)
                continue

            if args.run_stalta:
                datasets, station_df = add_stalta_candidates(datasets, station_df, args)
                event_meta.update(
                    {
                        "P_Pick_STA_Feature": args.stalta_feature,
                        "P_Pick_STA_Search_Pre_S": args.stalta_search_pre,
                        "P_Pick_STA_Search_Post_S": args.stalta_search_post,
                        "P_Pick_STA_Threshold": args.stalta_threshold,
                        "P_Pick_STA_Short_STA_S": args.stalta_short_sta,
                        "P_Pick_STA_Short_LTA_S": args.stalta_short_lta,
                        "P_Pick_STA_Long_STA_S": args.stalta_long_sta,
                        "P_Pick_STA_Long_LTA_S": args.stalta_long_lta,
                    }
                )

            if args.run_diting:
                datasets, station_df = add_diting_candidates(datasets, station_df, japan_h5, args)
                event_meta["P_Pick_DiTing_Model"] = args.diting_model_name

            datasets, station_df = apply_final_pick(datasets, station_df, args.final_pick)
            event_meta["P_Pick_Final_Source"] = args.final_pick

            write_event_group(data_grp, event_name, datasets, compression_level=args.compression_level)
            event_rows.append(event_meta)
            station_rows.append(station_df)
            stats["events_written"] += 1
            stats["stations_written"] += len(station_df)

        event_df = pd.DataFrame(event_rows)
        station_df_all = pd.concat(station_rows, ignore_index=True) if station_rows else pd.DataFrame()
        write_dataframe_group(
            meta_grp.create_group("event_metadata"),
            event_df,
            compression="gzip",
            compression_opts=args.compression_level,
        )
        write_dataframe_group(
            meta_grp.create_group("station_metadata"),
            station_df_all,
            compression="gzip",
            compression_opts=args.compression_level,
        )

    event_df.to_csv(output_events_csv, index=False)
    station_df_all.to_csv(output_stations_csv, index=False)

    diagnostics_dir = Path(args.diagnostics_dir).expanduser().resolve() if args.diagnostics_dir else output_dir / f"diagnostics_{args.year}"
    if not station_df_all.empty:
        write_diagnostics(station_df_all, hdf5_path=output_hdf5, diagnostics_dir=diagnostics_dir, n_plots=args.n_diagnostic_plots)

    print(f"\nYear {args.year} complete")
    print(f"  HDF5: {output_hdf5}")
    print(f"  Events CSV: {output_events_csv}")
    print(f"  Stations CSV: {output_stations_csv}")
    print(f"  Diagnostics: {diagnostics_dir}")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")


if __name__ == "__main__":
    main()
