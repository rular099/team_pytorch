from __future__ import annotations

import gzip
import io
import math
import os
import re
import sys
import tarfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO, Iterable

import h5py
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, detrend, resample_poly, sosfiltfilt
from tqdm import tqdm


JST = timezone(timedelta(hours=9))
PRETRIGGER_SECONDS = 15.0
COMPONENT_ORDER = ("NS", "EW", "UD")
DIR_COMPONENT_INDEX = {name: i for i, name in enumerate(COMPONENT_ORDER)}
OUTER_EVENT_RE = re.compile(r"(?P<event_id>\d{14})\.tar$")
INNER_COMPONENT_RE = re.compile(r"^(?P<base>.+)\.(?P<comp>NS|EW|UD)(?P<suffix>[12]?)$")
DEFAULT_JMA2001A_ZIP = Path(__file__).resolve().parent / "resources" / "jma_travel_times" / "tjma2001h.zip"


@dataclass
class ComponentTrace:
    component: str
    sensor_suffix: str
    member_name: str
    metadata_raw: dict[str, str]
    data_mps2: np.ndarray
    raw_dir: str
    scale_factor_raw: str
    max_acc_header_gal: float


@dataclass
class StationTrace:
    event_id: str
    archive_relpath: str
    inner_archive_name: str
    component_base: str
    source_network: str
    sensor_suffix: str
    sensor_class: str
    station_code: str
    station_lat: float
    station_lon: float
    station_height_m: float
    event_lat: float
    event_lon: float
    event_depth_km: float
    magnitude: float
    origin_time_raw: str
    record_time_raw: str
    last_correction_raw: str
    record_time_ts: float
    record_start_ts: float
    trigger_ts: float
    duration_header_s: float
    sampling_rate_raw_hz: float
    sampling_rate_target_hz: float
    raw_dirs: tuple[str, str, str]
    scale_factors_raw: tuple[str, str, str]
    max_acc_header_gal: np.ndarray
    waveform_native: np.ndarray
    waveform_resampled: np.ndarray
    p_pick_raw: int
    p_pick_resampled: int
    pga_vector_native_mps2: np.ndarray
    pga_norm_native_mps2: float
    pga_norm_native_loc: int
    pga_vector_resampled_mps2: np.ndarray
    pga_norm_resampled_mps2: float
    pga_norm_resampled_loc: int


def parse_jst_timestamp(time_str: str) -> tuple[datetime, float]:
    dt = datetime.strptime(time_str, "%Y/%m/%d %H:%M:%S").replace(tzinfo=JST)
    return dt, dt.timestamp()


def format_jst_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=JST).isoformat()


def try_float(val: str) -> float:
    return float(val.strip())


def _parse_scale_factor(scale_factor: str) -> tuple[float, float]:
    clean = scale_factor.replace("(gal)", "")
    match = re.match(r"\s*([+-]?\d+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)\s*$", clean)
    if not match:
        raise ValueError(f"Unsupported scale factor: {scale_factor}")
    gal_per_count = float(match.group(1)) / float(match.group(2))
    return gal_per_count, gal_per_count * 0.01


def _parse_header_and_data(raw_bytes: bytes) -> tuple[dict[str, str], np.ndarray]:
    text = raw_bytes.decode("utf-8", errors="ignore").splitlines()
    metadata: dict[str, str] = {}
    data_started = False
    data_chunks: list[np.ndarray] = []

    for line in text:
        stripped = line.strip()
        if not data_started:
            if not stripped:
                continue
            if stripped.startswith("Memo."):
                data_started = True
                continue
            key = line[:18].strip()
            value = line[18:].strip()
            if not key or not value:
                continue
            metadata[key] = value
        else:
            if not stripped:
                continue
            arr = np.fromstring(stripped, sep=" ", dtype=np.float64)
            if arr.size:
                data_chunks.append(arr)

    if not data_chunks:
        raise ValueError("No waveform samples found after header")

    data = np.concatenate(data_chunks)
    return metadata, data


def parse_component_fileobj(fileobj: BinaryIO, member_name: str) -> ComponentTrace:
    raw_bytes = fileobj.read()
    metadata_raw, raw_counts = _parse_header_and_data(raw_bytes)
    filename = Path(member_name).name
    match = INNER_COMPONENT_RE.match(filename)
    if not match:
        raise ValueError(f"Unsupported component filename: {member_name}")

    scale_factor_raw = metadata_raw["Scale Factor"]
    _, mps2_per_count = _parse_scale_factor(scale_factor_raw)
    data_mps2 = raw_counts * mps2_per_count

    return ComponentTrace(
        component=match.group("comp"),
        sensor_suffix=match.group("suffix"),
        member_name=filename,
        metadata_raw=metadata_raw,
        data_mps2=data_mps2.astype(np.float64, copy=False),
        raw_dir=metadata_raw.get("Dir.", ""),
        scale_factor_raw=scale_factor_raw,
        max_acc_header_gal=try_float(metadata_raw.get("Max. Acc. (gal)", "nan")),
    )


def _resample_waveform(waveform: np.ndarray, raw_fs: float, target_fs: float) -> np.ndarray:
    if math.isclose(raw_fs, target_fs):
        return waveform.copy()

    ratio = Fraction(str(target_fs / raw_fs)).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    resampled = resample_poly(waveform, up=up, down=down, axis=0)
    expected_len = int(round(waveform.shape[0] * target_fs / raw_fs))
    if resampled.shape[0] > expected_len:
        resampled = resampled[:expected_len]
    elif resampled.shape[0] < expected_len:
        pad = np.zeros((expected_len - resampled.shape[0], waveform.shape[1]), dtype=resampled.dtype)
        resampled = np.concatenate([resampled, pad], axis=0)
    return resampled


def _compute_pga_stats(waveform: np.ndarray) -> tuple[np.ndarray, float, int]:
    per_component = np.max(np.abs(waveform), axis=0)
    norm_series = np.linalg.norm(waveform, axis=1)
    norm_loc = int(np.argmax(norm_series))
    norm_val = float(norm_series[norm_loc])
    return per_component.astype(np.float64), norm_val, norm_loc


def _sensor_class(source_network: str, sensor_suffix: str) -> str:
    if source_network == "knt":
        return "single_surface"
    if sensor_suffix == "1":
        return "borehole"
    if sensor_suffix == "2":
        return "surface"
    return "unknown"


def build_station_trace(
    event_id: str,
    archive_relpath: str,
    inner_archive_name: str,
    component_base: str,
    source_network: str,
    traces: dict[str, ComponentTrace],
    target_sampling_rate_hz: float,
) -> StationTrace:
    missing = [comp for comp in COMPONENT_ORDER if comp not in traces]
    if missing:
        raise ValueError(f"Incomplete station group {component_base}: missing {missing}")

    ref = traces["UD"].metadata_raw
    sensor_suffix = traces["UD"].sensor_suffix

    for comp in COMPONENT_ORDER:
        meta = traces[comp].metadata_raw
        for key in (
            "Origin Time",
            "Lat.",
            "Long.",
            "Depth. (km)",
            "Mag.",
            "Station Code",
            "Station Lat.",
            "Station Long.",
            "Station Height(m)",
            "Record Time",
            "Sampling Freq(Hz)",
            "Duration Time(s)",
            "Last Correction",
        ):
            if meta.get(key) != ref.get(key):
                raise ValueError(f"Metadata mismatch for {component_base}: key={key}")

    lengths = [traces[comp].data_mps2.shape[0] for comp in COMPONENT_ORDER]
    if len(set(lengths)) != 1:
        raise ValueError(f"Waveform length mismatch for {component_base}: {lengths}")

    raw_fs = try_float(ref["Sampling Freq(Hz)"].replace("Hz", ""))
    duration_header_s = try_float(ref["Duration Time(s)"])
    _, record_time_ts = parse_jst_timestamp(ref["Record Time"])
    record_start_ts = record_time_ts - PRETRIGGER_SECONDS
    trigger_ts = record_time_ts

    waveform_native = np.stack(
        [traces[comp].data_mps2 for comp in COMPONENT_ORDER],
        axis=1,
    )
    waveform_native = waveform_native - np.mean(waveform_native, axis=0, keepdims=True)

    pga_vector_native, pga_norm_native, pga_norm_native_loc = _compute_pga_stats(waveform_native)

    waveform_resampled = _resample_waveform(waveform_native, raw_fs=raw_fs, target_fs=target_sampling_rate_hz)
    waveform_resampled = waveform_resampled - np.mean(waveform_resampled, axis=0, keepdims=True)
    pga_vector_resampled, pga_norm_resampled, pga_norm_resampled_loc = _compute_pga_stats(waveform_resampled)

    p_pick_raw = int(round(PRETRIGGER_SECONDS * raw_fs))
    p_pick_resampled = int(round(PRETRIGGER_SECONDS * target_sampling_rate_hz))

    return StationTrace(
        event_id=event_id,
        archive_relpath=archive_relpath,
        inner_archive_name=inner_archive_name,
        component_base=component_base,
        source_network=source_network,
        sensor_suffix=sensor_suffix,
        sensor_class=_sensor_class(source_network, sensor_suffix),
        station_code=ref["Station Code"],
        station_lat=try_float(ref["Station Lat."]),
        station_lon=try_float(ref["Station Long."]),
        station_height_m=try_float(ref["Station Height(m)"]),
        event_lat=try_float(ref["Lat."]),
        event_lon=try_float(ref["Long."]),
        event_depth_km=try_float(ref["Depth. (km)"]),
        magnitude=try_float(ref["Mag."]),
        origin_time_raw=ref["Origin Time"],
        record_time_raw=ref["Record Time"],
        last_correction_raw=ref["Last Correction"],
        record_time_ts=record_time_ts,
        record_start_ts=record_start_ts,
        trigger_ts=trigger_ts,
        duration_header_s=duration_header_s,
        sampling_rate_raw_hz=raw_fs,
        sampling_rate_target_hz=target_sampling_rate_hz,
        raw_dirs=tuple(traces[comp].raw_dir for comp in COMPONENT_ORDER),
        scale_factors_raw=tuple(traces[comp].scale_factor_raw for comp in COMPONENT_ORDER),
        max_acc_header_gal=np.array(
            [traces[comp].max_acc_header_gal for comp in COMPONENT_ORDER],
            dtype=np.float64,
        ),
        waveform_native=waveform_native,
        waveform_resampled=waveform_resampled,
        p_pick_raw=p_pick_raw,
        p_pick_resampled=p_pick_resampled,
        pga_vector_native_mps2=pga_vector_native,
        pga_norm_native_mps2=pga_norm_native,
        pga_norm_native_loc=pga_norm_native_loc,
        pga_vector_resampled_mps2=pga_vector_resampled,
        pga_norm_resampled_mps2=pga_norm_resampled,
        pga_norm_resampled_loc=pga_norm_resampled_loc,
    )


def _load_inner_archive_members(outer_tar_path: Path, inner_member_name: str) -> list[tarfile.TarInfo]:
    with tarfile.open(outer_tar_path, "r") as outer_tar:
        member = outer_tar.getmember(inner_member_name)
        payload = outer_tar.extractfile(member).read()
    inner_bytes = io.BytesIO(gzip.decompress(payload))
    with tarfile.open(fileobj=inner_bytes, mode="r:") as inner_tar:
        return inner_tar.getmembers()


def _open_inner_archive(outer_tar_path: Path, inner_member_name: str) -> tarfile.TarFile:
    with tarfile.open(outer_tar_path, "r") as outer_tar:
        member = outer_tar.getmember(inner_member_name)
        payload = outer_tar.extractfile(member).read()
    inner_bytes = io.BytesIO(gzip.decompress(payload))
    return tarfile.open(fileobj=inner_bytes, mode="r:")


def load_station_traces_from_event_archive(
    outer_tar_path: Path,
    archive_relpath: str,
    target_sampling_rate_hz: float,
) -> list[StationTrace]:
    outer_match = OUTER_EVENT_RE.match(outer_tar_path.name)
    if not outer_match:
        raise ValueError(f"Unexpected outer archive name: {outer_tar_path}")
    event_id = outer_match.group("event_id")

    station_traces: list[StationTrace] = []
    with tarfile.open(outer_tar_path, "r") as outer_tar:
        inner_members = [m for m in outer_tar.getmembers() if m.isfile() and m.name.endswith((".knt.tar.gz", ".kik.tar.gz"))]
        for inner_member in inner_members:
            source_network = "knt" if inner_member.name.endswith(".knt.tar.gz") else "kik"
            payload = outer_tar.extractfile(inner_member).read()
            with tarfile.open(fileobj=io.BytesIO(gzip.decompress(payload)), mode="r:") as inner_tar:
                grouped: dict[str, dict[str, tarfile.TarInfo]] = defaultdict(dict)
                for member in inner_tar.getmembers():
                    if not member.isfile():
                        continue
                    filename = Path(member.name).name
                    match = INNER_COMPONENT_RE.match(filename)
                    if not match:
                        continue
                    group_key = f"{match.group('base')}__{match.group('suffix') or '0'}"
                    grouped[group_key][match.group("comp")] = member

                for group_key, component_members in grouped.items():
                    if any(comp not in component_members for comp in COMPONENT_ORDER):
                        continue
                    traces: dict[str, ComponentTrace] = {}
                    component_base = group_key.split("__", 1)[0]
                    for comp in COMPONENT_ORDER:
                        with inner_tar.extractfile(component_members[comp]) as fileobj:
                            traces[comp] = parse_component_fileobj(fileobj, component_members[comp].name)
                    station_traces.append(
                        build_station_trace(
                            event_id=event_id,
                            archive_relpath=archive_relpath,
                            inner_archive_name=inner_member.name,
                            component_base=component_base,
                            source_network=source_network,
                            traces=traces,
                            target_sampling_rate_hz=target_sampling_rate_hz,
                        )
                    )
    return station_traces


def build_aligned_event(stations: list[StationTrace]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    global_start = min(st.record_start_ts for st in stations)
    offsets = [int(round((st.record_start_ts - global_start) * st.sampling_rate_target_hz)) for st in stations]
    event_length = max(offset + st.waveform_resampled.shape[0] for st, offset in zip(stations, offsets))
    n_stations = len(stations)

    aligned = np.zeros((n_stations, event_length, 3), dtype=np.float64)
    coords = np.zeros((n_stations, 3), dtype=np.float64)
    p_picks = np.zeros((n_stations,), dtype=np.int64)
    pga = np.zeros((n_stations,), dtype=np.float64)
    pga_native = np.zeros((n_stations,), dtype=np.float64)
    pga_vector_native = np.zeros((n_stations, 3), dtype=np.float64)
    pga_vector_resampled = np.zeros((n_stations, 3), dtype=np.float64)
    max_acc_header = np.zeros((n_stations, 3), dtype=np.float64)
    record_start_sample = np.zeros((n_stations,), dtype=np.int64)
    valid_n_samples = np.zeros((n_stations,), dtype=np.int64)
    pga_norm_native_loc_raw = np.zeros((n_stations,), dtype=np.int64)
    pga_norm_resampled_loc = np.zeros((n_stations,), dtype=np.int64)
    pga_norm_aligned_loc = np.zeros((n_stations,), dtype=np.int64)
    trigger_sample_raw = np.zeros((n_stations,), dtype=np.int64)
    trigger_sample_resampled = np.zeros((n_stations,), dtype=np.int64)

    station_codes: list[str] = []
    source_networks: list[str] = []
    sensor_classes: list[str] = []
    raw_dir_ns: list[str] = []
    raw_dir_ew: list[str] = []
    raw_dir_ud: list[str] = []
    scale_factor_ns: list[str] = []
    scale_factor_ew: list[str] = []
    scale_factor_ud: list[str] = []
    archive_relpaths: list[str] = []
    inner_archives: list[str] = []
    component_bases: list[str] = []
    sensor_suffixes: list[str] = []
    record_time_raws: list[str] = []
    record_start_times: list[str] = []
    trigger_times: list[str] = []
    duration_header_s = np.zeros((n_stations,), dtype=np.float64)
    sampling_rate_raw_hz = np.zeros((n_stations,), dtype=np.float64)

    for i, (station, offset) in enumerate(zip(stations, offsets)):
        n = station.waveform_resampled.shape[0]
        aligned[i, offset:offset + n, :] = station.waveform_resampled
        coords[i] = [station.station_lat, station.station_lon, station.station_height_m / 1000.0]
        p_picks[i] = offset + station.p_pick_resampled
        pga[i] = station.pga_norm_resampled_mps2
        pga_native[i] = station.pga_norm_native_mps2
        pga_vector_native[i] = station.pga_vector_native_mps2
        pga_vector_resampled[i] = station.pga_vector_resampled_mps2
        max_acc_header[i] = station.max_acc_header_gal
        record_start_sample[i] = offset
        valid_n_samples[i] = n
        pga_norm_native_loc_raw[i] = station.pga_norm_native_loc
        pga_norm_resampled_loc[i] = station.pga_norm_resampled_loc
        pga_norm_aligned_loc[i] = offset + station.pga_norm_resampled_loc
        trigger_sample_raw[i] = station.p_pick_raw
        trigger_sample_resampled[i] = station.p_pick_resampled
        duration_header_s[i] = station.duration_header_s
        sampling_rate_raw_hz[i] = station.sampling_rate_raw_hz

        station_codes.append(station.station_code)
        source_networks.append(station.source_network)
        sensor_classes.append(station.sensor_class)
        raw_dir_ns.append(station.raw_dirs[0])
        raw_dir_ew.append(station.raw_dirs[1])
        raw_dir_ud.append(station.raw_dirs[2])
        scale_factor_ns.append(station.scale_factors_raw[0])
        scale_factor_ew.append(station.scale_factors_raw[1])
        scale_factor_ud.append(station.scale_factors_raw[2])
        archive_relpaths.append(station.archive_relpath)
        inner_archives.append(station.inner_archive_name)
        component_bases.append(station.component_base)
        sensor_suffixes.append(station.sensor_suffix)
        record_time_raws.append(station.record_time_raw)
        record_start_times.append(format_jst_iso(station.record_start_ts))
        trigger_times.append(format_jst_iso(station.trigger_ts))

    source_mix = sorted(set(source_networks))
    event_meta = {
        "EVENT": stations[0].event_id,
        "Origin_Time(JST)": stations[0].origin_time_raw,
        "Latitude": stations[0].event_lat,
        "Longitude": stations[0].event_lon,
        "DEPTH": stations[0].event_depth_km,
        "Magnitude": stations[0].magnitude,
        "N_Stations": n_stations,
        "Source_Mix": ",".join(source_mix),
        "Archive_Path": stations[0].archive_relpath,
        "Event_Record_Start(JST)": format_jst_iso(global_start),
        "Event_Record_End(JST)": format_jst_iso(global_start + event_length / stations[0].sampling_rate_target_hz),
        "Event_Length_Samples": event_length,
        "Event_Length_Seconds": event_length / stations[0].sampling_rate_target_hz,
        "Sampling_Rate_Hz": stations[0].sampling_rate_target_hz,
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
        "sampling_rate_raw_hz": sampling_rate_raw_hz,
        "duration_header_s": duration_header_s,
        "station_codes": np.array(station_codes, dtype=object),
        "source_network": np.array(source_networks, dtype=object),
        "sensor_class": np.array(sensor_classes, dtype=object),
        "raw_dir_ns": np.array(raw_dir_ns, dtype=object),
        "raw_dir_ew": np.array(raw_dir_ew, dtype=object),
        "raw_dir_ud": np.array(raw_dir_ud, dtype=object),
        "scale_factor_ns": np.array(scale_factor_ns, dtype=object),
        "scale_factor_ew": np.array(scale_factor_ew, dtype=object),
        "scale_factor_ud": np.array(scale_factor_ud, dtype=object),
        "archive_relpath": np.array(archive_relpaths, dtype=object),
        "inner_archive": np.array(inner_archives, dtype=object),
        "component_base": np.array(component_bases, dtype=object),
        "sensor_suffix": np.array(sensor_suffixes, dtype=object),
        "record_time_raw": np.array(record_time_raws, dtype=object),
        "record_start_time_jst": np.array(record_start_times, dtype=object),
        "trigger_time_jst": np.array(trigger_times, dtype=object),
    }
    return datasets, event_meta


def stations_to_rows(stations: list[StationTrace], event_meta: dict[str, object]) -> list[dict[str, object]]:
    global_start = min(st.record_start_ts for st in stations)
    rows: list[dict[str, object]] = []
    for wave_idx, station in enumerate(stations):
        record_start_sample = int(round((station.record_start_ts - global_start) * station.sampling_rate_target_hz))
        row = {
            "EVENT": station.event_id,
            "wave_idx": wave_idx,
            "Origin_Time(JST)": station.origin_time_raw,
            "Latitude": station.event_lat,
            "Longitude": station.event_lon,
            "DEPTH": station.event_depth_km,
            "Magnitude": station.magnitude,
            "station_code": station.station_code,
            "station_lat": station.station_lat,
            "station_lon": station.station_lon,
            "station_height_m": station.station_height_m,
            "source_network": station.source_network,
            "sensor_class": station.sensor_class,
            "sensor_suffix": station.sensor_suffix,
            "raw_dir_ns": station.raw_dirs[0],
            "raw_dir_ew": station.raw_dirs[1],
            "raw_dir_ud": station.raw_dirs[2],
            "scale_factor_ns": station.scale_factors_raw[0],
            "scale_factor_ew": station.scale_factors_raw[1],
            "scale_factor_ud": station.scale_factors_raw[2],
            "sampling_rate_raw_hz": station.sampling_rate_raw_hz,
            "sampling_rate_hz": station.sampling_rate_target_hz,
            "duration_header_s": station.duration_header_s,
            "raw_n_samples": station.waveform_native.shape[0],
            "resampled_n_samples": station.waveform_resampled.shape[0],
            "record_time_raw": station.record_time_raw,
            "record_time_jst": format_jst_iso(station.record_time_ts),
            "record_start_time_jst": format_jst_iso(station.record_start_ts),
            "trigger_time_jst": format_jst_iso(station.trigger_ts),
            "record_start_sample": record_start_sample,
            "valid_n_samples": station.waveform_resampled.shape[0],
            "p_pick_raw": station.p_pick_raw,
            "p_pick_resampled": station.p_pick_resampled,
            "p_pick_aligned": record_start_sample + station.p_pick_resampled,
            "pga_norm_native_mps2": station.pga_norm_native_mps2,
            "pga_norm_resampled_mps2": station.pga_norm_resampled_mps2,
            "pga_norm_native_loc_raw": station.pga_norm_native_loc,
            "pga_norm_resampled_loc": station.pga_norm_resampled_loc,
            "pga_norm_aligned_loc": record_start_sample + station.pga_norm_resampled_loc,
            "pga_vector_native_ns_mps2": station.pga_vector_native_mps2[0],
            "pga_vector_native_ew_mps2": station.pga_vector_native_mps2[1],
            "pga_vector_native_ud_mps2": station.pga_vector_native_mps2[2],
            "pga_vector_resampled_ns_mps2": station.pga_vector_resampled_mps2[0],
            "pga_vector_resampled_ew_mps2": station.pga_vector_resampled_mps2[1],
            "pga_vector_resampled_ud_mps2": station.pga_vector_resampled_mps2[2],
            "max_acc_header_ns_gal": station.max_acc_header_gal[0],
            "max_acc_header_ew_gal": station.max_acc_header_gal[1],
            "max_acc_header_ud_gal": station.max_acc_header_gal[2],
            "archive_relpath": station.archive_relpath,
            "inner_archive": station.inner_archive_name,
            "component_base": station.component_base,
            "event_length_samples": event_meta["Event_Length_Samples"],
            "event_length_seconds": event_meta["Event_Length_Seconds"],
            "source_mix": event_meta["Source_Mix"],
        }
        rows.append(row)
    return rows


def apply_repaired_and_refined_p_picks(
    event_meta: dict[str, object],
    station_rows: list[dict[str, object]],
    datasets: dict[str, np.ndarray | object],
    pick_mode: str = "trigger_repair",
    final_pick: str = "stalta",
    travel_time_model: str = "constant",
    jma_table: JMATravelTimeTable | None = None,
    jma_search_margin_seconds: float = 10.0,
    diting_model: object | None = None,
    diting_device: str = "cuda:0",
    diting_batch_size: int = 100,
    diting_p_th: float = 0.1,
    diting_s_th: float = 0.1,
    diting_d_th: float = 0.3,
    diting_target_half_window_seconds: float = 10.0,
    diting_window_seconds: float = 100.0,
    velocity_highpass_hz: float = 0.05,
    threshold_seconds: float = 3.0,
    default_velocity_km_s: float = 6.0,
    min_velocity_km_s: float | None = None,
    max_velocity_km_s: float | None = None,
    travel_time_intercept_s: float = 0.0,
    min_margin_seconds: float = 0.5,
    stalta_pre_seconds: float = 4.0,
    stalta_post_seconds: float = 1.0,
    stalta_sta_seconds: float = 0.2,
    stalta_lta_seconds: float = 1.0,
    stalta_threshold_ratio: float = 0.2,
    stalta_feature: str = "vertical",
) -> tuple[dict[str, np.ndarray | object], dict[str, object], pd.DataFrame]:
    station_df = pd.DataFrame(station_rows)
    if pick_mode == "trigger_repair":
        repaired_df, fit_summary = estimate_event_repaired_p_picks(
            station_df,
            threshold_seconds=threshold_seconds,
            default_velocity_km_s=default_velocity_km_s,
            min_margin_seconds=min_margin_seconds,
        )
    elif pick_mode == "travel_time":
        repaired_df, fit_summary = estimate_event_travel_time_p_picks(
            station_df,
            travel_time_model=travel_time_model,
            jma_table=jma_table,
            jma_search_margin_seconds=jma_search_margin_seconds,
            p_velocity_km_s=default_velocity_km_s,
            min_velocity_km_s=min_velocity_km_s,
            max_velocity_km_s=max_velocity_km_s,
            travel_time_intercept_s=travel_time_intercept_s,
            threshold_seconds=threshold_seconds,
            min_margin_seconds=min_margin_seconds,
        )
    else:
        raise ValueError(f"Unsupported pick_mode: {pick_mode}")
    refined_df = refine_event_p_picks_from_arrays(
        event_df=repaired_df,
        waveforms=np.asarray(datasets["waveforms"]),
        record_start_sample=np.asarray(datasets["record_start_sample"]),
        valid_n_samples=np.asarray(datasets["valid_n_samples"]),
        coarse_pick_col="p_pick_repaired_aligned",
        pre_seconds=stalta_pre_seconds,
        post_seconds=stalta_post_seconds,
        sta_seconds=stalta_sta_seconds,
        lta_seconds=stalta_lta_seconds,
        threshold_ratio=stalta_threshold_ratio,
        feature=stalta_feature,
        search_left_col="p_pick_search_left_aligned",
        search_right_col="p_pick_search_right_aligned",
    )
    refined_df["p_pick_trigger_aligned"] = refined_df["p_pick_observed_aligned"].astype(np.int64)
    refined_df["p_pick_refined_aligned"] = refined_df["stalta_refined_pick_aligned"].astype(np.int64)
    refined_df["p_pick_refined_source"] = refined_df["stalta_method"].astype(str)
    refined_df["p_pick_aligned"] = refined_df["stalta_refined_pick_aligned"].astype(np.int64)

    if diting_model is not None:
        refined_df = add_diting_picks_to_event(
            event_df=refined_df,
            waveforms=np.asarray(datasets["waveforms"]),
            record_start_sample=np.asarray(datasets["record_start_sample"]),
            valid_n_samples=np.asarray(datasets["valid_n_samples"]),
            coarse_pick_col="p_pick_repaired_aligned",
            sampling_rate_hz=float(refined_df["sampling_rate_hz"].iloc[0]),
            model=diting_model,
            device=diting_device,
            batch_size=diting_batch_size,
            p_th=diting_p_th,
            s_th=diting_s_th,
            d_th=diting_d_th,
            target_half_window_seconds=diting_target_half_window_seconds,
            model_window_seconds=diting_window_seconds,
            velocity_highpass_hz=velocity_highpass_hz,
            search_left_col="p_pick_search_left_aligned",
            search_right_col="p_pick_search_right_aligned",
        )
        datasets["p_pick_diting_acc_aligned"] = refined_df["p_pick_diting_acc_aligned"].fillna(refined_df["p_pick_repaired_aligned"]).to_numpy(dtype=np.int64)
        datasets["p_pick_diting_vel_aligned"] = refined_df["p_pick_diting_vel_aligned"].fillna(refined_df["p_pick_repaired_aligned"]).to_numpy(dtype=np.int64)
        datasets["diting_acc_score"] = refined_df["diting_acc_score"].to_numpy(dtype=np.float64)
        datasets["diting_vel_score"] = refined_df["diting_vel_score"].to_numpy(dtype=np.float64)
        datasets["diting_acc_error"] = refined_df["diting_acc_error"].to_numpy(dtype=object)
        datasets["diting_vel_error"] = refined_df["diting_vel_error"].to_numpy(dtype=object)

    final_pick_columns = {
        "travel_time": "p_pick_repaired_aligned",
        "stalta": "p_pick_refined_aligned",
        "diting_acc": "p_pick_diting_acc_aligned",
        "diting_vel": "p_pick_diting_vel_aligned",
    }
    if final_pick not in final_pick_columns:
        raise ValueError(f"Unsupported final_pick: {final_pick}")
    final_col = final_pick_columns[final_pick]
    if final_col not in refined_df.columns:
        raise ValueError(f"final_pick={final_pick} requires --run_diting")
    final_values = pd.to_numeric(refined_df[final_col], errors="coerce").fillna(refined_df["p_pick_repaired_aligned"]).astype(np.int64)
    refined_df["p_pick_refined_aligned"] = final_values
    refined_df["p_pick_refined_source"] = final_pick
    refined_df["p_pick_refine_method"] = final_pick
    refined_df["p_pick_aligned"] = final_values

    datasets["p_pick_trigger_aligned"] = refined_df["p_pick_trigger_aligned"].to_numpy(dtype=np.int64)
    datasets["p_pick_repaired_aligned"] = refined_df["p_pick_repaired_aligned"].to_numpy(dtype=np.int64)
    datasets["p_pick_refined_aligned"] = refined_df["p_pick_refined_aligned"].to_numpy(dtype=np.int64)
    for col in (
        "p_pick_search_left_aligned",
        "p_pick_search_right_aligned",
        "p_pick_travel_time_fast_aligned",
        "p_pick_travel_time_slow_aligned",
        "p_pick_jma_grid_clipped",
    ):
        if col in refined_df.columns:
            datasets[col] = refined_df[col].to_numpy(dtype=np.int64)
    datasets["p_picks"] = datasets["p_pick_refined_aligned"]
    datasets["trigger_is_pick"] = refined_df["trigger_is_pick"].to_numpy(dtype=np.int8)
    datasets["p_pick_repaired_source"] = refined_df["p_pick_repaired_source"].to_numpy(dtype=object)
    datasets["p_pick_refined_source"] = refined_df["p_pick_refined_source"].to_numpy(dtype=object)
    datasets["p_pick_refine_method"] = refined_df["p_pick_refine_method"].to_numpy(dtype=object)
    datasets["stalta_ratio_peak"] = refined_df["stalta_ratio_peak"].to_numpy(dtype=np.float64)
    datasets["stalta_ratio_at_pick"] = refined_df["stalta_ratio_at_pick"].to_numpy(dtype=np.float64)
    datasets["stalta_search_left_aligned"] = refined_df["stalta_search_left_aligned"].to_numpy(dtype=np.int64)
    datasets["stalta_search_right_aligned"] = refined_df["stalta_search_right_aligned"].to_numpy(dtype=np.int64)

    event_meta = event_meta.copy()
    event_meta.update(
        {
            "P_Pick_Trigger_Threshold_S": float(threshold_seconds),
            "P_Pick_Mode": pick_mode,
            "P_Pick_Final_Source": final_pick,
            "P_Pick_Travel_Time_Model": fit_summary.get("travel_time_model", travel_time_model),
            "P_Pick_JMA_Search_Margin_S": float(fit_summary.get("jma_search_margin_seconds", jma_search_margin_seconds)),
            "P_Pick_Fit_Default_Velocity_Km_S": float(default_velocity_km_s),
            "P_Pick_Search_Min_Velocity_Km_S": float(fit_summary.get("min_velocity_km_s", default_velocity_km_s)),
            "P_Pick_Search_Max_Velocity_Km_S": float(fit_summary.get("max_velocity_km_s", default_velocity_km_s)),
            "P_Pick_Fit_Velocity_Km_S": float(fit_summary["velocity_km_s"]),
            "P_Pick_Fit_Slope_S_Per_Km": float(fit_summary["slope_s_per_km"]),
            "P_Pick_Fit_Intercept_S": float(fit_summary["intercept_s"]),
            "P_Pick_Fit_N_Trusted": int(fit_summary["n_trusted"]),
            "P_Pick_Fit_N_Clipped": int(fit_summary.get("n_clipped", 0)),
            "P_Pick_JMA_Grid_Clipped": int(fit_summary.get("n_jma_grid_clipped", 0)),
            "P_Pick_STA_Pre_S": float(stalta_pre_seconds),
            "P_Pick_STA_Post_S": float(stalta_post_seconds),
            "P_Pick_STA_STA_S": float(stalta_sta_seconds),
            "P_Pick_STA_LTA_S": float(stalta_lta_seconds),
            "P_Pick_STA_Threshold": float(stalta_threshold_ratio),
            "P_Pick_STA_Feature": stalta_feature,
            "P_Pick_DiTing_Enabled": int(diting_model is not None),
            "P_Pick_DiTing_Target_Half_Window_S": float(diting_target_half_window_seconds),
            "P_Pick_DiTing_Window_S": float(diting_window_seconds),
            "P_Pick_DiTing_Velocity_Highpass_Hz": float(velocity_highpass_hz),
        }
    )

    return datasets, event_meta, refined_df


def load_diting_model(
    ditingbench_root: str | None,
    model_name: str,
    weights: str,
    device: str,
) -> object:
    if ditingbench_root:
        sys.path.insert(0, str(Path(ditingbench_root).expanduser().resolve()))
    from dtbench.models import get_model

    return get_model(name=model_name, pretrain_weights=weights, device=device)


def prepare_diting_input_window(
    waveform_aligned_mps2: np.ndarray,
    coarse_pick_aligned: int,
    sampling_rate_hz: float,
    target_half_window_seconds: float = 10.0,
    model_window_seconds: float = 100.0,
    mode: str = "velocity",
    velocity_highpass_hz: float = 0.05,
    search_left_aligned: int | None = None,
    search_right_aligned: int | None = None,
) -> tuple[np.ndarray, int, int]:
    target_fs = 100.0
    if not math.isclose(sampling_rate_hz, target_fs):
        waveform = _resample_waveform(waveform_aligned_mps2, raw_fs=sampling_rate_hz, target_fs=target_fs)
        coarse_pick = int(round(coarse_pick_aligned * target_fs / sampling_rate_hz))
        search_left = None if search_left_aligned is None else int(round(search_left_aligned * target_fs / sampling_rate_hz))
        search_right = None if search_right_aligned is None else int(round(search_right_aligned * target_fs / sampling_rate_hz))
    else:
        waveform = waveform_aligned_mps2.copy()
        coarse_pick = int(coarse_pick_aligned)
        search_left = search_left_aligned
        search_right = search_right_aligned

    if search_left is not None and search_right is not None:
        crop_start = max(0, min(int(search_left), int(search_right)))
        crop_end = min(waveform.shape[0], max(int(search_left), int(search_right)) + 1)
    else:
        half = int(round(target_half_window_seconds * target_fs))
        crop_start = max(0, coarse_pick - half)
        crop_end = min(waveform.shape[0], coarse_pick + half)
    crop = waveform[crop_start:crop_end].astype(np.float64, copy=False)
    if crop.size:
        crop = detrend(crop, axis=0, type="linear")
        crop = crop - np.mean(crop, axis=0, keepdims=True)

    if mode == "velocity":
        if crop.shape[0] > 32 and velocity_highpass_hz > 0:
            sos = butter(4, velocity_highpass_hz, btype="highpass", fs=target_fs, output="sos")
            crop = sosfiltfilt(sos, crop, axis=0)
        crop = cumulative_trapezoid(crop, dx=1.0 / target_fs, axis=0, initial=0.0)
        if crop.size:
            crop = detrend(crop, axis=0, type="linear")
            crop = crop - np.mean(crop, axis=0, keepdims=True)
    elif mode != "acceleration":
        raise ValueError(f"Unsupported DiTing mode: {mode}")

    model_n = int(round(model_window_seconds * target_fs))
    window = np.zeros((model_n, 3), dtype=np.float32)
    if crop.shape[0]:
        tail_start = max(0, model_n - crop.shape[0])
        copy_n = min(crop.shape[0], model_n)
        window[tail_start:tail_start + copy_n, :] = crop[-copy_n:]
        placed_start = crop_start + crop.shape[0] - copy_n
    else:
        tail_start = model_n
        placed_start = crop_start
    return window, placed_start, tail_start


def run_diting_single_station(
    model: object,
    waveform_100s: np.ndarray,
    station_code: str,
    device: str,
    batch_size: int,
    p_th: float,
    s_th: float,
    d_th: float,
) -> tuple[float, float]:
    from obspy import Stream, Trace, UTCDateTime
    from dtbench.inference.dt_infer import diting_dpk_inference

    stream = Stream()
    for comp, channel in enumerate(("BHN", "BHE", "BHZ")):
        trace = Trace(data=waveform_100s[:, comp].astype(np.float32, copy=False))
        trace.stats.network = "JP"
        trace.stats.station = str(station_code)
        trace.stats.location = ""
        trace.stats.channel = channel
        trace.stats.sampling_rate = 100.0
        trace.stats.starttime = UTCDateTime(0)
        stream.append(trace)

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
            pick = pred[1][0]
            idx = float(pick[0])
            score = float(pick[1]) if len(pick) > 1 else np.nan
            if np.isnan(best_idx) or idx < best_idx:
                best_idx = idx
                best_score = score
    return best_idx, best_score


def add_diting_picks_to_event(
    event_df: pd.DataFrame,
    waveforms: np.ndarray,
    record_start_sample: np.ndarray,
    valid_n_samples: np.ndarray,
    coarse_pick_col: str,
    sampling_rate_hz: float,
    model: object,
    device: str,
    batch_size: int,
    p_th: float,
    s_th: float,
    d_th: float,
    target_half_window_seconds: float,
    model_window_seconds: float,
    velocity_highpass_hz: float,
    search_left_col: str | None = None,
    search_right_col: str | None = None,
) -> pd.DataFrame:
    df = event_df.copy().reset_index(drop=True)
    for col in (
        "p_pick_diting_acc_aligned",
        "p_pick_diting_vel_aligned",
        "diting_acc_score",
        "diting_vel_score",
        "diting_acc_error",
        "diting_vel_error",
    ):
        df[col] = "" if col.endswith("_error") else np.nan

    for idx, row in df.iterrows():
        wave_idx = int(row["wave_idx"])
        start = int(record_start_sample[wave_idx])
        valid_end = start + int(valid_n_samples[wave_idx])
        waveform = waveforms[wave_idx].copy()
        if start > 0:
            waveform[:start] = 0.0
        if valid_end < waveform.shape[0]:
            waveform[valid_end:] = 0.0

        coarse_pick = int(row[coarse_pick_col])
        search_left = int(row[search_left_col]) if search_left_col and search_left_col in df.columns and pd.notna(row[search_left_col]) else None
        search_right = int(row[search_right_col]) if search_right_col and search_right_col in df.columns and pd.notna(row[search_right_col]) else None
        for mode, suffix in (("acceleration", "acc"), ("velocity", "vel")):
            try:
                window, crop_start, tail_start = prepare_diting_input_window(
                    waveform_aligned_mps2=waveform,
                    coarse_pick_aligned=coarse_pick,
                    sampling_rate_hz=sampling_rate_hz,
                    target_half_window_seconds=target_half_window_seconds,
                    model_window_seconds=model_window_seconds,
                    mode=mode,
                    velocity_highpass_hz=velocity_highpass_hz,
                    search_left_aligned=search_left,
                    search_right_aligned=search_right,
                )
                pred_idx, score = run_diting_single_station(
                    model=model,
                    waveform_100s=window,
                    station_code=str(row["station_code"]),
                    device=device,
                    batch_size=batch_size,
                    p_th=p_th,
                    s_th=s_th,
                    d_th=d_th,
                )
                if not np.isfinite(pred_idx):
                    df.loc[idx, f"diting_{suffix}_error"] = "no_p_pick"
                    continue
                model_n = int(round(model_window_seconds * 100.0))
                crop_n = model_n - tail_start
                if pred_idx < tail_start or pred_idx >= tail_start + crop_n:
                    df.loc[idx, f"diting_{suffix}_error"] = "p_pick_outside_search_window"
                    continue
                aligned_pick_100hz = crop_start + int(round(pred_idx - tail_start))
                aligned_pick = int(round(aligned_pick_100hz * sampling_rate_hz / 100.0))
                aligned_pick = int(np.clip(aligned_pick, start, max(start, valid_end - 1)))
                df.loc[idx, f"p_pick_diting_{suffix}_aligned"] = aligned_pick
                df.loc[idx, f"diting_{suffix}_score"] = score
            except Exception as exc:
                df.loc[idx, f"diting_{suffix}_error"] = str(exc)

    return df


def write_pick_diagnostics(
    station_df: pd.DataFrame,
    hdf5_path: Path,
    diagnostics_dir: Path,
    n_waveform_plots: int = 24,
    random_seed: int = 2024,
) -> None:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    if station_df.empty:
        return

    base_col = "p_pick_repaired_aligned"
    if base_col not in station_df.columns:
        return

    pick_cols = [
        col for col in (
            "p_pick_trigger_aligned",
            "p_pick_observed_aligned",
            "p_pick_predicted_aligned",
            "p_pick_repaired_aligned",
            "stalta_refined_pick_aligned",
            "p_pick_diting_acc_aligned",
            "p_pick_diting_vel_aligned",
            "p_pick_refined_aligned",
        )
        if col in station_df.columns
    ]
    out_df = station_df.copy()
    sr = pd.to_numeric(out_df["sampling_rate_hz"], errors="coerce").replace(0, np.nan)
    base = pd.to_numeric(out_df[base_col], errors="coerce")

    summary_rows = []
    for col in pick_cols:
        values = pd.to_numeric(out_df[col], errors="coerce")
        diff_samples = values - base
        diff_seconds = diff_samples / sr
        out_df[f"{col}_minus_travel_time_samples"] = diff_samples
        out_df[f"{col}_minus_travel_time_s"] = diff_seconds
        finite = diff_seconds[np.isfinite(diff_seconds)]
        summary_rows.append(
            {
                "pick": col,
                "base_pick": base_col,
                "count": int(finite.size),
                "missing": int(diff_seconds.isna().sum()),
                "mean_diff_s": float(finite.mean()) if finite.size else np.nan,
                "std_diff_s": float(finite.std()) if finite.size else np.nan,
                "median_diff_s": float(finite.median()) if finite.size else np.nan,
                "mean_abs_diff_s": float(finite.abs().mean()) if finite.size else np.nan,
                "median_abs_diff_s": float(finite.abs().median()) if finite.size else np.nan,
                "p05_diff_s": float(finite.quantile(0.05)) if finite.size else np.nan,
                "p25_diff_s": float(finite.quantile(0.25)) if finite.size else np.nan,
                "p75_diff_s": float(finite.quantile(0.75)) if finite.size else np.nan,
                "p95_diff_s": float(finite.quantile(0.95)) if finite.size else np.nan,
                "min_diff_s": float(finite.min()) if finite.size else np.nan,
                "max_diff_s": float(finite.max()) if finite.size else np.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(diagnostics_dir / "pick_difference_summary.csv", index=False)
    if "p_pick_search_left_aligned" in out_df.columns and "p_pick_search_right_aligned" in out_df.columns:
        range_width_s = (
            pd.to_numeric(out_df["p_pick_search_right_aligned"], errors="coerce")
            - pd.to_numeric(out_df["p_pick_search_left_aligned"], errors="coerce")
        ) / sr
        out_df["p_pick_search_width_s"] = range_width_s
        finite_width = range_width_s[np.isfinite(range_width_s)]
        pd.DataFrame(
            [
                {
                    "count": int(finite_width.size),
                    "mean_width_s": float(finite_width.mean()) if finite_width.size else np.nan,
                    "median_width_s": float(finite_width.median()) if finite_width.size else np.nan,
                    "p05_width_s": float(finite_width.quantile(0.05)) if finite_width.size else np.nan,
                    "p95_width_s": float(finite_width.quantile(0.95)) if finite_width.size else np.nan,
                    "min_width_s": float(finite_width.min()) if finite_width.size else np.nan,
                    "max_width_s": float(finite_width.max()) if finite_width.size else np.nan,
                }
            ]
        ).to_csv(diagnostics_dir / "pick_search_range_summary.csv", index=False)
    out_df.to_csv(diagnostics_dir / "station_pick_differences.csv", index=False)

    try:
        os.environ.setdefault("MPLCONFIGDIR", str(diagnostics_dir / ".mplconfig"))
        (diagnostics_dir / ".mplconfig").mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not summary_df.empty:
        table_cols = ["pick", "count", "median_diff_s", "mean_abs_diff_s", "p05_diff_s", "p95_diff_s"]
        table_df = summary_df[table_cols].copy()
        for col in table_cols[2:]:
            table_df[col] = table_df[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        fig_height = max(2.2, 0.34 * (len(table_df) + 1))
        fig, ax = plt.subplots(figsize=(12, fig_height))
        ax.axis("off")
        table = ax.table(cellText=table_df.values, colLabels=table_df.columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.25)
        fig.tight_layout()
        fig.savefig(diagnostics_dir / "pick_difference_summary_table.png", dpi=180)
        plt.close(fig)

    for col in pick_cols:
        if col == base_col:
            continue
        diff = out_df[f"{col}_minus_travel_time_s"].dropna()
        if diff.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(diff, bins=80, color="0.25")
        ax.axvline(0.0, color="tab:blue", linewidth=1.2)
        ax.set_title(f"{col} - travel_time coarse pick")
        ax.set_xlabel("seconds")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(diagnostics_dir / f"{col}_minus_travel_time_hist.png", dpi=160)
        plt.close(fig)

    if n_waveform_plots <= 0:
        return
    sample_df = out_df.dropna(subset=["Magnitude"]).copy()
    if sample_df.empty:
        return
    try:
        n_mag_bins = min(4, int(sample_df["Magnitude"].nunique()))
        if n_mag_bins > 1:
            sample_df["mag_bin"] = pd.qcut(sample_df["Magnitude"], q=n_mag_bins, duplicates="drop")
        else:
            sample_df["mag_bin"] = "all"
    except Exception:
        sample_df["mag_bin"] = "all"

    sampled_parts = []
    n_bins = max(1, sample_df["mag_bin"].nunique())
    per_bin = max(1, math.ceil(n_waveform_plots / n_bins))
    for _, part in sample_df.groupby("mag_bin", observed=True):
        sampled_parts.append(part.sample(min(len(part), per_bin), random_state=random_seed))
    sampled = pd.concat(sampled_parts, ignore_index=True).sample(frac=1.0, random_state=random_seed).head(n_waveform_plots)
    sampled.to_csv(diagnostics_dir / "waveform_pick_plot_samples.csv", index=False)

    colors = {
        "p_pick_trigger_aligned": "tab:gray",
        "p_pick_predicted_aligned": "tab:cyan",
        "p_pick_repaired_aligned": "tab:blue",
        "stalta_refined_pick_aligned": "tab:orange",
        "p_pick_diting_acc_aligned": "tab:red",
        "p_pick_diting_vel_aligned": "tab:purple",
        "p_pick_refined_aligned": "black",
    }
    labels = {
        "p_pick_trigger_aligned": "trigger",
        "p_pick_predicted_aligned": "travel_pred",
        "p_pick_repaired_aligned": "travel_coarse",
        "stalta_refined_pick_aligned": "stalta",
        "p_pick_diting_acc_aligned": "diting_acc",
        "p_pick_diting_vel_aligned": "diting_vel",
        "p_pick_refined_aligned": "final",
    }

    with h5py.File(hdf5_path, "r") as h5:
        for plot_idx, row in enumerate(sampled.itertuples(index=False), start=1):
            row_dict = row._asdict()
            event = str(row_dict["EVENT"])
            wave_idx = int(row_dict["wave_idx"])
            if event not in h5["data"]:
                continue
            grp = h5["data"][event]
            waveform = grp["waveforms"][wave_idx]
            sr_hz = float(row_dict["sampling_rate_hz"])
            center = row_dict.get("p_pick_refined_aligned", row_dict.get(base_col))
            if not np.isfinite(center):
                center = row_dict.get(base_col)
            center = int(center)
            left = max(0, center - int(round(15.0 * sr_hz)))
            right = min(waveform.shape[0], center + int(round(25.0 * sr_hz)))
            if right <= left:
                continue
            t = (np.arange(left, right) - center) / sr_hz
            vertical = waveform[left:right, 2]
            norm = np.linalg.norm(waveform[left:right], axis=1)

            fig, axes = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True)
            axes[0].plot(t, vertical, color="0.2", linewidth=0.8)
            axes[0].set_ylabel("UD m/s^2")
            axes[1].plot(t, norm, color="0.2", linewidth=0.8)
            axes[1].set_ylabel("norm m/s^2")
            axes[1].set_xlabel("seconds relative to final pick")

            if "p_pick_search_left_aligned" in row_dict and "p_pick_search_right_aligned" in row_dict:
                search_left = row_dict["p_pick_search_left_aligned"]
                search_right = row_dict["p_pick_search_right_aligned"]
                if np.isfinite(search_left) and np.isfinite(search_right):
                    x0 = (float(min(search_left, search_right)) - center) / sr_hz
                    x1 = (float(max(search_left, search_right)) - center) / sr_hz
                    for ax in axes:
                        ax.axvspan(x0, x1, color="tab:blue", alpha=0.12, label="velocity_range")

            for col, color in colors.items():
                if col not in row_dict:
                    continue
                pick = row_dict[col]
                if not np.isfinite(pick):
                    continue
                x = (float(pick) - center) / sr_hz
                for ax in axes:
                    ax.axvline(x, color=color, linewidth=1.0, alpha=0.9, label=labels.get(col, col))
            handles, legend_labels = axes[0].get_legend_handles_labels()
            if handles:
                by_label = dict(zip(legend_labels, handles))
                axes[0].legend(by_label.values(), by_label.keys(), fontsize=7, ncol=4)
            title = (
                f"{event} wave_idx={wave_idx} station={row_dict.get('station_code', '')} "
                f"M={float(row_dict.get('Magnitude', np.nan)):.1f}"
            )
            axes[0].set_title(title)
            fig.tight_layout()
            fig.savefig(diagnostics_dir / f"waveform_pick_check_{plot_idx:03d}.png", dpi=160)
            plt.close(fig)


def _write_string_or_numeric_dataset(group: h5py.Group, name: str, values, compression: str | None = None, compression_opts: int | None = None):
    arr = np.asarray(values)
    kwargs = {}
    if compression is not None and arr.dtype.kind not in {"S", "O", "U"} and arr.ndim > 0:
        kwargs["compression"] = compression
        if compression_opts is not None:
            kwargs["compression_opts"] = compression_opts
    if arr.dtype.kind in {"U", "O"}:
        arr = np.asarray([str(v).encode("utf-8") for v in arr], dtype="S")
    group.create_dataset(name, data=arr, **kwargs)


def write_dataframe_group(group: h5py.Group, df: pd.DataFrame, compression: str | None = None, compression_opts: int | None = None):
    for col in df.columns:
        _write_string_or_numeric_dataset(group, col, df[col].to_numpy(), compression=compression, compression_opts=compression_opts)


def _interp_axis(grid: np.ndarray, value: float) -> tuple[int, int, float, bool]:
    clipped = False
    if value <= grid[0]:
        return 0, 0, 0.0, value < grid[0]
    if value >= grid[-1]:
        return len(grid) - 1, len(grid) - 1, 0.0, value > grid[-1]
    hi = int(np.searchsorted(grid, value, side="right"))
    lo = hi - 1
    span = float(grid[hi] - grid[lo])
    weight = 0.0 if span == 0 else float((value - grid[lo]) / span)
    return lo, hi, weight, clipped


def _lerp(v0: float, v1: float, weight: float) -> float:
    return float(v0 * (1.0 - weight) + v1 * weight)


class JMATravelTimeTable:
    def __init__(self, zip_path: Path):
        self.zip_path = Path(zip_path).expanduser().resolve()
        if not self.zip_path.exists():
            raise FileNotFoundError(f"JMA travel-time table not found: {self.zip_path}")
        self.altitudes_m, self.depths_km, self.distances_km, self.p_seconds = self._load_zip(self.zip_path)

    @staticmethod
    def _member_altitude_m(name: str) -> int | None:
        match = re.search(r"tjma2001h[./]tjma2001h\.([+-]\d{5})$", name)
        if not match:
            return None
        return int(match.group(1))

    @classmethod
    def _load_zip(cls, zip_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tables: list[tuple[int, np.ndarray]] = []
        depths_ref = None
        distances_ref = None
        with zipfile.ZipFile(zip_path) as zf:
            members = []
            for name in zf.namelist():
                altitude = cls._member_altitude_m(name)
                if altitude is not None:
                    members.append((altitude, name))
            if not members:
                raise ValueError(f"No tjma2001h table files found in {zip_path}")

            for altitude, name in sorted(members):
                rows = []
                with zf.open(name) as fp:
                    for raw_line in fp:
                        parts = raw_line.decode("ascii", errors="ignore").split()
                        if len(parts) < 6 or parts[0] != "P":
                            continue
                        rows.append((float(parts[1]), float(parts[3]), float(parts[4]), float(parts[5])))
                arr = np.asarray(rows, dtype=np.float64)
                if arr.size == 0:
                    raise ValueError(f"No travel-time rows found in {name}")
                depths = np.unique(arr[:, 2])
                distances = np.unique(arr[:, 3])
                if depths_ref is None:
                    depths_ref = depths
                    distances_ref = distances
                elif not (np.array_equal(depths_ref, depths) and np.array_equal(distances_ref, distances)):
                    raise ValueError(f"Inconsistent JMA grid in {name}")

                depth_index = {float(v): i for i, v in enumerate(depths)}
                distance_index = {float(v): i for i, v in enumerate(distances)}
                table = np.empty((len(depths), len(distances)), dtype=np.float64)
                table.fill(np.nan)
                for p_time, _s_time, depth, distance in arr:
                    table[depth_index[float(depth)], distance_index[float(distance)]] = p_time
                if np.isnan(table).any():
                    raise ValueError(f"Incomplete JMA grid in {name}")
                tables.append((altitude, table))

        altitudes = np.asarray([alt for alt, _ in tables], dtype=np.float64)
        p_seconds = np.stack([table for _, table in tables], axis=0)
        return altitudes, np.asarray(depths_ref, dtype=np.float64), np.asarray(distances_ref, dtype=np.float64), p_seconds

    def predict_p_seconds(self, depth_km: float, distance_km: float, station_height_m: float) -> tuple[float, bool]:
        a0, a1, aw, a_clipped = _interp_axis(self.altitudes_m, float(station_height_m))
        d0, d1, dw, d_clipped = _interp_axis(self.depths_km, float(depth_km))
        x0, x1, xw, x_clipped = _interp_axis(self.distances_km, float(distance_km))

        def interp_for_alt(ai: int) -> float:
            v00 = self.p_seconds[ai, d0, x0]
            v01 = self.p_seconds[ai, d0, x1]
            v10 = self.p_seconds[ai, d1, x0]
            v11 = self.p_seconds[ai, d1, x1]
            vx0 = _lerp(v00, v01, xw)
            vx1 = _lerp(v10, v11, xw)
            return _lerp(vx0, vx1, dw)

        t0 = interp_for_alt(a0)
        t1 = interp_for_alt(a1)
        return _lerp(t0, t1, aw), bool(a_clipped or d_clipped or x_clipped)


def build_year_dataset(
    waveform_root: Path,
    output_hdf5: Path,
    output_events_csv: Path,
    output_stations_csv: Path,
    year: int,
    target_sampling_rate_hz: float = 100.0,
    min_stations: int = 3,
    limit_events: int | None = None,
    compression_level: int = 4,
    pick_mode: str = "trigger_repair",
    final_pick: str = "stalta",
    travel_time_model: str = "jma2001a",
    jma_travel_time_zip: Path | None = None,
    jma_search_margin_seconds: float = 10.0,
    p_velocity_km_s: float = 6.0,
    p_velocity_min_km_s: float | None = None,
    p_velocity_max_km_s: float | None = None,
    travel_time_intercept_s: float = 0.0,
    stalta_pre_seconds: float = 4.0,
    stalta_post_seconds: float = 1.0,
    stalta_sta_seconds: float = 0.2,
    stalta_lta_seconds: float = 1.0,
    stalta_threshold_ratio: float = 0.2,
    stalta_feature: str = "vertical",
    run_diting: bool = False,
    ditingbench_root: str | None = None,
    diting_model_name: str = "diting1200m",
    diting_weights: str | None = None,
    diting_device: str = "cuda:0",
    diting_batch_size: int = 100,
    diting_p_th: float = 0.1,
    diting_s_th: float = 0.1,
    diting_d_th: float = 0.3,
    diting_target_half_window_seconds: float = 10.0,
    diting_window_seconds: float = 100.0,
    velocity_highpass_hz: float = 0.05,
    diagnostics_dir: Path | None = None,
    n_diagnostic_plots: int = 24,
    diagnostic_random_seed: int = 2024,
) -> dict[str, int]:
    year_dir = waveform_root / str(year)
    outer_archives = sorted(year_dir.glob("*.tar"))
    if limit_events is not None:
        outer_archives = outer_archives[:limit_events]

    stats = Counter()
    event_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    jma_table = None
    if travel_time_model == "jma2001a":
        jma_table = JMATravelTimeTable(jma_travel_time_zip or DEFAULT_JMA2001A_ZIP)
    elif travel_time_model != "constant":
        raise ValueError(f"Unsupported travel_time_model: {travel_time_model}")

    diting_model = None
    if run_diting:
        if diting_weights is None:
            raise ValueError("diting_weights must be set when run_diting=True")
        diting_model = load_diting_model(
            ditingbench_root=ditingbench_root,
            model_name=diting_model_name,
            weights=diting_weights,
            device=diting_device,
        )

    compression = "gzip"
    with h5py.File(output_hdf5, "w") as out_h5:
        meta_grp = out_h5.create_group("metadata")
        data_grp = out_h5.create_group("data")
        meta_grp.create_dataset("sampling_rate", data=target_sampling_rate_hz)
        meta_grp.create_dataset("pretrigger_seconds", data=PRETRIGGER_SECONDS)
        _write_string_or_numeric_dataset(meta_grp, "alignment_mode", np.array(["earliest_record_start"], dtype=object))
        _write_string_or_numeric_dataset(meta_grp, "waveform_root", np.array([str(waveform_root)], dtype=object))
        meta_grp.create_dataset("year", data=year)

        for outer_tar_path in tqdm(outer_archives, desc=f"year {year}", unit="event"):
            stats["events_seen"] += 1
            archive_relpath = str(outer_tar_path.relative_to(waveform_root))
            try:
                stations = load_station_traces_from_event_archive(
                    outer_tar_path=outer_tar_path,
                    archive_relpath=archive_relpath,
                    target_sampling_rate_hz=target_sampling_rate_hz,
                )
            except Exception:
                stats["events_failed"] += 1
                continue

            if not stations:
                stats["events_empty"] += 1
                continue

            if len(stations) < min_stations:
                stats["events_below_min_stations"] += 1
                stats["stations_dropped_below_min"] += len(stations)
                continue

            datasets, event_meta = build_aligned_event(stations)
            event_station_rows = stations_to_rows(stations, event_meta)
            datasets, event_meta, refined_station_df = apply_repaired_and_refined_p_picks(
                event_meta=event_meta,
                station_rows=event_station_rows,
                datasets=datasets,
                pick_mode=pick_mode,
                final_pick=final_pick,
                travel_time_model=travel_time_model,
                jma_table=jma_table,
                jma_search_margin_seconds=jma_search_margin_seconds,
                diting_model=diting_model,
                diting_device=diting_device,
                diting_batch_size=diting_batch_size,
                diting_p_th=diting_p_th,
                diting_s_th=diting_s_th,
                diting_d_th=diting_d_th,
                diting_target_half_window_seconds=diting_target_half_window_seconds,
                diting_window_seconds=diting_window_seconds,
                velocity_highpass_hz=velocity_highpass_hz,
                default_velocity_km_s=p_velocity_km_s,
                min_velocity_km_s=p_velocity_min_km_s,
                max_velocity_km_s=p_velocity_max_km_s,
                travel_time_intercept_s=travel_time_intercept_s,
                stalta_pre_seconds=stalta_pre_seconds,
                stalta_post_seconds=stalta_post_seconds,
                stalta_sta_seconds=stalta_sta_seconds,
                stalta_lta_seconds=stalta_lta_seconds,
                stalta_threshold_ratio=stalta_threshold_ratio,
                stalta_feature=stalta_feature,
            )
            event_rows.append(event_meta)
            station_rows.extend(refined_station_df.to_dict(orient="records"))

            event_grp = data_grp.create_group(event_meta["EVENT"])
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

            stats["events_written"] += 1
            stats["stations_written"] += len(stations)

        event_df = pd.DataFrame(event_rows)
        station_df = pd.DataFrame(station_rows)
        write_dataframe_group(meta_grp.create_group("event_metadata"), event_df, compression=compression, compression_opts=compression_level)
        write_dataframe_group(meta_grp.create_group("station_metadata"), station_df, compression=compression, compression_opts=compression_level)

    event_df.to_csv(output_events_csv, index=False)
    station_df.to_csv(output_stations_csv, index=False)
    if diagnostics_dir is not None and not station_df.empty:
        write_pick_diagnostics(
            station_df=station_df,
            hdf5_path=output_hdf5,
            diagnostics_dir=diagnostics_dir,
            n_waveform_plots=n_diagnostic_plots,
            random_seed=diagnostic_random_seed,
        )
    return dict(stats)


def discover_years(waveform_root: Path) -> list[int]:
    years = []
    for child in waveform_root.iterdir():
        if child.is_dir() and re.fullmatch(r"\d{4}", child.name):
            years.append(int(child.name))
    return sorted(years)


def load_converted_station(
    hdf5_path: Path,
    event_id: str,
    wave_idx: int,
) -> dict[str, np.ndarray | str | float | int]:
    with h5py.File(hdf5_path, "r") as f:
        grp = f["data"][event_id]
        out = {}
        for key in grp.keys():
            data = grp[key][()]
            if isinstance(data, np.ndarray) and data.dtype.kind == "S":
                data = np.array([v.decode("utf-8") for v in data])
            out[key] = data
        return {
            "waveform": out["waveforms"][wave_idx],
            "record_start_sample": int(out["record_start_sample"][wave_idx]),
            "valid_n_samples": int(out["valid_n_samples"][wave_idx]),
            "p_pick": int(out["p_picks"][wave_idx]),
            "p_pick_trigger_aligned": int(out["p_pick_trigger_aligned"][wave_idx]) if "p_pick_trigger_aligned" in out else int(out["p_picks"][wave_idx]),
            "p_pick_repaired_aligned": int(out["p_pick_repaired_aligned"][wave_idx]) if "p_pick_repaired_aligned" in out else int(out["p_picks"][wave_idx]),
            "p_pick_refined_aligned": int(out["p_pick_refined_aligned"][wave_idx]) if "p_pick_refined_aligned" in out else int(out["p_picks"][wave_idx]),
            "pga_norm_resampled_mps2": float(out["pga_norm_resampled_mps2"][wave_idx]),
            "pga_norm_native_mps2": float(out["pga_norm_native_mps2"][wave_idx]),
            "pga_norm_aligned_loc": int(out["pga_norm_aligned_loc"][wave_idx]),
            "pga_norm_resampled_loc": int(out["pga_norm_resampled_loc"][wave_idx]),
            "max_acc_header_gal": out["max_acc_header_gal"][wave_idx],
        }


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1e-15, 1.0 - a)))
    return radius_km * c


def hypocentral_distance_km(
    event_lat: float,
    event_lon: float,
    event_depth_km: float,
    station_lat: float,
    station_lon: float,
    station_height_m: float,
) -> float:
    horizontal = haversine_distance_km(event_lat, event_lon, station_lat, station_lon)
    station_height_km = station_height_m / 1000.0
    station_depth_km = -station_height_km
    vertical = event_depth_km - station_depth_km
    return float(math.sqrt(horizontal ** 2 + vertical ** 2))


def classify_trigger_is_pick(
    p_pick_aligned: np.ndarray,
    pga_aligned_loc: np.ndarray,
    sampling_rate_hz: float,
    threshold_seconds: float = 3.0,
) -> np.ndarray:
    delta_sec = (np.asarray(pga_aligned_loc) - np.asarray(p_pick_aligned)) / float(sampling_rate_hz)
    return delta_sec > threshold_seconds


def robust_linear_fit_distance_to_tp(
    distances_km: np.ndarray,
    tp_seconds: np.ndarray,
    min_slope: float = 1.0 / 8.0,
    max_slope: float = 1.0 / 3.0,
) -> dict[str, np.ndarray | float | int]:
    x = np.asarray(distances_km, dtype=np.float64)
    y = np.asarray(tp_seconds, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        raise ValueError("No finite samples to fit")

    if x.size == 1:
        slope = 1.0 / 6.0
        intercept = float(y[0] - slope * x[0])
        residuals = y - (intercept + slope * x)
        return {
            "intercept": intercept,
            "slope": slope,
            "velocity_km_s": 1.0 / slope,
            "inlier_mask": np.array([True]),
            "residuals": residuals,
            "n_obs": 1,
        }

    slope, intercept = np.polyfit(x, y, 1)
    slope = float(np.clip(slope, min_slope, max_slope))
    intercept = float(intercept)
    residuals = y - (intercept + slope * x)
    mad = np.median(np.abs(residuals - np.median(residuals)))
    limit = max(0.75, 3.0 * 1.4826 * mad)
    inlier_mask = np.abs(residuals) <= limit
    if np.sum(inlier_mask) >= 2:
        slope, intercept = np.polyfit(x[inlier_mask], y[inlier_mask], 1)
        slope = float(np.clip(slope, min_slope, max_slope))
        intercept = float(intercept)
        residuals = y - (intercept + slope * x)
    else:
        inlier_mask = np.ones_like(x, dtype=bool)

    return {
        "intercept": intercept,
        "slope": slope,
        "velocity_km_s": 1.0 / slope,
        "inlier_mask": inlier_mask,
        "residuals": residuals,
        "n_obs": int(x.size),
    }


def estimate_event_repaired_p_picks(
    event_df: pd.DataFrame,
    threshold_seconds: float = 3.0,
    default_velocity_km_s: float = 6.0,
    min_margin_seconds: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if event_df.empty:
        raise ValueError("event_df must not be empty")

    df = event_df.copy().reset_index(drop=True)
    sampling_rate_hz = float(df["sampling_rate_hz"].iloc[0])
    origin_raw = str(df["Origin_Time(JST)"].iloc[0])
    _, origin_ts = parse_jst_timestamp(origin_raw)
    global_start_ts = pd.to_datetime(df["record_start_time_jst"], utc=True).min().timestamp()

    distances_km = []
    epicentral_distances_km = []
    for _, row in df.iterrows():
        epicentral_distances_km.append(
            haversine_distance_km(
                float(row["Latitude"]),
                float(row["Longitude"]),
                float(row["station_lat"]),
                float(row["station_lon"]),
            )
        )
        distances_km.append(
            hypocentral_distance_km(
                event_lat=float(row["Latitude"]),
                event_lon=float(row["Longitude"]),
                event_depth_km=float(row["DEPTH"]),
                station_lat=float(row["station_lat"]),
                station_lon=float(row["station_lon"]),
                station_height_m=float(row["station_height_m"]),
            )
        )
    df["hypocentral_distance_km"] = distances_km
    df["epicentral_distance_km"] = epicentral_distances_km
    df["trigger_to_pga_seconds"] = (df["pga_norm_aligned_loc"] - df["p_pick_aligned"]) / sampling_rate_hz
    df["trigger_is_pick"] = classify_trigger_is_pick(
        p_pick_aligned=df["p_pick_aligned"].to_numpy(),
        pga_aligned_loc=df["pga_norm_aligned_loc"].to_numpy(),
        sampling_rate_hz=sampling_rate_hz,
        threshold_seconds=threshold_seconds,
    )

    observed_abs_ts = global_start_ts + df["p_pick_aligned"].to_numpy(dtype=np.float64) / sampling_rate_hz
    observed_tp_seconds = observed_abs_ts - origin_ts

    trusted = df["trigger_is_pick"].to_numpy(dtype=bool)
    if int(np.sum(trusted)) >= 1:
        fit = robust_linear_fit_distance_to_tp(
            distances_km=df.loc[trusted, "hypocentral_distance_km"].to_numpy(),
            tp_seconds=observed_tp_seconds[trusted],
        )
        intercept = float(fit["intercept"])
        slope = float(fit["slope"])
        velocity_km_s = float(fit["velocity_km_s"])
    else:
        slope = 1.0 / float(default_velocity_km_s)
        intercept = 0.0
        velocity_km_s = float(default_velocity_km_s)
        fit = {
            "intercept": intercept,
            "slope": slope,
            "velocity_km_s": velocity_km_s,
            "inlier_mask": np.zeros((0,), dtype=bool),
            "residuals": np.zeros((0,), dtype=np.float64),
            "n_obs": 0,
        }

    predicted_tp_seconds = intercept + slope * df["hypocentral_distance_km"].to_numpy()
    predicted_abs_ts = origin_ts + predicted_tp_seconds
    predicted_aligned = np.rint((predicted_abs_ts - global_start_ts) * sampling_rate_hz).astype(np.int64)
    pga_margin_samples = int(round(min_margin_seconds * sampling_rate_hz))
    max_allowed = df["pga_norm_aligned_loc"].to_numpy(dtype=np.int64) - pga_margin_samples
    repaired_aligned = np.minimum(predicted_aligned, max_allowed)
    repaired_aligned = np.maximum(repaired_aligned, df["record_start_sample"].to_numpy(dtype=np.int64))

    df["p_pick_observed_aligned"] = df["p_pick_aligned"].astype(np.int64)
    df["p_pick_observed_seconds_after_origin"] = observed_tp_seconds
    df["p_pick_predicted_aligned"] = predicted_aligned
    df["p_pick_predicted_seconds_after_origin"] = predicted_tp_seconds
    df["p_pick_repaired_aligned"] = np.where(trusted, df["p_pick_aligned"], repaired_aligned)
    df["p_pick_repaired_source"] = np.where(trusted, "trigger", "distance_fit")
    df["p_pick_repaired_seconds_after_origin"] = (
        global_start_ts + df["p_pick_repaired_aligned"].to_numpy(dtype=np.float64) / sampling_rate_hz - origin_ts
    )

    fit_summary = {
        "threshold_seconds": float(threshold_seconds),
        "default_velocity_km_s": float(default_velocity_km_s),
        "velocity_km_s": float(velocity_km_s),
        "slope_s_per_km": float(slope),
        "intercept_s": float(intercept),
        "n_total": int(len(df)),
        "n_trusted": int(np.sum(trusted)),
    }
    if fit["n_obs"]:
        fit_summary["residual_mad_s"] = float(np.median(np.abs(np.asarray(fit["residuals"]) - np.median(np.asarray(fit["residuals"])))))
    return df, fit_summary


def estimate_event_travel_time_p_picks(
    event_df: pd.DataFrame,
    travel_time_model: str = "constant",
    jma_table: JMATravelTimeTable | None = None,
    jma_search_margin_seconds: float = 10.0,
    p_velocity_km_s: float = 6.0,
    min_velocity_km_s: float | None = None,
    max_velocity_km_s: float | None = None,
    travel_time_intercept_s: float = 0.0,
    threshold_seconds: float = 3.0,
    min_margin_seconds: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if event_df.empty:
        raise ValueError("event_df must not be empty")

    df = event_df.copy().reset_index(drop=True)
    sampling_rate_hz = float(df["sampling_rate_hz"].iloc[0])
    origin_raw = str(df["Origin_Time(JST)"].iloc[0])
    _, origin_ts = parse_jst_timestamp(origin_raw)
    global_start_ts = pd.to_datetime(df["record_start_time_jst"], utc=True).min().timestamp()

    distances_km = []
    epicentral_distances_km = []
    for _, row in df.iterrows():
        epicentral_distances_km.append(
            haversine_distance_km(
                float(row["Latitude"]),
                float(row["Longitude"]),
                float(row["station_lat"]),
                float(row["station_lon"]),
            )
        )
        distances_km.append(
            hypocentral_distance_km(
                event_lat=float(row["Latitude"]),
                event_lon=float(row["Longitude"]),
                event_depth_km=float(row["DEPTH"]),
                station_lat=float(row["station_lat"]),
                station_lon=float(row["station_lon"]),
                station_height_m=float(row["station_height_m"]),
            )
        )
    df["hypocentral_distance_km"] = distances_km
    df["epicentral_distance_km"] = epicentral_distances_km
    df["trigger_to_pga_seconds"] = (df["pga_norm_aligned_loc"] - df["p_pick_aligned"]) / sampling_rate_hz
    df["trigger_is_pick"] = classify_trigger_is_pick(
        p_pick_aligned=df["p_pick_aligned"].to_numpy(),
        pga_aligned_loc=df["pga_norm_aligned_loc"].to_numpy(),
        sampling_rate_hz=sampling_rate_hz,
        threshold_seconds=threshold_seconds,
    )

    observed_abs_ts = global_start_ts + df["p_pick_aligned"].to_numpy(dtype=np.float64) / sampling_rate_hz
    observed_tp_seconds = observed_abs_ts - origin_ts
    jma_clipped = np.zeros((len(df),), dtype=bool)
    if travel_time_model == "jma2001a":
        if jma_table is None:
            raise ValueError("jma_table is required when travel_time_model='jma2001a'")
        predicted = []
        for i, row in df.iterrows():
            p_seconds, clipped = jma_table.predict_p_seconds(
                depth_km=float(row["DEPTH"]),
                distance_km=float(row["epicentral_distance_km"]),
                station_height_m=float(row["station_height_m"]),
            )
            predicted.append(float(p_seconds) + float(travel_time_intercept_s))
            jma_clipped[i] = clipped
        predicted_tp_seconds = np.asarray(predicted, dtype=np.float64)
        fast_tp_seconds = predicted_tp_seconds - float(jma_search_margin_seconds)
        slow_tp_seconds = predicted_tp_seconds + float(jma_search_margin_seconds)
        min_velocity_km_s = np.nan if min_velocity_km_s is None else float(min_velocity_km_s)
        max_velocity_km_s = np.nan if max_velocity_km_s is None else float(max_velocity_km_s)
    elif travel_time_model == "constant":
        if min_velocity_km_s is None:
            min_velocity_km_s = p_velocity_km_s
        if max_velocity_km_s is None:
            max_velocity_km_s = p_velocity_km_s
        min_velocity_km_s = float(min_velocity_km_s)
        max_velocity_km_s = float(max_velocity_km_s)
        if min_velocity_km_s <= 0 or max_velocity_km_s <= 0:
            raise ValueError("Velocity bounds must be positive")
        if min_velocity_km_s > max_velocity_km_s:
            min_velocity_km_s, max_velocity_km_s = max_velocity_km_s, min_velocity_km_s
        if not (min_velocity_km_s <= float(p_velocity_km_s) <= max_velocity_km_s):
            raise ValueError(
                f"p_velocity_km_s ({p_velocity_km_s}) must be within "
                f"[{min_velocity_km_s}, {max_velocity_km_s}]"
            )
        distances = df["hypocentral_distance_km"].to_numpy(dtype=np.float64)
        predicted_tp_seconds = float(travel_time_intercept_s) + distances / float(p_velocity_km_s)
        fast_tp_seconds = float(travel_time_intercept_s) + distances / max_velocity_km_s
        slow_tp_seconds = float(travel_time_intercept_s) + distances / min_velocity_km_s
    else:
        raise ValueError(f"Unsupported travel_time_model: {travel_time_model}")
    predicted_abs_ts = origin_ts + predicted_tp_seconds
    predicted_aligned = np.rint((predicted_abs_ts - global_start_ts) * sampling_rate_hz).astype(np.int64)
    fast_aligned = np.rint((origin_ts + fast_tp_seconds - global_start_ts) * sampling_rate_hz).astype(np.int64)
    slow_aligned = np.rint((origin_ts + slow_tp_seconds - global_start_ts) * sampling_rate_hz).astype(np.int64)

    pga_margin_samples = int(round(min_margin_seconds * sampling_rate_hz))
    record_start = df["record_start_sample"].to_numpy(dtype=np.int64)
    valid_n = df["valid_n_samples"].to_numpy(dtype=np.int64)
    valid_end = record_start + np.maximum(valid_n - 1, 0)
    max_allowed = np.minimum(valid_end, df["pga_norm_aligned_loc"].to_numpy(dtype=np.int64) - pga_margin_samples)
    max_allowed = np.maximum(max_allowed, record_start)
    repaired_aligned = np.clip(predicted_aligned, record_start, max_allowed).astype(np.int64)
    search_left = np.clip(np.minimum(fast_aligned, slow_aligned), record_start, valid_end).astype(np.int64)
    search_right = np.clip(np.maximum(fast_aligned, slow_aligned), record_start, valid_end).astype(np.int64)
    swapped = search_left > search_right
    if np.any(swapped):
        tmp = search_left.copy()
        search_left[swapped] = search_right[swapped]
        search_right[swapped] = tmp[swapped]

    df["p_pick_observed_aligned"] = df["p_pick_aligned"].astype(np.int64)
    df["p_pick_observed_seconds_after_origin"] = observed_tp_seconds
    df["p_pick_predicted_aligned"] = predicted_aligned
    df["p_pick_predicted_seconds_after_origin"] = predicted_tp_seconds
    df["p_pick_travel_time_fast_aligned"] = fast_aligned
    df["p_pick_travel_time_slow_aligned"] = slow_aligned
    df["p_pick_travel_time_fast_seconds_after_origin"] = fast_tp_seconds
    df["p_pick_travel_time_slow_seconds_after_origin"] = slow_tp_seconds
    df["p_pick_jma_grid_clipped"] = jma_clipped.astype(np.int8)
    df["p_pick_search_left_aligned"] = search_left
    df["p_pick_search_right_aligned"] = search_right
    df["p_pick_search_left_seconds_after_origin"] = global_start_ts + search_left.astype(np.float64) / sampling_rate_hz - origin_ts
    df["p_pick_search_right_seconds_after_origin"] = global_start_ts + search_right.astype(np.float64) / sampling_rate_hz - origin_ts
    df["p_pick_search_width_seconds"] = (search_right - search_left) / sampling_rate_hz
    df["p_pick_repaired_aligned"] = repaired_aligned
    source_name = "jma2001a" if travel_time_model == "jma2001a" else "travel_time"
    clipped_name = f"{source_name}_clipped"
    df["p_pick_repaired_source"] = np.where(predicted_aligned == repaired_aligned, source_name, clipped_name)
    df["p_pick_repaired_seconds_after_origin"] = (
        global_start_ts + df["p_pick_repaired_aligned"].to_numpy(dtype=np.float64) / sampling_rate_hz - origin_ts
    )

    fit_summary = {
        "travel_time_model": travel_time_model,
        "jma_search_margin_seconds": float(jma_search_margin_seconds),
        "threshold_seconds": float(threshold_seconds),
        "default_velocity_km_s": float(p_velocity_km_s),
        "min_velocity_km_s": float(min_velocity_km_s),
        "max_velocity_km_s": float(max_velocity_km_s),
        "velocity_km_s": float(p_velocity_km_s) if travel_time_model == "constant" else float("nan"),
        "slope_s_per_km": float(1.0 / float(p_velocity_km_s)) if travel_time_model == "constant" else float("nan"),
        "intercept_s": float(travel_time_intercept_s),
        "n_total": int(len(df)),
        "n_trusted": int(np.sum(df["trigger_is_pick"].to_numpy(dtype=bool))),
        "n_clipped": int(np.sum(predicted_aligned != repaired_aligned)),
        "n_jma_grid_clipped": int(np.sum(jma_clipped)),
    }
    return df, fit_summary


def _moving_average_same(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.astype(np.float64, copy=True)
    kernel = np.ones((window,), dtype=np.float64) / float(window)
    return np.convolve(x, kernel, mode="same")


def refine_pick_stalta(
    waveform_aligned: np.ndarray,
    coarse_pick_aligned: int,
    sampling_rate_hz: float,
    record_start_sample: int = 0,
    valid_n_samples: int | None = None,
    search_left_aligned: int | None = None,
    search_right_aligned: int | None = None,
    pre_seconds: float = 4.0,
    post_seconds: float = 1.0,
    sta_seconds: float = 0.2,
    lta_seconds: float = 1.0,
    threshold_ratio: float = 2.5,
    feature: str = "vertical",
) -> dict[str, int | float | str | np.ndarray]:
    if valid_n_samples is None:
        valid_n_samples = waveform_aligned.shape[0] - record_start_sample

    start = int(record_start_sample)
    end = min(waveform_aligned.shape[0], start + int(valid_n_samples))
    valid = waveform_aligned[start:end]
    if valid.size == 0:
        return {
            "refined_pick_aligned": int(coarse_pick_aligned),
            "search_left_aligned": int(coarse_pick_aligned),
            "search_right_aligned": int(coarse_pick_aligned),
            "ratio_peak": float("nan"),
            "ratio_at_pick": float("nan"),
            "method": "empty",
            "search_feature": feature,
        }

    coarse_rel = int(np.clip(coarse_pick_aligned - start, 0, max(0, valid.shape[0] - 1)))
    if search_left_aligned is not None and search_right_aligned is not None:
        left_abs = min(int(search_left_aligned), int(search_right_aligned))
        right_abs = max(int(search_left_aligned), int(search_right_aligned))
        left = int(np.clip(left_abs - start, 0, max(0, valid.shape[0] - 1)))
        right = int(np.clip(right_abs - start + 1, left + 1, valid.shape[0]))
    else:
        left = max(0, coarse_rel - int(round(pre_seconds * sampling_rate_hz)))
        right = min(valid.shape[0], coarse_rel + int(round(post_seconds * sampling_rate_hz)) + 1)

    if feature == "vertical":
        char = np.abs(valid[:, 2])
    elif feature == "norm":
        char = np.linalg.norm(valid, axis=1)
    else:
        raise ValueError(f"Unsupported feature: {feature}")

    sta_n = max(1, int(round(sta_seconds * sampling_rate_hz)))
    lta_n = max(sta_n + 1, int(round(lta_seconds * sampling_rate_hz)))
    sta = _moving_average_same(char, sta_n)
    lta = _moving_average_same(char, lta_n)
    ratio = sta / np.maximum(lta, 5e-4)
    ratio[: max(lta_n, left)] = 0.0

    local_ratio = ratio[left:right]
    if local_ratio.size == 0:
        refined_rel = coarse_rel
        method = "fallback_empty_window"
    else:
        above = np.flatnonzero(local_ratio >= threshold_ratio)
        if above.size:
            refined_rel = left + int(above[0])
            method = "stalta_threshold"
        else:
            refined_rel = left + int(np.argmax(local_ratio))
            method = "stalta_argmax"

    return {
        "refined_pick_aligned": int(start + refined_rel),
        "search_left_aligned": int(start + left),
        "search_right_aligned": int(start + max(left, right - 1)),
        "ratio_peak": float(np.max(local_ratio)) if local_ratio.size else float("nan"),
        "ratio_at_pick": float(ratio[refined_rel]) if ratio.size else float("nan"),
        "method": method,
        "search_feature": feature,
        "coarse_pick_aligned": int(coarse_pick_aligned),
    }


def refine_event_p_picks_from_arrays(
    event_df: pd.DataFrame,
    waveforms: np.ndarray,
    record_start_sample: np.ndarray,
    valid_n_samples: np.ndarray,
    coarse_pick_col: str = "p_pick_repaired_aligned",
    pre_seconds: float = 4.0,
    post_seconds: float = 1.0,
    sta_seconds: float = 0.2,
    lta_seconds: float = 1.0,
    threshold_ratio: float = 0.2,
    feature: str = "vertical",
    search_left_col: str | None = None,
    search_right_col: str | None = None,
) -> pd.DataFrame:
    if event_df.empty:
        raise ValueError("event_df must not be empty")

    df = event_df.copy().reset_index(drop=True)
    sampling_rate_hz = float(df["sampling_rate_hz"].iloc[0])

    refined_rows = []
    for _, row in df.iterrows():
        wave_idx = int(row["wave_idx"])
        search_left = int(row[search_left_col]) if search_left_col and search_left_col in df.columns and pd.notna(row[search_left_col]) else None
        search_right = int(row[search_right_col]) if search_right_col and search_right_col in df.columns and pd.notna(row[search_right_col]) else None
        refined = refine_pick_stalta(
            waveform_aligned=waveforms[wave_idx],
            coarse_pick_aligned=int(row[coarse_pick_col]),
            sampling_rate_hz=sampling_rate_hz,
            record_start_sample=int(record_start_sample[wave_idx]),
            valid_n_samples=int(valid_n_samples[wave_idx]),
            search_left_aligned=search_left,
            search_right_aligned=search_right,
            pre_seconds=pre_seconds,
            post_seconds=post_seconds,
            sta_seconds=sta_seconds,
            lta_seconds=lta_seconds,
            threshold_ratio=threshold_ratio,
            feature=feature,
        )
        refined_rows.append(refined)

    refined_df = pd.DataFrame(refined_rows)
    for col in refined_df.columns:
        df[f"stalta_{col}"] = refined_df[col].values
    return df


def refine_event_p_picks_from_hdf5(
    hdf5_path: Path,
    event_df: pd.DataFrame,
    coarse_pick_col: str = "p_pick_repaired_aligned",
    pre_seconds: float = 4.0,
    post_seconds: float = 1.0,
    sta_seconds: float = 0.2,
    lta_seconds: float = 1.0,
    threshold_ratio: float = 0.2,
    feature: str = "vertical",
) -> pd.DataFrame:
    if event_df.empty:
        raise ValueError("event_df must not be empty")

    event_id = str(event_df["EVENT"].iloc[0])
    with h5py.File(hdf5_path, "r") as f:
        grp = f["data"][event_id]
        waveforms = grp["waveforms"][()]
        record_start_sample = grp["record_start_sample"][()]
        valid_n_samples = grp["valid_n_samples"][()]

    return refine_event_p_picks_from_arrays(
        event_df=event_df,
        waveforms=waveforms,
        record_start_sample=record_start_sample,
        valid_n_samples=valid_n_samples,
        coarse_pick_col=coarse_pick_col,
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
        sta_seconds=sta_seconds,
        lta_seconds=lta_seconds,
        threshold_ratio=threshold_ratio,
        feature=feature,
    )
