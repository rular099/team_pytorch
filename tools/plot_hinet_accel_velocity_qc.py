#!/usr/bin/env python3
"""Plot K-NET/KiK-net acceleration against matched Hi-net velocity windows.

The plot is anchored on the theoretical P arrival recorded by
``tools/download_hinet_velocity.py``.  Training-data picks are converted back to
absolute station time and drawn on the same axis so station-clock offsets are
visible directly.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shlex
import struct
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
JST = timezone(timedelta(hours=9))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/codex-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


ACC_COMPONENT_INDEX = {
    "vertical": 2,
    "ns": 0,
    "ew": 1,
}
HINET_COMPONENT_BY_MODE = {
    "vertical": ("U", "Z"),
    "ns": ("N", "1"),
    "ew": ("E", "2"),
}
PICK_COLUMNS = [
    "p_picks",
    "p_pick_trigger_aligned",
    "p_pick_predicted_aligned",
    "p_pick_repaired_aligned",
    "stalta_refined_pick_aligned",
    "p_pick_diting_acc_aligned",
    "p_pick_diting_vel_aligned",
    "p_pick_refined_aligned",
    "pga_norm_aligned_loc",
]
PICK_LABELS = {
    "p_picks": "p_picks",
    "p_pick_trigger_aligned": "trigger",
    "p_pick_predicted_aligned": "travel_pred",
    "p_pick_repaired_aligned": "travel_coarse",
    "stalta_refined_pick_aligned": "stalta",
    "p_pick_diting_acc_aligned": "diting_acc",
    "p_pick_diting_vel_aligned": "diting_vel",
    "p_pick_refined_aligned": "final",
    "pga_norm_aligned_loc": "pga",
}
PICK_COLORS = {
    "p_picks": "black",
    "p_pick_trigger_aligned": "0.45",
    "p_pick_predicted_aligned": "tab:cyan",
    "p_pick_repaired_aligned": "tab:blue",
    "stalta_refined_pick_aligned": "tab:orange",
    "p_pick_diting_acc_aligned": "tab:red",
    "p_pick_diting_vel_aligned": "tab:purple",
    "p_pick_refined_aligned": "black",
    "pga_norm_aligned_loc": "tab:olive",
}
SPAN_COLUMNS = [
    ("p_pick_search_left_aligned", "p_pick_search_right_aligned", "pick_search", "tab:blue"),
    ("p_pick_search_raw_left_aligned", "p_pick_search_raw_right_aligned", "raw_search", "tab:cyan"),
    ("stalta_search_left_aligned", "stalta_search_right_aligned", "stalta_search", "tab:orange"),
]


@dataclass
class TrainingStation:
    event_id: str
    wave_idx: int
    station_code: str
    sensor_class: str
    height_m: float
    sampling_rate_hz: float
    record_start_sample: int
    valid_n_samples: int
    record_start_timestamp: float
    valid_start_rel_seconds: float
    valid_end_rel_seconds: float
    acceleration_t_rel: np.ndarray
    acceleration: np.ndarray
    pick_rel_seconds: dict[str, float]
    span_rel_seconds: dict[str, tuple[float, float]]
    summary: dict[str, object]


@dataclass
class VelocitySeries:
    t_rel: np.ndarray
    values: np.ndarray
    source: str
    status: str
    label: str
    path: str
    trace_count: int


def decode_array(values) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind == "S":
        return np.asarray([x.decode("utf-8", errors="replace") for x in arr])
    return arr


def finite_float(value, default=np.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def parse_jst_timestamp(value: object) -> float:
    text = str(value).strip()
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=JST)
    return dt.timestamp()


def parse_download_script(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace").replace("\\\n", " ")
    try:
        tokens = shlex.split(text)
    except ValueError:
        return {}
    out: dict[str, str] = {}
    for i, token in enumerate(tokens[:-1]):
        if token in {"--hdf5", "--output-root"}:
            out[token] = tokens[i + 1]
    return out


def resolve_inputs(args: argparse.Namespace) -> None:
    parsed = parse_download_script(args.download_script)
    if args.hdf5 is None and "--hdf5" in parsed:
        args.hdf5 = Path(parsed["--hdf5"])
    if args.download_root is None and "--output-root" in parsed:
        args.download_root = Path(parsed["--output-root"])
    if args.hdf5 is None:
        raise SystemExit("Provide --hdf5 or keep it in --download-script.")
    if args.download_root is None:
        raise SystemExit("Provide --download-root or keep --output-root in --download-script.")
    args.hdf5 = args.hdf5.expanduser().resolve()
    args.download_root = args.download_root.expanduser().resolve()
    args.output_dir = (args.output_dir or (REPO_ROOT / "hinet_velocity_qc")).expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.raw_search_dir = [p.expanduser().resolve() for p in args.raw_search_dir]


def load_manifest(download_root: Path) -> pd.DataFrame:
    manifest_path = download_root / "manifests" / "download_manifest.csv"
    if manifest_path.exists():
        df = pd.read_csv(manifest_path, dtype={"event_id": str, "knet_station": str, "hinet_station": str})
    else:
        event_manifests = sorted((download_root / "manifests" / "events").glob("*/download_manifest.csv"))
        if not event_manifests:
            raise FileNotFoundError(f"No download manifest found under {download_root}")
        df = pd.concat((pd.read_csv(p, dtype={"event_id": str, "knet_station": str, "hinet_station": str}) for p in event_manifests), ignore_index=True)
    subset = [c for c in ("event_id", "knet_station", "knet_height_m", "hinet_station", "ppick_timestamp") if c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    return df


def select_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    if args.event_id:
        out = out[out["event_id"].astype(str) == str(args.event_id)]
    if args.station:
        station = str(args.station)
        mask = out["knet_station"].astype(str).eq(station)
        if "hinet_station" in out.columns:
            mask = mask | out["hinet_station"].astype(str).eq(station)
        out = out[mask]
    if out.empty:
        raise SystemExit("No manifest rows matched the requested event/station filters.")
    if args.prefer_written_mseed and "mseed_status" in out.columns:
        out = out.assign(_mseed_rank=out["mseed_status"].astype(str).isin(["written", "partial_written"]).astype(int))
        out = out.sort_values(["_mseed_rank", "event_id", "knet_station"], ascending=[False, True, True]).drop(columns=["_mseed_rank"])
    if args.max_plots is not None and args.max_plots > 0:
        out = out.head(args.max_plots)
    return out.reset_index(drop=True)


def station_index_for_row(group: h5py.Group, row: pd.Series, wave_idx: int | None) -> tuple[int, str, float, str]:
    n = int(group["waveforms"].shape[0])
    if wave_idx is not None:
        if wave_idx < 0 or wave_idx >= n:
            raise IndexError(f"wave_idx={wave_idx} is outside [0, {n})")
        codes = decode_array(group["station_codes"][()]) if "station_codes" in group else np.asarray([""] * n)
        sensors = decode_array(group["sensor_class"][()]) if "sensor_class" in group else np.asarray([""] * n)
        coords = np.asarray(group["coords"][()]) if "coords" in group else np.full((n, 3), np.nan)
        return wave_idx, str(codes[wave_idx]), float(coords[wave_idx, 2] * 1000.0), str(sensors[wave_idx])

    station = str(row["knet_station"])
    codes = decode_array(group["station_codes"][()]) if "station_codes" in group else np.asarray([""] * n)
    candidates = np.flatnonzero(codes.astype(str) == station)
    if candidates.size == 0:
        raise KeyError(f"station {station!r} not found in HDF5 event {group.name}")
    coords = np.asarray(group["coords"][()]) if "coords" in group else np.full((n, 3), np.nan)
    sensors = decode_array(group["sensor_class"][()]) if "sensor_class" in group else np.asarray([""] * n)
    if candidates.size == 1:
        idx = int(candidates[0])
    else:
        target_height = finite_float(row.get("knet_height_m"), default=np.nan)
        if np.isfinite(target_height) and coords.shape[1] >= 3:
            heights = coords[candidates, 2] * 1000.0
            idx = int(candidates[int(np.nanargmin(np.abs(heights - target_height)))])
        else:
            idx = int(candidates[0])
    return idx, str(codes[idx]), float(coords[idx, 2] * 1000.0), str(sensors[idx])


def sample_to_rel_seconds(sample: float, record_start_sample: int, record_start_ts: float, sr: float, ppick_ts: float) -> float:
    return record_start_ts + (float(sample) - float(record_start_sample)) / float(sr) - float(ppick_ts)


def load_training_station(h5: h5py.File, row: pd.Series, args: argparse.Namespace) -> TrainingStation:
    event_id = str(row["event_id"])
    if event_id not in h5["data"]:
        raise KeyError(f"event {event_id!r} not found in HDF5")
    group = h5["data"][event_id]
    wave_idx, station_code, height_m, sensor_class = station_index_for_row(group, row, args.wave_idx)

    sr = float(h5["metadata"]["sampling_rate"][()]) if "sampling_rate" in h5["metadata"] else 100.0
    waveform_shape = group["waveforms"].shape
    record_start_sample = int(group["record_start_sample"][wave_idx]) if "record_start_sample" in group else 0
    valid_n_samples = int(group["valid_n_samples"][wave_idx]) if "valid_n_samples" in group else int(waveform_shape[1])
    if "record_start_time_jst" in group:
        record_start_ts = parse_jst_timestamp(decode_array(group["record_start_time_jst"][()])[wave_idx])
    else:
        record_start_ts = float(row["origin_timestamp"]) - float(group["p_picks"][wave_idx] - record_start_sample) / sr

    ppick_ts = float(row["ppick_timestamp"])
    window_start = ppick_ts - args.pre_seconds
    window_end = ppick_ts + args.post_seconds
    left = math.floor(record_start_sample + (window_start - record_start_ts) * sr)
    right = math.ceil(record_start_sample + (window_end - record_start_ts) * sr)
    left = max(0, min(int(waveform_shape[1]), left))
    right = max(left + 1, min(int(waveform_shape[1]), right))

    waveform = np.asarray(group["waveforms"][wave_idx, left:right, :], dtype=np.float64)
    if args.component == "norm":
        acc = np.linalg.norm(waveform, axis=1)
    else:
        acc = waveform[:, ACC_COMPONENT_INDEX[args.component]]
    samples = np.arange(left, right)
    t_rel = record_start_ts + (samples - record_start_sample) / sr - ppick_ts

    picks: dict[str, float] = {}
    for col in PICK_COLUMNS:
        if col in group:
            try:
                picks[col] = sample_to_rel_seconds(float(group[col][wave_idx]), record_start_sample, record_start_ts, sr, ppick_ts)
            except Exception:
                pass
    spans: dict[str, tuple[float, float]] = {}
    for left_col, right_col, label, _ in SPAN_COLUMNS:
        if left_col in group and right_col in group:
            x0 = sample_to_rel_seconds(float(group[left_col][wave_idx]), record_start_sample, record_start_ts, sr, ppick_ts)
            x1 = sample_to_rel_seconds(float(group[right_col][wave_idx]), record_start_sample, record_start_ts, sr, ppick_ts)
            spans[label] = (min(x0, x1), max(x0, x1))

    valid_start_rel = sample_to_rel_seconds(record_start_sample, record_start_sample, record_start_ts, sr, ppick_ts)
    valid_end_rel = sample_to_rel_seconds(record_start_sample + valid_n_samples - 1, record_start_sample, record_start_ts, sr, ppick_ts)
    summary = {
        "event_id": event_id,
        "knet_station": str(row["knet_station"]),
        "hinet_station": str(row.get("hinet_station", "")),
        "wave_idx": wave_idx,
        "station_code": station_code,
        "sensor_class": sensor_class,
        "station_height_m_hdf5": height_m,
        "station_height_m_manifest": finite_float(row.get("knet_height_m"), default=np.nan),
        "record_start_time_jst_hdf5": datetime.fromtimestamp(record_start_ts, tz=JST).isoformat(),
        "theoretical_p_time_jst": datetime.fromtimestamp(ppick_ts, tz=JST).isoformat(),
        "valid_start_minus_theoretical_p_s": valid_start_rel,
        "valid_end_minus_theoretical_p_s": valid_end_rel,
    }
    for col, rel in picks.items():
        summary[f"{col}_minus_theoretical_p_s"] = rel
    return TrainingStation(
        event_id=event_id,
        wave_idx=wave_idx,
        station_code=station_code,
        sensor_class=sensor_class,
        height_m=height_m,
        sampling_rate_hz=sr,
        record_start_sample=record_start_sample,
        valid_n_samples=valid_n_samples,
        record_start_timestamp=record_start_ts,
        valid_start_rel_seconds=valid_start_rel,
        valid_end_rel_seconds=valid_end_rel,
        acceleration_t_rel=t_rel,
        acceleration=acc,
        pick_rel_seconds=picks,
        span_rel_seconds=spans,
        summary=summary,
    )


def candidate_mseed_paths(row: pd.Series, download_root: Path) -> list[Path]:
    candidates: list[Path] = []
    value = row.get("mseed_path")
    if value is not None and not pd.isna(value) and str(value).strip():
        candidates.append(Path(str(value)).expanduser())
    event_id = str(row["event_id"])
    knet = str(row["knet_station"]).replace("/", "_")
    hinet = str(row["hinet_station"]).replace("/", "_")
    candidates.append(download_root / "mseed" / event_id / f"{knet}__{hinet}.mseed")
    candidates.extend(sorted((download_root / "mseed" / event_id).glob(f"{knet}__*.mseed")))
    seen: set[Path] = set()
    out = []
    for path in candidates:
        path = path.expanduser()
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            out.append(path)
    return out


def traces_to_series(stream, ppick_ts: float, start_ts: float, end_ts: float, component: str) -> VelocitySeries:
    stream = stream.copy()
    try:
        from obspy import UTCDateTime

        stream.trim(UTCDateTime(start_ts), UTCDateTime(end_ts), pad=False, nearest_sample=False)
    except Exception:
        pass
    traces = [tr for tr in stream if getattr(tr.stats, "npts", 0) > 0]
    if not traces:
        return VelocitySeries(np.array([]), np.array([]), "mseed", "empty_after_trim", "Hi-net MiniSEED", "", 0)
    if component == "norm":
        base = max(traces, key=lambda tr: tr.stats.npts)
        t_abs = np.asarray(base.times("timestamp"), dtype=np.float64)
        vals = []
        for tr in traces:
            tt = np.asarray(tr.times("timestamp"), dtype=np.float64)
            yy = np.asarray(tr.data, dtype=np.float64)
            if tt.size > 1 and yy.size > 1:
                vals.append(np.interp(t_abs, tt, yy, left=np.nan, right=np.nan))
        if not vals:
            return VelocitySeries(np.array([]), np.array([]), "mseed", "empty_norm", "Hi-net MiniSEED norm", "", len(traces))
        arr = np.vstack(vals)
        values = np.sqrt(np.nansum(arr * arr, axis=0))
        return VelocitySeries(t_abs - ppick_ts, values, "mseed", "loaded", "Hi-net MiniSEED norm counts", "", len(traces))
    wanted_suffixes = HINET_COMPONENT_BY_MODE[component]
    selected = None
    for tr in traces:
        channel = str(getattr(tr.stats, "channel", "")).upper()
        if channel.endswith(("Z", "U")) and any(channel.endswith(s) for s in ("Z", "U")) and component == "vertical":
            selected = tr
            break
        if component == "ns" and channel.endswith(("N", "1")):
            selected = tr
            break
        if component == "ew" and channel.endswith(("E", "2")):
            selected = tr
            break
    if selected is None:
        for tr in traces:
            channel = str(getattr(tr.stats, "channel", "")).upper()
            if any(channel.endswith(s) for s in wanted_suffixes):
                selected = tr
                break
    if selected is None:
        selected = traces[0]
    t_abs = np.asarray(selected.times("timestamp"), dtype=np.float64)
    values = np.asarray(selected.data, dtype=np.float64)
    label = f"Hi-net MiniSEED {getattr(selected.stats, 'channel', '')} counts"
    return VelocitySeries(t_abs - ppick_ts, values, "mseed", "loaded", label, "", len(traces))


def load_mseed_velocity(row: pd.Series, args: argparse.Namespace) -> VelocitySeries | None:
    paths = candidate_mseed_paths(row, args.download_root)
    if not paths:
        return None
    ppick_ts = float(row["ppick_timestamp"])
    start_ts = ppick_ts - args.pre_seconds
    end_ts = ppick_ts + args.post_seconds
    try:
        from obspy import read
    except Exception as exc:
        return VelocitySeries(np.array([]), np.array([]), "mseed", f"obspy_import_failed:{exc!r}", "Hi-net MiniSEED", str(paths[0]), 0)
    for path in paths:
        try:
            series = traces_to_series(read(str(path)), ppick_ts, start_ts, end_ts, args.component)
            series.path = str(path)
            if series.t_rel.size:
                return series
        except Exception as exc:
            last = VelocitySeries(np.array([]), np.array([]), "mseed", f"read_failed:{exc!r}", "Hi-net MiniSEED", str(path), 0)
    return last if "last" in locals() else None


def parse_channel_table_text(ch_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for raw in ch_path.read_text(errors="ignore").splitlines():
        parts = raw.split()
        if len(parts) < 5 or parts[0].startswith("#"):
            continue
        station = parts[3] if "." in parts[3] else parts[1]
        component = parts[4] if "." in parts[3] else parts[2]
        rows.append({
            "channel_id": str(parts[0]).lower(),
            "hinet_station": str(station),
            "component": str(component).upper(),
            "raw_line": raw,
        })
    return pd.DataFrame(rows)


def bcd_to_int(value: int) -> int:
    return (value >> 4) * 10 + (value & 0x0F)


def parse_hinet_vm_timestamp(buf: bytes) -> float:
    if len(buf) < 8:
        raise ValueError("timestamp buffer too short")
    year = bcd_to_int(buf[0]) * 100 + bcd_to_int(buf[1])
    dt = datetime(
        year,
        bcd_to_int(buf[2]),
        bcd_to_int(buf[3]),
        bcd_to_int(buf[4]),
        bcd_to_int(buf[5]),
        bcd_to_int(buf[6]),
        tzinfo=JST,
    )
    return dt.timestamp()


def decode_win_diffs(first: int, encoded: bytes, datawide: float, srate: int) -> np.ndarray:
    if srate <= 0:
        return np.asarray([], dtype=np.float64)
    values = np.empty(srate, dtype=np.float64)
    values[0] = first
    if datawide == 0.5:
        idx = 1
        previous = first
        for i, byte in enumerate(encoded):
            high = byte >> 4
            if high & 0x8:
                high -= 0x10
            previous += high
            if idx < srate:
                values[idx] = previous
                idx += 1
            low = byte & 0x0F
            if low & 0x8:
                low -= 0x10
            previous += low
            if i == len(encoded) - 1 and srate % 2 == 0:
                break
            if idx < srate:
                values[idx] = previous
                idx += 1
        return values[:idx]
    if datawide == 1:
        diffs = np.frombuffer(encoded, dtype=np.int8).astype(np.int64)
    elif datawide == 2:
        diffs = np.frombuffer(encoded, dtype=">i2").astype(np.int64)
    elif datawide == 3:
        diffs = np.empty(len(encoded) // 3, dtype=np.int64)
        for i in range(diffs.size):
            raw = int.from_bytes(encoded[3 * i:3 * i + 3], "big", signed=False)
            if raw & 0x800000:
                raw -= 0x1000000
            diffs[i] = raw
    elif datawide == 4:
        diffs = np.frombuffer(encoded, dtype=">i4").astype(np.int64)
    else:
        raise NotImplementedError(f"unsupported WIN32 data width: {datawide}")
    n = min(diffs.size + 1, srate)
    if n > 1:
        values[1:n] = first + np.cumsum(diffs[: n - 1])
    return values[:n]


def read_hinet_vm_cnt(paths: Iterable[Path], channel_ids: set[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    values_by_channel: dict[str, list[np.ndarray]] = {channel_id: [] for channel_id in channel_ids}
    times_by_channel: dict[str, list[np.ndarray]] = {channel_id: [] for channel_id in channel_ids}
    for path in sorted(paths):
        data = path.read_bytes()
        offset = 4 if len(data) > 20 and data[:4] == b"\x00\x00\x00\x00" else 0
        while offset + 16 <= len(data):
            try:
                record_ts = parse_hinet_vm_timestamp(data[offset:offset + 8])
            except Exception:
                break
            payload_len = struct.unpack(">i", data[offset + 12:offset + 16])[0]
            payload_start = offset + 16
            payload_end = payload_start + payload_len
            if payload_len <= 0 or payload_end > len(data):
                break
            cursor = payload_start
            while cursor + 10 <= payload_end:
                if data[cursor:cursor + 2] in (b"\x01\x01", b"\x01\x03"):
                    cursor += 2
                channel_id = f"{data[cursor]:02x}{data[cursor + 1]:02x}"
                raw_width = data[cursor + 2] >> 4
                srate = int(data[cursor + 3])
                cursor += 4
                datawide = 0.5 if raw_width == 0 else float(raw_width)
                encoded_len = srate // 2 if raw_width == 0 else (srate - 1) * raw_width
                if cursor + 4 + encoded_len > payload_end:
                    break
                first = struct.unpack(">i", data[cursor:cursor + 4])[0]
                cursor += 4
                encoded = data[cursor:cursor + encoded_len]
                cursor += encoded_len
                if channel_id not in channel_ids:
                    continue
                decoded = decode_win_diffs(first, encoded, datawide, srate)
                if decoded.size == 0:
                    continue
                values_by_channel[channel_id].append(decoded)
                times_by_channel[channel_id].append(record_ts + np.arange(decoded.size, dtype=np.float64) / float(srate))
            offset = payload_end

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for channel_id in channel_ids:
        if values_by_channel[channel_id]:
            t = np.concatenate(times_by_channel[channel_id])
            y = np.concatenate(values_by_channel[channel_id])
            order = np.argsort(t)
            out[channel_id] = (t[order], y[order])
    return out


def raw_search_dirs(args: argparse.Namespace, event_id: str) -> list[Path]:
    dirs = [
        args.download_root / "raw" / event_id,
        args.download_root / "raw",
        args.download_root,
        REPO_ROOT,
        Path.cwd(),
    ]
    dirs.extend(args.raw_search_dir)
    out = []
    seen = set()
    for path in dirs:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            out.append(path)
    return out


def filename_minute_timestamp(path: Path) -> float | None:
    match = re.match(r"(?P<ymdhm>\d{12})", path.name)
    if not match:
        return None
    try:
        dt = datetime.strptime(match.group("ymdhm"), "%Y%m%d%H%M").replace(tzinfo=JST)
        return dt.timestamp()
    except ValueError:
        return None


def candidate_raw_files(row: pd.Series, args: argparse.Namespace) -> tuple[list[Path], Path | None]:
    event_id = str(row["event_id"])
    ppick_ts = float(row["ppick_timestamp"])
    window_start = ppick_ts - args.pre_seconds
    window_end = ppick_ts + args.post_seconds
    cnts: list[Path] = []
    ch_files: list[Path] = []
    for directory in raw_search_dirs(args, event_id):
        cnts.extend(directory.glob(f"{event_id}_*.cnt"))
        cnts.extend(directory.glob(f"{event_id[:12]}*.cnt"))
        cnts.extend(directory.glob(f"{event_id[:8]}*.cnt"))
        ch_files.extend(directory.glob(f"*{event_id[:8]}*.ch"))
        ch_files.extend(directory.glob("*.ch"))
    unique_cnts = []
    seen_cnt = set()
    for path in cnts:
        if path in seen_cnt or not path.exists():
            continue
        seen_cnt.add(path)
        minute_ts = filename_minute_timestamp(path)
        if minute_ts is not None and (minute_ts > window_end or minute_ts + 60.0 < window_start):
            continue
        unique_cnts.append(path)
    unique_ch = []
    seen_ch = set()
    for path in ch_files:
        if path in seen_ch or not path.exists():
            continue
        seen_ch.add(path)
        unique_ch.append(path)
    dated = [p for p in unique_ch if event_id[:8] in p.name]
    ch_path = sorted(dated or unique_ch)[0] if (dated or unique_ch) else None
    return sorted(unique_cnts), ch_path


def channel_ids_for_station(ch_path: Path, hinet_station: str, component: str) -> dict[str, str]:
    ctable = parse_channel_table_text(ch_path)
    if ctable.empty:
        return {}
    station_rows = ctable[ctable["hinet_station"].astype(str).str.upper() == str(hinet_station).upper()]
    if station_rows.empty:
        return {}
    if component == "norm":
        wanted = ("U", "Z", "N", "1", "E", "2")
    else:
        wanted = HINET_COMPONENT_BY_MODE[component]
    rows = station_rows[station_rows["component"].astype(str).str.upper().isin(wanted)]
    out: dict[str, str] = {}
    for _, row in rows.iterrows():
        comp = str(row["component"]).upper()
        out[comp] = str(row["channel_id"]).lower()
    return out


def load_raw_velocity(row: pd.Series, args: argparse.Namespace) -> VelocitySeries | None:
    cnt_paths, ch_path = candidate_raw_files(row, args)
    if not cnt_paths or ch_path is None:
        return None
    hinet_station = str(row["hinet_station"])
    channel_map = channel_ids_for_station(ch_path, hinet_station, args.component)
    if not channel_map:
        return VelocitySeries(np.array([]), np.array([]), "raw_win32", "channel_not_found", "Hi-net raw WIN32", str(ch_path), 0)

    selected_ids = set(channel_map.values())
    try:
        series_by_id = read_hinet_vm_cnt(cnt_paths, selected_ids)
    except Exception as exc:
        return VelocitySeries(np.array([]), np.array([]), "raw_win32", f"read_failed:{exc!r}", "Hi-net raw WIN32", ",".join(str(p) for p in cnt_paths), 0)
    if not series_by_id:
        return VelocitySeries(np.array([]), np.array([]), "raw_win32", "no_selected_channel_samples", "Hi-net raw WIN32", ",".join(str(p) for p in cnt_paths), 0)

    ppick_ts = float(row["ppick_timestamp"])
    start_rel = -float(args.pre_seconds)
    end_rel = float(args.post_seconds)
    if args.component == "norm":
        arrays = []
        base_t = None
        for channel_id, (t_abs, y) in series_by_id.items():
            if base_t is None or t_abs.size > base_t.size:
                base_t = t_abs
        if base_t is None:
            return VelocitySeries(np.array([]), np.array([]), "raw_win32", "empty_norm", "Hi-net raw WIN32 norm counts", "", len(series_by_id))
        for t_abs, y in series_by_id.values():
            arrays.append(np.interp(base_t, t_abs, y, left=np.nan, right=np.nan))
        values = np.sqrt(np.nansum(np.vstack(arrays) ** 2, axis=0))
        t_rel = base_t - ppick_ts
        mask = (t_rel >= start_rel) & (t_rel <= end_rel)
        return VelocitySeries(t_rel[mask], values[mask], "raw_win32", "loaded", "Hi-net raw WIN32 norm counts", ",".join(str(p) for p in cnt_paths), len(series_by_id))

    preferred = HINET_COMPONENT_BY_MODE[args.component]
    selected_channel = None
    for comp in preferred:
        if comp in channel_map and channel_map[comp] in series_by_id:
            selected_channel = channel_map[comp]
            break
    if selected_channel is None:
        selected_channel = next(iter(series_by_id.keys()))
    t_abs, values = series_by_id[selected_channel]
    t_rel = t_abs - ppick_ts
    mask = (t_rel >= start_rel) & (t_rel <= end_rel)
    label = f"Hi-net raw WIN32 {hinet_station} {selected_channel} counts"
    return VelocitySeries(t_rel[mask], values[mask], "raw_win32", "loaded", label, ",".join(str(p) for p in cnt_paths), 1)


def load_velocity(row: pd.Series, args: argparse.Namespace) -> VelocitySeries:
    mseed = load_mseed_velocity(row, args)
    if mseed is not None and mseed.t_rel.size:
        return mseed
    raw = load_raw_velocity(row, args)
    if raw is not None and raw.t_rel.size:
        return raw
    if raw is not None:
        return raw
    if mseed is not None:
        return mseed
    return VelocitySeries(np.array([]), np.array([]), "none", "missing_velocity", "Hi-net velocity unavailable", "", 0)


def add_marks(ax, picks: dict[str, float], spans: dict[str, tuple[float, float]]) -> None:
    span_colors = {
        "pick_search": "tab:blue",
        "raw_search": "tab:cyan",
        "stalta_search": "tab:orange",
    }
    for label, (x0, x1) in spans.items():
        color = span_colors.get(label, "0.5")
        ax.axvspan(x0, x1, color=color, alpha=0.10, label=label)
    for col, x in picks.items():
        if not np.isfinite(x):
            continue
        ax.axvline(x, color=PICK_COLORS.get(col, "0.2"), linewidth=1.0, alpha=0.9, label=PICK_LABELS.get(col, col))


def add_invalid_acceleration_shading(ax, training: TrainingStation, args: argparse.Namespace) -> None:
    left_edge = -float(args.pre_seconds)
    right_edge = float(args.post_seconds)
    if training.valid_start_rel_seconds > left_edge:
        ax.axvspan(left_edge, min(training.valid_start_rel_seconds, right_edge), color="0.85", alpha=0.45, label="no_acc_record")
    if training.valid_end_rel_seconds < right_edge:
        ax.axvspan(max(training.valid_end_rel_seconds, left_edge), right_edge, color="0.85", alpha=0.45, label="no_acc_record")


def dedupe_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, ncol=4, loc="upper right")


def safe_slug(*parts: object) -> str:
    text = "__".join(str(p) for p in parts if str(p))
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def plot_pair(training: TrainingStation, velocity: VelocitySeries, row: pd.Series, args: argparse.Namespace) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.2), sharex=True)
    axes[0].plot(training.acceleration_t_rel, training.acceleration, color="0.20", linewidth=0.8)
    add_invalid_acceleration_shading(axes[0], training, args)
    axes[0].axvline(0.0, color="tab:green", linewidth=1.3, label="theoretical_P")
    add_marks(axes[0], training.pick_rel_seconds, training.span_rel_seconds)
    axes[0].set_ylabel("K-NET/KiK-net acc. m/s^2" if args.component != "norm" else "K-NET/KiK-net |acc.| m/s^2")
    dedupe_legend(axes[0])

    if velocity.t_rel.size:
        axes[1].plot(velocity.t_rel, velocity.values, color="0.20", linewidth=0.8)
    else:
        axes[1].text(0.5, 0.5, f"Velocity unavailable: {velocity.status}", ha="center", va="center", transform=axes[1].transAxes, fontsize=10)
    axes[1].axvline(0.0, color="tab:green", linewidth=1.3, label="theoretical_P")
    add_marks(axes[1], training.pick_rel_seconds, training.span_rel_seconds)
    axes[1].set_ylabel(velocity.label)
    axes[1].set_xlabel("seconds relative to theoretical P arrival")
    axes[1].set_xlim(-args.pre_seconds, args.post_seconds)
    dedupe_legend(axes[1])

    final_offset = training.pick_rel_seconds.get("p_picks", np.nan)
    match_distance = finite_float(row.get("match_distance_km"), default=np.nan)
    epi_distance = finite_float(row.get("epicentral_distance_km"), default=np.nan)
    title = (
        f"{training.event_id} {training.station_code}->"
        f"{row.get('hinet_station', '')} wave_idx={training.wave_idx} "
        f"{training.sensor_class} h={training.height_m:.1f}m "
        f"epi={epi_distance:.1f}km match={match_distance:.3f}km "
        f"p_pick-P={final_offset:.2f}s vel={velocity.source}:{velocity.status}"
    )
    axes[0].set_title(title, fontsize=10, loc="left")
    for ax in axes:
        ax.grid(True, axis="x", alpha=0.18)
    fig.tight_layout()
    filename = safe_slug(training.event_id, training.station_code, f"wave{training.wave_idx}", row.get("hinet_station", ""), args.component) + ".png"
    out_path = args.output_dir / filename
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    return out_path


def process(args: argparse.Namespace) -> pd.DataFrame:
    manifest = select_rows(load_manifest(args.download_root), args)
    rows = []
    with h5py.File(args.hdf5, "r") as h5:
        for _, row in manifest.iterrows():
            summary = dict(row)
            try:
                training = load_training_station(h5, row, args)
                velocity = load_velocity(row, args)
                plot_path = plot_pair(training, velocity, row, args)
                summary.update(training.summary)
                summary.update({
                    "velocity_source": velocity.source,
                    "velocity_status": velocity.status,
                    "velocity_path": velocity.path,
                    "velocity_trace_count": velocity.trace_count,
                    "plot_path": str(plot_path),
                    "plot_status": "written" if velocity.t_rel.size else "written_missing_velocity",
                })
                final_offset = training.pick_rel_seconds.get("p_picks", np.nan)
                summary["large_pick_offset"] = int(np.isfinite(final_offset) and abs(final_offset) >= args.large_offset_seconds)
                summary["possible_timezone_offset"] = int(np.isfinite(final_offset) and min(abs(final_offset - 9 * 3600), abs(final_offset + 9 * 3600)) <= 120)
            except Exception as exc:
                summary.update({
                    "plot_status": "failed",
                    "plot_error": repr(exc),
                    "plot_path": "",
                })
            rows.append(summary)
    summary_df = pd.DataFrame(rows)
    summary_path = args.output_dir / "qc_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    return summary_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="QC plot for matched Hi-net velocity and training acceleration waveforms."
    )
    parser.add_argument("--hdf5", type=Path, default=None, help="Training HDF5. Defaults to --hdf5 parsed from download_hinet.sh.")
    parser.add_argument("--download-root", type=Path, default=None, help="Hi-net downloader output root. Defaults to --output-root parsed from download_hinet.sh.")
    parser.add_argument("--download-script", type=Path, default=REPO_ROOT / "download_hinet.sh", help="Shell script used to infer --hdf5 and --download-root.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for PNG plots and qc_summary.csv. Defaults to team_pytorch/hinet_velocity_qc.")
    parser.add_argument("--event-id", default=None, help="Optional event id filter.")
    parser.add_argument("--station", default=None, help="Optional K-NET/KiK-net or Hi-net station filter.")
    parser.add_argument("--wave-idx", type=int, default=None, help="Force a specific station index in the HDF5 event.")
    parser.add_argument("--max-plots", type=int, default=40, help="Maximum manifest rows to plot. Set <=0 for all selected rows.")
    parser.add_argument("--pre-seconds", type=float, default=50.0, help="Seconds before theoretical P arrival to show.")
    parser.add_argument("--post-seconds", type=float, default=50.0, help="Seconds after theoretical P arrival to show.")
    parser.add_argument("--component", choices=["vertical", "ns", "ew", "norm"], default="vertical")
    parser.add_argument("--large-offset-seconds", type=float, default=20.0, help="Flag p_picks offsets larger than this threshold.")
    parser.add_argument("--prefer-written-mseed", action="store_true", help="In batch mode, plot rows with MiniSEED first.")
    parser.add_argument("--raw-search-dir", type=Path, action="append", default=[], help="Extra directory containing raw *.cnt/*.ch files.")
    parser.add_argument("--dpi", type=int, default=160)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    resolve_inputs(args)
    if not args.hdf5.exists():
        raise FileNotFoundError(args.hdf5)
    if not args.download_root.exists():
        raise FileNotFoundError(args.download_root)
    summary = process(args)
    status_counts = summary["plot_status"].value_counts(dropna=False).to_dict() if "plot_status" in summary else {}
    velocity_counts = summary["velocity_status"].value_counts(dropna=False).to_dict() if "velocity_status" in summary else {}
    print(f"Wrote QC summary: {args.output_dir / 'qc_summary.csv'}")
    print(f"Plot status counts: {status_counts}")
    print(f"Velocity status counts: {velocity_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
