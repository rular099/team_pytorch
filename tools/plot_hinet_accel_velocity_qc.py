#!/usr/bin/env python3
"""Plot K-NET/KiK-net acceleration against matched Hi-net velocity windows.

The plot is anchored on the theoretical P arrival recorded by
``tools/download_hinet_velocity.py``.  Training-data picks are converted back to
absolute station time; the default plot keeps only the theoretical-P line and
uses compact axis markers for trigger/final/PGA so the waveforms stay readable.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shlex
import struct
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.hinet_raw_archive import AnnualHinetArchiveReader  # noqa: E402

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
SPAN_COLUMNS = [
    ("p_pick_search_left_aligned", "p_pick_search_right_aligned", "pick_search", "tab:blue"),
    ("p_pick_search_raw_left_aligned", "p_pick_search_raw_right_aligned", "raw_search", "tab:cyan"),
    ("stalta_search_left_aligned", "stalta_search_right_aligned", "stalta_search", "tab:orange"),
]
DEFAULT_BOTTOM_MARKERS = [
    ("p_pick_trigger_aligned", "trigger", "0.45", 0.03),
]
DEFAULT_TOP_MARKERS = [
    ("pga_norm_aligned_loc", "pga", "tab:olive", 0.03),
]
CANDIDATE_BOTTOM_MARKERS = [
    ("stalta_refined_pick_aligned", "stalta", "tab:orange", 0.17),
    ("p_pick_diting_acc_aligned", "diting_acc", "tab:red", 0.24),
    ("p_pick_diting_vel_aligned", "diting_vel", "tab:purple", 0.31),
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
        if token in {"--hdf5", "--output-root", "--match-distance-km"}:
            out[token] = tokens[i + 1]
    return out


def resolve_inputs(args: argparse.Namespace) -> None:
    parsed = parse_download_script(args.download_script)
    if args.hdf5 is None and "--hdf5" in parsed:
        args.hdf5 = Path(parsed["--hdf5"])
    if args.download_root is None and "--output-root" in parsed:
        args.download_root = Path(parsed["--output-root"])
    if args.max_match_distance_km is None and "--match-distance-km" in parsed:
        args.max_match_distance_km = float(parsed["--match-distance-km"])
    if args.hdf5 is None:
        raise SystemExit("Provide --hdf5 or keep it in --download-script.")
    if args.download_root is None:
        raise SystemExit("Provide --download-root or keep --output-root in --download-script.")
    args.hdf5 = args.hdf5.expanduser().resolve()
    args.download_root = args.download_root.expanduser().resolve()
    args.output_dir = (args.output_dir or (REPO_ROOT / "hinet_velocity_qc")).expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.raw_search_dir = [p.expanduser().resolve() for p in args.raw_search_dir]
    if args.max_match_distance_km is None:
        args.max_match_distance_km = 1.0


def load_manifest(args: argparse.Namespace) -> pd.DataFrame:
    download_root = args.download_root
    manifest_path = download_root / "manifests" / "download_manifest.csv"
    if manifest_path.exists():
        df = pd.read_csv(manifest_path, dtype={"event_id": str, "knet_station": str, "hinet_station": str})
    else:
        event_manifests = sorted((download_root / "manifests" / "events").glob("*/download_manifest.csv"))
        if event_manifests:
            df = pd.concat((pd.read_csv(p, dtype={"event_id": str, "knet_station": str, "hinet_station": str}) for p in event_manifests), ignore_index=True)
        else:
            if args.event_id and str(args.event_id)[:4].isdigit():
                archive_year = str(args.event_id)[:4]
            else:
                year_match = re.search(r"japan_(20\d{2})\.hdf5$", args.hdf5.name)
                archive_year = year_match.group(1) if year_match else ""
            archive_candidates = []
            if archive_year:
                archive_candidates.extend((download_root / "archive").glob(f"hinet_raw_{archive_year}*.h5"))
            if not archive_candidates:
                archive_candidates.extend((download_root / "archive").glob("hinet_raw_*.h5"))
            frames = []
            for archive_path in sorted(set(archive_candidates)):
                with AnnualHinetArchiveReader(archive_path) as reader:
                    event_ids = reader.event_ids()
                    if args.event_id:
                        event_ids = [event_id for event_id in event_ids if event_id == str(args.event_id)]
                    for event_id in event_ids:
                        event_manifest = reader.manifest(event_id)
                        if event_manifest.empty:
                            continue
                        event_manifest = event_manifest.copy()
                        event_manifest["archive_path"] = str(archive_path)
                        event_manifest["archive_event_id"] = event_id
                        frames.append(event_manifest)
            if not frames:
                raise FileNotFoundError(f"No legacy or annual-archive manifest found under {download_root}")
            df = pd.concat(frames, ignore_index=True)
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


def load_archive_velocity(row: pd.Series, args: argparse.Namespace) -> VelocitySeries | None:
    archive_value = row.get("archive_path", "")
    if archive_value is None or pd.isna(archive_value) or not str(archive_value).strip():
        return None
    archive_path = Path(str(archive_value)).expanduser()
    if not archive_path.exists():
        return VelocitySeries(
            np.array([]),
            np.array([]),
            "annual_hdf5",
            "archive_missing",
            "Hi-net annual CNT archive",
            str(archive_path),
            0,
        )
    event_id = str(row.get("archive_event_id", row["event_id"]))
    hinet_station = str(row["hinet_station"])
    if args.component == "norm":
        components = ("U", "Z", "N", "1", "E", "2")
    else:
        components = HINET_COMPONENT_BY_MODE[args.component]
    try:
        with AnnualHinetArchiveReader(archive_path) as reader:
            component_series = reader.read_station_series(event_id, hinet_station, components)
    except Exception as exc:
        return VelocitySeries(
            np.array([]),
            np.array([]),
            "annual_hdf5",
            f"read_failed:{exc!r}",
            "Hi-net annual CNT archive",
            str(archive_path),
            0,
        )
    if not component_series:
        return VelocitySeries(
            np.array([]),
            np.array([]),
            "annual_hdf5",
            "channel_not_found",
            "Hi-net annual CNT archive",
            str(archive_path),
            0,
        )

    ppick_ts = float(row["ppick_timestamp"])
    if args.component == "norm":
        base_t = max((series[0] for series in component_series.values()), key=lambda values: values.size)
        arrays = [
            np.interp(base_t, times, values, left=np.nan, right=np.nan)
            for times, values in component_series.values()
        ]
        values = np.sqrt(np.nansum(np.vstack(arrays) ** 2, axis=0))
        times = base_t
        label = "Hi-net annual CNT archive norm counts"
    else:
        preferred = HINET_COMPONENT_BY_MODE[args.component]
        selected_component = next((name for name in preferred if name in component_series), None)
        if selected_component is None:
            selected_component = next(iter(component_series))
        times, values = component_series[selected_component]
        label = f"Hi-net annual CNT archive {hinet_station} {selected_component} counts"
    t_rel = times - ppick_ts
    mask = (t_rel >= -float(args.pre_seconds)) & (t_rel <= float(args.post_seconds))
    return VelocitySeries(
        t_rel[mask],
        np.asarray(values)[mask],
        "annual_hdf5",
        "loaded",
        label,
        f"{archive_path}#{event_id}",
        len(component_series),
    )


def load_velocity(row: pd.Series, args: argparse.Namespace) -> VelocitySeries:
    archived = load_archive_velocity(row, args)
    if archived is not None and archived.t_rel.size:
        return archived
    mseed = load_mseed_velocity(row, args)
    if mseed is not None and mseed.t_rel.size:
        return mseed
    raw = load_raw_velocity(row, args)
    if raw is not None and raw.t_rel.size:
        return raw
    if raw is not None:
        return raw
    if archived is not None:
        return archived
    if mseed is not None:
        return mseed
    return VelocitySeries(np.array([]), np.array([]), "none", "missing_velocity", "Hi-net velocity unavailable", "", 0)


def add_search_windows(ax, spans: dict[str, tuple[float, float]]) -> None:
    span_colors = {
        "pick_search": "tab:blue",
        "raw_search": "tab:cyan",
        "stalta_search": "tab:orange",
    }
    for label, (x0, x1) in spans.items():
        color = span_colors.get(label, "0.5")
        ax.axvspan(x0, x1, color=color, alpha=0.10, label=label)


def add_axis_marker(ax, x: float, label: str, color: str, y_axes: float, marker: str = "^") -> None:
    if not np.isfinite(x):
        return
    ax.scatter(
        [x],
        [y_axes],
        transform=ax.get_xaxis_transform(),
        marker=marker,
        s=44,
        color=color,
        edgecolors="white",
        linewidths=0.45,
        zorder=6,
        clip_on=False,
        label=label,
    )


def first_finite_pick(picks: dict[str, float], columns: tuple[str, ...]) -> float:
    for col in columns:
        value = picks.get(col, np.nan)
        if np.isfinite(value):
            return float(value)
    return float("nan")


def add_compact_pick_markers(ax, picks: dict[str, float], marker_specs: list[tuple[str, str, str, float]]) -> None:
    for col, label, color, y_axes in marker_specs:
        add_axis_marker(ax, picks.get(col, np.nan), label, color, y_axes)


def add_candidate_pick_markers(ax, picks: dict[str, float]) -> None:
    add_compact_pick_markers(ax, picks, CANDIDATE_BOTTOM_MARKERS)


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


def waveform_qc_metrics(
    training: TrainingStation,
    velocity: VelocitySeries,
    row: pd.Series,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Return timestamp, coverage, finite-value, and station-match checks."""
    metrics: dict[str, object] = {}
    fail_reasons: list[str] = []
    warn_reasons: list[str] = []

    acc_finite = np.isfinite(training.acceleration)
    metrics["acceleration_n_samples"] = int(training.acceleration.size)
    metrics["acceleration_finite_fraction"] = float(acc_finite.mean()) if acc_finite.size else 0.0
    acc_contains_p = bool(
        training.valid_start_rel_seconds <= 0.0 <= training.valid_end_rel_seconds
    )
    metrics["acceleration_contains_theoretical_p"] = int(acc_contains_p)
    if not acc_contains_p:
        fail_reasons.append("theoretical_P_outside_acc_record")
    if metrics["acceleration_finite_fraction"] < 0.999:
        fail_reasons.append("nonfinite_acceleration")

    metrics["velocity_n_samples"] = int(velocity.values.size)
    if velocity.values.size == 0 or velocity.t_rel.size == 0:
        metrics.update({
            "velocity_finite_fraction": 0.0,
            "velocity_start_minus_p_s": np.nan,
            "velocity_end_minus_p_s": np.nan,
            "velocity_median_dt_s": np.nan,
            "velocity_gap_count": 0,
            "velocity_nonmonotonic_count": 0,
            "velocity_contains_theoretical_p": 0,
            "velocity_covers_requested_window": 0,
        })
        fail_reasons.append("velocity_missing")
    else:
        finite = np.isfinite(velocity.values) & np.isfinite(velocity.t_rel)
        finite_fraction = float(finite.mean())
        metrics["velocity_finite_fraction"] = finite_fraction
        metrics["velocity_start_minus_p_s"] = float(np.nanmin(velocity.t_rel))
        metrics["velocity_end_minus_p_s"] = float(np.nanmax(velocity.t_rel))
        diffs = np.diff(velocity.t_rel[np.isfinite(velocity.t_rel)])
        positive = diffs[diffs > 0]
        median_dt = float(np.median(positive)) if positive.size else np.nan
        gap_count = int(np.sum(diffs > 1.5 * median_dt)) if np.isfinite(median_dt) else 0
        nonmonotonic_count = int(np.sum(diffs <= 0))
        tolerance = max(0.05, 2.0 * median_dt) if np.isfinite(median_dt) else 0.05
        contains_p = bool(
            metrics["velocity_start_minus_p_s"] <= tolerance
            and metrics["velocity_end_minus_p_s"] >= -tolerance
        )
        covers_window = bool(
            metrics["velocity_start_minus_p_s"] <= -float(args.pre_seconds) + tolerance
            and metrics["velocity_end_minus_p_s"] >= float(args.post_seconds) - tolerance
        )
        metrics.update({
            "velocity_median_dt_s": median_dt,
            "velocity_gap_count": gap_count,
            "velocity_nonmonotonic_count": nonmonotonic_count,
            "velocity_contains_theoretical_p": int(contains_p),
            "velocity_covers_requested_window": int(covers_window),
        })
        if finite_fraction < 0.999:
            fail_reasons.append("nonfinite_velocity")
        if nonmonotonic_count:
            fail_reasons.append("nonmonotonic_velocity_time")
        if not contains_p:
            fail_reasons.append("theoretical_P_outside_velocity")
        if gap_count:
            warn_reasons.append(f"velocity_gaps={gap_count}")
        if not covers_window:
            warn_reasons.append("velocity_short_coverage")

    match_distance = finite_float(row.get("match_distance_km"), default=np.nan)
    match_ok = bool(
        np.isfinite(match_distance)
        and match_distance <= float(args.max_match_distance_km) + 1e-9
    )
    metrics["station_match_distance_km"] = match_distance
    metrics["station_match_threshold_km"] = float(args.max_match_distance_km)
    metrics["station_match_within_threshold"] = int(match_ok)
    if not match_ok:
        fail_reasons.append("station_match_distance_exceeded")

    if fail_reasons:
        status = "FAIL"
    elif warn_reasons:
        status = "WARN"
    else:
        status = "PASS"
    metrics["qc_status"] = status
    metrics["qc_fail_reasons"] = ";".join(fail_reasons)
    metrics["qc_warn_reasons"] = ";".join(warn_reasons)
    return metrics


def set_relative_and_absolute_time_ticks(ax, ppick_ts: float, args: argparse.Namespace) -> None:
    """Use two-line ticks: relative seconds above the corresponding JST time."""
    ticks = np.linspace(-float(args.pre_seconds), float(args.post_seconds), 5)
    timestamps = [datetime.fromtimestamp(ppick_ts + float(x), tz=JST) for x in ticks]
    crosses_date = len({stamp.date() for stamp in timestamps}) > 1
    labels = [
        (
            f"{offset:+.0f} s\n{stamp.strftime('%m-%d %H:%M:%S')}"
            if crosses_date
            else f"{offset:+.0f} s\n{stamp.strftime('%H:%M:%S')}"
        )
        for offset, stamp in zip(ticks, timestamps)
    ]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8)
    date_label = timestamps[len(timestamps) // 2].strftime("%Y-%m-%d")
    ax.set_xlabel(
        f"seconds relative to theoretical P / absolute time JST ({date_label})",
        fontsize=9,
    )


def plot_pair(
    training: TrainingStation,
    velocity: VelocitySeries,
    row: pd.Series,
    qc: dict[str, object],
    args: argparse.Namespace,
) -> Path:
    import matplotlib.pyplot as plt

    acceleration_color = "#0072B2"
    velocity_color = "#D55E00"
    p_color = "#009E73"
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True, constrained_layout=True)
    axes[0].plot(
        training.acceleration_t_rel,
        training.acceleration,
        color=acceleration_color,
        linewidth=0.8,
    )
    add_invalid_acceleration_shading(axes[0], training, args)
    axes[0].axvline(0.0, color=p_color, linewidth=1.3, linestyle="--", label="theoretical_P")
    if args.show_search_windows:
        add_search_windows(axes[0], training.span_rel_seconds)
    add_compact_pick_markers(axes[0], training.pick_rel_seconds, DEFAULT_TOP_MARKERS)
    axes[0].set_ylabel(
        "K-NET/KiK-net original acceleration (m s⁻²)"
        if args.component != "norm"
        else "K-NET/KiK-net original |acceleration| (m s⁻²)"
    )
    dedupe_legend(axes[0])

    if velocity.t_rel.size:
        axes[1].plot(velocity.t_rel, velocity.values, color=velocity_color, linewidth=0.8)
    else:
        axes[1].text(0.5, 0.5, f"Velocity unavailable: {velocity.status}", ha="center", va="center", transform=axes[1].transAxes, fontsize=10)
    axes[1].axvline(0.0, color=p_color, linewidth=1.3, linestyle="--", label="theoretical_P")
    if args.show_search_windows:
        add_search_windows(axes[1], training.span_rel_seconds)
    add_compact_pick_markers(axes[1], training.pick_rel_seconds, DEFAULT_BOTTOM_MARKERS)
    final_pick = first_finite_pick(training.pick_rel_seconds, ("p_picks", "p_pick_refined_aligned"))
    add_axis_marker(axes[1], final_pick, "final", "black", 0.10)
    if args.show_candidate_picks:
        add_candidate_pick_markers(axes[1], training.pick_rel_seconds)
    axes[1].set_ylabel(velocity.label.replace("Hi-net", "Hi-net velocity sensor"))
    axes[1].set_xlim(-args.pre_seconds, args.post_seconds)
    dedupe_legend(axes[1])

    ppick_ts = float(row["ppick_timestamp"])
    set_relative_and_absolute_time_ticks(axes[1], ppick_ts, args)

    final_offset = first_finite_pick(training.pick_rel_seconds, ("p_picks", "p_pick_refined_aligned"))
    match_distance = finite_float(row.get("match_distance_km"), default=np.nan)
    epi_distance = finite_float(row.get("epicentral_distance_km"), default=np.nan)
    origin_ts = finite_float(row.get("origin_timestamp"), default=np.nan)
    origin_jst = (
        datetime.fromtimestamp(origin_ts, tz=JST).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " JST"
        if np.isfinite(origin_ts)
        else str(row.get("origin_time_jst", ""))
    )
    p_jst = datetime.fromtimestamp(ppick_ts, tz=JST).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " JST"
    magnitude = finite_float(row.get("event_magnitude"), default=np.nan)
    depth = finite_float(row.get("event_depth_km"), default=np.nan)
    title = (
        f"Hi-net download QC [{qc['qc_status']}] | event {training.event_id} | "
        f"M {magnitude:.1f}, depth {depth:.1f} km | origin {origin_jst}\n"
        f"K-NET/KiK-net {training.station_code} ({training.sensor_class}, wave {training.wave_idx}) ↔ "
        f"Hi-net {row.get('hinet_station', '')} | station separation {match_distance:.3f} km | "
        f"epicentral distance {epi_distance:.1f} km\n"
        f"theoretical P {p_jst} | final pick − P {final_offset:.2f} s | "
        f"waveform source {velocity.source}:{velocity.status}"
    )
    fig.suptitle(title, fontsize=9.5, x=0.01, ha="left")
    qc_note = (
        f"n={int(qc['velocity_n_samples'])}, Δt={finite_float(qc['velocity_median_dt_s']):.5f} s, "
        f"gaps={int(qc['velocity_gap_count'])}, finite={100.0 * finite_float(qc['velocity_finite_fraction']):.2f}%"
    )
    axes[1].text(
        0.995,
        0.02,
        qc_note,
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.25",
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85, "pad": 2.5},
    )
    for ax in axes:
        ax.grid(True, axis="x", alpha=0.18)
    filename = safe_slug(training.event_id, training.station_code, f"wave{training.wave_idx}", row.get("hinet_station", ""), args.component) + ".png"
    out_path = args.output_dir / filename
    fig.savefig(out_path, dpi=args.dpi)
    if args.save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


def process(args: argparse.Namespace) -> pd.DataFrame:
    manifest = select_rows(load_manifest(args), args)
    rows = []
    with h5py.File(args.hdf5, "r") as h5:
        for _, row in manifest.iterrows():
            summary = dict(row)
            try:
                training = load_training_station(h5, row, args)
                velocity = load_velocity(row, args)
                qc = waveform_qc_metrics(training, velocity, row, args)
                plot_path = plot_pair(training, velocity, row, qc, args)
                summary.update(training.summary)
                summary.update(qc)
                summary.update({
                    "velocity_source": velocity.source,
                    "velocity_status": velocity.status,
                    "velocity_path": velocity.path,
                    "velocity_trace_count": velocity.trace_count,
                    "plot_path": str(plot_path),
                    "plot_status": "written" if velocity.t_rel.size else "written_missing_velocity",
                })
                final_offset = first_finite_pick(training.pick_rel_seconds, ("p_picks", "p_pick_refined_aligned"))
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
    parser.add_argument(
        "--max-match-distance-km",
        type=float,
        default=None,
        help="QC threshold for station separation. Defaults to --match-distance-km parsed from download_hinet.sh, then 1 km.",
    )
    parser.add_argument("--large-offset-seconds", type=float, default=20.0, help="Flag p_picks offsets larger than this threshold.")
    parser.add_argument("--prefer-written-mseed", action="store_true", help="In batch mode, plot rows with MiniSEED first.")
    parser.add_argument("--raw-search-dir", type=Path, action="append", default=[], help="Extra directory containing raw *.cnt/*.ch files.")
    parser.add_argument("--show-candidate-picks", action="store_true", help="Also mark STALTA and DiTing candidate picks on the velocity-panel time axis.")
    parser.add_argument("--show-search-windows", action="store_true", help="Also shade travel-time/STALTA search windows.")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--save-pdf", action="store_true", help="Also save a vector PDF beside each PNG QC plot.")
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
