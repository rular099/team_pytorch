#!/usr/bin/env python3
"""Download Hi-net velocity waveforms for existing K-NET/KiK-net events.

The script is intentionally conservative:

* Hi-net credentials are read from HINET_USER/HINET_PASSWORD.
* Hi-net station metadata and response channel tables are cached on disk.
* K-NET/KiK-net stations are matched to Hi-net stations by horizontal distance.
* Raw WIN32 count files are always kept. MiniSEED cuts are best-effort raw-count
  exports from the raw WIN32 files; conversion failures are recorded in the
  manifest without discarding the raw download.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import math
import os
import shutil
import struct
import sys
import time
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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/codex-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

from japan_dataset_builder import (  # noqa: E402
    AK135TravelTimeModel,
    DEFAULT_JMA2001A_ZIP,
    JMATravelTimeTable,
    haversine_distance_km,
)


JST = timezone(timedelta(hours=9))
COMPONENT_MAP = {
    "U": "HHZ",
    "Z": "HHZ",
    "N": "HHN",
    "1": "HHN",
    "E": "HHE",
    "2": "HHE",
}


@dataclass(frozen=True)
class EventInfo:
    event_id: str
    origin_time_jst: str
    origin_timestamp: float
    latitude: float
    longitude: float
    depth_km: float
    magnitude: float | None
    origin_source: str


@dataclass(frozen=True)
class RawDownloadResult:
    cnt_path: Path | None
    ch_path: Path | None
    segment_paths: tuple[Path, ...]
    raw_status: str
    raw_error: str


def _decode_array(values):
    arr = np.asarray(values)
    if arr.dtype.kind == "S":
        return np.asarray([v.decode("utf-8", errors="replace") for v in arr])
    return arr


def read_metadata_table(hdf5_path: Path, table_name: str) -> pd.DataFrame:
    key = f"metadata/{table_name}"
    with h5py.File(hdf5_path, "r") as f:
        if key not in f:
            raise KeyError(f"{key} not found in {hdf5_path}")
        node = f[key]
        if isinstance(node, h5py.Group):
            return pd.DataFrame({col: _decode_array(node[col][()]) for col in node.keys()})
    return pd.read_hdf(hdf5_path, key)


def find_column(columns: Iterable[str], candidates: Iterable[str], required: bool = True) -> str | None:
    column_set = {str(c).lower(): str(c) for c in columns}
    for candidate in candidates:
        if candidate.lower() in column_set:
            return column_set[candidate.lower()]
    if required:
        raise KeyError(f"None of {list(candidates)} found in columns {list(columns)}")
    return None


def parse_origin_time(value: object, event_id: str) -> tuple[str, float, str]:
    if value is not None and not pd.isna(value):
        text = str(value).strip()
        formats = [
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(text, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=JST)
                return dt.astimezone(JST).isoformat(), dt.timestamp(), "metadata"
            except ValueError:
                continue
        try:
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=JST)
            return dt.astimezone(JST).isoformat(), dt.timestamp(), "metadata"
        except ValueError:
            pass
    if len(str(event_id)) == 14 and str(event_id).isdigit():
        dt = datetime.strptime(str(event_id), "%Y%m%d%H%M%S").replace(tzinfo=JST)
        return dt.isoformat(), dt.timestamp(), "event_id_yyyymmddHHMMSS"
    raise ValueError(
        f"Cannot determine origin time for event {event_id!r}. "
        "Expected Origin_Time(JST)/origin_time in metadata or a YYYYmmddHHMMSS event id."
    )


def load_event_table(hdf5_path: Path) -> pd.DataFrame:
    event_df = read_metadata_table(hdf5_path, "event_metadata")
    event_key = find_column(event_df.columns, ["EVENT", "KiK_File", "#EventID"])
    event_df[event_key] = event_df[event_key].astype(str)
    return event_df.drop_duplicates(event_key).reset_index(drop=True)


def load_station_table(hdf5_path: Path) -> pd.DataFrame:
    try:
        station_df = read_metadata_table(hdf5_path, "station_metadata")
    except KeyError:
        rows: list[dict[str, object]] = []
        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                raise KeyError(f"data group not found in {hdf5_path}")
            for event_id, group in f["data"].items():
                if "coords" not in group or "station_codes" not in group:
                    continue
                coords = np.asarray(group["coords"][()])
                codes = _decode_array(group["station_codes"][()])
                source = _decode_array(group["source_network"][()]) if "source_network" in group else [""] * len(codes)
                sensor = _decode_array(group["sensor_class"][()]) if "sensor_class" in group else [""] * len(codes)
                for i, code in enumerate(codes):
                    rows.append({
                        "EVENT": str(event_id),
                        "wave_idx": i,
                        "station_code": str(code),
                        "station_lat": float(coords[i, 0]),
                        "station_lon": float(coords[i, 1]),
                        "station_height_m": float(coords[i, 2]) * 1000.0 if coords.shape[1] > 2 else 0.0,
                        "source_network": str(source[i]) if i < len(source) else "",
                        "sensor_class": str(sensor[i]) if i < len(sensor) else "",
                    })
        if not rows:
            raise KeyError(
                "metadata/station_metadata is missing and data/* lacks station_codes; "
                "cannot match K-NET/KiK-net stations to Hi-net."
            )
        station_df = pd.DataFrame(rows)
    return station_df


def load_origin_corrections(path: Path | None, require_accepted: bool = True) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    df = pd.read_csv(path, dtype={"event_id": str, "EVENT": str, "jma_event_id": str})
    if df.empty:
        return {}
    event_key = find_column(df.columns, ["event_id", "EVENT", "KiK_File", "#EventID"])
    ts_key = find_column(df.columns, ["origin_timestamp_corrected", "jma_origin_timestamp"], required=False)
    time_key = find_column(df.columns, ["origin_time_jst_corrected", "jma_origin_time_jst"], required=False)
    if ts_key is None and time_key is None:
        raise KeyError(f"{path} must contain origin_timestamp_corrected/jma_origin_timestamp or origin_time_jst_corrected/jma_origin_time_jst")
    if require_accepted and "accepted" in df.columns:
        df = df[pd.to_numeric(df["accepted"], errors="coerce").fillna(0).astype(int) == 1].copy()
    out: dict[str, dict[str, object]] = {}
    for _, row in df.iterrows():
        event_id = str(row[event_key])
        if ts_key is not None and not pd.isna(row[ts_key]):
            origin_ts = float(row[ts_key])
            origin_dt = datetime.fromtimestamp(origin_ts, tz=JST)
            origin_iso = origin_dt.isoformat(timespec="milliseconds")
        else:
            origin_iso, origin_ts, _ = parse_origin_time(row[time_key], event_id)
        if time_key is not None and not pd.isna(row[time_key]):
            origin_iso = str(row[time_key])
        out[event_id] = {
            "origin_time_jst": origin_iso,
            "origin_timestamp": origin_ts,
            "match_status": row.get("match_status", ""),
            "origin_time_correction_s": row.get("origin_time_correction_s", ""),
            "source": str(path),
        }
    return out


def load_events(
    hdf5_path: Path,
    selected_event_ids: set[str] | None,
    origin_corrections: dict[str, dict[str, object]] | None = None,
) -> dict[str, EventInfo]:
    event_df = load_event_table(hdf5_path)
    event_key = find_column(event_df.columns, ["EVENT", "KiK_File", "#EventID"])
    lat_key = find_column(event_df.columns, ["Latitude", "LAT", "lat", "event_lat"])
    lon_key = find_column(event_df.columns, ["Longitude", "LON", "lon", "event_lon"])
    depth_key = find_column(event_df.columns, ["DEPTH", "Depth", "depth_km", "event_depth_km"])
    mag_key = find_column(event_df.columns, ["Magnitude", "M_J", "MA", "Mag"], required=False)
    origin_key = find_column(
        event_df.columns,
        ["Origin_Time(JST)", "Origin Time", "origin_time", "origin_time_jst", "Time"],
        required=False,
    )

    events: dict[str, EventInfo] = {}
    for _, row in event_df.iterrows():
        event_id = str(row[event_key])
        if selected_event_ids is not None and event_id not in selected_event_ids:
            continue
        origin_value = row[origin_key] if origin_key else None
        origin_iso, origin_ts, origin_source = parse_origin_time(origin_value, event_id)
        if origin_corrections and event_id in origin_corrections:
            correction = origin_corrections[event_id]
            origin_iso = str(correction["origin_time_jst"])
            origin_ts = float(correction["origin_timestamp"])
            status = correction.get("match_status", "")
            delta = correction.get("origin_time_correction_s", "")
            origin_source = f"jma_origin_correction:{status}:delta_s={delta}"
        events[event_id] = EventInfo(
            event_id=event_id,
            origin_time_jst=origin_iso,
            origin_timestamp=float(origin_ts),
            latitude=float(row[lat_key]),
            longitude=float(row[lon_key]),
            depth_km=float(row[depth_key]),
            magnitude=None if mag_key is None or pd.isna(row[mag_key]) else float(row[mag_key]),
            origin_source=origin_source,
        )
    return events


def select_event_ids(
    hdf5_path: Path,
    mode: str,
    event_ids_path: Path | None,
    num_events: int,
    seed: int,
) -> set[str] | None:
    if event_ids_path is not None:
        ids = [line.strip() for line in event_ids_path.read_text().splitlines() if line.strip()]
        return set(ids)
    if mode == "all":
        return None
    event_df = load_event_table(hdf5_path)
    event_key = find_column(event_df.columns, ["EVENT", "KiK_File", "#EventID"])
    ids = event_df[event_key].astype(str).drop_duplicates().to_numpy()
    if len(ids) <= num_events:
        return set(map(str, ids))
    rng = np.random.default_rng(seed)
    picked = rng.choice(ids, size=num_events, replace=False)
    return set(map(str, picked))


def load_or_download_hinet_inventory(args) -> pd.DataFrame:
    inventory_path = args.inventory_csv
    if inventory_path.exists() and not args.overwrite_inventory:
        return pd.read_csv(inventory_path, dtype=str)

    client = make_hinet_client()
    if args.dry_run and not args.allow_network_in_dry_run:
        raise FileNotFoundError(
            f"Inventory cache not found: {inventory_path}. "
            "Run without --dry-run, or add --allow-network-in-dry-run, after HINET_USER/HINET_PASSWORD are set."
        )
    rows = fetch_hinet_station_rows(client, args.hinet_network)
    if not rows:
        raise RuntimeError(f"Hi-net station inventory for network {args.hinet_network!r} is empty")
    write_csv(inventory_path, rows, sorted(rows[0].keys()))
    return pd.DataFrame(rows)


def make_hinet_client():
    user = os.environ.get("HINET_USER")
    password = os.environ.get("HINET_PASSWORD")
    if not user or not password:
        raise RuntimeError("Set HINET_USER and HINET_PASSWORD before accessing Hi-net.")
    try:
        from HinetPy import Client
    except ImportError as exc:
        raise ImportError("HinetPy is required for Hi-net downloads. Install it with `pip install HinetPy`.") from exc
    return Client(user, password)


def fetch_hinet_station_rows(client, network_code: str) -> list[dict[str, object]]:
    # HinetPy versions have changed method names over time. Keep the access
    # small and explicit so local installations can be handled without editing
    # the download workflow.
    method = None
    for name in ("get_station_list", "get_station_list_for_network", "get_stations"):
        if hasattr(client, name):
            method = getattr(client, name)
            break
    if method is None:
        raise AttributeError("HinetPy Client has no recognized station-list method")

    try:
        stations = method(network_code)
    except TypeError:
        stations = method()

    rows: list[dict[str, object]] = []
    for item in stations:
        if isinstance(item, dict):
            row = {str(k): v for k, v in item.items()}
        else:
            row = {
                name: getattr(item, name)
                for name in dir(item)
                if not name.startswith("_") and not callable(getattr(item, name))
            }
        name = first_present(row, ["name", "station", "code", "station_code"])
        lat = first_present(row, ["latitude", "lat"])
        lon = first_present(row, ["longitude", "lon", "lng"])
        elev = first_present(row, ["elevation", "elevation_m", "height", "height_m"], default=0.0)
        if name is None or lat is None or lon is None:
            continue
        rows.append({
            "hinet_network": network_code,
            "hinet_station": str(name),
            "hinet_lat": float(lat),
            "hinet_lon": float(lon),
            "hinet_elevation_m": float(elev),
        })
    deduped: dict[str, dict[str, object]] = {}
    for row in rows:
        deduped.setdefault(str(row["hinet_station"]), row)
    return list(deduped.values())


def first_present(row: dict[str, object], keys: Iterable[str], default=None):
    lowered = {str(k).lower(): k for k in row.keys()}
    for key in keys:
        if key.lower() in lowered:
            value = row[lowered[key.lower()]]
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                return value
    return default


def build_station_matches(args, station_df: pd.DataFrame, hinet_df: pd.DataFrame) -> pd.DataFrame:
    match_path = args.match_csv
    if match_path.exists() and not args.overwrite_matches:
        return pd.read_csv(match_path, dtype={"knet_station": str, "hinet_station": str})

    station_key = find_column(station_df.columns, ["station_code", "Station Code"])
    lat_key = find_column(station_df.columns, ["station_lat", "Station Lat.", "latitude"])
    lon_key = find_column(station_df.columns, ["station_lon", "Station Long.", "longitude"])
    height_key = find_column(station_df.columns, ["station_height_m", "Station Height(m)"], required=False)
    source_key = find_column(station_df.columns, ["source_network"], required=False)
    sensor_key = find_column(station_df.columns, ["sensor_class"], required=False)

    unique_cols = [station_key, lat_key, lon_key]
    if height_key:
        unique_cols.append(height_key)
    if source_key:
        unique_cols.append(source_key)
    if sensor_key:
        unique_cols.append(sensor_key)
    stations = station_df[unique_cols].drop_duplicates().reset_index(drop=True)

    hinet_lat = pd.to_numeric(hinet_df["hinet_lat"], errors="coerce").to_numpy(dtype=float)
    hinet_lon = pd.to_numeric(hinet_df["hinet_lon"], errors="coerce").to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for _, sta in stations.iterrows():
        lat = float(sta[lat_key])
        lon = float(sta[lon_key])
        dists = np.asarray([
            haversine_distance_km(lat, lon, float(h_lat), float(h_lon))
            for h_lat, h_lon in zip(hinet_lat, hinet_lon)
        ])
        order = np.argsort(dists)
        best = int(order[0])
        second = int(order[1]) if len(order) > 1 else best
        accepted = float(dists[best]) <= float(args.match_distance_km)
        rows.append({
            "knet_station": str(sta[station_key]),
            "knet_lat": lat,
            "knet_lon": lon,
            "knet_height_m": float(sta[height_key]) if height_key else "",
            "source_network": str(sta[source_key]) if source_key else "",
            "sensor_class": str(sta[sensor_key]) if sensor_key else "",
            "hinet_network": str(hinet_df.iloc[best]["hinet_network"]),
            "hinet_station": str(hinet_df.iloc[best]["hinet_station"]),
            "hinet_lat": float(hinet_df.iloc[best]["hinet_lat"]),
            "hinet_lon": float(hinet_df.iloc[best]["hinet_lon"]),
            "hinet_elevation_m": float(hinet_df.iloc[best].get("hinet_elevation_m", 0.0)),
            "match_distance_km": float(dists[best]),
            "second_hinet_station": str(hinet_df.iloc[second]["hinet_station"]),
            "second_match_distance_km": float(dists[second]),
            "ambiguous_within_2x": int(float(dists[second]) <= max(float(dists[best]) * 2.0, 0.1)),
            "accepted": int(accepted),
            "match_threshold_km": float(args.match_distance_km),
        })
    matches = pd.DataFrame(rows)
    write_csv(match_path, matches.to_dict(orient="records"), list(matches.columns))
    return matches


def build_travel_time_model(jma_zip: Path | None):
    jma_table = None
    try:
        jma_table = JMATravelTimeTable(jma_zip or DEFAULT_JMA2001A_ZIP)
    except Exception as exc:
        print(f"[WARN] failed to load JMA2001 table ({exc}); all picks will use ak135", file=sys.stderr)
    return jma_table, AK135TravelTimeModel()


def predict_p_seconds(event: EventInfo, station_lat: float, station_lon: float, station_elevation_m: float, jma_table, ak135):
    epicentral_km = haversine_distance_km(event.latitude, event.longitude, station_lat, station_lon)
    if jma_table is None:
        p_seconds = ak135.predict_p_seconds(event.depth_km, epicentral_km)
        return float(p_seconds), "ak135", 1, float(epicentral_km)
    try:
        p_seconds, clipped = jma_table.predict_p_seconds(event.depth_km, epicentral_km, station_elevation_m)
        if clipped:
            p_seconds = ak135.predict_p_seconds(event.depth_km, epicentral_km)
            return float(p_seconds), "ak135", 1, float(epicentral_km)
        return float(p_seconds), "jma2001a", 0, float(epicentral_km)
    except Exception:
        p_seconds = ak135.predict_p_seconds(event.depth_km, epicentral_km)
        return float(p_seconds), "ak135", 1, float(epicentral_km)


def floor_to_minute(ts: float) -> datetime:
    dt = datetime.fromtimestamp(ts, tz=JST)
    return dt.replace(second=0, microsecond=0)


def ceil_span_minutes(start_ts: float, end_ts: float) -> int:
    return max(1, int(math.ceil((end_ts - start_ts) / 60.0)))


def hinet_time_string(dt: datetime) -> str:
    return dt.astimezone(JST).strftime("%Y%m%d%H%M")


def _list_segment_cnts(segment_dir: Path) -> tuple[Path, ...]:
    return tuple(sorted(p for p in segment_dir.glob("*.cnt") if p.is_file()))


def _find_segment_channel_table(segment_dir: Path) -> Path | None:
    candidates = sorted([p for p in segment_dir.glob("*.euc.ch") if p.is_file()])
    candidates.extend(sorted([p for p in segment_dir.glob("*.ch") if p.is_file() and p not in candidates]))
    return candidates[0] if candidates else None


def _call_hinet_in_directory(
    client,
    network: str,
    starttime: str,
    span: int,
    data: Path,
    ctable: Path,
    outdir: Path,
    cwd: Path,
) -> None:
    cwd.mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd()
    try:
        os.chdir(cwd)
        call_get_continuous_waveform(
            client=client,
            network=network,
            starttime=starttime,
            span=span,
            data=data,
            ctable=ctable,
            outdir=outdir,
        )
    finally:
        os.chdir(old_cwd)


def download_raw_event(args, client, event: EventInfo, arrivals: pd.DataFrame) -> RawDownloadResult:
    event_raw_dir = args.output_root / "raw" / event.event_id
    event_raw_dir.mkdir(parents=True, exist_ok=True)
    segment_dir = event_raw_dir / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)

    start_ts = float(arrivals["cut_start_timestamp"].min())
    end_ts = float(arrivals["cut_end_timestamp"].max())
    request_start_dt = floor_to_minute(start_ts)
    request_start_ts = request_start_dt.timestamp()
    span_min = ceil_span_minutes(request_start_ts, end_ts)

    cnt_name = f"{event.event_id}_{request_start_dt.strftime('%Y%m%dT%H%M')}_{span_min:04d}min.cnt"
    ch_name = f"{event.event_id}_{request_start_dt.strftime('%Y%m%dT%H%M')}_{span_min:04d}min.ch"
    cnt_path = event_raw_dir / cnt_name
    ch_path = event_raw_dir / ch_name

    if args.overwrite_raw:
        for path in (cnt_path, ch_path):
            if path.exists():
                path.unlink()
        if segment_dir.exists():
            shutil.rmtree(segment_dir)
        segment_dir.mkdir(parents=True, exist_ok=True)

    if cnt_path.exists() and ch_path.exists() and not args.overwrite_raw:
        return RawDownloadResult(cnt_path, ch_path, tuple(), "skipped_existing", "")
    if args.dry_run:
        return RawDownloadResult(None, None, tuple(), "dry_run", "")

    stations = sorted(set(str(x) for x in arrivals["hinet_station"]))
    select_hinet_stations(client, args.hinet_network, stations)

    try:
        _call_hinet_in_directory(
            client=client,
            network=args.hinet_network,
            starttime=hinet_time_string(request_start_dt),
            span=span_min,
            data=cnt_path,
            ctable=ch_path,
            outdir=event_raw_dir,
            cwd=segment_dir,
        )
    except Exception as exc:
        segment_paths = _list_segment_cnts(segment_dir)
        segment_ch_path = _find_segment_channel_table(segment_dir)
        if segment_paths and segment_ch_path is not None:
            return RawDownloadResult(
                None,
                segment_ch_path,
                segment_paths,
                "downloaded_unmerged",
                f"merge_failed:{exc!r}",
            )
        return RawDownloadResult(None, segment_ch_path, segment_paths, "download_failed", repr(exc))

    if not cnt_path.exists():
        found = list(event_raw_dir.glob("*.cnt"))
        cnt_path = found[0] if found else cnt_path
    if not ch_path.exists():
        found = list(event_raw_dir.glob("*.ch"))
        ch_path = found[0] if found else ch_path
    if not cnt_path.exists() or not ch_path.exists():
        segment_paths = _list_segment_cnts(segment_dir)
        segment_ch_path = _find_segment_channel_table(segment_dir)
        if segment_paths and segment_ch_path is not None:
            return RawDownloadResult(
                None,
                segment_ch_path,
                segment_paths,
                "downloaded_unmerged",
                f"missing merged cnt={cnt_path.exists()} ch={ch_path.exists()}",
            )
        return RawDownloadResult(
            None,
            segment_ch_path,
            segment_paths,
            "download_missing_files",
            f"missing cnt={cnt_path.exists()} ch={ch_path.exists()}",
        )
    return RawDownloadResult(cnt_path, ch_path, tuple(), "downloaded", "")


def select_hinet_stations(client, network: str, stations: list[str]) -> None:
    if not stations:
        return
    if hasattr(client, "select_stations"):
        method = getattr(client, "select_stations")
        try:
            method(network, stations=stations)
            return
        except TypeError:
            pass
        try:
            method(network, stations)
            return
        except TypeError:
            pass
        method(stations)
        return
    raise AttributeError("HinetPy Client has no select_stations method")


def call_get_continuous_waveform(client, network: str, starttime: str, span: int, data: str, ctable: str, outdir: Path) -> None:
    if not hasattr(client, "get_continuous_waveform"):
        raise AttributeError("HinetPy Client has no get_continuous_waveform method")
    method = getattr(client, "get_continuous_waveform")
    kwargs = {
        "code": network,
        "starttime": starttime,
        "span": span,
        "data": str(data),
        "ctable": str(ctable),
        "outdir": str(outdir),
    }
    sig = inspect.signature(method)
    filtered = {key: value for key, value in kwargs.items() if key in sig.parameters}
    if "code" in sig.parameters:
        method(**filtered)
    else:
        filtered.pop("code", None)
        method(network, starttime, span, **filtered)


def cache_response_files(args, ch_path: Path, event_id: str) -> str:
    response_dir = args.output_root / "responses" / event_id
    response_dir.mkdir(parents=True, exist_ok=True)
    copied = response_dir / ch_path.name
    if not copied.exists() or args.overwrite_responses:
        shutil.copy2(ch_path, copied)
    status = "channel_table_copied"
    if args.dry_run:
        return status
    try:
        from HinetPy.win32 import extract_sacpz

        extract_sacpz(str(ch_path), outdir=str(response_dir), keep_sensitivity=True)
        status = "sacpz_extracted"
    except Exception as exc:
        status = f"sacpz_failed:{exc!r}"
    return status


def read_channel_table(ch_path: Path) -> pd.DataFrame:
    try:
        from HinetPy.win32 import read_ctable

        channels = read_ctable(str(ch_path))
        rows: list[dict[str, object]] = []
        for ch in channels:
            row = {
                name: getattr(ch, name)
                for name in dir(ch)
                if not name.startswith("_") and not callable(getattr(ch, name))
            }
            channel_id = first_present(row, ["id", "chid", "channel_id", "code"])
            name = first_present(row, ["name", "station", "station_name"])
            component = first_present(row, ["component", "comp"])
            lat = first_present(row, ["latitude", "lat"], default=np.nan)
            lon = first_present(row, ["longitude", "lon"], default=np.nan)
            rows.append({
                "channel_id": str(channel_id),
                "hinet_station": str(name),
                "component": str(component),
                "latitude": lat,
                "longitude": lon,
                "raw_repr": repr(ch),
            })
        return pd.DataFrame(rows)
    except Exception:
        return parse_channel_table_text(ch_path)


def parse_channel_table_text(ch_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for raw in ch_path.read_text(errors="ignore").splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        channel_id = parts[0]
        station = parts[3] if "." in parts[3] else parts[1]
        component = parts[4] if "." in parts[3] else parts[2]
        lat = np.nan
        lon = np.nan
        for i in range(len(parts) - 1):
            try:
                a = float(parts[i])
                b = float(parts[i + 1])
            except ValueError:
                continue
            if 20.0 <= a <= 50.0 and 120.0 <= b <= 155.0:
                lat, lon = a, b
                break
        rows.append({
            "channel_id": str(channel_id),
            "hinet_station": str(station),
            "component": str(component),
            "latitude": lat,
            "longitude": lon,
            "raw_line": raw,
        })
    return pd.DataFrame(rows)


def safe_station_code(value: str) -> str:
    text = str(value).strip().replace(".", "_").replace("/", "_")
    return text[:8] if len(text) > 8 else text


def bcd_to_int(value: int) -> int:
    return (value >> 4) * 10 + (value & 0x0F)


def parse_hinet_vm_timestamp(buf: bytes) -> float:
    if len(buf) < 8:
        raise ValueError("timestamp buffer too short")
    dt = datetime(
        bcd_to_int(buf[0]) * 100 + bcd_to_int(buf[1]),
        bcd_to_int(buf[2]),
        bcd_to_int(buf[3]),
        bcd_to_int(buf[4]),
        bcd_to_int(buf[5]),
        bcd_to_int(buf[6]),
        tzinfo=JST,
    )
    return dt.timestamp()


def decode_win32_diffs(first: int, encoded: bytes, datawide: float, srate: int) -> np.ndarray:
    if srate <= 0:
        return np.asarray([], dtype=np.int32)
    values = np.empty(srate, dtype=np.int64)
    values[0] = first
    if datawide == 0.5:
        previous = first
        idx = 1
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
        return values[:idx].astype(np.int32, copy=False)
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
        raise NotImplementedError(f"Unsupported WIN32 data width: {datawide}")
    n = min(diffs.size + 1, srate)
    if n > 1:
        values[1:n] = first + np.cumsum(diffs[: n - 1])
    return values[:n].astype(np.int32, copy=False)


def read_hinet_vm_cnt_segments(cnt_paths: Iterable[Path], channel_ids: set[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    wanted = {str(channel_id).lower() for channel_id in channel_ids}
    values_by_channel: dict[str, list[np.ndarray]] = {channel_id: [] for channel_id in wanted}
    times_by_channel: dict[str, list[np.ndarray]] = {channel_id: [] for channel_id in wanted}
    for path in sorted(cnt_paths):
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
                if channel_id not in wanted:
                    continue
                decoded = decode_win32_diffs(first, encoded, datawide, srate)
                if decoded.size == 0:
                    continue
                values_by_channel[channel_id].append(decoded)
                times_by_channel[channel_id].append(record_ts + np.arange(decoded.size, dtype=np.float64) / float(srate))
            offset = payload_end

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for channel_id in wanted:
        if values_by_channel[channel_id]:
            t = np.concatenate(times_by_channel[channel_id])
            y = np.concatenate(values_by_channel[channel_id])
            order = np.argsort(t)
            out[channel_id] = (t[order], y[order])
    return out


def infer_sampling_rate_hz(times: np.ndarray) -> float:
    if times.size < 2:
        return 100.0
    diffs = np.diff(times)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 100.0
    return float(round(1.0 / float(np.median(diffs)), 6))


def contiguous_slices(times: np.ndarray, sampling_rate_hz: float) -> list[slice]:
    if times.size == 0:
        return []
    if times.size == 1:
        return [slice(0, 1)]
    max_gap = 1.5 / max(float(sampling_rate_hz), 1.0)
    breaks = np.flatnonzero(np.diff(times) > max_gap) + 1
    starts = np.r_[0, breaks]
    stops = np.r_[breaks, times.size]
    return [slice(int(start), int(stop)) for start, stop in zip(starts, stops) if stop > start]


def make_trace(data: np.ndarray, start_timestamp: float, sampling_rate_hz: float, hinet_station: str, component: str, channel: str):
    from obspy import Trace, UTCDateTime

    tr = Trace(data=np.asarray(data, dtype=np.int32))
    tr.stats.network = str(hinet_station).split(".", 1)[0][:2] or "HN"
    tr.stats.station = safe_station_code(str(hinet_station).split(".")[-1])
    tr.stats.location = ""
    tr.stats.channel = channel
    tr.stats.starttime = UTCDateTime(float(start_timestamp))
    tr.stats.sampling_rate = float(sampling_rate_hz)
    tr.stats.mseed = {"dataquality": "D"}
    tr.stats.hinet_component = component
    return tr


def traces_for_channel(
    times: np.ndarray,
    values: np.ndarray,
    cut_start: float,
    cut_end: float,
    pad: bool,
    hinet_station: str,
    component: str,
    out_channel: str,
) -> list:
    sampling_rate_hz = infer_sampling_rate_hz(times)
    if pad:
        npts = max(1, int(math.ceil((cut_end - cut_start) * sampling_rate_hz)))
        data = np.zeros(npts, dtype=np.int32)
        sample_idx = np.rint((times - cut_start) * sampling_rate_hz).astype(np.int64)
        mask = (sample_idx >= 0) & (sample_idx < npts)
        data[sample_idx[mask]] = values[mask].astype(np.int32, copy=False)
        return [make_trace(data, cut_start, sampling_rate_hz, hinet_station, component, out_channel)]

    mask = (times >= cut_start) & (times <= cut_end)
    if not np.any(mask):
        return []
    t = times[mask]
    y = values[mask].astype(np.int32, copy=False)
    traces = []
    for part in contiguous_slices(t, sampling_rate_hz):
        traces.append(make_trace(y[part], float(t[part][0]), sampling_rate_hz, hinet_station, component, out_channel))
    return traces


def write_station_mseed_from_series(args, arr: pd.Series, station_channels: pd.DataFrame, series_by_id: dict[str, tuple[np.ndarray, np.ndarray]], out_path: Path) -> dict[str, object]:
    from obspy import Stream

    traces = []
    component_ids = []
    component_names = []
    for _, ch in station_channels.iterrows():
        comp = str(ch.get("component", "")).strip().upper()
        out_channel = COMPONENT_MAP.get(comp)
        channel_id = str(ch["channel_id"]).lower()
        if out_channel is None or channel_id not in series_by_id:
            continue
        times, values = series_by_id[channel_id]
        new_traces = traces_for_channel(
            times=times,
            values=values,
            cut_start=float(arr["cut_start_timestamp"]),
            cut_end=float(arr["cut_end_timestamp"]),
            pad=bool(args.pad_mseed),
            hinet_station=str(arr["hinet_station"]),
            component=comp,
            out_channel=out_channel,
        )
        if new_traces:
            traces.extend(new_traces)
            component_ids.append(str(ch["channel_id"]))
            component_names.append(comp)
    if not traces:
        return {"mseed_status": "no_matching_traces", "mseed_error": "", "mseed_path": ""}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Stream(traces=traces).write(str(out_path), format="MSEED")
    n_components = len(set(component_names))
    return {
        "mseed_status": "written" if n_components >= 3 else "partial_written",
        "mseed_error": "",
        "mseed_path": str(out_path),
        "mseed_trace_count": len(traces),
        "mseed_component_count": n_components,
        "mseed_channel_ids": ",".join(component_ids),
    }


def convert_event_segments_to_mseed(
    args,
    event: EventInfo,
    cnt_paths: Iterable[Path],
    ch_path: Path,
    arrivals: pd.DataFrame,
    source: str = "python_win32",
    inherited_error: str = "",
) -> pd.DataFrame:
    mseed_dir = args.output_root / "mseed" / event.event_id
    mseed_dir.mkdir(parents=True, exist_ok=True)
    ctable = read_channel_table(ch_path)
    ctable_path = args.output_root / "responses" / event.event_id / f"{ch_path.stem}.channels.csv"
    write_csv(ctable_path, ctable.to_dict(orient="records"), list(ctable.columns) if not ctable.empty else ["channel_id"])

    hinet_stations = set(arrivals["hinet_station"].astype(str))
    wanted_rows = ctable[
        ctable["hinet_station"].astype(str).isin(hinet_stations)
        & ctable["component"].astype(str).str.strip().str.upper().isin(list(COMPONENT_MAP))
    ].copy()
    wanted_ids = {str(x).lower() for x in wanted_rows["channel_id"]} if not wanted_rows.empty else set()
    rows: list[dict[str, object]] = []
    if not wanted_ids:
        out = arrivals.copy()
        out["mseed_status"] = "no_channels_in_ctable"
        out["mseed_error"] = inherited_error
        out["mseed_path"] = ""
        out["mseed_source"] = source
        return out

    try:
        series_by_id = read_hinet_vm_cnt_segments(tuple(cnt_paths), wanted_ids)
    except Exception as exc:
        out = arrivals.copy()
        out["mseed_status"] = "python_win_read_failed"
        out["mseed_error"] = f"{inherited_error}; {exc!r}".strip("; ")
        out["mseed_path"] = ""
        out["mseed_source"] = source
        return out

    for _, arr in arrivals.iterrows():
        row = dict(arr)
        station_channels = wanted_rows[wanted_rows["hinet_station"].astype(str) == str(arr["hinet_station"])]
        if station_channels.empty:
            row.update({"mseed_status": "no_channels_in_ctable", "mseed_error": inherited_error, "mseed_path": "", "mseed_source": source})
            rows.append(row)
            continue
        out_path = mseed_dir / f"{arr['knet_station']}__{arr['hinet_station']}.mseed".replace("/", "_")
        try:
            result = write_station_mseed_from_series(args, arr, station_channels, series_by_id, out_path)
            result["mseed_source"] = source
            if inherited_error and result.get("mseed_status") in {"written", "partial_written"}:
                result["mseed_error"] = ""
                result["mseed_fallback_note"] = inherited_error
            row.update(result)
        except Exception as exc:
            row.update({
                "mseed_status": "write_failed",
                "mseed_error": f"{inherited_error}; {exc!r}".strip("; "),
                "mseed_path": "",
                "mseed_source": source,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def convert_event_to_mseed(args, event: EventInfo, cnt_path: Path, ch_path: Path, arrivals: pd.DataFrame) -> pd.DataFrame:
    mseed_dir = args.output_root / "mseed" / event.event_id
    mseed_dir.mkdir(parents=True, exist_ok=True)
    ctable = read_channel_table(ch_path)
    ctable_path = args.output_root / "responses" / event.event_id / f"{ch_path.stem}.channels.csv"
    write_csv(ctable_path, ctable.to_dict(orient="records"), list(ctable.columns) if not ctable.empty else ["channel_id"])

    try:
        from obspy import UTCDateTime, read

        stream = read(str(cnt_path), format="WIN", century=args.win_century)
    except Exception as exc:
        return convert_event_segments_to_mseed(
            args,
            event,
            [cnt_path],
            ch_path,
            arrivals,
            source="python_win32_after_obspy_failed",
            inherited_error=f"obspy_win_read_failed:{exc!r}",
        )

    rows: list[dict[str, object]] = []
    for _, arr in arrivals.iterrows():
        row = dict(arr)
        station_channels = ctable[ctable["hinet_station"].astype(str) == str(arr["hinet_station"])]
        if station_channels.empty:
            row.update({"mseed_status": "no_channels_in_ctable", "mseed_error": "", "mseed_path": ""})
            rows.append(row)
            continue
        traces = []
        component_ids = []
        for _, ch in station_channels.iterrows():
            comp = str(ch.get("component", "")).strip().upper()
            out_channel = COMPONENT_MAP.get(comp)
            if out_channel is None:
                continue
            selected = stream.select(channel=str(ch["channel_id"]))
            if len(selected) == 0:
                continue
            tr = selected[0].copy()
            tr.stats.network = str(ch.get("hinet_station", arr["hinet_station"])).split(".", 1)[0][:2] or "HN"
            tr.stats.station = safe_station_code(str(arr["hinet_station"]).split(".")[-1])
            tr.stats.location = ""
            tr.stats.channel = out_channel
            tr.stats.mseed = {"dataquality": "D"}
            tr.trim(
                UTCDateTime(float(arr["cut_start_timestamp"])),
                UTCDateTime(float(arr["cut_end_timestamp"])),
                pad=bool(args.pad_mseed),
                fill_value=0,
                nearest_sample=False,
            )
            if tr.stats.npts > 0:
                traces.append(tr)
                component_ids.append(str(ch["channel_id"]))
        if not traces:
            row.update({"mseed_status": "no_matching_traces", "mseed_error": "", "mseed_path": ""})
            rows.append(row)
            continue
        out_path = mseed_dir / f"{arr['knet_station']}__{arr['hinet_station']}.mseed".replace("/", "_")
        try:
            from obspy import Stream

            Stream(traces=traces).write(str(out_path), format="MSEED")
            row.update({
                "mseed_status": "written" if len(traces) >= 3 else "partial_written",
                "mseed_error": "",
                "mseed_path": str(out_path),
                "mseed_trace_count": len(traces),
                "mseed_component_count": len(traces),
                "mseed_channel_ids": ",".join(component_ids),
                "mseed_source": "obspy_win",
            })
        except Exception as exc:
            row.update({"mseed_status": "write_failed", "mseed_error": repr(exc), "mseed_path": ""})
        rows.append(row)
    return pd.DataFrame(rows)


def build_event_arrivals(
    event: EventInfo,
    station_rows: pd.DataFrame,
    matches: pd.DataFrame,
    jma_table,
    ak135,
    pre_seconds: float,
    post_seconds: float,
) -> pd.DataFrame:
    station_key = find_column(station_rows.columns, ["station_code", "Station Code"])
    event_key = find_column(station_rows.columns, ["EVENT", "KiK_File", "#EventID"])
    lat_key = find_column(station_rows.columns, ["station_lat", "Station Lat.", "latitude"])
    lon_key = find_column(station_rows.columns, ["station_lon", "Station Long.", "longitude"])
    height_key = find_column(station_rows.columns, ["station_height_m", "Station Height(m)"], required=False)

    event_stations = station_rows[station_rows[event_key].astype(str) == event.event_id].copy()
    accepted = matches[matches["accepted"].astype(int) == 1].copy()
    accepted = accepted.sort_values("match_distance_km").drop_duplicates("knet_station")
    merged = event_stations.merge(accepted, left_on=station_key, right_on="knet_station", how="inner")
    optional_training_cols = (
        "record_start_time_jst",
        "trigger_time_jst",
        "record_start_sample",
        "valid_n_samples",
        "sampling_rate_hz",
        "p_pick_predicted_seconds_after_origin",
        "p_pick_theoretical_record_offset_seconds",
        "p_pick_theoretical_inside_record",
        "p_pick_theoretical_inside_allowed_window",
        "p_pick_repair_clip_reason",
        "p_pick_repaired_source",
        "p_pick_search_source",
        "p_pick_refined_source",
    )
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        p_seconds, model, fallback, epi_km = predict_p_seconds(
            event,
            float(row["hinet_lat"]),
            float(row["hinet_lon"]),
            float(row.get("hinet_elevation_m", row[height_key] if height_key else 0.0)),
            jma_table,
            ak135,
        )
        ppick_ts = event.origin_timestamp + p_seconds
        out = {
            "event_id": event.event_id,
            "origin_time_jst": event.origin_time_jst,
            "origin_timestamp": event.origin_timestamp,
            "event_lat": event.latitude,
            "event_lon": event.longitude,
            "event_depth_km": event.depth_km,
            "event_magnitude": "" if event.magnitude is None else event.magnitude,
            "origin_source": event.origin_source,
            "knet_station": str(row[station_key]),
            "knet_lat": float(row[lat_key]),
            "knet_lon": float(row[lon_key]),
            "knet_height_m": float(row[height_key]) if height_key else "",
            "hinet_network": str(row["hinet_network"]),
            "hinet_station": str(row["hinet_station"]),
            "hinet_lat": float(row["hinet_lat"]),
            "hinet_lon": float(row["hinet_lon"]),
            "hinet_elevation_m": float(row.get("hinet_elevation_m", 0.0)),
            "match_distance_km": float(row["match_distance_km"]),
            "epicentral_distance_km": epi_km,
            "jma2001_p_travel_time_sec": p_seconds if model == "jma2001a" else "",
            "ak135_p_travel_time_sec": p_seconds if model == "ak135" else "",
            "p_seconds_after_origin": p_seconds,
            "ppick_timestamp": ppick_ts,
            "ppick_time_jst": datetime.fromtimestamp(ppick_ts, tz=JST).isoformat(),
            "cut_start_timestamp": ppick_ts - pre_seconds,
            "cut_end_timestamp": ppick_ts + post_seconds,
            "cut_start_jst": datetime.fromtimestamp(ppick_ts - pre_seconds, tz=JST).isoformat(),
            "cut_end_jst": datetime.fromtimestamp(ppick_ts + post_seconds, tz=JST).isoformat(),
            "travel_time_model": model,
            "travel_time_status": "fallback" if fallback else "ok",
        }
        for col in optional_training_cols:
            if col in row.index:
                value = row[col]
                out[f"training_{col}"] = "" if pd.isna(value) else value
        if "record_start_time_jst" in row.index and "valid_n_samples" in row.index and "sampling_rate_hz" in row.index:
            try:
                record_start = pd.Timestamp(row["record_start_time_jst"])
                if record_start.tzinfo is None:
                    record_start = record_start.tz_localize(JST)
                record_start_ts = float(record_start.timestamp())
                sampling_rate_hz = float(row["sampling_rate_hz"])
                valid_n_samples = int(row["valid_n_samples"])
                record_end_ts = record_start_ts + max(valid_n_samples - 1, 0) / sampling_rate_hz
                out["training_record_start_minus_hinet_theoretical_p_s"] = record_start_ts - ppick_ts
                out["training_record_end_minus_hinet_theoretical_p_s"] = record_end_ts - ppick_ts
                out["training_hinet_theoretical_p_inside_record"] = int(record_start_ts <= ppick_ts <= record_end_ts)
            except Exception as exc:
                out["training_record_window_error"] = repr(exc)
        rows.append(out)
    return pd.DataFrame(rows)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_dataframe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def process_events(args) -> None:
    selected_ids = select_event_ids(args.hdf5, args.mode, args.event_ids, args.num_events, args.seed)
    origin_corrections = load_origin_corrections(args.origin_corrections, require_accepted=not args.use_unaccepted_origin_corrections)
    events = load_events(args.hdf5, selected_ids, origin_corrections=origin_corrections)
    if selected_ids is not None:
        missing = sorted(set(selected_ids) - set(events))
        if missing:
            preview = ", ".join(missing[:5])
            suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} missing total)"
            print(f"[WARN] {len(missing)} selected event ids were not found in HDF5: {preview}{suffix}", file=sys.stderr)
    station_df = load_station_table(args.hdf5)
    hinet_df = load_or_download_hinet_inventory(args)
    matches = build_station_matches(args, station_df, hinet_df)
    accepted_count = int((matches["accepted"].astype(int) == 1).sum()) if not matches.empty else 0
    print(f"[INFO] loaded {len(events)} events; accepted station matches: {accepted_count}/{len(matches)}")
    if origin_corrections:
        used_corrections = len(set(events) & set(origin_corrections))
        print(f"[INFO] loaded {len(origin_corrections)} origin corrections; used {used_corrections} for selected events")

    jma_table, ak135 = build_travel_time_model(args.jma_travel_time_zip)
    client = None if args.dry_run else make_hinet_client()
    all_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for event_id in sorted(events):
        event = events[event_id]
        print(f"[INFO] event {event_id} origin={event.origin_time_jst}")
        try:
            arrivals = build_event_arrivals(
                event,
                station_df,
                matches,
                jma_table,
                ak135,
                pre_seconds=args.pre_seconds,
                post_seconds=args.post_seconds,
            )
            if arrivals.empty:
                summary_rows.append({"event_id": event_id, "status": "no_matched_stations", "stations": 0})
                continue
            write_dataframe(args.output_root / "manifests" / "events" / event_id / "arrivals.csv", arrivals)
            raw = download_raw_event(args, client, event, arrivals)
            arrivals["raw_status"] = raw.raw_status
            arrivals["raw_error"] = raw.raw_error
            arrivals["raw_count_path"] = "" if raw.cnt_path is None else str(raw.cnt_path)
            arrivals["raw_segment_paths"] = ";".join(str(path) for path in raw.segment_paths)
            arrivals["raw_segment_count"] = len(raw.segment_paths)
            arrivals["channel_table_path"] = "" if raw.ch_path is None else str(raw.ch_path)

            if raw.ch_path is not None:
                arrivals["response_status"] = cache_response_files(args, raw.ch_path, event_id)
            else:
                arrivals["response_status"] = "not_available"

            if raw.cnt_path is not None and raw.ch_path is not None and args.write_mseed:
                arrivals = convert_event_to_mseed(args, event, raw.cnt_path, raw.ch_path, arrivals)
            elif raw.segment_paths and raw.ch_path is not None and args.write_mseed:
                arrivals = convert_event_segments_to_mseed(
                    args,
                    event,
                    raw.segment_paths,
                    raw.ch_path,
                    arrivals,
                    source="python_win32_segments",
                    inherited_error=raw.raw_error,
                )
            elif not args.write_mseed:
                arrivals["mseed_status"] = "disabled"
            elif args.dry_run:
                arrivals["mseed_status"] = "dry_run"
            else:
                arrivals["mseed_status"] = "raw_unavailable"

            write_dataframe(args.output_root / "manifests" / "events" / event_id / "download_manifest.csv", arrivals)
            all_rows.append(arrivals)
            summary_rows.append({
                "event_id": event_id,
                "status": raw.raw_status,
                "stations": len(arrivals),
                "raw_error": raw.raw_error,
            })
            if args.sleep_seconds > 0 and not args.dry_run:
                time.sleep(args.sleep_seconds)
        except Exception as exc:
            print(f"[WARN] event {event_id} failed: {exc!r}", file=sys.stderr)
            summary_rows.append({"event_id": event_id, "status": "failed", "stations": 0, "raw_error": repr(exc)})

        if all_rows:
            write_dataframe(args.output_root / "manifests" / "download_manifest.csv", pd.concat(all_rows, ignore_index=True))
        write_csv(args.output_root / "manifests" / "summary.csv", summary_rows, ["event_id", "status", "stations", "raw_error"])

    if all_rows:
        write_dataframe(args.output_root / "manifests" / "download_manifest.csv", pd.concat(all_rows, ignore_index=True))
    write_csv(args.output_root / "manifests" / "summary.csv", summary_rows, ["event_id", "status", "stations", "raw_error"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download matched Hi-net velocity windows for K-NET/KiK-net events.")
    parser.add_argument("--hdf5", type=Path, required=True, help="Converted K-NET/KiK-net HDF5 file.")
    parser.add_argument("--output-root", type=Path, default=Path("hinet_velocity_downloads"))
    parser.add_argument("--mode", choices=["all", "smoketest"], default="all")
    parser.add_argument("--num-events", type=int, default=3, help="Number of events for --mode smoketest.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --mode smoketest.")
    parser.add_argument("--event-ids", type=Path, default=None, help="Optional event-id file such as stage2_512_event_ids.txt.")
    parser.add_argument("--origin-corrections", type=Path, default=None, help="CSV from tools/fetch_jma_hypocenters.py.")
    parser.add_argument(
        "--use-unaccepted-origin-corrections",
        action="store_true",
        help="Use all rows in --origin-corrections instead of filtering accepted==1.",
    )
    parser.add_argument("--hinet-network", default="0101", help="Hi-net network code used by HinetPy.")
    parser.add_argument("--inventory-csv", type=Path, default=None)
    parser.add_argument("--match-csv", type=Path, default=None)
    parser.add_argument("--match-distance-km", type=float, default=1.0)
    parser.add_argument("--pre-seconds", type=float, default=120.0)
    parser.add_argument("--post-seconds", type=float, default=120.0)
    parser.add_argument("--jma-travel-time-zip", type=Path, default=None)
    parser.add_argument("--win-century", default="20", help="Century prefix for ObsPy WIN timestamp parser.")
    parser.add_argument("--write-mseed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pad-mseed", action="store_true", help="Pad cut MiniSEED traces with zero counts if coverage is short.")
    parser.add_argument("--dry-run", action="store_true", help="Plan events/windows without downloading raw waveforms.")
    parser.add_argument("--allow-network-in-dry-run", action="store_true")
    parser.add_argument("--overwrite-inventory", action="store_true")
    parser.add_argument("--overwrite-matches", action="store_true")
    parser.add_argument("--overwrite-raw", action="store_true")
    parser.add_argument("--overwrite-responses", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.hdf5 = args.hdf5.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.inventory_csv is None:
        args.inventory_csv = args.output_root / "inventory" / "hinet_stations.csv"
    else:
        args.inventory_csv = args.inventory_csv.expanduser().resolve()
    if args.match_csv is None:
        args.match_csv = args.output_root / "matches" / "hinet_kiknet_station_matches.csv"
    else:
        args.match_csv = args.match_csv.expanduser().resolve()
    if args.event_ids is not None:
        args.event_ids = args.event_ids.expanduser().resolve()
    if args.origin_corrections is not None:
        args.origin_corrections = args.origin_corrections.expanduser().resolve()
    if args.jma_travel_time_zip is not None:
        args.jma_travel_time_zip = args.jma_travel_time_zip.expanduser().resolve()
    if not args.hdf5.exists():
        raise FileNotFoundError(args.hdf5)
    process_events(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
