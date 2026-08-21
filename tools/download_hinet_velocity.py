#!/usr/bin/env python3
"""Download Hi-net raw-count waveforms for existing K-NET/KiK-net events.

The default storage policy is intentionally conservative:

* Hi-net credentials are read from HINET_USER/HINET_PASSWORD.
* K-NET/KiK-net stations are matched to Hi-net stations by horizontal distance.
* Native CNT and channel-table bytes are committed once to an annual HDF5
  archive, with checksums and byte-offset indexes.
* Temporary Hi-net files are removed only after the event commit is flushed.
* MiniSEED and per-channel SAC PZ files are opt-in legacy products; the annual
  archive never persists a second waveform representation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import platform
import re
import shlex
import shutil
import sys
import tempfile
import time
import zipfile
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
from tools.hinet_raw_archive import (  # noqa: E402
    AnnualHinetArchiveReader,
    AnnualHinetArchiveWriter,
    partial_archive_path,
    read_hinet_vm_cnt_paths,
    scan_hinet_vm_cnt_paths,
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
    origin_time_jst_raw: str
    origin_time_correction_s: float | None
    origin_time_correction_status: str
    origin_time_correction_source: str
    origin_time_jma_event_id: str


@dataclass(frozen=True)
class RawDownloadResult:
    cnt_path: Path | None
    ch_path: Path | None
    segment_paths: tuple[Path, ...]
    raw_status: str
    raw_error: str
    batch_count: int = 1
    download_strategy: str = "single_request"


@dataclass(frozen=True)
class RawBatchResult:
    segment_paths: tuple[Path, ...]
    channel_path: Path | None
    success: bool
    used_minute_fallback: bool
    detail: str


class HinetDownloadError(RuntimeError):
    """A Hi-net transport failure with the swallowed HinetPy cause restored."""


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


def clean_metadata_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def clean_metadata_float(value: object) -> float | None:
    text = clean_metadata_text(value)
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


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
    origin_raw_key = find_column(
        event_df.columns,
        ["Origin_Time(JST)_Raw", "origin_time_jst_raw"],
        required=False,
    )
    correction_s_key = find_column(
        event_df.columns,
        ["Origin_Time_Correction_S", "origin_time_correction_s"],
        required=False,
    )
    correction_status_key = find_column(
        event_df.columns,
        ["Origin_Time_Correction_Status", "origin_time_correction_status"],
        required=False,
    )
    correction_source_key = find_column(
        event_df.columns,
        ["Origin_Time_Correction_Source", "origin_time_correction_source"],
        required=False,
    )
    correction_jma_id_key = find_column(
        event_df.columns,
        ["Origin_Time_JMA_Event_ID", "origin_time_jma_event_id"],
        required=False,
    )

    events: dict[str, EventInfo] = {}
    for _, row in event_df.iterrows():
        event_id = str(row[event_key])
        if selected_event_ids is not None and event_id not in selected_event_ids:
            continue
        origin_value = row[origin_key] if origin_key else None
        origin_iso, origin_ts, origin_source = parse_origin_time(origin_value, event_id)
        origin_time_jst_raw = clean_metadata_text(row[origin_raw_key]) if origin_raw_key else ""
        correction_s = clean_metadata_float(row[correction_s_key]) if correction_s_key else None
        correction_status = clean_metadata_text(row[correction_status_key]) if correction_status_key else ""
        correction_source = clean_metadata_text(row[correction_source_key]) if correction_source_key else ""
        correction_jma_id = clean_metadata_text(row[correction_jma_id_key]) if correction_jma_id_key else ""
        if origin_corrections and event_id in origin_corrections:
            correction = origin_corrections[event_id]
            origin_iso = str(correction["origin_time_jst"])
            origin_ts = float(correction["origin_timestamp"])
            status = correction.get("match_status", "")
            delta = correction.get("origin_time_correction_s", "")
            origin_source = f"jma_origin_correction:{status}:delta_s={delta}"
            correction_s = clean_metadata_float(delta)
            correction_status = clean_metadata_text(status)
            correction_source = clean_metadata_text(correction.get("source", ""))
        events[event_id] = EventInfo(
            event_id=event_id,
            origin_time_jst=origin_iso,
            origin_timestamp=float(origin_ts),
            latitude=float(row[lat_key]),
            longitude=float(row[lon_key]),
            depth_km=float(row[depth_key]),
            magnitude=None if mag_key is None or pd.isna(row[mag_key]) else float(row[mag_key]),
            origin_source=origin_source,
            origin_time_jst_raw=origin_time_jst_raw,
            origin_time_correction_s=correction_s,
            origin_time_correction_status=correction_status,
            origin_time_correction_source=correction_source,
            origin_time_jma_event_id=correction_jma_id,
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

    client = make_hinet_client(args)
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


def _diagnostic_hinet_client_class(base_client_class):
    """Return a Client subclass that propagates timeout and preserves errors.

    HinetPy 0.12.0 creates a fresh download client with its default 60-second
    timeout and catches every download/unzip exception without reporting it.
    The override intentionally mirrors only that private transport method.  It
    keeps the public request/splitting logic while making failures actionable.
    """

    class DiagnosticHinetClient(base_client_class):
        def _download_cont_waveform(self, job):
            download_client = base_client_class(
                self.user,
                self.password,
                timeout=self.timeout,
                retries=self.retries,
                sleep_time_in_seconds=self.sleep_time_in_seconds,
                max_sleep_count=self.max_sleep_count,
            )
            failures: list[str] = []
            last_exception: Exception | None = None
            try:
                for attempt in range(1, int(self.retries) + 1):
                    response = None
                    downloaded_bytes = 0
                    try:
                        response = download_client.session.post(
                            self._CONT_DOWNLOAD,
                            data={"id": job.id},
                            stream=True,
                            timeout=self.timeout,
                        )
                        response.raise_for_status()
                        with tempfile.NamedTemporaryFile() as temporary:
                            for chunk in response.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    temporary.write(chunk)
                                    downloaded_bytes += len(chunk)
                            temporary.flush()
                            with zipfile.ZipFile(temporary.name) as archive:
                                names = archive.namelist()
                                cnt_names = sorted(name for name in names if name.endswith(".cnt"))
                                channel_names = sorted(
                                    name for name in names if name.endswith(".euc.ch")
                                )
                                if not cnt_names:
                                    raise HinetDownloadError(
                                        "Hi-net ZIP contained no CNT members "
                                        f"(job={job.id}, bytes={downloaded_bytes}, "
                                        f"members={names[:8]})"
                                    )
                                if not channel_names:
                                    raise HinetDownloadError(
                                        "Hi-net ZIP contained no .euc.ch member "
                                        f"(job={job.id}, bytes={downloaded_bytes}, "
                                        f"members={names[:8]})"
                                    )
                                channel_name = channel_names[0]
                                archive.extractall(members=[*cnt_names, channel_name])
                                return cnt_names, channel_name
                    except Exception as exc:  # noqa: BLE001 - preserve provider cause
                        last_exception = exc
                        status = getattr(response, "status_code", "not_received")
                        detail = (
                            f"attempt={attempt}/{self.retries} status={status} "
                            f"bytes={downloaded_bytes} error={type(exc).__name__}: {exc}"
                        )
                        failures.append(detail)
                        print(f"[WARN] Hi-net download job {job.id}: {detail}", file=sys.stderr)
                    finally:
                        if response is not None:
                            response.close()
                joined = " | ".join(failures)
                raise HinetDownloadError(
                    f"Hi-net download job {job.id} failed after {self.retries} attempts: {joined}"
                ) from last_exception
            finally:
                session = getattr(download_client, "session", None)
                if session is not None:
                    session.close()

    DiagnosticHinetClient.__name__ = "DiagnosticHinetClient"
    return DiagnosticHinetClient


def make_hinet_client(args=None):
    user = os.environ.get("HINET_USER")
    password = os.environ.get("HINET_PASSWORD")
    if not user or not password:
        raise RuntimeError("Set HINET_USER and HINET_PASSWORD before accessing Hi-net.")
    try:
        from HinetPy import Client
    except ImportError as exc:
        raise ImportError("HinetPy is required for Hi-net downloads. Install it with `pip install HinetPy`.") from exc
    timeout = float(getattr(args, "hinet_timeout_seconds", 300.0))
    retries = int(getattr(args, "hinet_retries", 3))
    if timeout <= 0:
        raise ValueError("--hinet-timeout-seconds must be positive")
    if retries <= 0:
        raise ValueError("--hinet-retries must be positive")
    client_class = _diagnostic_hinet_client_class(Client)
    return client_class(user, password, timeout=timeout, retries=retries)


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
        cached = pd.read_csv(match_path, dtype={"knet_station": str, "hinet_station": str})
        if "match_threshold_km" in cached.columns and not cached.empty:
            cached_thresholds = pd.to_numeric(cached["match_threshold_km"], errors="coerce").dropna()
            if not cached_thresholds.empty and not np.allclose(
                cached_thresholds.to_numpy(dtype=float),
                float(args.match_distance_km),
            ):
                raise ValueError(
                    f"Cached station matches in {match_path} use a different threshold. "
                    "Pass --overwrite-matches and use a new annual archive path."
                )
        return cached

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
    return tuple(sorted(p for p in segment_dir.rglob("*.cnt") if p.is_file()))


def _find_segment_channel_table(segment_dir: Path) -> Path | None:
    candidates = sorted([p for p in segment_dir.rglob("*.euc.ch") if p.is_file()])
    candidates.extend(sorted([p for p in segment_dir.rglob("*.ch") if p.is_file() and p not in candidates]))
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
    *,
    max_span: int | None = None,
    threads: int = 1,
    cleanup: bool = False,
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
            max_span=max_span,
            threads=threads,
            cleanup=cleanup,
        )
    finally:
        os.chdir(old_cwd)


def _compact_error(value: object, maximum: int = 1200) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= maximum else text[: maximum - 3] + "..."


def _reset_staging_directory(path: Path, allowed_root: Path) -> None:
    path = Path(path)
    allowed_root = Path(allowed_root).resolve()
    resolved = path.resolve()
    if resolved == allowed_root or allowed_root not in resolved.parents:
        raise RuntimeError(f"Refusing to reset staging directory outside {allowed_root}: {resolved}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _normalize_channel_id(value: object) -> str:
    channel_id = str(value).strip().lower()
    if channel_id.startswith("0x"):
        channel_id = channel_id[2:]
    return channel_id.zfill(4)


def _requested_station_channels(
    ch_path: Path,
    stations: Iterable[str],
) -> tuple[dict[str, dict[str, str]], list[str]]:
    table = read_channel_table(ch_path)
    if table.empty:
        return {}, sorted(set(str(value).upper() for value in stations))
    station_rows: dict[str, pd.DataFrame] = {}
    for station, rows in table.groupby(table["hinet_station"].astype(str).str.upper()):
        station_rows[str(station)] = rows
    requested: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    for station in sorted(set(str(value).upper() for value in stations)):
        rows = station_rows.get(station)
        if rows is None:
            missing.append(station)
            continue
        components = rows["component"].astype(str).str.strip().str.upper()
        channel_ids: dict[str, str] = {}
        for canonical, aliases in (("U", {"U", "Z"}), ("N", {"N", "1"}), ("E", {"E", "2"})):
            matches = rows[components.isin(aliases)]
            if matches.empty:
                break
            channel_ids[canonical] = _normalize_channel_id(matches.iloc[0]["channel_id"])
        if len(channel_ids) != 3:
            missing.append(station)
            continue
        requested[station] = channel_ids
    return requested, missing


def _channel_table_missing_stations(ch_path: Path, stations: Iterable[str]) -> list[str]:
    _, missing = _requested_station_channels(ch_path, stations)
    return missing


def _validate_station_channel_samples(
    cnt_paths: Iterable[Path],
    requested_channels: dict[str, dict[str, str]],
    *,
    request_start_timestamp: float,
    request_end_timestamp: float,
    tolerance_seconds: float,
) -> tuple[bool, str]:
    wanted = {
        channel_id
        for station_channels in requested_channels.values()
        for channel_id in station_channels.values()
    }
    try:
        series = read_hinet_vm_cnt_paths(cnt_paths, wanted)
    except Exception as exc:
        return False, f"requested-channel CNT decode failed: {exc!r}"

    failures: list[str] = []
    expected_start_second = math.floor(request_start_timestamp)
    expected_end_second = math.ceil(request_end_timestamp)
    expected_seconds = max(0, expected_end_second - expected_start_second)
    allowed_missing_seconds = max(0, int(math.floor(tolerance_seconds)))
    for station, component_channels in requested_channels.items():
        for component, channel_id in component_channels.items():
            item = series.get(channel_id)
            label = f"{station}.{component}/{channel_id}"
            if item is None or item[0].size == 0:
                failures.append(f"{label}:absent")
                continue
            times = np.asarray(item[0], dtype=np.float64)
            times = times[np.isfinite(times)]
            if times.size == 0:
                failures.append(f"{label}:no-finite-times")
                continue
            unique_seconds = np.unique(np.floor(times + 1.0e-6).astype(np.int64))
            covered_seconds = unique_seconds[
                (unique_seconds >= expected_start_second)
                & (unique_seconds < expected_end_second)
            ].size
            missing_seconds = max(0, expected_seconds - int(covered_seconds))
            if (
                float(times[0]) > request_start_timestamp + tolerance_seconds
                or float(times[-1]) < request_end_timestamp - tolerance_seconds
                or missing_seconds > allowed_missing_seconds
            ):
                failures.append(
                    f"{label}:samples={times.size},seconds={covered_seconds}/{expected_seconds},"
                    f"range=[{times[0]:.3f},{times[-1]:.3f}]"
                )
    if failures:
        preview = "; ".join(failures[:6])
        suffix = "" if len(failures) <= 6 else f"; ...({len(failures)} channels failed)"
        return False, "incomplete requested-channel coverage: " + preview + suffix
    return True, "ok"


def _validate_raw_request(
    cnt_paths: Iterable[Path],
    ch_path: Path | None,
    *,
    request_start_timestamp: float,
    request_end_timestamp: float,
    stations: Iterable[str],
    tolerance_seconds: float = 1.1,
) -> tuple[bool, str]:
    cnt_paths = tuple(Path(path) for path in cnt_paths)
    if not cnt_paths:
        return False, "no CNT files returned"
    if ch_path is None or not Path(ch_path).is_file():
        return False, "no channel table returned"
    empty = [str(path) for path in [*cnt_paths, Path(ch_path)] if path.stat().st_size <= 0]
    if empty:
        return False, "empty files: " + ", ".join(empty)
    try:
        coverage = scan_hinet_vm_cnt_paths(cnt_paths)
    except Exception as exc:
        return False, f"CNT coverage scan failed: {exc!r}"
    if (
        coverage.record_count <= 0
        or not math.isfinite(coverage.start_timestamp)
        or not math.isfinite(coverage.end_timestamp)
        or coverage.start_timestamp > request_start_timestamp + tolerance_seconds
        or coverage.end_timestamp < request_end_timestamp - tolerance_seconds
    ):
        return (
            False,
            "incomplete CNT coverage: "
            f"records={coverage.record_count} "
            f"coverage=[{coverage.start_timestamp}, {coverage.end_timestamp}) "
            f"requested=[{request_start_timestamp}, {request_end_timestamp})",
        )
    requested_channels, missing = _requested_station_channels(Path(ch_path), stations)
    if missing:
        preview = ",".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f",...({len(missing)} total)"
        return False, f"channel table lacks complete 3-component stations: {preview}{suffix}"
    samples_valid, samples_detail = _validate_station_channel_samples(
        cnt_paths,
        requested_channels,
        request_start_timestamp=request_start_timestamp,
        request_end_timestamp=request_end_timestamp,
        tolerance_seconds=tolerance_seconds,
    )
    if not samples_valid:
        return False, samples_detail
    return True, "ok"


def _merge_channel_tables(paths: Iterable[Path], output_path: Path) -> Path:
    """Losslessly concatenate unique provider channel-table payloads."""
    payloads: list[bytes] = []
    seen_hashes: set[str] = set()
    for path in paths:
        payload = Path(path).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if payload and digest not in seen_hashes:
            seen_hashes.add(digest)
            payloads.append(payload)
    if not payloads:
        raise RuntimeError("Cannot merge empty Hi-net channel tables")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined = bytearray()
    for payload in payloads:
        if combined and not combined.endswith((b"\n", b"\r")):
            combined.extend(b"\n")
        combined.extend(payload)
    output_path.write_bytes(bytes(combined))
    return output_path


def _request_raw_window(
    args,
    client,
    *,
    request_dir: Path,
    allowed_root: Path,
    request_label: str,
    request_start_dt: datetime,
    span_minutes: int,
    stations: list[str],
) -> RawBatchResult:
    _reset_staging_directory(request_dir, allowed_root)
    segment_dir = request_dir / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{request_label}_{request_start_dt.strftime('%Y%m%dT%H%M')}_{span_minutes:04d}min"
    cnt_path = request_dir / f"{stem}.cnt"
    ch_path = request_dir / f"{stem}.ch"
    call_error = ""
    try:
        _call_hinet_in_directory(
            client=client,
            network=args.hinet_network,
            starttime=hinet_time_string(request_start_dt),
            span=int(span_minutes),
            data=cnt_path,
            ctable=ch_path,
            outdir=request_dir,
            cwd=segment_dir,
            max_span=min(int(span_minutes), 60),
            threads=int(getattr(args, "hinet_download_threads", 1)),
            cleanup=False,
        )
    except Exception as exc:  # native files may still be usable after merge failure
        call_error = _compact_error(repr(exc))

    cnt_paths = _list_segment_cnts(segment_dir)
    if not cnt_paths and cnt_path.is_file():
        cnt_paths = (cnt_path,)
    channel_path = _find_segment_channel_table(segment_dir)
    if channel_path is None and ch_path.is_file():
        channel_path = ch_path
    start_timestamp = request_start_dt.timestamp()
    end_timestamp = start_timestamp + int(span_minutes) * 60.0
    valid, validation = _validate_raw_request(
        cnt_paths,
        channel_path,
        request_start_timestamp=start_timestamp,
        request_end_timestamp=end_timestamp,
        stations=stations,
    )
    if valid:
        detail = "primary request valid"
        if call_error:
            detail += f"; HinetPy post-processing failed but native files passed: {call_error}"
        return RawBatchResult(cnt_paths, channel_path, True, False, detail)
    detail_parts = [validation]
    if call_error:
        detail_parts.append(f"transport={call_error}")
    return RawBatchResult(tuple(), channel_path, False, False, _compact_error("; ".join(detail_parts)))


def _download_station_batch(
    args,
    client,
    *,
    event_raw_dir: Path,
    event_id: str,
    batch_index: int,
    stations: list[str],
    request_start_dt: datetime,
    span_minutes: int,
) -> RawBatchResult:
    batch_root = event_raw_dir / "requests" / f"batch_{batch_index:03d}"
    batch_root.mkdir(parents=True, exist_ok=True)
    select_hinet_stations(client, args.hinet_network, stations)
    primary = _request_raw_window(
        args,
        client,
        request_dir=batch_root / "primary",
        allowed_root=event_raw_dir,
        request_label=f"{event_id}_b{batch_index:03d}",
        request_start_dt=request_start_dt,
        span_minutes=span_minutes,
        stations=stations,
    )
    if primary.success:
        return primary
    if not bool(getattr(args, "minute_fallback", True)):
        return primary

    fallback_span = max(1, int(getattr(args, "fallback_span_minutes", 1)))
    fallback_segments: list[Path] = []
    fallback_channels: list[Path] = []
    offset = 0
    while offset < span_minutes:
        sub_span = min(fallback_span, span_minutes - offset)
        sub_start = request_start_dt + timedelta(minutes=offset)
        result = _request_raw_window(
            args,
            client,
            request_dir=batch_root / "minute_fallback" / f"minute_{offset:04d}",
            allowed_root=event_raw_dir,
            request_label=f"{event_id}_b{batch_index:03d}_m{offset:04d}",
            request_start_dt=sub_start,
            span_minutes=sub_span,
            stations=stations,
        )
        if not result.success or result.channel_path is None:
            return RawBatchResult(
                tuple(),
                result.channel_path,
                False,
                True,
                _compact_error(
                    f"primary failed: {primary.detail}; minute fallback offset={offset} "
                    f"span={sub_span} failed: {result.detail}"
                ),
            )
        fallback_segments.extend(result.segment_paths)
        fallback_channels.append(result.channel_path)
        offset += sub_span
        delay = float(getattr(args, "subrequest_sleep_seconds", 0.0))
        if delay > 0 and offset < span_minutes:
            time.sleep(delay)

    combined_channel = _merge_channel_tables(
        fallback_channels,
        batch_root / f"{event_id}_b{batch_index:03d}_minute_fallback.euc.ch",
    )
    missing = _channel_table_missing_stations(combined_channel, stations)
    if missing:
        return RawBatchResult(
            tuple(),
            combined_channel,
            False,
            True,
            f"minute fallback channel table incomplete: {','.join(missing[:8])}",
        )
    return RawBatchResult(
        tuple(fallback_segments),
        combined_channel,
        True,
        True,
        _compact_error(f"minute fallback succeeded after primary failure: {primary.detail}"),
    )


def _stage_complete_event_download(
    event_raw_dir: Path,
    event_id: str,
    batch_results: list[RawBatchResult],
    stations: list[str],
) -> tuple[tuple[Path, ...], Path]:
    aggregate_root = event_raw_dir / "complete"
    _reset_staging_directory(aggregate_root, event_raw_dir)
    staged_segments: list[Path] = []
    channel_paths: list[Path] = []
    for batch_index, result in enumerate(batch_results):
        if not result.success or result.channel_path is None:
            raise RuntimeError(f"Cannot stage incomplete batch {batch_index}")
        channel_paths.append(result.channel_path)
        # Keep the files in their isolated request directories. The archive
        # index ordinal disambiguates duplicate provider filenames across
        # station batches, so their original names do not need to be rewritten.
        staged_segments.extend(result.segment_paths)
    channel_path = _merge_channel_tables(
        channel_paths,
        aggregate_root / f"{event_id}_combined.euc.ch",
    )
    missing = _channel_table_missing_stations(channel_path, stations)
    if missing:
        raise RuntimeError(
            "combined channel table lacks requested stations: " + ",".join(missing[:8])
        )
    return tuple(staged_segments), channel_path


def download_raw_event(args, client, event: EventInfo, arrivals: pd.DataFrame) -> RawDownloadResult:
    raw_work_root = getattr(args, "raw_work_root", args.output_root / "raw")
    event_raw_dir = raw_work_root / event.event_id
    event_raw_dir.mkdir(parents=True, exist_ok=True)
    segment_dir = event_raw_dir / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)

    start_ts = float(arrivals["cut_start_timestamp"].min())
    end_ts = float(arrivals["cut_end_timestamp"].max())
    request_start_dt = floor_to_minute(start_ts)
    request_start_ts = request_start_dt.timestamp()
    span_min = ceil_span_minutes(request_start_ts, end_ts)
    request_end_ts = request_start_ts + span_min * 60.0
    stations = sorted(set(str(x) for x in arrivals["hinet_station"]))

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
        valid, detail = _validate_raw_request(
            (cnt_path,),
            ch_path,
            request_start_timestamp=request_start_ts,
            request_end_timestamp=request_end_ts,
            stations=stations,
        )
        if valid:
            return RawDownloadResult(cnt_path, ch_path, tuple(), "skipped_existing", "")
        print(
            f"[WARN] event {event.event_id}: ignoring incomplete staged merged files: {detail}",
            file=sys.stderr,
        )
    if not args.overwrite_raw:
        existing_segments = _list_segment_cnts(segment_dir)
        existing_segment_ch = _find_segment_channel_table(segment_dir)
        valid, detail = _validate_raw_request(
            existing_segments,
            existing_segment_ch,
            request_start_timestamp=request_start_ts,
            request_end_timestamp=request_end_ts,
            stations=stations,
        )
        if valid:
            return RawDownloadResult(
                None,
                existing_segment_ch,
                existing_segments,
                "skipped_existing_segments",
                "",
            )
        if existing_segments or existing_segment_ch is not None:
            print(
                f"[WARN] event {event.event_id}: ignoring incomplete staged segments: {detail}",
                file=sys.stderr,
            )
    if args.dry_run:
        return RawDownloadResult(None, None, tuple(), "dry_run", "")

    configured_batch_size = int(getattr(args, "station_batch_size", 40))
    batch_size = len(stations) if configured_batch_size <= 0 else configured_batch_size
    station_batches = [stations[start : start + batch_size] for start in range(0, len(stations), batch_size)]
    batch_results: list[RawBatchResult] = []
    for batch_index, station_batch in enumerate(station_batches):
        result = _download_station_batch(
            args,
            client,
            event_raw_dir=event_raw_dir,
            event_id=event.event_id,
            batch_index=batch_index,
            stations=station_batch,
            request_start_dt=request_start_dt,
            span_minutes=span_min,
        )
        if not result.success:
            return RawDownloadResult(
                None,
                result.channel_path,
                tuple(),
                "download_failed",
                _compact_error(
                    f"station batch {batch_index + 1}/{len(station_batches)} failed "
                    f"for {len(station_batch)} stations: {result.detail}",
                    maximum=1800,
                ),
                batch_count=len(station_batches),
                download_strategy="batched_with_minute_fallback",
            )
        batch_results.append(result)

    staged_segments, combined_channel = _stage_complete_event_download(
        event_raw_dir,
        event.event_id,
        batch_results,
        stations,
    )
    used_fallback = any(result.used_minute_fallback for result in batch_results)
    if len(station_batches) > 1 and used_fallback:
        raw_status = "downloaded_batched_minute_fallback"
        strategy = "station_batches+minute_fallback"
    elif len(station_batches) > 1:
        raw_status = "downloaded_batched"
        strategy = "station_batches"
    elif used_fallback:
        raw_status = "downloaded_minute_fallback"
        strategy = "minute_fallback"
    else:
        raw_status = "downloaded_unmerged"
        strategy = "single_request_native_segments"
    notes = [result.detail for result in batch_results if result.detail != "primary request valid"]
    return RawDownloadResult(
        None,
        combined_channel,
        staged_segments,
        raw_status,
        _compact_error(" | ".join(notes), maximum=1800),
        batch_count=len(station_batches),
        download_strategy=strategy,
    )


def canonical_raw_files(args, event_id: str, raw: RawDownloadResult) -> tuple[tuple[Path, ...], Path | None]:
    """Choose exactly one native representation for an archive commit.

    HinetPy may leave both its original one-minute segments and a merged CNT.
    The annual archive keeps the native segments when present and otherwise the
    merged file, never both.
    """
    # Explicit paths are authoritative for the new batched workflow. This also
    # prevents stale files from an older failed attempt taking precedence over
    # the fully validated aggregate staged under complete/.
    if raw.segment_paths:
        return tuple(raw.segment_paths), raw.ch_path
    if raw.cnt_path is not None:
        return (raw.cnt_path,), raw.ch_path
    event_raw_dir = getattr(args, "raw_work_root", args.output_root / "raw") / str(event_id)
    segment_dir = event_raw_dir / "segments"
    native_segments = _list_segment_cnts(segment_dir) if segment_dir.exists() else tuple()
    if native_segments:
        native_ch = _find_segment_channel_table(segment_dir)
        return native_segments, native_ch or raw.ch_path
    return tuple(), raw.ch_path


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


def call_get_continuous_waveform(
    client,
    network: str,
    starttime: str,
    span: int,
    data: str,
    ctable: str,
    outdir: Path,
    *,
    max_span: int | None = None,
    threads: int = 1,
    cleanup: bool = False,
) -> None:
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
        "max_span": max_span,
        "threads": threads,
        "cleanup": cleanup,
    }
    sig = inspect.signature(method)
    filtered = {key: value for key, value in kwargs.items() if key in sig.parameters}
    if "code" in sig.parameters:
        method(**filtered)
    else:
        for positional_name in ("code", "starttime", "span"):
            filtered.pop(positional_name, None)
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


def read_hinet_vm_cnt_segments(cnt_paths: Iterable[Path], channel_ids: set[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return read_hinet_vm_cnt_paths(cnt_paths, channel_ids)


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
            "origin_time_jst_raw": event.origin_time_jst_raw,
            "origin_timestamp": event.origin_timestamp,
            "origin_time_correction_s": "" if event.origin_time_correction_s is None else event.origin_time_correction_s,
            "origin_time_correction_status": event.origin_time_correction_status,
            "origin_time_correction_source": event.origin_time_correction_source,
            "origin_time_jma_event_id": event.origin_time_jma_event_id,
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


def load_download_context(args):
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
    return events, station_df, matches, jma_table, ak135


def archive_provenance(args) -> dict[str, object]:
    source_stat = args.hdf5.stat()
    match_csv = getattr(args, "match_csv", None)
    event_ids_file = getattr(args, "event_ids", None)
    origin_corrections_file = getattr(args, "origin_corrections", None)
    travel_time_path = getattr(args, "jma_travel_time_zip", None) or DEFAULT_JMA2001A_ZIP

    def small_file_identity(path: Path | None) -> dict[str, object]:
        if path is None:
            return {"path": "", "sha256": ""}
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            return {"path": str(resolved), "sha256": "missing"}
        digest = hashlib.sha256()
        with resolved.open("rb") as fp:
            for block in iter(lambda: fp.read(1024 * 1024), b""):
                digest.update(block)
        return {"path": str(resolved), "sha256": digest.hexdigest()}

    identity = {
        "source_hdf5": str(args.hdf5),
        "source_hdf5_size_bytes": int(source_stat.st_size),
        "source_hdf5_mtime_ns": int(source_stat.st_mtime_ns),
        "year": int(args.year),
        "hinet_network": args.hinet_network,
        "match_distance_km": float(args.match_distance_km),
        "pre_seconds": float(args.pre_seconds),
        "post_seconds": float(args.post_seconds),
        "mode": args.mode,
        "station_matches": small_file_identity(match_csv),
        "event_ids": small_file_identity(event_ids_file),
        "origin_corrections": small_file_identity(origin_corrections_file),
        "travel_time_table": small_file_identity(travel_time_path),
    }
    return {
        "archive_identity": identity,
        "command": shlex.join(sys.argv),
        "source_hdf5": str(args.hdf5),
        "source_hdf5_size_bytes": int(source_stat.st_size),
        "source_hdf5_mtime_ns": int(source_stat.st_mtime_ns),
        "year": int(args.year),
        "hinet_network": args.hinet_network,
        "match_distance_km": float(args.match_distance_km),
        "pre_seconds": float(args.pre_seconds),
        "post_seconds": float(args.post_seconds),
        "mode": args.mode,
        "event_ids_file": "" if event_ids_file is None else str(event_ids_file),
        "origin_corrections_file": "" if origin_corrections_file is None else str(origin_corrections_file),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "archive_policy": "native_cnt_segments_preferred;sha256_deduplicated;no_mseed;no_sacpz",
        # These transport controls may change between retries without changing
        # the requested scientific dataset or invalidating a partial archive.
        "download_transport": {
            "hinet_timeout_seconds": float(getattr(args, "hinet_timeout_seconds", 300.0)),
            "hinet_retries": int(getattr(args, "hinet_retries", 3)),
            "station_batch_size": int(getattr(args, "station_batch_size", 40)),
            "hinet_download_threads": int(getattr(args, "hinet_download_threads", 1)),
            "minute_fallback": bool(getattr(args, "minute_fallback", True)),
            "fallback_span_minutes": int(getattr(args, "fallback_span_minutes", 1)),
            "subrequest_sleep_seconds": float(getattr(args, "subrequest_sleep_seconds", 0.0)),
        },
    }


def cleanup_staged_event(args, event_id: str) -> None:
    if args.keep_staging:
        return
    root = Path(args.raw_work_root).resolve()
    event_dir = root / str(event_id)
    if event_dir.parent != root:
        raise RuntimeError(f"Refusing to clean unexpected staging path: {event_dir}")
    if event_dir.exists():
        shutil.rmtree(event_dir)


def export_archive_catalog(archive_path: Path, output_root: Path, year: int) -> None:
    catalog_dir = output_root / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    with AnnualHinetArchiveReader(archive_path) as reader:
        reader.events_dataframe().to_csv(catalog_dir / f"hinet_events_{year}.csv", index=False)
        reader.attempts_dataframe().to_csv(catalog_dir / f"hinet_attempts_{year}.csv", index=False)
        summary = {
            "archive": str(archive_path),
            "year": reader.year,
            "complete": reader.complete,
            "committed_event_count": len(reader.event_ids()),
        }
    (catalog_dir / f"hinet_archive_{year}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def resolve_archive_work_path(final_path: Path) -> tuple[Path, bool]:
    """Return the append target and whether the final archive is already complete."""
    final_path = Path(final_path)
    partial_path = partial_archive_path(final_path)
    if final_path.exists():
        with AnnualHinetArchiveReader(final_path) as reader:
            if reader.complete:
                return final_path, True
        if partial_path.exists():
            raise RuntimeError(f"Both incomplete final and partial archives exist: {final_path}, {partial_path}")
        return final_path, False
    if partial_path.exists():
        return partial_path, False
    return partial_path, False


def process_events_archive(args) -> dict[str, object]:
    if args.write_mseed:
        raise ValueError("--write-mseed is incompatible with --storage-mode annual-hdf5")
    if args.response_mode != "none":
        raise ValueError("Response side products are disabled in annual archive mode; raw .ch bytes are archived instead")
    if args.overwrite_raw:
        raise ValueError("--overwrite-raw is not supported for append-only archives; use a new archive path")

    events, station_df, matches, jma_table, ak135 = load_download_context(args)
    expected_ids = set(events)
    if args.dry_run:
        station_total = 0
        for event_number, event_id in enumerate(sorted(events, reverse=True), start=1):
            arrivals = build_event_arrivals(
                events[event_id],
                station_df,
                matches,
                jma_table,
                ak135,
                pre_seconds=args.pre_seconds,
                post_seconds=args.post_seconds,
            )
            station_total += len(arrivals)
            print(
                f"[DRY-RUN] [{event_number}/{len(events)}] event={event_id} "
                f"matched_stations={len(arrivals)}"
            )
        print(f"[DRY-RUN] events={len(events)} matched_event_stations={station_total}; no archive was created")
        return {"complete": True, "failed": 0, "dry_run": True, "archive_path": ""}

    provenance = archive_provenance(args)
    archive_work_path, already_complete = resolve_archive_work_path(args.archive_path)
    if already_complete:
        with AnnualHinetArchiveReader(archive_work_path) as reader:
            missing = expected_ids - set(reader.event_ids())
            identity_matches = reader.archive_identity == provenance["archive_identity"]
        if missing:
            raise RuntimeError(
                f"Completed archive {archive_work_path} is missing {len(missing)} events from the current source"
            )
        if not identity_matches:
            raise RuntimeError(
                f"Completed archive identity differs from the current source/configuration: {archive_work_path}"
            )
        print(f"[INFO] annual archive is already complete: {archive_work_path}")
        export_archive_catalog(archive_work_path, args.output_root, args.year)
        return {"complete": True, "failed": 0, "archive_path": str(archive_work_path)}

    args.raw_work_root = args.output_root / ".staging" / str(args.year) / "raw"
    args.raw_work_root.mkdir(parents=True, exist_ok=True)
    client = None
    run_failures: list[str] = []
    newly_committed = 0
    skipped_committed = 0

    writer = AnnualHinetArchiveWriter(
        archive_work_path,
        year=args.year,
        source_hdf5=args.hdf5,
        provenance=provenance,
        chunk_bytes=args.archive_chunk_bytes,
    )
    try:
        for event_number, event_id in enumerate(sorted(events, reverse=True), start=1):
            event = events[event_id]
            if writer.has_event(event_id):
                skipped_committed += 1
                cleanup_staged_event(args, event_id)
                print(f"[INFO] [{event_number}/{len(events)}] event {event_id}: already committed")
                continue

            print(
                f"[INFO] [{event_number}/{len(events)}] event {event_id} "
                f"origin={event.origin_time_jst}"
            )
            station_count = 0
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
                station_count = len(arrivals)
                if arrivals.empty:
                    writer.commit_event_bytes(
                        event,
                        arrivals,
                        [],
                        ("", b""),
                        raw_status="no_matched_stations",
                    )
                    writer.verify_event(event_id)
                    cleanup_staged_event(args, event_id)
                    newly_committed += 1
                    continue
                if client is None:
                    client = make_hinet_client(args)
                raw = download_raw_event(args, client, event, arrivals)
                if not (
                    raw.raw_status.startswith("downloaded")
                    or raw.raw_status.startswith("skipped_existing")
                ):
                    raise RuntimeError(
                        f"raw download did not complete: status={raw.raw_status}; error={raw.raw_error}"
                    )
                cnt_paths, channel_table_path = canonical_raw_files(args, event_id, raw)
                if not cnt_paths or channel_table_path is None:
                    raise RuntimeError(
                        f"raw download is incomplete: status={raw.raw_status}; "
                        f"cnt_segments={len(cnt_paths)}; channel_table={channel_table_path}; "
                        f"error={raw.raw_error}"
                    )
                empty_files = [str(path) for path in cnt_paths if path.stat().st_size <= 0]
                if channel_table_path.stat().st_size <= 0:
                    empty_files.append(str(channel_table_path))
                if empty_files:
                    raise RuntimeError("downloaded empty raw files: " + ", ".join(empty_files))

                missing_stations = _channel_table_missing_stations(
                    channel_table_path,
                    arrivals["hinet_station"].astype(str),
                )
                if missing_stations:
                    raise RuntimeError(
                        "raw channel table lacks complete 3-component stations: "
                        + ",".join(missing_stations[:8])
                    )

                coverage = scan_hinet_vm_cnt_paths(cnt_paths)
                request_start = float(arrivals["cut_start_timestamp"].min())
                request_end = float(arrivals["cut_end_timestamp"].max())
                coverage_tolerance_s = 1.1
                if (
                    coverage.record_count <= 0
                    or not math.isfinite(coverage.start_timestamp)
                    or not math.isfinite(coverage.end_timestamp)
                    or coverage.start_timestamp > request_start + coverage_tolerance_s
                    or coverage.end_timestamp < request_end - coverage_tolerance_s
                ):
                    raise RuntimeError(
                        "raw CNT time coverage is incomplete: "
                        f"records={coverage.record_count}; "
                        f"coverage=[{coverage.start_timestamp}, {coverage.end_timestamp}); "
                        f"requested=[{request_start}, {request_end}]"
                    )

                arrivals = arrivals.copy()
                arrivals["raw_status"] = raw.raw_status
                arrivals["raw_error"] = raw.raw_error
                arrivals["raw_batch_count"] = raw.batch_count
                arrivals["raw_download_strategy"] = raw.download_strategy
                arrivals["hinet_timeout_seconds"] = float(
                    getattr(args, "hinet_timeout_seconds", 300.0)
                )
                arrivals["hinet_retries"] = int(getattr(args, "hinet_retries", 3))
                arrivals["station_batch_size"] = int(getattr(args, "station_batch_size", 40))
                arrivals["minute_fallback_enabled"] = int(
                    bool(getattr(args, "minute_fallback", True))
                )
                arrivals["fallback_span_minutes"] = int(
                    getattr(args, "fallback_span_minutes", 1)
                )
                arrivals["raw_storage"] = "annual_hdf5_native_cnt_bytes"
                arrivals["archive_path"] = str(args.archive_path)
                arrivals["archive_event_id"] = event_id
                arrivals["raw_segment_count"] = len(cnt_paths)
                arrivals["raw_segment_filenames"] = ";".join(path.name for path in cnt_paths)
                arrivals["raw_record_count"] = coverage.record_count
                arrivals["raw_coverage_start_timestamp"] = coverage.start_timestamp
                arrivals["raw_coverage_end_timestamp_exclusive"] = coverage.end_timestamp
                arrivals["raw_time_coverage_complete"] = 1
                arrivals["channel_table_filename"] = channel_table_path.name
                arrivals["response_status"] = "raw_channel_table_archived"
                arrivals["mseed_status"] = "disabled_archive"

                writer.commit_event_files(
                    event,
                    arrivals,
                    cnt_paths,
                    channel_table_path,
                    raw_status=raw.raw_status,
                    raw_error=raw.raw_error,
                )
                writer.verify_event(event_id)
                cleanup_staged_event(args, event_id)
                newly_committed += 1
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                # If an archive append failed, discard its uncommitted tail
                # before recording the failure or processing another event.
                writer.recover()
                if writer.has_event(event_id):
                    # A committed row is authoritative. Never downgrade or
                    # silently finalize it if readback verification failed.
                    try:
                        writer.verify_event(event_id)
                    except Exception as verify_exc:
                        raise RuntimeError(
                            f"Committed archive data failed verification for event {event_id}"
                        ) from verify_exc
                    newly_committed += 1
                    print(
                        f"[WARN] event {event_id} was committed and verified, but post-commit cleanup failed: {exc!r}",
                        file=sys.stderr,
                    )
                    continue
                writer.record_attempt(
                    event_id,
                    "failed",
                    station_count=station_count,
                    error=repr(exc),
                )
                run_failures.append(event_id)
                print(f"[WARN] event {event_id} failed: {exc!r}", file=sys.stderr)

        missing_after_run = expected_ids - writer.committed_event_ids()
        can_finalize = (
            not args.dry_run
            and args.mode == "all"
            and args.event_ids is None
            and not missing_after_run
        )
        if can_finalize:
            writer.mark_complete(expected_ids)
    finally:
        writer.close()

    completed = bool(can_finalize)
    output_archive = archive_work_path
    if completed and archive_work_path != args.archive_path:
        os.replace(archive_work_path, args.archive_path)
        output_archive = args.archive_path
    export_archive_catalog(output_archive, args.output_root, args.year)

    missing_count = len(expected_ids - writer.committed_event_ids())
    print(
        f"[INFO] archive summary year={args.year}: total={len(events)} "
        f"new={newly_committed} resumed={skipped_committed} missing={missing_count} "
        f"path={output_archive}"
    )
    return {
        "complete": completed,
        "failed": missing_count,
        "run_failures": run_failures,
        "archive_path": str(output_archive),
    }


def process_events_files(args) -> dict[str, object]:
    events, station_df, matches, jma_table, ak135 = load_download_context(args)

    client = None if args.dry_run else make_hinet_client(args)
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
            if not args.dry_run and not (
                raw.raw_status.startswith("downloaded")
                or raw.raw_status.startswith("skipped_existing")
            ):
                raise RuntimeError(
                    f"raw download did not complete: status={raw.raw_status}; error={raw.raw_error}"
                )
            arrivals["raw_status"] = raw.raw_status
            arrivals["raw_error"] = raw.raw_error
            arrivals["raw_batch_count"] = raw.batch_count
            arrivals["raw_download_strategy"] = raw.download_strategy
            arrivals["raw_count_path"] = "" if raw.cnt_path is None else str(raw.cnt_path)
            arrivals["raw_segment_paths"] = ";".join(str(path) for path in raw.segment_paths)
            arrivals["raw_segment_count"] = len(raw.segment_paths)
            arrivals["channel_table_path"] = "" if raw.ch_path is None else str(raw.ch_path)

            if raw.ch_path is not None and args.response_mode == "pz":
                arrivals["response_status"] = cache_response_files(args, raw.ch_path, event_id)
            elif raw.ch_path is not None:
                arrivals["response_status"] = "raw_channel_table_only"
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
    failed = sum(row.get("status") == "failed" for row in summary_rows)
    return {"complete": failed == 0, "failed": failed, "archive_path": ""}


def process_events(args) -> dict[str, object]:
    if args.storage_mode == "annual-hdf5":
        return process_events_archive(args)
    return process_events_files(args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and archive matched Hi-net raw-count windows.")
    parser.add_argument("--hdf5", type=Path, required=True, help="Converted K-NET/KiK-net HDF5 file.")
    parser.add_argument("--output-root", type=Path, default=Path("hinet_velocity_downloads"))
    parser.add_argument(
        "--storage-mode",
        choices=("annual-hdf5", "files"),
        default="annual-hdf5",
        help="Store byte-exact CNT/CH data in one annual HDF5 archive (default) or use the legacy file tree.",
    )
    parser.add_argument("--year", type=int, default=None, help="Archive year; inferred from japan_YYYY.hdf5 when omitted.")
    parser.add_argument("--archive-path", type=Path, default=None, help="Final annual archive path.")
    parser.add_argument("--archive-chunk-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--keep-staging", action="store_true", help="Keep temporary native CNT/CH files after verified archive commits.")
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
    parser.add_argument(
        "--hinet-timeout-seconds",
        type=float,
        default=300.0,
        help="Read timeout for each Hi-net ZIP download (HinetPy defaults to 60 seconds).",
    )
    parser.add_argument(
        "--hinet-retries",
        type=int,
        default=3,
        help="Transport retries per Hi-net download job.",
    )
    parser.add_argument(
        "--station-batch-size",
        type=int,
        default=40,
        help="Maximum selected Hi-net stations per request; 0 disables station splitting.",
    )
    parser.add_argument(
        "--hinet-download-threads",
        type=int,
        default=1,
        help="HinetPy download threads per request; one is safest for provider throttling.",
    )
    parser.add_argument(
        "--minute-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry a failed full window as short consecutive requests.",
    )
    parser.add_argument(
        "--fallback-span-minutes",
        type=int,
        default=1,
        help="Minutes per request during fallback (1-60).",
    )
    parser.add_argument(
        "--subrequest-sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between consecutive fallback subrequests.",
    )
    parser.add_argument("--inventory-csv", type=Path, default=None)
    parser.add_argument("--match-csv", type=Path, default=None)
    parser.add_argument("--match-distance-km", type=float, default=1.0)
    parser.add_argument("--pre-seconds", type=float, default=120.0)
    parser.add_argument("--post-seconds", type=float, default=120.0)
    parser.add_argument("--jma-travel-time-zip", type=Path, default=None)
    parser.add_argument("--win-century", default="20", help="Century prefix for ObsPy WIN timestamp parser.")
    parser.add_argument("--write-mseed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pad-mseed", action="store_true", help="Pad cut MiniSEED traces with zero counts if coverage is short.")
    parser.add_argument(
        "--response-mode",
        choices=("none", "pz"),
        default="none",
        help="Legacy file mode only: optionally extract per-channel SAC PZ files.",
    )
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
    if args.year is None:
        match = re.search(r"(?:japan_|/)(?P<year>20\d{2})(?:\.hdf5|/)", str(args.hdf5))
        if match is None:
            raise ValueError("Pass --year because it could not be inferred from --hdf5")
        args.year = int(match.group("year"))
    if not 1900 <= int(args.year) <= 2200:
        raise ValueError(f"Invalid --year: {args.year}")
    if args.archive_chunk_bytes <= 0:
        raise ValueError("--archive-chunk-bytes must be positive")
    if args.hinet_timeout_seconds <= 0:
        raise ValueError("--hinet-timeout-seconds must be positive")
    if args.hinet_retries <= 0:
        raise ValueError("--hinet-retries must be positive")
    if args.station_batch_size < 0:
        raise ValueError("--station-batch-size must be zero or positive")
    if args.hinet_download_threads <= 0:
        raise ValueError("--hinet-download-threads must be positive")
    if not 1 <= args.fallback_span_minutes <= 60:
        raise ValueError("--fallback-span-minutes must be in [1, 60]")
    if args.subrequest_sleep_seconds < 0:
        raise ValueError("--subrequest-sleep-seconds must be non-negative")
    if args.archive_path is None:
        args.archive_path = args.output_root / "archive" / f"hinet_raw_{args.year}.h5"
    else:
        args.archive_path = args.archive_path.expanduser().resolve()
    if args.inventory_csv is None:
        args.inventory_csv = args.output_root / "catalog" / "hinet_stations.csv"
    else:
        args.inventory_csv = args.inventory_csv.expanduser().resolve()
    if args.match_csv is None:
        args.match_csv = args.output_root / "catalog" / f"hinet_kiknet_station_matches_{args.year}.csv"
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
    result = process_events(args)
    return 0 if result.get("complete", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
