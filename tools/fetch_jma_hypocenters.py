#!/usr/bin/env python3
"""Fetch JMA hypocenter catalogs and match precise origin times to events.

The K-NET/KiK-net waveform headers used by japan_dataset_builder.py often carry
minute-level origin times.  JMA daily hypocenter lists and Seismological
Bulletin files provide origin seconds, which is enough to repair the
travel-time based P picks.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


JST = timezone(timedelta(hours=9))
JMA_DAILY_URL_TEMPLATE = "https://www.data.jma.go.jp/eqev/data/daily_map/{date}.html"
JMA_BULLETIN_URL_TEMPLATE = "https://www.data.jma.go.jp/eqev/data/bulletin/data/hypo/h{year}.zip"

JMA_LINE_RE = re.compile(
    r"^\s*"
    r"(?P<year>\d{4})\s+"
    r"(?P<month>\d{1,2})\s+"
    r"(?P<day>\d{1,2})\s+"
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2})\s+"
    r"(?P<second>\d{1,2}(?:\.\d+)?)\s+"
    r"(?P<lat_deg>\d{1,2})[°ﾟ]\s*(?P<lat_min>\d{1,2}(?:\.\d+)?)[\'′](?P<lat_hemi>[NS])\s+"
    r"(?P<lon_deg>\d{1,3})[°ﾟ]\s*(?P<lon_min>\d{1,2}(?:\.\d+)?)[\'′](?P<lon_hemi>[EW])\s+"
    r"(?P<depth>-?\d+(?:\.\d+)?|-)\s+"
    r"(?P<magnitude>-?\d+(?:\.\d+)?|-)\s*"
    r"(?P<region>.*?)\s*$"
)


@dataclass
class TrainingEvent:
    event_id: str
    origin_time_jst_raw: str
    origin_timestamp_raw: float
    latitude: float
    longitude: float
    depth_km: float
    magnitude: float | None


@dataclass
class JMAHypocenter:
    jma_event_id: str
    origin_time_jst: str
    origin_timestamp: float
    latitude: float
    longitude: float
    depth_km: float | None
    magnitude: float | None
    region: str
    source_url: str
    source_date: str
    source_line: str


def decode_array(values) -> np.ndarray:
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
            return pd.DataFrame({col: decode_array(node[col][()]) for col in node.keys()})
    return pd.read_hdf(hdf5_path, key)


def find_column(columns: Iterable[str], candidates: Iterable[str], required: bool = True) -> str | None:
    column_set = {str(c).lower(): str(c) for c in columns}
    for candidate in candidates:
        found = column_set.get(candidate.lower())
        if found is not None:
            return found
    if required:
        raise KeyError(f"None of {list(candidates)} found in columns {list(columns)}")
    return None


def parse_training_origin_time(value: object, event_id: str) -> tuple[str, float]:
    if value is not None and not pd.isna(value):
        text = str(value).strip()
        for fmt in (
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
        ):
            try:
                dt = datetime.strptime(text, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=JST)
                dt = dt.astimezone(JST)
                return dt.isoformat(), dt.timestamp()
            except ValueError:
                pass
        try:
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=JST)
            dt = dt.astimezone(JST)
            return dt.isoformat(), dt.timestamp()
        except ValueError:
            pass
    event_text = str(event_id)
    if len(event_text) == 14 and event_text.isdigit():
        dt = datetime.strptime(event_text, "%Y%m%d%H%M%S").replace(tzinfo=JST)
        return dt.isoformat(), dt.timestamp()
    raise ValueError(f"Cannot parse origin time for event {event_id!r}: {value!r}")


def load_training_events(args: argparse.Namespace) -> pd.DataFrame:
    if args.events_csv is not None:
        df = pd.read_csv(args.events_csv)
    elif args.hdf5 is not None:
        df = read_metadata_table(args.hdf5, "event_metadata")
    elif args.stations_csv is not None:
        df = pd.read_csv(args.stations_csv)
    else:
        raise SystemExit("Provide one of --events-csv, --hdf5, or --stations-csv.")

    event_key = find_column(df.columns, ["EVENT", "KiK_File", "#EventID"])
    lat_key = find_column(df.columns, ["Latitude", "LAT", "lat", "event_lat"])
    lon_key = find_column(df.columns, ["Longitude", "LON", "lon", "event_lon"])
    depth_key = find_column(df.columns, ["DEPTH", "Depth", "depth_km", "event_depth_km"])
    mag_key = find_column(df.columns, ["Magnitude", "M_J", "MA", "Mag", "event_magnitude"], required=False)
    origin_key = find_column(
        df.columns,
        ["Origin_Time(JST)", "Origin Time", "origin_time", "origin_time_jst", "Time"],
        required=False,
    )

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        event_id = str(row[event_key])
        if event_id in seen:
            continue
        seen.add(event_id)
        if args.event_id and event_id not in args.event_id:
            continue
        origin_value = row[origin_key] if origin_key is not None else None
        origin_iso, origin_ts = parse_training_origin_time(origin_value, event_id)
        rows.append(
            {
                "event_id": event_id,
                "origin_time_jst_raw": origin_iso,
                "origin_timestamp_raw": origin_ts,
                "latitude": float(row[lat_key]),
                "longitude": float(row[lon_key]),
                "depth_km": float(row[depth_key]),
                "magnitude": np.nan if mag_key is None or pd.isna(row[mag_key]) else float(row[mag_key]),
            }
        )
        if args.limit_events is not None and len(rows) >= args.limit_events:
            break
    if not rows:
        raise SystemExit("No training events were loaded.")
    return pd.DataFrame(rows)


def decimal_degrees(degrees: str, minutes: str, hemisphere: str) -> float:
    value = float(degrees) + float(minutes) / 60.0
    if hemisphere in {"S", "W"}:
        value = -value
    return value


def clean_daily_html(text: str) -> str:
    pre_match = re.search(r"<pre[^>]*>(.*?)</pre>", text, flags=re.IGNORECASE | re.DOTALL)
    if pre_match:
        text = pre_match.group(1)
    else:
        text = re.sub(r"</(?:tr|p|div|br|li|h\d)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(text)


def parse_jma_daily_text(text: str, source_url: str, source_date: str) -> list[JMAHypocenter]:
    out: list[JMAHypocenter] = []
    cleaned = clean_daily_html(text)
    for raw_line in cleaned.splitlines():
        line = " ".join(raw_line.strip().split())
        if not line or not re.match(r"^\d{4}\s+", line):
            continue
        match = JMA_LINE_RE.match(line)
        if not match:
            continue
        parts = match.groupdict()
        base = datetime(
            int(parts["year"]),
            int(parts["month"]),
            int(parts["day"]),
            int(parts["hour"]),
            int(parts["minute"]),
            tzinfo=JST,
        )
        origin_dt = base + timedelta(seconds=float(parts["second"]))
        lat = decimal_degrees(parts["lat_deg"], parts["lat_min"], parts["lat_hemi"])
        lon = decimal_degrees(parts["lon_deg"], parts["lon_min"], parts["lon_hemi"])
        depth = None if parts["depth"] == "-" else float(parts["depth"])
        mag = None if parts["magnitude"] == "-" else float(parts["magnitude"])
        origin_id = origin_dt.strftime("%Y%m%d%H%M%S") + f"{int(round(origin_dt.microsecond / 1000.0)):03d}"
        out.append(
            JMAHypocenter(
                jma_event_id=origin_id,
                origin_time_jst=origin_dt.isoformat(timespec="milliseconds"),
                origin_timestamp=origin_dt.timestamp(),
                latitude=lat,
                longitude=lon,
                depth_km=depth,
                magnitude=mag,
                region=parts["region"].strip(),
                source_url=source_url,
                source_date=source_date,
                source_line=line,
            )
        )
    return out


def parse_compact_decimal(text: str, scale: float) -> float | None:
    text = text.strip()
    if not text:
        return None
    if "." in text:
        return float(text)
    return float(int(text)) / scale


def parse_bulletin_depth(text: str) -> float | None:
    if not text.strip():
        return None
    if "." in text:
        return float(text)
    # The bulletin format stores either F5.2 depth without a decimal point or
    # depth-slice I3 followed by two blanks.
    if len(text) >= 5 and not text[-2:].strip():
        return float(int(text[:3]))
    return float(int(text.strip())) / 100.0


def parse_bulletin_magnitude(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    if "." in text:
        return float(text)
    if text.startswith("-") and len(text) == 2 and text[1].isdigit():
        return -float(text[1]) / 10.0
    if len(text) == 2 and text[0].isalpha() and text[1].isdigit():
        return -(ord(text[0].upper()) - ord("A") + 1) - float(text[1]) / 10.0
    return float(int(text)) / 10.0


def decode_bulletin_member(payload: bytes) -> str:
    for encoding in ("utf-8", "cp932", "shift_jis", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def parse_jma_bulletin_text(text: str, source_url: str, source_year: str) -> list[JMAHypocenter]:
    out: list[JMAHypocenter] = []
    seen_ids: dict[str, int] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.rstrip("\r\n")
        if len(line) < 92 or not line.startswith("J"):
            continue
        try:
            year = int(line[1:5])
            month = int(line[5:7])
            day = int(line[7:9])
            hour = int(line[9:11])
            minute = int(line[11:13])
            second = parse_compact_decimal(line[13:17], 100.0)
            lat_min = parse_compact_decimal(line[24:28], 100.0)
            lon_min = parse_compact_decimal(line[36:40], 100.0)
            if second is None or lat_min is None or lon_min is None:
                continue
            base = datetime(year, month, day, hour, minute, tzinfo=JST)
            origin_dt = base + timedelta(seconds=second)
            lat = float(int(line[21:24])) + lat_min / 60.0
            lon = float(int(line[32:36])) + lon_min / 60.0
            depth = parse_bulletin_depth(line[44:49])
            mag = parse_bulletin_magnitude(line[52:54])
            if mag is None:
                mag = parse_bulletin_magnitude(line[55:57])
        except (ValueError, OverflowError):
            continue

        base_id = origin_dt.strftime("%Y%m%d%H%M%S") + f"{int(round(origin_dt.microsecond / 1000.0)):03d}"
        duplicate_index = seen_ids.get(base_id, 0)
        seen_ids[base_id] = duplicate_index + 1
        event_id = base_id if duplicate_index == 0 else f"{base_id}_{duplicate_index:03d}"
        out.append(
            JMAHypocenter(
                jma_event_id=event_id,
                origin_time_jst=origin_dt.isoformat(timespec="milliseconds"),
                origin_timestamp=origin_dt.timestamp(),
                latitude=lat,
                longitude=lon,
                depth_km=depth,
                magnitude=mag,
                region=line[68:92].strip(),
                source_url=source_url,
                source_date=source_year,
                source_line=line,
            )
        )
    return out


def parse_jma_bulletin_zip(payload: bytes, source_url: str, source_year: str) -> list[JMAHypocenter]:
    rows: list[JMAHypocenter] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            rows.extend(parse_jma_bulletin_text(decode_bulletin_member(archive.read(name)), source_url, source_year))
    return rows


def fetch_jma_daily_page(date_text: str, cache_dir: Path, args: argparse.Namespace) -> str | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{date_text}.html"
    if cache_path.exists() and not args.refresh:
        return cache_path.read_text(encoding="utf-8", errors="replace")
    if args.cache_only:
        if args.allow_missing_days:
            return None
        raise FileNotFoundError(f"Missing cached JMA daily page: {cache_path}")
    url = JMA_DAILY_URL_TEMPLATE.format(date=date_text)
    request = urllib.request.Request(url, headers={"User-Agent": args.user_agent})
    try:
        with urllib.request.urlopen(request, timeout=args.timeout_seconds) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            payload = response.read().decode(charset, errors="replace")
    except urllib.error.HTTPError as exc:
        if args.allow_missing_days and exc.code == 404:
            print(f"[WARN] missing JMA daily page for {date_text}: {url}", file=sys.stderr)
            return None
        raise RuntimeError(f"Failed to fetch JMA daily page for {date_text}: {url}") from exc
    cache_path.write_text(payload, encoding="utf-8")
    if args.request_sleep_seconds > 0:
        time.sleep(args.request_sleep_seconds)
    return payload


def fetch_jma_bulletin_zip(year: int, cache_dir: Path, args: argparse.Namespace) -> bytes | None:
    bulletin_cache_dir = cache_dir / "bulletin"
    bulletin_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = bulletin_cache_dir / f"h{year}.zip"
    if cache_path.exists() and not args.refresh:
        return cache_path.read_bytes()
    if args.cache_only:
        if args.allow_missing_days:
            return None
        raise FileNotFoundError(f"Missing cached JMA bulletin zip: {cache_path}")
    url = JMA_BULLETIN_URL_TEMPLATE.format(year=year)
    request = urllib.request.Request(url, headers={"User-Agent": args.user_agent})
    try:
        with urllib.request.urlopen(request, timeout=args.timeout_seconds) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        if args.allow_missing_days and exc.code == 404:
            print(f"[WARN] missing JMA bulletin zip for {year}: {url}", file=sys.stderr)
            return None
        raise RuntimeError(f"Failed to fetch JMA bulletin zip for {year}: {url}") from exc
    cache_path.write_bytes(payload)
    if args.request_sleep_seconds > 0:
        time.sleep(args.request_sleep_seconds)
    return payload


def dates_for_events(events: pd.DataFrame, padding_days: int) -> list[str]:
    dates: set[str] = set()
    for ts in events["origin_timestamp_raw"].astype(float):
        day = datetime.fromtimestamp(ts, tz=JST).date()
        for delta in range(-padding_days, padding_days + 1):
            dates.add((day + timedelta(days=delta)).strftime("%Y%m%d"))
    return sorted(dates)


def load_jma_catalog(events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    date_texts = dates_for_events(events, args.date_padding_days)
    if args.catalog_source == "daily":
        daily_dates = date_texts
        bulletin_years: list[int] = []
    elif args.catalog_source == "bulletin":
        daily_dates = []
        bulletin_years = sorted({int(date_text[:4]) for date_text in date_texts})
    else:
        daily_dates = [date_text for date_text in date_texts if int(date_text[:4]) > args.bulletin_max_year]
        bulletin_years = sorted({int(date_text[:4]) for date_text in date_texts if int(date_text[:4]) <= args.bulletin_max_year})

    for year in bulletin_years:
        payload = fetch_jma_bulletin_zip(year, args.cache_dir, args)
        if payload is None:
            continue
        source_url = JMA_BULLETIN_URL_TEMPLATE.format(year=year)
        rows.extend(asdict(item) for item in parse_jma_bulletin_zip(payload, source_url, str(year)))
    for date_text in daily_dates:
        page = fetch_jma_daily_page(date_text, args.cache_dir, args)
        if page is None:
            continue
        source_url = JMA_DAILY_URL_TEMPLATE.format(date=date_text)
        rows.extend(asdict(item) for item in parse_jma_daily_text(page, source_url, date_text))
    if not rows:
        raise SystemExit("No JMA hypocenters were parsed. Check network/cache and JMA page format.")
    catalog = pd.DataFrame(rows).drop_duplicates("jma_event_id").reset_index(drop=True)
    needed_dates = set(date_texts)
    catalog = catalog[
        [
            datetime.fromtimestamp(float(ts), tz=JST).strftime("%Y%m%d") in needed_dates
            for ts in catalog["origin_timestamp"].astype(float)
        ]
    ].reset_index(drop=True)
    if catalog.empty:
        raise SystemExit("No JMA hypocenters remain after filtering to training-event dates.")
    return catalog.sort_values("origin_timestamp").reset_index(drop=True)


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


def finite_abs_diff(a: float | None, b: float | None) -> float:
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    return abs(float(a) - float(b))


def build_candidate_scores(event: TrainingEvent, catalog: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if catalog.empty:
        return pd.DataFrame()
    candidates = catalog.copy()
    candidates["time_diff_s"] = candidates["origin_timestamp"].astype(float) - event.origin_timestamp_raw
    candidates["abs_time_diff_s"] = candidates["time_diff_s"].abs()
    candidates = candidates[candidates["abs_time_diff_s"] <= args.max_time_diff_seconds].copy()
    if candidates.empty:
        return candidates

    candidates["horizontal_distance_km"] = [
        haversine_distance_km(event.latitude, event.longitude, float(row.latitude), float(row.longitude))
        for row in candidates.itertuples(index=False)
    ]
    candidates["depth_diff_km"] = [
        finite_abs_diff(event.depth_km, None if pd.isna(value) else float(value))
        for value in candidates["depth_km"]
    ]
    candidates["magnitude_diff"] = [
        finite_abs_diff(event.magnitude, None if pd.isna(value) else float(value))
        for value in candidates["magnitude"]
    ]

    mask = candidates["horizontal_distance_km"] <= args.max_distance_km
    mask &= candidates["depth_diff_km"].isna() | (candidates["depth_diff_km"] <= args.max_depth_diff_km)
    mask &= candidates["magnitude_diff"].isna() | (candidates["magnitude_diff"] <= args.max_magnitude_diff)
    candidates = candidates[mask].copy()
    if candidates.empty:
        return candidates

    depth_term = candidates["depth_diff_km"].fillna(0.0) / args.depth_weight_km
    mag_term = candidates["magnitude_diff"].fillna(0.0) / args.magnitude_weight
    candidates["match_score"] = (
        candidates["abs_time_diff_s"] / args.time_weight_seconds
        + candidates["horizontal_distance_km"] / args.distance_weight_km
        + depth_term
        + mag_term
    )
    return candidates.sort_values(["match_score", "abs_time_diff_s", "horizontal_distance_km"]).reset_index(drop=True)


def match_one_event(event_row: pd.Series, catalog: pd.DataFrame, args: argparse.Namespace) -> dict[str, object]:
    mag = None if pd.isna(event_row["magnitude"]) else float(event_row["magnitude"])
    event = TrainingEvent(
        event_id=str(event_row["event_id"]),
        origin_time_jst_raw=str(event_row["origin_time_jst_raw"]),
        origin_timestamp_raw=float(event_row["origin_timestamp_raw"]),
        latitude=float(event_row["latitude"]),
        longitude=float(event_row["longitude"]),
        depth_km=float(event_row["depth_km"]),
        magnitude=mag,
    )
    candidates = build_candidate_scores(event, catalog, args)
    base = {
        "event_id": event.event_id,
        "origin_time_jst_raw": event.origin_time_jst_raw,
        "origin_timestamp_raw": event.origin_timestamp_raw,
        "training_latitude": event.latitude,
        "training_longitude": event.longitude,
        "training_depth_km": event.depth_km,
        "training_magnitude": np.nan if event.magnitude is None else event.magnitude,
        "candidate_count": int(len(candidates)),
    }
    if candidates.empty:
        base.update({"match_status": "no_candidate", "accepted": 0})
        return base

    best = candidates.iloc[0]
    second = candidates.iloc[1] if len(candidates) > 1 else None
    second_score = np.nan if second is None else float(second["match_score"])
    score_margin = np.inf if second is None else second_score - float(best["match_score"])
    ambiguous = bool(second is not None and score_margin < args.min_score_margin)
    accepted = bool(not ambiguous or args.accept_ambiguous)
    corrected_dt = datetime.fromtimestamp(float(best["origin_timestamp"]), tz=JST)
    out = {
        **base,
        "match_status": "ambiguous" if ambiguous else "matched",
        "accepted": int(accepted),
        "origin_time_jst_corrected": corrected_dt.isoformat(timespec="milliseconds"),
        "origin_timestamp_corrected": float(best["origin_timestamp"]),
        "origin_time_correction_s": float(best["origin_timestamp"]) - event.origin_timestamp_raw,
        "jma_event_id": str(best["jma_event_id"]),
        "jma_origin_time_jst": str(best["origin_time_jst"]),
        "jma_origin_timestamp": float(best["origin_timestamp"]),
        "jma_latitude": float(best["latitude"]),
        "jma_longitude": float(best["longitude"]),
        "jma_depth_km": np.nan if pd.isna(best["depth_km"]) else float(best["depth_km"]),
        "jma_magnitude": np.nan if pd.isna(best["magnitude"]) else float(best["magnitude"]),
        "jma_region": str(best["region"]),
        "jma_source_url": str(best["source_url"]),
        "jma_source_date": str(best["source_date"]),
        "time_diff_s": float(best["time_diff_s"]),
        "abs_time_diff_s": float(best["abs_time_diff_s"]),
        "horizontal_distance_km": float(best["horizontal_distance_km"]),
        "depth_diff_km": np.nan if pd.isna(best["depth_diff_km"]) else float(best["depth_diff_km"]),
        "magnitude_diff": np.nan if pd.isna(best["magnitude_diff"]) else float(best["magnitude_diff"]),
        "match_score": float(best["match_score"]),
        "second_match_score": second_score,
        "score_margin": score_margin,
    }
    if second is not None:
        out.update(
            {
                "second_jma_event_id": str(second["jma_event_id"]),
                "second_jma_origin_time_jst": str(second["origin_time_jst"]),
                "second_time_diff_s": float(second["time_diff_s"]),
                "second_horizontal_distance_km": float(second["horizontal_distance_km"]),
                "second_depth_diff_km": np.nan if pd.isna(second["depth_diff_km"]) else float(second["depth_diff_km"]),
                "second_magnitude_diff": np.nan if pd.isna(second["magnitude_diff"]) else float(second["magnitude_diff"]),
            }
        )
    return out


def match_events(events: pd.DataFrame, catalog: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = [match_one_event(row, catalog, args) for _, row in events.iterrows()]
    return pd.DataFrame(rows)


def write_outputs(events: pd.DataFrame, catalog: pd.DataFrame, matches: pd.DataFrame, args: argparse.Namespace) -> None:
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.output_csv, index=False)
    if args.catalog_csv is not None:
        args.catalog_csv.parent.mkdir(parents=True, exist_ok=True)
        catalog.to_csv(args.catalog_csv, index=False)
    if args.summary_json is not None:
        accepted = matches["accepted"].astype(int) if "accepted" in matches else pd.Series(dtype=int)
        summary = {
            "training_event_count": int(len(events)),
            "jma_catalog_count": int(len(catalog)),
            "match_status_counts": matches["match_status"].value_counts(dropna=False).to_dict()
            if "match_status" in matches
            else {},
            "accepted_count": int(accepted.sum()) if not accepted.empty else 0,
            "output_csv": str(args.output_csv),
            "catalog_csv": "" if args.catalog_csv is None else str(args.catalog_csv),
        }
        if "origin_time_correction_s" in matches:
            correction = pd.to_numeric(matches.loc[matches["accepted"].astype(int) == 1, "origin_time_correction_s"], errors="coerce")
            correction = correction[np.isfinite(correction)]
            summary.update(
                {
                    "accepted_correction_median_s": float(correction.median()) if correction.size else None,
                    "accepted_correction_p05_s": float(correction.quantile(0.05)) if correction.size else None,
                    "accepted_correction_p95_s": float(correction.quantile(0.95)) if correction.size else None,
                }
            )
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_argument_group("inputs")
    inputs.add_argument("--hdf5", type=Path, default=None, help="Training HDF5 containing metadata/event_metadata.")
    inputs.add_argument("--events-csv", type=Path, default=None, help="Training events CSV.")
    inputs.add_argument("--stations-csv", type=Path, default=None, help="Training stations CSV; events are de-duplicated.")
    inputs.add_argument("--event-id", action="append", default=None, help="Restrict to event id. Repeatable.")
    inputs.add_argument("--limit-events", type=int, default=None, help="Limit number of training events.")

    outputs = parser.add_argument_group("outputs")
    outputs.add_argument("--output-csv", type=Path, required=True, help="Matched origin correction CSV.")
    outputs.add_argument("--catalog-csv", type=Path, default=None, help="Optional parsed JMA daily catalog CSV.")
    outputs.add_argument("--summary-json", type=Path, default=None, help="Optional run summary JSON.")
    outputs.add_argument("--cache-dir", type=Path, default=Path("jma_daily_cache"), help="Directory for cached JMA daily HTML.")

    fetch = parser.add_argument_group("fetching")
    fetch.add_argument("--refresh", action="store_true", help="Re-download cached JMA daily pages.")
    fetch.add_argument("--cache-only", action="store_true", help="Use existing cached pages only.")
    fetch.add_argument("--allow-missing-days", action="store_true", help="Skip missing/404 daily pages.")
    fetch.add_argument(
        "--catalog-source",
        choices=["auto", "daily", "bulletin"],
        default="auto",
        help=(
            "JMA source for hypocenters. auto uses annual bulletin zips through "
            "--bulletin-max-year and daily_map pages for newer years."
        ),
    )
    fetch.add_argument(
        "--bulletin-max-year",
        type=int,
        default=2023,
        help="Latest year to read from annual Seismological Bulletin zips when --catalog-source=auto.",
    )
    fetch.add_argument("--date-padding-days", type=int, default=0, help="Also fetch neighboring days around each raw origin date.")
    fetch.add_argument("--timeout-seconds", type=float, default=30.0)
    fetch.add_argument("--request-sleep-seconds", type=float, default=0.2)
    fetch.add_argument("--user-agent", default="team-pytorch-jma-origin-correction/1.0")

    match = parser.add_argument_group("matching")
    match.add_argument("--max-time-diff-seconds", type=float, default=90.0)
    match.add_argument("--max-distance-km", type=float, default=20.0)
    match.add_argument("--max-depth-diff-km", type=float, default=30.0)
    match.add_argument("--max-magnitude-diff", type=float, default=1.0)
    match.add_argument("--time-weight-seconds", type=float, default=30.0)
    match.add_argument("--distance-weight-km", type=float, default=5.0)
    match.add_argument("--depth-weight-km", type=float, default=10.0)
    match.add_argument("--magnitude-weight", type=float, default=0.3)
    match.add_argument("--min-score-margin", type=float, default=0.5)
    match.add_argument("--accept-ambiguous", action="store_true", help="Set accepted=1 even when the best/second-best margin is small.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    for attr in ("hdf5", "events_csv", "stations_csv", "output_csv", "catalog_csv", "summary_json", "cache_dir"):
        value = getattr(args, attr)
        if value is not None:
            setattr(args, attr, Path(value).expanduser().resolve())
    if args.date_padding_days < 0:
        raise SystemExit("--date-padding-days must be non-negative")
    if args.bulletin_max_year < 0:
        raise SystemExit("--bulletin-max-year must be non-negative")
    for name in (
        "max_time_diff_seconds",
        "max_distance_km",
        "max_depth_diff_km",
        "max_magnitude_diff",
        "time_weight_seconds",
        "distance_weight_km",
        "depth_weight_km",
        "magnitude_weight",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")

    events = load_training_events(args)
    catalog = load_jma_catalog(events, args)
    matches = match_events(events, catalog, args)
    write_outputs(events, catalog, matches, args)
    accepted = int(matches["accepted"].astype(int).sum()) if "accepted" in matches else 0
    print(f"Loaded {len(events)} training events, parsed {len(catalog)} JMA events, accepted {accepted} matches.")
    print(f"Origin corrections: {args.output_csv}")
    if args.catalog_csv is not None:
        print(f"JMA catalog: {args.catalog_csv}")


if __name__ == "__main__":
    main()
