#!/usr/bin/env python3
"""Download F-net strong-motion windows for existing Japanese strong-motion events.

The input strong-motion tree is expected to match japan_dataset_builder.py:

  strong_root/
    2024/
      20240101000000.tar

For each event archive, this script reads event origin metadata from the
existing strong-motion headers, queries FDSN station metadata for F-net, predicts
P arrival times at all matching stations, and downloads raw miniSEED windows
around those predicted arrivals.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import gzip
import io
import json
import math
import sys
import tarfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from japan_dataset_builder import (  # noqa: E402
    AK135TravelTimeModel,
    COMPONENT_ORDER,
    DEFAULT_JMA2001A_ZIP,
    INNER_COMPONENT_RE,
    JMATravelTimeTable,
    OUTER_EVENT_RE,
    build_station_trace,
    haversine_distance_km,
    parse_component_fileobj,
    parse_jst_timestamp,
)


@dataclass
class EventInfo:
    event_id: str
    archive_path: str
    origin_time_jst: str
    origin_time_utc: str
    origin_timestamp: float
    latitude: float
    longitude: float
    depth_km: float
    magnitude: float
    source_station_code: str
    source_network: str


def parse_years(text: str | None) -> set[int] | None:
    if not text:
        return None
    years: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            years.update(range(start, end + 1))
        else:
            years.add(int(part))
    return years


def iter_event_archives(strong_root: Path, years: set[int] | None, limit: int | None) -> Iterable[Path]:
    year_dirs = [p for p in strong_root.iterdir() if p.is_dir() and p.name.isdigit()]
    year_dirs = sorted(p for p in year_dirs if years is None or int(p.name) in years)
    count = 0
    for year_dir in year_dirs:
        for archive in sorted(year_dir.glob("*.tar")):
            if limit is not None and count >= limit:
                return
            count += 1
            yield archive


def read_event_info(archive_path: Path, strong_root: Path) -> EventInfo:
    match = OUTER_EVENT_RE.match(archive_path.name)
    if not match:
        raise ValueError(f"Unexpected event archive name: {archive_path}")
    event_id = match.group("event_id")
    archive_relpath = str(archive_path.relative_to(strong_root))

    with tarfile.open(archive_path, "r") as outer_tar:
        inner_members = [
            m for m in outer_tar.getmembers()
            if m.isfile() and m.name.endswith((".knt.tar.gz", ".kik.tar.gz"))
        ]
        for inner_member in inner_members:
            source_network = "knt" if inner_member.name.endswith(".knt.tar.gz") else "kik"
            payload = outer_tar.extractfile(inner_member).read()
            with tarfile.open(fileobj=io.BytesIO(gzip.decompress(payload)), mode="r:") as inner_tar:
                grouped: dict[str, dict[str, tarfile.TarInfo]] = {}
                for member in inner_tar.getmembers():
                    if not member.isfile():
                        continue
                    filename = Path(member.name).name
                    inner_match = INNER_COMPONENT_RE.match(filename)
                    if not inner_match:
                        continue
                    group_key = f"{inner_match.group('base')}__{inner_match.group('suffix') or '0'}"
                    grouped.setdefault(group_key, {})[inner_match.group("comp")] = member

                for group_key, component_members in grouped.items():
                    if any(comp not in component_members for comp in COMPONENT_ORDER):
                        continue
                    traces = {}
                    component_base = group_key.split("__", 1)[0]
                    for comp in COMPONENT_ORDER:
                        with inner_tar.extractfile(component_members[comp]) as fileobj:
                            traces[comp] = parse_component_fileobj(fileobj, component_members[comp].name)
                    station = build_station_trace(
                        event_id=event_id,
                        archive_relpath=archive_relpath,
                        inner_archive_name=inner_member.name,
                        component_base=component_base,
                        source_network=source_network,
                        traces=traces,
                        target_sampling_rate_hz=100.0,
                    )
                    origin_dt, origin_ts = parse_jst_timestamp(station.origin_time_raw)
                    return EventInfo(
                        event_id=event_id,
                        archive_path=str(archive_path),
                        origin_time_jst=origin_dt.isoformat(),
                        origin_time_utc=datetime.fromtimestamp(origin_ts, tz=timezone.utc).isoformat(),
                        origin_timestamp=float(origin_ts),
                        latitude=float(station.event_lat),
                        longitude=float(station.event_lon),
                        depth_km=float(station.event_depth_km),
                        magnitude=float(station.magnitude),
                        source_station_code=station.station_code,
                        source_network=source_network,
                    )
    raise ValueError(f"No complete station metadata found in {archive_path}")


def build_travel_time_model(model_name: str, jma_zip: Path | None):
    jma_table = None
    ak135 = None
    if model_name == "jma2001a":
        try:
            jma_table = JMATravelTimeTable(jma_zip or DEFAULT_JMA2001A_ZIP)
        except Exception as exc:
            print(f"[WARN] failed to load JMA travel-time table ({exc}); falling back to ak135", file=sys.stderr)
            ak135 = AK135TravelTimeModel()
        if ak135 is None:
            ak135 = AK135TravelTimeModel()
    elif model_name == "ak135":
        ak135 = AK135TravelTimeModel()
    elif model_name == "constant":
        pass
    else:
        raise ValueError(f"Unsupported travel-time model: {model_name}")
    return jma_table, ak135


def predict_p_seconds(
    event: EventInfo,
    station_lat: float,
    station_lon: float,
    station_elevation_m: float,
    model_name: str,
    jma_table,
    ak135,
    constant_velocity_km_s: float,
) -> tuple[float, str, bool, float, float]:
    epicentral_km = haversine_distance_km(event.latitude, event.longitude, station_lat, station_lon)
    station_height_km = station_elevation_m / 1000.0
    hypocentral_km = math.sqrt(epicentral_km ** 2 + (event.depth_km + station_height_km) ** 2)
    clipped = False
    used = model_name
    if model_name == "jma2001a":
        if jma_table is None:
            p_seconds = ak135.predict_p_seconds(event.depth_km, epicentral_km)
            used = "ak135"
            clipped = True
        else:
            p_seconds, clipped = jma_table.predict_p_seconds(
                depth_km=event.depth_km,
                distance_km=epicentral_km,
                station_height_m=station_elevation_m,
            )
            if clipped:
                p_seconds = ak135.predict_p_seconds(event.depth_km, epicentral_km)
                used = "ak135"
    elif model_name == "ak135":
        p_seconds = ak135.predict_p_seconds(event.depth_km, epicentral_km)
    else:
        p_seconds = hypocentral_km / float(constant_velocity_km_s)
    return float(p_seconds), used, bool(clipped), float(epicentral_km), float(hypocentral_km)


def split_patterns(patterns: str) -> list[str]:
    return [part.strip() for part in patterns.split(",") if part.strip()]


def matches_any(value: str, patterns: str) -> bool:
    return any(fnmatch.fnmatchcase(value, pattern) for pattern in split_patterns(patterns))


def inventory_station_rows(inventory, channel_patterns: str) -> list[dict[str, object]]:
    rows = []
    seen = set()
    for net in inventory:
        for sta in net:
            channels = [
                cha for cha in sta.channels
                if matches_any(cha.code, channel_patterns)
            ]
            if not channels:
                continue
            key = (net.code, sta.code)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "network": net.code,
                "station": sta.code,
                "latitude": float(sta.latitude),
                "longitude": float(sta.longitude),
                "elevation_m": float(sta.elevation),
                "site_name": getattr(sta.site, "name", "") if sta.site is not None else "",
                "channel_codes": ",".join(sorted({cha.code for cha in channels})),
                "location_codes": ",".join(sorted({cha.location_code or "" for cha in channels})),
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=2, ensure_ascii=False)


def make_client(args):
    from obspy.clients.fdsn import Client

    if args.fdsn_base_url:
        return Client(base_url=args.fdsn_base_url, timeout=args.timeout)
    return Client(args.fdsn_client, timeout=args.timeout)


def utc_from_timestamp(ts: float):
    from obspy import UTCDateTime

    return UTCDateTime(ts)


def process_event(args, client, event: EventInfo, jma_table, ak135) -> dict[str, int]:
    event_dir = args.output_root / "events" / event.event_id
    waveform_dir = event_dir / "waveforms"
    waveform_dir.mkdir(parents=True, exist_ok=True)

    event_origin = utc_from_timestamp(event.origin_timestamp)
    inventory_start = event_origin - args.inventory_pre_seconds
    inventory_end = event_origin + args.inventory_post_seconds

    stats = {"events": 1, "stations": 0, "downloaded": 0, "skipped_existing": 0, "failed": 0}
    log_rows: list[dict[str, object]] = []
    arrival_rows: list[dict[str, object]] = []

    write_json(event_dir / "event.json", asdict(event))

    try:
        inventory = client.get_stations(
            network=args.network,
            station=args.station,
            location=args.location,
            channel=args.channels,
            starttime=inventory_start,
            endtime=inventory_end,
            level="response",
        )
    except Exception as exc:
        log_rows.append({
            "event_id": event.event_id,
            "network": args.network,
            "station": args.station,
            "status": "inventory_failed",
            "error": repr(exc),
        })
        write_csv(event_dir / "download_log.csv", log_rows, DOWNLOAD_LOG_FIELDS)
        stats["failed"] += 1
        return stats

    inventory.write(str(event_dir / "inventory.xml"), format="STATIONXML")
    station_rows = inventory_station_rows(inventory, args.channels)
    write_csv(event_dir / "stations.csv", station_rows, STATION_FIELDS)

    for row in station_rows:
        stats["stations"] += 1
        try:
            p_seconds, used_model, clipped, epi_km, hypo_km = predict_p_seconds(
                event=event,
                station_lat=float(row["latitude"]),
                station_lon=float(row["longitude"]),
                station_elevation_m=float(row["elevation_m"]),
                model_name=args.travel_time_model,
                jma_table=jma_table,
                ak135=ak135,
                constant_velocity_km_s=args.constant_p_velocity_km_s,
            )
            arrival = utc_from_timestamp(event.origin_timestamp + p_seconds)
            start = arrival - args.pre_seconds
            end = arrival + args.post_seconds
            channel_label = args.channels.replace(",", "_").replace("?", "Q").replace("*", "STAR")
            location_label = args.location.replace(",", "_").replace("*", "STAR")
            out_name = (
                f"{row['network']}.{row['station']}."
                f"{location_label}.{channel_label}."
                f"{start.strftime('%Y%m%dT%H%M%S')}_{end.strftime('%Y%m%dT%H%M%S')}.mseed"
            )
            out_path = waveform_dir / out_name
            arrival_rows.append({
                "event_id": event.event_id,
                "network": row["network"],
                "station": row["station"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "elevation_m": row["elevation_m"],
                "epicentral_distance_km": epi_km,
                "hypocentral_distance_km": hypo_km,
                "p_seconds_after_origin": p_seconds,
                "p_arrival_utc": arrival.isoformat(),
                "window_start_utc": start.isoformat(),
                "window_end_utc": end.isoformat(),
                "travel_time_model": used_model,
                "jma_grid_clipped": int(clipped),
                "waveform_file": str(out_path.relative_to(event_dir)),
            })
            if out_path.exists() and not args.overwrite:
                stats["skipped_existing"] += 1
                log_rows.append({
                    "event_id": event.event_id,
                    "network": row["network"],
                    "station": row["station"],
                    "status": "skipped_existing",
                    "start_utc": start.isoformat(),
                    "end_utc": end.isoformat(),
                    "path": str(out_path),
                })
                continue
            if args.dry_run:
                log_rows.append({
                    "event_id": event.event_id,
                    "network": row["network"],
                    "station": row["station"],
                    "status": "dry_run",
                    "start_utc": start.isoformat(),
                    "end_utc": end.isoformat(),
                    "path": str(out_path),
                })
                continue
            stream = client.get_waveforms(
                network=str(row["network"]),
                station=str(row["station"]),
                location=args.location,
                channel=args.channels,
                starttime=start,
                endtime=end,
                attach_response=False,
            )
            if len(stream) == 0:
                raise RuntimeError("empty stream")
            stream.write(str(out_path), format="MSEED")
            stats["downloaded"] += 1
            log_rows.append({
                "event_id": event.event_id,
                "network": row["network"],
                "station": row["station"],
                "status": "downloaded",
                "start_utc": start.isoformat(),
                "end_utc": end.isoformat(),
                "path": str(out_path),
                "n_traces": len(stream),
            })
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
        except Exception as exc:
            stats["failed"] += 1
            log_rows.append({
                "event_id": event.event_id,
                "network": row.get("network", ""),
                "station": row.get("station", ""),
                "status": "failed",
                "error": repr(exc),
            })

    write_csv(event_dir / "arrivals.csv", arrival_rows, ARRIVAL_FIELDS)
    write_csv(event_dir / "download_log.csv", log_rows, DOWNLOAD_LOG_FIELDS)
    return stats


STATION_FIELDS = [
    "network",
    "station",
    "latitude",
    "longitude",
    "elevation_m",
    "site_name",
    "channel_codes",
    "location_codes",
]

ARRIVAL_FIELDS = [
    "event_id",
    "network",
    "station",
    "latitude",
    "longitude",
    "elevation_m",
    "epicentral_distance_km",
    "hypocentral_distance_km",
    "p_seconds_after_origin",
    "p_arrival_utc",
    "window_start_utc",
    "window_end_utc",
    "travel_time_model",
    "jma_grid_clipped",
    "waveform_file",
]

DOWNLOAD_LOG_FIELDS = [
    "event_id",
    "network",
    "station",
    "status",
    "start_utc",
    "end_utc",
    "path",
    "n_traces",
    "error",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download F-net windows around predicted P arrivals for existing Japanese events."
    )
    parser.add_argument("--strong-root", type=Path, required=True, help="Existing strong-motion waveform root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory for downloaded F-net data.")
    parser.add_argument("--years", type=str, default=None, help="Years to process, e.g. 2024 or 2020-2024.")
    parser.add_argument("--limit-events", type=int, default=None, help="Maximum number of events to process.")
    parser.add_argument("--network", default="BO", help="FDSN network code. F-net is BO.")
    parser.add_argument("--station", default="*", help="FDSN station selector.")
    parser.add_argument("--location", default="*", help="FDSN location selector.")
    parser.add_argument("--channels", default="HN?", help="FDSN channel selector for strong-motion channels.")
    parser.add_argument("--fdsn-client", default="IRIS", help="ObsPy FDSN client name when --fdsn-base-url is unset.")
    parser.add_argument("--fdsn-base-url", default=None, help="Explicit FDSN base URL for direct provider access.")
    parser.add_argument("--timeout", type=float, default=120.0, help="FDSN request timeout in seconds.")
    parser.add_argument("--pre-seconds", type=float, default=100.0, help="Seconds before predicted P arrival.")
    parser.add_argument("--post-seconds", type=float, default=100.0, help="Seconds after predicted P arrival.")
    parser.add_argument("--inventory-pre-seconds", type=float, default=300.0, help="Station query start offset before origin.")
    parser.add_argument("--inventory-post-seconds", type=float, default=1800.0, help="Station query end offset after origin.")
    parser.add_argument(
        "--travel-time-model",
        choices=["jma2001a", "ak135", "constant"],
        default="jma2001a",
    )
    parser.add_argument("--jma-travel-time-zip", type=Path, default=None)
    parser.add_argument("--constant-p-velocity-km-s", type=float, default=6.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Sleep after each successful waveform request.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing miniSEED files.")
    parser.add_argument("--dry-run", action="store_true", help="Write metadata and planned requests without downloading waveforms.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.strong_root = args.strong_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    years = parse_years(args.years)

    if not args.strong_root.exists():
        raise FileNotFoundError(f"strong root not found: {args.strong_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    jma_table, ak135 = build_travel_time_model(args.travel_time_model, args.jma_travel_time_zip)
    client = make_client(args)

    summary_rows = []
    total = {"events": 0, "stations": 0, "downloaded": 0, "skipped_existing": 0, "failed": 0}
    for archive in iter_event_archives(args.strong_root, years, args.limit_events):
        try:
            event = read_event_info(archive, args.strong_root)
            print(f"[INFO] event {event.event_id} origin={event.origin_time_utc}")
            stats = process_event(args, client, event, jma_table, ak135)
            summary_rows.append({"event_id": event.event_id, **stats})
            for key, value in stats.items():
                total[key] = total.get(key, 0) + int(value)
        except Exception as exc:
            event_id = archive.stem
            print(f"[WARN] event {event_id} failed: {exc!r}", file=sys.stderr)
            summary_rows.append({"event_id": event_id, "events": 1, "stations": 0, "downloaded": 0, "skipped_existing": 0, "failed": 1})
            total["events"] += 1
            total["failed"] += 1

    summary_fields = ["event_id", "events", "stations", "downloaded", "skipped_existing", "failed"]
    write_csv(args.output_root / "summary.csv", summary_rows, summary_fields)
    write_json(args.output_root / "summary.json", total)
    print(f"[INFO] done: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
