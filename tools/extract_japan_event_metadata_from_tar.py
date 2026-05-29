#!/usr/bin/env python
"""Extract event-level metadata from raw Japan K-NET/KiK-net tar archives.

This is a lightweight pre-scan used before building the final training HDF5.
It reads only the text headers of component files, writes one row per event,
and is intended as input to tools/fetch_jma_hypocenters.py --events-csv.
"""

from __future__ import annotations

import argparse
import gzip
import io
import sys
import tarfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from japan_dataset_builder import (  # noqa: E402
    COMPONENT_ORDER,
    INNER_ARCHIVE_SUFFIXES,
    INNER_COMPONENT_RE,
    OUTER_EVENT_RE,
)


REQUIRED_EVENT_KEYS = (
    "Origin Time",
    "Lat.",
    "Long.",
    "Depth. (km)",
    "Mag.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waveform-root", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--min-stations", type=int, default=3)
    parser.add_argument("--limit-events", type=int, default=None)
    return parser.parse_args()


def parse_header_fileobj(fileobj) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for raw_line in fileobj:
        line = raw_line.decode("utf-8", errors="ignore")
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Memo."):
            break
        key = line[:18].strip()
        value = line[18:].strip()
        if key and value:
            metadata[key] = value
    return metadata


def source_network(inner_member_name: str) -> str:
    if ".knt." in inner_member_name:
        return "knt"
    if ".kik." in inner_member_name:
        return "kik"
    return "unknown"


def scan_event_archive(outer_tar_path: Path, waveform_root: Path, min_stations: int) -> dict[str, object] | None:
    outer_match = OUTER_EVENT_RE.match(outer_tar_path.name)
    if not outer_match:
        return None
    event_id = outer_match.group("event_id")
    archive_relpath = str(outer_tar_path.relative_to(waveform_root))

    complete_station_count = 0
    source_networks: set[str] = set()
    first_event_header: dict[str, str] | None = None

    with tarfile.open(outer_tar_path, "r") as outer_tar:
        inner_members = [
            member
            for member in outer_tar.getmembers()
            if member.isfile() and member.name.endswith(INNER_ARCHIVE_SUFFIXES)
        ]
        for inner_member in inner_members:
            source_networks.add(source_network(inner_member.name))
            payload_file = outer_tar.extractfile(inner_member)
            if payload_file is None:
                continue
            payload = payload_file.read()
            if inner_member.name.endswith(".gz"):
                payload = gzip.decompress(payload)
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as inner_tar:
                grouped: dict[str, dict[str, tarfile.TarInfo]] = {}
                for member in inner_tar.getmembers():
                    if not member.isfile():
                        continue
                    filename = Path(member.name).name
                    match = INNER_COMPONENT_RE.match(filename)
                    if not match:
                        continue
                    group_key = f"{match.group('base')}__{match.group('suffix') or '0'}"
                    grouped.setdefault(group_key, {})[match.group("comp")] = member

                for component_members in grouped.values():
                    if any(comp not in component_members for comp in COMPONENT_ORDER):
                        continue
                    complete_station_count += 1
                    if first_event_header is None:
                        with inner_tar.extractfile(component_members["UD"]) as fileobj:
                            if fileobj is None:
                                continue
                            header = parse_header_fileobj(fileobj)
                        if all(key in header for key in REQUIRED_EVENT_KEYS):
                            first_event_header = header

    if complete_station_count < min_stations or first_event_header is None:
        return None

    return {
        "EVENT": event_id,
        "Origin_Time(JST)": first_event_header["Origin Time"],
        "Latitude": float(first_event_header["Lat."]),
        "Longitude": float(first_event_header["Long."]),
        "DEPTH": float(first_event_header["Depth. (km)"]),
        "Magnitude": float(first_event_header["Mag."]),
        "N_Stations": int(complete_station_count),
        "Source_Mix": ",".join(sorted(source_networks)),
        "Archive_Path": archive_relpath,
    }


def main() -> None:
    args = parse_args()
    waveform_root = args.waveform_root.expanduser().resolve()
    year_dir = waveform_root / str(args.year)
    if not year_dir.is_dir():
        raise SystemExit(f"[ERROR] waveform year directory not found: {year_dir}")

    archives = sorted(year_dir.glob("*.tar"))
    if args.limit_events is not None:
        archives = archives[: args.limit_events]
    if not archives:
        raise SystemExit(f"[ERROR] no event tar archives found under {year_dir}")

    rows: list[dict[str, object]] = []
    failed = 0
    for archive in archives:
        try:
            row = scan_event_archive(archive, waveform_root, min_stations=args.min_stations)
        except Exception as exc:
            failed += 1
            print(f"[WARN] failed to scan {archive}: {exc}", file=sys.stderr)
            continue
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit(f"[ERROR] no usable event metadata rows extracted from {year_dir}")

    out = pd.DataFrame(rows)
    args.output_csv.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"[INFO] wrote event metadata CSV: {args.output_csv}")
    print(f"[INFO] events scanned: {len(archives)}, usable: {len(out)}, failed: {failed}")


if __name__ == "__main__":
    main()
