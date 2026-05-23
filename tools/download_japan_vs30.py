#!/usr/bin/env python3
"""Download station VS30 values for Japan from the J-SHIS mesh API.

J-SHIS exposes AVS, the 30 m average S-wave velocity, on the Japan 250 m mesh.
This script queries the nearest mesh cell for each station coordinate and writes
a reusable station-level CSV for japan_dataset_builder.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


JSHIS_MESHSEARCH_URL = "https://www.j-shis.bosai.go.jp/map/api/meshsearch"
VS30_SOURCE_LABEL = "j-shis_meshsearch_avs"


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
    lookup = {str(c).lower(): str(c) for c in columns}
    for candidate in candidates:
        found = lookup.get(candidate.lower())
        if found is not None:
            return found
    if required:
        raise KeyError(f"None of {list(candidates)} found in columns {list(columns)}")
    return None


def load_station_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"station_code": str})
    code_key = find_column(df.columns, ["station_code", "Station Code", "code", "station"])
    lat_key = find_column(df.columns, ["station_lat", "Station Lat.", "lat", "latitude"])
    lon_key = find_column(df.columns, ["station_lon", "Station Long.", "lon", "longitude", "long"])
    out = pd.DataFrame(
        {
            "station_code": df[code_key].astype(str),
            "station_lat": pd.to_numeric(df[lat_key], errors="coerce"),
            "station_lon": pd.to_numeric(df[lon_key], errors="coerce"),
            "input_source": str(path),
        }
    )
    return out


def load_station_hdf5(path: Path) -> pd.DataFrame:
    try:
        df = read_metadata_table(path, "station_metadata")
        code_key = find_column(df.columns, ["station_code", "Station Code", "code", "station"])
        lat_key = find_column(df.columns, ["station_lat", "Station Lat.", "lat", "latitude"])
        lon_key = find_column(df.columns, ["station_lon", "Station Long.", "lon", "longitude", "long"])
        return pd.DataFrame(
            {
                "station_code": df[code_key].astype(str),
                "station_lat": pd.to_numeric(df[lat_key], errors="coerce"),
                "station_lon": pd.to_numeric(df[lon_key], errors="coerce"),
                "input_source": str(path),
            }
        )
    except KeyError:
        rows: list[dict[str, object]] = []
        with h5py.File(path, "r") as h5:
            for event_id, group in h5.get("data", {}).items():
                if "coords" not in group:
                    continue
                coords = np.asarray(group["coords"][()])
                if "station_codes" in group:
                    codes = _decode_array(group["station_codes"][()])
                else:
                    codes = [f"{event_id}:{i}" for i in range(coords.shape[0])]
                for i, code in enumerate(codes):
                    rows.append(
                        {
                            "station_code": str(code),
                            "station_lat": float(coords[i, 0]),
                            "station_lon": float(coords[i, 1]),
                            "input_source": str(path),
                        }
                    )
        if not rows:
            raise
        return pd.DataFrame(rows)


def load_station_inputs(input_csvs: list[Path], hdf5_paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in input_csvs:
        frames.append(load_station_csv(path.expanduser().resolve()))
    for path in hdf5_paths:
        frames.append(load_station_hdf5(path.expanduser().resolve()))
    if not frames:
        raise SystemExit("Provide at least one --input-csv or --hdf5.")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["station_lat", "station_lon"]).copy()
    df["station_code"] = df["station_code"].astype(str).str.strip()
    df["station_lat_round"] = df["station_lat"].round(6)
    df["station_lon_round"] = df["station_lon"].round(6)
    df = df.drop_duplicates(["station_code", "station_lat_round", "station_lon_round"])
    df = df.sort_values(["station_code", "station_lat_round", "station_lon_round"]).reset_index(drop=True)
    return df.drop(columns=["station_lat_round", "station_lon_round"])


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    lat1_rad = math.radians(float(lat1))
    lon1_rad = math.radians(float(lon1))
    lat2_rad = math.radians(float(lat2))
    lon2_rad = math.radians(float(lon2))
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1e-15, 1.0 - a)))
    return radius_km * c


def _flatten_points(coords):
    if not isinstance(coords, (list, tuple)):
        return
    if len(coords) >= 2 and all(isinstance(v, (int, float)) for v in coords[:2]):
        yield float(coords[1]), float(coords[0])
        return
    for item in coords:
        yield from _flatten_points(item)


def geometry_distance_km(geometry: dict, station_lat: float, station_lon: float) -> float:
    points = list(_flatten_points(geometry.get("coordinates", [])))
    if not points:
        return float("nan")
    lat = float(np.mean([p[0] for p in points]))
    lon = float(np.mean([p[1] for p in points]))
    return haversine_distance_km(station_lat, station_lon, lat, lon)


def parse_vs30_response(payload: dict, station_lat: float, station_lon: float) -> dict[str, object]:
    features = payload.get("features")
    if not features:
        return {
            "vs30_mps": np.nan,
            "vs30_valid": 0,
            "vs30_source": VS30_SOURCE_LABEL,
            "vs30_mesh_code": "",
            "vs30_query_distance_km": np.nan,
            "arv": np.nan,
            "status": str(payload.get("status", "no_features")),
            "error": "",
        }
    feature = features[0]
    props = feature.get("properties", {}) or {}
    avs_value = props.get("AVS", props.get("avs", props.get("Vs30", props.get("vs30"))))
    try:
        vs30 = float(avs_value)
    except (TypeError, ValueError):
        vs30 = float("nan")
    valid = int(np.isfinite(vs30) and vs30 > 0)
    dist = props.get("DIST", props.get("distance", np.nan))
    try:
        dist = float(dist)
    except (TypeError, ValueError):
        dist = geometry_distance_km(feature.get("geometry", {}) or {}, station_lat, station_lon)
    return {
        "vs30_mps": vs30,
        "vs30_valid": valid,
        "vs30_source": VS30_SOURCE_LABEL,
        "vs30_mesh_code": str(props.get("meshcode", props.get("CODE", props.get("code", "")))),
        "vs30_jcode": str(props.get("JCODE", props.get("jcode", ""))),
        "vs30_query_distance_km": dist,
        "arv": props.get("ARV", props.get("arv", np.nan)),
        "status": str(payload.get("status", "ok")),
        "error": "",
    }


def query_jshis_vs30(
    station_lat: float,
    station_lon: float,
    radius_km: float,
    limit: int,
    timeout: float,
    version: str | None,
) -> dict[str, object]:
    params = {
        "center": f"{float(station_lon):.8f},{float(station_lat):.8f}",
        "epsg": "4326",
        "format": "geojson",
        "filter": "AVS_gt_0",
        "radius": f"{float(radius_km):.3f}",
        "limit": str(int(limit)),
        "order": "DIST",
    }
    if version and version.lower() not in {"default", "latest", "none"}:
        params["version"] = version
    url = JSHIS_MESHSEARCH_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "team-pytorch-vs30/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8")
    payload = json.loads(text)
    row = parse_vs30_response(payload, station_lat=station_lat, station_lon=station_lon)
    row["api_url"] = url
    return row


def row_key(row: pd.Series) -> tuple[str, float, float]:
    return (
        str(row["station_code"]),
        round(float(row["station_lat"]), 6),
        round(float(row["station_lon"]), 6),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Japan station VS30 from J-SHIS AVS mesh data.")
    parser.add_argument("--input-csv", action="append", default=[], type=Path, help="Station CSV. Repeatable.")
    parser.add_argument("--hdf5", action="append", default=[], type=Path, help="Training HDF5 with station metadata. Repeatable.")
    parser.add_argument("--output", required=True, type=Path, help="Output station VS30 CSV.")
    parser.add_argument("--radius-km", type=float, default=1.0, help="Mesh search radius in km.")
    parser.add_argument("--limit", type=int, default=1, help="Number of mesh candidates requested from J-SHIS.")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds.")
    parser.add_argument("--sleep", type=float, default=0.1, help="Polite pause between API requests.")
    parser.add_argument("--version", default="V4", help="J-SHIS API version. Use 'default' to omit.")
    parser.add_argument("--force", action="store_true", help="Re-query rows already present in --output.")
    parser.add_argument("--max-stations", type=int, default=None, help="Optional cap for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stations = load_station_inputs(args.input_csv, args.hdf5)
    if args.max_stations is not None:
        stations = stations.head(int(args.max_stations)).copy()
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    existing = pd.DataFrame()
    done: set[tuple[str, float, float]] = set()
    if args.output.exists() and not args.force:
        existing = pd.read_csv(args.output, dtype={"station_code": str})
        if not existing.empty:
            for _, row in existing.iterrows():
                done.add(row_key(row))

    rows = [] if existing.empty or args.force else existing.to_dict(orient="records")
    total = len(stations)
    for idx, station in stations.iterrows():
        key = row_key(station)
        if key in done:
            continue
        base = {
            "station_code": station["station_code"],
            "station_lat": float(station["station_lat"]),
            "station_lon": float(station["station_lon"]),
            "input_source": station.get("input_source", ""),
            "query_radius_km": float(args.radius_km),
            "api_version": "" if args.version.lower() in {"default", "latest", "none"} else args.version,
        }
        try:
            result = query_jshis_vs30(
                station_lat=base["station_lat"],
                station_lon=base["station_lon"],
                radius_km=args.radius_km,
                limit=args.limit,
                timeout=args.timeout,
                version=args.version,
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            result = {
                "vs30_mps": np.nan,
                "vs30_valid": 0,
                "vs30_source": VS30_SOURCE_LABEL,
                "vs30_mesh_code": "",
                "vs30_jcode": "",
                "vs30_query_distance_km": np.nan,
                "arv": np.nan,
                "status": "request_failed",
                "error": str(exc),
                "api_url": "",
            }
        rows.append({**base, **result})
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(
            f"[{len(rows)}/{total}] {base['station_code']} "
            f"vs30={result.get('vs30_mps')} valid={result.get('vs30_valid')} status={result.get('status')}",
            flush=True,
        )
        if args.sleep > 0:
            time.sleep(float(args.sleep))

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    valid = int(pd.to_numeric(out.get("vs30_valid", 0), errors="coerce").fillna(0).sum()) if not out.empty else 0
    print(f"Wrote {len(out)} station rows to {args.output} ({valid} valid VS30 values).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
