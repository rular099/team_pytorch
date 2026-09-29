#!/usr/bin/env python3
"""Hi-net raw-archive backend used by the V01 derived-cache builder.

Raw CNT samples are converted with the archived CH sensitivity only.  No full
response deconvolution or filtering is performed.  Readers are opened lazily
per process and kept in a bounded LRU, matching the archive's worker contract.
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np

from tools.hinet_raw_archive import AnnualHinetArchiveReader


BACKEND_VERSION = "v01-hinet-sensitivity-only-v1"
COMPONENT_ORDER = ("E", "N", "U")  # existing model order: EW, NS, vertical


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_archives(root: str | Path, allow_partial: bool = True) -> list[Path]:
    root = Path(root).expanduser().resolve()
    search_root = root / "archive" if (root / "archive").is_dir() else root
    selected: dict[int, Path] = {}
    for path in sorted(search_root.glob("hinet_raw_*.h5")):
        if path.name.endswith(".partial.h5"):
            continue
        try:
            year = int(path.name.split("_")[-1].split(".")[0])
        except ValueError:
            continue
        selected[year] = path
    if allow_partial:
        for path in sorted(search_root.glob("hinet_raw_*.partial.h5")):
            try:
                year = int(path.name.split("_")[-1].split(".")[0])
            except ValueError:
                continue
            if year not in selected or not selected[year].exists():
                selected[year] = path
            else:
                # A non-empty final archive is authoritative.  A partial file
                # is used only when the final path is absent/empty.
                if selected[year].stat().st_size == 0:
                    selected[year] = path
    return [selected[year] for year in sorted(selected)]


def counts_to_velocity_mps(
    counts: np.ndarray,
    counts_per_physical_unit: float,
    unit: str,
) -> np.ndarray:
    unit_norm = str(unit).strip().lower().replace(" ", "")
    if unit_norm not in {"m/s", "m/sec", "mps"}:
        raise ValueError(f"Expected CH velocity unit m/s, got {unit!r}")
    sensitivity = float(counts_per_physical_unit)
    if not np.isfinite(sensitivity) or sensitivity <= 0:
        raise ValueError(f"Invalid counts_per_physical_unit: {sensitivity!r}")
    return np.asarray(counts, dtype=np.float64) / sensitivity


def assemble_station_window(
    reader: AnnualHinetArchiveReader,
    event_id: str,
    station: str,
    start_timestamp: float,
    *,
    n_samples: int = 10000,
    sampling_rate: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Decode and calibrate one E/N/U station on a fixed absolute grid."""
    return assemble_event_windows(
        reader,
        event_id,
        [station],
        start_timestamp,
        n_samples=n_samples,
        sampling_rate=sampling_rate,
    )[str(station)]


def assemble_event_windows(
    reader: AnnualHinetArchiveReader,
    event_id: str,
    stations: Iterable[str],
    start_timestamp: float,
    *,
    n_samples: int = 10000,
    sampling_rate: float = 100.0,
) -> dict[str, tuple[np.ndarray, np.ndarray, dict[str, object]]]:
    """Decode an event's requested channels once, then assemble station views."""
    table = reader.channel_table(event_id)
    requested = list(dict.fromkeys(str(station) for station in stations))
    station_rows: dict[str, dict[str, object]] = {}
    channel_ids: set[str] = set()
    for station in requested:
        rows = table[
            table["hinet_station"].astype(str).str.upper() == station.upper()
        ]
        by_component = {
            str(row["component"]).upper(): row for _, row in rows.iterrows()
        }
        missing = [component for component in COMPONENT_ORDER if component not in by_component]
        if missing:
            raise ValueError(f"{event_id}/{station} lacks components {missing}")
        station_rows[station] = by_component
        channel_ids.update(
            str(by_component[component]["channel_id"]).lower()
            for component in COMPONENT_ORDER
        )
    decoded = reader.read_series(event_id, channel_ids)
    result = {}
    for station in requested:
        by_component = station_rows[station]
        wave = np.zeros((n_samples, 3), dtype=np.float32)
        valid = np.zeros(n_samples, dtype=bool)
        channel_hashes = []
        for channel_index, component in enumerate(COMPONENT_ORDER):
            row = by_component[component]
            channel_id = str(row["channel_id"]).lower()
            if channel_id not in decoded:
                raise ValueError(f"{event_id}/{station} component {component} was not decoded")
            times, counts = decoded[channel_id]
            values = counts_to_velocity_mps(
                counts,
                row["counts_per_physical_unit"],
                row["unit"],
            )
            positions = np.rint((np.asarray(times) - float(start_timestamp)) * sampling_rate).astype(np.int64)
            inside = (positions >= 0) & (positions < n_samples)
            positions = positions[inside]
            values = values[inside]
            if positions.size and np.unique(positions).size != positions.size:
                raise ValueError(f"{event_id}/{station}/{component} has overlapping samples")
            wave[positions, channel_index] = values.astype(np.float32)
            component_valid = np.zeros(n_samples, dtype=bool)
            component_valid[positions] = True
            valid = component_valid if channel_index == 0 else (valid & component_valid)
            channel_hashes.append(str(row.get("raw_line", "")))
        wave[~valid] = 0.0
        provenance = {
            "backend_version": BACKEND_VERSION,
            "quantity": "velocity_sensor_output",
            "units": "m/s",
            "response_correction": "sensitivity_only",
            "component_order": list(COMPONENT_ORDER),
            "channel_metadata_sha256": hashlib.sha256(
                "\n".join(channel_hashes).encode("utf-8")
            ).hexdigest(),
        }
        result[station] = wave, valid, provenance
    return result


class WorkerArchivePool:
    def __init__(self, archive_paths: Iterable[str | Path], max_open_archives: int = 4):
        self.paths = {int(Path(path).name.split("_")[-1].split(".")[0]): Path(path) for path in archive_paths}
        self.max_open_archives = max(1, int(max_open_archives))
        self._pid = os.getpid()
        self._readers: OrderedDict[int, AnnualHinetArchiveReader] = OrderedDict()

    def reader(self, year: int) -> AnnualHinetArchiveReader:
        if os.getpid() != self._pid:
            self.close()
            self._pid = os.getpid()
        year = int(year)
        reader = self._readers.pop(year, None)
        if reader is None:
            reader = AnnualHinetArchiveReader(self.paths[year])
        self._readers[year] = reader
        while len(self._readers) > self.max_open_archives:
            _, old = self._readers.popitem(last=False)
            old.close()
        return reader

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_readers"] = OrderedDict()
        state["_pid"] = os.getpid()
        return state


__all__ = [
    "BACKEND_VERSION",
    "COMPONENT_ORDER",
    "WorkerArchivePool",
    "assemble_event_windows",
    "assemble_station_window",
    "counts_to_velocity_mps",
    "discover_archives",
    "sha256_file",
]
