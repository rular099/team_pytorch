#!/usr/bin/env python3
"""Annual, byte-exact Hi-net WIN32 archive support.

The archive stores the original ``*.cnt`` and channel-table bytes once in
append-only byte pools.  Small fixed-width index rows map events and native
segments to offsets in those pools.  No decoded waveform array is persisted.

The writer uses an event row as the commit marker.  On reopen, any uncommitted
tail left by an interrupted append is truncated before new events are added.
Readers must open their own HDF5 handle in each process (for example, lazily in
each PyTorch DataLoader worker).
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import struct
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd


FORMAT_NAME = "hinet-annual-raw-count-archive"
FORMAT_VERSION = 1
JST = timezone(timedelta(hours=9))
UTC = timezone.utc


EVENT_DTYPE = np.dtype([
    ("event_id", "S32"),
    ("origin_time_jst", "S48"),
    ("origin_time_jst_raw", "S48"),
    ("origin_timestamp", "<f8"),
    ("origin_time_correction_s", "<f8"),
    ("origin_time_correction_status", "S64"),
    ("origin_time_correction_source", "S128"),
    ("origin_time_jma_event_id", "S64"),
    ("latitude", "<f8"),
    ("longitude", "<f8"),
    ("depth_km", "<f8"),
    ("magnitude", "<f8"),
    ("origin_source", "S256"),
    ("raw_status", "S48"),
    ("raw_error", "S2048"),
    ("station_count", "<i4"),
    ("segment_start", "<u8"),
    ("segment_count", "<u4"),
    ("channel_offset", "<u8"),
    ("channel_length", "<u8"),
    ("channel_sha256", "S64"),
    ("channel_filename", "S256"),
    ("manifest_offset", "<u8"),
    ("manifest_length", "<u8"),
    ("request_start_timestamp", "<f8"),
    ("request_end_timestamp", "<f8"),
    ("committed_at_utc", "S40"),
])


SEGMENT_DTYPE = np.dtype([
    ("event_id", "S32"),
    ("ordinal", "<u4"),
    ("original_filename", "S256"),
    ("byte_offset", "<u8"),
    ("byte_length", "<u8"),
    ("sha256", "S64"),
])


ATTEMPT_DTYPE = np.dtype([
    ("event_id", "S32"),
    ("status", "S64"),
    ("station_count", "<i4"),
    ("error", "S2048"),
    ("attempted_at_utc", "S40"),
])


def _now_utc() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _fixed_bytes(value: object, width: int) -> bytes:
    text = "" if value is None else str(value)
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= width:
        return raw
    return raw[:width].decode("utf-8", errors="ignore").encode("utf-8")


def _decode_fixed(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    return str(value)


def _event_value(event: object, name: str, default=None):
    if isinstance(event, Mapping):
        return event.get(name, default)
    return getattr(event, name, default)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def partial_archive_path(final_path: Path) -> Path:
    final_path = Path(final_path)
    return final_path.with_name(f"{final_path.stem}.partial{final_path.suffix}")


def _append_structured(dataset: h5py.Dataset, row: np.void | np.ndarray) -> int:
    index = int(dataset.shape[0])
    dataset.resize((index + 1,))
    dataset[index] = row
    return index


def _append_bytes(dataset: h5py.Dataset, data: bytes) -> tuple[int, int]:
    offset = int(dataset.shape[0])
    length = len(data)
    if length:
        dataset.resize((offset + length,))
        dataset[offset:offset + length] = np.frombuffer(data, dtype=np.uint8)
    return offset, length


def _slice_bytes(dataset: h5py.Dataset, offset: int, length: int) -> bytes:
    if length <= 0:
        return b""
    return np.asarray(dataset[offset:offset + length], dtype=np.uint8).tobytes()


class ArchiveFormatError(RuntimeError):
    """Raised when an archive is incompatible or internally inconsistent."""


class AnnualHinetArchiveWriter:
    """Single-writer append interface for one annual Hi-net raw archive."""

    def __init__(
        self,
        path: Path,
        *,
        year: int,
        source_hdf5: Path,
        provenance: Mapping[str, object] | None = None,
        chunk_bytes: int = 1024 * 1024,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.year = int(year)
        self.source_hdf5 = Path(source_hdf5).expanduser().resolve()
        self._closed = False
        self._lock_fp = None
        self._acquire_lock()
        try:
            existed = self.path.exists()
            self.h5 = h5py.File(self.path, "a", libver="latest")
            if not existed or "raw" not in self.h5:
                self._initialize(chunk_bytes=max(64 * 1024, int(chunk_bytes)), provenance=provenance or {})
            else:
                self._validate()
                self._validate_identity(provenance or {})
                self._merge_provenance(provenance or {})
            self.recover_uncommitted_tail()
            self._rebuild_indexes()
        except Exception:
            if hasattr(self, "h5"):
                try:
                    self.h5.close()
                except Exception:
                    pass
            self._release_lock()
            raise

    def _acquire_lock(self) -> None:
        import fcntl

        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._lock_fp = lock_path.open("a+")
        try:
            fcntl.flock(self._lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_fp.close()
            self._lock_fp = None
            raise RuntimeError(f"Archive is already being written: {self.path}") from exc

    def _release_lock(self) -> None:
        if self._lock_fp is None:
            return
        import fcntl

        try:
            fcntl.flock(self._lock_fp.fileno(), fcntl.LOCK_UN)
        finally:
            self._lock_fp.close()
            self._lock_fp = None

    def _initialize(self, *, chunk_bytes: int, provenance: Mapping[str, object]) -> None:
        self.h5.attrs["format_name"] = FORMAT_NAME
        self.h5.attrs["format_version"] = FORMAT_VERSION
        self.h5.attrs["year"] = self.year
        self.h5.attrs["source_hdf5"] = str(self.source_hdf5)
        self.h5.attrs["created_at_utc"] = _now_utc()
        self.h5.attrs["updated_at_utc"] = _now_utc()
        self.h5.attrs["complete"] = 0
        self.h5.attrs["expected_event_count"] = -1
        identity = provenance.get("archive_identity", {})
        self.h5.attrs["archive_identity_json"] = json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        raw = self.h5.require_group("raw")
        raw.create_dataset(
            "cnt_bytes",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint8,
            chunks=(chunk_bytes,),
            fletcher32=True,
        )
        raw.create_dataset(
            "channel_table_bytes",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint8,
            chunks=(max(64 * 1024, min(chunk_bytes, 1024 * 1024)),),
            fletcher32=True,
        )
        raw.create_dataset(
            "manifest_bytes",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint8,
            chunks=(max(64 * 1024, min(chunk_bytes, 1024 * 1024)),),
            fletcher32=True,
        )

        index = self.h5.require_group("index")
        index.create_dataset("events", shape=(0,), maxshape=(None,), dtype=EVENT_DTYPE, chunks=(256,))
        index.create_dataset("segments", shape=(0,), maxshape=(None,), dtype=SEGMENT_DTYPE, chunks=(1024,))
        index.create_dataset("attempts", shape=(0,), maxshape=(None,), dtype=ATTEMPT_DTYPE, chunks=(1024,))

        metadata = self.h5.require_group("metadata")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        metadata.create_dataset("provenance_json", shape=(), dtype=string_dtype)
        metadata["provenance_json"][()] = json.dumps(dict(provenance), ensure_ascii=False, sort_keys=True)
        self.h5.flush()

    def _validate(self) -> None:
        if str(self.h5.attrs.get("format_name", "")) != FORMAT_NAME:
            raise ArchiveFormatError(f"Not a {FORMAT_NAME} file: {self.path}")
        if int(self.h5.attrs.get("format_version", -1)) != FORMAT_VERSION:
            raise ArchiveFormatError(
                f"Unsupported archive version {self.h5.attrs.get('format_version')} in {self.path}"
            )
        if int(self.h5.attrs.get("year", -1)) != self.year:
            raise ArchiveFormatError(
                f"Archive year={self.h5.attrs.get('year')} does not match requested year={self.year}"
            )

    def _merge_provenance(self, update: Mapping[str, object]) -> None:
        dataset = self.h5["metadata/provenance_json"]
        current_raw = dataset[()]
        if isinstance(current_raw, bytes):
            current_raw = current_raw.decode("utf-8", errors="replace")
        try:
            current = json.loads(str(current_raw)) if current_raw else {}
        except json.JSONDecodeError:
            current = {"previous_raw": str(current_raw)}
        current.update(dict(update))
        dataset[()] = json.dumps(current, ensure_ascii=False, sort_keys=True)
        self.h5.attrs["updated_at_utc"] = _now_utc()

    def _validate_identity(self, provenance: Mapping[str, object]) -> None:
        incoming = provenance.get("archive_identity")
        if incoming is None:
            return
        incoming_json = json.dumps(
            incoming,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        stored = str(self.h5.attrs.get("archive_identity_json", ""))
        if not stored and int(self.events.shape[0]) == 0:
            self.h5.attrs["archive_identity_json"] = incoming_json
            return
        if stored != incoming_json:
            raise ArchiveFormatError(
                "Archive identity does not match this run. Source metadata, station matches, "
                "event selection, or waveform-window parameters changed; use a new archive path."
            )

    @property
    def events(self) -> h5py.Dataset:
        return self.h5["index/events"]

    @property
    def segments(self) -> h5py.Dataset:
        return self.h5["index/segments"]

    @property
    def attempts(self) -> h5py.Dataset:
        return self.h5["index/attempts"]

    def recover_uncommitted_tail(self) -> None:
        """Drop append tails not referenced by a committed event row."""
        events = self.events[()]
        segment_rows = 0
        channel_end = 0
        manifest_end = 0
        for row in events:
            segment_rows = max(segment_rows, int(row["segment_start"]) + int(row["segment_count"]))
            channel_end = max(channel_end, int(row["channel_offset"]) + int(row["channel_length"]))
            manifest_end = max(manifest_end, int(row["manifest_offset"]) + int(row["manifest_length"]))

        if int(self.segments.shape[0]) < segment_rows:
            raise ArchiveFormatError("Committed event references missing segment-index rows")
        if int(self.segments.shape[0]) != segment_rows:
            self.segments.resize((segment_rows,))

        cnt_end = 0
        if segment_rows:
            kept = self.segments[:segment_rows]
            cnt_end = max(int(row["byte_offset"]) + int(row["byte_length"]) for row in kept)

        pools = (
            (self.h5["raw/cnt_bytes"], cnt_end),
            (self.h5["raw/channel_table_bytes"], channel_end),
            (self.h5["raw/manifest_bytes"], manifest_end),
        )
        for dataset, referenced_end in pools:
            if int(dataset.shape[0]) < referenced_end:
                raise ArchiveFormatError(f"Committed index references missing bytes in {dataset.name}")
            if int(dataset.shape[0]) != referenced_end:
                dataset.resize((referenced_end,))
        self.h5.flush()

    def recover(self) -> None:
        """Recover an interrupted append and rebuild in-memory lookup tables."""
        self.recover_uncommitted_tail()
        self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        self._event_ids: set[str] = {
            _decode_fixed(row["event_id"]) for row in self.events[()]
        }
        self._cnt_hash_index: dict[str, tuple[int, int]] = {}
        for row in self.segments[()]:
            digest = _decode_fixed(row["sha256"])
            self._cnt_hash_index.setdefault(
                digest,
                (int(row["byte_offset"]), int(row["byte_length"])),
            )
        self._channel_hash_index: dict[str, tuple[int, int]] = {}
        for row in self.events[()]:
            digest = _decode_fixed(row["channel_sha256"])
            length = int(row["channel_length"])
            if digest and length:
                self._channel_hash_index.setdefault(
                    digest,
                    (int(row["channel_offset"]), length),
                )

    def committed_event_ids(self) -> set[str]:
        return set(self._event_ids)

    def has_event(self, event_id: str) -> bool:
        return str(event_id) in self._event_ids

    def verify_event(self, event_id: str) -> None:
        """Read committed bytes back through HDF5 and verify stored hashes."""
        target = str(event_id)
        event_rows = [row for row in self.events[()] if _decode_fixed(row["event_id"]) == target]
        if len(event_rows) != 1:
            raise ArchiveFormatError(f"Expected one committed row for {target}, found {len(event_rows)}")
        event = event_rows[0]
        start = int(event["segment_start"])
        count = int(event["segment_count"])
        cnt_pool = self.h5["raw/cnt_bytes"]
        for row in self.segments[start:start + count]:
            data = _slice_bytes(cnt_pool, int(row["byte_offset"]), int(row["byte_length"]))
            if sha256_bytes(data) != _decode_fixed(row["sha256"]):
                raise ArchiveFormatError(
                    f"CNT checksum mismatch after commit: {target}/{_decode_fixed(row['original_filename'])}"
                )
        channel_length = int(event["channel_length"])
        if channel_length:
            channel_data = _slice_bytes(
                self.h5["raw/channel_table_bytes"],
                int(event["channel_offset"]),
                channel_length,
            )
            if sha256_bytes(channel_data) != _decode_fixed(event["channel_sha256"]):
                raise ArchiveFormatError(f"Channel-table checksum mismatch after commit: {target}")

    def _put_deduplicated(
        self,
        pool: h5py.Dataset,
        data: bytes,
        digest_index: dict[str, tuple[int, int]],
    ) -> tuple[int, int, str]:
        digest = sha256_bytes(data)
        existing = digest_index.get(digest)
        if existing is not None:
            return existing[0], existing[1], digest
        offset, length = _append_bytes(pool, data)
        digest_index[digest] = (offset, length)
        return offset, length, digest

    def record_attempt(
        self,
        event_id: str,
        status: str,
        *,
        station_count: int = 0,
        error: str = "",
    ) -> None:
        row = np.zeros((), dtype=ATTEMPT_DTYPE)
        row["event_id"] = _fixed_bytes(event_id, 32)
        row["status"] = _fixed_bytes(status, 64)
        row["station_count"] = int(station_count)
        row["error"] = _fixed_bytes(error, 2048)
        row["attempted_at_utc"] = _fixed_bytes(_now_utc(), 40)
        _append_structured(self.attempts, row)
        self.h5.attrs["updated_at_utc"] = _now_utc()
        self.h5.flush()

    def commit_event_files(
        self,
        event: object,
        arrivals: pd.DataFrame,
        cnt_paths: Sequence[Path],
        channel_table_path: Path | None,
        *,
        raw_status: str,
        raw_error: str = "",
    ) -> None:
        cnt_items = [(Path(path).name, Path(path).read_bytes()) for path in cnt_paths]
        if channel_table_path is None:
            channel_item = ("", b"")
        else:
            channel_path = Path(channel_table_path)
            channel_item = (channel_path.name, channel_path.read_bytes())
        self.commit_event_bytes(
            event,
            arrivals,
            cnt_items,
            channel_item,
            raw_status=raw_status,
            raw_error=raw_error,
        )

    def commit_event_bytes(
        self,
        event: object,
        arrivals: pd.DataFrame,
        cnt_items: Sequence[tuple[str, bytes]],
        channel_item: tuple[str, bytes],
        *,
        raw_status: str,
        raw_error: str = "",
    ) -> None:
        event_id = str(_event_value(event, "event_id", ""))
        if not event_id:
            raise ValueError("event_id is required")
        if self.has_event(event_id):
            raise ValueError(f"Event is already committed: {event_id}")

        segment_start = int(self.segments.shape[0])
        cnt_pool = self.h5["raw/cnt_bytes"]
        for ordinal, (filename, data) in enumerate(cnt_items):
            offset, length, digest = self._put_deduplicated(
                cnt_pool,
                bytes(data),
                self._cnt_hash_index,
            )
            row = np.zeros((), dtype=SEGMENT_DTYPE)
            row["event_id"] = _fixed_bytes(event_id, 32)
            row["ordinal"] = ordinal
            row["original_filename"] = _fixed_bytes(filename, 256)
            row["byte_offset"] = offset
            row["byte_length"] = length
            row["sha256"] = _fixed_bytes(digest, 64)
            _append_structured(self.segments, row)

        channel_filename, channel_bytes = channel_item
        if channel_bytes:
            channel_offset, channel_length, channel_digest = self._put_deduplicated(
                self.h5["raw/channel_table_bytes"],
                bytes(channel_bytes),
                self._channel_hash_index,
            )
        else:
            channel_offset = int(self.h5["raw/channel_table_bytes"].shape[0])
            channel_length = 0
            channel_digest = ""

        manifest_bytes = arrivals.to_csv(index=False).encode("utf-8")
        manifest_offset, manifest_length = _append_bytes(
            self.h5["raw/manifest_bytes"],
            manifest_bytes,
        )

        if arrivals.empty:
            request_start = math.nan
            request_end = math.nan
        else:
            request_start = float(pd.to_numeric(arrivals["cut_start_timestamp"], errors="coerce").min())
            request_end = float(pd.to_numeric(arrivals["cut_end_timestamp"], errors="coerce").max())

        magnitude = _event_value(event, "magnitude", math.nan)
        if magnitude is None:
            magnitude = math.nan
        row = np.zeros((), dtype=EVENT_DTYPE)
        row["event_id"] = _fixed_bytes(event_id, 32)
        row["origin_time_jst"] = _fixed_bytes(_event_value(event, "origin_time_jst", ""), 48)
        row["origin_time_jst_raw"] = _fixed_bytes(
            _event_value(event, "origin_time_jst_raw", ""),
            48,
        )
        row["origin_timestamp"] = float(_event_value(event, "origin_timestamp", math.nan))
        correction_s = _event_value(event, "origin_time_correction_s", math.nan)
        try:
            correction_s = float(correction_s)
        except (TypeError, ValueError):
            correction_s = math.nan
        row["origin_time_correction_s"] = correction_s
        row["origin_time_correction_status"] = _fixed_bytes(
            _event_value(event, "origin_time_correction_status", ""),
            64,
        )
        row["origin_time_correction_source"] = _fixed_bytes(
            _event_value(event, "origin_time_correction_source", ""),
            128,
        )
        row["origin_time_jma_event_id"] = _fixed_bytes(
            _event_value(event, "origin_time_jma_event_id", ""),
            64,
        )
        row["latitude"] = float(_event_value(event, "latitude", math.nan))
        row["longitude"] = float(_event_value(event, "longitude", math.nan))
        row["depth_km"] = float(_event_value(event, "depth_km", math.nan))
        row["magnitude"] = float(magnitude)
        row["origin_source"] = _fixed_bytes(_event_value(event, "origin_source", ""), 256)
        row["raw_status"] = _fixed_bytes(raw_status, 48)
        row["raw_error"] = _fixed_bytes(raw_error, 2048)
        row["station_count"] = int(len(arrivals))
        row["segment_start"] = segment_start
        row["segment_count"] = len(cnt_items)
        row["channel_offset"] = channel_offset
        row["channel_length"] = channel_length
        row["channel_sha256"] = _fixed_bytes(channel_digest, 64)
        row["channel_filename"] = _fixed_bytes(channel_filename, 256)
        row["manifest_offset"] = manifest_offset
        row["manifest_length"] = manifest_length
        row["request_start_timestamp"] = request_start
        row["request_end_timestamp"] = request_end
        row["committed_at_utc"] = _fixed_bytes(_now_utc(), 40)

        # The event row is the commit marker and must be written last.
        self.h5.flush()
        _append_structured(self.events, row)
        self._event_ids.add(event_id)
        self.record_attempt(
            event_id,
            raw_status,
            station_count=len(arrivals),
            error=raw_error,
        )

    def mark_complete(self, expected_event_ids: Iterable[str]) -> None:
        expected = {str(event_id) for event_id in expected_event_ids}
        missing = sorted(expected - self._event_ids)
        if missing:
            preview = ", ".join(missing[:5])
            raise RuntimeError(
                f"Cannot finalize {self.path}: {len(missing)} events are not committed ({preview})"
            )
        self.h5.attrs["expected_event_count"] = len(expected)
        self.h5.attrs["complete"] = 1
        self.h5.attrs["completed_at_utc"] = _now_utc()
        self.h5.attrs["updated_at_utc"] = _now_utc()
        self.h5.flush()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.h5.flush()
            self.h5.close()
        finally:
            self._release_lock()
            self._closed = True

    def __enter__(self) -> "AnnualHinetArchiveWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@dataclass(frozen=True)
class ArchivedSegment:
    event_id: str
    ordinal: int
    original_filename: str
    byte_offset: int
    byte_length: int
    sha256: str


@dataclass(frozen=True)
class CntCoverage:
    start_timestamp: float
    end_timestamp: float
    record_count: int
    blob_count: int


class AnnualHinetArchiveReader:
    """Read-only, process-local access to an annual raw archive."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.h5 = h5py.File(self.path, "r", libver="latest", swmr=True)
        if str(self.h5.attrs.get("format_name", "")) != FORMAT_NAME:
            self.h5.close()
            raise ArchiveFormatError(f"Not a {FORMAT_NAME} file: {self.path}")
        self._event_rows = {
            _decode_fixed(row["event_id"]): row for row in self.h5["index/events"][()]
        }

    @property
    def year(self) -> int:
        return int(self.h5.attrs["year"])

    @property
    def complete(self) -> bool:
        return bool(int(self.h5.attrs.get("complete", 0)))

    @property
    def archive_identity(self) -> dict[str, object]:
        raw = str(self.h5.attrs.get("archive_identity_json", "{}"))
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ArchiveFormatError(f"Invalid archive_identity_json in {self.path}") from exc
        return value if isinstance(value, dict) else {}

    def event_ids(self) -> list[str]:
        return list(self._event_rows)

    def has_event(self, event_id: str) -> bool:
        return str(event_id) in self._event_rows

    def event_record(self, event_id: str) -> dict[str, object]:
        row = self._event_rows[str(event_id)]
        out: dict[str, object] = {}
        for name in EVENT_DTYPE.names or ():
            value = row[name]
            if np.issubdtype(EVENT_DTYPE[name], np.bytes_):
                out[name] = _decode_fixed(value)
            elif isinstance(value, np.generic):
                out[name] = value.item()
            else:
                out[name] = value
        return out

    def event_segments(self, event_id: str) -> list[ArchivedSegment]:
        event = self._event_rows[str(event_id)]
        start = int(event["segment_start"])
        count = int(event["segment_count"])
        rows = self.h5["index/segments"][start:start + count]
        return [
            ArchivedSegment(
                event_id=_decode_fixed(row["event_id"]),
                ordinal=int(row["ordinal"]),
                original_filename=_decode_fixed(row["original_filename"]),
                byte_offset=int(row["byte_offset"]),
                byte_length=int(row["byte_length"]),
                sha256=_decode_fixed(row["sha256"]),
            )
            for row in rows
        ]

    def cnt_items(self, event_id: str, *, verify: bool = False) -> list[tuple[str, bytes]]:
        pool = self.h5["raw/cnt_bytes"]
        out: list[tuple[str, bytes]] = []
        for segment in self.event_segments(event_id):
            data = _slice_bytes(pool, segment.byte_offset, segment.byte_length)
            if verify and sha256_bytes(data) != segment.sha256:
                raise ArchiveFormatError(
                    f"CNT checksum mismatch for {event_id}/{segment.original_filename}"
                )
            out.append((segment.original_filename, data))
        return out

    def channel_table_item(self, event_id: str, *, verify: bool = False) -> tuple[str, bytes]:
        row = self._event_rows[str(event_id)]
        filename = _decode_fixed(row["channel_filename"])
        offset = int(row["channel_offset"])
        length = int(row["channel_length"])
        data = _slice_bytes(self.h5["raw/channel_table_bytes"], offset, length)
        digest = _decode_fixed(row["channel_sha256"])
        if verify and digest and sha256_bytes(data) != digest:
            raise ArchiveFormatError(f"Channel-table checksum mismatch for {event_id}/{filename}")
        return filename, data

    def channel_table(self, event_id: str) -> pd.DataFrame:
        """Return parsed response and station metadata from the archived .ch bytes."""
        return parse_hinet_channel_table_bytes(self.channel_table_item(event_id)[1])

    def manifest(self, event_id: str) -> pd.DataFrame:
        row = self._event_rows[str(event_id)]
        data = _slice_bytes(
            self.h5["raw/manifest_bytes"],
            int(row["manifest_offset"]),
            int(row["manifest_length"]),
        )
        if not data.strip():
            return pd.DataFrame()
        try:
            return pd.read_csv(io.BytesIO(data), dtype={"event_id": str})
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def read_series(
        self,
        event_id: str,
        channel_ids: set[str],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        blobs = [data for _, data in self.cnt_items(event_id)]
        return read_hinet_vm_cnt_bytes(blobs, channel_ids)

    def read_station_series(
        self,
        event_id: str,
        hinet_station: str,
        components: Sequence[str] = ("U", "N", "E"),
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Decode requested components for one station, keyed by component name."""
        table = self.channel_table(event_id)
        if table.empty:
            return {}
        station_rows = table[
            table["hinet_station"].astype(str).str.upper() == str(hinet_station).upper()
        ]
        wanted_components = {str(component).upper() for component in components}
        station_rows = station_rows[
            station_rows["component"].astype(str).str.upper().isin(wanted_components)
        ]
        channel_to_component = {
            str(row["channel_id"]).lower(): str(row["component"]).upper()
            for _, row in station_rows.iterrows()
        }
        decoded = self.read_series(event_id, set(channel_to_component))
        return {
            channel_to_component[channel_id]: series
            for channel_id, series in decoded.items()
            if channel_id in channel_to_component
        }

    def events_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.event_record(event_id) for event_id in self.event_ids()])

    def attempts_dataframe(self) -> pd.DataFrame:
        rows = []
        for row in self.h5["index/attempts"][()]:
            rows.append({
                "event_id": _decode_fixed(row["event_id"]),
                "status": _decode_fixed(row["status"]),
                "station_count": int(row["station_count"]),
                "error": _decode_fixed(row["error"]),
                "attempted_at_utc": _decode_fixed(row["attempted_at_utc"]),
            })
        return pd.DataFrame(rows)

    def close(self) -> None:
        self.h5.close()

    def __enter__(self) -> "AnnualHinetArchiveReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class HinetArchiveCollection:
    """Small index over multiple annual archives without persisting waveforms."""

    def __init__(self, archive_paths: Iterable[Path]) -> None:
        self.archive_paths = [Path(path).expanduser().resolve() for path in archive_paths]
        self._event_to_path: dict[str, Path] = {}
        for path in self.archive_paths:
            with AnnualHinetArchiveReader(path) as reader:
                for event_id in reader.event_ids():
                    if event_id in self._event_to_path:
                        raise ValueError(f"Duplicate event_id {event_id} in annual archives")
                    self._event_to_path[event_id] = path

    def event_ids(self) -> list[str]:
        return list(self._event_to_path)

    def archive_path(self, event_id: str) -> Path:
        return self._event_to_path[str(event_id)]

    def load_event(self, event_id: str, channel_ids: set[str] | None = None) -> dict[str, object]:
        with AnnualHinetArchiveReader(self.archive_path(event_id)) as reader:
            out: dict[str, object] = {
                "event": reader.event_record(event_id),
                "manifest": reader.manifest(event_id),
                "channel_table": reader.channel_table(event_id),
                "channel_table_bytes": reader.channel_table_item(event_id)[1],
            }
            if channel_ids is None:
                out["cnt_items"] = reader.cnt_items(event_id)
            else:
                out["series"] = reader.read_series(event_id, channel_ids)
            return out


class HinetArchiveEventDataset:
    """Map-style, worker-safe event dataset over annual raw archives.

    ``event_ids`` is the frozen train/validation/test split. Each worker opens
    archive handles lazily in its own process and keeps only a bounded LRU of
    annual files open. Returned waveforms are decoded transiently from CNT and
    are never written back to disk.
    """

    def __init__(
        self,
        archive_paths: Iterable[Path],
        *,
        event_ids: Iterable[str] | None = None,
        components: Sequence[str] = ("U", "N", "E", "Z", "1", "2"),
        decode_counts: bool = True,
        max_open_archives: int = 4,
    ) -> None:
        collection = HinetArchiveCollection(archive_paths)
        self._event_to_path = dict(collection._event_to_path)
        if event_ids is None:
            self._event_ids = sorted(self._event_to_path)
        else:
            self._event_ids = [str(event_id) for event_id in event_ids]
            missing = [event_id for event_id in self._event_ids if event_id not in self._event_to_path]
            if missing:
                preview = ", ".join(missing[:5])
                raise KeyError(f"{len(missing)} split events are absent from the archives: {preview}")
        self.components = tuple(str(component).upper() for component in components)
        self.decode_counts = bool(decode_counts)
        self.max_open_archives = max(1, int(max_open_archives))
        self._owner_pid = os.getpid()
        self._readers: OrderedDict[Path, AnnualHinetArchiveReader] = OrderedDict()

    @classmethod
    def from_event_id_file(
        cls,
        archive_paths: Iterable[Path],
        event_id_file: Path,
        **kwargs,
    ) -> "HinetArchiveEventDataset":
        ids = [line.strip() for line in Path(event_id_file).read_text().splitlines() if line.strip()]
        return cls(archive_paths, event_ids=ids, **kwargs)

    def __len__(self) -> int:
        return len(self._event_ids)

    def _ensure_process_local_state(self) -> None:
        current_pid = os.getpid()
        if current_pid == self._owner_pid:
            return
        # Handles inherited through fork must not be reused.
        self.close()
        self._owner_pid = current_pid
        self._readers = OrderedDict()

    def _reader(self, path: Path) -> AnnualHinetArchiveReader:
        self._ensure_process_local_state()
        reader = self._readers.pop(path, None)
        if reader is None:
            reader = AnnualHinetArchiveReader(path)
        self._readers[path] = reader
        while len(self._readers) > self.max_open_archives:
            _, evicted = self._readers.popitem(last=False)
            evicted.close()
        return reader

    def __getitem__(self, index: int) -> dict[str, object]:
        event_id = self._event_ids[int(index)]
        archive_path = self._event_to_path[event_id]
        reader = self._reader(archive_path)
        channel_table = reader.channel_table(event_id)
        sample: dict[str, object] = {
            "event_id": event_id,
            "archive_path": str(archive_path),
            "event": reader.event_record(event_id),
            "manifest": reader.manifest(event_id),
            "channel_table": channel_table,
        }
        if self.decode_counts and not channel_table.empty:
            selected = channel_table[
                channel_table["component"].astype(str).str.upper().isin(self.components)
            ]
            channel_ids = set(selected["channel_id"].astype(str).str.lower())
            sample["series"] = reader.read_series(event_id, channel_ids)
        elif self.decode_counts:
            sample["series"] = {}
        return sample

    def close(self) -> None:
        for reader in self._readers.values():
            try:
                reader.close()
            except Exception:
                pass
        self._readers.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Never pickle or fork live h5py handles.
        state["_readers"] = OrderedDict()
        state["_owner_pid"] = os.getpid()
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def bcd_to_int(value: int) -> int:
    return (value >> 4) * 10 + (value & 0x0F)


def parse_hinet_channel_table_bytes(data: bytes) -> pd.DataFrame:
    """Parse HinetPy channel-table text without materializing a temporary file.

    The named response fields follow ``HinetPy.win32.read_ctable``. Original
    lines and otherwise undocumented positional fields are retained as well.
    """
    rows: list[dict[str, object]] = []
    for raw_line in data.decode("utf-8", errors="replace").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        items = stripped.split()
        if len(items) < 15:
            continue
        try:
            gain = float(items[7])
            period = float(items[9])
            damping = float(items[10])
            preamplification = float(items[11])
            lsb_value = float(items[12])
            sensitivity = (
                gain * math.pow(10.0, preamplification / 20.0) / lsb_value
                if lsb_value != 0.0
                else math.nan
            )
            latitude = float(items[13])
            longitude = float(items[14])
        except (TypeError, ValueError):
            continue
        rows.append({
            "channel_id": items[0].lower(),
            "hinet_station": items[3],
            "component": items[4].upper(),
            "latitude": latitude,
            "longitude": longitude,
            "unit": items[8],
            "gain": gain,
            "period": period,
            "damping": damping,
            "preamplification": preamplification,
            "lsb_value": lsb_value,
            "counts_per_physical_unit": sensitivity,
            "field_1": items[1],
            "field_2": items[2],
            "field_5": items[5],
            "field_6": items[6],
            "field_15": items[15] if len(items) > 15 else "",
            "field_16": items[16] if len(items) > 16 else "",
            "field_17": items[17] if len(items) > 17 else "",
            "description": " ".join(items[18:]) if len(items) > 18 else "",
            "raw_line": raw_line,
        })
    return pd.DataFrame(rows)


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


def read_hinet_vm_cnt_bytes(
    cnt_blobs: Iterable[bytes | bytearray | memoryview],
    channel_ids: set[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Decode selected channels directly from native CNT byte strings."""
    wanted = {str(channel_id).lower() for channel_id in channel_ids}
    values_by_channel: dict[str, list[np.ndarray]] = {channel_id: [] for channel_id in wanted}
    times_by_channel: dict[str, list[np.ndarray]] = {channel_id: [] for channel_id in wanted}
    for blob in cnt_blobs:
        data = bytes(blob)
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
                times_by_channel[channel_id].append(
                    record_ts + np.arange(decoded.size, dtype=np.float64) / float(srate)
                )
            offset = payload_end

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for channel_id in wanted:
        if values_by_channel[channel_id]:
            times = np.concatenate(times_by_channel[channel_id])
            values = np.concatenate(values_by_channel[channel_id])
            order = np.argsort(times)
            out[channel_id] = (times[order], values[order])
    return out


def scan_hinet_vm_cnt_bytes(
    cnt_blobs: Iterable[bytes | bytearray | memoryview],
) -> CntCoverage:
    """Scan one-second WIN32 record headers without decoding channel samples."""
    start = math.inf
    end = -math.inf
    records = 0
    blobs = 0
    for blob in cnt_blobs:
        blobs += 1
        data = bytes(blob)
        offset = 4 if len(data) > 20 and data[:4] == b"\x00\x00\x00\x00" else 0
        while offset + 16 <= len(data):
            try:
                record_ts = parse_hinet_vm_timestamp(data[offset:offset + 8])
            except Exception:
                break
            payload_len = struct.unpack(">i", data[offset + 12:offset + 16])[0]
            payload_end = offset + 16 + payload_len
            if payload_len <= 0 or payload_end > len(data):
                break
            start = min(start, record_ts)
            # A VM record represents one second; sample timestamps end just
            # before this exclusive bound.
            end = max(end, record_ts + 1.0)
            records += 1
            offset = payload_end
    if records == 0:
        start = math.nan
        end = math.nan
    return CntCoverage(start, end, records, blobs)


def read_hinet_vm_cnt_paths(
    cnt_paths: Iterable[Path],
    channel_ids: set[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return read_hinet_vm_cnt_bytes(
        (Path(path).read_bytes() for path in sorted(cnt_paths)),
        channel_ids,
    )


def scan_hinet_vm_cnt_paths(cnt_paths: Iterable[Path]) -> CntCoverage:
    return scan_hinet_vm_cnt_bytes(Path(path).read_bytes() for path in sorted(cnt_paths))


__all__ = [
    "AnnualHinetArchiveReader",
    "AnnualHinetArchiveWriter",
    "ArchiveFormatError",
    "ArchivedSegment",
    "CntCoverage",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "HinetArchiveCollection",
    "HinetArchiveEventDataset",
    "decode_win32_diffs",
    "parse_hinet_vm_timestamp",
    "parse_hinet_channel_table_bytes",
    "partial_archive_path",
    "read_hinet_vm_cnt_bytes",
    "read_hinet_vm_cnt_paths",
    "scan_hinet_vm_cnt_bytes",
    "scan_hinet_vm_cnt_paths",
    "sha256_bytes",
]
