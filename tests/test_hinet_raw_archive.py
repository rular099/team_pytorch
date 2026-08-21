from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from tools.hinet_raw_archive import (
    AnnualHinetArchiveReader,
    AnnualHinetArchiveWriter,
    ArchiveFormatError,
    HinetArchiveCollection,
    HinetArchiveEventDataset,
    partial_archive_path,
    read_hinet_vm_cnt_bytes,
    scan_hinet_vm_cnt_bytes,
)


def make_cnt_record(channel_ids: tuple[int, ...] = (0x1234,)) -> bytes:
    # 2024-01-02 03:04:05 JST, 4 Hz channels with int8 differences.
    timestamp = bytes((0x20, 0x24, 0x01, 0x02, 0x03, 0x04, 0x05, 0x00))
    samples = struct.pack(">i", 10) + bytes((1, 0xFE, 3))
    payload = b"".join(
        int(channel_id).to_bytes(2, "big") + bytes((0x10, 0x04)) + samples
        for channel_id in channel_ids
    )
    return timestamp + bytes(4) + struct.pack(">i", len(payload)) + payload


def event(event_id: str) -> dict[str, object]:
    return {
        "event_id": event_id,
        "origin_time_jst": "2024-01-02T03:03:55+09:00",
        "origin_time_jst_raw": "2024-01-02T03:04:00+09:00",
        "origin_timestamp": 1704132235.0,
        "origin_time_correction_s": -5.0,
        "origin_time_correction_status": "matched",
        "origin_time_correction_source": "jma_daily",
        "origin_time_jma_event_id": "test-jma-id",
        "latitude": 35.0,
        "longitude": 139.0,
        "depth_km": 10.0,
        "magnitude": 5.0,
        "origin_source": "metadata",
    }


def arrivals(event_id: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "event_id": event_id,
        "hinet_station": "N.TESTH",
        "cut_start_timestamp": 1704132230.0,
        "cut_end_timestamp": 1704132290.0,
    }])


class AnnualHinetArchiveTest(unittest.TestCase):
    def test_roundtrip_dedup_decode_and_collection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "hinet_raw_2024.h5"
            cnt = make_cnt_record()
            channel_table = b"1234 1 0 N.TESTH U 1 1 1.0 m/s 1.0 0.7 0 1e-7 35 139 0\n"
            ids = ("20240102030355", "20240102030455")
            with AnnualHinetArchiveWriter(
                archive,
                year=2024,
                source_hdf5=Path(tmp) / "japan_2024.hdf5",
                provenance={"test": True},
                chunk_bytes=65536,
            ) as writer:
                for event_id in ids:
                    writer.commit_event_bytes(
                        event(event_id),
                        arrivals(event_id),
                        [("2024010203040101VM.cnt", cnt)],
                        ("01_01_20240102.euc.ch", channel_table),
                        raw_status="downloaded_unmerged",
                    )
                    writer.verify_event(event_id)
                # Identical bytes are referenced twice but physically stored once.
                self.assertEqual(writer.h5["raw/cnt_bytes"].shape[0], len(cnt))
                self.assertEqual(writer.h5["raw/channel_table_bytes"].shape[0], len(channel_table))
                writer.mark_complete(ids)

            with AnnualHinetArchiveReader(archive) as reader:
                self.assertTrue(reader.complete)
                self.assertEqual(set(reader.event_ids()), set(ids))
                self.assertEqual(reader.cnt_items(ids[0], verify=True)[0][1], cnt)
                self.assertEqual(reader.channel_table_item(ids[0], verify=True)[1], channel_table)
                self.assertEqual(reader.manifest(ids[0]).iloc[0]["hinet_station"], "N.TESTH")
                record = reader.event_record(ids[0])
                self.assertEqual(record["origin_time_correction_status"], "matched")
                channel_df = reader.channel_table(ids[0])
                self.assertEqual(channel_df.iloc[0]["component"], "U")
                self.assertGreater(channel_df.iloc[0]["counts_per_physical_unit"], 0)
                series = reader.read_series(ids[0], {"1234"})
                np.testing.assert_array_equal(series["1234"][1], np.asarray([10, 11, 9, 12]))
                station_series = reader.read_station_series(ids[0], "N.TESTH", ("U",))
                np.testing.assert_array_equal(station_series["U"][1], np.asarray([10, 11, 9, 12]))

            collection = HinetArchiveCollection([archive])
            loaded = collection.load_event(ids[1], {"1234"})
            np.testing.assert_array_equal(loaded["series"]["1234"][1], np.asarray([10, 11, 9, 12]))
            dataset = HinetArchiveEventDataset([archive], event_ids=[ids[1]], components=("U",))
            sample = dataset[0]
            np.testing.assert_array_equal(sample["series"]["1234"][1], np.asarray([10, 11, 9, 12]))
            dataset.close()

    def test_recover_uncommitted_blob_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "hinet_raw_2024.partial.h5"
            cnt = make_cnt_record()
            with AnnualHinetArchiveWriter(
                archive,
                year=2024,
                source_hdf5=Path(tmp) / "japan_2024.hdf5",
                chunk_bytes=65536,
            ) as writer:
                writer.commit_event_bytes(
                    event("20240102030355"),
                    arrivals("20240102030355"),
                    [("segment.cnt", cnt)],
                    ("channels.ch", b"channels"),
                    raw_status="downloaded",
                )

            with h5py.File(archive, "a") as h5:
                pool = h5["raw/cnt_bytes"]
                committed_size = int(pool.shape[0])
                pool.resize((committed_size + 100,))
                pool[committed_size:] = 255

            with AnnualHinetArchiveWriter(
                archive,
                year=2024,
                source_hdf5=Path(tmp) / "japan_2024.hdf5",
                chunk_bytes=65536,
            ) as writer:
                self.assertEqual(writer.h5["raw/cnt_bytes"].shape[0], len(cnt))
                writer.verify_event("20240102030355")

    def test_resume_rejects_changed_archive_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "hinet_raw_2024.partial.h5"
            source = Path(tmp) / "japan_2024.hdf5"
            with AnnualHinetArchiveWriter(
                archive,
                year=2024,
                source_hdf5=source,
                provenance={"archive_identity": {"pre_seconds": 120}},
                chunk_bytes=65536,
            ):
                pass
            with self.assertRaises(ArchiveFormatError):
                AnnualHinetArchiveWriter(
                    archive,
                    year=2024,
                    source_hdf5=source,
                    provenance={"archive_identity": {"pre_seconds": 20}},
                    chunk_bytes=65536,
                )

    def test_byte_decoder_and_partial_name(self) -> None:
        cnt = make_cnt_record()
        decoded = read_hinet_vm_cnt_bytes([cnt], {"1234", "ffff"})
        self.assertEqual(set(decoded), {"1234"})
        np.testing.assert_array_equal(decoded["1234"][1], np.asarray([10, 11, 9, 12]))
        coverage = scan_hinet_vm_cnt_bytes([cnt])
        self.assertEqual(coverage.record_count, 1)
        self.assertAlmostEqual(coverage.end_timestamp - coverage.start_timestamp, 1.0)
        self.assertEqual(
            partial_archive_path(Path("hinet_raw_2024.h5")).name,
            "hinet_raw_2024.partial.h5",
        )


if __name__ == "__main__":
    unittest.main()
