from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from tools.download_hinet_velocity import (
    JST,
    EventInfo,
    HinetDownloadError,
    RawBatchResult,
    RawDownloadResult,
    _download_station_batch,
    _diagnostic_hinet_client_class,
    _merge_channel_tables,
    _validate_raw_request,
    archive_provenance,
    download_raw_event,
    process_events_archive,
)
from tools.hinet_raw_archive import AnnualHinetArchiveReader, scan_hinet_vm_cnt_bytes
from tools.plot_hinet_accel_velocity_qc import load_archive_velocity, load_manifest
from tests.test_hinet_raw_archive import make_cnt_record


def write_three_component_channel_table(path: Path, stations: list[str]) -> None:
    rows = []
    for station_index, station in enumerate(stations):
        for component_index, component in enumerate(("U", "N", "E")):
            channel_id = 0x1234 + station_index * 3 + component_index
            rows.append(
                f"{channel_id:04x} 1 0 {station} {component} "
                "1 1 1.0 m/s 1.0 0.7 0 1e-7 35 139 0"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def example_event(event_id: str = "20240102030355") -> EventInfo:
    return EventInfo(
        event_id=event_id,
        origin_time_jst="2024-01-02T03:03:55+09:00",
        origin_timestamp=1704132235.0,
        latitude=35.0,
        longitude=139.0,
        depth_km=10.0,
        magnitude=5.0,
        origin_source="metadata",
        origin_time_jst_raw="2024/01/02 03:04:00",
        origin_time_correction_s=-5.0,
        origin_time_correction_status="matched",
        origin_time_correction_source="jma_daily",
        origin_time_jma_event_id="test-jma-id",
    )


class DownloadArchiveFlowTest(unittest.TestCase):
    def test_diagnostic_client_propagates_timeout_and_preserves_zip_error(self) -> None:
        class FakeResponse:
            status_code = 200

            def __init__(self):
                self.closed = False

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                self.chunk_size = chunk_size
                yield b"not-a-zip"

            def close(self):
                self.closed = True

        class FakeSession:
            def __init__(self):
                self.responses = []
                self.closed = False
                self.timeouts = []

            def post(self, *_args, **kwargs):
                response = FakeResponse()
                self.responses.append(response)
                self.timeouts.append(kwargs["timeout"])
                return response

            def close(self):
                self.closed = True

        class FakeClient:
            _CONT_DOWNLOAD = "download"
            instances = []

            def __init__(
                self,
                user,
                password,
                timeout=60,
                retries=3,
                sleep_time_in_seconds=5,
                max_sleep_count=30,
            ):
                self.user = user
                self.password = password
                self.timeout = timeout
                self.retries = retries
                self.sleep_time_in_seconds = sleep_time_in_seconds
                self.max_sleep_count = max_sleep_count
                self.session = FakeSession()
                self.instances.append(self)

        diagnostic_class = _diagnostic_hinet_client_class(FakeClient)
        client = diagnostic_class("user", "password", timeout=17, retries=2)
        with self.assertRaises(HinetDownloadError) as caught:
            client._download_cont_waveform(SimpleNamespace(id="job-1"))

        download_client = FakeClient.instances[-1]
        self.assertEqual(download_client.timeout, 17)
        self.assertEqual(download_client.session.timeouts, [17, 17])
        self.assertTrue(download_client.session.closed)
        self.assertTrue(all(response.closed for response in download_client.session.responses))
        self.assertIn("BadZipFile", str(caught.exception))

    def test_channel_table_aggregation_preserves_unique_provider_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.euc.ch"
            duplicate = root / "duplicate.euc.ch"
            second = root / "second.euc.ch"
            first_payload = b"# first\r\n1234 row\r\n"
            second_payload = b"# second\n1235 row\n"
            first.write_bytes(first_payload)
            duplicate.write_bytes(first_payload)
            second.write_bytes(second_payload)
            output = _merge_channel_tables((first, duplicate, second), root / "combined.euc.ch")
            self.assertEqual(output.read_bytes(), first_payload + second_payload)

    def test_raw_validation_requires_all_requested_channel_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            channel_path = root / "channels.euc.ch"
            write_three_component_channel_table(channel_path, ["N.TESTH"])
            complete_path = root / "complete.cnt"
            complete_path.write_bytes(make_cnt_record((0x1234, 0x1235, 0x1236)))
            coverage = scan_hinet_vm_cnt_bytes([complete_path.read_bytes()])
            valid, detail = _validate_raw_request(
                (complete_path,),
                channel_path,
                request_start_timestamp=coverage.start_timestamp,
                request_end_timestamp=coverage.end_timestamp,
                stations=["N.TESTH"],
            )
            self.assertTrue(valid, detail)

            incomplete_path = root / "incomplete.cnt"
            incomplete_path.write_bytes(make_cnt_record())
            valid, detail = _validate_raw_request(
                (incomplete_path,),
                channel_path,
                request_start_timestamp=coverage.start_timestamp,
                request_end_timestamp=coverage.end_timestamp,
                stations=["N.TESTH"],
            )
            self.assertFalse(valid)
            self.assertIn("N.TESTH.N/1235:absent", detail)

    def test_transport_tuning_does_not_change_archive_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_hdf5 = Path(tmp) / "japan_2024.hdf5"
            source_hdf5.write_bytes(b"source")
            args = SimpleNamespace(
                hdf5=source_hdf5,
                year=2024,
                hinet_network="0101",
                match_distance_km=0.5,
                pre_seconds=120.0,
                post_seconds=120.0,
                mode="all",
                match_csv=None,
                event_ids=None,
                origin_corrections=None,
                jma_travel_time_zip=None,
                hinet_timeout_seconds=300.0,
                hinet_retries=3,
                station_batch_size=40,
                hinet_download_threads=1,
                minute_fallback=True,
                fallback_span_minutes=1,
                subrequest_sleep_seconds=0.0,
            )
            first = archive_provenance(args)
            args.hinet_timeout_seconds = 900.0
            args.station_batch_size = 10
            args.fallback_span_minutes = 2
            second = archive_provenance(args)

        self.assertEqual(first["archive_identity"], second["archive_identity"])
        self.assertNotEqual(first["download_transport"], second["download_transport"])

    def test_station_batches_preserve_provider_filenames_in_separate_directories(self) -> None:
        stations = [f"N.T{i:03d}H" for i in range(5)]
        arrivals = pd.DataFrame(
            {
                "hinet_station": stations,
                "cut_start_timestamp": [1704132245.0] * len(stations),
                "cut_end_timestamp": [1704132295.0] * len(stations),
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                output_root=root,
                raw_work_root=root / "raw",
                overwrite_raw=False,
                dry_run=False,
                station_batch_size=2,
                hinet_network="0101",
            )
            source_root = root / "synthetic_downloads"
            calls: list[list[str]] = []

            def fake_batch(*_args, **kwargs):
                batch_index = kwargs["batch_index"]
                batch_stations = list(kwargs["stations"])
                calls.append(batch_stations)
                cnt_path = source_root / f"batch_{batch_index}" / "native.cnt"
                cnt_path.parent.mkdir(parents=True, exist_ok=True)
                cnt_path.write_bytes(f"batch-{batch_index}".encode())
                channel_path = source_root / f"batch_{batch_index}" / "channels.euc.ch"
                write_three_component_channel_table(channel_path, batch_stations)
                return RawBatchResult((cnt_path,), channel_path, True, False, "primary request valid")

            with patch("tools.download_hinet_velocity._download_station_batch", side_effect=fake_batch):
                raw = download_raw_event(args, object(), example_event(), arrivals)

            self.assertEqual(calls, [stations[:2], stations[2:4], stations[4:]])
            self.assertEqual(raw.raw_status, "downloaded_batched")
            self.assertEqual(raw.download_strategy, "station_batches")
            self.assertEqual(raw.batch_count, 3)
            self.assertEqual(len(raw.segment_paths), 3)
            self.assertEqual({path.name for path in raw.segment_paths}, {"native.cnt"})
            self.assertEqual(len({path.parent for path in raw.segment_paths}), 3)
            self.assertTrue(all(path.is_file() for path in raw.segment_paths))
            self.assertIsNotNone(raw.ch_path)

    def test_failed_station_batch_returns_no_committable_segments(self) -> None:
        stations = ["N.AAAH", "N.BBBH", "N.CCCH"]
        arrivals = pd.DataFrame(
            {
                "hinet_station": stations,
                "cut_start_timestamp": [1704132245.0] * 3,
                "cut_end_timestamp": [1704132295.0] * 3,
            }
        )
        success = RawBatchResult((Path("first.cnt"),), Path("first.ch"), True, False, "ok")
        failure = RawBatchResult(tuple(), Path("second.ch"), False, True, "no CNT files returned")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                output_root=root,
                raw_work_root=root / "raw",
                overwrite_raw=False,
                dry_run=False,
                station_batch_size=2,
                hinet_network="0101",
            )
            with patch(
                "tools.download_hinet_velocity._download_station_batch",
                side_effect=(success, failure),
            ):
                raw = download_raw_event(args, object(), example_event(), arrivals)

        self.assertEqual(raw.raw_status, "download_failed")
        self.assertEqual(raw.segment_paths, tuple())
        self.assertIn("batch 2/2", raw.raw_error)

    def test_failed_full_window_uses_consecutive_minute_fallback(self) -> None:
        stations = ["N.TESTH"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            minute_results = []
            for minute in range(3):
                cnt_path = root / "provider" / f"minute_{minute}.cnt"
                cnt_path.parent.mkdir(parents=True, exist_ok=True)
                cnt_path.write_bytes(bytes([minute]))
                channel_path = root / "provider" / f"minute_{minute}.euc.ch"
                write_three_component_channel_table(channel_path, stations)
                minute_results.append(
                    RawBatchResult((cnt_path,), channel_path, True, False, "primary request valid")
                )
            primary_failure = RawBatchResult(
                tuple(),
                None,
                False,
                False,
                "transport timeout",
            )
            args = SimpleNamespace(
                hinet_network="0101",
                minute_fallback=True,
                fallback_span_minutes=1,
                subrequest_sleep_seconds=0.0,
            )
            client = SimpleNamespace(select_stations=lambda *_args, **_kwargs: None)
            with patch(
                "tools.download_hinet_velocity._request_raw_window",
                side_effect=(primary_failure, *minute_results),
            ) as request_mock:
                result = _download_station_batch(
                    args,
                    client,
                    event_raw_dir=root / "raw" / "20240102030355",
                    event_id="20240102030355",
                    batch_index=0,
                    stations=stations,
                    request_start_dt=datetime(2024, 1, 2, 3, 4, tzinfo=JST),
                    span_minutes=3,
                )

            self.assertTrue(result.success)
            self.assertTrue(result.used_minute_fallback)
            self.assertEqual(len(result.segment_paths), 3)
            self.assertEqual(request_mock.call_count, 4)
            self.assertTrue(result.channel_path.is_file())

    def test_event_commit_verify_cleanup_and_finalize(self) -> None:
        cnt = make_cnt_record((0x1234, 0x1235, 0x1236))
        coverage = scan_hinet_vm_cnt_bytes([cnt])
        event_id = "20240102030355"
        event = example_event(event_id)
        manifest = pd.DataFrame([{
            "event_id": event_id,
            "hinet_station": "N.TESTH",
            "cut_start_timestamp": coverage.start_timestamp,
            "cut_end_timestamp": coverage.end_timestamp,
            "ppick_timestamp": coverage.start_timestamp,
        }])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_hdf5 = root / "japan_2024.hdf5"
            source_hdf5.write_bytes(b"source-metadata-placeholder")
            cnt_path = root / "native.cnt"
            cnt_path.write_bytes(cnt)
            channel_path = root / "channels.euc.ch"
            write_three_component_channel_table(channel_path, ["N.TESTH"])
            final_archive = root / "archive" / "hinet_raw_2024.h5"
            args = SimpleNamespace(
                write_mseed=False,
                response_mode="none",
                overwrite_raw=False,
                hdf5=source_hdf5,
                output_root=root,
                archive_path=final_archive,
                archive_chunk_bytes=65536,
                year=2024,
                hinet_network="0101",
                match_distance_km=0.5,
                pre_seconds=0.0,
                post_seconds=1.0,
                mode="all",
                event_ids=None,
                origin_corrections=None,
                dry_run=False,
                keep_staging=False,
                sleep_seconds=0.0,
                component="vertical",
            )
            raw = RawDownloadResult(
                cnt_path=None,
                ch_path=channel_path,
                segment_paths=(cnt_path,),
                raw_status="downloaded_unmerged",
                raw_error="",
            )

            with (
                patch(
                    "tools.download_hinet_velocity.load_download_context",
                    return_value=({event_id: event}, pd.DataFrame(), pd.DataFrame(), None, None),
                ),
                patch("tools.download_hinet_velocity.build_event_arrivals", return_value=manifest),
                patch("tools.download_hinet_velocity.make_hinet_client", return_value=object()),
                patch("tools.download_hinet_velocity.download_raw_event", return_value=raw),
            ):
                result = process_events_archive(args)

            self.assertTrue(result["complete"])
            self.assertTrue(final_archive.exists())
            self.assertFalse(final_archive.with_name("hinet_raw_2024.partial.h5").exists())
            with AnnualHinetArchiveReader(final_archive) as reader:
                self.assertTrue(reader.complete)
                self.assertEqual(reader.event_ids(), [event_id])
                self.assertEqual(reader.manifest(event_id).iloc[0]["raw_time_coverage_complete"], 1)
                self.assertEqual(set(reader.read_series(event_id, {"1234"})), {"1234"})
                archived_row = reader.manifest(event_id).iloc[0]
            velocity = load_archive_velocity(archived_row, args)
            self.assertIsNotNone(velocity)
            self.assertEqual(velocity.status, "loaded")
            self.assertEqual(velocity.values.tolist(), [10, 11, 9, 12])
            qc_args = SimpleNamespace(
                download_root=root,
                hdf5=source_hdf5,
                event_id=event_id,
            )
            archive_manifest = load_manifest(qc_args)
            self.assertEqual(archive_manifest.iloc[0]["archive_event_id"], event_id)


if __name__ == "__main__":
    unittest.main()
