import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from tools import diagnose_query_geometry_sensitivity as querydiag
from tools import merge_query_geometry_diagnostics_shards as querymerge


class _SyntheticDataset:
    def __init__(self, *, protocol="normal", samples=None):
        self.deterministic_sampling_seed = 42
        self.oversample = 1
        self.shuffle = False
        self.realtime_training = {
            "enabled": True,
            "mode": "val",
            "reference": "first_p_pick",
            "val_times": [1, 3, 5, 10, 20, 40, 90],
            "train_times": None,
            "train_time_bins": [[0, 1], [1, 3]],
            "bins_per_event_per_epoch": 1,
            "bin_sampling": "without_replacement",
        }
        self.realtime_target_sampling = {
            "enabled": True,
            "input_ratio": 0.3,
            "triggered_noninput_ratio": 0.2,
            "untriggered_ratio": 0.5,
            "fill_missing": True,
            "exclude_inputs": False,
        }
        if protocol == "random":
            self.causal_random_input_mask = {
                "enabled": True,
                "apply_probability": 1.0,
                "station_counts": [1, 3, 5, 8, 12, 16],
                "order_selected_by_pick": True,
                "target_sampling": {
                    "enabled": True,
                    "input_ratio": 0.0,
                    "triggered_noninput_ratio": 0.2,
                    "untriggered_ratio": 0.5,
                    "fill_missing": True,
                    "exclude_inputs": True,
                },
            }
        else:
            self.causal_random_input_mask = {"enabled": False}
        self.samples = list(samples or self._default_samples())

    @staticmethod
    def _sample(event_id, station_valid, query_x, truth, target_type):
        n_stations = len(station_valid)
        n_targets = len(query_x)
        waveforms = torch.zeros((n_stations, 3, 8), dtype=torch.float32)
        station_coords = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [999.0, 999.0, 999.0]][:n_stations],
            dtype=torch.float32,
        )
        query_coords = torch.tensor(
            [[value, 0.0, 0.0] for value in query_x],
            dtype=torch.float32,
        )
        query_valid = torch.tensor(
            [index < n_targets - 1 for index in range(n_targets)],
            dtype=torch.bool,
        )
        inputs = [
            waveforms,
            station_coords,
            torch.tensor(station_valid, dtype=torch.bool),
            query_coords,
            query_valid,
        ]
        labels = [torch.tensor(truth, dtype=torch.float32).reshape(-1, 1)]
        info = {
            "event_id": event_id,
            "realtime_elapsed_time": torch.tensor(3.0),
            "realtime_target_type": torch.tensor(target_type, dtype=torch.int64),
        }
        return inputs, labels, info

    @classmethod
    def _default_samples(cls):
        return [
            cls._sample(
                "event-a",
                [True, True, False],
                [0.0, 2.0, 4.0, 999.0],
                [0.2, 1.8, 3.7, -999.0],
                [0, 1, 2, -1],
            ),
            cls._sample(
                "event-b",
                [True, False, False],
                [-1.0, 1.0, 3.0, -999.0],
                [-0.8, 1.2, 2.5, 999.0],
                [1, 2, 2, -1],
            ),
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class _CoordinateSensitiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output_layout = ["pga"]
        self.pga_query_token = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self._last_diag = {}

    def forward(
        self,
        waveform,
        station_coords,
        station_valid,
        query_coords,
        query_valid,
        *extra_inputs,
    ):
        del waveform, station_coords, station_valid, extra_inputs
        self._last_diag = {
            "coords_emb_norm": torch.tensor(2.0, device=query_coords.device),
            "wave_emb_norm": torch.tensor(4.0, device=query_coords.device),
            "pga_cross_attn_entropy": torch.tensor(0.25, device=query_coords.device),
            "pga_cross_attn_max_weight": torch.tensor(0.75, device=query_coords.device),
            "pga_cross_attn_valid_mass": torch.tensor(1.0, device=query_coords.device),
            "station_raw_pairwise_cosine": torch.tensor(0.9, device=query_coords.device),
        }
        prediction = query_coords[..., :1]
        return [prediction * query_valid[..., None].to(prediction.dtype)]


class _QueryInvariantModel(_CoordinateSensitiveModel):
    def forward(
        self,
        waveform,
        station_coords,
        station_valid,
        query_coords,
        query_valid,
        *extra_inputs,
    ):
        del waveform, station_coords, station_valid, extra_inputs
        self._last_diag = {}
        prediction = torch.zeros_like(query_coords[..., :1])
        return [prediction * query_valid[..., None].to(prediction.dtype)]


class QueryGeometryHelperTests(unittest.TestCase):
    def test_validation_split_is_enforced(self):
        self.assertEqual(querydiag.require_validation_split("dev"), "val")
        with self.assertRaisesRegex(ValueError, "validation-only"):
            querydiag.require_validation_split("test")
        with self.assertRaisesRegex(ValueError, "validation-only"):
            querydiag.require_validation_split("train")

    def test_scale_one_is_exact_and_invalid_slots_are_unchanged(self):
        query = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [999.0, 999.0, 999.0]]
        )
        query_valid = torch.tensor([True, True, False])
        stations = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [500.0, 500.0, 500.0]]
        )
        station_valid = torch.tensor([True, True, False])

        scale_one, centroid = querydiag.radial_scale_query_coordinates(
            query, query_valid, stations, station_valid, 1.0
        )
        torch.testing.assert_close(scale_one, query, rtol=0, atol=0)
        torch.testing.assert_close(centroid, torch.tensor([1.0, 0.0, 0.0]))

        collapsed, _ = querydiag.radial_scale_query_coordinates(
            query, query_valid, stations, station_valid, 0.0
        )
        torch.testing.assert_close(collapsed[:2], centroid.expand(2, -1))
        torch.testing.assert_close(collapsed[2], query[2], rtol=0, atol=0)

    def test_invalid_targets_cannot_affect_spatial_metrics(self):
        metrics = querydiag.compute_spatial_field_metrics(
            truth=np.array([0.0, 2.0, 1e9]),
            prediction=np.array([0.5, 1.5, -1e9]),
            valid=np.array([True, True, False]),
            pair_sample_limit=100,
        )
        self.assertEqual(metrics["valid_target_count"], 2)
        self.assertEqual(metrics["pair_count"], 1)
        self.assertAlmostEqual(metrics["event_centered_mae"], 0.5)
        self.assertAlmostEqual(metrics["pairwise_delta_mae"], 1.0)

    def test_query_permutation_restores_original_order(self):
        dataset = _SyntheticDataset()
        inputs, _, _ = dataset[0]
        permutation = np.array([2, 0, 3, 1])
        permuted = querydiag.permute_query_aligned_inputs(inputs, permutation)
        restored = querydiag.inverse_permute(
            permuted[3].numpy(), permutation
        )
        np.testing.assert_array_equal(restored, inputs[3].numpy())

    def test_missing_optional_gates_are_explicit(self):
        report = querydiag.inspect_checkpoint_parameters(_CoordinateSensitiveModel())
        self.assertEqual(report["pga_query_token"]["status"], "present")
        self.assertEqual(report["waveform_scale_gate"]["status"], "missing")
        self.assertIsNone(report["waveform_scale_gate"]["parameters"])
        self.assertIn("reason", report["waveform_scale_gate"])

    def test_checkpoint_epoch_and_tensor_provenance_are_inspected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint.pth"
            torch.save(
                {
                    "epoch": 32,
                    "loss": 0.25,
                    "checkpoint_format": "non_encoder_v1",
                    "excluded_prefixes": ["waveform_model.0."],
                    "excluded_tensor_count": 3,
                    "saved_tensor_count": 2,
                    "total_tensor_count": 5,
                    "encoder_source": "/encoder/source.pt",
                    "model_state_dict": {"head.weight": torch.ones(1)},
                },
                checkpoint_path,
            )
            identity = querydiag.inspect_checkpoint_file(checkpoint_path)
            self.assertEqual(identity["epoch"], 32)
            self.assertEqual(identity["loss"], 0.25)
            self.assertEqual(identity["checkpoint_format"], "non_encoder_v1")
            self.assertEqual(identity["excluded_tensor_count"], 3)
            self.assertEqual(identity["saved_tensor_count"], 2)
            self.assertEqual(identity["total_tensor_count"], 5)
            self.assertTrue(identity["external_encoder_required"])
            validation = querydiag.validate_checkpoint_epoch(identity, 32)
            self.assertTrue(validation["verified"])
            with self.assertRaisesRegex(ValueError, "epoch mismatch"):
                querydiag.validate_checkpoint_epoch(identity, 6)

    def test_wrong_expected_epoch_fails_before_dataset_construction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_path = root / "checkpoint.pth"
            config_path = root / "config.json"
            diting_path = root / "diting.yml"
            torch.save(
                {"epoch": 5, "model_state_dict": {}},
                checkpoint_path,
            )
            config_path.write_text("{}\n", encoding="utf-8")
            diting_path.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(
                querydiag.eval_checkpoint, "build_datasets"
            ) as build_datasets:
                with self.assertRaisesRegex(ValueError, "epoch mismatch"):
                    querydiag.main([
                        "--config", str(config_path),
                        "--checkpoint", str(checkpoint_path),
                        "--expected-checkpoint-epoch", "6",
                        "--protocol", "normal",
                        "--split", "val",
                        "--output-prefix", str(root / "result"),
                        "--diting-config", str(diting_path),
                    ])
                build_datasets.assert_not_called()

    def test_non_encoder_checkpoint_requires_matching_explicit_encoder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            actual_encoder = root / "actual.pt"
            expected_encoder = root / "expected.pt"
            actual_encoder.write_bytes(b"actual")
            expected_encoder.write_bytes(b"expected")
            identity = {
                "external_encoder_required": True,
                "encoder_source": str(expected_encoder),
            }
            with self.assertRaisesRegex(ValueError, "explicit"):
                querydiag.validate_encoder_source(
                    identity,
                    None,
                    encoder_was_explicit=False,
                )
            with self.assertRaisesRegex(ValueError, "mismatch"):
                querydiag.validate_encoder_source(
                    identity,
                    actual_encoder,
                    encoder_was_explicit=True,
                )
            unsafe = querydiag.validate_encoder_source(
                identity,
                actual_encoder,
                encoder_was_explicit=True,
                allow_unsafe_mismatch=True,
            )
            self.assertEqual(unsafe["status"], "mismatch_allowed_unsafe")
            self.assertTrue(unsafe["unsafe_mismatch_override"])
            matched = querydiag.validate_encoder_source(
                identity,
                expected_encoder,
                encoder_was_explicit=True,
            )
            self.assertEqual(matched["status"], "matched")

    def test_provenance_records_configs_encoder_architecture_and_sampling(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.json"
            checkpoint_path = root / "checkpoint.pth"
            diting_config_path = root / "diting.yml"
            encoder_path = root / "encoder.pt"
            for path, content in (
                (config_path, b"{}\n"),
                (checkpoint_path, b"checkpoint"),
                (diting_config_path, b"base_width: 128\n"),
                (encoder_path, b"encoder"),
            ):
                path.write_bytes(content)
            generator_validation = {
                "status": "passed",
                "generators": [{"deterministic_sampling_seed": 42}],
            }
            provenance = querydiag.build_provenance(
                config_identity=querydiag.file_provenance(
                    config_path, compute_sha256=True
                ),
                checkpoint_identity={
                    "path": str(checkpoint_path),
                    "epoch": 32,
                    "checkpoint_format": "non_encoder_v1",
                    "excluded_prefixes": ["waveform_model.0."],
                    "excluded_tensor_count": 1,
                    "saved_tensor_count": 2,
                    "total_tensor_count": 3,
                },
                checkpoint_epoch_validation={"verified": True},
                diting_config_identity=querydiag.file_provenance(
                    diting_config_path, compute_sha256=True
                ),
                diting_encoder_identity=querydiag.file_provenance(
                    encoder_path, compute_sha256=True
                ),
                encoder_source_validation={"status": "matched"},
                diting_args=argparse.Namespace(base_width=128, model_depth=24),
                generator_protocol_validation=generator_validation,
                config_source_mode="resolved",
                deployment_source_identity={
                    "mode": "uploaded_sha256",
                    "sha256": "abc",
                },
                protocol="normal",
                split="val",
                seed=17,
                station_counts=[1, 3, 5, 8, 12, 16],
                radial_scales=[0.0, 0.5, 1.0, 1.5],
                max_events=8,
                pair_sample_limit=4096,
                equivariance_tolerance=1e-5,
                checkpoint_sha256=True,
                invocation_argv=["diagnose", "--split", "val"],
            )
            self.assertEqual(provenance["config_source_mode"], "resolved")
            self.assertEqual(
                provenance["deployment_source_identity"]["mode"],
                "uploaded_sha256",
            )
            self.assertIsNotNone(provenance["resolved_run_config"]["sha256"])
            self.assertIsNotNone(provenance["checkpoint"]["sha256"])
            self.assertIsNotNone(
                provenance["diting"]["pretrained_encoder"]["sha256"]
            )
            self.assertEqual(
                provenance["diting"]["architecture_arguments"]["base_width"],
                128,
            )
            self.assertEqual(provenance["diagnostic_seed"], 17)
            self.assertEqual(
                provenance["generator_sampling"], generator_validation
            )

    def test_output_overwrite_requires_force_and_npz_has_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = querydiag.diagnostic_output_paths(Path(temp_dir) / "run")
            config = {"model_params": {"n_pga_targets": 4}}
            querydiag.write_outputs(
                paths,
                config=config,
                summary={"counts": {"events": 1}},
                arrays={"value": np.array([1.0])},
                provenance={"split": "val"},
                force=False,
            )
            with paths["summary"].open(encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["provenance"]["split"], "val")
            with paths["completion"].open(encoding="utf-8") as handle:
                completion = json.load(handle)
            self.assertEqual(completion["status"], "complete")
            self.assertEqual(set(completion["artifacts"]), {
                "resolved_config", "samples", "summary"
            })
            with np.load(paths["samples"]) as archive:
                self.assertEqual(json.loads(str(archive["provenance_json"]))["split"], "val")
                self.assertEqual(
                    json.loads(str(archive["resolved_config_json"])), config
                )
            with self.assertRaises(FileExistsError):
                querydiag.write_outputs(
                    paths,
                    config=config,
                    summary={},
                    arrays={},
                    provenance={},
                    force=False,
                )

    def test_serialization_failure_leaves_no_final_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = querydiag.diagnostic_output_paths(root / "run")
            with mock.patch.object(
                querydiag, "_serialize_npz", side_effect=RuntimeError("boom")
            ):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    querydiag.write_outputs(
                        paths,
                        config={},
                        summary={},
                        arrays={},
                        provenance={},
                        force=False,
                    )
            self.assertFalse(any(path.exists() for path in paths.values()))
            self.assertEqual(list(root.glob(".*.tmp.*")), [])

    def test_partial_publication_is_detected_and_force_can_recover(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = querydiag.diagnostic_output_paths(Path(temp_dir) / "run")
            real_replace = os.replace
            calls = 0

            def fail_second_replace(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("publish interrupted")
                return real_replace(source, destination)

            with mock.patch.object(
                querydiag.os, "replace", side_effect=fail_second_replace
            ):
                with self.assertRaisesRegex(RuntimeError, "publish interrupted"):
                    querydiag.write_outputs(
                        paths,
                        config={},
                        summary={},
                        arrays={"x": np.ones(1)},
                        provenance={},
                        force=False,
                    )
            self.assertFalse(paths["completion"].exists())
            with self.assertRaisesRegex(FileExistsError, "partial/incomplete"):
                querydiag.refuse_existing_outputs(paths, force=False)
            querydiag.write_outputs(
                paths,
                config={},
                summary={},
                arrays={"x": np.ones(1)},
                provenance={},
                force=True,
            )
            self.assertTrue(all(path.is_file() for path in paths.values()))


class QueryGeometryEndToEndTests(unittest.TestCase):
    @staticmethod
    def _run(
        model,
        dataset=None,
        *,
        protocol="normal",
        max_events=0,
        num_event_shards=1,
        event_shard_id=0,
    ):
        return querydiag.run_query_geometry_diagnostics(
            model,
            dataset or _SyntheticDataset(protocol=protocol),
            torch.device("cpu"),
            {},
            protocol=protocol,
            station_counts=[1, 3, 5],
            radial_scales=[0.0, 0.5, 1.0, 1.5],
            seed=17,
            max_events=max_events,
            num_event_shards=num_event_shards,
            event_shard_id=event_shard_id,
            pair_sample_limit=100,
            equivariance_tolerance=1e-7,
        )

    @staticmethod
    def _event_block_dataset(event_count=5, *, protocol="normal"):
        samples = []
        for event_ordinal in range(event_count):
            for elapsed_time in querydiag.PINNED_VALIDATION_TIMES:
                sample = _SyntheticDataset._sample(
                    f"event-{event_ordinal}",
                    [True, False, False],
                    [0.0, 1.0, 2.0, 999.0],
                    [0.1, 1.1, 2.1, -999.0],
                    [0, 1, 2, -1],
                )
                sample[2]["realtime_elapsed_time"] = torch.tensor(elapsed_time)
                samples.append(sample)
        return _SyntheticDataset(protocol=protocol, samples=samples)

    def test_coordinate_sensitive_model_has_nonzero_radial_sensitivity(self):
        summary, arrays = self._run(_CoordinateSensitiveModel())

        self.assertEqual(summary["counts"]["events"], 2)
        self.assertEqual(summary["counts"]["realtime_samples"], 2)
        self.assertEqual(summary["counts"]["valid_targets"], 6)
        self.assertEqual(summary["counts"]["station_count_histogram"], {"1": 1, "2": 1})
        self.assertEqual(summary["counts"]["target_type_counts"]["input"], 1)
        for metric_name in ("correlation", "r2", "slope", "intercept"):
            self.assertIsNotNone(summary["baseline"]["point_metrics"][metric_name])
        groups = summary["baseline"]["target_groups"]
        self.assertEqual(
            set(groups),
            {"all", "non_input", "triggered_noninput", "untriggered", "input"},
        )
        self.assertEqual(groups["all"]["point_metrics"]["targets"], 6)
        self.assertEqual(groups["non_input"]["point_metrics"]["targets"], 5)
        self.assertEqual(groups["triggered_noninput"]["point_metrics"]["targets"], 2)
        self.assertEqual(groups["untriggered"]["point_metrics"]["targets"], 3)
        self.assertEqual(groups["input"]["point_metrics"]["targets"], 1)
        self.assertEqual(
            groups["all"]["spatial_field_metrics"][
                "valid_target_count_at_least_2"
            ]["realtime_samples"],
            2,
        )
        self.assertEqual(
            groups["input"]["spatial_field_metrics"][
                "valid_target_count_at_least_2"
            ]["realtime_samples"],
            0,
        )
        self.assertEqual(
            groups["all"]["spatial_field_metrics"][
                "valid_target_count_at_least_5"
            ]["realtime_samples"],
            0,
        )
        self.assertGreater(
            summary["radial_interventions"]["0.0"][
                "mean_abs_prediction_change_from_scale_1"
            ],
            0.0,
        )
        scale_one_index = int(np.where(arrays["radial_scales"] == 1.0)[0][0])
        np.testing.assert_array_equal(
            arrays["radial_query_coords"][:, scale_one_index],
            arrays["query_coords"],
        )
        np.testing.assert_array_equal(
            arrays["radial_prediction_change_from_scale_1"][:, scale_one_index],
            np.zeros_like(arrays["baseline_prediction"]),
        )
        self.assertEqual(summary["query_order_equivariance"]["failed_samples"], 0)
        self.assertAlmostEqual(
            summary["model_internal_diagnostics"][
                "coordinate_to_wave_embedding_norm_ratio"
            ]["mean"],
            0.5,
        )
        radial_groups = summary["radial_interventions"]["0.0"]["target_groups"]
        self.assertEqual(set(radial_groups), set(groups))
        self.assertEqual(radial_groups["non_input"]["targets"], 5)
        self.assertIn("1", radial_groups["non_input"]["by_station_count"])
        self.assertIn("2", radial_groups["non_input"]["by_station_count"])
        self.assertEqual(
            radial_groups["input"]["predicted_p95_p05_range"][
                "valid_target_count_at_least_2"
            ]["realtime_samples"],
            0,
        )

    def test_query_invariant_model_has_zero_radial_sensitivity(self):
        summary, arrays = self._run(_QueryInvariantModel())
        for scale_summary in summary["radial_interventions"].values():
            self.assertEqual(
                scale_summary["mean_abs_prediction_change_from_scale_1"], 0.0
            )
        np.testing.assert_array_equal(
            arrays["radial_prediction_change_from_scale_1"],
            np.zeros_like(arrays["radial_prediction_change_from_scale_1"]),
        )

    def test_max_events_keeps_all_realtime_samples_for_selected_event(self):
        first = _SyntheticDataset._default_samples()[0]
        repeated = _SyntheticDataset._sample(
            "event-a",
            [True, False, False],
            [0.0, 1.0, 2.0, 999.0],
            [0.0, 1.0, 2.0, -999.0],
            [1, 2, 2, -1],
        )
        next_event = _SyntheticDataset._default_samples()[1]
        dataset = _SyntheticDataset(samples=[first, repeated, next_event])
        summary, _ = self._run(
            _CoordinateSensitiveModel(), dataset=dataset, max_events=1
        )
        self.assertEqual(summary["counts"]["events"], 1)
        self.assertEqual(summary["counts"]["realtime_samples"], 2)
        self.assertEqual(summary["selection"]["examined_realtime_samples"], 2)

    def test_event_shards_are_disjoint_complete_and_keep_whole_events(self):
        dataset = self._event_block_dataset(event_count=5)
        shard_arrays = []
        for shard_id in range(3):
            summary, arrays = self._run(
                _CoordinateSensitiveModel(),
                dataset=dataset,
                num_event_shards=3,
                event_shard_id=shard_id,
            )
            shard_arrays.append(arrays)
            sharding = summary["selection"]["event_sharding"]
            self.assertEqual(sharding["mode"], "shard")
            self.assertEqual(sharding["event_shard_id"], shard_id)
            self.assertTrue(np.all(arrays["event_ordinal"] % 3 == shard_id))
            for ordinal in np.unique(arrays["event_ordinal"]):
                self.assertEqual(np.sum(arrays["event_ordinal"] == ordinal), 7)
        concatenated = np.concatenate(
            [arrays["event_index"] for arrays in shard_arrays]
        )
        np.testing.assert_array_equal(
            np.sort(concatenated), np.arange(len(dataset))
        )
        self.assertEqual(np.unique(concatenated).size, len(dataset))

    def test_event_sharding_rejects_max_events_and_noncontiguous_blocks(self):
        dataset = self._event_block_dataset(event_count=2)
        with self.assertRaisesRegex(ValueError, "max_events must be 0"):
            self._run(
                _CoordinateSensitiveModel(),
                dataset=dataset,
                max_events=1,
                num_event_shards=2,
            )
        dataset.samples[3] = _SyntheticDataset._sample(
            "wrong-event",
            [True, False, False],
            [0.0, 1.0, 2.0, 999.0],
            [0.1, 1.1, 2.1, -999.0],
            [0, 1, 2, -1],
        )
        with self.assertRaisesRegex(ValueError, "not event-contiguous"):
            self._run(
                _CoordinateSensitiveModel(),
                dataset=dataset,
                num_event_shards=2,
                event_shard_id=0,
            )

    def test_verified_merge_reconstructs_global_sample_order(self):
        dataset = self._event_block_dataset(event_count=4)
        # Resampling can repeat one physical event ID as multiple complete
        # seven-time occurrences. Occurrences may land in different shards,
        # while merged unique-event counts must remain based on event_key.
        for sample in dataset.samples[14:21]:
            sample[2]["event_id"] = "event-1"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_base = root / "run"
            for shard_id in range(2):
                summary, arrays = self._run(
                    _CoordinateSensitiveModel(),
                    dataset=dataset,
                    num_event_shards=2,
                    event_shard_id=shard_id,
                )
                provenance = {
                    "protocol": "normal",
                    "split": "val",
                    "diagnostic_seed": 17,
                    "station_counts": [1, 3, 5],
                    "radial_scales": [0.0, 0.5, 1.0, 1.5],
                    "max_events": 0,
                    "pair_sample_limit": 100,
                    "equivariance_tolerance": 1e-7,
                    "event_sharding": summary["selection"]["event_sharding"],
                }
                prefix = querymerge.shard_prefix(input_base, shard_id, 2)
                querydiag.write_outputs(
                    querydiag.diagnostic_output_paths(prefix),
                    config={"model": "synthetic"},
                    summary=summary,
                    arrays=arrays,
                    provenance=provenance,
                    force=False,
                )
            output = root / "merged"
            querymerge.merge_shards(input_base, 2, output)
            paths = querydiag.diagnostic_output_paths(output)
            with np.load(paths["samples"], allow_pickle=False) as archive:
                np.testing.assert_array_equal(
                    archive["event_index"], np.arange(len(dataset))
                )
                np.testing.assert_array_equal(
                    archive["event_ordinal"], np.arange(len(dataset)) // 7
                )
            with paths["summary"].open(encoding="utf-8") as handle:
                merged_summary = json.load(handle)
            self.assertEqual(merged_summary["counts"]["events"], 3)
            self.assertEqual(merged_summary["counts"]["realtime_samples"], 28)
            self.assertEqual(
                merged_summary["selection"]["selected_event_occurrences"], 4
            )
            self.assertEqual(
                merged_summary["provenance"]["event_sharding"][
                    "selected_unique_event_keys"
                ],
                3,
            )
            self.assertEqual(
                merged_summary["provenance"]["event_sharding"]["mode"],
                "merged",
            )
            self.assertEqual(len(merged_summary["provenance"]["merge"]["source_shards"]), 2)

    def test_merge_rejects_artifact_changed_after_completion(self):
        dataset = self._event_block_dataset(event_count=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_base = root / "run"
            for shard_id in range(2):
                summary, arrays = self._run(
                    _CoordinateSensitiveModel(),
                    dataset=dataset,
                    num_event_shards=2,
                    event_shard_id=shard_id,
                )
                provenance = {
                    "diagnostic_seed": 17,
                    "pair_sample_limit": 100,
                    "equivariance_tolerance": 1e-7,
                    "event_sharding": summary["selection"]["event_sharding"],
                }
                prefix = querymerge.shard_prefix(input_base, shard_id, 2)
                querydiag.write_outputs(
                    querydiag.diagnostic_output_paths(prefix),
                    config={},
                    summary=summary,
                    arrays=arrays,
                    provenance=provenance,
                    force=False,
                )
            changed = querydiag.diagnostic_output_paths(
                querymerge.shard_prefix(input_base, 1, 2)
            )["summary"]
            with changed.open("a", encoding="utf-8") as handle:
                handle.write(" ")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                querymerge.merge_shards(input_base, 2, root / "merged")

    def test_event_counts_include_joint_generator_source_identity(self):
        sample_a = _SyntheticDataset._default_samples()[0]
        sample_b = _SyntheticDataset._sample(
            "event-a",
            [True, False, False],
            [0.0, 1.0, 2.0, 999.0],
            [0.0, 1.0, 2.0, -999.0],
            [1, 2, 2, -1],
        )
        dataset = _SyntheticDataset(samples=[sample_a, sample_b])
        dataset.indexes = [(0, 0), (1, 0)]
        summary, arrays = self._run(_CoordinateSensitiveModel(), dataset=dataset)
        self.assertEqual(summary["counts"]["events"], 2)
        np.testing.assert_array_equal(arrays["dataset_source_index"], [0, 1])

    def test_spatial_threshold_at_least_five_includes_only_eligible_samples(self):
        large_sample = _SyntheticDataset._sample(
            "event-large",
            [True, True, False],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 999.0],
            [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, -999.0],
            [0, 1, 2, 1, 2, 2, -1],
        )
        summary, _ = self._run(
            _CoordinateSensitiveModel(),
            dataset=_SyntheticDataset(samples=[large_sample]),
        )
        field = summary["baseline"]["target_groups"]["all"][
            "spatial_field_metrics"
        ]
        self.assertEqual(
            field["valid_target_count_at_least_5"]["realtime_samples"], 1
        )
        radial = summary["radial_interventions"]["0.0"]["target_groups"]["all"]
        self.assertEqual(
            radial["predicted_p95_p05_range"][
                "valid_target_count_at_least_5"
            ]["realtime_samples"],
            1,
        )

    def test_protocol_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            self._run(
                _CoordinateSensitiveModel(),
                dataset=_SyntheticDataset(protocol="random"),
                protocol="normal",
            )
        with self.assertRaisesRegex(ValueError, "requires"):
            self._run(
                _CoordinateSensitiveModel(),
                dataset=_SyntheticDataset(protocol="normal"),
                protocol="random",
            )

    def test_random_protocol_rejects_disabled_or_input_including_targets(self):
        disabled = _SyntheticDataset(protocol="random")
        disabled.causal_random_input_mask["target_sampling"]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "target_sampling.enabled"):
            self._run(
                _CoordinateSensitiveModel(), dataset=disabled, protocol="random"
            )

        includes_input = _SyntheticDataset(protocol="random")
        includes_input.causal_random_input_mask["target_sampling"][
            "exclude_inputs"
        ] = False
        with self.assertRaisesRegex(ValueError, "exclude_inputs"):
            self._run(
                _CoordinateSensitiveModel(),
                dataset=includes_input,
                protocol="random",
            )

    def test_random_protocol_rejects_wrong_station_set(self):
        dataset = _SyntheticDataset(protocol="random")
        dataset.causal_random_input_mask["station_counts"] = [1, 3, 5]
        with self.assertRaisesRegex(ValueError, "station_counts set"):
            self._run(
                _CoordinateSensitiveModel(), dataset=dataset, protocol="random"
            )

    def test_protocol_rejects_wrong_realtime_mode_or_times(self):
        wrong_mode = _SyntheticDataset(protocol="random")
        wrong_mode.realtime_training["mode"] = "train"
        with self.assertRaisesRegex(ValueError, "mode='val'"):
            self._run(
                _CoordinateSensitiveModel(), dataset=wrong_mode, protocol="random"
            )

        wrong_times = _SyntheticDataset(protocol="normal")
        wrong_times.realtime_training["val_times"] = [1, 3, 5]
        with self.assertRaisesRegex(ValueError, "pinned validation val_times"):
            self._run(
                _CoordinateSensitiveModel(), dataset=wrong_times, protocol="normal"
            )

    def test_normal_protocol_rejects_active_random_mask(self):
        dataset = _SyntheticDataset(protocol="random")
        with self.assertRaisesRegex(ValueError, "does not match"):
            self._run(
                _CoordinateSensitiveModel(), dataset=dataset, protocol="normal"
            )


class QueryGeometryLauncherTests(unittest.TestCase):
    repo_root = Path(__file__).resolve().parents[1]
    launcher = repo_root / "tools" / "run_query_geometry_diagnostics_slurm.sh"

    def _launcher_environment(self, root):
        root = Path(root)
        placeholder = root / "placeholder"
        placeholder.write_bytes(b"placeholder")
        environment = os.environ.copy()
        environment.update({
            "WORKDIR": str(self.repo_root),
            "DIAGNOSTIC_SCRIPT": str(placeholder),
            "RT55_CONFIG": str(placeholder),
            "RT56_CONFIG": str(placeholder),
            "RT55_CHECKPOINT": str(placeholder),
            "RT56_CHECKPOINT": str(placeholder),
            "DITING_CONFIG": str(placeholder),
            "DITING_PRETRAINED": str(placeholder),
            "OUT": str(root / "outputs"),
            "ACTION": "rt55_normal",
            "DRY_RUN": "1",
            "CONFIG_SOURCE_MODE": "resolved",
        })
        return environment

    def test_submission_rejects_same_name_active_job(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            squeue = fake_bin / "squeue"
            squeue.write_text("#!/bin/sh\necho '12345 RUNNING'\n", encoding="utf-8")
            squeue.chmod(0o755)
            environment = self._launcher_environment(root)
            environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
            result = subprocess.run(
                ["bash", str(self.launcher)],
                cwd=self.repo_root,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("same-name Slurm job", result.stderr)

    def test_uploaded_sha256_mode_does_not_require_git_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            environment = self._launcher_environment(root)
            environment["WORKDIR"] = str(root)
            environment["ALLOW_ACTIVE_JOB"] = "1"
            environment["SOURCE_IDENTITY_MODE"] = "uploaded_sha256"
            diagnostic = Path(environment["DIAGNOSTIC_SCRIPT"])
            environment["EXPECTED_DIAGNOSTIC_SHA256"] = hashlib.sha256(
                diagnostic.read_bytes()
            ).hexdigest()
            environment["EXPECTED_LAUNCHER_SHA256"] = hashlib.sha256(
                self.launcher.read_bytes()
            ).hexdigest()
            result = subprocess.run(
                ["bash", str(self.launcher)],
                cwd=self.repo_root,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("source_identity_mode=uploaded_sha256", result.stdout)
            self.assertIn("SHA-256 identities matched", result.stdout)

    def test_sharded_submission_uses_bounded_slurm_array(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            environment = self._launcher_environment(temp_dir)
            environment["ALLOW_ACTIVE_JOB"] = "1"
            environment["NUM_EVENT_SHARDS"] = "8"
            environment["QUERYDIAG_SHARD_CONCURRENCY"] = "4"
            environment["MAX_EVENTS"] = "0"
            result = subprocess.run(
                ["bash", str(self.launcher)],
                cwd=self.repo_root,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--array=0-7%4", result.stdout)
            self.assertIn("%A_%a", result.stdout)
            self.assertIn("event_shards=8", result.stdout)

    def test_sharded_worker_uses_task_specific_output_lock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            environment = self._launcher_environment(root)
            environment["SLURM_JOB_ID"] = "999"
            environment["SLURM_ARRAY_TASK_ID"] = "3"
            environment["NUM_EVENT_SHARDS"] = "8"
            environment["MAX_EVENTS"] = "0"
            environment["EXPECTED_GIT_COMMIT"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_root,
                text=True,
            ).strip()
            lock = root / "outputs" / (
                "rt55_normal_querydiag.shard-00003-of-00008.lock"
            )
            lock.mkdir(parents=True)
            result = subprocess.run(
                ["bash", str(self.launcher), "rt55_normal"],
                cwd=self.repo_root,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already owns output lock", result.stderr)

    def test_submission_does_not_export_reserved_slurm_gpus_option(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            sbatch = fake_bin / "sbatch"
            sbatch.write_text(
                "#!/bin/sh\n"
                "if env | grep -q '^SLURM_GPUS='; then\n"
                "  echo 'reserved SLURM_GPUS leaked into sbatch' >&2\n"
                "  exit 44\n"
                "fi\n"
                "printf '%s\\n' \"$*\"\n",
                encoding="utf-8",
            )
            sbatch.chmod(0o755)
            environment = self._launcher_environment(root)
            environment.pop("SLURM_GPUS", None)
            environment["PATH"] = (
                f"{fake_bin}{os.pathsep}{environment['PATH']}"
            )
            environment["DRY_RUN"] = "0"
            environment["CONFIRM_QUERY_DIAGNOSTICS"] = "1"
            environment["ALLOW_ACTIVE_JOB"] = "1"
            result = subprocess.run(
                ["bash", str(self.launcher)],
                cwd=self.repo_root,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--gres=dcu:1", result.stdout)
            self.assertNotIn("reserved SLURM_GPUS", result.stderr)

    def test_worker_rejects_git_commit_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            environment = self._launcher_environment(temp_dir)
            environment["SLURM_JOB_ID"] = "999"
            environment["EXPECTED_GIT_COMMIT"] = "0" * 40
            result = subprocess.run(
                ["bash", str(self.launcher), "rt55_normal"],
                cwd=self.repo_root,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Worker Git commit mismatch", result.stderr)

    def test_worker_rejects_existing_output_lock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            environment = self._launcher_environment(root)
            environment["SLURM_JOB_ID"] = "999"
            environment["EXPECTED_GIT_COMMIT"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_root,
                text=True,
            ).strip()
            lock = root / "outputs" / "rt55_normal_querydiag.lock"
            lock.mkdir(parents=True)
            result = subprocess.run(
                ["bash", str(self.launcher), "rt55_normal"],
                cwd=self.repo_root,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already owns output lock", result.stderr)


if __name__ == "__main__":
    unittest.main()
