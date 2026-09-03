import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tools import diagnose_query_geometry_sensitivity as querydiag


class _SyntheticDataset:
    def __init__(self, *, protocol="normal", samples=None):
        if protocol == "random":
            self.causal_random_input_mask = {
                "enabled": True,
                "apply_probability": 1.0,
                "station_counts": [1, 3],
                "target_sampling": {"exclude_inputs": True},
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


class QueryGeometryEndToEndTests(unittest.TestCase):
    @staticmethod
    def _run(model, dataset=None, *, protocol="normal", max_events=0):
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
            pair_sample_limit=100,
            equivariance_tolerance=1e-7,
        )

    def test_coordinate_sensitive_model_has_nonzero_radial_sensitivity(self):
        summary, arrays = self._run(_CoordinateSensitiveModel())

        self.assertEqual(summary["counts"]["events"], 2)
        self.assertEqual(summary["counts"]["realtime_samples"], 2)
        self.assertEqual(summary["counts"]["valid_targets"], 6)
        self.assertEqual(summary["counts"]["station_count_histogram"], {"1": 1, "2": 1})
        self.assertEqual(summary["counts"]["target_type_counts"]["input"], 1)
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


if __name__ == "__main__":
    unittest.main()
