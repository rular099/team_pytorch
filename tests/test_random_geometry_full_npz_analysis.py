import unittest

import numpy as np

from tools import analyze_random_geometry_full_npz as analysis


class RandomGeometryFullNpzAnalysisTests(unittest.TestCase):
    def _arrays(self):
        truth = np.asarray(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            ]
        )
        valid = np.ones_like(truth, dtype=bool)
        event_id = np.asarray(["event-a", "event-a", "event-b", "event-b"])
        common = {
            "truth": truth,
            "sigma": np.full_like(truth, 0.5),
            "valid": valid,
            "target_type": np.asarray([[1, 1, 2, 2, 2, 2]] * 4),
            "event_id": event_id,
            "event_index": np.arange(4),
            "elapsed_time": np.asarray([1.0, 3.0, 1.0, 3.0]),
            "requested_station_count": np.asarray([1, 1, 3, 3]),
            "selected_station_count": np.asarray([1, 1, 2, 2]),
        }
        baseline = dict(common)
        candidate = dict(common)
        row_mean = truth.mean(axis=1, keepdims=True)
        baseline["prediction"] = (
            row_mean + 0.25 * (truth[:, ::-1] - row_mean) + 0.2
        )
        candidate["prediction"] = truth.copy()
        alignment = {
            key: np.asarray([0]) for key in analysis.ALIGNMENT_KEYS
        }
        baseline["alignment"] = alignment
        candidate["alignment"] = {
            key: value.copy() for key, value in alignment.items()
        }
        return baseline, candidate

    def test_event_aggregates_support_target_matrices(self):
        event_ids = np.asarray(["a", "a", "b"])
        values = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        valid = np.asarray([[True, False], [True, True], [False, True]])

        events, sums, counts = analysis._event_aggregates(event_ids, values, valid)

        np.testing.assert_array_equal(events, ["a", "b"])
        np.testing.assert_allclose(sums, [8.0, 6.0])
        np.testing.assert_array_equal(counts, [3, 1])

    def test_paired_point_bootstrap_keeps_event_clusters(self):
        baseline, candidate = self._arrays()

        result = analysis.paired_point_summary(
            baseline,
            candidate,
            baseline["valid"],
            sample_mask=None,
            bootstrap_replicates=50,
            rng=np.random.default_rng(17),
        )

        self.assertLess(result["mae_delta_candidate_minus_baseline"], 0.0)
        self.assertLess(result["rmse_delta_candidate_minus_baseline"], 0.0)
        self.assertEqual(
            result["event_cluster_bootstrap"][
                "mae_delta_candidate_minus_baseline"
            ]["event_clusters"],
            2,
        )
        self.assertEqual(result["target_improved_fraction"], 1.0)

    def test_build_analysis_reports_spatial_improvement(self):
        baseline, candidate = self._arrays()

        result = analysis.build_analysis(
            baseline,
            candidate,
            baseline_metrics=None,
            candidate_metrics=None,
            pair_sample_limit=4096,
            bootstrap_replicates=20,
            seed=23,
        )

        spatial = result["paired_comparison"]["spatial_by_target_population"][
            "all"
        ]["valid_target_count_at_least_5"]["metrics"]
        self.assertGreater(spatial["pearson"]["delta_candidate_minus_baseline"], 0.0)
        self.assertLess(
            spatial["event_centered_mae"]["delta_candidate_minus_baseline"],
            0.0,
        )
        self.assertEqual(result["alignment"]["status"], "passed")


if __name__ == "__main__":
    unittest.main()
