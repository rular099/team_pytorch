import unittest

import numpy as np
import pandas as pd

from gemini_util_light import PreloadedEventGenerator
from tools.analyze_v01_padding_controls import paired_cluster_ci
from tools.prep_padding_protocol import (
    apply_prep_intervention,
    assign_retained_prep_samples,
    prep_keep_mask,
)


class V01PaddingControlTests(unittest.TestCase):
    def test_only_source_prep_prefix_is_deleted(self):
        wave = np.arange(2 * 12 * 3, dtype=np.float32).reshape(2, 12, 3)
        valid = np.ones((2, 12), dtype=bool)
        picks = np.array([7, 6])
        retained = np.array([2, 0])
        roles = np.array([1, 0])
        changed, keep = apply_prep_intervention(
            wave, valid, picks, retained, roles
        )
        self.assertFalse(keep[0, :5].any())
        self.assertTrue(keep[0, 5:].all())
        self.assertTrue(keep[1].all())
        np.testing.assert_array_equal(changed[0, 7:], wave[0, 7:])
        np.testing.assert_array_equal(changed[1], wave[1])

    def test_p_and_post_p_are_never_deleted(self):
        valid = np.array([[1, 0, 1, 1, 1, 1, 0]], dtype=bool)
        keep = prep_keep_mask(valid, np.array([3]), np.array([0]))
        np.testing.assert_array_equal(keep[0, 3:], valid[0, 3:])
        self.assertTrue(keep[0, 3])

    def test_intervention_mask_is_subset_of_storage_support(self):
        valid = np.array([[0, 1, 1, 0, 1, 1]], dtype=bool)
        keep = prep_keep_mask(valid, np.array([4]), np.array([2]))
        self.assertFalse(np.any(keep & ~valid))

    def test_template_assignment_is_stable_and_identity_specific(self):
        templates = np.arange(1, 1000)
        first = assign_retained_prep_samples(templates, "event-a", "station-a")
        second = assign_retained_prep_samples(templates, "event-a", "station-a")
        other = assign_retained_prep_samples(templates, "event-a", "station-b")
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

    def test_masked_filler_cannot_change_allowed_samples(self):
        rng = np.random.default_rng(7)
        wave = rng.normal(size=(1, 20, 3)).astype(np.float32)
        valid = np.ones((1, 20), dtype=bool)
        zero, keep = apply_prep_intervention(
            wave, valid, np.array([10]), np.array([3]), fill_value=0.0
        )
        finite, keep_again = apply_prep_intervention(
            wave, valid, np.array([10]), np.array([3]), fill_value=123.0
        )
        np.testing.assert_array_equal(keep, keep_again)
        np.testing.assert_array_equal(zero[keep], finite[keep])

    def test_event_cluster_ci_is_deterministic(self):
        base = pd.DataFrame({
            "event_id": ["a", "a", "b"],
            "decision_time_s": [1.0, 1.0, 1.0],
            "target_index": [0, 1, 0],
            "protocol": ["normal"] * 3,
            "abs_error": [0.1, 0.2, 0.3],
        })
        worse = base.copy()
        worse["abs_error"] += 0.05
        first = paired_cluster_ci(base, worse, 200, 20260915)
        second = paired_cluster_ci(base, worse, 200, 20260915)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["delta_mae"], 0.05)
        self.assertEqual(first["paired_events"], 2)

    def test_absolute_reference_pick_overrides_arm_specific_pick_minimum(self):
        generator = PreloadedEventGenerator.__new__(PreloadedEventGenerator)
        generator.v01_reference_p_pick = 500
        generator.sampling_rate = 100.0
        generator.realtime_training = {
            "val_times": [1.0],
            "train_times": [1.0],
            "train_time_bins": [(0.0, 1.0)],
        }
        context = {"mode": "val", "time_index": 0}
        first = generator._select_realtime_cutout(
            np.array([[450, 520]]), np.array([[True, True]]),
            np.random.default_rng(1), context, 10000,
        )
        second = generator._select_realtime_cutout(
            np.array([[490, 700]]), np.array([[True, True]]),
            np.random.default_rng(2), context, 10000,
        )
        self.assertEqual(first["current_sample"], 600)
        self.assertEqual(first["current_sample"], second["current_sample"])


if __name__ == "__main__":
    unittest.main()
