import unittest

import numpy as np
import pandas as pd

from tools.summarize_v01_results import cluster_ci, pair_rows, point_metrics


def sample_frame():
    return pd.DataFrame({"event_id": ["a", "a", "b"], "time_s": [1., 1., 1.],
                         "query_lat": [35., 36., 35.], "query_lon": [140.,140.,140.],
                         "query_depth": [0.,0.,0.], "truth": [-1.,-1.5,-1.],
                         "prediction": [-1.1,-1.4,-1.2], "current_sample": [600.,600.,600.],
                         "first_pick_sample": [500.,500.,500.], "target_type": [1,2,1],
                         "abs_error": [.1,.1,.2], "sigma": [.2,.2,.2],
                         "prob": [.8,.2,.8], "nll": [.1,.2,.1], "target_index": [3,4,3]})


class TestV01ResultSummary(unittest.TestCase):
    def test_physical_coordinates_not_slot_indices(self):
        left = sample_frame(); right = sample_frame().iloc[::-1].copy()
        right["target_index"] += 100
        self.assertEqual(len(pair_rows(left,right)),3)

    def test_duplicate_query_keys_rejected(self):
        frame = sample_frame()
        with self.assertRaisesRegex(ValueError,"duplicate"):
            pair_rows(pd.concat([frame, frame.iloc[:1]]), frame)

    def test_different_truth_or_cutoff_rejected(self):
        left = sample_frame()
        for key in ("truth", "current_sample", "first_pick_sample", "target_type"):
            right = left.copy(); right[key] = right[key].astype(float); right.loc[0,key] += .1
            with self.assertRaisesRegex(ValueError,key):
                pair_rows(left,right)

    def test_cluster_sign_and_determinism(self):
        left = sample_frame(); right = left.copy(); right.abs_error += .03
        paired = pair_rows(left,right)
        a, stats = cluster_ci(paired,100,42); b,_ = cluster_ci(paired,100,42)
        self.assertEqual(a,b); self.assertEqual(a["paired_events"],2)
        for key in ("delta_mae", "ci_low", "ci_high", "event_macro_delta_mae"):
            self.assertAlmostEqual(a[key],.03)
        self.assertEqual(stats["count"].sum(),3)

    def test_metrics_coordinate_and_population(self):
        result = point_metrics(sample_frame())
        self.assertEqual(result["events"],2); self.assertEqual(result["targets"],3)
        self.assertAlmostEqual(result["mae"],.4/3)
        self.assertAlmostEqual(result["brier"],.04)
        self.assertAlmostEqual(result["event_macro_mae"],.15)


if __name__ == "__main__":
    unittest.main()
