"""Small numerical regression tests for report-only derived PR metrics."""
import unittest

from prepare_report_assets import pr_statistics


class PRStatisticsTests(unittest.TestCase):
    def test_perfect_ranking_and_fixed_cutoff(self):
        result = pr_statistics([True, False, True, False], [.9, .1, .8, .2])
        self.assertEqual(result['targets'], 4)
        self.assertEqual(result['positive_targets'], 2)
        self.assertAlmostEqual(result['f1_at_alpha_0p5'], 1)
        self.assertAlmostEqual(result['pr_auc_trapezoid'], 1)
        self.assertAlmostEqual(result['average_precision'], 1)

    def test_ties_are_grouped_and_ap_not_trapezoid_area(self):
        first = pr_statistics([True, False], [.5, .5])
        second = pr_statistics([False, True], [.5, .5])
        self.assertEqual(first, second)
        self.assertAlmostEqual(first['f1_at_alpha_0p5'], 2/3)
        self.assertAlmostEqual(first['pr_auc_trapezoid'], .75)
        self.assertAlmostEqual(first['average_precision'], .5)

    def test_zero_selected_cutoff(self):
        result = pr_statistics([True, False], [.4, .1])
        self.assertEqual(result['f1_at_alpha_0p5'], 0)
        self.assertAlmostEqual(result['average_precision'], 1)

    def test_single_class_not_reported_as_discrimination(self):
        for labels in ([False, False], [True, True]):
            result = pr_statistics(labels, [.7, .2])
            self.assertIsNone(result['pr_auc_trapezoid'])
            self.assertIsNone(result['average_precision'])


if __name__ == '__main__':
    unittest.main()
