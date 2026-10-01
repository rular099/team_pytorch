import unittest
import numpy as np
from fe01.sampling import TimeSampler
from fe01.windows import CAPABILITIES


class TimeSamplingTests(unittest.TestCase):
    def test_100000_draws_and_early_mass(self):
        sampler=TimeSampler()
        audit=sampler.audit(100000)
        self.assertTrue(audit['passed'])
        self.assertAlmostEqual(sum(audit['expected'][:3])/100000,.6)
        self.assertEqual(audit['min_seconds'],1)
        self.assertEqual(audit['max_seconds'],90)

    def test_truncated_bin_keeps_density_and_normalizes(self):
        sampler=TimeSampler(max_sample=2500)
        # 20..40 bin contributes only the grid mass through 25 s.
        expected=.75+.15*501/2000
        self.assertAlmostEqual(sampler.retained_mass,expected)
        self.assertAlmostEqual(sampler.probabilities.sum(),1)
        self.assertEqual(sampler.ticks.max(),2500)
        self.assertTrue(sampler.audit()['passed'])

    def test_native_capacities_and_reproducibility(self):
        for family in ['phasenet_pretrained_frozen','eqt_pretrained_frozen','diting_pretrained_frozen']:
            cap=CAPABILITIES[family]
            sampler=TimeSampler(max_sample=cap.max_elapsed_sample())
            first=[sampler.sample('data','E',3,i,42) for i in range(100)]
            np.random.seed(1);np.random.random(10000)
            resumed=[sampler.sample('data','E',3,i,42) for i in range(100)]
            self.assertEqual(first,resumed)
            self.assertNotEqual(first,[sampler.sample('data','E',4,i,42) for i in range(100)])
            self.assertLessEqual(max(first),min(90,cap.max_elapsed_sample()/100))

    def test_invalid_probability_tables_fail(self):
        for probabilities in [[1,1,1,1,1,1],[-.1,.3,.2,.2,.2,.2],[np.nan]*6]:
            with self.assertRaises(ValueError):
                TimeSampler(probabilities=probabilities)

    def test_model_rng_worker_or_rank_does_not_change_samples(self):
        sampler=TimeSampler()
        baseline={i:sampler.sample('shard','event',2,i,42) for i in range(20)}
        for order in [range(19,-1,-1),list(range(0,20,2))+list(range(1,20,2))]:
            self.assertEqual(baseline,{i:sampler.sample('shard','event',2,i,42) for i in order})
