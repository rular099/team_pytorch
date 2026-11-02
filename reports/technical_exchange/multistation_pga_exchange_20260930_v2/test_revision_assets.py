import unittest
import numpy as np
from prepare_revision_assets import best_f1,_field_metrics


class RevisionTests(unittest.TestCase):
    def test_oracle_cutoff_perfect(self):
        value,cutoff=best_f1([True,False,True,False],[.9,.1,.8,.2])
        self.assertAlmostEqual(value,1);self.assertAlmostEqual(cutoff,.8)

    def test_ties_no_order_advantage(self):
        self.assertEqual(best_f1([True,False],[.5,.5]),best_f1([False,True],[.5,.5]))
        self.assertAlmostEqual(best_f1([True,False],[.5,.5])[0],2/3)

    def test_no_positive_not_discrimination(self):
        self.assertEqual(best_f1([False,False],[.9,.1]),(None,None))

    def test_field_ratio_is_translation_invariant_not_full_shape(self):
        y=np.arange(5,dtype=float)[None,:];valid=np.ones_like(y,bool);k=np.ones(1)
        identity=_field_metrics(y,y+1,valid,k)
        reversed_=_field_metrics(y,y[:,::-1],valid,k)
        self.assertAlmostEqual(identity['range_ratio_mean'],1)
        self.assertAlmostEqual(identity['one_station_pairwise_delta_mae'],0)
        self.assertAlmostEqual(reversed_['range_ratio_mean'],1)
        self.assertGreater(reversed_['one_station_pairwise_delta_mae'],0)

    def test_zero_truth_range_not_divided_by_epsilon(self):
        y=np.zeros((1,5));r=_field_metrics(y,y,np.ones_like(y,bool),np.ones(1))
        self.assertIsNone(r['range_ratio_mean'])


if __name__=='__main__':unittest.main()
