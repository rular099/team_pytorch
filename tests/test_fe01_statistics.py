import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy.special import ndtr

from fe01.metrics import paired_bootstrap,quantile,score_rows
from fe01.spatial import fit_reference,distance_km,field_metrics,make_holdout


class StatisticalTests(unittest.TestCase):
    def test_single_gaussian_mixture_quantiles_and_crps(self):
        w=np.ones((3,1));mu=np.zeros((3,1));sigma=np.ones((3,1))
        np.testing.assert_allclose(quantile(.975,w,mu,sigma),1.95996398454,atol=1e-9)
        scores=score_rows(np.zeros(3),w,mu,sigma)
        np.testing.assert_allclose(scores['crps'],(np.sqrt(2)-1)/np.sqrt(np.pi),atol=1e-12)
        np.testing.assert_allclose(scores['pit'],.5)
        np.testing.assert_allclose(scores['linear_mixture_mean_mps2'],np.exp(.5*np.log(10)**2))
        self.assertTrue((scores['covered95']==1).all())

    def test_bimodal_interval_uses_full_cdf(self):
        w=np.array([[.5,.5]]);mu=np.array([[-3,3]]);sigma=np.ones((1,2))*.1
        self.assertAlmostEqual(float(quantile(.975,w,mu,sigma)[0]),3.164485,places=5)

    def test_cluster_bootstrap_moves_all_times_and_targets_together(self):
        rows=[]
        for event in range(8):
            for time in [1,3]:
                for station in range(4):
                    rows.append(dict(dataset_id='a',event_id=str(event),elapsed_time=time,station_id=str(station),
                        geometry_protocol='normal',truth=0.,prediction=float(event/10),status='supported'))
        a=pd.DataFrame(rows);b=a.copy();b['prediction']+=.1
        result=paired_bootstrap(a,b,draws=500)
        self.assertAlmostEqual(result['delta_mae'],.1)
        self.assertAlmostEqual(result['ci_low'],.1)
        self.assertEqual(result['events'],8)
        with self.assertRaises(ValueError):
            paired_bootstrap(a,b.iloc[1:],draws=10)
        b.loc[0,'status']='failure'
        with self.assertRaises(ValueError):
            paired_bootstrap(a,b,draws=10)

    def test_reference_refuses_val_and_test_fitting(self):
        for split in ['val','test']:
            with self.assertRaises(ValueError):
                fit_reference(pd.DataFrame({'split':[split]}))

    def test_geographic_distance_in_km(self):
        self.assertAlmostEqual(float(distance_km(0,0,0,1)),111.19508,places=4)
        self.assertLess(float(distance_km(60,0,60,1)),56)
