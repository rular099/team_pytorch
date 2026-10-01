from dataclasses import replace
import unittest
import numpy as np
from fe01.windows import CAPABILITIES,build_window,clock


class NativeWindowTests(unittest.TestCase):
    def setUp(self):
        self.raw=np.ones((2,3,13000),dtype=np.float32)
        self.mask=np.ones((2,13000),dtype=bool)

    def test_native_prefix_capacity_boundaries(self):
        for family,last in [('phasenet_pretrained_frozen',25),('eqt_pretrained_frozen',54.99),('diting_pretrained_frozen',94.99)]:
            cap=CAPABILITIES[family]
            _,_,info=build_window(self.raw,self.mask,500,last,cap)
            self.assertEqual(info['required_n_samples'],cap.native_n_samples)
            self.assertEqual(info['status'],'supported')
            _,_,info=build_window(self.raw,self.mask,500,last+.01,cap)
            self.assertEqual(info['reason'],'unsupported_history')

    def test_zeros_are_observed_and_storage_gaps_do_not_reduce_capacity(self):
        raw=np.zeros_like(self.raw);mask=np.zeros_like(self.mask)
        mask[:,500:700]=True
        _,valid,info=build_window(raw,mask,500,1,CAPABILITIES['phasenet_pretrained_frozen'])
        self.assertTrue(valid.any())
        _,_,unsupported=build_window(raw,mask,500,30,CAPABILITIES['phasenet_pretrained_frozen'])
        self.assertEqual(unsupported['reason'],'unsupported_history')

    def test_right_alignment_and_no_rounding_into_future(self):
        self.raw[:]=np.arange(13000)
        out,mask,info=build_window(self.raw,self.mask,500,1.009,CAPABILITIES['phasenet_pretrained_frozen'])
        self.assertEqual(info['current_sample'],600)
        self.assertEqual(info['cutout_exclusive'],601)
        self.assertEqual(out[0,0,-1],600)
        self.assertFalse(mask[0,0,0])
        self.assertEqual(clock(500,1.009,100),(600,601))

    def test_rolling_retains_only_recent_native_window(self):
        self.raw[:]=np.arange(13000)
        out,_,info=build_window(self.raw,self.mask,500,90,CAPABILITIES['phasenet_pretrained_frozen'],'native_rolling_v2')
        self.assertEqual(info['history_start_sample'],6500)
        self.assertEqual(out[0,0,0],6500)
        self.assertEqual(out[0,0,-1],9500)

    def test_future_nan_inf_and_missing_components(self):
        cap=CAPABILITIES['phasenet_pretrained_frozen']
        baseline=build_window(self.raw,self.mask,500,1,cap)
        changed=self.raw.copy();changed[...,601:]=np.nan
        result=build_window(changed,self.mask,500,1,cap)
        np.testing.assert_array_equal(result[0],baseline[0])
        component_mask=np.broadcast_to(self.mask[:,None],self.raw.shape).copy()
        component_mask[0,1,500:601]=False
        out,valid,_=build_window(self.raw,component_mask,500,1,cap)
        self.assertEqual(out[0,1,-1],0)
        self.assertFalse(valid[0,1,-1])

    def test_empty_history_and_other_sampling_rate(self):
        cap=replace(CAPABILITIES['phasenet_pretrained_frozen'],sampling_rate=50,native_n_samples=1501)
        out,mask,info=build_window(self.raw,self.mask,250,1,cap)
        self.assertEqual(info['cutout_exclusive'],301)
        self.assertEqual(out.shape[-1],1501)
        _,mask,_=build_window(self.raw,np.zeros_like(self.mask),250,1,cap)
        self.assertFalse(mask.any())
