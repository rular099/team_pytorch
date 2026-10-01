import copy
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd
import torch

from tests.fe01_helpers import config,synthetic_archive
from fe01.data import collate,prepare_sample,read_event,split_catalog
from fe01.model import build_model
from fe01.engine import inputs_to


class CausalReplayTests(unittest.TestCase):
    def test_loader_feature_prediction_invariance_from_raw_future(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=synthetic_archive(tmp);cfg=config(tmp)
            row=split_catalog(cfg,'val').iloc[0]
            event=read_event(row,cfg,1,full_for_audit=True)
            sample=prepare_sample(event,cfg,1)
            changed=copy.deepcopy(event);changed['waveform'][...,event['cutout_exclusive']:]=np.nan
            altered=prepare_sample(changed,cfg,1)
            for a,b in zip(sample['inputs'],altered['inputs']):
                self.assertTrue(torch.equal(a,b))
            model=build_model(cfg).eval()
            with torch.no_grad():
                first=model(*inputs_to(sample,'cpu'))
                first_features=model._last_raw_station_emb.clone()
                second=model(*inputs_to(altered,'cpu'))
                self.assertTrue(torch.equal(first_features,model._last_raw_station_emb))
            for a,b in zip(first,second):
                self.assertTrue(torch.equal(a,b))
            # Integration at the HDF5 boundary: prefix reader never reads future.
            with h5py.File(path,'r+') as f:
                f['data'][row.event_id]['waveforms'][:,601:,:]=np.nan
            reread=read_event(row,cfg,1)
            rebuilt=prepare_sample(reread,cfg,1)
            for a,b in zip(sample['inputs'],rebuilt['inputs']):
                self.assertTrue(torch.equal(a,b))

    def test_replay_matches_independent_forwards_and_cache_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_archive(tmp);cfg=config(tmp);row=split_catalog(cfg,'val').iloc[0]
            model=build_model(cfg).eval()
            for elapsed in [1,2,3,5]:
                event=read_event(row,cfg,elapsed)
                sample=prepare_sample(event,cfg,elapsed,replay=True)
                with torch.no_grad():
                    standalone=model(*inputs_to(sample,'cpu'))
                    with model.same_cutoff_query_cache():
                        first=model(*inputs_to(sample,'cpu'))
                        cached=model(*inputs_to(sample,'cpu'))
                        for a,b,c in zip(standalone,first,cached):
                            self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(a,c))
                    self.assertIsNone(model._fe01_query_cache)

    def test_random_replay_uses_stable_station_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_archive(tmp);cfg=config(tmp);cfg['geometry']['random_station_counts']=[2]
            row=split_catalog(cfg,'val').iloc[0]
            selected=[]
            for elapsed in [1,3,5,10,20]:
                result=prepare_sample(read_event(row,cfg,elapsed),cfg,elapsed,'random',replay=True)
                selected.append(result['info']['input_ids'])
            self.assertEqual(selected[-1],selected[-2])

    def test_legacy_exclusive_cutoff_counterexample(self):
        from gemini_util_light import PreloadedEventGenerator
        with tempfile.TemporaryDirectory() as tmp:
            path=synthetic_archive(tmp,events=1,length=11000)
            event_id='20200101000000'
            metadata=pd.DataFrame(dict(EVENT=[event_id]*6,wave_idx=np.arange(6),Magnitude=[4.]*6,
                                       Latitude=[35.]*6,Longitude=[139.]*6,DEPTH=[10.]*6))
            def generate():
                generator=PreloadedEventGenerator(metadata,{'sampling_rate':100},str(path),
                    {'key':'Magnitude','noise_seconds':5},key='Magnitude',windowlen=10000,
                    shuffle=False,max_stations=4,pga_targets=3,pga_from_inactive=True,sampling_rate=100,
                    trigger_based=True,magnitude_resampling=1,scale_metadata=False,
                    emit_waveform_padding_mask=True,deterministic_sampling_seed=42,
                    realtime_training={'enabled':True,'mode':'val','val_times':[1]},
                    realtime_target_sampling={'enabled':True,'input_ratio':.3,'triggered_noninput_ratio':.2,'untriggered_ratio':.5})
                result=generator[0]
                return result,generator.crop_start
            baseline,crop_start=generate()
            cutoff=int(baseline[2]['realtime_current_sample'])+1+crop_start
            with h5py.File(path,'r+') as f:
                f['data'][event_id]['waveforms'][:,cutoff,:]+=100
            altered,_=generate()
            self.assertFalse(torch.equal(baseline[0][0],altered[0][0]))
            # Legacy FullModel, lightweight synthetic encoder. Real 1200M weights
            # are unavailable; this tests the entire legacy preprocessing path.
            import gemini_models as legacy
            class Adapter(torch.nn.Module):
                def forward(self,x,token_mask=None):
                    return x.mean(-1)
            sequence=torch.nn.Sequential(torch.nn.Conv1d(3,20,1),Adapter())
            model=legacy.build_transformer_model(**config(tmp)['model_params'],trace_length=10000,
                                                  station_waveform_model=sequence).eval()
            with torch.no_grad():
                first=model(*[x.unsqueeze(0) for x in baseline[0]])
                features=model._last_raw_station_emb.clone()
                second=model(*[x.unsqueeze(0) for x in altered[0]])
            self.assertFalse(torch.equal(features,model._last_raw_station_emb))
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(first,second)))

    def test_test_split_is_refused_before_waveform_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_archive(tmp);cfg=config(tmp)
            with self.assertRaisesRegex(ValueError,'Held-out test'):
                split_catalog(cfg,'test')
