import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from tests.fe01_helpers import ROOT,config,synthetic_archive
from fe01.data import collate,prepare_sample,read_event,split_catalog
from fe01.engine import inputs_to,loss
from fe01.extractors import (EQTSharedTrunk,OriginalTEAM,PhaseNetBottleneck,
                             StationFrontend,load_seisbench)
from fe01.model import build_model,common_state,shape_and_scale,state_fingerprint


MANIFEST=ROOT/'offline_weights/stead_v2/pretrained_manifest.json'


class ExtractorTests(unittest.TestCase):
    def test_shared_shape_and_amplitude_x10(self):
        raw=torch.randn(2,4,3,1000)
        mask=torch.zeros(2,4,1000,dtype=torch.bool);mask[...,200:700]=True
        first,scale=shape_and_scale(raw,mask)
        second,scaled=shape_and_scale(raw*10,mask)
        torch.testing.assert_close(first,second,atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(scaled[...,:10]-scale[...,:10],torch.ones_like(scale[...,:10]),atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(scaled[...,-1],scale[...,-1])

    def test_masked_fillers_and_all_empty_are_finite(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_archive(tmp);cfg=config(tmp);row=split_catalog(cfg,'val').iloc[0]
            sample=prepare_sample(read_event(row,cfg,1),cfg,1)
            model=build_model(cfg).eval()
            inputs=inputs_to(sample,'cpu')
            changed=[x.clone() for x in inputs]
            changed[0]=torch.where(inputs[5][:,:,None,:],inputs[0],torch.full_like(inputs[0],12345))
            with torch.no_grad():
                first=model(*inputs);second=model(*changed)
                for a,b in zip(first,second):
                    torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-7)
                inputs[2].zero_();inputs[5].zero_();inputs[0].zero_()
                with self.assertRaisesRegex(ValueError,'no_input'):
                    model(*inputs)

    @unittest.skipUnless(MANIFEST.is_file(),'Offline weights absent; see manual downloader')
    def test_real_phasenet_native_intermediate_matches_official_forward(self):
        picker,_=load_seisbench(MANIFEST,'phasenet');picker.eval()
        x=torch.randn(2,3,3001);seen=[]
        handle=picker.down_branch[-1][1].register_forward_hook(lambda m,i,o:seen.append(o.detach()))
        with torch.no_grad():
            picker(x)
        handle.remove()
        with torch.no_grad():
            feature=PhaseNetBottleneck(picker)(x)
        self.assertEqual(len(seen),1)
        torch.testing.assert_close(feature,torch.relu(seen[0]),atol=0,rtol=0)
        self.assertEqual(feature.shape[1],128)

    @unittest.skipUnless(MANIFEST.is_file(),'Offline weights absent; see manual downloader')
    def test_real_eqt_native_shared_trunk_matches_official_forward(self):
        picker,_=load_seisbench(MANIFEST,'eqtransformer');picker.eval()
        x=torch.randn(2,3,6000);seen=[]
        handle=picker.transformer_d.register_forward_hook(lambda m,i,o:seen.append(o[0].detach()))
        with torch.no_grad():
            picker(x)
        handle.remove()
        with torch.no_grad():
            feature=EQTSharedTrunk(picker)(x)
        self.assertEqual(len(seen),1)
        torch.testing.assert_close(feature,seen[0],atol=0,rtol=0)
        self.assertEqual(feature.shape[1],16)

    @unittest.skipUnless(MANIFEST.is_file(),'Offline weights absent; see manual downloader')
    def test_common_initial_state_frozen_encoder_and_checkpoint_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_archive(tmp);row=split_catalog(config(tmp),'val').iloc[0]
            fingerprints=[]
            for family in ['team_original_scratch','phasenet_pretrained_frozen','eqt_pretrained_frozen','amplitude_only','coords_only']:
                cfg=config(tmp,family)
                with mock.patch('gemini_models.get_diting_model',side_effect=AssertionError('Unexpected DiTing construction')):
                    model=build_model(cfg)
                fingerprints.append(state_fingerprint(common_state(model)))
                sample=prepare_sample(read_event(row,cfg,1),cfg,1)
                before=state_fingerprint(model.waveform_model.encoder.state_dict())
                model.train()
                optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
                batch=collate([sample,sample])
                outputs=model(*batch['inputs'])
                value=loss(outputs,batch['labels'],model,cfg,batch['inputs'][4]);value.backward();optimizer.step()
                self.assertTrue(torch.isfinite(value))
                if model.waveform_model.frozen:
                    self.assertFalse(model.waveform_model.encoder.training)
                    self.assertEqual(before,state_fingerprint(model.waveform_model.encoder.state_dict()))
                    self.assertTrue(all(p.grad is None for p in model.waveform_model.encoder.parameters()))
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.waveform_model.projection.parameters()))
                model.eval()
                restored=build_model(cfg).eval();restored.load_state_dict(model.state_dict(),strict=True)
                with torch.no_grad():
                    a=model(*batch['inputs']);b=restored(*batch['inputs'])
                for x,y in zip(a,b):
                    torch.testing.assert_close(x,y,atol=0,rtol=0)
            self.assertEqual(len(set(fingerprints)),1)

    def test_missing_weight_manifest_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=config(tmp,'phasenet_pretrained_frozen');cfg['pretrained_manifest']=str(Path(tmp)/'missing.json')
            with self.assertRaises(FileNotFoundError):
                build_model(cfg)

    def test_team_cnn_axes_flatten_and_synthetic_learning(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=config(tmp);model=build_model(cfg)
            encoder=model.waveform_model.encoder
            self.assertEqual(encoder.conv1.kernel_size,(5,1))
            self.assertEqual(encoder.conv2.kernel_size,(16,3))
            self.assertEqual(encoder(torch.randn(2,3,10000)).shape,(2,20))
            # Tiny learnable continuous task; only observed amplitude/shape differs.
            wave=torch.zeros(4,4,3,10000)
            pattern=torch.sin(torch.arange(800,dtype=torch.float32)*.04)
            for i,amplitude in enumerate([.01,.03,.1,.3]):
                wave[i,0,:,-800:]=amplitude*pattern
            station_valid=torch.zeros(4,4,dtype=torch.bool);station_valid[:,0]=True
            mask=torch.zeros(4,4,10000,dtype=torch.bool);mask[:,0,-800:]=True
            inputs=[wave,torch.zeros(4,4,3),station_valid,torch.ones(4,3,3)*.1,torch.ones(4,3,dtype=torch.bool),mask]
            target=torch.log10(torch.tensor([.01,.03,.1,.3]))[:,None,None].repeat(1,3,1)
            optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.005)
            history=[]
            for _ in range(30):
                optimizer.zero_grad();output=model(*inputs)[-1]
                mixture_mean=(torch.softmax(output[...,0],-1)*output[...,1]).sum(-1)
                # A learning check, not a new FE01 training loss or real-data result.
                value=((mixture_mean-target[...,0])**2).mean()
                history.append(float(value.detach()));value.backward();optimizer.step()
            self.assertLess(np.mean(history[-5:]),np.mean(history[:5])*.75)
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in encoder.parameters()))

    def test_query_chunk_order_and_same_cutoff_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_archive(tmp);cfg=config(tmp);row=split_catalog(cfg,'val').iloc[0]
            model=build_model(cfg).eval();event=read_event(row,cfg,3)
            queries=[3,4,5]
            together=prepare_sample(event,cfg,3,queries=queries)
            reversed_sample=prepare_sample(event,cfg,3,queries=list(reversed(queries)))
            with torch.no_grad():
                a=model(*inputs_to(together,'cpu'))[-1][0,:3]
                b=model(*inputs_to(reversed_sample,'cpu'))[-1][0,:3].flip(0)
                torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
                single=[]
                with model.same_cutoff_query_cache():
                    for q in queries:
                        part=prepare_sample(event,cfg,3,queries=[q])
                        single.append(model(*inputs_to(part,'cpu'))[-1][0,0])
                torch.testing.assert_close(a,torch.stack(single),atol=1e-6,rtol=1e-6)

    def test_legacy_default_factory_is_unchanged(self):
        import gemini_models as legacy
        class Adapter(nn.Module):
            def forward(self,x,token_mask=None):
                return x.mean(-1)
        sequence=nn.Sequential(nn.Conv1d(3,20,1),Adapter())
        with tempfile.TemporaryDirectory() as tmp:
            cfg=config(tmp)
            torch.manual_seed(123)
            with mock.patch.object(legacy,'get_diting_model',return_value=sequence) as factory:
                old=legacy.build_transformer_model(**cfg['model_params'],trace_length=10000)
                factory.assert_called_once()
            torch.manual_seed(123)
            new=legacy.build_transformer_model(**cfg['model_params'],trace_length=10000,station_waveform_model=sequence)
            self.assertEqual(state_fingerprint(old.state_dict()),state_fingerprint(new.state_dict()))
            x=[torch.randn(1,4,3,10000),torch.randn(1,4,3),torch.ones(1,4,dtype=torch.bool),torch.randn(1,3,3),torch.ones(1,3,dtype=torch.bool)]
            old.eval();new.eval()
            with torch.no_grad():
                for a,b in zip(old(*x),new(*x)):
                    torch.testing.assert_close(a,b,atol=0,rtol=0)
