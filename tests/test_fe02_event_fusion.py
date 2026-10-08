"""Synthetic/unit evidence only; not real DiTing loading or HPC performance."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)
from tests.fe01_helpers import config, synthetic_archive, ROOT
import gemini_models as legacy
from fe01.config import load, sha256, write_json
from fe01.model import FE01FullModel, common_state, state_fingerprint
from fe01.extractors import StationFrontend
from scripts.fe02.prepare_runs import make_config
from scripts.fe02.select_run import select
from scripts.fe02.register_diting import register
from fe02.config import validate


def small_params(tmp, variant='A'):
    params=config(tmp)['model_params']
    from fe02.config import VARIANTS
    _,_,mode,post,memory,dims=VARIANTS[variant]
    params.update(pga_readout_mode=mode,pga_use_event_context=post,pga_event_memory=memory,
                  pga_output_mlp_dims=None if dims is None else [8,8])
    return params


def small_model(tmp, variant='A', factory=legacy.build_transformer_model):
    torch.manual_seed(42)
    front=StationFrontend(nn.Conv1d(3,20,1),20,native_dim=20,frozen=True)
    return factory(**small_params(tmp,variant),trace_length=10000,
                   station_waveform_model=front,full_model_class=FE01FullModel)


def inputs(stations=1):
    torch.manual_seed(9)
    wave=torch.randn(2,4,3,10000)
    coords=torch.randn(2,4,3)*.2
    valid=torch.zeros(2,4,dtype=torch.bool);valid[:,:stations]=True
    mask=valid[:,:,None].expand(2,4,10000).clone()
    query=torch.tensor([[[.2,.2,0.],[.6,-.2,0.],[-.1,.4,0.]]]).expand(2,-1,-1).clone()
    qvalid=torch.tensor([[True,True,False]]).expand(2,-1).clone()
    return [wave,coords,valid,query,qvalid,mask]


class ConfigTests(unittest.TestCase):
    def test_all_twenty_configs_and_indices_are_real_diting(self):
        for stage,total in [('formal',15),('pilot',5)]:
            for i in range(total):
                row=select(ROOT,stage,i)
                cfg=load(ROOT/row['config'],expand=False)
                validate(cfg)
                self.assertEqual(cfg['model_family'],'diting_pretrained_frozen')
                self.assertEqual(cfg['variant_id'],['R0','A','B','C','M'][i%5])
                self.assertEqual(cfg['seed'],42+i//5)
        with self.assertRaises(ValueError): select(ROOT,'formal',15)

    def test_typos_conflicts_protocol_drift_fail_fast(self):
        changes=[lambda c:c['model_params'].update(pga_event_memroy=True),
                 lambda c:c.update(event_fusion='invalid'),
                 lambda c:c['model_params'].update(pga_use_event_context=True),
                 lambda c:c['model_params'].update(output_mlp_dims=[64,64]),
                 lambda c:c['training'].update(epochs=6),
                 lambda c:c.update(model_family='diting_random_frozen'),
                 lambda c:c['model_params'].update(pga_distance_bias=True),
                 lambda c:c.update(audit_limits=dict(train_events=16)),
                 lambda c:c['model_params'].update(pga_output_mlp_dims=[64,False])]
        for change in changes:
            cfg=make_config('M');change(cfg)
            with self.assertRaises(ValueError): validate(cfg)

    def test_registry_is_hash_only_immutable_and_torch_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint=Path(tmp)/'unit_not_real.pt';checkpoint.write_bytes(b'not DiTing')
            output=Path(tmp)/'manifest.json';register(checkpoint,output,'synthetic unit fixture')
            manifest=json.loads(output.read_text())['models']['diting']
            self.assertEqual(manifest['files']['weights']['sha256'],sha256(checkpoint))
            self.assertEqual(manifest['device_forward_status'],'NOT_RUN')
            with self.assertRaises(FileExistsError):register(checkpoint,output,'unit')
        result=subprocess.run([sys.executable,'-c',
            'import scripts.fe02.prepare_runs,scripts.fe02.select_run,scripts.fe02.register_diting; import sys; assert "torch" not in sys.modules'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_printer_does_not_submit_or_request_memory_or_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'weights').mkdir()
            (path/'weights/pretrained_manifest.json').write_text('{}')
            env_file=path/'private.env'
            values=dict(FE02_CODE_ROOT=str(ROOT),FE02_OUTPUT_ROOT=str(path/'output'),FE02_DATA_ROOT=str(path/'data'),
                FE02_SPLIT_MANIFEST=str(path/'split.csv'),FE02_WEIGHTS_ROOT=str(path/'weights'),FE02_PARTITION='diting',
                FE02_NODES=4,FE02_DEVICES_PER_NODE=4,FE02_CPUS_PER_TASK=8,FE02_WALLTIME='23:50:00',FE02_GRES_KIND='dcu')
            env_file.write_text('\n'.join('export '+k+'='+shlex.quote(str(v)) for k,v in values.items()))
            bins=path/'bin';bins.mkdir()
            for name in ('sbatch','srun'):
                command=bins/name;command.write_text('#!/bin/sh\ntouch '+shlex.quote(str(path/'called'))+'\nexit 99\n');command.chmod(0o755)
            result=subprocess.run(['bash',str(ROOT/'scripts/fe02/print_submit_commands.sh')],capture_output=True,text=True,
                env={**os.environ,'FE02_ENV_FILE':str(env_file),'PATH':str(bins)+':'+os.environ['PATH']})
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('--array=0-4%1',result.stdout)
            self.assertNotIn('--mem=',result.stdout)
            self.assertNotIn('--allow-test',result.stdout)
            self.assertFalse((path/'called').exists())


class ModelTests(unittest.TestCase):
    def test_default_factory_matches_base_state_and_inference_exactly(self):
        # Execute the frozen baseline source without touching any old worktree.
        source=subprocess.check_output(['git','show','ba4fa7740d9fde5fde87a7b0ae0397037209baf4:gemini_models.py'],cwd=ROOT).decode()
        import types
        module=types.ModuleType('_fe02_baseline_legacy');module.__file__=str(ROOT/'gemini_models.py')
        exec(compile(source,module.__file__,'exec'),module.__dict__)
        with tempfile.TemporaryDirectory() as tmp:
            params=small_params(tmp,'R0');params.pop('pga_event_memory');params.pop('pga_output_mlp_dims')
            def build(factory):
                torch.manual_seed(42)
                front=StationFrontend(nn.Conv1d(3,20,1),20,native_dim=20,frozen=True)
                return factory(**params,trace_length=10000,station_waveform_model=front,full_model_class=FE01FullModel).eval()
            old=build(module.build_transformer_model);new=build(legacy.build_transformer_model)
            self.assertEqual(set(old.state_dict()),set(new.state_dict()))
            self.assertEqual(state_fingerprint(old.state_dict()),state_fingerprint(new.state_dict()))
            new.load_state_dict(old.state_dict(),strict=True)
            for count in (1,3):
                with torch.no_grad():
                    a=old(*inputs(count));b=new(*inputs(count))
                for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))

    def test_pga_only_relu_and_output_dim_and_auxiliary_heads(self):
        with tempfile.TemporaryDirectory() as tmp:
            params=small_params(tmp,'R0');old=small_model(tmp,'R0')
            for distribution in ('mdn','point','gaussian'):
                params.update(pga_output_mlp_dims=[8,5],output_distribution=distribution)
                front=StationFrontend(nn.Conv1d(3,20,1),20,native_dim=20,frozen=True)
                model=legacy.build_transformer_model(**params,trace_length=10000,station_waveform_model=front,full_model_class=FE01FullModel)
                self.assertTrue(any(isinstance(m,nn.ReLU) for m in model.mlp_pga.modules()))
                self.assertEqual([m.out_features for m in model.mlp_pga.modules() if isinstance(m,nn.Linear)],[8,5])
                self.assertEqual(str(model.mlp_mag),str(old.mlp_mag));self.assertEqual(str(model.mlp_loc),str(old.mlp_loc))
                self.assertTrue(all(torch.isfinite(y).all() for y in model(*inputs())))
            for invalid in ([],[0],[True],[-3],[8,1.5],'8'):
                params['pga_output_mlp_dims']=invalid
                with self.assertRaises(ValueError):legacy.build_transformer_model(**params,trace_length=10000,station_waveform_model=front)

    def test_memory_keys_padding_permutation_single_and_multi_gradients(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=small_model(tmp,'M').eval()
            for count in (1,3):
                tensors=inputs(count)
                seen=[]
                hook=model.pga_cross_attention.register_forward_pre_hook(lambda m,x:seen.append([v.detach().clone() for v in x]))
                outputs=model(*tensors);hook.remove()
                self.assertEqual(seen[0][1].shape[1],5)
                self.assertTrue(seen[0][2][:,-1].all())
                self.assertEqual(seen[0][2][:,:4].tolist(),tensors[2].tolist())
                changed=[x.clone() for x in tensors]
                changed[0][:,count:]=torch.randn_like(changed[0][:,count:])*1e5
                changed[1][:,count:]=9999
                for a,b in zip(outputs,model(*changed)):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
                reverse=[x.clone() for x in tensors]
                reverse[3]=reverse[3].flip(1);reverse[4]=reverse[4].flip(1)
                torch.testing.assert_close(outputs[-1],model(*reverse)[-1].flip(1),atol=1e-5,rtol=1e-5)
                model.zero_grad();outputs[-1][:,:2,:,1].sum().backward()
                self.assertIsNotNone(model.pga_event_memory_mapper.weight.grad)
                self.assertGreater(float(model.pga_event_memory_mapper.weight.grad.abs().max()),0)
                self.assertTrue(all(p.grad is None for p in model.waveform_model.encoder.parameters()))

    def test_event_only_skips_cross_attention_and_post_gate_zero_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=small_model(tmp,'C').eval()
            with mock.patch.object(c.pga_cross_attention,'forward',side_effect=AssertionError('C invoked station readout')):
                self.assertTrue(torch.isfinite(c(*inputs())[-1]).all())
            b=small_model(tmp,'B').eval()
            tensors=inputs()
            base=b(*tensors)[-1]
            with torch.no_grad():b.pga_event_context_gate.fill_(1.0)
            self.assertGreater(float((base-b(*tensors)[-1]).detach().abs().max()),1e-6)

    def test_synthetic_boundary_audit_rolls_back_all_parameters_and_rng(self):
        from fe02.model import extra_audit
        from fe01.data import split_catalog
        class Adapter(nn.Module):
            def __init__(self):super().__init__();self.linear=nn.Linear(4,20)
            def forward(self,x,token_mask=None):return self.linear(x.mean(-1))
        for variant in ('R0','A','B','C','M'):
            with tempfile.TemporaryDirectory() as tmp:
                synthetic_archive(tmp);cfg=config(tmp)
                encoder=nn.Sequential(nn.Conv1d(3,4,50,stride=50),nn.BatchNorm1d(4),nn.Dropout(.5))
                front=StationFrontend(encoder,20,frozen=True,diting_adapter=Adapter())
                model=legacy.build_transformer_model(**small_params(tmp,variant),trace_length=10000,
                    station_waveform_model=front,full_model_class=FE01FullModel)
                before=state_fingerprint(model.state_dict());rng=torch.get_rng_state().clone()
                extra_audit(model,cfg,split_catalog(cfg,'train').iloc[0],'cpu',Path(tmp))
                self.assertEqual(before,state_fingerprint(model.state_dict()))
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                report=json.loads((Path(tmp)/'freeze_gradient_audit.json').read_text())
                self.assertEqual(report['status'],'PASS')
                self.assertTrue(report['encoder_parameters_and_buffers_unchanged'])

    def test_no_input_and_memory_conflicts_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=small_model(tmp,'M');tensors=inputs();tensors[2].zero_();tensors[5].zero_()
            with self.assertRaises(ValueError):model(*tensors)
            for key,value in [('pga_use_event_context',True),('no_event_token',True),
                              ('pga_distance_bias',True),('pga_readout_mode','query_no_transformer')]:
                params=small_params(tmp,'M');params[key]=value
                front=StationFrontend(nn.Conv1d(3,20,1),20,native_dim=20,frozen=True)
                with self.assertRaises(ValueError):legacy.build_transformer_model(**params,trace_length=10000,station_waveform_model=front)

    def test_production_pga_shape_and_shared_seed_initialization(self):
        # Mock the 1200M frontend allocation ONLY; actual RT55 downstream at dim1000.
        # No test below may be represented as true DiTing encoder verification.
        from fe02.model import build_model
        class Adapter(nn.Module):
            def __init__(self):super().__init__();self.linear=nn.Linear(4,1000)
            def forward(self,x,token_mask=None):return self.linear(x.mean(-1))
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            seq=nn.Sequential(nn.Conv1d(3,4,50,stride=50),Adapter())
            checkpoint=Path(tmp)/'mock.pth';torch.save({'model_dict':seq.state_dict()},checkpoint)
            manifest=Path(tmp)/'manifest.json'
            write_json(manifest,dict(models=dict(diting=dict(files=dict(weights=dict(path='mock.pth',sha256=sha256(checkpoint)))))))
            initial=None
            for variant in ['R0','A','B','C','M']:
                cfg=make_config(variant);cfg.update(pretrained_manifest=str(manifest))
                sequence=nn.Sequential(nn.Conv1d(3,4,50,stride=50),Adapter())
                with mock.patch('train_light.build_diting_args',return_value=SimpleNamespace(pretrain_method='mae')),mock.patch('gemini_models.get_diting_model',return_value=sequence):
                    model=build_model(cfg)
                unchanged={n:t for n,t in model.state_dict().items() if not n.startswith(('waveform_model.','mlp_pga.','pga_event_context_','pga_event_memory_'))}
                current=state_fingerprint(unchanged)
                if initial is not None:self.assertEqual(initial,current)
                initial=current
                dims=[m.out_features for m in model.mlp_pga.modules() if isinstance(m,nn.Linear)]
                self.assertEqual(dims,[64] if variant=='R0' else [64,64])
                self.assertEqual(any(isinstance(m,nn.ReLU) for m in model.mlp_pga.modules()),variant!='R0')
                del model

    def test_explicit_engine_hooks_update_resume_reload_and_export(self):
        # End-to-end CPU unit fixture with injected tiny encoder and protocol
        # hooks. NOT a real DiTing/HPC audit and NOT a production config.
        from types import SimpleNamespace
        from fe01 import engine as core
        from fe01.model import model_audit
        from fe02.model import extra_audit
        class Adapter(nn.Module):
            def __init__(self):super().__init__();self.linear=nn.Linear(4,20)
            def forward(self,x,token_mask=None):return self.linear(x.mean(-1))
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_archive(tmp,events=9);cfg=config(tmp,'diting_pretrained_frozen')
            cfg.update(fe02=dict(enabled=True,version=1),variant_id='M',event_fusion='memory')
            cfg['model_params'].update(pga_event_memory=True,pga_output_mlp_dims=[8,8])
            cfg['training'].update(epochs=1,max_updates=1,global_batch=3,microbatch=1)
            manifest=Path(tmp)/'unit.json'
            write_json(manifest,dict(models=dict(diting=dict(files=dict(weights=dict(sha256='synthetic_test_only'))))))
            cfg['pretrained_manifest']=str(manifest)
            def builder(c,device='cpu'):
                torch.manual_seed(c['seed'])
                front=StationFrontend(nn.Conv1d(3,4,50,stride=50),20,frozen=True,diting_adapter=Adapter())
                return legacy.build_transformer_model(**c['model_params'],trace_length=10000,
                    station_waveform_model=front,full_model_class=FE01FullModel).to(device)
            experiment=SimpleNamespace(BASE_COMMIT='synthetic_test_only',validate=lambda c:None,
                code_identity=lambda:'synthetic_test_only',build_model=builder,
                model_audit=lambda m,c:model_audit(m),extra_audit=extra_audit,
                check_evaluation_contract=lambda c,p:None)
            destination=Path(tmp)/'audit';core.audit(cfg,destination,experiment=experiment)
            capability=json.loads((destination/'capability_manifest.json').read_text())
            self.assertEqual(set(capability),{'diting_pretrained_frozen'})
            core.train(cfg,destination,experiment=experiment)
            run=Path(tmp)/cfg['run_id'];before=sha256(run/'last.pth')
            core.train(cfg,destination,resume=True,experiment=experiment)
            self.assertEqual(before,sha256(run/'last.pth'))
            with self.assertRaises(FileExistsError):core.train(cfg,destination,experiment=experiment)
            changed=copy.deepcopy(cfg);changed['variant_id']='A'
            with self.assertRaises(ValueError):core.train(changed,destination,resume=True,experiment=experiment)
            core.export_evaluation(copy.deepcopy(cfg),run/'best.pth',run/'evaluation',experiment=experiment)
            frame=__import__('pandas').read_csv(run/'evaluation/predictions.csv.gz')
            self.assertTrue(frame.variant_id.eq('M').all())
            self.assertTrue(frame.event_fusion.eq('memory').all())
            self.assertTrue(frame.split.eq('val').all())
            self.assertTrue(frame.status.eq('supported').all())
            self.assertEqual(json.loads((run/'evaluation/provenance.json').read_text())['variant_id'],'M')

if __name__=='__main__':unittest.main()
