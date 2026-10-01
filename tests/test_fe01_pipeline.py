import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
import pandas as pd
import torch

from tests.fe01_helpers import config,synthetic_archive,ROOT
from fe01.config import fingerprint,load,safe_output,write_json
from fe01.data import read_event,split_catalog
from fe01.engine import audit,causal_audit,check_audit,export_evaluation,load_checkpoint,train
from fe01.analysis import fit_train_reference,summarize_run,training_histograms
from fe01.replay import replay,select_cases
from scripts.fe01.select_run import select


class PipelineTests(unittest.TestCase):
    def test_nonfinite_model_cannot_pass_causal_audit(self):
        from fe01.model import build_model
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_archive(tmp);cfg=config(tmp);model=build_model(cfg)
            with torch.no_grad():
                for parameter in model.parameters(): parameter.fill_(float('nan'))
            with self.assertRaises(FloatingPointError):
                causal_audit(model,cfg,split_catalog(cfg,'train').iloc[0],'cpu')

    def test_synthetic_audit_update_reload_evaluation_replay_and_map(self):
        # One CPU optimizer update on synthetic records, deliberately tiny model.
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_archive(tmp,events=9)
            cfg=config(tmp);cfg['training'].update(epochs=1,max_updates=1,global_batch=3,microbatch=1)
            destination=Path(tmp)/'audit';audit(cfg,destination)
            lock=check_audit(cfg,destination)
            capabilities=json.loads((destination/'capability_manifest.json').read_text())
            self.assertTrue(capabilities['team_original_scratch']['verified_native_forward'])
            self.assertFalse(capabilities['diting_pretrained_frozen']['verified_native_forward'])
            self.assertEqual(len(lock['effective_config']['audited_cohorts']['train']),3)
            train(cfg,destination)
            run=Path(tmp)/cfg['run_id'];self.assertTrue((run/'best.pth').is_file())
            before=(run/'last.pth').read_bytes()
            train(cfg,destination,resume=True)
            self.assertEqual(before,(run/'last.pth').read_bytes())
            changed=copy.deepcopy(cfg);changed['seed']=43
            with self.assertRaises(ValueError): train(changed,destination,resume=True)
            with self.assertRaises(FileExistsError): train(cfg,destination)
            evaluation=run/'val'
            export_evaluation(copy.deepcopy(cfg),run/'best.pth',evaluation)
            frame=pd.read_csv(evaluation/'predictions.csv.gz')
            self.assertTrue(frame.status.eq('supported').all())
            self.assertTrue(frame.split.eq('val').all())
            training_histograms(run,run/'actual_training_time_histograms.csv')
            effective=copy.deepcopy(cfg);load_checkpoint(effective,run/'best.pth','cpu')
            reference=run/'reference.json';fit_train_reference(effective,reference)
            summarize_run(evaluation,run/'analysis',reference)
            cases=run/'cases.json';select_cases(effective,cases,count=3)
            replay(copy.deepcopy(cfg),run/'best.pth',cases,run/'replay',start=1,end=2,spacing_km=100)
            consistency=pd.read_csv(run/'replay/replay_consistency.csv')
            self.assertLessEqual(consistency.max_prediction_delta.max(),1e-6)
            from scripts.fe01.render_maps import render
            render(run/'replay',reference,run/'maps')
            self.assertEqual(len(list((run/'maps').glob('*.gif'))),3)

    def test_spatial_stations_removed_before_train_reference_or_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=synthetic_archive(tmp);cfg=config(tmp);cfg['spatial']['enabled']=True
            table=pd.DataFrame(dict(station_id=[f'S{i}' for i in range(6)],
                spatial_split=['test','train','train','train','val','test'],buffer_excluded=[False]*6))
            table.to_csv(cfg['spatial']['manifest'],index=False)
            with h5py.File(path,'r+') as f:
                g=f['data']['20200101000000']
                g['pga'][0]=np.nan;g['waveforms'][0]=np.nan;g['p_picks'][0]=1
            event=read_event(split_catalog(cfg,'train').iloc[0],cfg,1)
            self.assertEqual(event['ids'].tolist(),['S1','S2','S3'])
            self.assertEqual(event['reference_sample'],700)
            self.assertTrue(np.isfinite(event['pga']).all())
            self.assertTrue(np.isfinite(event['waveform']).all())

    def test_configuration_paths_and_array_mapping(self):
        for i in range(12):
            row=select(ROOT,'formal',i)
            self.assertEqual(int(row['seed']),42+i%3)
            self.assertEqual(row['main'],'True')
        self.assertEqual(select(ROOT,'pilot',3)['model_family'],'eqt_pretrained_frozen')
        with self.assertRaises(ValueError): select(ROOT,'formal',12)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): safe_output(tmp,Path(tmp).parent/'escape')
            write_json(Path(tmp)/'base.json',{'a':{'x':1}})
            write_json(Path(tmp)/'child.json',{'extends':'base.json','a':{'y':2}})
            self.assertEqual(load(Path(tmp)/'child.json'),{'a':{'x':1,'y':2}})

    def test_submit_printer_cannot_invoke_scheduler(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);environment=root/'env.sh'
            values=dict(FE01_CODE_ROOT=str(ROOT),FE01_OUTPUT_ROOT=str(root/'output'),DATA_ROOT=str(root),
                FE01_SPLIT_MANIFEST=str(root/'split.csv'),FE01_WEIGHTS_ROOT=str(root),FE01_PYTHON=str(ROOT/'.venv-fe01/bin/python'),
                FE01_MODULE_LOADS='none',FE01_PARTITION='example',FE01_ACCOUNT='example',FE01_QOS='example',
                FE01_NODES='1',FE01_DEVICES_PER_NODE='1',FE01_CPUS_PER_TASK='1',FE01_MEMORY='4G',FE01_WALLTIME='00:10:00',FE01_GRES_KIND='dcu')
            import shlex
            environment.write_text('\n'.join(f'export {k}={shlex.quote(v)}' for k,v in values.items()))
            bin_dir=root/'bin';bin_dir.mkdir()
            for command in ['sbatch','srun']:
                fake=bin_dir/command;fake.write_text('#!/bin/sh\ntouch "'+str(root/'scheduler_was_called')+'"\nexit 91\n');fake.chmod(0o755)
            result=subprocess.run(['bash',str(ROOT/'scripts/fe01/print_submit_commands.sh')],capture_output=True,text=True,
                env={**os.environ,'FE01_ENV_FILE':str(environment),'PATH':str(bin_dir)+':'+os.environ['PATH']})
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('--array=0-11',result.stdout)
            self.assertFalse((root/'scheduler_was_called').exists())

    def test_weight_hash_corruption_is_fatal(self):
        from fe01.extractors import checked_asset
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'w.pt').write_bytes(b'bad')
            write_json(root/'manifest.json',dict(models=dict(phasenet=dict(files=dict(weights=dict(path='w.pt',sha256='0'*64))))))
            with self.assertRaisesRegex(ValueError,'SHA mismatch'): checked_asset(root/'manifest.json','phasenet')

    def test_diting_unregistered_fails_before_large_model_allocation(self):
        from fe01.model import build_model
        with tempfile.TemporaryDirectory() as tmp:
            cfg=config(tmp,'diting_pretrained_frozen')
            with mock.patch('gemini_models.get_diting_model',side_effect=AssertionError('allocated large encoder')):
                with self.assertRaisesRegex(ValueError,'unregistered'): build_model(cfg)

    def test_small_diting_mock_strict_loading_and_common_initialization(self):
        # Explicit mock; not evidence of a true 1200M checkpoint forward.
        from types import SimpleNamespace
        from fe01.config import sha256
        from fe01.model import build_model,common_state,state_fingerprint
        class Adapter(torch.nn.Module):
            def __init__(self): super().__init__();self.projection=torch.nn.Linear(4,20)
            def forward(self,x,token_mask=None): return self.projection(x.mean(-1))
        with tempfile.TemporaryDirectory() as tmp:
            sequence=torch.nn.Sequential(torch.nn.Conv1d(3,4,1),Adapter())
            checkpoint=Path(tmp)/'mock.pth';torch.save({'model_dict':sequence.state_dict()},checkpoint)
            manifest=Path(tmp)/'mock.json'
            write_json(manifest,dict(models=dict(diting=dict(files=dict(weights=dict(path='mock.pth',sha256=sha256(checkpoint)))))))
            cfg=config(tmp,'diting_pretrained_frozen');cfg['pretrained_manifest']=str(manifest)
            with (mock.patch('train_light.build_diting_args',return_value=SimpleNamespace(pretrain_method='mae')),
                  mock.patch('gemini_models.get_diting_model',return_value=sequence)):
                model=build_model(cfg)
            other=build_model(config(tmp))
            self.assertEqual(state_fingerprint(common_state(model)),state_fingerprint(common_state(other)))
            self.assertTrue(all(not p.requires_grad for p in model.waveform_model.encoder.parameters()))
