import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from tools.analyze_v01_validation_closure import pairing, cluster_statistics
from tools.audit_v01_validation_contract import prepare, audit_aa_duplicates, plan_and_sidecar, signature
from tests import test_v01_validation_closure as fixture

ROOT = Path(__file__).resolve().parents[1]


class ClosureAnalysisTests(unittest.TestCase):
    def frames(self):
        left = pd.DataFrame({'event_id': ['a','b'], 'query_sensor_id': ['Q','Q'],
                             'time_s': [1.,1.], 'absolute_cutoff_utc': [11.,21.],
                             'truth': [-2.,-1.], 'prediction': [-1.8,-.8], 'nll': [1.,1.],
                             'prob': [.2,.8], 'sigma': [.2,.2], 'input_count': [1,1], 'target_type': [2,1]})
        left['abs_error'] = abs(left.prediction-left.truth)
        right = left.copy(); right.prediction -= .1; right.nll -= .1
        right['abs_error'] = abs(right.prediction-right.truth)
        return left, right

    def test_outer_pairing_reports_missing_and_duplicate_ids_fail(self):
        left, right = self.frames()
        paired, audit, unmatched = pairing(left, right.iloc[:1])
        self.assertEqual(audit['matched'], 1)
        self.assertEqual(audit['left_only'], 1)
        self.assertEqual(len(unmatched), 1)
        with self.assertRaisesRegex(ValueError, 'one-to-one'):
            pairing(pd.concat([left,left]),right)

    def test_label_or_exact_absolute_cutoff_mismatch_cannot_hide(self):
        left, right = self.frames()
        right.loc[0,'truth'] += .01
        with self.assertRaisesRegex(ValueError, 'label mismatch'):
            pairing(left,right)
        right = left.copy(); right.loc[0,'absolute_cutoff_utc'] += .01
        _, audit, _ = pairing(left,right)
        self.assertEqual(audit['left_only'], 1)
        self.assertEqual(audit['right_only'], 1)

    def test_event_cluster_ci_and_mse_decomposition_from_actual_errors(self):
        left, right = self.frames()
        paired, _, _ = pairing(left,right)
        rows, sufficient = cluster_statistics(paired, draws=5000)
        by_stat = {row['statistic']:row for row in rows}
        self.assertAlmostEqual(by_stat['micro_mae']['estimate'], -.1)
        self.assertAlmostEqual(by_stat['macro_rmse']['estimate'], -.1)
        self.assertAlmostEqual(by_stat['left_mse']['estimate'], .04)
        self.assertAlmostEqual(by_stat['right_bias_squared']['estimate'], .01)
        self.assertAlmostEqual(by_stat['left_centered_variance']['estimate'], 0.)
        second,_ = cluster_statistics(paired, draws=5000)
        self.assertEqual(rows,second)
        self.assertEqual(len(sufficient),2)

    def test_aa_duplicates_with_different_prediction_cannot_deduplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'aa.npz'
            np.savez(path, val_event_id=np.array(['E','E']),
                     val_realtime_requested_elapsed_time=np.array([1.,1.]),
                     val_event_index=np.array([0,1]), val_pga_mu_best=np.array([[.1],[.10001]]))
            result = audit_aa_duplicates(path)
            self.assertFalse(result['can_deduplicate'])
            self.assertEqual(result['decision'],'rerun_one_AA_random')

    def test_default_dry_run_and_aa_exception_never_submit(self):
        script = ROOT/'tools/complete_v01_validation_slurm.sh'
        for extra, expected in (({},5), ({'INCLUDE_AA_RANDOM':'1'},6)):
            env = dict(os.environ, DRY_RUN='1', **extra)
            env.pop('SLURM_JOB_ID', None)
            result = subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stderr.count('--gres=dcu:1'), expected)
            self.assertIn('afterok:dry_v01-closure-audit',result.stderr)
            self.assertNotIn('--mem=',result.stderr)
            self.assertNotIn('train_light.py', result.stdout+result.stderr)

    def test_formal_submission_requires_explicit_confirmation(self):
        result = subprocess.run(['bash',str(ROOT/'tools/complete_v01_validation_slurm.sh')],
                                env={**os.environ,'DRY_RUN':'0','CONFIRM_V01_CLOSURE':'0'},
                                capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('requires',result.stderr)

    def test_login_prepare_uses_only_stdlib_and_refuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'existing'; target.mkdir()
            result = subprocess.run([sys.executable,'-S',str(ROOT/'tools/audit_v01_validation_contract.py'),
                                     'prepare','--run-root',tmp,'--output',str(target)],
                                    capture_output=True,text=True)
            self.assertIn('Refusing existing closure directory',result.stderr)
            self.assertNotIn('ModuleNotFoundError',result.stderr)

    def test_validation_only_rejection_occurs_before_loader_reads(self):
        import eval_checkpoint
        config = {'training_params': {'v01_validation_closure':True, 'generator_params':[{}]}}
        with mock.patch.object(eval_checkpoint.loader,'load_events') as load:
            for splits in (['test'],['train'],['val','test']):
                with self.assertRaisesRegex(ValueError,'validation only'):
                    eval_checkpoint.build_datasets(config,splits=splits)
            load.assert_not_called()

    def test_real_evaluator_export_and_cache_identity_sidecar_roundtrip(self):
        import h5py
        import torch
        from eval_checkpoint import run_inference
        from tools.analyze_v01_validation_closure import load_rows
        from tools.v01_validation_contract import run_closure_inference
        case = fixture.V01ValidationClosureTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        with h5py.File(case.path,'r+') as h5:
            h5['data/E0/waveforms'][0] = h5['data/E1/waveforms'][0]
            h5['data/E0/waveform_valid_mask'][0] = True
        gen = case.generator(True,True)
        gen.no_event_token = True
        gen.max_stations = 2  # force source-first reordering + padded source slot
        class TinyModel(torch.nn.Module):
            output_layout = ['pga']
            def forward(self,*inputs):
                params = torch.zeros((1,3,1,3),dtype=torch.float32)
                params[...,1] = -1.; params[...,2] = .2
                return [params]
        model = TinyModel().eval()
        config = {'training_params':{'pga_loss_weighting':{'threshold':-1.2}}, 'model_params':{}}
        root = Path(case.tmp.name)
        results, summary = run_closure_inference(model,gen,torch.device('cpu'),config,
                                                range(len(gen)),root/'ledger.jsonl',run_inference)
        path = root/'saved.npz'
        np.savez(path, **{'val_'+k:np.asarray(v,dtype=object) for k,v in results.items()})
        records = plan_and_sidecar(gen,path,root,'certified')
        frame = load_rows(path,records)
        self.assertEqual(len(frame),18)
        self.assertEqual(set(frame.query_sensor_id),{'Qneg','Qlate','Qunknown'})
        self.assertTrue(all(r['source_sensor_ids'][0] == 'S' for r in records))
        self.assertEqual(summary['outcomes']['predicted'],6)

    def test_invalid_label_is_explicit_not_a_neighbour_or_model_error(self):
        import h5py
        from tools.v01_validation_contract import V01NoPrediction
        case = fixture.V01ValidationClosureTests(); case.setUp()
        self.addCleanup(case.doCleanups)
        with h5py.File(case.path,'r+') as h5:
            h5['data/E1/pga'][:] = np.nan
        gen = case.generator()
        with self.assertRaises(V01NoPrediction) as ctx:
            gen[3]
        self.assertEqual(ctx.exception.outcome,'invalid_label_or_metadata')

    def test_complete_login_prepare_is_torch_free_and_preserves_original_configs(self):
        from tools.audit_v01_validation_contract import CELLS
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root/'eval_retry1'; old.mkdir()
            source_text = {}
            for cell in CELLS:
                arm = cell.split('__')[0]
                weight = root/('weights_'+arm); weight.mkdir(exist_ok=True)
                (weight/'full_model_last.pth').touch()
                config = {'v01_validation_protocol':'normal', 'training_params':{
                    'weight_path':str(weight), 'data_path':['unchanged.hdf5'],
                    'metadata_cache_dir':'historical-cache'}}
                text = json.dumps(config)
                (old/(cell+'.config.json')).write_text(text)
                source_text[cell] = text
            output = root/'closure'
            result = subprocess.run([sys.executable,'-S',str(ROOT/'tools/audit_v01_validation_contract.py'),
                                     'prepare','--run-root',str(root),'--output',str(output)],
                                    capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            for cell,text in source_text.items():
                self.assertEqual((old/(cell+'.config.json')).read_text(),text)
                prepared = json.loads((output/'configs'/(cell+'.json')).read_text())
                self.assertTrue(prepared['training_params']['v01_validation_closure'])
                self.assertEqual(prepared['training_params']['data_path'],['unchanged.hdf5'])

    def test_real_idx7_tracer_mechanism_with_production_generator_fixture(self):
        import h5py
        from tools.audit_v01_validation_contract import trace_idx7
        case = fixture.V01ValidationClosureTests(); case.setUp()
        self.addCleanup(case.doCleanups)
        with h5py.File(case.path,'r+') as h5:
            for i in range(2,10):
                h5.copy('data/E1','data/E'+str(i))
        rows = []
        for i in range(10):
            part = case.meta[case.meta.EVENT == 'E1'].copy()
            part['EVENT'] = 'E'+str(i)
            rows.append(part)
        case.meta = pd.concat(rows,ignore_index=True)
        result = trace_idx7(case.generator())
        self.assertTrue(result['confirmed'])
        self.assertEqual(result['request']['event_id'],'E7')
        self.assertEqual(result['after_clock'][0]['changed_slots'],0)
        with h5py.File(case.path,'r+') as h5:
            h5['data/E7/waveforms'][:] = 0.
            h5['data/E7/waveform_valid_mask'][:] = False
        abstention = trace_idx7(case.generator())
        self.assertTrue(abstention['confirmed'])
        self.assertEqual(abstention['after']['outcome'],'no_available_source')
        self.assertGreater(abstention['after']['query_count'],0)

    def test_fixed_axis_figure_source_counts_are_reconciled(self):
        from tools.analyze_v01_validation_closure import figures
        left,right = self.frames()
        frames = {arm+'__'+arm+'__'+geometry: frame
                  for geometry in ('normal','random')
                  for arm,frame in (('vfull',left),('vmissing',right))}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            figures(frames,output)
            counts = pd.read_csv(output/'density_bin_counts.csv')
            self.assertTrue((counts.groupby('cell')['count'].sum() == 2).all())
            audit = json.loads((output/'figure_count_audit.json').read_text())
            self.assertTrue(all(r['targets'] == r['binned_targets']+r['outside_fixed_axes'] for r in audit))
            self.assertTrue((output/'pga_density.svg').is_file())


if __name__ == '__main__':
    unittest.main()
