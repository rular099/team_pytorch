"""CPU/stdlib operation checks; genuine pandas parser, no Torch or cluster jobs.

Run with: python -m unittest discover -s tests -p test_fe01_a01_diting_eval_fix.py
"""
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fe01.config import fingerprint, load, sha256, write_json
from fe01_review import requests
from fe01_review.provenance import TRAIN_CODE
from scripts.fe01_a01_ops import eval_pack as ep, eval_revision as revision, eval_status as status
from scripts.fe01_a01_ops.serialized_on import read_serialized_on
from test_fe01_a01_scheduler import ROOT, settings, gates

BASE = 'e1ad3bf547392738ee5e120b7d4d33f8e42d7d30'
TEAM_BASE = 'f29ecdcb4dcc4bdce9eaf560bcbc9453c2b4506e'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def fixture(tmp):
    env, log = settings(tmp)
    project = tmp/'project';(project/'outputs').mkdir(parents=True)
    tool = tmp/'original_upload';tool.mkdir()
    for name, digest in revision.LEGACY_OPERATIONS.items():
        (tool/name).write_bytes(subprocess.check_output(['git', 'show', BASE+':scripts/fe01_a01_ops/'+name], cwd=ROOT))
        assert sha256(tool/name) == digest
    matrix = json.loads((ROOT/'configs/fe01_a01/matrix.json').read_text())['new_training']
    gates(env, range(5))
    with patch.dict(os.environ, env):
        for index, entry in enumerate(matrix):
            cfg = load(ROOT/entry['config']);root = Path(env['A01_OUTPUT_ROOT'])
            audit = root/'audits'/cfg['run_id'];lock = json.loads((audit/'protocol.lock.json').read_text())
            lock.update(code_sha256=TRAIN_CODE, world_size=16, data_identity={'SYNTHETIC': 'DATA'},
                        validation_population_sha256='SYNTHETIC_POPULATION',
                        effective_config=dict(target_normalization={'mean': -1.23, 'std': .42},
                                              audited_cohorts={'train': ['SYNTHETIC'], 'val': ['SYNTHETIC']}))
            lock['sampling']['budget']['updates_per_epoch'] = 212
            write_json(audit/'protocol.lock.json', lock)
            for stage in ('diagnostics', 'pilot'):
                path = root/stage/cfg['run_id']/(stage+'.json');gate = json.loads(path.read_text())
                gate['audit_lock_sha256'] = sha256(audit/'protocol.lock.json');write_json(path, gate)
            run = root/cfg['run_id'];run.mkdir()
            effective = copy.deepcopy(cfg);effective.update(lock['effective_config'])
            write_json(run/'resolved_config.json', effective);write_json(run/'protocol.lock.json', lock)
            for name in ('best.pth', 'last.pth'):(run/name).write_bytes(b'SYNTHETIC_NOT_WEIGHTS')
            (run/'training_curves.csv').write_text('epoch,updates\n'+''.join(f'{i},{i*212}\n' for i in range(1,13)))
        module(tool/'eval_pack.py', 'original_eval_tools').prepare(project, tool, env['A01_ENV_FILE'])
    session = Path((tool/'eval_session.path').read_text().strip())
    fixed = tmp/'diting_fix';fixed.mkdir()
    for name in revision.FILES+('serialized_on.py', 'submit_diting_eval_fix.sh', 'check_eval_status.sh', 'eval_status.py'):
        shutil.copy2(ROOT/'scripts/fe01_a01_ops'/name, fixed/name)
    env.update(A01_EVAL_SESSION=str(session), A01_CONFIG_PYTHON=sys.executable)
    return env, log, session, fixed


def old_TEAM_revision(session, tmp):
    tool = tmp/'historical_TEAM';tool.mkdir()
    for name in revision.FILES:
        (tool/name).write_bytes(subprocess.check_output(['git', 'show', TEAM_BASE+':scripts/fe01_a01_ops/'+name], cwd=ROOT))
    old = module(tool/'eval_revision.py', 'actual_frozen_TEAM_revision')
    return old.register(session, tool)


def seal(session, index, execution=None):
    plan = json.loads((session/'plan.json').read_text());run_id = plan['runs'][index]['run_id']
    root = session/'results'/f'index{index}';root.mkdir(parents=True)
    for label in ('selected', 'epoch12'):
        for run in [run_id]+([run_id+'_reused_ON'] if index < 3 else []):
            for name in ('predictions.csv.gz', 'support.csv', 'population.json', 'scores.csv', 'fields.csv.gz', 'checkpoint_identity.json'):
                path = root/'evaluation'/run/label/name;path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}' if name.endswith('.json') else 'SYNTHETIC\n')
        if index != 3:
            for name in ('comparison.json', 'paired_event_bootstrap.csv', 'ON_scores.csv', 'OFF_scores.csv'):
                path = root/'comparisons'/label/name;path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{"status":"PASS"}' if name.endswith('.json') else 'SYNTHETIC\n')
        if index == 4 and execution:
            on_id = plan['runs'][3]['run_id']
            digest = sha256(session/'results/index3/evaluation'/on_id/label/'predictions.csv.gz')
            write_json(root/'paired_input_checks'/f'{label}.json', dict(status='PASS', parser=dict(engine='c', float_precision='round_trip'),
                       source_expected_sha256=digest, source_actual_sha256=digest))
    record = dict(status='PASS', index=index, split='validation', plan_sha256=sha256(session/'plan.json'),
                  source_core_sha256=plan['source_core_sha256'], execution_operations_sha256=plan['operations_sha256'],
                  files_sha256={str(p.relative_to(root)):sha256(p) for p in root.rglob('*') if p.is_file()})
    if execution:
        record.update(execution_operations_sha256=json.loads((execution/'revision.json').read_text())['operations_sha256'],
                      operations_revision=dict(path=str((execution/'revision.json').relative_to(session)), sha256=sha256(execution/'revision.json')))
    write_json(root/'EVAL_COMPLETE.json', record)
    return root


def snapshot(root):
    return {str(p.relative_to(root)):sha256(p) for p in root.rglob('*') if p.is_file()}


def canonical_frame():
    # These decimals arise from the existing original-read -> to_csv sequence.
    values = [-2.6145308017730717, -1.9146628379821775]
    return pd.DataFrame([dict(dataset_id='2000', event_id='00001', station_id=f'S{i:02}', elapsed_time=1.,
        geometry_protocol='normal', target_role='untriggered_noninput', input_ids='["S0"]', truth=value,
        requested_decision_sample=600., current_sample=600, cutout_exclusive=601, history_start_sample=0,
        history_end_sample=600, latest_received_sample=600, input_count=1, split='val', units='log10(m/s^2)',
        status='supported', prediction=-1., nll=2., mdn_logits='[0.0, -1000.0]') for i,value in enumerate(values)])


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='a01_serialized_on_');self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name);self.frame = canonical_frame()
        self.path = self.root/'predictions.csv.gz';self.frame.to_csv(self.path, index=False)
        self.expected = requests.population_audit(self.frame, production=False)
        self.original_audit = requests.population_audit

    def strict_synthetic(self, frame, production=True):
        actual = self.original_audit(frame, production=False)
        if production:
            if actual != self.expected:raise ValueError('Locked population/labels/input/clock mismatch')
        return actual

    def test_default_parser_changes_identity_and_round_trip_exactly_restores_all_values(self):
        before = self.path.read_bytes()
        default = requests.read_frame(self.path)
        self.assertFalse(default.truth.equals(self.frame.truth))
        with self.assertRaisesRegex(ValueError, 'Locked'):self.strict_synthetic(default)
        with patch.object(requests, 'population_audit', self.strict_synthetic):
            result = read_serialized_on(self.path, sha256(self.path), self.root/'check.json')
        pd.testing.assert_frame_equal(result, self.frame)
        self.assertEqual(self.path.read_bytes(), before)
        check = json.loads((self.root/'check.json').read_text())
        self.assertEqual(check['status'], 'PASS');self.assertEqual(check['default_parser_identity_differences']['truth'], 2)

    def test_real_identity_changes_missing_keys_duplicates_and_units_are_all_rejected(self):
        for kind in ('truth', 'input_ids', 'current_sample', 'cutout_exclusive', 'event_id', 'missing', 'duplicate', 'units', 'split'):
            with self.subTest(kind=kind):
                changed = self.frame.copy()
                if kind == 'truth':changed.loc[0,kind] += .001
                elif kind == 'input_ids':changed.loc[0,kind] = '["OTHER"]'
                elif kind == 'current_sample':changed.loc[0,kind] += 1
                elif kind == 'cutout_exclusive':changed.loc[0,kind] += 1
                elif kind == 'event_id':changed.loc[0,kind] = 'OTHER'
                elif kind == 'missing':changed = changed.iloc[:1]
                elif kind == 'duplicate':changed = pd.concat([changed, changed.iloc[:1]])
                elif kind == 'units':changed.loc[0,kind] = 'g'
                else:changed.loc[0,kind] = 'test'
                changed.to_csv(self.path, index=False)
                report = self.root/(kind+'.json')
                with patch.object(requests, 'population_audit', self.strict_synthetic):
                    with self.assertRaisesRegex(ValueError, 'details'):read_serialized_on(self.path, sha256(self.path), report)
                self.assertEqual(json.loads(report.read_text())['status'], 'FAIL')

    def test_sealed_bytes_must_match_before_any_population_acceptance(self):
        with patch.object(requests, 'population_audit') as audit:
            with self.assertRaisesRegex(ValueError, 'details'):read_serialized_on(self.path, '0'*64, self.root/'badsha.json')
            audit.assert_not_called()
        self.assertIn('bytes changed', json.loads((self.root/'badsha.json').read_text())['error'])

    def test_actual_production_gate_is_still_called_without_waivers(self):
        with self.assertRaisesRegex(ValueError, 'details'):read_serialized_on(self.path, sha256(self.path), self.root/'production.json')
        check = json.loads((self.root/'production.json').read_text())
        self.assertEqual(check['status'], 'FAIL')
        self.assertEqual(check['expected_population_sha256'], requests.EXPECTED_POP)
        self.assertEqual(check['observed_population']['rows'], 2)


class OperationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='a01_diting_fix_');self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.env, self.log, self.session, self.tool = fixture(self.root)
        self.environment = patch.dict(os.environ, self.env);self.environment.start();self.addCleanup(self.environment.stop)
        self.team = old_TEAM_revision(self.session, self.root)
        seal(self.session, 0, self.team)
        for index in (1,2,3):seal(self.session, index)

    def register(self):
        return revision.register(self.session, self.tool, revision.DITING_PURPOSE)

    def test_revision_accepts_actual_frozen_TEAM_and_restricts_new_evaluation_to4(self):
        before = snapshot(self.session);directory = self.register()
        self.assertEqual(revision.validate_revision(self.session, directory/'revision.json', index=4)['allowed_eval_indices'], [4])
        revision.validate_revision(self.session, self.team/'revision.json', index=0)
        for index in range(4):
            with self.assertRaisesRegex(ValueError, 'only index4'):revision.validate_revision(self.session, directory/'revision.json', index=index)
            ep.result_gate(self.session, index)
        after = snapshot(self.session)
        self.assertTrue(all(after[name] == value for name,value in before.items()))

    def test_manual_default_submits_only4_and_preserves_all_four_completed_results(self):
        before = snapshot(self.session)
        deny = self.root/'deny';deny.mkdir();(deny/'torch.py').write_text('raise RuntimeError("LOGIN_IMPORTED_TORCH")\n')
        result = subprocess.run(['bash', str(self.tool/'submit_diting_eval_fix.sh')], env={**self.env, 'PYTHONPATH':str(deny)}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.log.read_text();self.assertEqual(calls.count('sbatch '), 1)
        self.assertIn('--array=4%4', calls);self.assertIn('--gres=dcu:1', calls);self.assertIn('_job_eval', calls)
        self.assertIn('eval_diting_off\t4\t12345', (self.session/'jobs.tsv').read_text())
        after = snapshot(self.session);self.assertTrue(all(after[name] == value for name,value in before.items()))

    def test_missing_ON_or_unsupported_stage_cannot_submit4(self):
        directory = self.register();env = {**self.env, 'A01_EVAL_REVISION_DIRECTORY':str(directory)}
        with patch.dict(os.environ, env):
            with self.assertRaisesRegex(ValueError, 'only eval_diting_off or pack'):ep.preflight(self.session, 'eval_first')
        target = self.session/'results/index3/EVAL_COMPLETE.json';target.rename(target.with_suffix('.preserved'))
        result = subprocess.run(['bash', str(self.tool/'submit_diting_eval_fix.sh')], env=self.env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0);self.assertFalse(self.log.exists())

    def test_new_execution_seal_requires_both_parser_reports_and_authenticates_sources(self):
        directory = self.register();root = seal(self.session, 4, directory)
        ep.result_gate(self.session, 4)
        path = root/'paired_input_checks/epoch12.json';path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):ep.result_gate(self.session, 4)

    def test_separate_CPU_pack_accepts_TEAM_old_and_DiTing_new_without_torch_or_resealing(self):
        directory = self.register();seal(self.session, 4, directory);before = snapshot(self.session/'results')
        deny = self.root/'deny';deny.mkdir();(deny/'torch.py').write_text('raise RuntimeError("PACK_IMPORTED_TORCH")\n')
        env = {**self.env, 'A01_EVAL_REVISION_DIRECTORY':str(directory), 'PYTHONPATH':str(deny)}
        result = subprocess.run([sys.executable, '-B', str(directory/'eval_pack.py'), 'pack', '--session', str(self.session)],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(snapshot(self.session/'results'), before)
        with tarfile.open(self.session/'review.tar.gz') as archive:
            names = archive.getnames()
            self.assertEqual(len([name for name in names if name.startswith('evaluation_execution/') and name.endswith('revision.json')]), 2)
            self.assertIn('results/index4/paired_input_checks/epoch12.json', names)
            self.assertFalse(any(name.endswith(('.pth', '.env')) for name in names))
        self.assertFalse(self.log.exists())

    def test_old_TEAM_pack_cannot_mistake_new4_for_old_execution(self):
        directory = self.register();seal(self.session, 4, directory)
        frozen = module(self.team/'eval_pack.py', 'frozen_TEAM_pack')
        with self.assertRaisesRegex(ValueError, 'Unsupported execution revision policy'):frozen.result_gate(self.session, 4)

    def test_tampered_reader_or_policy_blocks_revision_validation(self):
        directory = self.register();path = directory/'serialized_on.py';path.chmod(0o600);path.write_text(path.read_text()+'\n# TAMPER\n')
        with self.assertRaisesRegex(ValueError, 'Revision source changed'):revision.validate_revision(self.session, directory/'revision.json')

    def test_status_is_readonly_and_distinguishes_completed_artifacts_from_failed4(self):
        (self.session/'jobs.tsv').write_text('eval_first\t0,1,2,3\t100\neval_team\t0\t200\neval_diting_off\t4\t300\n')
        before = snapshot(self.session);lines = []
        value = status.inspect(self.session, ep=ep, query=lambda _:({'300_4':dict(state='FAILED')}, []), emit=lines.append)
        self.assertEqual([row['status'] for row in value['rows']], ['COMPLETE']*4+['FAILED'])
        self.assertFalse(value['pack_ready']);self.assertEqual(snapshot(self.session), before)
        self.assertFalse(self.log.exists())

    def test_status_and_pack_submission_entrypoints_are_distinct_and_CPU_pack_has_no_gres(self):
        directory = self.register();seal(self.session, 4, directory)
        result = subprocess.run(['bash', str(self.tool/'submit_diting_eval_fix.sh'), 'status'], env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr);self.assertIn('PACK_READY=YES', result.stdout)
        self.assertFalse(self.log.exists())
        result = subprocess.run(['bash', str(self.tool/'submit_diting_eval_fix.sh'), 'pack'], env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.log.read_text();self.assertEqual(calls.count('sbatch '), 1)
        self.assertNotIn('--gres', calls);self.assertIn('_job_pack', calls)

    def exercise_evaluate_loop(self, bad_ON=False):
        directory = self.register();plan = ep.load_plan(self.session)
        frame = canonical_frame();on_root = self.session/'results/index3'
        for label in ('selected', 'epoch12'):
            path = on_root/'evaluation'/plan['runs'][3]['run_id']/label/'predictions.csv.gz'
            saved = frame.copy()
            if bad_ON:saved.loc[0,'truth'] += .001
            saved.to_csv(path, index=False)
        on_gate = json.loads((on_root/'EVAL_COMPLETE.json').read_text())
        on_gate['files_sha256'] = {str(p.relative_to(on_root)):sha256(p) for p in on_root.rglob('*') if p.is_file() and p.name != 'EVAL_COMPLETE.json'}
        write_json(on_root/'EVAL_COMPLETE.json', on_gate)
        row = plan['runs'][4];lock = json.loads((Path(row['audit_dir'])/'protocol.lock.json').read_text())
        original_audit = requests.population_audit;expected = original_audit(frame, production=False)
        def synthetic_audit(actual, production=True):
            audit = original_audit(actual, production=False)
            if production and audit != expected:raise ValueError('Locked population/labels/input/clock mismatch')
            return audit
        calls = []
        def metadata(path, row, lock):
            epoch = 12 if Path(path).name == 'last.pth' else 7
            return dict(path=str(path), epoch=epoch, updates=epoch*212, best_epoch=7, best_selection_score=.1,
                        curve_records=[dict(epoch=i, validation_selection=abs(i-7)+.1) for i in range(1,13)])
        engine = types.ModuleType('fe01_a01.engine');engine.check_audit = lambda *args:lock
        evaluation = types.ModuleType('fe01_a01.evaluation')
        def export(cfg, audit, checkpoint, epoch, target, device):
            label = target.name;check = target.parents[2]/'paired_input_checks'/(label+'.json')
            self.assertEqual(json.loads(check.read_text())['status'], 'PASS')
            calls.append(('export', epoch))
            target.mkdir(parents=True)
            frame.to_csv(target/'predictions.csv.gz', index=False)
            for name in ('support.csv','population.json','scores.csv','fields.csv.gz'):
                (target/name).write_text('{}' if name.endswith('.json') else 'SYNTHETIC_EXPORT\n')
            return frame.copy()
        def compare(on, off, target):
            pd.testing.assert_frame_equal(on, off);calls.append(('compare', target.name))
            target.mkdir(parents=True)
            for name in ('paired_event_bootstrap.csv', 'ON_scores.csv', 'OFF_scores.csv'):
                (target/name).write_text('SYNTHETIC_COMPARE\n')
            write_json(target/'comparison.json', dict(status='PASS'))
        evaluation.export = export;evaluation.compare = compare;evaluation.export_old_ON = None
        before = snapshot(self.session/'results')
        with patch.dict(sys.modules, {'fe01_a01.engine':engine, 'fe01_a01.evaluation':evaluation}), \
                patch.object(ep, 'checkpoint_metadata', metadata), \
                patch.object(requests, 'population_audit', synthetic_audit), \
                patch.dict(os.environ, {'A01_EVAL_REVISION_DIRECTORY':str(directory)}):
            if bad_ON:
                with self.assertRaisesRegex(ValueError, 'Serialized ON identity check failed'):
                    ep.evaluate_one(self.session, 4, device='cpu')
            else:
                ep.evaluate_one(self.session, 4, device='cpu');ep.result_gate(self.session, 4)
        after = snapshot(self.session/'results')
        self.assertTrue(all(after[name] == digest for name,digest in before.items()))
        return calls, directory

    def test_evaluate_loop_checks_ON_before_each_export_and_seals_both_epochs_with_new_identity(self):
        calls, directory = self.exercise_evaluate_loop()
        self.assertEqual(calls, [('export',7),('compare','selected'),('export',12),('compare','epoch12')])
        gate = json.loads((self.session/'results/index4/EVAL_COMPLETE.json').read_text())
        self.assertEqual(gate['operations_revision']['sha256'], sha256(directory/'revision.json'))
        self.assertFalse(self.log.exists())

    def test_real_population_difference_stops_before_any_OFF_export_and_preserves_failed_attempt(self):
        calls, _ = self.exercise_evaluate_loop(bad_ON=True)
        self.assertEqual(calls, [])
        self.assertFalse((self.session/'results/index4').exists())
        reports = list((self.session/'attempts').glob('eval_index4_*/paired_input_checks/selected.json'))
        self.assertEqual(len(reports), 1);self.assertEqual(json.loads(reports[0].read_text())['status'], 'FAIL')


class SchedulerTests(unittest.TestCase):
    def test_old_sacct_without_array_option_reports_failed_task_and_preserves_live_queue_priority(self):
        calls = []
        def run(command, **kwargs):
            calls.append(command)
            if command[0] == 'sacct':
                if '--array' in command:return subprocess.CompletedProcess(command, 1, '', "sacct: unrecognized option '--array'")
                output = '100_0|COMPLETED|0:0\n300_4|FAILED|1:0\n400_4|FAILED|1:0\n'
            else:output = '400_4|RUNNING\n'
            return subprocess.CompletedProcess(command, 0, output, '')
        with patch.object(status.subprocess, 'run', run):
            states, warnings = status.scheduler({0:dict(job='100', task='100_0'),4:dict(job='300', task='300_4')})
        self.assertFalse(warnings);self.assertNotIn('--array', calls[0]);self.assertIn('-r', calls[1])
        self.assertEqual(states['300_4']['state'], 'FAILED');self.assertEqual(states['400_4']['state'], 'RUNNING')

    def test_unavailable_or_parent_only_accounting_never_means_task_complete(self):
        def run(command, **kwargs):return subprocess.CompletedProcess(command, 0, '300|COMPLETED|0:0\n', '')
        with patch.object(status.subprocess, 'run', run):
            states, _ = status.scheduler({4:dict(job='300', task='300_4')})
        self.assertEqual(status.pending_status(dict(task='300_4'), states), 'UNKNOWN')


if __name__ == '__main__':
    unittest.main()
