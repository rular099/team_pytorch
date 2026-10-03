import errno
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np

import loader_light
from tools.launcher_config import load_config_file


ROOT = Path(__file__).resolve().parents[1]
RECOVER = ROOT / 'tools/recover_v01_prep_padding_controls_slurm.sh'


def config_environment(root):
    return {
        name: str(root / name.lower()) for name in (
            'V01_RUN_ROOT', 'V01_CACHE_ROOT', 'V01_VFULL_WEIGHT_PATH',
            'V01_VMISSING_WEIGHT_PATH', 'V01_APAIR_WEIGHT_PATH',
            'RT55_EP32_CHECKPOINT', 'FROZEN_SPLIT_MANIFEST',
            'JAPAN_FULL_DATA_ROOT', 'JAPAN_FULL_WEIGHT_PATH', 'RT56_WEIGHT_PATH',
        )
    }


class LauncherConfigTests(unittest.TestCase):
    def test_real_configs_match_training_loader_including_rt55(self):
        from train_light import load_config_file as training_loader
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, config_environment(Path(tmp))):
            for name in (
                'transformer_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_chaosuan.json',
                'v01_velocity_full.json', 'v01_velocity_missing.json', 'v01_acc_pair.json',
            ):
                path = ROOT / 'pga_configs' / name
                self.assertEqual(load_config_file(path), training_loader(str(path)))
            for arm in ('velocity_full', 'velocity_missing', 'acc_pair'):
                os.environ['V01_ARM_CONFIG'] = str(ROOT / 'pga_configs' / ('v01_' + arm + '.json'))
                for protocol in ('normal', 'random'):
                    path = ROOT / 'pga_configs' / ('v01_validation_' + protocol + '.json')
                    self.assertEqual(load_config_file(path), training_loader(str(path)))

    def test_json_cli_runs_without_site_packages_or_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'base.json'
            child = root / 'child.json'
            base.write_text(json.dumps({'training_params': {
                'weight_path': '${UNSET_OBSOLETE_PATH}',
                'single_station_pretrain': {'enabled': True},
            }}))
            child.write_text(json.dumps({'extends': 'base.json', 'training_params': {'weight_path': '/safe/weights'}}))
            for field, expected in [('weight-path', '/safe/weights'), ('single-station-enabled', '1')]:
                result = subprocess.run(
                    [sys.executable, '-S', str(ROOT / 'tools/launcher_config.py'), str(child), field],
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), expected)

    def test_cycles_and_unresolved_values_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            path.write_text(json.dumps({'extends': 'config.json'}))
            with self.assertRaisesRegex(ValueError, 'cycle'):
                load_config_file(path)
            path.write_text(json.dumps({'training_params': {'weight_path': '${UNSET_V01_TEST_VALUE}'}}))
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, 'Unresolved environment'):
                    load_config_file(path)

    def test_resolved_job_config_changes_only_cache_location(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, config_environment(Path(tmp))):
            root = Path(tmp)
            os.environ['V01_ARM_CONFIG'] = str(ROOT / 'pga_configs/v01_velocity_missing.json')
            path = ROOT / 'pga_configs/v01_validation_random.json'
            expected = load_config_file(path)
            output = root / 'resolved/config.json'
            cache = str(root / 'isolated-cache')
            result = subprocess.run([
                sys.executable, '-S', str(ROOT / 'tools/launcher_config.py'), str(path), 'resolved-json',
                '--metadata-cache-dir', cache, '--output', str(output),
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            expected['training_params']['metadata_cache_dir'] = cache
            self.assertEqual(json.loads(output.read_text()), expected)
            self.assertNotIn('extends', expected)


class MetadataCachePublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'event.hdf5'
        self.cache = self.root / 'station.csv'
        with h5py.File(self.source, 'w') as h5:
            metadata = h5.create_group('metadata')
            metadata.create_dataset('sampling_rate', data=100)
            table = metadata.create_group('station_metadata')
            table.create_dataset('EVENT', data=np.array(['E1', 'E1'], dtype='S8'))
            table.create_dataset('wave_idx', data=np.arange(2))
            table.create_dataset('Magnitude', data=np.array([4.0, 4.0]))
            h5.create_group('data/E1').create_dataset('waveforms', data=np.zeros((2, 10, 3)))

    def test_cluster_eexist_accepts_identical_completed_cache(self):
        loader_light.build_event_metadata(str(self.source), str(self.cache))
        original = self.cache.read_bytes()
        with mock.patch.object(loader_light.os, 'replace', side_effect=FileExistsError(errno.EEXIST, 'File exists')):
            result = loader_light.build_event_metadata(str(self.source), str(self.cache))
        self.assertEqual(len(result), 2)
        self.assertEqual(self.cache.read_bytes(), original)
        self.assertFalse(list(self.root.glob('.*.tmp')))

    def test_conflicting_cache_is_not_silently_accepted_or_overwritten(self):
        self.cache.write_text('EVENT,wave_idx,Magnitude\nwrong,0,9\n')
        original = self.cache.read_bytes()
        with mock.patch.object(loader_light.os, 'replace', side_effect=FileExistsError(errno.EEXIST, 'File exists')):
            with self.assertRaisesRegex(RuntimeError, 'differs from generated CSV'):
                loader_light.build_event_metadata(str(self.source), str(self.cache))
        self.assertEqual(self.cache.read_bytes(), original)
        self.assertFalse(list(self.root.glob('.*.tmp')))

    def test_unrelated_publication_failure_propagates(self):
        with mock.patch.object(loader_light.os, 'replace', side_effect=PermissionError('denied')):
            with self.assertRaises(PermissionError):
                loader_light.build_event_metadata(str(self.source), str(self.cache))
        self.assertFalse(self.cache.exists())
        self.assertFalse(list(self.root.glob('.*.tmp')))


class RecoverySubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / 'v01_run'
        self.run.mkdir()
        self.env = os.environ.copy()
        for key in ('EXPECTED_SOURCE_MANIFEST_SHA256', 'EXPECTED_GIT_COMMIT', 'SLURM_JOB_ID', 'V01_VFULL_WEIGHT_PATH',
                    'V01_VMISSING_WEIGHT_PATH', 'V01_APAIR_WEIGHT_PATH', 'V01_EVAL_ROOT', 'V01_REPORT_ROOT',
                    'RESUME_V01', 'ALLOW_EXISTING_EVAL', 'RESET_WEIGHT_PATH', 'V01_RECOVERY_TAG',
                    'V01_SUBMISSION_GUARD_DIR', 'ACTION'):
            self.env.pop(key, None)
        self.env.update({
            'WORKDIR': str(ROOT), 'V01_RUN_ROOT': str(self.run),
            'V01_CACHE_ROOT': str(self.run / 'derived_cache'),
            'SOURCE_IDENTITY_MODE': 'uploaded_sha256', 'DRY_RUN': '1',
            'ACC_DATA_ROOT': str(self.root / 'acc'), 'VELOCITY_DATA_ROOT': str(self.root / 'velocity'),
            'FROZEN_SPLIT_MANIFEST': str(self.root / 'split.csv'),
            'RT55_EP32_CHECKPOINT': str(self.root / 'parent.pth'),
        })
        (self.root / 'acc').mkdir()
        (self.root / 'velocity').mkdir()
        (self.root / 'split.csv').write_text('event_id,split\nE1,train\n')
        (self.root / 'parent.pth').write_bytes(b'fixture-only-not-a-checkpoint')
        for arm in ('vmissing', 'apair'):
            weight = self.run / ('weights_' + arm)
            weight.mkdir()
            (weight / 'full_model_last.pth').write_bytes(b'fixture-only-not-a-checkpoint')
            (weight / 'config.json').write_text('{}')
        cache = self.run / 'derived_cache'
        cache.mkdir()
        for name in ('protocol_lock.json', 'preflight_summary.json'):
            (cache / name).write_text('{}')
        for year in range(2004, 2025):
            shard = cache / str(year)
            shard.mkdir()
            for kind in ('velocity', 'acc_pair'):
                (shard / f'japan_{year}_v01_{kind}.hdf5').write_bytes(b'fixture-only')

    def call(self, **overrides):
        return subprocess.run(['bash', str(RECOVER)], cwd=ROOT,
                              env=dict(self.env, **overrides), capture_output=True, text=True)

    def formal(self):
        dry = self.call()
        self.assertEqual(dry.returncode, 0, dry.stderr)
        digest = re.search(r'source_manifest_sha256=([0-9a-f]{64})', dry.stdout).group(1)
        bin_dir = self.root / 'bin'
        bin_dir.mkdir(exist_ok=True)
        sbatch = bin_dir / 'sbatch'
        sbatch.write_text(
            '#!' + sys.executable + '\nimport json, os, sys\n'
            'from pathlib import Path\np=Path(os.environ["MOCK_SBATCH_CALLS"])\n'
            'rows=p.read_text().splitlines() if p.exists() else []\n'
            'with p.open("a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n'
            'print(90001+len(rows))\n'
        )
        sbatch.chmod(0o755)
        calls = self.root / 'calls.jsonl'
        result = self.call(DRY_RUN='0', CONFIRM_V01='1', EXPECTED_SOURCE_MANIFEST_SHA256=digest,
                           PATH=str(bin_dir) + os.pathsep + self.env['PATH'], MOCK_SBATCH_CALLS=str(calls))
        rows = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
        return result, rows

    def test_recovery_has_only_one_training_ten_validation_and_one_analysis(self):
        result, calls = self.formal()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 12)
        names = [next(arg.split('=', 1)[1] for arg in call if arg.startswith('--job-name=')) for call in calls]
        self.assertEqual(names[0], 'v01-vfull-train')
        self.assertNotIn('v01-preflight', names)
        self.assertNotIn('v01-vmissing-train', names)
        self.assertNotIn('v01-apair-train', names)
        self.assertEqual(names[-1], 'v01-analyze')
        self.assertFalse(any('--resume_full_model' in arg for call in calls for arg in call))
        wraps = [next(arg.split('=', 1)[1] for arg in call if arg.startswith('--wrap=')) for call in calls]
        for command in wraps[1:-1]:
            self.assertIn('--splits val', command)
            self.assertNotIn('--splits test', command)
        for call, name in zip(calls[1:-1], names[1:-1]):
            deps = [arg for arg in call if arg.startswith('--dependency=')]
            self.assertEqual(deps, ['--dependency=afterok:90001'] if name.startswith('v01-vfull-') else [])
        self.assertIn('--dependency=afterok:' + ':'.join(str(i) for i in range(90002, 90012)), calls[-1])
        cache_dirs = [shlex.split(command)[shlex.split(command).index('--metadata-cache-dir') + 1] for command in wraps[:-1]]
        self.assertEqual(len(set(cache_dirs)), 11)
        exports = [next(arg for arg in call if arg.startswith('--export=')) for call in calls]
        self.assertIn('weights_vfull_retry1', exports[0])
        self.assertTrue(all('eval_retry1/' in item for item in exports[1:-1]))
        self.assertTrue(all('EXPECTED_CHECKPOINT_EPOCH=8' in item for item in exports[1:-1]))
        # Mocking submissions must not execute training or touch old artifacts.
        self.assertEqual((self.run / 'weights_vmissing/config.json').read_text(), '{}')

    def test_missing_trained_checkpoint_fails_before_any_submission(self):
        (self.run / 'weights_apair/full_model_last.pth').unlink()
        result, calls = self.formal()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('existing trained-arm checkpoint missing', result.stderr)
        self.assertEqual(calls, [])

    def test_duplicate_recovery_is_refused_even_before_training_starts(self):
        first, calls = self.formal()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(len(calls), 12)
        second, calls_again = self.formal()
        self.assertNotEqual(second.returncode, 0)
        self.assertIn('Recovery already submitted or reserved', second.stderr)
        self.assertEqual(calls_again, calls)

    def test_existing_eval_is_refused_before_training_is_submitted(self):
        path = self.run / 'eval_retry1/vfull__vfull__normal.npz'
        path.parent.mkdir()
        path.write_bytes(b'preserve existing output')
        result = self.call()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Refusing to overwrite existing evaluation', result.stderr)
        self.assertNotIn('[DRY-RUN] sbatch', result.stderr)
        self.assertEqual(path.read_bytes(), b'preserve existing output')

    def test_existing_vfull_checkpoint_is_refused_without_explicit_resume(self):
        weight = self.run / 'weights_vfull_retry1'
        weight.mkdir()
        (weight / 'full_model_last.pth').write_bytes(b'preserve existing checkpoint')
        result = self.call()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Refusing to overwrite non-empty arm output', result.stderr)
        self.assertNotIn('[DRY-RUN] sbatch', result.stderr)


if __name__ == '__main__':
    unittest.main()
