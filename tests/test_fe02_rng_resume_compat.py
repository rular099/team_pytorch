"""Synthetic legacy-device API checks; not a real ROCm/DCU resume test."""
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
SHIM = ROOT / 'compat/fe02_rng_cpu/sitecustomize.py'
BASE = '9f70371838ec31044e559096a939f023d2924f37'


def shim_module():
    spec = importlib.util.spec_from_file_location('_fe02_rng_compat_test', SHIM)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(os.environ, {'FE02_RNG_CPU_COMPAT': '0'}):
        spec.loader.exec_module(module)
    return module


class RelocatedState(torch.Tensor):
    """CPU-backed fixture pretending to have been mapped to a DCU by torch.load."""
    @property
    def device(self):
        return torch.device('cuda:0')

    def cpu(self):
        return self.as_subclass(torch.Tensor).clone()


class RngResumeCompatibilityTests(unittest.TestCase):
    def test_preserves_bytes_order_and_next_random_values_without_reseeding(self):
        module = shim_module()
        states = [torch.Generator().manual_seed(s).get_state() for s in (42, 43, 44, 45)]
        relocated = [s.as_subclass(RelocatedState) for s in states]
        observed = []

        def legacy_restore(values):
            for v in values:
                if v.device.type != 'cpu' or v.dtype != torch.uint8:
                    raise TypeError('RNG state must be a torch.ByteTensor')
            observed.extend(values)

        with self.assertRaisesRegex(TypeError, 'ByteTensor'):
            legacy_restore(relocated)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {
                'FE02_RNG_COMPAT_SHA256': hashlib.sha256(SHIM.read_bytes()).hexdigest(),
                'FE02_OUTPUT_ROOT': tmp, 'RANK': '7', 'SLURM_JOB_ID': '30444656', 'SLURM_ARRAY_TASK_ID': '4'}), \
                mock.patch.object(torch.cuda, 'set_rng_state_all', new=legacy_restore):
            module.install()
            torch.cuda.set_rng_state_all(relocated)
            self.assertEqual(len(observed), 4)
            for before, after in zip(states, observed):
                self.assertTrue(torch.equal(before, after))
                a = torch.Generator().set_state(before)
                b = torch.Generator().set_state(after)
                self.assertTrue(torch.equal(torch.rand(16, generator=a), torch.rand(16, generator=b)))
            receipt = json.loads(next((Path(tmp) / 'compatibility_receipts').glob('*.json')).read_text())
            self.assertEqual(receipt['rank'], 7)
            self.assertTrue(receipt['state_bytes_preserved'])
            self.assertEqual(receipt['states'][0]['sha256'], hashlib.sha256(bytes(states[0].tolist())).hexdigest())

    def test_rejects_invalid_dtype_or_shape_instead_of_casting_or_skipping(self):
        module = shim_module()
        for bad in ([torch.ones(4)], [torch.zeros((2, 2), dtype=torch.uint8)], [b'not a tensor']):
            with self.assertRaises(TypeError):
                module.cpu_byte_states(bad)

    def test_inactive_on_login_helpers_and_without_opt_in(self):
        for overrides in ({'FE02_RNG_CPU_COMPAT': '1'}, {'FE02_RNG_CPU_COMPAT': '0', 'RANK': '0'}):
            env = {k: v for k, v in os.environ.items() if k not in ('RANK', 'FE02_RNG_CPU_COMPAT')}
            env.update(overrides, PYTHONPATH=str(SHIM.parent))
            result = subprocess.run([sys.executable, '-c',
                'import sys; assert "sitecustomize" in sys.modules; assert "torch" not in sys.modules'],
                env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_changed_shim_hash_fails_closed_at_rank_startup(self):
        env = {**os.environ, 'PYTHONPATH': str(SHIM.parent), 'FE02_RNG_CPU_COMPAT': '1',
               'FE02_RNG_COMPAT_SHA256': 'wrong', 'RANK': '0'}
        result = subprocess.run([sys.executable, '-c', 'print("MUST_NOT_TRAIN")'],
                                env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('MUST_NOT_TRAIN', result.stdout)
        self.assertIn('hash was not pinned', result.stderr)

    def test_original_training_code_identity_matches_baseline(self):
        from fe01 import engine as core
        from fe02 import engine as experiment
        from fe01.config import sha256

        @functools.lru_cache(None)
        def baseline_sha(path):
            path = Path(path).resolve()
            if ROOT not in path.parents:
                return sha256(path)
            contents = subprocess.check_output(['git', 'show', BASE + ':' + str(path.relative_to(ROOT))], cwd=ROOT)
            return hashlib.sha256(contents).hexdigest()

        current = experiment.code_identity()
        with mock.patch.object(core, 'sha256', baseline_sha), mock.patch.object(experiment, 'sha256', baseline_sha):
            self.assertEqual(current, experiment.code_identity())

    def test_launcher_submits_only_existing_m_run_with_pinned_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'compat/fe02_rng_cpu').mkdir(parents=True)
            (root / 'scripts/fe02').mkdir(parents=True)
            shutil.copyfile(ROOT / 'resume_fe02_rng_cpu.sh', root / 'resume_fe02_rng_cpu.sh')
            shutil.copyfile(SHIM, root / 'compat/fe02_rng_cpu/sitecustomize.py')
            (root / 'SOURCE_IDENTITY.json').write_text(json.dumps({'commit': BASE}))
            (root / 'scripts/fe02/submit.sh').write_text('''#!/bin/bash
set -euo pipefail
[[ "$FE02_RESUME" == 1 && "$FE02_INDICES" == 4 && "$FE02_RNG_CPU_COMPAT" == 1 ]]
[[ "$PYTHONPATH" == "$FE02_TEST_ROOT/compat/fe02_rng_cpu"* ]]
[[ "$(sha256sum "$FE02_TEST_ROOT/compat/fe02_rng_cpu/sitecustomize.py" | cut -d ' ' -f 1)" == "$FE02_RNG_COMPAT_SHA256" ]]
echo 'Submitted batch job 98765'
''')
            env = {k: v for k, v in os.environ.items() if not k.startswith('FE02_')}
            result = subprocess.run(['bash', str(root / 'resume_fe02_rng_cpu.sh')],
                                    env={**env, 'FE02_TEST_ROOT': str(root)}, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), 'Submitted batch job 98765')
            with (root / 'compat/fe02_rng_cpu/sitecustomize.py').open('a') as stream:
                stream.write('\n# tampered\n')
            changed = subprocess.run(['bash', str(root / 'resume_fe02_rng_cpu.sh')],
                                     env=env, capture_output=True, text=True)
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn('checksum mismatch', changed.stderr)


if __name__ == '__main__':
    unittest.main()
