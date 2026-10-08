"""Launcher-only synthetic checks. No torch, GPU, Slurm, or real encoder runs."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

from scripts.fe02.register_diting import ensure_registered

ROOT = Path(__file__).resolve().parents[1]


class SimpleSubmitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'source with spaces'
        self.scripts = self.root / 'scripts/fe02'
        self.scripts.mkdir(parents=True)
        for name in ('submit.sh', 'job.sh'):
            shutil.copyfile(ROOT / 'scripts/fe02' / name, self.scripts / name)
        shutil.copyfile(ROOT / 'fe02_cluster.env.example', self.root / 'fe02_cluster.env.example')
        # Compute environment and numerical work are stubs; these checks certify
        # orchestration/order/guards, not real environment or model correctness.
        (self.scripts / 'env.sh').write_text(
            'source "${FE02_ENV_FILE:?}"\ncd "$FE02_CODE_ROOT"\n')
        self.bins = Path(self.temp.name) / 'bin'
        self.bins.mkdir()
        self.trace = Path(self.temp.name) / 'trace'
        self.out = Path(self.temp.name) / 'results'
        fake_python = self.bins / 'python'
        fake_python.write_text('''#!/usr/bin/env bash
set -euo pipefail
case "$1" in
  *select_run.py)
    printf '%s\\n%s\\n' "$FE02_CODE_ROOT/unit.json" fe02__formal__R0__seed42;;
  *register_diting.py)
    echo register >> "$FE02_TEST_TRACE";;
  *run.py)
    action=$2
    printf '%s\\n' "$action" >> "$FE02_TEST_TRACE"
    [[ "$action" != audit || "${FE02_TEST_FAIL_AUDIT:-0}" != 1 ]] || exit 17
    if [[ "$action" == audit ]]; then
        mkdir -p "$FE02_OUTPUT_ROOT/audits/fe02__formal__R0__seed42"
        echo '{}' > "$FE02_OUTPUT_ROOT/audits/fe02__formal__R0__seed42/protocol.lock.json"
    fi
    if [[ "$action" == train && " $* " == *' --resume '* ]]; then
        echo resume >> "$FE02_TEST_TRACE"
    fi;;
  *) exit 98;;
esac
''')
        fake_python.chmod(0o755)
        srun = self.bins / 'srun'
        srun.write_text('''#!/usr/bin/env bash
set -euo pipefail
echo srun >> "$FE02_TEST_TRACE"
while [[ "$1" == --* ]]; do shift; done
export SLURM_PROCID=0 SLURM_LOCALID=0
exec "$@"
''')
        srun.chmod(0o755)
        scontrol = self.bins / 'scontrol'
        scontrol.write_text('#!/bin/sh\necho unit-node\n')
        scontrol.chmod(0o755)
        for name in ('sbatch', 'module', 'conda'):
            stub = self.bins / name
            stub.write_text('#!/bin/sh\nexit 99\n')
            stub.chmod(0o755)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith('FE02_')}
        self.env.update(FE02_OUTPUT_ROOT=str(self.out), FE02_PYTHON=str(fake_python),
                        FE02_TEST_TRACE=str(self.trace), PATH=str(self.bins) + ':' + os.environ['PATH'])

    def print_command(self, **overrides):
        return subprocess.run(['bash', str(self.scripts / 'submit.sh')],
                              env={**self.env, **overrides}, capture_output=True, text=True)

    def worker(self, command, spool_copy=False):
        args = shlex.split(command)[-4:]
        if spool_copy:
            spool = Path(self.temp.name) / 'slurm_script'
            shutil.copyfile(args[0], spool)
            args[0] = str(spool)
        return subprocess.run(['bash', *args], capture_output=True, text=True,
            env={**self.env, 'SLURM_JOB_ID': '12345', 'SLURM_ARRAY_TASK_ID': '0',
                 'SLURM_NTASKS_PER_NODE': '4', 'SLURM_NTASKS': '16', 'SLURM_JOB_NODELIST': 'unit-node'})

    def test_no_manual_env_registry_python_or_submission_needed(self):
        result = self.print_command()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.strip().splitlines()), 1)
        args = shlex.split(result.stdout)
        self.assertEqual(args[0], 'sbatch')
        for expected in ('--array=0-4%1', '--nodes=4', '--ntasks-per-node=4', '--gres=dcu:4', '--time=23:50:00'):
            self.assertIn(expected, args)
        self.assertFalse(any(a.startswith('--mem=') or 'dependency' in a or 'test_job' in a for a in args))
        self.assertFalse(self.trace.exists())
        self.assertFalse((self.out / 'offline_weights/pretrained_manifest.json').exists())
        snapshot = Path(args[-2])
        self.assertEqual(hashlib.sha256(snapshot.read_bytes()).hexdigest(), args[-1])
        exports = [shlex.split(line)[1] for line in snapshot.read_text().splitlines()]
        self.assertIn('FE02_CODE_ROOT=' + str(self.root), exports)

    def test_pipeline_registers_audits_trains_and_validates_in_order(self):
        result = self.worker(self.print_command().stdout)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.trace.read_text().splitlines(), ['register', 'audit', 'srun', 'train', 'eval'])

    def test_slurm_spool_copy_keeps_real_code_root(self):
        result = self.worker(self.print_command().stdout, spool_copy=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.trace.read_text().splitlines(), ['register', 'audit', 'srun', 'train', 'eval'])

    def test_audit_failure_never_trains(self):
        command = self.print_command(FE02_TEST_FAIL_AUDIT='1').stdout
        result = self.worker(command)
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertEqual(self.trace.read_text().splitlines(), ['register', 'audit'])

    def test_resume_reuses_audit_and_requires_last(self):
        run = self.out / 'fe02__formal__R0__seed42'
        run.mkdir(parents=True)
        (run / 'last.pth').write_bytes(b'synthetic, not a checkpoint')
        audit = self.out / 'audits/fe02__formal__R0__seed42'
        audit.mkdir(parents=True)
        (audit / 'protocol.lock.json').write_text('{}')
        command = self.print_command(FE02_RESUME='1', FE02_INDICES='0').stdout
        result = self.worker(command)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.trace.read_text().splitlines(), ['register', 'srun', 'train', 'resume', 'eval'])
        (run / 'last.pth').unlink()
        result = self.worker(command)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Resume requires', result.stderr)

    def test_existing_results_and_changed_snapshot_are_preserved_or_rejected(self):
        run = self.out / 'fe02__formal__R0__seed42'
        run.mkdir(parents=True)
        marker = run / 'user-result'
        marker.write_bytes(b'preserve me')
        command = self.print_command().stdout
        result = self.worker(command)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(marker.read_bytes(), b'preserve me')
        self.assertFalse(self.trace.exists())
        snapshot = Path(shlex.split(command)[-2])
        snapshot.chmod(0o600)
        snapshot.write_text(snapshot.read_text() + '\n# changed\n')
        result = self.worker(command)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('changed after command generation', result.stderr)

    def test_bad_indices_and_world_are_rejected(self):
        for overrides in ({'FE02_INDICES': '15'}, {'FE02_INDICES': '4-1'},
                          {'FE02_INDICES': '0,,2'}, {'FE02_NODES': '3'},
                          {'FE02_STAGE': 'pilot', 'FE02_INDICES': '5'}):
            with self.subTest(overrides=overrides):
                self.assertNotEqual(self.print_command(**overrides).returncode, 0)

    def test_registry_reuses_only_identical_file_without_overwriting(self):
        checkpoint = Path(self.temp.name) / 'unit.pt'
        checkpoint.write_bytes(b'synthetic, not DiTing')
        manifest = Path(self.temp.name) / 'manifest.json'
        ensure_registered(checkpoint, manifest, 'unit fixture')
        original = manifest.read_bytes()
        ensure_registered(checkpoint, manifest, 'different label must not rewrite')
        self.assertEqual(original, manifest.read_bytes())
        checkpoint.write_bytes(b'changed synthetic file')
        with self.assertRaises(ValueError):
            ensure_registered(checkpoint, manifest, 'unit fixture')
        self.assertEqual(original, manifest.read_bytes())
        self.assertEqual(json.loads(original)['models']['diting']['device_forward_status'], 'NOT_RUN')


if __name__ == '__main__':
    unittest.main()
