"""Regression for wrong-device RNG state, bounded diagnostics and safe restart."""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import pytest
import torch

from scripts.fe01_a01_ops import compatibility_contract as contract
from scripts.fe01_a01_ops import continue_training as ct
from scripts.fe01_a01_ops import training_compat
from scripts.fe01_a01_ops.startup_trace import StartupTrace
from test_fe01_a01_continue import get_config, partial, run_dir
from test_fe01_a01_dcu_recovery import settings
from test_fe01_a01_scheduler import ROOT, gates


def rng_expression(module_path):
    tree = ast.parse(module_path.read_text())
    return next(n for n in ast.walk(tree) if isinstance(n, ast.Expr) and
                isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'torch.cuda.set_rng_state_all')


def execute_expression(expression, rng, api):
    code = compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(expression)], type_ignores=[])),
                   '<isolated-rng-restoration-regression>', 'exec')
    exec(code, {'torch': api, 'rng': rng})


def test_wrong_device_state_reproduces_failure_and_cpu_conversion_preserves_next_draws():
    # A simulated device carrier lets the CPU-only suite exercise the real
    # ByteTensor generator API, without claiming execution on a DCU.
    reference = torch.Generator().manual_seed(4521)
    saved = reference.get_state()
    class DeviceCarrier:
        def cpu(self):
            return saved.clone()
    destination = torch.Generator().manual_seed(999)
    def strict_set(states):
        assert len(states) == 1
        destination.set_state(states[0])  # actual ByteTensor contract
    api = SimpleNamespace(cuda=SimpleNamespace(set_rng_state_all=strict_set))
    rng = {'cuda_rng': [DeviceCarrier()]}
    before = rng_expression(ROOT/'fe01_a01/engine.py')
    after = rng_expression(ROOT/'scripts/fe01_a01_ops/training_compat.py')
    with pytest.raises(TypeError):
        execute_expression(before, rng, api)
    execute_expression(after, rng, api)
    assert torch.equal(destination.get_state(), saved)
    assert torch.equal(torch.rand(32, generator=destination), torch.rand(32, generator=reference))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Local CUDA/ROCm unavailable; actual DCU resume remains NOT_RUN')
def test_actual_device_mapped_rng_state_requires_cpu_before_restoration():
    local = torch.cuda.current_device()
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.cuda.manual_seed_all(4521)
        states = torch.cuda.get_rng_state_all()
        expected = torch.rand(32, device=f'cuda:{local}')
        mapped = {'cuda_rng': [state.to(f'cuda:{local}') for state in states]}
        with pytest.raises(TypeError):
            execute_expression(rng_expression(ROOT/'fe01_a01/engine.py'), mapped, torch)
        execute_expression(rng_expression(ROOT/'scripts/fe01_a01_ops/training_compat.py'), mapped, torch)
        assert torch.equal(torch.rand(32, device=f'cuda:{local}'), expected)


@pytest.mark.parametrize('old,new', [
    ('torch.optim.Adam', 'torch.optim.SGD'),
    ("[state.cpu() for state in rng['cuda_rng']]", "rng['cuda_rng']"),
    ('sampler.set_epoch(epoch)', 'sampler.set_epoch(epoch+1)'),
    ("observe('model_build.begin')", "observe(torch.rand(1))"),
])
def test_source_contract_rejects_training_drift_and_rng_regression(tmp_path, old, new):
    assert contract.verify(ROOT)['source_delta_check'] == 'PASS'
    path = tmp_path/'training_compat.py'
    path.write_text((ROOT/'scripts/fe01_a01_ops/training_compat.py').read_text().replace(old, new))
    with pytest.raises(ValueError):
        contract.verify(ROOT, path)


def test_explicit_entry_keeps_original_dependency_bindings():
    from fe01_a01 import engine
    original_train, original_load = engine.train, torch.load
    contract.verify_bindings(engine, training_compat)
    assert engine.train is original_train and torch.load is original_load


def test_execution_records_new_source_separately_and_reaches_original_config_gate(tmp_path, monkeypatch):
    from fe01_a01 import engine
    monkeypatch.setenv('A01_CODE_ROOT', str(ROOT))
    monkeypatch.setenv('A01_OPERATION_DIRECTORY', str(tmp_path))
    monkeypatch.setenv('RANK', '0')
    original_train, original_load = engine.train, torch.load
    stages = []
    # Invalid synthetic config stops before allocation, model building or data I/O.
    with pytest.raises((KeyError, ValueError)):
        ct.execute_training({}, Path('/SYNTHETIC_UNUSED'), True, SimpleNamespace(stage=stages.append), 3)
    identity = json.loads((tmp_path/'execution_identity_index3.json').read_text())
    assert identity['compatibility']['source_delta_check'] == 'PASS'
    assert identity['authenticated_core_sha256'] == '3e1e4398b01d724f793f88b19be2a2c03ca67a71e881d876c173a8fbc5b8177f'
    assert identity['compatibility']['executed_compatibility_sha256'] != identity['authenticated_core_sha256']
    assert stages == ['compatibility_verified', 'validate.begin']
    assert engine.train is original_train and torch.load is original_load


def test_archive_only_uncommitted_off_preserves_whole_old_directory_and_old_gates(tmp_path):
    env, scheduler = settings(tmp_path);gates(env, [3, 4])
    run = run_dir(env, 4);run.mkdir()
    (run/'init.pth').write_bytes(b'SYNTHETIC_UNCOMMITTED_INITIAL_STATE')
    (run/'sample_journals').mkdir()
    (run/'sample_journals/epoch0001_OLD_rank0.jsonl').touch()
    pins = {p: p.read_bytes() for p in Path(env['A01_OUTPUT_ROOT']).rglob('*.json')}
    operation = Path(env['A01_OUTPUT_ROOT'])/'operations/NEW_SYNTHETIC';operation.mkdir(parents=True)
    env.update(A01_ARCHIVE_UNCOMMITTED_DITING_OFF='1', A01_OPERATION_DIRECTORY=str(operation), SLURM_JOB_ID='SYNTHETIC')
    with patch.dict(os.environ, env):
        assert ct.plan([4])[0]['action'] == 'archive_then_fresh'
        ct.prepare(4)
        assert ct.plan([4])[0]['action'] == 'fresh'
    assert not run.exists()
    archive = operation/'uncommitted_before_index4'
    assert (archive/'init.pth').read_bytes() == b'SYNTHETIC_UNCOMMITTED_INITIAL_STATE'
    assert (archive/'sample_journals/epoch0001_OLD_rank0.jsonl').is_file()
    assert all(p.read_bytes() == value for p, value in pins.items())
    assert not scheduler.exists()


@pytest.mark.parametrize('bad', ['best.pth', 'last.pth.tmp', 'unknown.txt', 'sample_journals/nonempty.jsonl'])
def test_archive_policy_blocks_possible_progress_or_unknown_files(tmp_path, bad):
    env, scheduler = settings(tmp_path);gates(env, [3, 4])
    run = run_dir(env, 4);run.mkdir()
    target = run/bad;target.parent.mkdir(exist_ok=True);target.write_bytes(b'SYNTHETIC_KEEP')
    env['A01_ARCHIVE_UNCOMMITTED_DITING_OFF'] = '1'
    with patch.dict(os.environ, env), pytest.raises(ValueError, match='preserve it for inspection'):
        ct.plan([4])
    assert target.read_bytes() == b'SYNTHETIC_KEEP' and not scheduler.exists()


def test_archive_policy_still_blocks_on_missing_its_epoch5_checkpoint(tmp_path):
    env, scheduler = settings(tmp_path);gates(env, [3, 4])
    run = partial(env);(run/'last.pth').unlink()
    env['A01_ARCHIVE_UNCOMMITTED_DITING_OFF'] = '1'
    with patch.dict(os.environ, env), pytest.raises(ValueError, match='preserve it for inspection'):
        ct.plan([3])
    assert (run/'training_curves.csv').is_file() and not scheduler.exists()


def test_trace_does_not_consume_rng_and_flushes_stage_without_sample_journal(tmp_path):
    import random
    import numpy as np
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with StartupTrace(tmp_path, 4, 0) as trace:
        trace.stage('first_update.complete')
        progress = tmp_path/'startup/index4_rank0.jsonl'
        assert 'first_update.complete' in progress.read_text()
    assert random.getstate() == python_state
    assert all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), numpy_state))
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_trace_captures_actual_python_stack_on_startup_failure(tmp_path):
    with pytest.raises(RuntimeError, match='SYNTHETIC_STARTUP_FAILURE'):
        with StartupTrace(tmp_path, 4, 0):
            raise RuntimeError('SYNTHETIC_STARTUP_FAILURE')
    assert 'test_trace_captures_actual_python_stack' in (tmp_path/'startup/index4_rank0.stack.log').read_text()
    assert 'SYNTHETIC_STARTUP_FAILURE' in (tmp_path/'startup/index4_rank0.jsonl').read_text()


def test_new_entry_freezes_companions_and_compute_failure_still_returns_diagnostics(tmp_path):
    env, log = settings(tmp_path);gates(env, [3, 4]);partial(env)
    env['A01_RECOVERY_TIME'] = '72:00:00'
    entry = ROOT/'scripts/fe01_a01_ops/submit_diting_resume_fix.sh'
    result = subprocess.run(['bash',str(entry),'3,4'],env=env,text=True,capture_output=True)
    assert result.returncode == 0, result.stderr
    operation = next((Path(env['A01_OUTPUT_ROOT'])/'operations').iterdir())
    for name in ct.COMPATIBILITY_FILES:
        assert (operation/name).read_bytes() == (entry.parent/name).read_bytes()
    assert '--array=3,4%2' in log.read_text()
    bin_dir = tmp_path/'bin'
    (bin_dir/'scontrol').write_text('#!/bin/bash\nprintf "synthetic_node\\n"\n')
    (bin_dir/'scontrol').chmod(0o755)
    (bin_dir/'srun').write_text('#!/bin/bash\nprintf "srun %s\\n" "$*" >> "$FAKE_SCHEDULER_LOG"\n'
                              '[[ "$*" != *"_rank_continue"* ]] || exit 124\n')
    env.update(SLURM_JOB_ID='778', SLURM_ARRAY_JOB_ID='777', SLURM_ARRAY_TASK_ID='4',
               SLURM_JOB_NUM_NODES='4', SLURM_JOB_NODELIST='synthetic_nodes', A01_MASTER_PORT='29607',
               A01_ENV_FILE=str(operation/'a01.private.env'), A01_RECOVERY_LAUNCHER=str(operation/'dcu_recovery.sh'),
               A01_RECOVERY_HELPER=str(operation/'continue_training.py'), A01_OPERATION_DIRECTORY=str(operation),
               A01_ARCHIVE_UNCOMMITTED_DITING_OFF='1')
    result = subprocess.run(['bash',str(operation/'dcu_recovery.sh'),'_job_continue'],env=env,text=True,capture_output=True)
    assert result.returncode == 124, result.stderr
    with zipfile.ZipFile(operation/'return_bundle_index4.zip') as archive:
        assert json.loads(archive.read('summary.json'))['batch_exit_code'] == 124
    assert 'A01_RETURN_BUNDLE=' in result.stdout


def test_return_bundle_includes_stacks_and_inventory_but_never_weights_or_private_env(tmp_path):
    env, scheduler = settings(tmp_path);gates(env, [3, 4]);run = partial(env)
    operation = Path(env['A01_OUTPUT_ROOT'])/'operations/NEW_SYNTHETIC';operation.mkdir(parents=True)
    (operation/'startup').mkdir()
    (operation/'startup/index3_rank0.jsonl').write_text('{"stage":"rng_restore.complete"}\n')
    (operation/'startup/index3_rank0.stack.log').write_text('SYNTHETIC_STACK')
    (operation/'a01.private.env').write_text('NEVER_EXPORT_PRIVATE_SETTINGS')
    (operation/'checkpoint_before_3').mkdir()
    (operation/'checkpoint_before_3/last.pth').write_bytes(b'NEVER_EXPORT_WEIGHTS')
    (operation/'777_3.err').write_text('SYNTHETIC_ERROR')
    env.update(A01_OPERATION_DIRECTORY=str(operation), SLURM_ARRAY_JOB_ID='777', SLURM_JOB_ID='778')
    with patch.dict(os.environ, env):
        ct.bundle(3, 124)
    with zipfile.ZipFile(operation/'return_bundle_index3.zip') as archive:
        assert 'startup/index3_rank0.stack.log' in archive.namelist() and '777_3.err' in archive.namelist()
        assert not any(n.endswith(('.pth','.env')) for n in archive.namelist())
        summary = json.loads(archive.read('summary.json'))
        assert summary['batch_exit_code'] == 124
        assert any(r['path'] == 'last.pth' for r in summary['run_files'])
        assert all(b'NEVER_EXPORT' not in archive.read(n) for n in archive.namelist())
    assert (run/'last.pth').is_file() and not scheduler.exists()
