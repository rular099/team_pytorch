"""No real scheduler/GPU/data; verify gated continuation and preserved state."""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

from scripts.fe01_a01_ops import continue_training as ct
from fe01.config import fingerprint
from test_fe01_a01_dcu_recovery import TOOL, settings, compute_fixture
from test_fe01_a01_scheduler import ROOT, gates

HELPER = ROOT/'scripts/fe01_a01_ops/continue_training.py'


def get_config(env, index=3):
    with patch.dict(os.environ, env):
        return ct.config(index)


def run_dir(env, index):
    return Path(env['A01_OUTPUT_ROOT'])/get_config(env, index)['run_id']


def partial(env, index=3):
    run = run_dir(env, index)
    run.mkdir(parents=True, exist_ok=True)
    (run/'last.pth').write_bytes(b'SYNTHETIC_CHECKPOINT_ONLY')
    (run/'training_curves.csv').write_text('epoch,updates\n'+''.join(f'{i},{i*212}\n' for i in range(1, 6)))
    return run


def test_plan_mixes_resume_and_empty_fresh_without_loading_torch(tmp_path):
    env, log = settings(tmp_path);gates(env, [3, 4]);partial(env)
    fake_module = tmp_path/'no_torch';fake_module.mkdir()
    (fake_module/'torch.py').write_text('raise RuntimeError("LOGIN_MUST_NOT_IMPORT_TORCH")\n')
    env['PYTHONPATH'] = str(fake_module)
    result = subprocess.run([sys.executable, str(HELPER), 'plan', '--indices', '3,4'], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert [(r['index'], r['action'], r['curve_epoch']) for r in json.loads(result.stdout)] == [(3, 'resume', 5), (4, 'fresh', None)]
    assert not log.exists()


def test_continue_blocks_completed_missing_last_and_stale_pilot_before_sbatch(tmp_path):
    env, log = settings(tmp_path);gates(env, [3, 4])
    run = partial(env)
    for bad in ('completed', 'missing_last', 'stale_pilot'):
        gates(env, [3, 4])
        (run/'last.pth').write_bytes(b'SYNTHETIC')
        (run/'training_curves.csv').write_text('epoch,updates\n'+''.join(f'{i},{i*212}\n' for i in range(1, 13 if bad == 'completed' else 6)))
        if bad == 'missing_last':
            (run/'last.pth').unlink()
        if bad == 'stale_pilot':
            cfg = get_config(env)
            gate = Path(env['A01_OUTPUT_ROOT'])/'pilot'/cfg['run_id']/'pilot.json'
            value = json.loads(gate.read_text());value['audit_lock_sha256'] = 'tampered';gate.write_text(json.dumps(value))
        result = subprocess.run(['bash', str(TOOL), 'continue', '3,4'], env=env, text=True, capture_output=True)
        assert result.returncode != 0 and not log.exists(), (bad, result.stdout, result.stderr)


def test_submit_freezes_helper_requests_only_two_indices_and_retains_source(tmp_path):
    env, log = settings(tmp_path);gates(env, [3, 4]);run = partial(env)
    before = (run/'last.pth').read_bytes()
    env.update(A01_RECOVERY_TIME='72:00:00', A01_RECOVERY_EXCLUDE_NODES='synthetic[01-04]')
    result = subprocess.run(['bash', str(TOOL), 'continue', '3,4'], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    call = log.read_text()
    assert call.count('sbatch ') == 1 and '--array=3,4%2' in call and '--time=72:00:00' in call
    assert '--exclude=synthetic[01-04]' in call and call.rstrip().endswith('_job_continue')
    operation = next((Path(env['A01_OUTPUT_ROOT'])/'operations').iterdir())
    assert (operation/'continue_training.py').read_bytes() == HELPER.read_bytes()
    assert (operation/'helper.sha256').is_file() and (operation/'continuation_plan.json').is_file()
    assert (run/'last.pth').read_bytes() == before


@pytest.mark.parametrize('maximum,expected', [('UNLIMITED','72:00:00'),('1-00:00:00','24:00:00'),('2-00:00:00','48:00:00')])
def test_walltime_respects_partition_and_falls_back_when_unknown(monkeypatch, maximum, expected):
    monkeypatch.delenv('A01_RECOVERY_TIME', raising=False);monkeypatch.setenv('A01_TRAIN_TIME','24:00:00');monkeypatch.setenv('A01_PARTITION','')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, 'PartitionName=synthetic Default=YES MaxTime='+maximum))
    assert ct.walltime()[0] == expected
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, 'unknown'))
    assert ct.walltime()[0] == '24:00:00'


def checkpoint():
    cfg = dict(run_id='SYNTHETIC', output_root='/SYNTHETIC', training=dict(epochs=12), absolute_amplitude_mode='physical_scale_keep_duration')
    lock = dict(effective_config={}, world_size=16, a01_code_sha256='SYNTHETIC_CODE',
                initial_state=dict(encoder_sha256='SYNTHETIC_ENCODER'), sampling=dict(budget=dict(updates_per_epoch=212)))
    payload = dict(config=cfg, identity=fingerprint(dict(config=cfg, audit_lock=lock, world_size=16)),
                   source_config_sha256=fingerprint(cfg), a01_code_sha256=lock['a01_code_sha256'], world_size=16,
                   absolute_amplitude_mode=cfg['absolute_amplitude_mode'], encoder_state_sha256='SYNTHETIC_ENCODER',
                   epoch=5, updates=1060, curve_records=[dict(epoch=i, updates=i*212) for i in range(1,6)],
                   rank_rng=[dict(rng=torch.get_rng_state(), python_rng=None, numpy_rng=None, cuda_rng=[]) for _ in range(16)],
                   optimizer={'SYNTHETIC':torch.tensor([5.])}, scheduler={'last_epoch':1060}, committed_journals=[],
                   common_initial_state_sha256='SYNTHETIC_INIT', model_state_dict={'SYNTHETIC':torch.tensor([7.])},
                   best_epoch=5, best_selection_score=1.)
    return cfg, lock, payload


@pytest.mark.parametrize('bad', ['identity','world','updates','rank_rng','amplitude','completed'])
def test_checkpoint_validation_rejects_incompatible_or_incomplete_resume(bad):
    cfg, lock, payload = checkpoint()
    assert ct.checkpoint_gate(payload, cfg, lock, 16) == 5
    key, value = dict(identity=('identity','tampered'), world=('world_size',8), updates=('updates',1061),
                      rank_rng=('rank_rng',[]), amplitude=('absolute_amplitude_mode','wrong'), completed=('epoch',12))[bad]
    payload[key] = value
    with pytest.raises(ValueError, match='Resume checkpoint'):
        ct.checkpoint_gate(payload, cfg, lock, 16)


def test_resume_calls_original_engine_and_preserves_old_checkpoint_after_atomic_publish(tmp_path, monkeypatch):
    from fe01_a01 import engine
    cfg, lock, payload = checkpoint()
    cfg['output_root'] = str(tmp_path/'out')
    payload['config'] = cfg;payload['source_config_sha256'] = fingerprint(cfg)
    payload['identity'] = fingerprint(dict(config=cfg, audit_lock=lock, world_size=16))
    run = Path(cfg['output_root'])/cfg['run_id'];run.mkdir(parents=True)
    torch.save(payload, run/'last.pth');before = (run/'last.pth').read_bytes()
    operation = tmp_path/'operation';operation.mkdir()
    monkeypatch.setenv('RANK','0');monkeypatch.setenv('WORLD_SIZE','16');monkeypatch.setenv('A01_OPERATION_DIRECTORY',str(operation))
    monkeypatch.setattr(ct, 'config', lambda index: cfg)
    monkeypatch.setattr(ct, 'plan', lambda indices: [dict(index=3, action='resume', checkpoint=str(run/'last.pth'))])
    monkeypatch.setattr(engine, 'check_audit', lambda *a: lock)
    monkeypatch.setattr(engine, 'check_matched_pair', lambda *a: None)
    calls = []
    def existing_engine(config, audit, resume=False):
        calls.append(resume)
        (run/'last.pth.tmp').write_bytes(b'SYNTHETIC_NEXT_ATOMIC_PUBLICATION')
        os.replace(run/'last.pth.tmp', run/'last.pth')
    monkeypatch.setattr(engine, 'train', existing_engine)
    ct.train_one(3)
    assert calls == [True] and (operation/'checkpoint_before_3/last.pth').read_bytes() == before
    assert torch.load(operation/'checkpoint_before_3/last.pth',weights_only=False)['optimizer']['SYNTHETIC'].item() == 5.


def test_collective_failure_stops_before_training_and_probe_is_time_bounded(tmp_path):
    env, log = compute_fixture(tmp_path)
    operations = Path(env['A01_OUTPUT_ROOT'])/'operations';operations.mkdir(parents=True)
    env['A01_RECOVERY_HELPER'] = str(HELPER)
    srun = tmp_path/'bin/srun'
    srun.write_text('#!/bin/bash\nprintf "srun %s\\n" "$*" >> "$FAKE_SCHEDULER_LOG"\nexit 124\n')
    result = subprocess.run(['bash',str(TOOL),'_job_continue'],env=env,text=True,capture_output=True)
    assert result.returncode != 0
    call = log.read_text()
    assert call.count('srun ') == 1 and '--time=00:05:00' in call and '240s' in call
    assert '_rank_continue' not in call


def test_spooled_continuation_uses_shared_helper_and_probes_before_training(tmp_path):
    env, log = compute_fixture(tmp_path)
    (Path(env['A01_OUTPUT_ROOT'])/'operations').mkdir(parents=True)
    env['A01_RECOVERY_HELPER'] = str(HELPER)
    python = tmp_path/'probe_python'
    python.write_text('#!/bin/bash\nprintf "SYNTHETIC_HELPER %s\\n" "$*" >> "$FAKE_SCHEDULER_LOG"\n')
    srun = tmp_path/'bin/srun'
    srun.write_text('#!/bin/bash\nset -e\nprintf "srun %s\\n" "$*" >> "$FAKE_SCHEDULER_LOG"\n'
                    'while [[ "$1" != timeout && "$1" != bash ]]; do shift;done\n'
                    'export SLURM_PROCID=0 SLURM_LOCALID=0\nexec "$@"\n')
    spool = tmp_path/'node_local_spool';spool.mkdir()
    spooled = spool/'slurm_script';spooled.write_bytes(TOOL.read_bytes())
    result = subprocess.run(['bash',str(spooled),'_job_continue'],env=env,text=True,capture_output=True)
    assert result.returncode == 0, result.stderr
    calls = log.read_text()
    assert calls.count('srun ') == 2 and str(spool) not in calls
    assert calls.index('continue_training.py collective') < calls.index('continue_training.py supervise --index 2')


def test_training_startup_watchdog_does_not_accept_old_journals_and_kills_stalled_child(tmp_path, monkeypatch):
    run = tmp_path/'SYNTHETIC';journal = run/'sample_journals';journal.mkdir(parents=True)
    (journal/'epoch0006_OLD_rank0.jsonl').write_text('SYNTHETIC_OLD_PROGRESS\n')
    monkeypatch.setattr(ct, 'config', lambda index: dict(output_root=str(tmp_path),run_id='SYNTHETIC'))
    monkeypatch.setenv('RANK','0');monkeypatch.setenv('A01_STARTUP_TIMEOUT_SECONDS','0.1')
    real_popen = subprocess.Popen
    children = []
    def stalled(*a, **k):
        child = real_popen([sys.executable, '-c', 'import time;time.sleep(30)'])
        children.append(child)
        return child
    monkeypatch.setattr(subprocess, 'Popen', stalled)
    assert ct.supervise(4) == 124
    assert children[0].poll() is not None and (journal/'epoch0006_OLD_rank0.jsonl').read_text() == 'SYNTHETIC_OLD_PROGRESS\n'


def test_training_startup_watchdog_releases_after_new_rank_progress(tmp_path, monkeypatch):
    journal = tmp_path/'SYNTHETIC/sample_journals';journal.mkdir(parents=True)
    monkeypatch.setattr(ct, 'config', lambda index: dict(output_root=str(tmp_path),run_id='SYNTHETIC'))
    monkeypatch.setenv('RANK','0');monkeypatch.setenv('A01_STARTUP_TIMEOUT_SECONDS','0.1')
    real_popen = subprocess.Popen
    def progresses(*a, **k):
        (journal/'epoch0006_NEW_rank0.jsonl').write_text('SYNTHETIC_NEW_PROGRESS\n')
        return real_popen([sys.executable, '-c', 'import time;time.sleep(0.2)'])
    monkeypatch.setattr(subprocess, 'Popen', progresses)
    assert ct.supervise(3) == 0  # total exceeds startup limit; progress disables it


def test_existing_shared_lock_blocks_another_continuation_before_srun(tmp_path):
    import fcntl
    env, log = compute_fixture(tmp_path)
    operations = Path(env['A01_OUTPUT_ROOT'])/'operations';operations.mkdir(parents=True)
    with (operations/'continue_index2.lock').open('w') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(['bash',str(TOOL),'_job_continue'],env=env,text=True,capture_output=True)
    assert result.returncode == 2 and 'Another continuation' in result.stderr
    assert not log.exists()
