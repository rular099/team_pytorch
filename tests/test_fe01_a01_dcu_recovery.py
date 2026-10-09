"""Operational launcher tests use a fake scheduler; no cluster jobs or GPUs."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from test_fe01_a01_scheduler import ROOT,settings,gates

TOOL=ROOT/'scripts/fe01_a01_ops/dcu_recovery.sh'


def run(env,*args):
    return subprocess.run(['bash',str(TOOL),*args],env=env,text=True,capture_output=True)


def error_logs(env,indices):
    root=Path(env['A01_OUTPUT_ROOT']);(root/'slurm').mkdir(parents=True)
    (root/'jobs.tsv').write_text('train\t0-4\t777777\n')
    for i in range(5):
        text='AssertionError: DCU unavailable in allocation\n' if i in indices else ''
        (root/'slurm'/f'train_777777_{i}.err').write_text(text)


def test_inspect_is_readonly_and_only_matches_DCU_startup_assertions(tmp_path):
    env,log=settings(tmp_path);error_logs(env,[0,2,4])
    result=run(env,'inspect')
    assert result.returncode==0,result.stderr
    value=json.loads(result.stdout)
    assert value['array_job_id']=='777777' and value['failed_indices']==['0','2','4']
    assert not log.exists() and not (Path(env['A01_OUTPUT_ROOT'])/'operations').exists()


def test_retry_failed_preserves_gates_freezes_launcher_and_submits_one_array(tmp_path):
    env,log=settings(tmp_path);error_logs(env,[0,2,4]);gates(env,[0,2,3,4])
    private=Path(env['A01_ENV_FILE']);previous=private.read_bytes()
    pins={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(env['A01_OUTPUT_ROOT']).rglob('*.json')}
    result=run(env,'retry','failed')
    assert result.returncode==0,result.stderr
    calls=log.read_text().splitlines()
    assert len(calls)==1 and '--array=0,2,4%3' in calls[0] and calls[0].endswith('_job_retry')
    assert '--nodes=4' in calls[0] and '--ntasks-per-node=4' in calls[0] and '--gres=dcu:4' in calls[0]
    assert private.read_bytes()==previous
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==digest for p,digest in pins.items())
    operation=next((Path(env['A01_OUTPUT_ROOT'])/'operations').iterdir())
    assert (operation/'dcu_recovery.sh').read_bytes()==TOOL.read_bytes()
    assert (operation/'dcu_recovery.sh').stat().st_mode&0o777==0o500
    assert (operation/'a01.private.env').stat().st_mode&0o777==0o600
    assert (operation/'launcher.sha256').is_file()


def test_partial_prerequisites_or_completed_run_blocks_entire_retry_array(tmp_path):
    env,log=settings(tmp_path);gates(env,[0])
    result=run(env,'retry','0,2')
    assert result.returncode!=0 and not log.exists()
    gates(env,[0,2]);output=Path(env['A01_OUTPUT_ROOT'])/'a01__team_original_scratch__off__seed42'
    output.mkdir();(output/'best.pth').write_bytes(b'SYNTHETIC_COMPLETED_CHECKPOINT')
    result=run(env,'retry','0,2')
    assert result.returncode!=0 and not log.exists()
    assert (output/'best.pth').read_bytes()==b'SYNTHETIC_COMPLETED_CHECKPOINT'


def test_probe_submits_only_small_device_checks_with_existing_resources(tmp_path):
    env,log=settings(tmp_path)
    result=run(env,'probe')
    assert result.returncode==0,result.stderr
    call=log.read_text()
    assert call.count('sbatch ')==1 and '--time=00:10:00' in call and '--gres=dcu:4' in call
    assert '--array=' not in call and call.rstrip().endswith('_job_probe')


def test_invalid_indices_and_world_never_submit(tmp_path):
    env,log=settings(tmp_path)
    for value in ('0-4','0,0','5','0;1'):
        assert run(env,'retry',value).returncode!=0
    private=Path(env['A01_ENV_FILE']);private.write_text(private.read_text()+'export A01_TRAIN_NODES=1\n')
    assert run(env,'probe').returncode!=0 and not log.exists()


def compute_fixture(tmp):
    env,log=settings(tmp)
    code=tmp/'compute_source';folder=code/'scripts/fe01_a01';folder.mkdir(parents=True)
    (folder/'preflight.py').touch()
    fake_python=tmp/'probe_python'
    fake_python.write_text('#!/bin/bash\ncat >/dev/null\nprintf "SYNTHETIC_PROBE %s\\n" "$A01_PROBE_CONTEXT" >> "$FAKE_SCHEDULER_LOG"\n')
    fake_python.chmod(0o755)
    # The fake batch process has no DCU; an srun step succeeds. Neither uses torch.
    (folder/'env.sh').write_text('set -euo pipefail\nexport A01_PYTHON='+shlex.quote(str(fake_python))+
        '\nprintf "ENV_CONTEXT rank=%s\\n" "${SLURM_PROCID:-batch}" >> "$FAKE_SCHEDULER_LOG"\n'
        '[[ -n "${SLURM_PROCID:-}" ]] || exit 1\n')
    (folder/'rank_exec.sh').write_text('printf "SYNTHETIC_TRAIN %s\\n" "$*" >> "$FAKE_SCHEDULER_LOG"\n')
    private=Path(env['A01_ENV_FILE']);private.write_text(private.read_text()+
        'export A01_CODE_ROOT='+shlex.quote(str(code))+'\nexport A01_MASTER_PORT=29607\n')
    fake=tmp/'bin'
    (fake/'srun').write_text('#!/bin/bash\nset -e\nprintf "srun %s\\n" "$*" >> "$FAKE_SCHEDULER_LOG"\n'
        'while [[ "$1" != bash ]]; do shift;done\nexport SLURM_PROCID=0 SLURM_LOCALID=0\nexec "$@"\n')
    (fake/'scontrol').write_text('#!/bin/bash\nprintf "synthetic_node\\n"\n')
    for p in fake.iterdir():p.chmod(0o755)
    env.update(SLURM_JOB_ID='SYNTHETIC_JOB',SLURM_JOB_NUM_NODES='4',SLURM_JOB_NODELIST='synthetic_nodes',SLURM_ARRAY_TASK_ID='2')
    env.pop('SLURM_PROCID',None);env.pop('SLURM_LOCALID',None)
    return env,log


def test_batch_probe_failure_does_not_skip_actual_step_probe(tmp_path):
    env,log=compute_fixture(tmp_path)
    result=run(env,'_job_probe')
    assert result.returncode==0,result.stderr
    assert 'batch_rc=1 step_rc=0' in result.stdout
    calls=log.read_text()
    assert 'ENV_CONTEXT rank=batch' in calls and 'ENV_CONTEXT rank=0' in calls
    assert 'SYNTHETIC_PROBE batch' in calls and 'SYNTHETIC_PROBE step' in calls
    assert 'SYNTHETIC_TRAIN' not in calls and 'sbatch ' not in calls


def test_retry_checks_actual_step_first_and_only_then_starts_original_training(tmp_path):
    env,log=compute_fixture(tmp_path)
    result=run(env,'_job_retry')
    assert result.returncode==0,result.stderr
    calls=log.read_text()
    assert calls.count('srun ')==2 and 'ENV_CONTEXT rank=batch' not in calls
    assert calls.index('SYNTHETIC_PROBE step')<calls.index('SYNTHETIC_TRAIN')
    assert 'run.py train --index 2' in calls and '--world 16' in calls and 'sbatch ' not in calls


def test_unusable_step_aborts_before_original_training(tmp_path):
    env,log=compute_fixture(tmp_path)
    fake=tmp_path/'probe_python';fake.write_text('#!/bin/bash\ncat >/dev/null\nexit 1\n');fake.chmod(0o755)
    result=run(env,'_job_retry')
    assert result.returncode!=0
    assert 'SYNTHETIC_TRAIN' not in log.read_text() and log.read_text().count('srun ')==1


def test_real_probe_program_records_initialization_exception_as_JSON(tmp_path):
    env,log=compute_fixture(tmp_path)
    module=tmp_path/'fake_torch';module.mkdir()
    (module/'torch.py').write_text(
        'from types import SimpleNamespace\n__version__="SYNTHETIC_TORCH"\nversion=SimpleNamespace(hip="SYNTHETIC_HIP")\n'
        'class CUDA:\n'
        ' def is_available(self): return False\n'
        ' def device_count(self): return 0\n'
        ' def init(self): raise RuntimeError("SYNTHETIC_INITIALIZATION_ERROR")\n'
        'cuda=CUDA()\n')
    environment=tmp_path/'compute_source/scripts/fe01_a01/env.sh'
    environment.write_text(environment.read_text().replace(shlex.quote(str(tmp_path/'probe_python')),shlex.quote(sys.executable)))
    env['PYTHONPATH']=str(module)
    result=run(env,'_job_retry')
    assert result.returncode!=0 and 'SYNTHETIC_TRAIN' not in log.read_text()
    record=json.loads(next(line.split('A01_DCU_PROBE ',1)[1] for line in result.stdout.splitlines() if line.startswith('A01_DCU_PROBE ')))
    assert record['status']=='FAIL' and not record['available'] and record['count']==0
    assert record['init_error']=='RuntimeError: SYNTHETIC_INITIALIZATION_ERROR'
