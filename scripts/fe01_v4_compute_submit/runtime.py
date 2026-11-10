"""Allocated pool supervisor and fresh-process original logical stage workers."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from support import (ALLOWED, ROOT, allocated, bind_attempt, digest, evidence_attempt, fingerprint,
                     read, require, revision, validate_plan, write)


def load_manifest(path, pin):
    require(digest(path) == read(pin)['sha256'], 'Pool mapping byte identity differs')
    m = read(path)
    require(m['stage'] in ALLOWED, 'Stage outside operational scope')
    expected_device = 'cuda' if m['stage'] in {'replay_stations', 'replay_grid_normal', 'latency'} else 'cpu'
    require(m['device'] == expected_device, 'Worker device differs from original stage contract')
    identity = revision()
    require(m['revision'] == identity, 'Pool operational revision differs')
    from fe01_a01_compare import orchestration as ops
    plan = ops.verify_plan(m['session'])
    validate_plan(plan)
    require(digest(Path(m['session']) / 'plan.json') == m['plan_sha256'], 'Pool plan differs')
    slot = os.environ.get('SLURM_ARRAY_TASK_ID')
    require(slot in m['worker_task_map'], 'Scheduler slot absent from pool mapping')
    tasks = m['worker_task_map'][slot]
    require(len(tasks) == len(set(tasks)) and tasks, 'Invalid pool assignment')
    for i in tasks:
        require(isinstance(i, int) and 0 <= i < len(plan['tasks'][m['stage']]), 'Invalid logical task index')
        require(fingerprint(plan['tasks'][m['stage']][i]) == m['logical_tasks_sha256'][str(i)],
                'Original logical task/32-event definition changed')
    return m, identity, tasks


def run_task(m, identity, path, index):
    if m['device'] == 'cuda':
        import torch
        require(torch.cuda.is_available(), 'DCU unavailable in allocated worker step')
    from fe01_a01_compare import stages
    evidence = dict(revision=identity, stage=m['stage'], logical_task=index, manifest_sha256=digest(path),
                    scheduler_slot_logical_tasks=m['worker_task_map'][os.environ['SLURM_ARRAY_TASK_ID']],
                    logical_task_sha256=m['logical_tasks_sha256'][str(index)],
                    scheduler={k:os.environ.get(k) for k in ('SLURM_JOB_ID','SLURM_ARRAY_JOB_ID','SLURM_ARRAY_TASK_ID')},
                    policy='Scheduler slot ID is unchanged; explicit mapping selects original logical task. Original execute bytecode and model/readout/loss/export are unchanged.')
    callback = evidence_attempt(stages.attempt, evidence)
    original = bind_attempt(stages.execute, callback)
    original(m['session'], m['stage'], index, m['device'])


def supervise(m, path, pin, tasks):
    failures = []
    for index in tasks:
        print('BEGIN_LOGICAL_TASK', m['stage'], index, 'scheduler_slot=' + os.environ['SLURM_ARRAY_TASK_ID'], flush=True)
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--manifest', str(path), '--pin', str(pin), '--task', str(index)]
        result = subprocess.run(command, check=False)
        print('END_LOGICAL_TASK', m['stage'], index, 'exit=' + str(result.returncode), flush=True)
        if result.returncode:
            failures.append(dict(index=index, returncode=result.returncode))
    parent = Path(m['session']) / 'operational_submit'
    parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='pool_' + os.environ['SLURM_JOB_ID'] + '_', dir=parent))
    write(out / 'worker_summary.json', dict(stage=m['stage'], scheduler_slot=os.environ['SLURM_ARRAY_TASK_ID'],
          manifest_sha256=digest(path), logical_tasks=tasks, failures=failures,
          policy='Each logical task runs in a fresh Python process; per-block model loading/memory lifetime remain original. Successful seals are retained; any failure makes this pool worker fail.'))
    print('POOL_SUMMARY=' + str(out / 'worker_summary.json'), flush=True)
    require(not failures, 'Some original logical tasks failed; inspect pool log and preserved attempts')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--pin', required=True)
    p.add_argument('--task', type=int)
    a = p.parse_args()
    allocated()
    m, identity, tasks = load_manifest(a.manifest, a.pin)
    if a.task is None:
        supervise(m, a.manifest, a.pin, tasks)
    else:
        require(a.task in tasks, 'Task not assigned to this scheduler slot')
        run_task(m, identity, a.manifest, a.task)


if __name__ == '__main__':
    main()
