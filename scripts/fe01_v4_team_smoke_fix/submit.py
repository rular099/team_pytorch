"""Explicit sbatch for task 3; original plan/source and other tasks stay unchanged."""
import argparse
import datetime as dt
import fcntl
import os
from pathlib import Path
import subprocess
import uuid
import re
from fix_support import TASK, ROOT, operational_identity, require, validate_plan, write


def submit(session, env_file):
    from fe01_a01_compare import orchestration as ops
    env_file = Path(env_file).resolve()
    revision = operational_identity()
    plan = ops.verify_plan(session)
    validate_plan(plan, os.environ['ON_EVAL_RELEASE_ROOT'])
    require(Path(plan['env_file']).resolve() == Path(env_file).resolve(), 'Private environment differs from run plan')
    ops.prerequisites(session, 'model_smoke')
    with (Path(session) / 'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if ops.completed(session, 'model_smoke', TASK):
            print('NO_SUBMISSION: TEAM model_smoke already complete and seal verified')
            return None
        state = ops.scheduler_states(session).get(('model_smoke', TASK), {}).get('status', 'NOT_SUBMITTED')
        if state in {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'RESIZING'}:
            print('NO_SUBMISSION: TEAM model_smoke ' + state)
            return None
        require(state in {'NOT_SUBMITTED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL',
                          'PREEMPTED', 'BOOT_FAIL'}, 'Refuse duplicate/unknown TEAM task: ' + state)
        args = ['sbatch', '--parsable', '--export=ALL', '--chdir=' + str(ROOT), '--job-name=onv4_TEAM_smoke_fix',
                '--nodes=1', '--ntasks=1', '--array=3%1', '--gres=' + os.environ['ON_EVAL_GRES'],
                '--cpus-per-task=' + os.environ.get('ON_EVAL_CPUS', '4'), '--mem=' + os.environ.get('ON_EVAL_MEM', '64G'),
                '--time=' + os.environ.get('ON_EVAL_TIME', '24:00:00'),
                '--output=' + str(Path(session) / 'slurm/model_smoke_%A_%a.out'),
                '--error=' + str(Path(session) / 'slurm/model_smoke_%A_%a.err')]
        for option, value in [('partition', os.environ.get('ON_EVAL_PARTITION', '')),
                              ('account', os.environ.get('ON_EVAL_ACCOUNT', '')),
                              ('exclude', os.environ.get('ON_EVAL_EXCLUDE_NODES', ''))]:
            if value:
                args.append('--' + option + '=' + value)
        args += [str(ROOT / 'scripts/fe01_v4_team_smoke_fix/worker.sbatch'), str(session), str(env_file), str(ROOT)]
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        job = result.stdout.strip().split(';')[0]
        require(bool(re.fullmatch(r'\d+', job)), 'Unrecognized sbatch job ID: ' + result.stdout)
        write(Path(session) / 'jobs' / (dt.datetime.now().strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex + '.json'),
              dict(stage='model_smoke', indices=[TASK], job_id=job, command=args, status='SUBMITTED_NOT_COMPLETED',
                   runtime_revision_sha256=revision['revision_sha256'], runtime_revision_commit=revision['commit']))
        print('SUBMITTED TEAM model_smoke task=3 job=' + job)
        print('Logs=' + str(Path(session) / 'slurm') + '/model_smoke_' + job + '_3.err/.out')
        print('Original run/status/next-stage scripts remain applicable; no other task submitted.')
        return job


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env', required=True)
    p.add_argument('--run-id', required=True)
    a = p.parse_args()
    from fe01_a01_compare.orchestration import session_path
    submit(session_path(a.run_id), a.env)


if __name__ == '__main__':
    main()
