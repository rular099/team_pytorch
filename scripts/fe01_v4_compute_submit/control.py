"""Compute-only coordinator; explicit requested stages, small worker pools."""
import argparse
from collections import Counter
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid
from support import (ACTIVE, ALLOWED, RETRY, ROOT, allocated, digest, fingerprint,
                     read, require, revision, validate_plan, write)


def scheduler_states(session):
    """Keep scheduler slot IDs separate from original logical task indices."""
    states = {}
    for path in sorted((Path(session) / 'jobs').glob('*.json')):
        record = read(path)
        job = str(record['job_id'])
        require(bool(re.fullmatch(r'\d+', job)), 'Invalid stored scheduler ID')
        found = {}
        for command in (['squeue', '-h', '-r', '-j', job, '-o', '%i|%T'],
                        ['sacct', '-n', '-P', '-j', job, '--format=JobID,State,ExitCode']):
            try:
                r = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
            except (OSError, subprocess.TimeoutExpired):
                continue
            if r.returncode:
                continue
            for line in r.stdout.splitlines():
                fields = line.strip().split('|')
                m = re.fullmatch(re.escape(job) + r'_(\d+)', fields[0]) if len(fields) >= 2 else None
                if m:
                    found[int(m[1])] = fields[1].split()[0].rstrip('+')
        mapping = record.get('worker_task_map', {str(i): [i] for i in record['indices']})
        require(sorted(i for values in mapping.values() for i in values) == sorted(record['indices']),
                'Stored worker mapping differs from logical task list')
        for slot, tasks in mapping.items():
            for task in tasks:
                states[(record['stage'], task)] = dict(status=found.get(int(slot), 'UNKNOWN'), job_id=job,
                                                      scheduler_slot=int(slot),
                                                      scheduler_state_scope='pool assignment; logical task may wait within worker'
                                                      if 'worker_task_map' in record else 'original array task')
    return states


def task_states(session, plan, stage, queue):
    from fe01_a01_compare import orchestration as ops
    from fe01_a01_compare.io import verify_seal
    result, base = {}, None
    for task in plan['tasks'][stage]:
        i = task['index']
        path = Path(session) / 'stages' / stage / str(i)
        if path.exists():
            if base is None:
                base = ops.stage_identity(session, stage, i)
            verify_seal(path, dict(base, task=i))
            result[i] = 'COMPLETE_SEAL_VERIFIED'
        else:
            result[i] = queue.get((stage, i), {}).get('status', 'NOT_SUBMITTED')
    return result


def choose_tasks(states):
    remaining = {i: s for i, s in states.items() if s != 'COMPLETE_SEAL_VERIFIED'}
    require(all(s in ACTIVE | RETRY for s in remaining.values()),
            'UNKNOWN or unrecognized task state: reconcile scheduler evidence before retry')
    if any(s in ACTIVE for s in remaining.values()):
        return []  # One pool per stage at a time, preserving its total concurrency cap.
    return sorted(remaining)


def pools(indices, workers):
    require(workers > 0 and len(indices) == len(set(indices)), 'Invalid worker/task set')
    count = min(workers, len(indices))
    return {str(i): indices[i::count] for i in range(count)}


def submit_checked(command, intent):
    """No swallowed stderr, no retry after an ambiguous scheduler response."""
    write(intent / 'intent.json', dict(command=command, status='SUBMISSION_INTENT'))
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except BaseException as exc:
        write(intent / 'result.json', dict(status='UNKNOWN', reason=str(exc)))
        raise
    answer = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    if result.returncode:
        write(intent / 'result.json', dict(answer, status='REJECTED'))
        print('SBATCH_REJECTED', 'returncode=' + str(result.returncode), flush=True)
        print(result.stderr or '(empty sbatch stderr)', file=sys.stderr, flush=True)
        raise RuntimeError('sbatch rejected request; exact stdout/stderr saved in ' + str(intent))
    job = result.stdout.strip().split(';')[0]
    if not re.fullmatch(r'\d+', job):
        write(intent / 'result.json', dict(answer, status='UNKNOWN'))
        raise RuntimeError('Unrecognized sbatch response; do not resubmit blindly: ' + str(intent))
    write(intent / 'result.json', dict(answer, status='ACCEPTED', job_id=job))
    return job


def unresolved_requests(session, stage):
    for p in (Path(session) / 'operational_submit').glob('control_*/*/pool_manifest.json'):
        manifest = read(p)
        if manifest['stage'] != stage:
            continue
        result = p.parent / 'result.json'
        if not (p.parent / 'intent.json').exists():
            continue
        require(result.exists() and read(result)['status'] != 'UNKNOWN',
                'Unresolved prior submission: ' + str(p.parent))
        r = read(result)
        if r['status'] == 'ACCEPTED':
            require(any(read(j).get('pool_manifest_sha256') == digest(p)
                        for j in (Path(session) / 'jobs').glob('*.json')),
                    'Accepted pool has no job receipt: reconcile ' + str(p.parent))


def effective_workers(requested, out):
    try:
        result = subprocess.run(['scontrol', 'show', 'config'], capture_output=True, text=True, check=False, timeout=20)
        limits = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        limits = dict(returncode=None, stdout='', stderr=str(exc))
    write(out / 'scheduler_config.json', limits)
    ceilings = [int(v) for v in re.findall(r'\b(?:MaxArraySize|max_array_tasks)\s*=\s*(\d+)', limits['stdout'], re.I)]
    count = min([requested] + ceilings)
    require(count > 0, 'Job arrays disabled or invalid worker count')
    return count


def submit_pool(session, plan, stage, workers, out, identity):
    from fe01_a01_compare import orchestration as ops
    require(stage in ALLOWED, 'Stage not allowed by this operational entry')
    ops.prerequisites(session, stage)
    unresolved_requests(session, stage)
    queue = scheduler_states(session)
    states = task_states(session, plan, stage, queue)
    chosen = choose_tasks(states)
    write(out / 'states_before.json', dict(stage=stage, counts=dict(Counter(states.values()))))
    if not chosen:
        print('NO_SUBMISSION', stage, dict(Counter(states.values())), flush=True)
        return None
    mapping = pools(chosen, workers)
    dcustage = stage in {'replay_stations', 'replay_grid_normal', 'latency'}
    manifest = dict(stage=stage, session=str(Path(session).resolve()), env_file=plan['env_file'],
                    plan_sha256=digest(Path(session) / 'plan.json'), worker_task_map=mapping,
                    logical_tasks_sha256={str(i): fingerprint(plan['tasks'][stage][i]) for i in chosen},
                    revision=identity, device='cuda' if dcustage else 'cpu',
                    purpose='Original tasks unchanged; each scheduler slot executes a list of original logical tasks in fresh Python processes')
    write(out / 'pool_manifest.json', manifest)
    write(out / 'pool_manifest.pin.json', dict(sha256=digest(out / 'pool_manifest.json')))
    command = ['sbatch', '--parsable', '--export=ALL', '--chdir=' + os.environ['ON_EVAL_RELEASE_ROOT'], '--nodes=1', '--ntasks=1',
               '--job-name=onv4_pool_' + stage, '--array=0-' + str(len(mapping)-1) + '%' + str(len(mapping)),
               '--cpus-per-task=' + os.environ.get('ON_EVAL_CPUS', '4'), '--mem=' + os.environ.get('ON_EVAL_MEM', '64G'),
               '--time=' + os.environ.get('ON_EVAL_TIME', '24:00:00'),
               '--output=' + str(Path(session) / 'slurm' / (stage + '_pool_%A_%a.out')),
               '--error=' + str(Path(session) / 'slurm' / (stage + '_pool_%A_%a.err'))]
    partition = os.environ.get('ON_EVAL_PARTITION', '') if dcustage else os.environ.get('ON_EVAL_CPU_PARTITION', os.environ.get('ON_EVAL_PARTITION', ''))
    for key, value in (('partition', partition), ('account', os.environ.get('ON_EVAL_ACCOUNT', '')),
                       ('exclude', os.environ.get('ON_EVAL_EXCLUDE_NODES', ''))):
        if value:
            command.append('--' + key + '=' + value)
    if dcustage:
        command.append('--gres=' + os.environ['ON_EVAL_GRES'])
    command += [str(ROOT / 'scripts/fe01_v4_compute_submit/pool.sbatch'), str(out / 'pool_manifest.json'),
                str(out / 'pool_manifest.pin.json'), plan['env_file'], str(ROOT), manifest['device']]
    job = submit_checked(command, out)
    write(Path(session) / 'jobs' / (dt.datetime.now().strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex + '.json'),
          dict(stage=stage, indices=chosen, worker_task_map=mapping, job_id=job, command=command,
               pool_manifest_sha256=digest(out / 'pool_manifest.json'), runtime_revision_commit=identity['commit'],
               runtime_revision_sha256=identity['revision_sha256'], status='SUBMITTED_NOT_COMPLETED'))
    print('SUBMITTED_POOL', stage, 'job=' + job, 'workers=' + str(len(mapping)), 'logical_tasks=' + str(len(chosen)), flush=True)
    return job


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--status', action='store_true')
    p.add_argument('--stage', action='append', choices=ALLOWED)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    allocated()
    require(0 < a.workers <= 6, 'Use 1-6 workers per stage; latency remains a separate existing job')
    identity = revision()
    from fe01_a01_compare import orchestration as ops
    session = ops.session_path(os.environ['FE01_V4_RUN_ID'])
    plan = ops.verify_plan(session)
    validate_plan(plan)
    out = session / 'operational_submit' / ('control_' + os.environ['SLURM_JOB_ID'])
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'operational_identity.json', identity)
    print('CONTROL_EVIDENCE=' + str(out), flush=True)
    failures = []
    with (session / 'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if a.status:
            require(not a.stage, 'Status never submits inference stages')
            print('Counts below are logical tasks. Pool RUNNING/PENDING means assignment to a live scheduler slot, not simultaneous execution of every logical task.', flush=True)
            queue = scheduler_states(session)
            rows = []
            for stage in plan['tasks']:
                states = task_states(session, plan, stage, queue)
                print(stage, json.dumps(dict(Counter(states.values()))), flush=True)
                rows.extend(dict(stage=stage, index=i, status=s, **{k:v for k,v in queue.get((stage,i),{}).items() if k!='status'}) for i,s in states.items())
            write(out / 'status.json', rows)
        else:
            workers = effective_workers(a.workers, out)
            for stage in a.stage or ['replay_stations', 'replay_grid_normal']:
                stage_out = out / stage
                stage_out.mkdir()
                try:
                    submit_pool(session, plan, stage, workers, stage_out, identity)
                except Exception as exc:
                    failures.append(dict(stage=stage, type=type(exc).__name__, reason=str(exc)))
                    print('STAGE_SUBMISSION_FAILED', stage, str(exc), file=sys.stderr, flush=True)
            write(out / 'summary.json', dict(failures=failures, latency='not submitted by default; preserve any existing latency job'))
    require('torch' not in sys.modules, 'Coordinator must not import model libraries')
    require(not failures, 'One or more submissions failed; see exact scheduler stderr and summary.json')


if __name__ == '__main__':
    main()
