#!/usr/bin/env python3
"""Recover frozen v2 launches with a Bash-only fix, preserving their original source/audit."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

BASE_COMMIT = 'd53c3e82184d9bb3d8f85b1907d73af2ccfcf08c'
OLD_WORKER_SHA256 = 'b9c0ed88dc3a74e10cb203fdba366be9d129bfd041a28f8d34ab09b29bdbcba5'
VIEWS = ('knet','kiknet_borehole','kiknet_surface')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fixed_worker(original):
    if hashlib.sha256(original).hexdigest() != OLD_WORKER_SHA256:
        raise ValueError('Worker differs from the reported v2 source; refusing an unknown patch')
    text = original.decode()
    text = text.replace('    train_args=()\n',
        '    # Bash <=4.3 treats an empty array as unset under nounset. Keep the complete\n'
        '    # command in this array so both a fresh run and resume expand a nonempty argv.\n'
        '    train_args=(train "${args[@]}")\n', 1)
    text = text.replace('scripts/fe02_lite/run.py train "${args[@]}" "${train_args[@]}"',
                        'scripts/fe02_lite/run.py "${train_args[@]}"', 1)
    return text.encode()


def read_exports(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        fields = shlex.split(line)
        if len(fields) != 2 or fields[0] != 'export' or '=' not in fields[1]:
            raise ValueError('Expected the original static export snapshot: '+str(path))
        key, value = fields[1].split('=', 1)
        result[key] = value
    return result


def select_launch(output, job_id, launch=None):
    paths = [Path(launch)/'submission.json'] if launch else sorted(Path(output).glob('launches/fe02-lite-*/submission.json'))
    found = []
    for path in paths:
        record = json.loads(path.read_text())
        for index, receipt in enumerate(record.get('receipts', [])):
            if str(receipt).split(';')[0] == str(job_id):
                found.append((path, record, index))
    if len(found) != 1:
        raise ValueError('Expected one saved submission for JobID %s; found %s. Set --output-root or --launch.' % (job_id, len(found)))
    path, record, index = found[0]
    return path, record, record['tasks'][index], record['commands'][index]


def prepare(output, job_id, launch=None, resume=False):
    path, submission, task, command = select_launch(output, job_id, launch)
    # Original suffix: worker.sh launch.env ENV_SHA TASK_MAP_SHA.
    if len(command) < 5 or Path(command[-4]).name != 'worker.sh':
        raise ValueError('Unexpected saved sbatch command')
    env_path = Path(command[-3])
    if digest(env_path) != command[-2]:
        raise ValueError('Original frozen launch environment changed')
    exports = read_exports(env_path)
    source = Path(exports['FE02_LITE_CODE_ROOT']).resolve()
    identity_path = source/'SOURCE_IDENTITY.json'
    if digest(identity_path) != exports['FE02_LITE_SOURCE_IDENTITY_SHA256']:
        raise ValueError('Original source identity changed')
    identity = json.loads(identity_path.read_text())
    if identity['commit'] != BASE_COMMIT or submission['source'] != BASE_COMMIT:
        raise ValueError('Recovery supports the original v2 commit '+BASE_COMMIT)
    for name, expected in identity['files_sha256'].items():
        member = (source/name).resolve()
        if source not in member.parents or digest(member) != expected:
            raise ValueError('Original source file changed: '+name)
    task_path = Path(exports['FE02_LITE_TASK_MAP'])
    if digest(task_path) != command[-1]:
        raise ValueError('Original task map changed')
    if json.loads(task_path.read_text())[task['index']] != task:
        raise ValueError('Original task/index mapping changed')
    if Path(exports['FE02_LITE_OUTPUT_ROOT']).resolve() != Path(output).resolve():
        raise ValueError('Output root differs from original launch')
    run = Path(output)/task['run_id']
    if (run/'COMPLETED.json').exists():
        raise FileExistsError('Completed run is preserved: '+str(run))
    action = 'resume' if resume or exports['FE02_LITE_ACTION']=='resume' else 'run'
    if exports['FE02_LITE_ACTION'] not in ('run','resume'):
        raise ValueError('This fix only applies to a training launch')
    if action == 'resume' and not (run/'last.pth').is_file():
        raise FileNotFoundError('Resume requires last.pth; never best: '+str(run))
    if action == 'run' and run.exists():
        raise FileExistsError('Run exists; preserve it and use --resume only with last.pth: '+str(run))
    worker = fixed_worker((source/'scripts/fe02_lite/worker.sh').read_bytes())
    prefix = []
    for argument in command[:-4]:
        if argument.startswith('--dependency='):
            continue
        if argument.startswith('--job-name='):
            argument += '-bashfix'
        prefix.append(argument)
    env_bytes = env_path.read_bytes()
    if action != exports['FE02_LITE_ACTION']:
        env_bytes += b'\nexport FE02_LITE_ACTION=resume\n'
    return dict(original_job_id=str(job_id),original_submission=str(path),source_root=str(source),
        source_commit=BASE_COMMIT,task=task,action=action,command_prefix=prefix,
        task_map_sha256=command[-1],source_identity_sha256=digest(identity_path),
        original_env_sha256=digest(env_path),worker_sha256=hashlib.sha256(worker).hexdigest(),
        env_sha256=hashlib.sha256(env_bytes).hexdigest(),worker=worker,environment=env_bytes,
        invariants='Original source, SOURCE_IDENTITY, audit lock, config, model and existing logs remain unchanged')


def submitted_repairs(output, run_id):
    """A receipt proves re-submission, not the current Slurm state."""
    result = []
    for path in sorted(Path(output).glob('launcher_fixes/bash-array-*/launcher_fix.json')):
        receipt = json.loads(path.read_text())
        if receipt.get('task',{}).get('run_id') != run_id:
            continue
        if receipt.get('status') == 'SUBMITTED':
            job = receipt['sbatch_stdout'].strip().split(';')[0]
            result.append(dict(job_id=job,receipt=str(path)))
        elif receipt.get('status') == 'PREPARED':
            raise ValueError('A prior repair has an uncertain submission outcome; inspect '+str(path))
    return result


def queued_jobs(job_ids):
    """Read-only queue check for dependencies; keep dependencies if unavailable."""
    active = []
    for job in dict.fromkeys(job_ids):
        if not shutil.which('squeue'):
            active.append(job)
            continue
        result = subprocess.run(['squeue','-h','-j',job,'-o','%i'],capture_output=True,text=True)
        if result.returncode:
            if 'Invalid job id' not in result.stderr:
                active.append(job)
        elif any(line.strip().split('_')[0] == job for line in result.stdout.splitlines()):
            active.append(job)
    return active


def prepare_all(output, anchor_job=None, launch=None, resume=False):
    scope = dict(stage='formal',variant='main_d128',seed=42)
    if anchor_job:
        _,_,anchor,_ = select_launch(output,anchor_job,launch)
        scope = {k:anchor[k] for k in scope}
    candidates = {}
    # Search all original submissions, including three separately invoked submit scripts.
    for path in sorted(Path(output).glob('launches/fe02-lite-*/submission.json')):
        record = json.loads(path.read_text())
        if record.get('source') != BASE_COMMIT:
            continue
        for index,receipt in enumerate(record.get('receipts',[])):
            task = record['tasks'][index]
            view = task['dataset_view']
            if view not in VIEWS or any(task.get(k) != value for k,value in scope.items()):
                continue
            priority = (path.stat().st_mtime_ns,str(path),index)
            if view not in candidates or priority > candidates[view][0]:
                candidates[view] = (priority,path,task,str(receipt).split(';')[0],record['commands'][index])
    missing = [v for v in VIEWS if v not in candidates]
    if missing:
        raise ValueError('Batch recovery requires all three original views; missing '+', '.join(missing)+
                         '. Nothing submitted. Check --output-root.')
    rows,plans,prior_jobs,original_jobs,limits = [],[],[],[],[]
    for view in VIEWS:
        _,path,task,job,command = candidates[view]
        exports = read_exports(command[-3])
        limits.append(int(exports.get('FE02_LITE_CONCURRENCY','1')))
        row = dict(dataset_view=view,run_id=task['run_id'],original_job_id=job,original_submission=str(path))
        run = Path(output)/task['run_id']
        if (run/'COMPLETED.json').exists():
            row.update(status='SKIPPED_COMPLETED')
        else:
            previous = submitted_repairs(output,task['run_id'])
            if previous:
                row.update(status='SKIPPED_ALREADY_RESUBMITTED',previous_repairs=previous,
                           note='Receipt proves submission only; current state is not inferred')
                prior_jobs.extend(r['job_id'] for r in previous)
            else:
                plan = prepare(output,job,path.parent,resume or (run/'last.pth').is_file())
                row.update(status='PLANNED',action=plan['action'])
                plans.append(plan)
                original_jobs.append(job)
        rows.append(row)
    if min(limits)<1:
        raise ValueError('Invalid original concurrency')
    return dict(scope=scope,rows=rows,plans=plans,concurrency=min(limits),
                previous_repair_jobs=prior_jobs,original_jobs=original_jobs,
                selection_rule='latest original submission per view within stage/variant/seed; require all three views')


def submit_plan(output, plan, dependencies=()):
    printable = {k:v for k,v in plan.items() if k not in ('worker','environment')}
    parent = Path(output)/'launcher_fixes'
    parent.mkdir(parents=True,exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix='bash-array-',dir=str(parent)))
    worker_path,env_path = destination/'worker.sh',destination/'launch.env'
    worker_path.write_bytes(plan['worker']);worker_path.chmod(0o400)
    env_path.write_bytes(plan['environment']);env_path.chmod(0o400)
    command = list(plan['command_prefix'])
    if dependencies:
        command.append('--dependency=afterany:'+':'.join(dict.fromkeys(dependencies)))
    command += [str(worker_path),str(env_path),plan['env_sha256'],plan['task_map_sha256']]
    receipt_path = destination/'launcher_fix.json'
    printable.update(command=command,patch_tool_sha256=digest(__file__),
                     prepared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PREPARED')
    receipt_path.write_text(json.dumps(printable,indent=2)+'\n')
    result = subprocess.run(command,capture_output=True,text=True)
    printable.update(sbatch_stdout=result.stdout,sbatch_stderr=result.stderr,
                     status='SUBMITTED' if result.returncode==0 else 'SUBMISSION_FAILED')
    receipt_path.write_text(json.dumps(printable,indent=2)+'\n')
    if result.returncode:
        raise RuntimeError('sbatch failed; original files preserved. See '+str(receipt_path))
    job_id = result.stdout.strip().split(';')[0]
    print('JobID '+plan['task']['dataset_view']+': '+result.stdout.strip()+'; launcher-fix receipt: '+str(receipt_path),flush=True)
    return dict(job_id=job_id,receipt=str(receipt_path),dataset_view=plan['task']['dataset_view'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job-id', help='Original JobID; with --all, anchors stage/variant/seed')
    p.add_argument('--all', action='store_true', help='Recover all three views, including separate original submissions')
    p.add_argument('--output-root', default=os.environ.get('FE02_LITE_OUTPUT_ROOT'))
    p.add_argument('--launch', help='Optional exact original launch directory for --job-id')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if not a.output_root or not (a.job_id or a.all):
        p.error('Set --output-root/FE02_LITE_OUTPUT_ROOT and provide --all or --job-id')
    if not a.all:
        plan = prepare(a.output_root,a.job_id,a.launch,a.resume)
        if a.dry_run:
            printable = {k:v for k,v in plan.items() if k not in ('worker','environment')}
            printable['command'] = plan['command_prefix']+['<launcher-fix>/worker.sh','<launcher-fix>/launch.env',
                                                          plan['env_sha256'],plan['task_map_sha256']]
            print(json.dumps(printable,indent=2))
            return
        if not shutil.which('sbatch'):
            raise FileNotFoundError('sbatch unavailable; run on the HPC login node')
        submit_plan(a.output_root,plan)
        return
    batch = prepare_all(a.output_root,a.job_id,a.launch,a.resume)
    printable = {k:v for k,v in batch.items() if k != 'plans'}
    if a.dry_run:
        printable['note'] = 'No writes, queue queries or sbatch; submission will refresh dependency state'
        print(json.dumps(printable,indent=2))
        return
    for row in batch['rows']:
        print(row['dataset_view']+': '+row['status'],flush=True)
    if not batch['plans']:
        print('No new submissions needed; completed or previously resubmitted views preserved.')
        return
    if not shutil.which('sbatch'):
        raise FileNotFoundError('sbatch unavailable; run on the HPC login node')
    # Preserve the default one-job resource limit, including an already re-submitted KNET job.
    previous_active = queued_jobs(batch['previous_repair_jobs'])
    originals_active = queued_jobs(batch['original_jobs'])
    concurrency = batch['concurrency']
    if len(previous_active)>concurrency:
        slots = [list(previous_active) for _ in range(concurrency)]
    else:
        slots = [[] for _ in range(concurrency-len(previous_active))]+[[j] for j in previous_active]
    parent = Path(a.output_root)/'launcher_fixes';parent.mkdir(parents=True,exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='bash-batch-',dir=str(parent)))
    receipt_path = folder/'batch_fix.json'
    printable.update(active_prior_repairs=previous_active,active_original_jobs=originals_active,submitted=[],status='PREPARED')
    receipt_path.write_text(json.dumps(printable,indent=2)+'\n')
    for i,plan in enumerate(batch['plans']):
        # Wait for any original audits/queued attempts, without cancelling or editing them.
        dependencies = slots[i%concurrency]+originals_active
        try:
            receipt = submit_plan(a.output_root,plan,dependencies)
        except Exception as exc:
            printable.update(status='PARTIAL_SUBMISSION_FAILED',error=str(exc))
            receipt_path.write_text(json.dumps(printable,indent=2)+'\n')
            raise
        printable['submitted'].append(receipt)
        slots[i%concurrency] = [receipt['job_id']]
        receipt_path.write_text(json.dumps(printable,indent=2)+'\n')
    printable['status'] = 'SUBMITTED'
    receipt_path.write_text(json.dumps(printable,indent=2)+'\n')
    print('Batch recovery receipt: '+str(receipt_path))


if __name__=='__main__':
    main()
