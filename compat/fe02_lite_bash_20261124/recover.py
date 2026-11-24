#!/usr/bin/env python3
"""Re-submit one frozen v2 launch with a Bash-only worker fix, without changing its source/audit."""
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job-id', required=True, help='Original failed JobID, e.g. 35455202')
    p.add_argument('--output-root', default=os.environ.get('FE02_LITE_OUTPUT_ROOT'))
    p.add_argument('--launch', help='Optional exact original launches/fe02-lite-* directory')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if not a.output_root:
        p.error('Set FE02_LITE_OUTPUT_ROOT or --output-root')
    plan = prepare(a.output_root,a.job_id,a.launch,a.resume)
    printable = {k:v for k,v in plan.items() if k not in ('worker','environment')}
    if a.dry_run:
        printable['command'] = plan['command_prefix']+['<launcher-fix>/worker.sh','<launcher-fix>/launch.env',
                                                      plan['env_sha256'],plan['task_map_sha256']]
        print(json.dumps(printable,indent=2))
        return
    if not shutil.which('sbatch'):
        raise FileNotFoundError('sbatch unavailable; run on the HPC login node')
    parent = Path(a.output_root)/'launcher_fixes'
    parent.mkdir(parents=True,exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix='bash-array-',dir=str(parent)))
    worker_path,env_path = destination/'worker.sh',destination/'launch.env'
    worker_path.write_bytes(plan['worker']);worker_path.chmod(0o400)
    env_path.write_bytes(plan['environment']);env_path.chmod(0o400)
    command = plan['command_prefix']+[str(worker_path),str(env_path),plan['env_sha256'],plan['task_map_sha256']]
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
    print('JobID: '+result.stdout.strip()+'; launcher-fix receipt: '+str(receipt_path))


if __name__=='__main__':
    main()
