#!/usr/bin/env python3
"""Login-node stdlib only: explicit task map, immutable source/env snapshot, sbatch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import datetime
import sys

ROOT = Path(__file__).resolve().parents[2]
VIEWS = ['knet','kiknet_borehole','kiknet_surface']
VARIANTS = ['main_d128','d64','amp16','amp32','pool4','no_source_aux']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_source(root, target, dtbench_root=None):
    root,target = Path(root),Path(target)
    identity_path = root/'SOURCE_IDENTITY.json'
    if identity_path.is_file():
        source = json.loads(identity_path.read_text())
        names = list(source['files_sha256'])
        commit = source['commit']
        for name,expected in source['files_sha256'].items():
            if digest(root/name) != expected: raise ValueError('Source package changed: '+name)
    else:
        if subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=no']).strip():
            raise ValueError('Commit tracked source changes before submission')
        commit = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD']).decode().strip()
        names = subprocess.check_output(['git','-C',str(root),'ls-tree','-r','--name-only',commit]).decode().splitlines()
        names = [n for n in names if not n.startswith(('reports/','ppt_figures/')) and
            Path(n).suffix not in ('.pth','.pt','.h5','.hdf5','.npz','.gz','.zip','.png','.pdf','.pptx')]
    target.mkdir(parents=True,exist_ok=False)
    files = {}
    for name in names:
        src = root/name
        if not src.is_file() or src.is_symlink() or root.resolve() not in src.resolve().parents:
            raise ValueError('Invalid source member '+name)
        dest = target/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dest)
        files[name] = digest(dest)
    if dtbench_root:
        dep = Path(dtbench_root)
        if not (dep/'dtbench/training/modeling.py').is_file() or not (dep/'LICENSE').is_file():
            raise ValueError('dtbench package and license required')
        for src in (dep/'dtbench').rglob('*'):
            if src.is_file() and '__pycache__' not in src.parts and src.suffix in ('.py','.yaml','.yml','.json'):
                name = 'vendor/'+str(src.relative_to(dep))
                dest = target/name
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(src,dest)
                files[name] = digest(dest)
        dest = target/'vendor/DITINGBENCH_LICENSE'
        shutil.copyfile(dep/'LICENSE',dest)
        files[str(dest.relative_to(target))] = digest(dest)
    if not (target/'vendor/dtbench/training/modeling.py').is_file():
        raise ValueError('Provide FE02_LITE_DTBENCH_ROOT or a source bundle with vendor/dtbench')
    identity = dict(commit=commit,files_sha256=files,snapshot='FE02-Lite queue submission',
                    excluded='historical reports, datasets, weights, binary artifacts')
    (target/'SOURCE_IDENTITY.json').write_text(json.dumps(identity,indent=2)+'\n')
    return identity


def task_map(view,variant,seed,stage):
    return [dict(index=i,dataset_view=v,variant=variant,seed=seed,stage=stage,
        run_id=f'fe02_lite__{v}__{stage}__{variant}__seed{seed}')
        for i,v in enumerate(VIEWS if view=='all' else [view])]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-view',choices=VIEWS+['all'],default=os.environ.get('FE02_LITE_VIEW','knet'))
    p.add_argument('--variant',choices=VARIANTS,default=os.environ.get('FE02_LITE_VARIANT','main_d128'))
    p.add_argument('--seed',type=int,choices=[42,43,44],default=int(os.environ.get('FE02_LITE_SEED','42')))
    p.add_argument('--stage',choices=['formal','pilot'],default=os.environ.get('FE02_LITE_STAGE','formal'))
    p.add_argument('--action',choices=['run','resume','eval'],default='run')
    p.add_argument('--dry-run',action='store_true')
    a = p.parse_args()
    tasks = task_map(a.dataset_view,a.variant,a.seed,a.stage)
    env = {k:v for k,v in os.environ.items() if k.startswith('FE02_LITE_')}
    output = Path(env['FE02_LITE_OUTPUT_ROOT']).resolve()
    if str(output)=='/' or output==ROOT or ROOT in output.parents and output.name in ('fe02_lite','scripts','configs'):
        raise ValueError('Use an independent output directory')
    nodes = int(env['FE02_LITE_NODES'])
    devices = int(env['FE02_LITE_DEVICES_PER_NODE'])
    concurrency = int(env['FE02_LITE_CONCURRENCY'])
    if min(nodes,devices,concurrency,int(env['FE02_LITE_CPUS_PER_TASK']))<1:
        raise ValueError('Resources/concurrency must be positive')
    global_batch,micro = (2,1) if a.stage=='pilot' else (128,8)
    if global_batch%(nodes*devices*micro):
        raise ValueError('microbatch*world must divide fixed global batch (pilot needs <=2 ranks)')
    command = ['sbatch','--parsable',
        '--partition='+env['FE02_LITE_PARTITION'],'--nodes='+str(nodes),'--ntasks-per-node='+str(devices),
        '--cpus-per-task='+env['FE02_LITE_CPUS_PER_TASK'],'--gres='+env['FE02_LITE_GRES_KIND']+':'+str(devices),
        '--time='+env['FE02_LITE_WALLTIME'],
        '--output='+str(output/'slurm/%x-%A_%a.out'),'--error='+str(output/'slurm/%x-%A_%a.err'),
        '--export=ALL']
    for key,flag in [('ACCOUNT','account'),('QOS','qos'),('MEMORY','mem')]:
        if env.get('FE02_LITE_'+key): command.append('--'+flag+'='+env['FE02_LITE_'+key])
    def task_command(task, dependency=None):
        job_name = 'fe02-lite-'+task['dataset_view']+'-'+a.stage+'-'+a.variant+'-s'+str(a.seed)
        result = command+['--job-name='+job_name,'--array='+str(task['index'])+'-'+str(task['index'])+'%1']
        if dependency: result.append('--dependency=afterany:'+dependency)
        return result
    if a.dry_run:
        print(json.dumps(dict(action=a.action,tasks=tasks,environment=env,source=str(ROOT),
            plan='Snapshot source/vendor/config/env at real submission; no files or sbatch in dry-run'),indent=2))
        for i,task in enumerate(tasks):
            dependency = '<JobID of task '+str(i-concurrency)+'>' if i>=concurrency else None
            print(shlex.join(task_command(task,dependency)+['<immutable snapshot>/scripts/fe02_lite/worker.sh',
                '<snapshot>/launch.env','<env sha256>','<tasks sha256>']))
        return
    for key in ['DATA_ROOT','SPLIT_MANIFEST','DITING_CHECKPOINT','PARTITION']:
        if not env.get('FE02_LITE_'+key): raise ValueError('Set FE02_LITE_'+key+' in the environment file')
    if not shutil.which('sbatch'): raise FileNotFoundError('sbatch unavailable; submit on the HPC login node')
    for task in tasks:
        run = output/task['run_id']
        if a.action=='run' and run.exists(): raise FileExistsError('Run exists; explicit resume/eval required: '+str(run))
        if a.action=='resume' and not (run/'last.pth').exists(): raise FileNotFoundError('Resume needs last: '+str(run))
        if a.action=='eval' and not all((run/(tag+'.pth')).is_file() for tag in ('best','last')):
            raise FileNotFoundError('Evaluation requires best and last task checkpoints: '+str(run))
    (output/'launches').mkdir(parents=True,exist_ok=True)
    (output/'slurm').mkdir(exist_ok=True)
    launch = Path(tempfile.mkdtemp(prefix='fe02-lite-',dir=output/'launches'))
    identity = snapshot_source(ROOT,launch/'source',env.get('FE02_LITE_DTBENCH_ROOT') or None)
    (launch/'tasks.json').write_text(json.dumps(tasks,indent=2)+'\n')
    env.update(FE02_LITE_CODE_ROOT=str(launch/'source'),FE02_LITE_OUTPUT_ROOT=str(output),
        FE02_LITE_TASK_MAP=str(launch/'tasks.json'),FE02_LITE_ACTION=a.action,
        FE02_LITE_SOURCE_IDENTITY_SHA256=digest(launch/'source/SOURCE_IDENTITY.json'))
    snapshot = launch/'launch.env'
    snapshot.write_text(''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in sorted(env.items())))
    snapshot.chmod(0o400)
    task_hash = digest(launch/'tasks.json')
    env_hash = digest(snapshot)
    suffix = ['--chdir='+str(launch/'source'),str(launch/'source/scripts/fe02_lite/worker.sh'),str(snapshot),env_hash,task_hash]
    submission = dict(tasks=tasks,commands=[],receipts=[],source=identity['commit'],
        source_identity_sha256=digest(launch/'source/SOURCE_IDENTITY.json'),env_sha256=env_hash,task_map_sha256=task_hash,
        concurrency_rule='afterany chains, at most configured concurrency within this submission')
    # Separate, view-named jobs preserve dataset identity in both %x logs and job names.
    # Dependency chains limit concurrency even though task indices share one snapshot.
    for i,task in enumerate(tasks):
        dependency = submission['receipts'][i-concurrency].split(';')[0] if i>=concurrency else None
        invocation = task_command(task,dependency)+suffix
        submission['commands'].append(invocation)
        (launch/'submission.json').write_text(json.dumps(submission,indent=2)+'\n')
        receipt = subprocess.check_output(invocation,text=True).strip()
        submission['receipts'].append(receipt)
        (launch/'submission.json').write_text(json.dumps(submission,indent=2)+'\n')
        with (launch/'sbatch_receipt.txt').open('a') as stream: stream.write(receipt+'\n')
        print('JobID '+task['dataset_view']+': '+receipt+'; snapshot: '+str(launch),flush=True)


if __name__=='__main__': main()
