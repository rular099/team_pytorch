"""Execute the complete worker with fake Slurm/model commands; no real jobs/data."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('lite_bash_recover',ROOT/'compat/fe02_lite_bash_20261124/recover.py')
recover = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recover)
SHELLS = [shutil.which('bash')]
if os.environ.get('FE02_LITE_TEST_LEGACY_BASH'):
    SHELLS.append(os.environ['FE02_LITE_TEST_LEGACY_BASH'])


def executable(path, content):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)
    return path


@pytest.fixture
def launch(tmp_path):
    output = tmp_path/'outputs with spaces'
    old = output/'launches/fe02-lite-original'
    source = old/'source'
    binary = tmp_path/'bin'
    binary.mkdir()
    fake_python = executable(binary/'fake-python', '#!'+sys.executable+'''\nimport json,os,sys
from pathlib import Path
args=sys.argv[1:]
with open(os.environ['FAKE_COMMAND_LOG'],'a') as stream:
    stream.write(json.dumps(dict(program='python',argv=args))+'\\n')
if args[0]=='-':
    sys.argv=args
    exec(compile(sys.stdin.read(),'<worker-stdin>','exec'))
elif len(args)>1 and args[0].endswith('/run.py'):
    if args[1]==os.environ.get('FAKE_FAIL_STAGE'): raise SystemExit(17)
    if args[1]=='train': Path(os.environ['FAKE_RUN_DIR']).mkdir(parents=True,exist_ok=True)
''')
    executable(binary/'scontrol','#!/bin/sh\nprintf "synthetic-node\\n"\n')
    executable(binary/'srun','#!'+sys.executable+'''\nimport json,os,sys
args=sys.argv[1:]
with open(os.environ['FAKE_COMMAND_LOG'],'a') as stream:
    stream.write(json.dumps(dict(program='srun',argv=args))+'\\n')
while args and args[0].startswith('--'): args=args[1:]
os.execvp(args[0],args)
''')
    old_worker = subprocess.check_output(['git','show',recover.BASE_COMMIT+':scripts/fe02_lite/worker.sh'],cwd=ROOT)
    worker = source/'scripts/fe02_lite/worker.sh'
    worker.parent.mkdir(parents=True)
    worker.write_bytes(old_worker)
    (source/'scripts/fe02_lite/env.sh').write_text('export FE02_LITE_PYTHON='+shlex.quote(str(fake_python))+'\ncd "$FE02_LITE_CODE_ROOT"\n')
    (source/'scientific.py').write_text('# synthetic scientific-source sentinel\n')
    members = {str(p.relative_to(source)):recover.digest(p) for p in source.rglob('*') if p.is_file()}
    identity = source/'SOURCE_IDENTITY.json'
    identity.write_text(json.dumps(dict(commit=recover.BASE_COMMIT,files_sha256=members)))
    run_id = 'fe02_lite__knet__formal__main_d128__seed42'
    task = dict(index=0,dataset_view='knet',variant='main_d128',seed=42,stage='formal',run_id=run_id)
    tasks = old/'tasks.json';tasks.write_text(json.dumps([task]))
    exports = dict(FE02_LITE_CODE_ROOT=str(source),FE02_LITE_OUTPUT_ROOT=str(output),FE02_LITE_TASK_MAP=str(tasks),
        FE02_LITE_SOURCE_IDENTITY_SHA256=recover.digest(identity),FE02_LITE_ACTION='run',
        FE02_LITE_DEVICES_PER_NODE='1',FE02_LITE_DITING_CHECKPOINT='/synthetic/external-checkpoint')
    frozen_env = old/'launch.env'
    frozen_env.write_text(''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in exports.items()))
    command = ['sbatch','--parsable','--job-name=fe02-lite-knet-formal-main_d128-s42','--partition=synthetic',
        '--nodes=1','--ntasks-per-node=1','--array=0-0%1','--dependency=afterany:111',
        '--output='+str(output/'slurm/%x-%A_%a.out'),'--chdir='+str(source),
        str(worker),str(frozen_env),recover.digest(frozen_env),recover.digest(tasks)]
    (old/'submission.json').write_text(json.dumps(dict(source=recover.BASE_COMMIT,tasks=[task],commands=[command],receipts=['35455202'])))
    audit = output/'audits'/run_id/'protocol.lock.json'
    audit.parent.mkdir(parents=True);audit.write_text(json.dumps(dict(status='AUDIT_PASS',code_sha256='synthetic-original-lock')))
    env = dict(os.environ,PATH=str(binary)+':'+os.environ['PATH'],SLURM_JOB_ID='99901',SLURM_ARRAY_TASK_ID='0',
        SLURM_JOB_NODELIST='synthetic-node',SLURM_NTASKS='1',SLURM_PROCID='0',SLURM_LOCALID='0',
        FAKE_COMMAND_LOG=str(tmp_path/'commands.jsonl'),FAKE_RUN_DIR=str(output/run_id))
    return dict(output=output,old=old,source=source,binary=binary,worker=worker,env_file=frozen_env,tasks=tasks,
                command=command,exports=exports,env=env,run=output/run_id,audit=audit)


@pytest.mark.parametrize('shell',SHELLS)
@pytest.mark.parametrize('action',['run','resume','eval'])
def test_worker_argv_and_modes(launch,shell,action):
    data=launch
    exports=dict(data['exports'],FE02_LITE_ACTION=action)
    data['env_file'].write_text(''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in exports.items()))
    if action in ('resume','eval'):
        data['run'].mkdir();(data['run']/'last.pth').write_text('synthetic last')
    result=subprocess.run([shell,str(ROOT/'scripts/fe02_lite/worker.sh'),str(data['env_file']),
        recover.digest(data['env_file']),recover.digest(data['tasks'])],env=data['env'],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    commands=[json.loads(line) for line in Path(data['env']['FAKE_COMMAND_LOG']).read_text().splitlines()]
    training=[c for c in commands if c['program']=='python' and c['argv'][:2]==['scripts/fe02_lite/run.py','train']]
    assert len(training)==(0 if action=='eval' else 1)
    if training:
        assert '' not in training[0]['argv']
        assert training[0]['argv'].count('--resume')==(1 if action=='resume' else 0)
        assert '--dataset-view' in training[0]['argv'] and 'knet' in training[0]['argv']
    evaluation=[c for c in commands if c['program']=='python' and c['argv'][:2]==['scripts/fe02_lite/run.py','eval']]
    assert len(evaluation)==2


@pytest.mark.parametrize('shell',SHELLS)
def test_audit_failure_never_reaches_srun(launch,shell):
    result=subprocess.run([shell,str(ROOT/'scripts/fe02_lite/worker.sh'),str(launch['env_file']),
        recover.digest(launch['env_file']),recover.digest(launch['tasks'])],
        env=dict(launch['env'],FAKE_FAIL_STAGE='audit'),capture_output=True,text=True)
    assert result.returncode==17
    assert '"program": "srun"' not in Path(launch['env']['FAKE_COMMAND_LOG']).read_text()


def test_legacy_failure_is_reproduced(launch):
    legacy=os.environ.get('FE02_LITE_TEST_LEGACY_BASH')
    if not legacy:
        pytest.skip('Set FE02_LITE_TEST_LEGACY_BASH to an actual Bash <=4.3 for reproduction')
    result=subprocess.run([legacy,str(launch['worker']),str(launch['env_file']),
        recover.digest(launch['env_file']),recover.digest(launch['tasks'])],env=launch['env'],capture_output=True,text=True)
    assert result.returncode!=0 and 'train_args[@]: unbound variable' in result.stderr
    assert '"program": "srun"' not in Path(launch['env']['FAKE_COMMAND_LOG']).read_text()


def test_recovery_preserves_frozen_source_and_audit(launch,monkeypatch,capsys):
    original={str(p):p.read_bytes() for p in launch['source'].rglob('*') if p.is_file()}
    original[str(launch['env_file'])]=launch['env_file'].read_bytes()
    original[str(launch['audit'])]=launch['audit'].read_bytes()
    plan=recover.prepare(launch['output'],'35455202')
    assert plan['worker']==(ROOT/'scripts/fe02_lite/worker.sh').read_bytes()
    assert not any(a.startswith('--dependency=') for a in plan['command_prefix'])
    monkeypatch.setattr(sys,'argv',['recover.py','--output-root',str(launch['output']),'--job-id','35455202','--dry-run'])
    recover.main();assert not (launch['output']/'launcher_fixes').exists()
    fake=executable(launch['binary']/'sbatch','#!/bin/sh\nprintf "99901\\n"\n')
    monkeypatch.setenv('PATH',str(fake.parent)+':'+os.environ['PATH'])
    monkeypatch.setattr(sys,'argv',['recover.py','--output-root',str(launch['output']),'--job-id','35455202'])
    recover.main()
    fix=next((launch['output']/'launcher_fixes').iterdir())
    receipt=json.loads((fix/'launcher_fix.json').read_text())
    assert receipt['original_job_id']=='35455202' and receipt['sbatch_stdout'].strip()=='99901'
    assert receipt['source_root']==str(launch['source'].resolve())
    assert (fix/'worker.sh').read_bytes()==plan['worker']
    for path,content in original.items():assert Path(path).read_bytes()==content
    # This standalone fixed worker runs against the unchanged v2 source tree.
    shell=os.environ.get('FE02_LITE_TEST_LEGACY_BASH') or shutil.which('bash')
    result=subprocess.run([shell,*receipt['command'][-4:]],env=launch['env'],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_recovery_checks_completed_changed_and_resume(launch):
    launch['run'].mkdir();(launch['run']/'best.pth').write_text('preserve best')
    with pytest.raises(FileExistsError):recover.prepare(launch['output'],'35455202')
    with pytest.raises(FileNotFoundError):recover.prepare(launch['output'],'35455202',resume=True)
    (launch['run']/'last.pth').write_text('synthetic last')
    plan=recover.prepare(launch['output'],'35455202',resume=True)
    assert plan['environment'].endswith(b'export FE02_LITE_ACTION=resume\n')
    assert plan['action']=='resume'
    (launch['run']/'COMPLETED.json').write_text('{}')
    with pytest.raises(FileExistsError):recover.prepare(launch['output'],'35455202',resume=True)
    (launch['source']/'scientific.py').write_text('changed')
    with pytest.raises(ValueError,match='source file changed'):recover.prepare(launch['output'],'35455202')


def three_views(data, separate=True, concurrency=1):
    """Both one submit_all invocation and three separate submit scripts are real supported paths."""
    tasks=[]
    for i,view in enumerate(recover.VIEWS):
        tasks.append(dict(index=0 if separate else i,dataset_view=view,variant='main_d128',seed=42,stage='formal',
                          run_id='fe02_lite__'+view+'__formal__main_d128__seed42'))
    if separate:
        locations=[data['old']]+[data['output']/'launches'/('fe02-lite-'+v) for v in recover.VIEWS[1:]]
        for location,task in zip(locations,tasks):
            if location!=data['old']:
                shutil.copytree(data['source'],location/'source')
            source=location/'source'
            task_path=location/'tasks.json';task_path.write_text(json.dumps([task]))
            env_path=location/'launch.env'
            exports=dict(data['exports'],FE02_LITE_CODE_ROOT=str(source),FE02_LITE_TASK_MAP=str(task_path),
                         FE02_LITE_CONCURRENCY=str(concurrency))
            env_path.write_text(''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in exports.items()))
            command=[a.replace('fe02-lite-knet-','fe02-lite-'+task['dataset_view']+'-') for a in data['command'][:-4]]
            command += [str(source/'scripts/fe02_lite/worker.sh'),str(env_path),recover.digest(env_path),recover.digest(task_path)]
            job='3545520'+str(2+tasks.index(task))
            (location/'submission.json').write_text(json.dumps(dict(source=recover.BASE_COMMIT,tasks=[task],commands=[command],receipts=[job])))
    else:
        data['tasks'].write_text(json.dumps(tasks))
        exports=dict(data['exports'],FE02_LITE_CONCURRENCY=str(concurrency))
        data['env_file'].write_text(''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in exports.items()))
        commands=[]
        for i,task in enumerate(tasks):
            command=[]
            for a in data['command'][:-4]:
                if a.startswith('--array='):a='--array='+str(i)+'-'+str(i)+'%1'
                command.append(a.replace('fe02-lite-knet-','fe02-lite-'+task['dataset_view']+'-'))
            commands.append(command+[str(data['worker']),str(data['env_file']),recover.digest(data['env_file']),recover.digest(data['tasks'])])
        (data['old']/'submission.json').write_text(json.dumps(dict(source=recover.BASE_COMMIT,tasks=tasks,commands=commands,
                                                               receipts=['35455202','35455203','35455204'])))


def scheduler(data,monkeypatch,queued=()):
    executable(data['binary']/'sbatch','#!'+sys.executable+'''\nimport json,os,sys
from pathlib import Path
path=Path(os.environ['FAKE_SBATCH_LOG'])
rows=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
with path.open('a') as stream: stream.write(json.dumps(sys.argv[1:])+'\\n')
print(99001+len(rows))
''')
    executable(data['binary']/'squeue','#!'+sys.executable+'''\nimport os,sys
job=sys.argv[sys.argv.index('-j')+1]
if job in os.environ.get('FAKE_QUEUED_JOBS','').split(','): print(job+'_0')
''')
    log=data['binary']/'sbatch.jsonl'
    monkeypatch.setenv('FAKE_SBATCH_LOG',str(log))
    monkeypatch.setenv('FAKE_QUEUED_JOBS',','.join(queued))
    monkeypatch.setenv('PATH',str(data['binary'])+':'+os.environ['PATH'])
    return log


@pytest.mark.parametrize('separate',[True,False])
def test_all_views_across_original_submission_styles(launch,monkeypatch,capsys,separate):
    three_views(launch,separate=separate)
    before={str(p):p.read_bytes() for p in launch['output'].rglob('*') if p.is_file()}
    monkeypatch.setattr(sys,'argv',['recover.py','--all','--output-root',str(launch['output']),'--dry-run'])
    recover.main()
    plan=json.loads(capsys.readouterr().out)
    assert [r['dataset_view'] for r in plan['rows']]==list(recover.VIEWS)
    assert [r['original_job_id'] for r in plan['rows']]==['35455202','35455203','35455204']
    assert not (launch['output']/'launcher_fixes').exists()
    log=scheduler(launch,monkeypatch)
    monkeypatch.setattr(sys,'argv',['recover.py','--all','--output-root',str(launch['output'])])
    recover.main()
    commands=[json.loads(line) for line in log.read_text().splitlines()]
    assert len(commands)==3
    assert not any(a.startswith('--dependency=') for a in commands[0])
    assert '--dependency=afterany:99001' in commands[1]
    assert '--dependency=afterany:99002' in commands[2]
    batch=json.loads(next(launch['output'].glob('launcher_fixes/bash-batch-*/batch_fix.json')).read_text())
    assert [r['dataset_view'] for r in batch['submitted']]==list(recover.VIEWS)
    for file,content in before.items():assert Path(file).read_bytes()==content
    # Idempotent after a successful batch: receipts identify every resubmitted view.
    recover.main()
    assert len(log.read_text().splitlines())==3


def test_all_skips_already_resubmitted_knet_and_waits_for_it(launch,monkeypatch,capsys):
    three_views(launch,separate=True)
    log=scheduler(launch,monkeypatch,queued=['99001'])
    monkeypatch.setattr(sys,'argv',['recover.py','--job-id','35455202','--output-root',str(launch['output'])])
    recover.main()  # Previous release only re-submitted KNET, as reported by user.
    monkeypatch.setattr(sys,'argv',['recover.py','--all','--output-root',str(launch['output'])])
    recover.main()
    commands=[json.loads(line) for line in log.read_text().splitlines()]
    assert len(commands)==3
    assert '--job-name=fe02-lite-kiknet_borehole-formal-main_d128-s42-bashfix' in commands[1]
    assert '--job-name=fe02-lite-kiknet_surface-formal-main_d128-s42-bashfix' in commands[2]
    assert '--dependency=afterany:99001' in commands[1]
    assert '--dependency=afterany:99002' in commands[2]
    batch=json.loads(next(launch['output'].glob('launcher_fixes/bash-batch-*/batch_fix.json')).read_text())
    assert batch['rows'][0]['status']=='SKIPPED_ALREADY_RESUBMITTED'
    assert [r['dataset_view'] for r in batch['submitted']]==['kiknet_borehole','kiknet_surface']


def test_all_preflights_every_view_before_any_submission(launch,monkeypatch):
    log=scheduler(launch,monkeypatch)
    monkeypatch.setattr(sys,'argv',['recover.py','--all','--output-root',str(launch['output'])])
    with pytest.raises(ValueError,match='missing kiknet_borehole, kiknet_surface'):
        recover.main()
    assert not log.exists()
    three_views(launch,separate=True)
    source=launch['output']/'launches/fe02-lite-kiknet_surface/source/scientific.py'
    source.write_text('changed')
    with pytest.raises(ValueError,match='source file changed'):recover.main()
    assert not log.exists() and not (launch['output']/'launcher_fixes').exists()


def test_all_preserves_completed_run_and_concurrency(launch,monkeypatch):
    three_views(launch,separate=False,concurrency=2)
    launch['run'].mkdir();(launch['run']/'COMPLETED.json').write_text('{}')
    batch=recover.prepare_all(launch['output'])
    assert batch['rows'][0]['status']=='SKIPPED_COMPLETED'
    assert batch['concurrency']==2 and len(batch['plans'])==2
    log=scheduler(launch,monkeypatch)
    monkeypatch.setattr(sys,'argv',['recover.py','--all','--output-root',str(launch['output'])])
    recover.main()
    commands=[json.loads(line) for line in log.read_text().splitlines()]
    assert len(commands)==2 and all(not any(a.startswith('--dependency=') for a in c) for c in commands)
