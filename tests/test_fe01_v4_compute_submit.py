"""Fake scheduler/backend tests; no cluster job, model or production data access."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'scripts/fe01_v4_compute_submit'
sys.path.insert(0, str(TOOLS))
import support as compute_support
import control as compute_control
import runtime as compute_runtime
from fe01_a01_compare import orchestration as ops
from fe01_a01_compare.io import attempt, verify_seal


@pytest.mark.parametrize('entry,tail', [('submit.sh', []), ('status.sh', ['--status'])])
def test_login_shell_only_sbatch_no_python_conda_or_private_env(entry, tail, tmp_path):
    project = tmp_path / 'project with spaces'; project.mkdir()
    marker = tmp_path / 'PRIVATE_ENV_WAS_SOURCED'
    (project / 'on_eval_v4_inventory_fix.private.env').write_text('touch "' + str(marker) + '"\nexit 99\n')
    bin_dir = tmp_path / 'bin'; bin_dir.mkdir()
    argv = tmp_path / 'args'
    sbatch = bin_dir / 'sbatch'
    sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$RECORD_ARGS"\nprintf "98765\\n"\n')
    sbatch.chmod(0o755)
    for name in ('python', 'python3', 'conda', 'sha256sum'):
        p = bin_dir / name
        p.write_text('#!/bin/bash\ntouch "$BANNED_COMMAND"\nexit 98\n'); p.chmod(0o755)
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'],
               FE01_V4_PROJECT_ROOT=str(project), RECORD_ARGS=str(argv), BANNED_COMMAND=str(tmp_path/'BANNED'))
    result = subprocess.run(['bash', str(TOOLS / entry)], cwd=project, env=env, text=True, capture_output=True)
    assert result.returncode == 0 and not marker.exists() and not (tmp_path/'BANNED').exists()
    args = argv.read_text().splitlines()
    assert args[-len(tail):] == tail if tail else True
    assert str(project / 'on_eval_v4_inventory_fix.private.env') in args
    assert str(TOOLS / 'control.sbatch') in args and '--mem=8G' in args
    assert not any(a.startswith('--array=') or a.startswith('--gres=') for a in args)


@pytest.mark.parametrize('step', ['control_step.sh', 'pool_step.sh'])
def test_compute_shell_refuses_before_reading_private_env_without_allocation(step, tmp_path):
    env = {k:v for k,v in os.environ.items() if k not in ('SLURM_JOB_ID','SLURM_STEP_ID')}
    result = subprocess.run(['bash', str(TOOLS / step)], env=env, capture_output=True, text=True)
    assert result.returncode and 'Allocated compute node required' in result.stderr


def test_python_requires_both_job_and_step(monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False); monkeypatch.delenv('SLURM_STEP_ID', raising=False)
    with pytest.raises(ValueError, match='allocated srun'):
        compute_support.allocated()
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    with pytest.raises(ValueError, match='allocated srun'):
        compute_support.allocated()
    monkeypatch.setenv('SLURM_STEP_ID', '0')
    compute_support.allocated()


@pytest.mark.parametrize('count', [164, 328])
def test_small_pool_preserves_all_original_logical_task_indices(count):
    mapping = compute_control.pools(list(range(count)), 6)
    assert set(mapping) == {str(i) for i in range(6)}
    assert sorted(i for values in mapping.values() for i in values) == list(range(count))
    assert all(len(values) > 1 for values in mapping.values())


@pytest.mark.parametrize('states,expected', [
    ({0:'COMPLETE_SEAL_VERIFIED',1:'FAILED',2:'NOT_SUBMITTED'},[1,2]),
    ({0:'RUNNING',1:'FAILED'},[]), ({0:'PENDING'},[]), ({0:'COMPLETE_SEAL_VERIFIED'},[])])
def test_task_selection_preserves_success_and_inflight_pool_cap(states, expected):
    assert compute_control.choose_tasks(states) == expected


def test_unknown_does_not_allow_duplicate_submission():
    with pytest.raises(ValueError, match='UNKNOWN'):
        compute_control.choose_tasks({0:'UNKNOWN',1:'FAILED'})


def test_site_array_task_limit_caps_pool_without_changing_logical_tasks(tmp_path,monkeypatch):
    monkeypatch.setattr(compute_control,'subprocess',SimpleNamespace(run=lambda *a,**k:
        SimpleNamespace(returncode=0,stdout='MaxArraySize = 1001\nSchedulerParameters = max_array_tasks=4\n',stderr='')))
    assert compute_control.effective_workers(6,tmp_path)==4
    assert len([i for values in compute_control.pools(list(range(328)),4).values() for i in values])==328


def test_default_compute_request_submits_only_two_replays_not_existing_latency(tmp_path,monkeypatch):
    session=tmp_path/'run';session.mkdir()
    monkeypatch.setenv('SLURM_JOB_ID','123');monkeypatch.setenv('SLURM_STEP_ID','0')
    monkeypatch.setenv('FE01_V4_RUN_ID','run')
    monkeypatch.setattr(compute_control,'revision',lambda:{'commit':'test'})
    monkeypatch.setattr(compute_control,'validate_plan',lambda p:None)
    monkeypatch.setattr(ops,'session_path',lambda r:session)
    monkeypatch.setattr(ops,'verify_plan',lambda s:{'tasks':{}})
    monkeypatch.setattr(compute_control,'effective_workers',lambda n,o:n)
    calls=[]
    monkeypatch.setattr(compute_control,'submit_pool',lambda s,p,stage,n,out,identity:calls.append((stage,n)))
    monkeypatch.setattr(sys,'argv',['control.py'])
    compute_control.main()
    assert calls==[('replay_stations',4),('replay_grid_normal',4)]
    assert all(stage!='latency' for stage,n in calls)


def test_mapping_aware_scheduler_preserves_latency_and_noncontiguous_indices(tmp_path, monkeypatch):
    jobs=tmp_path/'jobs';jobs.mkdir()
    (jobs/'1.json').write_text(json.dumps(dict(stage='replay_stations',job_id='999',indices=[64,70,90],worker_task_map={'0':[64,70],'1':[90]})))
    (jobs/'2.json').write_text(json.dumps(dict(stage='latency',job_id='33441392',indices=[0,1,2,3])))
    def run(args,**kwargs):
        value=('999_0|RUNNING\n999_1|FAILED\n' if '999' in args else
               '\n'.join('33441392_'+str(i)+'|RUNNING' for i in range(4)))
        return SimpleNamespace(returncode=0,stdout=value,stderr='')
    monkeypatch.setattr(compute_control,'subprocess',SimpleNamespace(run=run))
    states=compute_control.scheduler_states(tmp_path)
    assert states[('replay_stations',64)]['status']=='RUNNING'
    assert states[('replay_stations',70)]['status']=='RUNNING'
    assert states[('replay_stations',90)]['status']=='FAILED'
    assert all(states[('latency',i)]['job_id']=='33441392' for i in range(4))
    assert ('replay_stations',0) not in states


@pytest.mark.parametrize('returncode,stdout,state', [(1,'','REJECTED'),(0,'unrecognized reply','UNKNOWN')])
def test_submission_failure_keeps_exact_stderr_and_never_retries(tmp_path,monkeypatch,capsys,returncode,stdout,state):
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=returncode,stdout=stdout,stderr='sbatch: exact site rejection\n')
    monkeypatch.setattr(compute_control,'subprocess',SimpleNamespace(run=run))
    with pytest.raises(RuntimeError):
        compute_control.submit_checked(['sbatch','--array=0-5%6'],tmp_path)
    evidence=json.loads((tmp_path/'result.json').read_text())
    assert evidence['status']==state and evidence['stderr']=='sbatch: exact site rejection\n'
    assert len(calls)==1
    if returncode:
        assert 'exact site rejection' in capsys.readouterr().err


def pool_fixture(tmp_path,monkeypatch,count=328):
    session=tmp_path/'run';(session/'jobs').mkdir(parents=True);(session/'slurm').mkdir()
    (session/'plan.json').write_text('{}')
    env=tmp_path/'private.env';env.write_text('# fake\n')
    plan={'env_file':str(env),'tasks':{'replay_stations':[dict(index=i,events=[['year.hdf5',str(i)]]) for i in range(count)]}}
    original=copy.deepcopy(plan)
    monkeypatch.setattr(ops,'prerequisites',lambda *args:None)
    monkeypatch.setenv('ON_EVAL_GRES','dcu:1')
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0,stdout='777;cluster\n',stderr='')
    monkeypatch.setattr(compute_control,'subprocess',SimpleNamespace(run=run))
    return session,plan,original,calls


def test_submit_328_logical_tasks_as_only_six_scheduler_slots(tmp_path,monkeypatch):
    session,plan,original,calls=pool_fixture(tmp_path,monkeypatch)
    out=session/'operational_submit/control_123/replay_stations';out.mkdir(parents=True)
    revision={'commit':'external','revision_sha256':'digest'}
    assert compute_control.submit_pool(session,plan,'replay_stations',6,out,revision)=='777'
    assert len(calls)==1 and '--array=0-5%6' in calls[0] and '--gres=dcu:1' in calls[0]
    record=json.loads(next((session/'jobs').glob('*.json')).read_text())
    assert record['indices']==list(range(328)) and len(record['worker_task_map'])==6
    assert record['stage']=='replay_stations' and plan==original
    assert json.loads((out/'pool_manifest.pin.json').read_text())['sha256']==compute_support.digest(out/'pool_manifest.json')


@pytest.mark.parametrize('result', [None, {'status':'UNKNOWN'}, {'status':'ACCEPTED','job_id':'999'}])
def test_unresolved_submission_is_fail_closed(tmp_path,result):
    p=tmp_path/'operational_submit/control_123/replay_stations';p.mkdir(parents=True)
    (tmp_path/'jobs').mkdir()
    (p/'pool_manifest.json').write_text(json.dumps({'stage':'replay_stations'}))
    (p/'intent.json').write_text('{}')
    if result is not None:
        (p/'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match='Unresolved|no job receipt'):
        compute_control.unresolved_requests(tmp_path,'replay_stations')


def test_original_execute_bytecode_and_global_bindings_unchanged():
    from fe01_a01_compare import stages
    before=dict(stages.execute.__globals__)
    revised=compute_support.bind_attempt(stages.execute,lambda *args:None)
    assert revised.__code__ is stages.execute.__code__
    assert all(stages.execute.__globals__[k] is v for k,v in before.items())
    assert all(revised.__globals__[k] is v for k,v in before.items() if k!='attempt')


def test_mapping_and_operational_identity_are_sealed_and_original_reader_compatible(tmp_path):
    evidence=dict(stage='replay_stations',logical_task=140,manifest_sha256='mapping',
                  revision=dict(commit='new',revision_sha256='runtime'),
                  scheduler=dict(SLURM_ARRAY_TASK_ID='2'))
    identity=dict(stage='replay_stations',task=140,execution_sha256='unchanged')
    callback=compute_support.evidence_attempt(attempt,evidence)
    with callback(tmp_path,'replay_stations',140,identity) as out:
        (out/'data.txt').write_text('synthetic')
    path=tmp_path/'stages/replay_stations/140'
    seal=verify_seal(path,identity)
    assert seal['runtime_revision_commit']=='new' and seal['operational_pool_manifest_sha256']=='mapping'
    assert json.loads((path/'operational_submission.json').read_text())['scheduler']['SLURM_ARRAY_TASK_ID']=='2'
    (path/'operational_submission.json').write_text('tampered')
    with pytest.raises(ValueError,match='changed'):
        verify_seal(path,identity)


def test_fresh_process_per_logical_task_preserves_partial_success_and_reports_failure(tmp_path,monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID','123');monkeypatch.setenv('SLURM_ARRAY_TASK_ID','2')
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1 if command[-1]=='140' else 0)
    monkeypatch.setattr(compute_runtime,'subprocess',SimpleNamespace(run=run))
    with pytest.raises(ValueError,match='Some original logical tasks failed'):
        compute_runtime.supervise(dict(session=str(tmp_path),stage='replay_stations'),manifest,tmp_path/'pin',[2,140,200])
    assert [c[-1] for c in calls]==['2','140','200']
    assert all(c[0]==sys.executable and '--task' in c for c in calls)
    summary=json.loads(next((tmp_path/'operational_submit').glob('*/worker_summary.json')).read_text())
    assert summary['failures']==[{'index':140,'returncode':1}]
    assert os.environ['SLURM_ARRAY_TASK_ID']=='2'


def test_actual_source_guard_rejects_wrong_main_commit(tmp_path,monkeypatch):
    monkeypatch.setenv('ON_EVAL_RELEASE_ROOT',str(ROOT))
    monkeypatch.setenv('FE01_V4_RUN_ID','run');monkeypatch.setenv('FE01_V4_ENV_FILE',str(tmp_path/'private.env'))
    plan={'run_id':'run','env_file':str(tmp_path/'private.env'),'source_identity':{'result_commit':'wrong','execution_sha256':compute_support.EXECUTION}}
    with pytest.raises(ValueError,match='Exact frozen'):
        compute_support.validate_plan(plan)


@pytest.mark.parametrize('corruption', [None, 'task', 'device'])
def test_runtime_checks_mapping_pin_original_task_and_device_without_rewriting_slot(tmp_path,monkeypatch,corruption):
    stage='replay_stations'
    plan={'tasks':{stage:[dict(index=i,events=[['synthetic',str(i)]]) for i in range(4)]}}
    identity={'commit':'external','revision_sha256':'sha'}
    m=dict(stage=stage,device='cuda',session=str(tmp_path),plan_sha256=compute_support.digest(tmp_path/'plan.json') if (tmp_path/'plan.json').exists() else '',
           revision=identity,worker_task_map={'0':[2,3]},logical_tasks_sha256={str(i):compute_support.fingerprint(plan['tasks'][stage][i]) for i in (2,3)})
    (tmp_path/'plan.json').write_text('{}');m['plan_sha256']=compute_support.digest(tmp_path/'plan.json')
    if corruption=='task':
        m['logical_tasks_sha256']['2']='wrong'
    if corruption=='device':
        m['device']='cpu'
    path=tmp_path/'manifest.json';path.write_text(json.dumps(m))
    pin=tmp_path/'pin.json';pin.write_text(json.dumps({'sha256':compute_support.digest(path)}))
    monkeypatch.setenv('SLURM_ARRAY_TASK_ID','0')
    monkeypatch.setattr(compute_runtime,'revision',lambda:identity)
    monkeypatch.setattr(compute_runtime,'validate_plan',lambda p:None)
    monkeypatch.setattr(ops,'verify_plan',lambda s:plan)
    if corruption:
        with pytest.raises(ValueError,match='definition changed|device differs'):
            compute_runtime.load_manifest(path,pin)
    else:
        assert compute_runtime.load_manifest(path,pin)[2]==[2,3]
        path.write_text(path.read_text()+' ')
        with pytest.raises(ValueError,match='byte identity differs'):
            compute_runtime.load_manifest(path,pin)
    assert os.environ['SLURM_ARRAY_TASK_ID']=='0'


def test_shell_syntax_and_torch_free_imports():
    for path in TOOLS.iterdir():
        if path.suffix in ('.sh','.sbatch'):
            subprocess.run(['bash','-n',str(path)],check=True)
    code='import sys;sys.path.insert(0,'+repr(str(TOOLS))+');import control,runtime,support;assert "torch" not in sys.modules'
    subprocess.run([sys.executable,'-B','-c',code],check=True)
