import os
import json
import subprocess
import sys
from pathlib import Path
import pytest
from fe01_review.provenance import evaluation_identity, sha256
ROOT=Path(__file__).resolve().parents[1]


def test_preparation_never_calls_scheduler(tmp_path):
    binary=tmp_path/'bin';binary.mkdir();marker=tmp_path/'scheduler_called'
    for name in ('sbatch','srun'):
        p=binary/name;p.write_text('#!/bin/bash\ntouch "'+str(marker)+'"\nexit99\n');p.chmod(0o755)
    models=json.loads((ROOT/'configs/fe01_review/checkpoint_manifest.example.json').read_text())
    manifest=tmp_path/'models.json';manifest.write_text(json.dumps(models))
    plan=tmp_path/'plan';plan.mkdir()
    for f in ('fixed_requests.csv.gz','random_time_draws.csv'):(plan/f).write_text('fixture')
    envpath=tmp_path/'cluster.env';values=dict(FE01_CODE_ROOT=str(ROOT),FE01_TRAIN_OUTPUT_ROOT=str(tmp_path/'original'),
        FE01_REVIEW_OUTPUT_ROOT=str(tmp_path/'review'),REVIEW_RUN_ID='synthetic',FE01_REVIEW_PLAN=str(plan),FE01_REVIEW_MANIFEST=str(manifest),
        FE01_PARTITION='fake',RT55_CHECKPOINT='absent',RT55_CONFIG='absent',RT61_CHECKPOINT='absent',RT61_CONFIG='absent',LEGACY_ENCODER_CHECKPOINT='absent')
    envpath.write_text('\n'.join('export '+k+'='+json.dumps(v) for k,v in values.items()))
    env=dict(os.environ,PATH=str(binary)+os.pathsep+os.environ['PATH'],DRY_RUN='1')
    result=subprocess.run(['bash',str(ROOT/'scripts/fe01_review/print_submit_commands.sh'),str(envpath)],env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert 'sbatch' in result.stdout;assert not marker.exists()
    jobs=json.loads((tmp_path/'review/synthetic/jobs_manifest.json').read_text());assert jobs['scheduler_called'] is False
    env['DRY_RUN']='0';result=subprocess.run(['bash',str(ROOT/'scripts/fe01_review/print_submit_commands.sh'),str(envpath)],env=env,capture_output=True)
    assert result.returncode!=0 and not marker.exists()


@pytest.mark.parametrize('action',['train','resume','test'])
def test_cli_has_no_unauthorized_action(action):
    result=subprocess.run([sys.executable,str(ROOT/'scripts/fe01_review/run.py'),action],capture_output=True)
    assert result.returncode==2


@pytest.fixture
def direct_cluster(tmp_path):
    """Fake Slurm records submissions; it never executes a job script."""
    binary=tmp_path/'bin';binary.mkdir();calls=tmp_path/'calls.jsonl'
    sbatch=binary/'sbatch'
    sbatch.write_text('#!'+sys.executable+'\nimport json,os,sys\nfrom pathlib import Path\n'
        'p=Path(os.environ["FAKE_SLURM_CALLS"])\n'
        'n=len(p.read_text().splitlines()) if p.exists() else 0\n'
        'with p.open("a") as f: f.write(json.dumps(dict(args=sys.argv[1:],cwd=os.getcwd()))+"\\n")\n'
        'if os.environ.get("FAKE_SLURM_ERROR"): print("fixture rejection",file=sys.stderr);sys.exit(1)\n'
        'print(str(9001+n)+";fixture_cluster")\n')
    sbatch.chmod(0o755)
    srun=binary/'srun';srun.write_text('#!/bin/bash\nexit 99\n');srun.chmod(0o755)
    # Use the actual frozen model matrix, with synthetic paths and CPU gates.
    models=json.loads((ROOT/'configs/fe01_review/checkpoint_manifest.example.json').read_text())['models']
    for m in models:
        for key in ('checkpoint','config','encoder','run_dir','diting_config'):
            if key in m:m[key]=str(tmp_path/key/m['run_id'])
    manifest=tmp_path/'models.json';manifest.write_text(json.dumps(dict(models=models)))
    plan=tmp_path/'plan';plan.mkdir()
    for name in ('fixed_requests.csv.gz','random_time_draws.csv'):(plan/name).write_text('fixture')
    root=tmp_path/'review/run'
    values=dict(FE01_CODE_ROOT=str(ROOT),FE01_TRAIN_OUTPUT_ROOT=str(tmp_path/'original'),
        FE01_REVIEW_OUTPUT_ROOT=str(root.parent),REVIEW_RUN_ID='run',FE01_REVIEW_PLAN=str(plan),
        FE01_REVIEW_MANIFEST=str(manifest),FE01_PARTITION='fake',FE01_ARRAY_CONCURRENCY='8',
        FE01_SUBMIT_PYTHON=sys.executable)
    envpath=tmp_path/'cluster.env'
    import shlex
    envpath.write_text('\n'.join('export '+k+'='+shlex.quote(v) for k,v in values.items()))
    import_blockers=tmp_path/'blocked_imports';import_blockers.mkdir()
    for name in ('torch','h5py'):
        (import_blockers/(name+'.py')).write_text('raise RuntimeError("Submission must not import models or HDF5")\n')
    env=dict(os.environ,PATH=str(binary)+os.pathsep+os.environ['PATH'],FAKE_SLURM_CALLS=str(calls),
             PYTHONPATH=str(import_blockers))
    def submit(stage):
        return subprocess.run(['bash',str(ROOT/'scripts/fe01_review/submit.sh'),stage,str(envpath)],
                              env=env,text=True,capture_output=True,cwd=tmp_path)
    def finish_identity(passed=(0,3,6,9)):
        identity=root/'identity';identity.mkdir()
        metadata=dict(evaluation_module_sha=evaluation_identity()['evaluation_module_sha'])
        (identity/'provenance.json').write_text(json.dumps(metadata))
        (identity/'artifact_manifest.sha256').write_text('')
        frozen=json.loads(json.dumps(models))
        inventory=[]
        for i,m in enumerate(frozen):
            m.update(inventory_status='IDENTITY_PASS' if i in passed else 'BLOCKED',checkpoint_sha256=str(i))
            inventory.append(dict(run_id=m['run_id'],selected=dict(reason='fixture missing/wrong epoch')))
        (identity/'frozen_checkpoint_manifest.json').write_text(json.dumps(dict(models=frozen)))
        (identity/'checkpoint_inventory.json').write_text(json.dumps(inventory))
        requests=root/'requests';requests.mkdir()
        (requests/'provenance.json').write_text(json.dumps(dict(**metadata,hydrated=True)))
        (requests/'artifact_manifest.sha256').write_text('')
        return frozen
    def finish_verify(indices=(0,3,6,9),stale=()):
        for i in indices:
            m=models[i];out=root/'verification'/m['run_id'];out.mkdir(parents=True)
            gate=dict(run_id=m['run_id'],checkpoint_epoch=m['epoch'],checkpoint_sha256=str(i),status='PASS',
                evaluation_module_sha='old' if i in stale else evaluation_identity()['evaluation_module_sha'],
                request_manifest_sha256=sha256(plan/'fixed_requests.csv.gz'))
            (out/'verification.json').write_text(json.dumps(gate));(out/'artifact_manifest.sha256').write_text('')
    return dict(submit=submit,finish_identity=finish_identity,finish_verify=finish_verify,root=root,
                env=env,calls=calls,models=models)


def read_calls(cluster):
    path=cluster['calls']
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_direct_identity_submits_once_from_project_root(direct_cluster):
    c=direct_cluster;result=c['submit']('identity')
    assert result.returncode==0,result.stderr
    assert 'SUBMITTED identity JobID=9001' in result.stdout
    calls=read_calls(c);assert len(calls)==1 and calls[0]['cwd']==str(ROOT)
    assert calls[0]['args'][-1].endswith('/identity_job.sbatch')
    assert '--partition=fake' in calls[0]['args'] and '--chdir='+str(ROOT) in calls[0]['args']
    jobs=json.loads((c['root']/'jobs_manifest.json').read_text())
    assert jobs['scheduler_called'] is True and jobs['jobs'][0]['job_id']=='9001'
    assert jobs['jobs'][0]['execution_status']=='SUBMITTED'
    assert c['submit']('identity').returncode!=0
    assert len(read_calls(c))==1


def test_direct_verify_requires_completed_cpu_plan_and_filters_identity(direct_cluster):
    c=direct_cluster;assert c['submit']('identity').returncode==0
    assert c['submit']('verify').returncode!=0
    assert len(read_calls(c))==1
    c['finish_identity']()
    result=c['submit']('verify');assert result.returncode==0,result.stderr
    args=read_calls(c)[-1]['args']
    assert '--array=0,3,6,9%4' in args and '--dependency=afterany:9001' in args
    assert args[-1]=='verify' and 'SKIP' in result.stdout


def test_direct_evaluation_rejects_stale_gates_without_rerunning_fixed_fe01(direct_cluster):
    c=direct_cluster;assert c['submit']('identity').returncode==0;c['finish_identity']()
    assert c['submit']('verify').returncode==0;c['finish_verify'](stale=(3,))
    for stage,array in [('random','0,6,9%3'),('long','0,6,9%3'),('legacy_fixed','9%1'),('replay','0,6,9%3')]:
        result=c['submit'](stage);assert result.returncode==0,result.stderr
        args=read_calls(c)[-1]['args'];assert '--array='+array in args
        assert '--dependency=afterany:9002' in args
    assert read_calls(c)[-2]['args'][-2:]==['evaluate','fixed']


def test_direct_stage_does_not_overwrite_output_or_submit_empty_array(direct_cluster):
    c=direct_cluster;assert c['submit']('identity').returncode==0;c['finish_identity'](passed=(0,))
    assert c['submit']('random').returncode!=0
    c['finish_verify'](indices=(0,))
    target=c['root']/'evaluations'/c['models'][0]['run_id']/'random';target.mkdir(parents=True)
    result=c['submit']('random')
    assert result.returncode!=0 and 'Existing output' in result.stderr
    assert c['submit']('legacy_fixed').returncode!=0
    assert len(read_calls(c))==1


def test_direct_reference_and_pack_record_dependencies_and_seal_submission(direct_cluster):
    c=direct_cluster;assert c['submit']('identity').returncode==0;c['finish_identity']()
    assert c['submit']('reference').returncode==0
    assert c['submit']('analyze').returncode!=0
    ref=c['root']/'reference';ref.mkdir()
    (ref/'train_only_reference.json').write_text('{}');(ref/'artifact_manifest.sha256').write_text('')
    assert c['submit']('analyze').returncode==0
    assert '--dependency=afterany:9002' in read_calls(c)[-1]['args']
    assert c['submit']('pack').returncode==0
    assert '--dependency=afterany:9001:9002:9003' in read_calls(c)[-1]['args']
    assert c['submit']('verify').returncode!=0
    assert len(read_calls(c))==4


def test_direct_scheduler_rejection_is_not_reported_as_success_or_retried(direct_cluster):
    c=direct_cluster;c['env']['FAKE_SLURM_ERROR']='1'
    result=c['submit']('identity')
    assert result.returncode!=0 and 'SUBMITTED identity' not in result.stdout
    jobs=json.loads((c['root']/'jobs_manifest.json').read_text())['jobs']
    assert jobs[0]['execution_status']=='SUBMISSION_UNCONFIRMED' and 'job_id' not in jobs[0]
    assert c['submit']('identity').returncode!=0 and len(read_calls(c))==1
