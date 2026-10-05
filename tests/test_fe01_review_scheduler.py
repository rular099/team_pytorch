import os
import json
import subprocess
import sys
from pathlib import Path
import pytest
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
