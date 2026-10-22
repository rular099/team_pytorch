import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from fe01_a01_compare import orchestration as ops
from fe01_a01_compare.io import attempt,read_json,write_json,verify_seal


def test_atomic_attempt_preserves_failure_and_success(tmp_path):
    identity={'task':0,'execution_sha256':'x'}
    with pytest.raises(ValueError):
        with attempt(tmp_path,'s',0,identity) as p:
            (p/'partial.txt').write_text('partial')
            raise ValueError('failure')
    assert len(list((tmp_path/'attempts/s').glob('*/FAILURE.json')))==1
    with attempt(tmp_path,'s',0,identity) as p:
        (p/'ok.txt').write_text('ok')
    verify_seal(tmp_path/'stages/s/0',identity)
    with attempt(tmp_path,'s',0,identity) as p:
        assert p is None
    (tmp_path/'stages/s/0/ok.txt').write_text('changed')
    with pytest.raises(ValueError,match='changed'):
        verify_seal(tmp_path/'stages/s/0',identity)


def fake_scheduler(tmp_path,monkeypatch):
    bin_dir=tmp_path/'bin';bin_dir.mkdir()
    logfile=tmp_path/'sbatch_calls'
    scripts={'sbatch':'printf "999;cluster\\n"\nprintf "%s\\n" "$*" >> "$FAKE_LOG"\n',
             'squeue':'printf "%s\\n" "${FAKE_QUEUE:-}"\n','sacct':'printf "%s\\n" "${FAKE_ACCT:-}"\n'}
    for name,text in scripts.items():
        p=bin_dir/name;p.write_text('#!/usr/bin/env bash\n'+text);p.chmod(0o755)
    monkeypatch.setenv('PATH',str(bin_dir)+os.pathsep+os.environ['PATH'])
    monkeypatch.setenv('FAKE_LOG',str(logfile))
    monkeypatch.setenv('ON_EVAL_GRES','dcu:1')
    return logfile


def plan_stub(tmp_path,monkeypatch):
    session=tmp_path/'run';session.mkdir();(session/'jobs').mkdir();(session/'slurm').mkdir()
    plan={'tasks':{'model_smoke':[{'index':0},{'index':1}]},'env_file':str(tmp_path/'private.env')}
    monkeypatch.setattr(ops,'verify_plan',lambda s:plan)
    monkeypatch.setattr(ops,'prerequisites',lambda s,k:None)
    monkeypatch.setattr(ops,'completed',lambda s,k,i:False)
    return session


def test_fake_scheduler_prepare_status_never_submit(tmp_path,monkeypatch):
    log=fake_scheduler(tmp_path,monkeypatch)
    session=plan_stub(tmp_path,monkeypatch)
    ops.status(session)
    assert not log.exists()
    # Plain import is Torch-free, without calling prepare/submit implicitly.
    code='import sys; import fe01_a01_compare.cli; assert "torch" not in sys.modules'
    subprocess.run([sys.executable,'-c',code],check=True)
    assert not log.exists()


def test_submit_no_running_or_success_duplicate_and_failed_resume(tmp_path,monkeypatch):
    log=fake_scheduler(tmp_path,monkeypatch)
    session=plan_stub(tmp_path,monkeypatch)
    ops.submit(session,'model_smoke')
    assert len(log.read_text().splitlines())==1
    monkeypatch.setenv('FAKE_QUEUE','999_0|RUNNING\n999_1|PENDING')
    ops.submit(session,'model_smoke')
    assert len(log.read_text().splitlines())==1
    monkeypatch.setattr(ops,'completed',lambda s,k,i:i==0)
    monkeypatch.setenv('FAKE_QUEUE','')
    monkeypatch.setenv('FAKE_ACCT','999_1|FAILED|1:0')
    with pytest.raises(ValueError,match='resume_failed'):
        ops.submit(session,'model_smoke')
    ops.submit(session,'model_smoke',resume=True)
    assert len(log.read_text().splitlines())==2
    assert '--array=1%' in log.read_text().splitlines()[-1]


def test_unknown_state_not_complete_or_resubmitted(tmp_path,monkeypatch):
    log=fake_scheduler(tmp_path,monkeypatch)
    session=plan_stub(tmp_path,monkeypatch)
    ops.submit(session,'model_smoke')
    with pytest.raises(ValueError,match='UNKNOWN'):
        ops.submit(session,'model_smoke',resume=True)
    assert len(log.read_text().splitlines())==1


def test_task_plan_counts_model_once_per_block():
    tasks=ops.make_tasks([[str(i),str(i)] for i in range(1310)],32)
    assert len(tasks['replay_stations'])==4*2*41
    assert len(tasks['replay_grid_normal'])==4*41
    assert len(tasks['replay_pilot'])==4
    assert set(tasks['replay_stations'][0])=={'index','family','geometry','events'}


def test_sbatch_scripts_use_private_release_not_spool_relative_root():
    root=Path(__file__).resolve().parents[1]
    for path in (root/'scripts/fe01_a01_compare/jobs').glob('*.sbatch'):
        text=path.read_text()
        assert 'release_root=${ON_EVAL_RELEASE_ROOT:?}' in text
        assert 'sbatch ' not in text.replace('# Submitted explicitly by submit_stage.sh. No child job submission.','')
        subprocess.run(['bash','-n',str(path)],check=True)


def test_prepare_with_fake_scheduler_is_readonly_toward_sources(tmp_path,monkeypatch):
    from fe01_a01_compare.sources import pins
    log=fake_scheduler(tmp_path,monkeypatch)
    old=tmp_path/'old';old.mkdir()
    write_json(old/'plan.json',{'source':'fake private plan; no checkpoint touched'})
    pin=next(p for p in pins() if p['family']=='TEAM' and p['checkpoint_policy']=='selected')
    path=old/pin['source'];path.parent.mkdir(parents=True)
    rows=[]
    for e in range(1310):
        for i in range(2):
            rows.append(dict(dataset_id='japan_2020.hdf5',event_id=f'{e:014d}',station_id=f'S{i}',latitude=35+i*.01,
                             longitude=139+i*.01,input_count=i+1,magnitude=3+e/1310))
    pd.DataFrame(rows).to_csv(path,index=False)
    monkeypatch.setenv('ON_EVAL_SOURCE_SESSION',str(old))
    monkeypatch.setenv('ON_EVAL_OUTPUT_ROOT',str(tmp_path/'new'))
    monkeypatch.setattr('fe01_a01_compare.sources.verify_export_seals',lambda p:[])
    private=tmp_path/'private.env';private.write_text('# fake only\n')
    ops.prepare('fake_run',private)
    plan=read_json(tmp_path/'new/fake_run/plan.json')
    assert plan['station_frame_upper_bound']==209600
    assert plan['grid_point_count_distribution']['max']>0
    assert plan['formal_hpc_status']=='NOT_SUBMITTED'
    assert not log.exists()
    assert read_json(old/'plan.json')=={'source':'fake private plan; no checkpoint touched'}


def test_shared_data_gate_hashes_copied_shards_once(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import time
    from fe01_a01_compare.stages import shared_data_gate,data_gate
    from fe01_a01_compare.io import sha256
    root=tmp_path/'data';(root/'2020').mkdir(parents=True)
    shard=root/'2020/japan_2020.hdf5';shard.write_bytes(b'copied shard bytes')
    split=tmp_path/'split.csv';split.write_text('immutable split')
    audit=tmp_path/'audit';audit.mkdir()
    write_json(audit/'protocol.lock.json',{'data_identity':{'split_manifest_sha256':sha256(split),'shards':{
        '/original/2020/japan_2020.hdf5':{'bytes':shard.stat().st_size,'mtime_ns':0,'sha256':sha256(shard)}}}})
    old=tmp_path/'source';old.mkdir();write_json(old/'plan.json',{'runs':[{'audit_dir':str(audit)}]})
    plan={'source_session':str(old)};cfg={'data':{'root':str(root),'split_manifest':str(split)}}
    calls=[]
    def count(*args):
        calls.append(1);time.sleep(.03);return data_gate(*args)
    monkeypatch.setattr('fe01_a01_compare.stages.data_gate',count)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda i:shared_data_gate(tmp_path/'run',plan,cfg),range(4)))
    assert len(calls)==1
    assert len({x['cache_seal_sha256'] for x in results})==1


def test_optional_scope_incomplete_is_unknown_but_complete_corruption_fails(tmp_path,monkeypatch):
    from fe01_a01_compare import stages
    from fe01_a01_compare.io import seal
    monkeypatch.setattr(stages,'verify_plan',lambda s:{'tasks':{'replay_grid_random':[{'index':0}]}})
    directory=tmp_path/'stages/replay_grid_random/0'
    directory.mkdir(parents=True)
    assert stages.store_index(tmp_path,'replay_grid_random',False)=={}
    (directory/'data.txt').write_text('valid')
    seal(directory)
    def directories(session,key):
        verify_seal(directory)
        return [directory]
    monkeypatch.setattr(stages,'all_stage_directories',directories)
    (directory/'data.txt').write_text('changed')
    with pytest.raises(ValueError,match='changed'):
        stages.store_index(tmp_path,'replay_grid_random',False)


def test_streamed_frame_tables_keep_schema_for_small_and_supported_events(tmp_path):
    from fe01_a01_compare.synthetic import fixture
    from fe01_a01_compare.map_readiness import evaluate
    from fe01_a01_compare.stages import stream_csv
    from fe01_a01_compare.io import csv_read
    f,_=fixture()
    small=f[f.station_id.isin(['S00','S01'])].copy();small['event_id']='small'
    a,_=evaluate(small);b,_=evaluate(f)
    path=tmp_path/'frames.csv.gz'
    stream_csv(a,path);stream_csv(b,path)
    result=csv_read(path)
    assert len(result)==len(a)+len(b)
    supported=result[result.event_id.eq('SYNTHETIC_EVENT')&result.panel.eq('fixed_all_targets')]
    assert supported.mae.notna().all()
    assert result[result.event_id.eq('small')].mae.isna().all()
