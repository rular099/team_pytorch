"""Multi-batch routing, gate order and CPU-only packing; fake scheduler/data."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from unittest.mock import patch

import pytest
import torch

from scripts.fe01_a01_ops import eval_pack as ep
from test_fe01_a01_scheduler import ROOT,gates
from test_fe01_a01_dcu_recovery import settings
from fe01.config import load,write_json,sha256,fingerprint
from fe01_review.provenance import TRAIN_CODE


def fixture(tmp):
    env,log=settings(tmp);project=tmp/'project';project.mkdir();(project/'outputs').mkdir()
    tool=tmp/'upload';tool.mkdir()
    for name in ('eval_pack.sh','eval_pack.py'):shutil.copy2(ROOT/'scripts/fe01_a01_ops'/name,tool/name)
    env.update(A01_PROJECT_ROOT=str(project),A01_SUBMISSION_LOG_DIR=str(tool))
    matrix=json.loads((ROOT/'configs/fe01_a01/matrix.json').read_text())['new_training']
    batches=[project/'outputs/early',project/'outputs/late']
    selected={0:0,1:1,2:0,3:1,4:1};configs={};locks={}
    for batch_index,batch in enumerate(batches):
        local={**env,'A01_OUTPUT_ROOT':str(batch)};gates(local,range(5))
        with patch.dict(os.environ,local):
            for index,entry in enumerate(matrix):
                cfg=load(ROOT/entry['config']);audit=batch/'audits'/cfg['run_id']
                lock=json.loads((audit/'protocol.lock.json').read_text())
                lock.update(code_sha256=TRAIN_CODE,world_size=16,data_identity={'SYNTHETIC':'DATA'},
                            validation_population_sha256='SYNTHETIC_POPULATION',
                            effective_config=dict(target_normalization={'mean':-1.23,'std':.42},
                                                  audited_cohorts={'train':['SYNTHETIC'],'val':['SYNTHETIC']}))
                lock['sampling']['budget']['updates_per_epoch']=212
                write_json(audit/'protocol.lock.json',lock)
                for stage in ('diagnostics','pilot'):
                    p=batch/stage/cfg['run_id']/(stage+'.json');gate=json.loads(p.read_text())
                    gate['audit_lock_sha256']=sha256(audit/'protocol.lock.json');write_json(p,gate)
                configs[batch_index,index]=cfg;locks[batch_index,index]=lock
                if selected[index]!=batch_index:continue
                make_run(batch,cfg,lock)
    return env,log,project,tool,batches,configs,locks


def make_run(batch,cfg,lock,epochs=12):
    run=batch/cfg['run_id'];run.mkdir(exist_ok=True)
    effective=copy.deepcopy(cfg);effective.update(lock['effective_config'])
    write_json(run/'resolved_config.json',effective);write_json(run/'protocol.lock.json',lock)
    for name in ('best.pth','last.pth'):(run/name).write_bytes(b'SYNTHETIC_NOT_REAL_WEIGHTS')
    (run/'training_curves.csv').write_text('epoch,updates\n'+''.join(f'{i},{i*212}\n' for i in range(1,epochs+1)))
    return run


def prepare(env,project,tool):
    with patch.dict(os.environ,env):ep.prepare(project,tool,env['A01_ENV_FILE'])
    return Path((tool/'eval_session.path').read_text().strip())


def seal_result(session,index):
    plan=json.loads((session/'plan.json').read_text());row=plan['runs'][index];run_id=row['run_id']
    root=session/'results'/f'index{index}';root.mkdir(parents=True)
    for label in ('selected','epoch12'):
        kinds=[run_id]+([run_id+'_reused_ON'] if index<3 else [])
        for kind in kinds:
            for name in ('predictions.csv.gz','support.csv','population.json','scores.csv','fields.csv.gz','checkpoint_identity.json'):
                p=root/'evaluation'/kind/label/name;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text('{}' if p.suffix=='.json' else 'SYNTHETIC_EXPORT\n')
        if index!=3:
            for name in ('comparison.json','paired_event_bootstrap.csv','ON_scores.csv','OFF_scores.csv'):
                p=root/'comparisons'/label/name;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text('{"status":"PASS"}' if p.suffix=='.json' else 'SYNTHETIC_COMPARE\n')
    files={str(p.relative_to(root)):sha256(p) for p in root.rglob('*') if p.is_file()}
    write_json(root/'EVAL_COMPLETE.json',dict(status='PASS',index=index,split='validation',plan_sha256=sha256(session/'plan.json'),
               source_core_sha256=plan['source_core_sha256'],execution_operations_sha256=plan['operations_sha256'],files_sha256=files))
    return root


def test_discovery_joins_five_complete_runs_from_two_batches_preserving_inputs(tmp_path):
    env,log,project,tool,batches,cfgs,locks=fixture(tmp_path)
    failed=make_run(batches[0],cfgs[0,1],locks[0,1],epochs=4)
    pins={p:sha256(p) for b in batches for p in b.rglob('*') if p.is_file()}
    session=prepare(env,project,tool);plan=json.loads((session/'plan.json').read_text())
    assert [Path(r['run_dir']).parent.name for r in plan['runs']]==['early','late','early','late','late']
    assert all(p.exists() and sha256(p)==digest for p,digest in pins.items())
    assert any(r['status']=='REJECTED' and r['path']==str(failed) for r in json.loads((tool/'eval_candidates.json').read_text()))
    assert not log.exists()
    with patch.dict(os.environ,env):ep.preflight(session,'eval_first')
    assert all(r['config']['output_root']!=str(session) for r in plan['runs'])


def test_login_prepare_and_submit_never_import_torch_and_first_stage_is_one_array(tmp_path):
    env,log,project,tool,*_=fixture(tmp_path)
    deny=tmp_path/'deny';deny.mkdir();(deny/'torch.py').write_text('raise RuntimeError("LOGIN_IMPORTED_TORCH")\n')
    env.update(PYTHONPATH=str(deny),A01_CONFIG_PYTHON=sys.executable)
    for stage in ('prepare','eval_first'):
        result=subprocess.run(['bash',str(tool/'eval_pack.sh'),stage],env=env,text=True,capture_output=True)
        assert result.returncode==0,result.stderr
    calls=log.read_text();assert calls.count('sbatch ')==1 and '--array=0,1,2,3%4' in calls
    assert '--nodes=1' in calls and '--gres=dcu:1' in calls and '--time=24:00:00' in calls
    assert calls.rstrip().endswith('_job_eval')
    result=subprocess.run(['bash',str(tool/'eval_pack.sh'),'eval_diting_off'],env=env,text=True,capture_output=True)
    assert result.returncode!=0 and log.read_text()==calls


def test_duplicate_complete_candidates_block_and_explicit_map_disambiguates(tmp_path):
    env,log,project,tool,batches,cfgs,locks=fixture(tmp_path)
    make_run(batches[1],cfgs[1,0],locks[1,0])
    with patch.dict(os.environ,env),pytest.raises(ValueError,match='found 2 complete'):
        ep.prepare(project,tool,env['A01_ENV_FILE'])
    assert not (tool/'eval_session.path').exists() and not log.exists()
    mapping=tmp_path/'map.json';write_json(mapping,{'0':str(batches[0]/cfgs[0,0]['run_id'])})
    env['A01_EVAL_RUN_MAP']=str(mapping)
    assert (prepare(env,project,tool)/'plan.json').is_file()


def test_cross_batch_diting_pair_mismatch_blocks_before_submission(tmp_path):
    env,log,project,tool,batches,cfgs,locks=fixture(tmp_path)
    cfg=cfgs[1,4];lock=locks[1,4];lock['initial_state']['common_sha256']='WRONG'
    audit=batches[1]/'audits'/cfg['run_id'];write_json(audit/'protocol.lock.json',lock)
    write_json(batches[1]/cfg['run_id']/'protocol.lock.json',lock)
    for stage in ('diagnostics','pilot'):
        p=batches[1]/stage/cfg['run_id']/(stage+'.json');gate=json.loads(p.read_text())
        gate['audit_lock_sha256']=sha256(audit/'protocol.lock.json');write_json(p,gate)
    with patch.dict(os.environ,env),pytest.raises(ValueError,match='init differs'):
        ep.prepare(project,tool,env['A01_ENV_FILE'])
    assert not log.exists()


def test_completed_indices_are_skipped_and_pack_requires_every_sealed_result(tmp_path):
    env,log,project,tool,*_=fixture(tmp_path);session=prepare(env,project,tool)
    with patch.dict(os.environ,env):
        with pytest.raises(OSError):ep.preflight(session,'pack')
        seal_result(session,0);seal_result(session,3)
        assert ep.preflight(session,'eval_first')==[1,2]
        assert ep.preflight(session,'eval_diting_off')==[4]
        for index in (1,2,4):seal_result(session,index)
        ep.preflight(session,'pack')
        (session/'results/index3/evaluation'/json.loads((session/'plan.json').read_text())['runs'][3]['run_id']/'selected/scores.csv').write_text('TAMPER')
        with pytest.raises(ValueError,match='artifact changed'):ep.preflight(session,'pack')
    assert not log.exists()


def test_cpu_pack_excludes_weights_and_failed_attempts_and_preserves_source_roots(tmp_path):
    env,log,project,tool,batches,*_=fixture(tmp_path);session=prepare(env,project,tool)
    for index in range(5):seal_result(session,index)
    bad=session/'attempts/FAILED';bad.mkdir(parents=True);(bad/'bad.csv').write_text('NEVER_INCLUDE_FAILED_RESULT')
    pin={p:sha256(p) for b in batches for p in b.rglob('*') if p.is_file()}
    deny=tmp_path/'deny';deny.mkdir();(deny/'torch.py').write_text('raise RuntimeError("PACK_IMPORTED_TORCH")\n')
    env.update(PYTHONPATH=str(deny),A01_CONFIG_PYTHON=sys.executable,A01_EVAL_SESSION=str(session))
    result=subprocess.run([sys.executable,str(session/'tools/eval_pack.py'),'pack','--session',str(session)],env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    with tarfile.open(session/'review.tar.gz') as archive:
        names=archive.getnames()
        assert any(n.endswith('A01_REVIEW_MANIFEST.json') for n in names)
        assert any(n.startswith('results/index4/') for n in names)
        assert not any(n.endswith(('.pth','.env','.py','.sh')) or 'FAILED' in n for n in names)
    assert all(sha256(p)==digest for p,digest in pin.items())
    assert (session/'review.tar.gz.sha256').read_text().split()[0]==sha256(session/'review.tar.gz')
    assert not log.exists()


def test_pack_submission_is_cpu_only_and_does_not_chain(tmp_path):
    env,log,project,tool,*_=fixture(tmp_path);session=prepare(env,project,tool)
    for index in range(5):seal_result(session,index)
    result=subprocess.run(['bash',str(tool/'eval_pack.sh'),'pack'],env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    call=log.read_text();assert call.count('sbatch ')==1 and '--gres' not in call and '--array' not in call
    assert call.rstrip().endswith('_job_pack')


def test_spooled_eval_launches_shared_script_and_initializes_environment_only_inside_step(tmp_path):
    from test_fe01_a01_dcu_recovery import compute_fixture
    env,log=compute_fixture(tmp_path)
    session=tmp_path/'eval_session';(session/'tools').mkdir(parents=True)
    for name in ('eval_pack.sh','eval_pack.py'):shutil.copy2(ROOT/'scripts/fe01_a01_ops'/name,session/'tools'/name)
    shutil.copy2(env['A01_ENV_FILE'],session/'runtime.private.env')
    env['A01_EVAL_SESSION']=str(session)
    spool=tmp_path/'local_spool';spool.mkdir();spooled=spool/'slurm_script'
    shutil.copy2(session/'tools/eval_pack.sh',spooled)
    result=subprocess.run(['bash',str(spooled),'_job_eval'],env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    calls=log.read_text()
    assert str(spool) not in calls and str(session/'tools/eval_pack.sh') in calls
    assert 'ENV_CONTEXT rank=batch' not in calls and 'ENV_CONTEXT rank=0' in calls
    assert 'eval_pack.py evaluate' in calls and '--index 2' in calls


def small_checkpoints(directory,cfg,lock):
    effective=copy.deepcopy(cfg);effective.update(lock['effective_config'])
    curves=[dict(epoch=i,updates=i*212,validation_selection=abs(i-7)+.1) for i in range(1,13)]
    common=dict(config=effective,source_config_sha256=fingerprint(cfg),world_size=16,
                a01_code_sha256=lock['a01_code_sha256'],a01_source_commit='SYNTHETIC_SOURCE',
                identity=fingerprint(dict(config=effective,audit_lock=lock,world_size=16)),
                absolute_amplitude_mode=cfg['absolute_amplitude_mode'],
                encoder_state_sha256=lock['initial_state']['encoder_sha256'],
                common_initial_state_sha256=lock['initial_state']['common_sha256'],best_epoch=7,best_selection_score=.1,
                model_state_dict={'SYNTHETIC_ONLY':torch.tensor([1.])})
    for name,epoch in (('best.pth',7),('last.pth',12)):
        torch.save(dict(common,epoch=epoch,updates=epoch*212,curve_records=curves[:epoch]),directory/name)


def test_eval_keeps_original_config_routes_both_checkpoints_and_retries_failed_attempt_without_overwrite(tmp_path,monkeypatch):
    from fe01_a01 import engine,evaluation
    env,log,project,tool,batches,cfgs,locks=fixture(tmp_path)
    cfg=cfgs[0,0];lock=locks[0,0];run=batches[0]/cfg['run_id'];small_checkpoints(run,cfg,lock)
    session=prepare(env,project,tool)
    calls=[];fail=[True]
    monkeypatch.setattr(engine,'check_audit',lambda *a:lock)
    def export(config,audit,checkpoint,epoch,target,device):
        calls.append((config['output_root'],str(checkpoint),epoch,str(target)))
        if epoch==12 and fail[0]:raise RuntimeError('SYNTHETIC_SECOND_EXPORT_FAILURE')
        target.mkdir(parents=True)
        for name in ('predictions.csv.gz','support.csv','population.json','scores.csv','fields.csv.gz'):
            (target/name).write_text('{}' if name.endswith('.json') else 'SYNTHETIC')
        return 'SYNTHETIC_OFF_FRAME'
    def old_on(config,audit,label,target,device):
        target.mkdir(parents=True)
        for name in ('predictions.csv.gz','population.json','checkpoint_identity.json'):
            (target/name).write_text('{}' if name.endswith('.json') else 'SYNTHETIC')
        return 'SYNTHETIC_ON_FRAME'
    def compare(on,off,target):
        assert on=='SYNTHETIC_ON_FRAME' and off=='SYNTHETIC_OFF_FRAME'
        target.mkdir(parents=True)
        for name in ('comparison.json','paired_event_bootstrap.csv','ON_scores.csv','OFF_scores.csv'):
            (target/name).write_text('{"status":"PASS"}' if name.endswith('.json') else 'SYNTHETIC')
    monkeypatch.setattr(evaluation,'export',export);monkeypatch.setattr(evaluation,'export_old_ON',old_on)
    monkeypatch.setattr(evaluation,'compare',compare)
    with patch.dict(os.environ,env):
        with pytest.raises(RuntimeError,match='SECOND_EXPORT_FAILURE'):ep.evaluate_one(session,0)
        assert not (session/'results/index0').exists()
        old_attempt=next((session/'attempts').iterdir());pins={p:sha256(p) for p in old_attempt.rglob('*') if p.is_file()}
        fail[0]=False;ep.evaluate_one(session,0);ep.result_gate(session,0)
    assert all(sha256(p)==digest for p,digest in pins.items()) and old_attempt.exists()
    assert all(call[0]==str(batches[0]) for call in calls)
    assert [call[2] for call in calls]==[7,12,7,12]
    assert all(str(session/'attempts') in call[3] for call in calls)
    metadata=json.loads((session/'results/index0/evaluation'/cfg['run_id']/'selected/checkpoint_identity.json').read_text())
    assert metadata['epoch']==7 and metadata['file_sha256']==sha256(run/'best.pth')
    assert not log.exists()


@pytest.mark.parametrize('bad', ['world','updates','mode','identity'])
def test_checkpoint_metadata_rejects_wrong_training_identity(tmp_path,bad):
    env,log,project,tool,batches,cfgs,locks=fixture(tmp_path)
    with patch.dict(os.environ,env):
        row=ep.candidate(3,batches[1]/cfgs[1,3]['run_id'])
        cfg=row['config'];lock=locks[1,3];effective=copy.deepcopy(cfg);effective.update(lock['effective_config'])
        cp=dict(epoch=12,updates=2544,config=effective,source_config_sha256=fingerprint(cfg),
                world_size=16,a01_code_sha256=lock['a01_code_sha256'],a01_source_commit='SYNTHETIC_SOURCE',
                identity=fingerprint(dict(config=effective,audit_lock=lock,world_size=16)),
                absolute_amplitude_mode=cfg['absolute_amplitude_mode'],encoder_state_sha256=lock['initial_state']['encoder_sha256'],
                common_initial_state_sha256=lock['initial_state']['common_sha256'],
                curve_records=[dict(epoch=i) for i in range(1,13)],best_epoch=12,best_selection_score=1.)
        path=tmp_path/'synthetic.pth';torch.save(cp,path)
        assert ep.checkpoint_metadata(path,row,lock)['epoch']==12
        key,value={'world':('world_size',8),'updates':('updates',1),'mode':('absolute_amplitude_mode','wrong'),'identity':('identity','wrong')}[bad]
        cp[key]=value;torch.save(cp,path)
        with pytest.raises(ValueError):ep.checkpoint_metadata(path,row,lock)
