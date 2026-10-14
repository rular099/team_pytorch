"""Real small TEAM update and immutable live-session repair; no cluster jobs."""
import copy
import importlib.util
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

from scripts.fe01_a01_ops import eval_pack as ep,eval_revision as revision
from test_fe01_a01_eval_pack_ops import fixture,seal_result,small_checkpoints
from test_fe01_a01_helpers import config as small_config
from test_fe01_a01_scheduler import ROOT
from fe01.config import fingerprint,sha256,write_json
from fe01.model import state_fingerprint,common_state

BASE='e1ad3bf547392738ee5e120b7d4d33f8e42d7d30'


def historical_tools(directory):
    for name,digest in revision.LEGACY_OPERATIONS.items():
        try:
            data=subprocess.check_output(['git','-C',str(ROOT),'show',BASE+':scripts/fe01_a01_ops/'+name])
        except subprocess.CalledProcessError:
            pytest.skip('Historical eval source requires parent commit in the local Git history')
        (directory/name).write_bytes(data)
        assert sha256(directory/name)==digest
    spec=importlib.util.spec_from_file_location('actual_original_eval_pack',directory/'eval_pack.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    return old


def legacy_session(tmp,trained_team=False):
    env,log,project,tool,batches,cfgs,locks=fixture(tmp)
    if trained_team:small_checkpoints(batches[0]/cfgs[0,0]['run_id'],cfgs[0,0],locks[0,0])
    old=historical_tools(tool)
    with patch.dict(os.environ,env):old.prepare(project,tool,env['A01_ENV_FILE'])
    session=Path((tool/'eval_session.path').read_text().strip())
    fixed=tmp/'team_fix';fixed.mkdir()
    for name in revision.FILES+('submit_team_eval_fix.sh',):shutil.copy2(ROOT/'scripts/fe01_a01_ops'/name,fixed/name)
    env.update(A01_EVAL_SESSION=str(session),A01_CONFIG_PYTHON=sys.executable)
    return env,log,session,fixed,old


def pins(session):
    targets=[session/'plan.json',session/'runtime.private.env',*list((session/'tools').iterdir())]
    targets.extend(p for p in (session/'results').rglob('*') if p.is_file())
    return {p:p.read_bytes() for p in targets}


def small_team_checkpoint(tmp,monkeypatch):
    from fe01_a01.model import build_model
    from fe01_a01.provenance import source_identity
    cfg=small_config(tmp,mode='no_absolute_scale_keep_duration')
    model=build_model(cfg);encoder=model.waveform_model.encoder
    assert not model.waveform_model.frozen and all(p.requires_grad for p in encoder.parameters())
    initial_encoder=state_fingerprint(encoder.state_dict());initial_common=state_fingerprint(common_state(model))
    lock=dict(effective_config={},world_size=16,a01_code_sha256=source_identity()['a01_code_sha256'],
              initial_state=dict(encoder_sha256=initial_encoder,common_sha256=initial_common),
              sampling=dict(budget=dict(updates_per_epoch=212)),pretrained_manifest_sha256=None)
    optimizer=torch.optim.Adam(encoder.parameters(),lr=.001)
    output=encoder(torch.randn(2,3,cfg['native_n_samples']))
    objective=output.square().mean()+.001*encoder.conv1.weight.square().mean()
    objective.backward();optimizer.step()
    trained=state_fingerprint(encoder.state_dict());assert trained!=initial_encoder
    payload=dict(epoch=12,updates=2544,config=cfg,source_config_sha256=fingerprint(cfg),world_size=16,
                 a01_code_sha256=lock['a01_code_sha256'],a01_source_commit='SYNTHETIC_SOURCE',
                 identity=fingerprint(dict(config=cfg,audit_lock=lock,world_size=16)),absolute_amplitude_mode=cfg['absolute_amplitude_mode'],
                 encoder_state_sha256=trained,common_initial_state_sha256=initial_common,
                 curve_records=[dict(epoch=i) for i in range(1,13)],best_epoch=12,best_selection_score=1.,
                 model_state_dict=model.state_dict())
    path=tmp/'trained_team.pth';torch.save(payload,path)
    monkeypatch.setenv('A01_CODE_ROOT',str(ROOT))
    return cfg,lock,payload,path


def test_real_TEAM_update_fails_old_frozen_gate_and_passes_trained_state_gate_and_original_loader(tmp_path,monkeypatch):
    old_dir=tmp_path/'old';old_dir.mkdir();old=historical_tools(old_dir)
    cfg,lock,payload,path=small_team_checkpoint(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='initialization/encoder mismatch'):
        old.checkpoint_metadata(path,{'config':cfg},lock)
    result=ep.checkpoint_metadata(path,{'config':cfg},lock)
    identity=result['encoder_identity']
    assert identity['actual_state_sha256']==identity['recorded_state_sha256']!=identity['initial_state_sha256']
    from fe01_a01 import engine
    from fe01_a01.identity import a01_checkpoint
    monkeypatch.setattr(engine,'check_audit',lambda *a:lock)
    loaded,_=a01_checkpoint(path,12,cfg,tmp_path,device='cpu')
    assert state_fingerprint(loaded.waveform_model.encoder.state_dict())==identity['actual_state_sha256']


@pytest.mark.parametrize('bad',['common_initialization','trained_bytes','recorded_encoder_sha','missing_encoder'])
def test_TEAM_still_rejects_corrupt_initialization_or_trained_encoder(tmp_path,monkeypatch,bad):
    cfg,lock,payload,path=small_team_checkpoint(tmp_path,monkeypatch)
    if bad=='common_initialization':payload['common_initial_state_sha256']='0'*64
    elif bad=='recorded_encoder_sha':payload['encoder_state_sha256']='0'*64
    elif bad=='missing_encoder':payload['model_state_dict']={}
    else:
        key=next(k for k in payload['model_state_dict'] if k.startswith('waveform_model.encoder.'))
        payload['model_state_dict'][key]=payload['model_state_dict'][key]+1
    torch.save(payload,path)
    with pytest.raises(ValueError):ep.checkpoint_metadata(path,{'config':cfg},lock)


@pytest.mark.parametrize('index',[1,2,3])
def test_frozen_families_still_reject_changed_encoder_SHA(tmp_path,index):
    env,_,_,_,batches,cfgs,locks=fixture(tmp_path)
    batch=0 if index==2 else 1;cfg=cfgs[batch,index];lock=locks[batch,index]
    effective=copy.deepcopy(cfg);effective.update(lock['effective_config'])
    cp=dict(epoch=12,updates=2544,config=effective,source_config_sha256=fingerprint(cfg),world_size=16,
            a01_code_sha256=lock['a01_code_sha256'],identity=fingerprint(dict(config=effective,audit_lock=lock,world_size=16)),
            absolute_amplitude_mode=cfg['absolute_amplitude_mode'],common_initial_state_sha256=lock['initial_state']['common_sha256'],
            encoder_state_sha256='CHANGED')
    path=tmp_path/'wrong_frozen.pth';torch.save(cp,path)
    with patch.dict(os.environ,env),pytest.raises(ValueError,match='Frozen encoder mismatch'):
        ep.checkpoint_metadata(path,{'config':cfg},lock)


def test_retry_submits_only0_without_touching_live_plan_tools_or_other_results(tmp_path):
    env,log,session,tool,old=legacy_session(tmp_path)
    for index in (1,2,3):seal_result(session,index)
    before=pins(session)
    deny=tmp_path/'deny';deny.mkdir();(deny/'torch.py').write_text('raise RuntimeError("LOGIN_IMPORTED_TORCH")\n')
    env['PYTHONPATH']=str(deny)
    result=subprocess.run(['bash',str(tool/'submit_team_eval_fix.sh')],env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    calls=log.read_text();assert calls.count('sbatch ')==1 and '--array=0%4' in calls
    assert '_job_eval' in calls and '--gres=dcu:1' in calls and 'eval_first' not in calls
    assert all(p.read_bytes()==data for p,data in before.items())
    with patch.dict(os.environ,env):old.result_gate(session,3)
    directory=next((session/'operations').glob('team_eval_fix_*'))
    with patch.dict(os.environ,env):
        revision.validate_revision(session,directory/'revision.json',index=0)
        with pytest.raises(ValueError,match='only index0'):revision.validate_revision(session,directory/'revision.json',index=1)
    with patch.dict(os.environ,{**env,'A01_EVAL_REVISION_DIRECTORY':str(directory)}),pytest.raises(ValueError,match='only eval_team or pack'):
        ep.preflight(session,'eval_first')


def revised_result(session,directory):
    root=seal_result(session,0);path=root/'EVAL_COMPLETE.json';gate=json.loads(path.read_text())
    gate.update(execution_operations_sha256=json.loads((directory/'revision.json').read_text())['operations_sha256'],
                operations_revision=dict(path=str((directory/'revision.json').relative_to(session)),sha256=sha256(directory/'revision.json')))
    write_json(path,gate)
    return root


def test_new_cpu_pack_accepts_mixed_sources_without_rewriting_any_eval_seal(tmp_path):
    env,log,session,tool,old=legacy_session(tmp_path)
    with patch.dict(os.environ,env):directory=revision.register(session,tool)
    for index in (1,2,3,4):seal_result(session,index)
    revised_result(session,directory);before=pins(session)
    with patch.dict(os.environ,env),pytest.raises(ValueError,match='provenance differs'):
        old.result_gate(session,0)
    deny=tmp_path/'deny';deny.mkdir();(deny/'torch.py').write_text('raise RuntimeError("PACK_IMPORTED_TORCH")\n')
    run_env={**env,'PYTHONPATH':str(deny),'A01_EVAL_REVISION_DIRECTORY':str(directory)}
    result=subprocess.run([sys.executable,str(directory/'eval_pack.py'),'pack','--session',str(session)],
                          env=run_env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    with tarfile.open(session/'review.tar.gz') as archive:
        names=archive.getnames()
        assert any(n.startswith('evaluation_execution/') and n.endswith('revision.json') for n in names)
        assert any(n.startswith('results/index0/') for n in names) and 'package_execution.json' in names
        assert not any(n.endswith(('.env','.pth')) for n in names)
    assert all(p.read_bytes()==data for p,data in before.items()) and not log.exists()


def test_revision_source_tampering_blocks_result_verification(tmp_path):
    env,log,session,tool,old=legacy_session(tmp_path)
    with patch.dict(os.environ,env):directory=revision.register(session,tool)
    revised_result(session,directory)
    helper=directory/'eval_pack.py';helper.chmod(0o600);helper.write_text(helper.read_text()+'\n# TAMPER\n')
    with patch.dict(os.environ,env),pytest.raises(ValueError,match='Revision source changed'):
        ep.result_gate(session,0)
    assert not log.exists()


def test_corrected_eval_publishes_actual_new_execution_identity_in_unchanged_parent_session(tmp_path,monkeypatch):
    from fe01_a01 import engine,evaluation
    env,log,session,tool,old=legacy_session(tmp_path,trained_team=True)
    with patch.dict(os.environ,env):directory=revision.register(session,tool)
    before=pins(session);plan=json.loads((session/'plan.json').read_text());row=plan['runs'][0]
    lock=json.loads((Path(row['audit_dir'])/'protocol.lock.json').read_text())
    monkeypatch.setattr(engine,'check_audit',lambda *a:lock)
    def files(target,names):
        target.mkdir(parents=True)
        for name in names:(target/name).write_text('{}' if name.endswith('.json') else 'SYNTHETIC')
    def export(cfg,audit,checkpoint,epoch,target,device):
        assert cfg==row['config'] and Path(checkpoint).parent==Path(row['run_dir'])
        files(target,('predictions.csv.gz','support.csv','population.json','scores.csv','fields.csv.gz'))
        return 'SYNTHETIC_OFF'
    def old_on(cfg,audit,label,target,device):
        files(target,('predictions.csv.gz','population.json','checkpoint_identity.json'))
        return 'SYNTHETIC_ON'
    def compare(left,right,target):
        assert (left,right)==('SYNTHETIC_ON','SYNTHETIC_OFF')
        files(target,('comparison.json','paired_event_bootstrap.csv','ON_scores.csv','OFF_scores.csv'))
        write_json(target/'comparison.json',{'status':'PASS'})
    monkeypatch.setattr(evaluation,'export',export);monkeypatch.setattr(evaluation,'export_old_ON',old_on)
    monkeypatch.setattr(evaluation,'compare',compare)
    with patch.dict(os.environ,{**env,'A01_EVAL_REVISION_DIRECTORY':str(directory)}):
        ep.evaluate_one(session,0,device='cpu');ep.result_gate(session,0)
    gate=json.loads((session/'results/index0/EVAL_COMPLETE.json').read_text())
    assert gate['plan_sha256']==sha256(session/'plan.json')
    assert gate['execution_operations_sha256']!=plan['operations_sha256']
    assert gate['operations_revision']['sha256']==sha256(directory/'revision.json')
    assert all(p.read_bytes()==data for p,data in before.items()) and not log.exists()
