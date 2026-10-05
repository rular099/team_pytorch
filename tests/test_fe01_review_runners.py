import copy
import json
import numpy as np
import pandas as pd
import pytest
import torch
from tests.fe01_helpers import config,synthetic_archive
from fe01.data import split_catalog,read_event,prepare_sample
from fe01.model import build_model,model_audit,state_fingerprint,common_state
from fe01.metrics import decode
from fe01_review.runners import SystemRunner,pack_inputs,predict_decision,buffer_pin
from fe01_review.provenance import training_identity,new_output,write_json,fingerprint
from fe01_review.checkpoint_inventory import inspect_checkpoint


def fixture(tmp_path):
    synthetic_archive(tmp_path);cfg=config(tmp_path);cfg['geometry']['random_station_counts']=[1]
    event=read_event(split_catalog(cfg,'val').iloc[0],cfg,1,full_for_audit=True)
    sample=prepare_sample(event,cfg,1,'random');info=sample['info']
    records=[]
    for s in event['ids'][np.isfinite(event['pga'])]:
        i=list(event['ids']).index(s)
        records.append(dict(**{k:info[k] for k in ('current_sample','cutout_exclusive','history_start_sample','history_end_sample')},
            dataset_id=event['dataset_id'],event_id=event['event_id'],elapsed_time=1.,geometry_protocol='random',input_mode='natural',
            station_id=s,truth=float(event['pga'][i]),latitude=float(event['coords'][i,0]),longitude=float(event['coords'][i,1]),
            input_ids=json.dumps(info['input_ids']),input_coords=json.dumps(info['input_coords']),target_role='untriggered_noninput',input_count=info['input_count']))
    return cfg,event,pd.DataFrame(records)


@pytest.mark.parametrize('kind',['fe01','rt55'])
def test_real_class_bridge_query_cache_future_and_buffers(tmp_path,kind):
    cfg,event,req=fixture(tmp_path)
    if kind=='fe01':model=build_model(cfg)
    else:
        import gemini_models as legacy
        class Adapter(torch.nn.Module):
            def forward(self,x,token_mask=None):return x.mean(-1)
        model=legacy.build_transformer_model(**cfg['model_params'],trace_length=10000,
            station_waveform_model=torch.nn.Sequential(torch.nn.Conv1d(3,20,1),Adapter()))
    runner=SystemRunner(model,cfg['target_normalization'],kind);before=buffer_pin(model)
    base,_=predict_decision(runner,cfg,event,req,10000,chunk=3)
    for chunk,other in [(1,req),(3,req.iloc[::-1])]:
        prediction,_=predict_decision(runner,cfg,event,other,10000,chunk=chunk)
        np.testing.assert_allclose(base.prediction,prediction.set_index('station_id').loc[base.station_id].prediction,rtol=1e-5,atol=1e-5)
    inp,_=pack_inputs(event,req.iloc[0],cfg,10000,kind,queries=req.station_id[:2].tolist())
    with torch.no_grad():
        direct=model(*inp);wrapped=runner.forward(inp)
        for a,b in zip(direct,wrapped):torch.testing.assert_close(a,b)
    extended,_=pack_inputs(event,req.iloc[0],cfg,10000,kind,queries=req.station_id[:3].tolist())
    a=runner.forward(inp);b=runner.forward(extended);idx=model.output_layout.index('pga')
    torch.testing.assert_close(a[idx][:,:2],b[idx][:,:2],atol=1e-5,rtol=1e-5)
    with runner.cutoff_cache('1'):
        with pytest.raises(ValueError):runner.forward(inp,'2')
    for value in (np.nan,1e10):
        altered=copy.deepcopy(event);altered['waveform'][...,event['cutout_exclusive']:]=value
        with runner.capture_features() as original_features:predict_decision(runner,cfg,event,req,10000,chunk=3)
        with runner.capture_features() as changed_features:changed,_=predict_decision(runner,cfg,altered,req,10000,chunk=3)
        assert base.model_input_sha256.tolist()==changed.model_input_sha256.tolist()
        for a,b in zip(original_features,changed_features):torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
        np.testing.assert_allclose(base.prediction,changed.prediction,atol=1e-5,rtol=1e-5)
    assert buffer_pin(model)==before
    assert all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters())


def test_legacy_prefix_demean_own_inverse_and_metadata_isolation(tmp_path):
    cfg,event,req=fixture(tmp_path);inp,_=pack_inputs(event,req.iloc[0],cfg,10000,'rt55',queries=req.station_id[:2].tolist())
    wave,_,_,_,_,mask=inp
    assert abs((wave*mask[:,:,None]).sum().item())<1e-5
    changed=copy.deepcopy(event);changed['magnitude']=99;changed['event_location'][:]=99;changed['pga'][:]=99
    other,_=pack_inputs(changed,req.iloc[0],cfg,10000,'rt55',queries=req.station_id[:2].tolist())
    assert all(torch.equal(a,b) for a,b in zip(inp,other))
    raw=np.array([[[0,0,1]]],dtype=float)
    _,mu,sigma=decode(raw,dict(enabled=True,mean=4.,std=2.))
    np.testing.assert_allclose(mu,4.);np.testing.assert_allclose(sigma,2.)


def test_wrong_epoch_cannot_be_relabelled_and_fresh_output(tmp_path):
    p=tmp_path/'ep32.pth';torch.save(dict(epoch=20,model_state_dict={'x':torch.zeros(1)}),p)
    state=inspect_checkpoint(p,32,'rt55');assert state['status']=='BLOCKED';assert state['internal_epoch']==20
    output=new_output(tmp_path/'out');(output/'x').write_text('x')
    with pytest.raises(ValueError):new_output(output)
    assert len(training_identity()['files_sha256'])==115


def test_checkpoint_authenticates_original_config_lock_world_and_init(tmp_path):
    cfg=config(tmp_path);model=build_model(cfg);audit=model_audit(model)
    lock=dict(code_sha256=training_identity()['code_sha256'],config_sha256='original-source-config-pin')
    write_json(tmp_path/'resolved_config.json',cfg);write_json(tmp_path/'protocol.lock.json',lock)
    write_json(tmp_path/'model_interface_audit.json',audit);write_json(tmp_path/'runtime.json',dict(git_commit='synthetic-test-only'))
    payload=dict(epoch=0,model_state_dict=model.state_dict(),config=cfg,source_config_sha256=lock['config_sha256'],
        identity=fingerprint(dict(config=cfg,audit_lock=lock,world_size=16)),common_initial_state_sha256=audit['common_initial_state_sha256'])
    path=tmp_path/'init.pth';torch.save(payload,path)
    state=inspect_checkpoint(path,0,'fe01',tmp_path/'resolved_config.json',tmp_path)
    assert state['status']=='IDENTITY_PASS',state
    assert state['authenticated_world_size']==16
    payload['common_initial_state_sha256']='wrong';torch.save(payload,path)
    assert inspect_checkpoint(path,0,'fe01',tmp_path/'resolved_config.json',tmp_path)['status']=='BLOCKED'
