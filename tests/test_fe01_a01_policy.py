import copy
import pytest
import torch

from fe01.model import build_model as legacy_build,state_fingerprint,shape_and_scale
from fe01_a01 import ON,OFF
from fe01_a01.model import build_model,apply_policy
from fe01_a01.availability import scale_audit
from fe01_a01.identity import reuse_equivalence,training_budget
from test_fe01_a01_helpers import config,sample


@pytest.mark.parametrize('family',['team_original_scratch','phasenet_pretrained_frozen','eqt_pretrained_frozen'])
def test_policy_initial_forward_small_update(tmp_path,family):
    cfg=config(tmp_path,family);event,result,inputs=sample(tmp_path,cfg)
    if family!='team_original_scratch' and not __import__('pathlib').Path(cfg['pretrained_manifest']).exists():
        pytest.skip('Actual registered offline weights unavailable')
    old=legacy_build(cfg);on=build_model(cfg)
    off_cfg=copy.deepcopy(cfg);off_cfg['absolute_amplitude_mode']=OFF;off=build_model(off_cfg)
    assert state_fingerprint(old.state_dict())==state_fingerprint(on.state_dict())==state_fingerprint(off.state_dict())
    assert reuse_equivalence(cfg,cfg,old.state_dict(),inputs)['status']=='PASS'
    for model,mode in ((on,ON),(off,OFF)):
        assert scale_audit(model,inputs,mode)['rows'][0]['status']=='PASS'
        if family!='team_original_scratch':
            model.train();assert not model.waveform_model.encoder.training
            assert all(not p.requires_grad for p in model.waveform_model.encoder.parameters())


def test_policy_retains_duration_and_eps_limit(tmp_path):
    cfg=config(tmp_path);_,_,inputs=sample(tmp_path,cfg)
    normalized,stats=shape_and_scale(inputs[0],inputs[5])
    off=apply_policy(stats,OFF)
    assert off.shape[-1]==11 and torch.equal(off[...,10:],stats[...,10:])
    assert not off[...,:10].any()
    cfg['absolute_amplitude_mode']=OFF;model=build_model(cfg)
    inputs[0]*=1e-12
    result=scale_audit(model,inputs,OFF)
    assert any(r['status']=='EPS_CLAMP_LIMITATION' for r in result['rows'])


@pytest.mark.parametrize('world',[1,2,4,8,16])
def test_budget_computed_from_pytorch_sampler(tmp_path,world):
    budget=training_budget(9084,config(tmp_path),world)
    assert budget['samples_per_epoch']==27252
    assert budget['updates_per_epoch']==212 and budget['total_updates']==2544
    assert budget['dropped_samples_per_epoch']==116
