import copy
from pathlib import Path
import pytest
import torch

from fe01.config import fingerprint,sha256
from fe01.model import common_state,state_fingerprint,model_audit
from fe01_review.provenance import TRAIN_CODE
from fe01_a01.model import build_model
from fe01_a01.identity import old_checkpoint,check_family,a01_checkpoint
from fe01_a01.provenance import write_json,source_identity
from test_fe01_a01_helpers import config


def original_fixture(tmp):
    cfg=config(tmp);model=build_model(cfg)
    lock=dict(code_sha256=TRAIN_CODE,config_sha256=fingerprint(cfg),pretrained_manifest_sha256=None)
    checkpoint=dict(epoch=0,config=cfg,model_state_dict=model.state_dict(),source_config_sha256=fingerprint(cfg),
        common_initial_state_sha256=state_fingerprint(common_state(model)),
        identity=fingerprint(dict(config=cfg,audit_lock=lock,world_size=1)))
    write_json(tmp/'resolved_config.json',cfg);write_json(tmp/'protocol.lock.json',lock)
    write_json(tmp/'model_interface_audit.json',model_audit(model));torch.save(checkpoint,tmp/'init.pth')
    return cfg,checkpoint,lock


def test_TEAM_null_and_wrong_epoch_encoder_config_training_code(tmp_path):
    cfg,checkpoint,lock=original_fixture(tmp_path)
    assert old_checkpoint(tmp_path/'init.pth',0,tmp_path,cfg)[1]['status']=='IDENTITY_PASS'
    with pytest.raises(ValueError,match='epoch'):old_checkpoint(tmp_path/'init.pth',12,tmp_path,cfg)
    wrong=copy.deepcopy(cfg);wrong['model_family']='eqt_pretrained_frozen'
    with pytest.raises(ValueError,match='encoder'):old_checkpoint(tmp_path/'init.pth',0,tmp_path,wrong)
    changed=copy.deepcopy(checkpoint);changed['config']=copy.deepcopy(cfg);changed['config']['seed']=43
    torch.save(changed,tmp_path/'init.pth')
    with pytest.raises(ValueError,match='config'):old_checkpoint(tmp_path/'init.pth',0,tmp_path,cfg)
    torch.save(checkpoint,tmp_path/'init.pth');lock['code_sha256']='0'*64
    write_json(tmp_path/'protocol.lock.json',lock)
    with pytest.raises(ValueError,match='code'):old_checkpoint(tmp_path/'init.pth',0,tmp_path,cfg)


def test_family_SHA_is_not_blanket_optional(tmp_path):
    assert check_family(config(tmp_path),{'pretrained_manifest_sha256':None})['policy'].startswith('scratch')
    with pytest.raises(ValueError):check_family(config(tmp_path),{'pretrained_manifest_sha256':'a'*64})
    cfg=config(tmp_path,'phasenet_pretrained_frozen')
    for bad in (None,'','garbage','0'*64):
        with pytest.raises(ValueError):check_family(cfg,{'pretrained_manifest_sha256':bad})
    if Path(cfg['pretrained_manifest']).exists():
        assert check_family(cfg,{'pretrained_manifest_sha256':sha256(cfg['pretrained_manifest'])})['encoder_assets_sha256']
    wrong=copy.deepcopy(cfg);wrong['model_family']='unknown'
    with pytest.raises(ValueError,match='Unknown'):check_family(wrong,{'pretrained_manifest_sha256':None})


def test_A01_loader_rejects_stale_config_source_and_frozen_encoder(tmp_path):
    cfg=config(tmp_path,'phasenet_pretrained_frozen')
    if not Path(cfg['pretrained_manifest']).exists():pytest.skip('Actual pinned PhaseNet assets absent')
    model=build_model(cfg);audit=tmp_path/'audit';audit.mkdir()
    split=tmp_path/'split_events.csv';split.write_text('synthetic split identity\n')
    population=audit/'validation_population.csv.gz';population.write_bytes(b'synthetic population identity')
    source=source_identity()
    lock=dict(status='AUDIT_PASS',config_sha256=fingerprint(cfg),effective_config={},
        a01_code_sha256=source['a01_code_sha256'],code_sha256=TRAIN_CODE,
        data_identity=dict(split_manifest_sha256=sha256(split),shards={}),
        validation_population_sha256=sha256(population),pretrained_manifest_sha256=sha256(cfg['pretrained_manifest']),
        initial_state=dict(encoder_sha256=state_fingerprint(model.waveform_model.encoder.state_dict())))
    write_json(audit/'protocol.lock.json',lock)
    cp=dict(epoch=12,source_config_sha256=fingerprint(cfg),config=cfg,world_size=1,
        identity=fingerprint(dict(config=cfg,audit_lock=lock,world_size=1)),
        absolute_amplitude_mode=cfg['absolute_amplitude_mode'],a01_code_sha256=source['a01_code_sha256'],
        encoder_state_sha256=lock['initial_state']['encoder_sha256'],model_state_dict=model.state_dict())
    path=tmp_path/'new.pth';torch.save(cp,path)
    assert a01_checkpoint(path,12,cfg,audit)[1]['epoch']==12
    for key,bad in (('epoch',11),('source_config_sha256','0'*64),('a01_code_sha256','0'*64),('encoder_state_sha256','0'*64)):
        changed=copy.deepcopy(cp);changed[key]=bad;torch.save(changed,path)
        with pytest.raises(ValueError):a01_checkpoint(path,12,cfg,audit)
    changed=copy.deepcopy(cp)
    name=next(k for k in changed['model_state_dict'] if k.startswith('waveform_model.encoder.') and changed['model_state_dict'][k].is_floating_point())
    changed['model_state_dict'][name]+=1
    changed['encoder_state_sha256']=state_fingerprint({k.removeprefix('waveform_model.encoder.'):v for k,v in changed['model_state_dict'].items() if k.startswith('waveform_model.encoder.')})
    torch.save(changed,path)
    with pytest.raises(ValueError,match='encoder'):a01_checkpoint(path,12,cfg,audit)
