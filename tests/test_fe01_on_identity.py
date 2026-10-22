import copy
import json
from pathlib import Path
import pytest
from fe01_a01_compare.sources import metadata_gate
from fe01_a01_compare.contracts import contract,verify_frozen_core,EPOCHS


def identity():
    cfg={'model_family':'diting_pretrained_frozen','seed':42,'window':{'protocol':'native_prefix_v2'},
         'absolute_amplitude_mode':'physical_scale_keep_duration'}
    lock={'config_sha256':'c','code_sha256':'3e537ff6693971d92b139e517f370b998237bd8b29989194e9ddcc06cc9dcfea',
          'a01_code_sha256':'3e1e4398b01d724f793f88b19be2a2c03ca67a71e881d876c173a8fbc5b8177f','pretrained_manifest_sha256':'a'*64}
    payload={'epoch':12,'config':copy.deepcopy(cfg),'source_config_sha256':'c','a01_code_sha256':lock['a01_code_sha256'],
             'absolute_amplitude_mode':cfg['absolute_amplitude_mode']}
    return payload,{'family':'DiTing','epoch':12},cfg,lock


@pytest.mark.parametrize('mutation',[lambda p,s,c,l:p.update(epoch=11),
    lambda p,s,c,l:p.update(source_config_sha256='wrong'),lambda p,s,c,l:p.update(a01_code_sha256='wrong'),
    lambda p,s,c,l:l.update(pretrained_manifest_sha256=None),lambda p,s,c,l:c.update(model_family='eqt_pretrained_frozen'),
    lambda p,s,c,l:c.update(seed=43),lambda p,s,c,l:p.update(absolute_amplitude_mode='no_absolute_scale_keep_duration'),
    lambda p,s,c,l:l.update(code_sha256='wrong')])
def test_wrong_identity_rejected(mutation):
    args=identity();mutation(*args)
    with pytest.raises(ValueError):metadata_gate(*args)


def test_valid_fixed_contract_and_frozen_core():
    metadata_gate(*identity())
    c=contract()
    assert {x['family']:x['selected_epoch'] for x in c['models']}==EPOCHS
    assert verify_frozen_core()['checked']==96


def test_team_scratch_metadata_does_not_require_pretrained_encoder():
    p,s,c,l=identity();s['family']='TEAM';s['epoch']=11;p['epoch']=11
    c['model_family']='team_original_scratch';p['config']=copy.deepcopy(c)
    l['pretrained_manifest_sha256']=None
    metadata_gate(p,s,c,l)


def test_allocated_forward_tests_are_explicitly_not_local_cpu_results():
    root=Path(__file__).resolve().parents[1]
    source=(root/'fe01_a01_compare/replay.py').read_text()
    for expected in ['future_', 'last_legal_positive_control', 'legacy_overlap', 'query_chunk_size']:
        assert expected in source
    # This is an implementation-presence check, not a production forward PASS.
