import copy
import numpy as np
import pandas as pd
import pytest
import torch

from fe01_a01.model import build_model
from fe01_a01 import OFF
from fe01_a01.diagnostics import query_trace,permutation_controls,select_probes,nested_sets
from fe01_a01.availability import future_audit
from fe01.windows import CAPABILITIES,build_window
from test_fe01_a01_helpers import config,sample


def test_single_key_attention_and_live_query_residual(tmp_path):
    cfg=config(tmp_path);_,_,inputs=sample(tmp_path,cfg)
    inputs[2][:,1:]=False;inputs[5][:,1:]=False;inputs[0][:,1:]=0;inputs[1][:,1:]=0
    model=build_model(cfg)
    result,stages,sens=query_trace(model,inputs,cfg['target_normalization'],tmp_path/'trace',evidence='SYNTHETIC')
    assert stages.set_index('stage').loc['pure_attention','cross_query_max_abs']<1e-6
    assert stages.set_index('stage').loc['query_position_encoding','query_rms']>0
    assert stages.set_index('stage').loc['residual_normalized','query_rms']>0
    assert sens.gradient_mean_max_abs.max()>0
    small=sens[sens.step_km.eq(.01)]
    assert small.mean_fd_gradient_max_error.max()<2e-3
    assert small.distribution_fd_gradient_max_error.max()<2e-3
    assert permutation_controls(model,inputs)['joint_max_abs']<2e-5
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize('case',['trigger','capacity','tail_missing','legal_zero','invalid_station'])
def test_future_cutoff_and_last_legal_positive_control(tmp_path,case):
    cfg=config(tmp_path,mode=OFF);event,_,_=sample(tmp_path,cfg)
    elapsed=0.4 if case=='trigger' else 3
    if case=='capacity':elapsed=CAPABILITIES[cfg['model_family']].max_elapsed_sample()/100
    if case=='tail_missing':event['storage'][1,...,650:]=False
    if case=='invalid_station':event['storage'][-1]=False
    if case=='legal_zero':event['waveform'][...,550:565]=0
    result=future_audit(build_model(cfg),cfg,event,elapsed,case=case)
    assert result['upstream_status']=='UPSTREAM_CAUSALITY_UNKNOWN'
    assert all(r['status'].startswith(('PASS','UNSUPPORTED')) for r in result['rows'])


def test_metadata_only_probes_and_nested_common_population(tmp_path):
    rows=[]
    for k in (1,3):
        for event in range(40):
            for q in range(6):
                rows.append(dict(dataset_id='2020',event_id=str(event+k*100),elapsed_time=1.,
                    geometry_protocol='normal',input_count=k,station_id=f'S{q}',truth=event))
    frame=pd.DataFrame(rows);chosen,strata=select_probes(frame)
    changed=frame.copy();changed['truth']=-1000
    assert chosen.equals(select_probes(changed)[0])
    assert chosen.groupby(['elapsed_time','geometry_protocol','k_group','target_group']).size().max()==32
    assert strata.missing.any()
    cfg=config(tmp_path);event,_,_=sample(tmp_path,cfg,5)
    _,mask,_=build_window(event['waveform'],event['storage'],event['reference_sample'],5,CAPABILITIES[cfg['model_family']])
    sets,queries,info=nested_sets(event,mask.all(1),cfg)
    if 8 in sets:
        assert not np.isin(queries,sets[8]).any()
        for k in sets:assert np.array_equal(sets[k],sets[8][:k])
