import numpy as np
import pandas as pd
import pytest
from fe01_a01.evaluation import cluster_interval,canonical_identity
from fe01_a01.provenance import validate
from test_fe01_a01_helpers import config


def test_bootstrap_clusters_and_sparse_cells_are_explicit():
    rows=[]
    for event in range(10):
        for time in (1,3,5,10,20):
            for protocol in ('normal','random'):
                rows.append(dict(dataset_id='2020',event_id=str(event),elapsed_time=time,
                    geometry_protocol=protocol,prediction=0.,truth=1.))
    left=pd.DataFrame(rows);right=left.copy();right['prediction']=-.2
    result=cluster_interval(left,right,'abs_error',draws=300)
    assert result['events']==10 and result['targets']==100
    assert result['effective_replicates']==300 and abs(result['estimate']-.2)<1e-8
    subset=left[left.elapsed_time.eq(1)]
    sparse=cluster_interval(subset,right[right.elapsed_time.eq(1)],'abs_error',draws=300)
    assert sparse['status']=='SPARSE_AVAILABLE_CELLS_ONLY' and sparse['missing_cells']==8


def test_untraced_energy_temporal_anchor_bypass_rejected(tmp_path):
    cfg=config(tmp_path)
    for flag in ('use_pga_temporal_residual','use_target_temporal_pooling','pga_use_event_context'):
        cfg['model_params'][flag]=True
        with pytest.raises(ValueError):validate(cfg)
        cfg['model_params'][flag]=False


def test_request_float_display_canonicalization_never_intersects_or_changes_labels(tmp_path):
    rows=[dict(dataset_id='2020',event_id='E1',elapsed_time=1.,station_id=s,geometry_protocol='normal',
        target_role='untriggered_noninput',input_ids='["S0"]',truth=float(np.float32(-1.2)),
        requested_decision_sample=600.,current_sample=600,cutout_exclusive=601,history_start_sample=0,
        history_end_sample=600,latest_received_sample=600,input_count=1) for s in ('S1','S2')]
    original=pd.DataFrame(rows);actual=original.copy()
    actual['truth']=np.nextafter(actual.truth,np.inf)
    assert canonical_identity(actual,original).truth.equals(original.truth)
    with pytest.raises(ValueError,match='keys'):canonical_identity(actual.iloc[:1],original,tmp_path)
    assert (tmp_path/'request_difference.csv.gz').exists()
    actual['truth']+=.001
    with pytest.raises(ValueError,match='label'):canonical_identity(actual,original)
    actual=original.copy();actual['cutout_exclusive']+=1
    with pytest.raises(ValueError,match='request'):canonical_identity(actual,original)


def test_exact_logit_NLL_survives_float32_softmax_underflow(tmp_path,monkeypatch):
    import torch
    import fe01_a01.evaluation as module
    from fe01.config import sha256
    truth=float(np.float32(-1.2))
    frame=pd.DataFrame([dict(dataset_id='2020',event_id='E1',elapsed_time=1.,station_id='S1',geometry_protocol='normal',
        target_role='untriggered_noninput',input_ids='["S0"]',truth=truth,requested_decision_sample=600.,
        current_sample=600,cutout_exclusive=601,history_start_sample=0,history_end_sample=600,
        latest_received_sample=600,input_count=1,status='supported')])
    path=tmp_path/'validation_epoch11.csv.gz';frame.to_csv(path,index=False)
    cfg=config(tmp_path);cfg['target_normalization']=dict(enabled=True,mean=0.,std=1.)
    cfg['a01']=dict(original_run=str(tmp_path),original_selected_epoch=11,original_request_csv_sha256=sha256(path))
    class SyntheticMDN(torch.nn.Module):
        output_layout=['pga']
        def forward(self,*inputs):
            return [torch.tensor([[[[0.,100.,1.],[-1000.,truth,1.]]]])]
    model=SyntheticMDN()
    def fake_evaluate(model,cfg,**kwargs):
        inputs=[None]*6;inputs[4]=torch.tensor([[True]])
        model(*inputs)
        return frame.copy(),pd.DataFrame([dict(status='supported')])
    # Test fixture isolates scoring; production population authentication remains
    # exercised by separate locked real-metadata diagnostics and strict tests.
    monkeypatch.setattr(module,'evaluate',fake_evaluate)
    monkeypatch.setattr(module,'population_audit',lambda f:None)
    result,_=module.evaluate_exact(model,cfg,'cpu')
    assert abs(result.nll.iloc[0]-(1000+.5*np.log(2*np.pi)))<1e-7
    assert 'mdn_logits' in result
