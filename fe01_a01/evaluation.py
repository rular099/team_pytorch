"""Strict locked validation populations and paired event-cluster statistics."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fe01.engine import evaluate
from fe01.metrics import summarize
from fe01_review.requests import population_audit, PAIR_KEYS, IDENTITY_KEYS, read_frame
from fe01_review.diagnostics import fields
from .provenance import require, new_output, write_json
from .identity import a01_checkpoint


def canonical_identity(frame,original,output=None):
    canonical=original[IDENTITY_KEYS].sort_values(PAIR_KEYS).reset_index(drop=True)
    order=frame[PAIR_KEYS].sort_values(PAIR_KEYS).index
    actual=frame.loc[order,IDENTITY_KEYS].reset_index(drop=True)
    if not actual[PAIR_KEYS].equals(canonical[PAIR_KEYS]):
        if output is not None:
            difference=actual.merge(canonical,on=PAIR_KEYS,how='outer',indicator=True,suffixes=('_actual','_locked'))
            difference.to_csv(Path(output)/'request_difference.csv.gz',index=False)
        raise ValueError('Missing/extra request keys; no intersection permitted')
    differences=[]
    for column in IDENTITY_KEYS:
        if column=='truth':
            equal=np.array_equal(actual[column].to_numpy(dtype=np.float32),canonical[column].to_numpy(dtype=np.float32))
        elif column=='requested_decision_sample':
            equal=np.allclose(actual[column],canonical[column],atol=1e-9,rtol=0)
        else:equal=actual[column].equals(canonical[column])
        if not equal:differences.append(column)
    if differences:
        if output is not None:
            pd.concat([actual.assign(source='actual'),canonical.assign(source='locked')]).to_csv(Path(output)/'request_difference.csv.gz',index=False)
        raise ValueError('Frozen physical label/request differs: '+str(differences))
    frame=frame.copy()
    for column in ('truth','requested_decision_sample'):
        frame.loc[order,column]=canonical[column].to_numpy()
    return frame


def evaluate_exact(model,cfg,device='cuda',output=None):
    """Capture raw logits per real query before any float32 softmax underflow."""
    from scipy.special import logsumexp
    from fe01.metrics import score_rows
    captured=[]
    def receive(module,args,outputs):
        raw=outputs[model.output_layout.index('pga')].detach().cpu().numpy()
        valid=args[4].detach().cpu().numpy().astype(bool)
        captured.append(raw[valid])
    handle=model.register_forward_hook(receive)
    try:
        frame,status=evaluate(model,cfg,device=device,times=(1,3,5,10,20))
    finally:handle.remove()
    # Authenticate against original CSV bytes, not a repeatedly serialized copy.
    # HDF labels are float32; CSV parsing can move their float64 display by1ULP.
    from fe01.config import sha256
    original_path=Path(cfg['a01']['original_run'])/f"validation_epoch{cfg['a01']['original_selected_epoch']}.csv.gz"
    require(sha256(original_path)==cfg['a01']['original_request_csv_sha256'],'Original request CSV bytes changed')
    original=read_frame(original_path)
    population_audit(original)
    frame=canonical_identity(frame,original,output)
    population_audit(frame)
    raw=np.concatenate(captured).astype(np.float64)
    require(len(raw)==len(frame),'MDN/query capture order differs')
    logits=raw[...,0];log_weights=logits-logsumexp(logits,axis=-1,keepdims=True)
    weights=np.exp(log_weights)
    mu=raw[...,1]*cfg['target_normalization']['std']+cfg['target_normalization']['mean']
    sigma=raw[...,2]*cfg['target_normalization']['std']
    require(np.isfinite(raw).all() and (sigma>0).all(),'Invalid MDN output')
    values=score_rows(frame.truth.to_numpy(),weights,mu,sigma)
    values['nll']=-logsumexp(log_weights-np.log(sigma*np.sqrt(2*np.pi))-
        (frame.truth.to_numpy()[:,None]-mu)**2/(2*sigma*sigma),axis=-1)
    for name,value in values.items():frame[name]=value
    for name,value in (('mdn_logits',logits),('mdn_weights',weights),('mdn_mu',mu),('mdn_sigma',sigma)):
        frame[name]=[json.dumps(row.tolist()) for row in value]
    return frame,status


def groups(frame):
    return {'all':frame,'noninput':frame[frame.target_role.ne('observed_input')],
        'triggered_noninput':frame[frame.target_role.eq('triggered_noninput')],
        'untriggered_noninput':frame[frame.target_role.eq('untriggered_noninput')],
        'K1_noninput':frame[frame.input_count.eq(1)&frame.target_role.ne('observed_input')],
        'Kmulti_noninput':frame[frame.input_count.ge(2)&frame.target_role.ne('observed_input')]}


def strict_pair(left,right):
    population_audit(left); population_audit(right)
    a=left.sort_values(PAIR_KEYS).reset_index(drop=True)
    b=right.sort_values(PAIR_KEYS).reset_index(drop=True)
    require(a[IDENTITY_KEYS].equals(b[IDENTITY_KEYS]),'Pair key/label/request mismatch; no silent intersection')
    return a,b


def score_table(frame):
    rows=[]
    for role,subset in groups(frame).items():
        for protocol in ('normal','random'):
            for time in (1,3,5,10,20):
                cell=subset[subset.geometry_protocol.eq(protocol)&subset.elapsed_time.eq(time)]
                record=summarize(cell) if len(cell) else dict(status='N/A',events=0,valid_targets=0)
                rows.append(dict(target_group=role,geometry_protocol=protocol,elapsed_time=time,**record))
    return pd.DataFrame(rows)


def cluster_interval(left,right,metric,draws=2000,field=False):
    """OFF minus ON, with all decision cells resampled together by event."""
    keys=['dataset_id','event_id','geometry_protocol','elapsed_time']
    if field:
        a=left.set_index(keys)[metric];b=right.set_index(keys)[metric]
        require(a.index.equals(b.index),'Field key mismatch')
        good=a.notna()&b.notna();d=(b-a)[good].reset_index(name='delta')
        d['count']=1
    else:
        a=left.assign(value=metric_values(left,metric)).groupby(keys).agg(value=('value','sum'),count=('value','size'))
        b=right.assign(value=metric_values(right,metric)).groupby(keys).agg(value=('value','sum'),count=('value','size'))
        require(a.index.equals(b.index) and a['count'].equals(b['count']),'Target denominator mismatch')
        d=(b.value-a.value).reset_index(name='delta');d['count']=a['count'].to_numpy()
    cells=pd.MultiIndex.from_product([('normal','random'),(1,3,5,10,20)],names=['geometry_protocol','elapsed_time'])
    if not len(d):
        return dict(status='N/A',reason='empty subgroup',events=0,targets=0,fields=0,draws=draws,effective_replicates=0,empty_cell_fraction=1.)
    p=d.pivot(index=['dataset_id','event_id'],columns=['geometry_protocol','elapsed_time'],values='delta').reindex(columns=cells)
    c=d.pivot(index=['dataset_id','event_id'],columns=['geometry_protocol','elapsed_time'],values='count').reindex(columns=cells)
    num=p.fillna(0).to_numpy();den=c.fillna(0).to_numpy()
    supported=den.sum(0)>0
    # Sparse subgroups report available cells and missing cells, never a disguised ten-cell score.
    num,den=num[:,supported],den[:,supported]
    rng=np.random.default_rng(20261007);sampled=[]
    for start in range(0,draws,100):
        idx=rng.integers(0,len(p),size=(min(100,draws-start),len(p)))
        n,count=num[idx].sum(1),den[idx].sum(1)
        valid=(count>0).all(1)
        sampled.extend((n[valid]/count[valid]).mean(1).tolist())
    result=dict(status='PASS' if supported.all() else 'SPARSE_AVAILABLE_CELLS_ONLY',
        estimate=float((num.sum(0)/den.sum(0)).mean()),events=len(p),targets=int(den.sum()) if not field else None,
        fields=int(den.sum()) if field else len(d),supported_cells=int(supported.sum()),missing_cells=int((~supported).sum()),
        draws=draws,effective_replicates=len(sampled),empty_cell_fraction=(draws-len(sampled))/draws,
        rng_seed=20261007,cluster='dataset,event; all station/time/geometry rows travel together',
        weighting='targets within cell then cells equal' if not field else 'fields within cell then cells equal',single_seed=True)
    result.update(ci_low=float(np.quantile(sampled,.025)) if sampled else None,
                  ci_high=float(np.quantile(sampled,.975)) if sampled else None)
    return result


def metric_values(frame,name):
    if name=='abs_error': return (frame.prediction-frame.truth).abs()
    if name=='squared_error': return (frame.prediction-frame.truth)**2
    return frame[name]


def compare(left,right,output,draws=2000):
    left,right=strict_pair(left,right)
    output=new_output(output)
    score_table(left).to_csv(output/'ON_scores.csv',index=False)
    score_table(right).to_csv(output/'OFF_scores.csv',index=False)
    intervals=[];all_fields=[]
    for role,a in groups(left).items():
        b=groups(right)[role]
        for metric in ('abs_error','squared_error','nll','crps','brier','predictive_sigma','covered68','covered95','width95'):
            intervals.append(dict(target_group=role,metric=metric,**cluster_interval(a,b,metric,draws)))
        af=fields(a);bf=fields(b)
        if len(af):
            af=af[af.target_group.eq('all')].sort_values(['dataset_id','event_id','elapsed_time','geometry_protocol'])
            bf=bf[bf.target_group.eq('all')].sort_values(['dataset_id','event_id','elapsed_time','geometry_protocol'])
            af=af[af.status.eq('supported')];bf=bf[bf.status.eq('supported')]
            require(af.targets.to_list()==bf.targets.to_list(),'Spatial target denominator changed')
            for metric in ('level_mse','shape_mse','field_mse','range_error','equal_distance_delta_mae'):
                intervals.append(dict(target_group=role,metric=metric,**cluster_interval(af,bf,metric,draws,field=True)))
            all_fields.extend([af.assign(model_mode='ON',a01_target_group=role),bf.assign(model_mode='OFF',a01_target_group=role)])
    pd.DataFrame(intervals).to_csv(output/'paired_event_bootstrap.csv',index=False)
    if all_fields: pd.concat(all_fields).to_csv(output/'spatial_fields.csv.gz',index=False)
    write_json(output/'comparison.json',dict(status='PASS',population=population_audit(left),
        sign='OFF minus ON; error increases are positive',seed=42,limitations=['single seed','upstream causality unknown',
            'eligible fields require >=5 targets; sparse cells explicitly reported','dynamic range alone is not improvement']))


def export(cfg,audit_dir,checkpoint,epoch,output,device='cuda'):
    model,payload=a01_checkpoint(checkpoint,epoch,cfg,audit_dir,device)
    output=new_output(output)
    resolved=payload['config']
    frame,status=evaluate_exact(model,resolved,device,output)
    status.to_csv(output/'support.csv',index=False)
    require(status.status.eq('supported').all(),'Incomplete validation support')
    audit=population_audit(frame)
    frame['absolute_amplitude_mode']=resolved['absolute_amplitude_mode']
    frame['comparison_kind']='trained_ablation'
    frame.to_csv(output/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    score_table(frame).to_csv(output/'scores.csv',index=False)
    fields(frame).to_csv(output/'fields.csv.gz',index=False)
    write_json(output/'population.json',{**audit,'identity_float_canonicalization':'original CSV decimal display after exact float32 label and integer-clock checks; all keys mandatory'})
    return frame


def export_old_ON(cfg,audit_dir,label,output,device='cuda'):
    """Frozen old epoch, same authenticated state and A01 ON-equivalent forward."""
    import torch
    from . import ON
    from .engine import check_audit
    from .identity import old_checkpoint
    from .model import build_model
    from fe01.config import sha256
    lock=check_audit(cfg,audit_dir)
    resolved=copy.deepcopy(cfg);resolved.update(lock['effective_config']);resolved['absolute_amplitude_mode']=ON
    old=Path(cfg['a01']['original_run'])
    epoch=cfg['a01']['original_selected_epoch'] if label=='selected' else 12
    path=old/('best.pth' if label=='selected' else 'last.pth')
    checkpoint,identity=old_checkpoint(path,epoch,old,resolved)
    model=build_model(resolved,device);model.load_state_dict(checkpoint['model_state_dict'],strict=True);model.eval()
    del checkpoint
    output=new_output(output)
    frame,status=evaluate_exact(model,resolved,device,output)
    status.to_csv(output/'support.csv',index=False)
    require(status.status.eq('supported').all(),'Old ON incomplete evaluation')
    audit=population_audit(frame)
    frame['absolute_amplitude_mode']=ON;frame['comparison_kind']='reused_trained_ON_frozen_epoch'
    frame.to_csv(output/'predictions.csv.gz',index=False)
    write_json(output/'population.json',audit)
    write_json(output/'checkpoint_identity.json',identity)
    return frame
