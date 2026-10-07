"""Prefix causality and positive-gain interventions; labels never enter forward."""
import copy

import numpy as np
import torch

from fe01.data import prepare_sample
from fe01.engine import inputs_to
from fe01.model import shape_and_scale
from .model import apply_policy
from . import ON, OFF
from .provenance import require, write_json
from .equivalence import compare_tensors, paired_execution, capture_rng, restore_rng


def snapshot(model, inputs):
    raw, mask = inputs[0], inputs[5].bool()
    norm, stats = shape_and_scale(raw,mask)
    clean = torch.where(mask.unsqueeze(-2),raw,torch.zeros_like(raw))
    count = mask.sum(-1,keepdim=True).clamp_min(1)
    mean = clean.sum(-1)/count
    centered = torch.where(mask.unsqueeze(-2),clean-mean.unsqueeze(-1),torch.zeros_like(clean))
    variance = centered.square().sum(-1)/count
    encodings, delivered = [], []
    handles = [model.waveform_model.register_forward_hook(lambda m,a,o:encodings.append(o.detach())),
               model.waveform_scale_proj.register_forward_pre_hook(lambda m,a:delivered.append(a[0].detach()))]
    model.eval()
    try:
        with torch.no_grad():
            mdn = model(*inputs)[model.output_layout.index('pga')]
    finally:
        for handle in handles:
            handle.remove()
    return dict(received_mask=mask,station_mask=inputs[2],mean=mean,variance=variance,
                normalized_peak=norm.abs().amax((-2,-1)),normalized=norm,raw_amplitude=stats[...,:10],
                duration=stats[...,10:],delivered_amplitude=delivered[0],
                encoder_adapter=torch.stack(encodings,1),mdn=mdn,
                probabilities=mdn[...,0].softmax(-1))


def errors(a,b):
    return {k:(float((a[k].to(torch.float64)-b[k].to(torch.float64)).abs().max())) for k in a}


def _future_comparison(base_inputs, other_inputs, base, other):
    # Complete model input equality is stronger evidence than a float output
    # threshold: a changed future must never reach even a masked tensor slot.
    inputs_equal = len(base_inputs) == len(other_inputs) and all(
        a.dtype == b.dtype and a.shape == b.shape and torch.equal(a,b)
        for a,b in zip(base_inputs,other_inputs))
    strict = ('received_mask','station_mask','mean','variance','normalized_peak',
              'normalized','raw_amplitude','duration','delivered_amplitude')
    stats_equal = all(torch.equal(base[key], other[key]) for key in strict)
    downstream = ('encoder_adapter','mdn','probabilities')
    comparison = compare_tensors({key:base[key] for key in downstream},
                                  {key:other[key] for key in downstream})
    return dict(passed=inputs_equal and stats_equal and comparison['passed'],
                complete_inputs_exact=inputs_equal, prefix_statistics_exact=stats_equal,
                downstream=comparison)


def future_audit(model,cfg,event,elapsed,device='cpu',case='real_prefix',report_path=None):
    with paired_execution(device) as devices:
        return _future_audit(model,cfg,event,elapsed,device,case,devices,report_path)


def _future_audit(model,cfg,event,elapsed,device,case,devices,report_path):
    original = prepare_sample(event,cfg,elapsed,'normal')
    require(original['inputs'] is not None,'Unsupported future audit case')
    cutoff = original['info']['cutout_exclusive']
    require(event['waveform'].shape[-1] > cutoff,'Future test requires stored future samples')
    base_inputs = inputs_to(original,device)
    paired_rng = capture_rng(devices)
    base = snapshot(model,base_inputs)
    rows=[]
    comparisons=[]
    def record(name, sample):
        other_inputs=inputs_to(sample,device)
        restore_rng(paired_rng)
        other=snapshot(model,other_inputs)
        comparison=_future_comparison(base_inputs,other_inputs,base,other)
        comparison.update(mutation=name,selected_ids_equal=sample['info']['input_ids']==original['info']['input_ids'])
        comparison['passed']=comparison['passed'] and comparison['selected_ids_equal']
        comparisons.append(comparison)
        if report_path is not None:
            write_json(report_path,dict(case=case,elapsed_time=elapsed,cutout_exclusive=cutoff,
                checks=comparisons,upstream_status='UPSTREAM_CAUSALITY_UNKNOWN'))
        require(comparison['passed'],'Prefix input equality / forward reproducibility failed: '+
                str(comparison)+('; details: '+str(report_path) if report_path else ''))
        return errors(base,other),comparison
    for name,value in (('pulse',1e6),('NaN',np.nan),('Inf',np.inf)):
        changed=copy.deepcopy(event)
        changed['waveform'][...,cutoff:]=value
        sample=prepare_sample(changed,cfg,elapsed,'normal')
        delta,comparison=record(name,sample)
        rows.append(dict(case=case,elapsed_time=elapsed,cutout_exclusive=cutoff,mutation=name,
                         status='PASS',selected_ids_equal=True,complete_inputs_exact=True,
                         prefix_statistics_exact=True,downstream_max_tolerance_ratio=max(
                             (r.get('max_tolerance_ratio',0.) for r in comparison['downstream']['largest_differences']),default=0.),**delta))
    changed=copy.deepcopy(event)
    legal=cutoff-1
    valid = event['storage']
    support=valid[...,legal] if valid.ndim==3 else np.broadcast_to(valid[...,legal,None],changed['waveform'][...,legal].shape)
    if support.any():
        changed['waveform'][...,legal]=np.where(support,changed['waveform'][...,legal]+100.,changed['waveform'][...,legal])
        sample=prepare_sample(changed,cfg,elapsed,'normal')
        restore_rng(paired_rng)
        delta=errors(base,snapshot(model,inputs_to(sample,device)))
        require(delta['raw_amplitude']>0 or delta['mean']>0,'Last legal sample positive control failed')
        rows.append(dict(case=case,elapsed_time=elapsed,cutout_exclusive=cutoff,mutation='last_legal_positive_control',
                         status='PASS',selected_ids_equal=True,**delta))
    else:
        rows.append(dict(case=case,elapsed_time=elapsed,cutout_exclusive=cutoff,
                         mutation='last_legal_positive_control',status='UNSUPPORTED_NO_LEGAL_SAMPLE'))
    # Labels may be arbitrarily different without changing any input or forward.
    changed=copy.deepcopy(event);changed['pga']=changed['pga']+5
    sample=prepare_sample(changed,cfg,elapsed,'normal',queries=original['info'].get('query_indices'))
    delta,comparison=record('label_value_only',sample)
    rows.append(dict(case=case,elapsed_time=elapsed,cutout_exclusive=cutoff,mutation='label_value_only',status='PASS',
                     complete_inputs_exact=True,prefix_statistics_exact=True,**delta))
    return dict(upstream_status='UPSTREAM_CAUSALITY_UNKNOWN',
                scope='HDF waveform -> exclusive prefix -> statistics -> encoder/adapter -> MDN; no raw prefilter certification',
                comparisons=comparisons,rows=rows)


def scale_audit(model,inputs,mode):
    require(model.absolute_amplitude_mode==mode,'Scale test mode differs from actual model')
    base=snapshot(model,inputs)
    rows=[]
    patterns=[('event_0.1',[.1]*inputs[0].shape[1]),('event_1',[1.]*inputs[0].shape[1]),
              ('event_10',[10.]*inputs[0].shape[1]),
              ('station_0.1_1_10',[([.1,1.,10.][i%3]) for i in range(inputs[0].shape[1])])]
    for name,gains in patterns:
        x=[v.clone() for v in inputs]
        c=x[0].new_tensor(gains).reshape(1,-1,1,1)
        x[0]=x[0]*c
        other=snapshot(model,x)
        delta=errors(base,other)
        valid=inputs[2].bool()
        expected=torch.log10(c.squeeze(-1).squeeze(-1)).unsqueeze(-1)
        shift_error=float((other['raw_amplitude']-base['raw_amplitude']-expected)[valid].abs().max())
        normal=bool((base['raw_amplitude'][valid]>-10).all())
        require(delta['duration']==0,'Duration changed under positive gain')
        if normal:
            require(delta['normalized']<=3e-6 and shift_error<=3e-5,'Normal-regime scale preprocessing failed')
            if mode==OFF:
                require(delta['delivered_amplitude']==0 and delta['encoder_adapter']<=3e-5 and delta['mdn']<=3e-5,
                        'OFF leaks absolute gain')
        rows.append(dict(gain_pattern=name,mode=mode,status='PASS' if normal else 'EPS_CLAMP_LIMITATION',
                         log_shift_max_error=shift_error,**delta))
    return dict(scope='inference_intervention; gains are mechanism controls, not scored new earthquakes',
                normal_regime='all std/RMS/peaks >1e-10; joint normalization peak >1e-8',rows=rows)
