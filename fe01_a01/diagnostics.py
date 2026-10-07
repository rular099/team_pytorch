"""Query information flow through genuine differentiable eval paths."""
import contextlib
import copy
import json
import math

import numpy as np
import pandas as pd
import torch

from fe01.config import fingerprint
from .provenance import require, write_json


def select_probes(requests, cap=32):
    """Metadata-only stratification: time, geometry, actual K, target count."""
    require(0 < cap <= 32, 'Probe budget per stratum must be <=32')
    keys = ['dataset_id','event_id','elapsed_time','geometry_protocol']
    counts = requests.groupby(keys).agg(input_count=('input_count','first'),
                                      target_count=('station_id','nunique')).reset_index()
    require((requests.groupby(keys).input_count.nunique() == 1).all(), 'Inconsistent K in decision')
    counts['k_group'] = np.where(counts.input_count.eq(1), 'K1', 'Kmulti')
    counts['target_group'] = np.where(counts.target_count.ge(5), 'Q5plus', 'Q1to4')
    counts['selection_hash'] = [fingerprint(['FE01-A01-metadata-probe',*[r[k] for k in keys]])
                                for r in counts.to_dict('records')]
    selected, strata = [], []
    for time in (1,3,5,10,20):
        for protocol in ('normal','random'):
            for k in ('K1','Kmulti'):
                for q in ('Q1to4','Q5plus'):
                    group = counts[counts.elapsed_time.eq(time) & counts.geometry_protocol.eq(protocol)
                                   & counts.k_group.eq(k) & counts.target_group.eq(q)]
                    chosen = group.sort_values('selection_hash').head(cap)
                    selected.append(chosen)
                    strata.append(dict(elapsed_time=time,geometry_protocol=protocol,k_group=k,
                                       target_group=q,available=len(group),selected=len(chosen),
                                       missing=len(group)==0,cap=cap))
    return pd.concat(selected,ignore_index=True), pd.DataFrame(strata)


@contextlib.contextmanager
def trace_hooks(model):
    values, handles, pe = {}, [], []
    def hook(name, tuple_index=None):
        def save(module, args, output):
            values[name] = output if tuple_index is None else output[tuple_index]
        return save
    handles.append(model.position_embedding.register_forward_hook(lambda m,a,o: pe.append(o)))
    readout = model.pga_cross_attention
    handles.append(readout.register_forward_pre_hook(lambda m,a: values.update(query_before_attention=a[0])))
    handles.append(readout.attn.register_forward_hook(hook('pure_attention',0)))
    handles.append(readout.attn.register_forward_hook(hook('attention_weights',1)))
    handles.append(readout.out_norm.register_forward_pre_hook(lambda m,a: values.update(residual_sum=a[0])))
    handles.append(readout.out_norm.register_forward_hook(hook('residual_normalized')))
    for i, layer in enumerate(readout.extra_layers,1):
        handles.append(layer.attn.register_forward_hook(hook(f'layer{i}_pure_attention',0)))
        handles.append(layer.attn.register_forward_hook(hook(f'layer{i}_weights',1)))
        handles.append(layer.register_forward_hook(hook(f'layer{i}_readout',0)))
    head = []
    handles.append(model.mlp_pga.register_forward_hook(lambda m,a,o: head.append(o)))
    try:
        yield values, pe, head
    finally:
        for handle in handles:
            handle.remove()


def decoded_tensor(mdn, norm):
    weight = mdn[...,0].softmax(-1)
    mean = mdn[...,1]*norm['std']+norm['mean']
    sigma = mdn[...,2]*norm['std']
    prediction = (weight*mean).sum(-1)
    return weight,mean,sigma,prediction


def query_trace(model, inputs, norm, output=None, evidence='real_checkpoint', fd_km=(.01,.1,1.)):
    require(not getattr(model,'_fe01_query_cache',None), 'Diagnostics forbid detached query cache')
    model.eval()
    x = [v.detach().clone() for v in inputs]
    x[3].requires_grad_(True)
    require(x[0].shape[0] == 1 and x[4].any(), 'Trace requires one decision with real queries')
    valid = x[4][0].bool()
    with torch.enable_grad(), trace_hooks(model) as (values,pe,head):
        mdn = model(*x)[model.output_layout.index('pga')]
        weight,mu,sigma,prediction = decoded_tensor(mdn,norm)
        # Query heads are independent; sum yields the per-query diagonal Jacobian.
        mean_grad = torch.autograd.grad(prediction[0,valid].sum(),x[3],retain_graph=True)[0]
        jac = []
        for component in range(mdn.shape[-2]):
            for column in range(3):
                item = (weight,mu,sigma)[column][0,valid,component].sum()
                jac.append(torch.autograd.grad(item,x[3],retain_graph=True)[0])
        distribution_grad = torch.stack(jac)
        values.update(query_coordinates=x[3], mdn_weights=weight, mdn_mu=mu,
                      mdn_sigma=sigma,prediction=prediction.unsqueeze(-1))
        # Match the PE to the actual readout query, rather than assuming call order.
        values['query_position_encoding'] = pe[-1]
        if head:
            values['pga_head_hidden'] = torch.stack(head,dim=1)
        arrays = {k:v.detach().cpu().numpy() for k,v in values.items()}
        base_branches = {k:float(v.detach()) for k,v in getattr(model,'_last_diag',{}).items()
                if torch.is_tensor(v) and v.numel()==1 and any(s in k for s in ('norm','ratio','gate'))
                and torch.isfinite(v)}
    rows = []
    for name,array in arrays.items():
        if array.ndim >= 3 and array.shape[1] == len(valid):
            z = array[0,valid.cpu().numpy()].reshape(int(valid.sum()),-1)
            centered = z-z.mean(0)
            rows.append(dict(stage=name,query_rms=float(np.sqrt(np.mean(centered**2))),
                             cross_query_max_abs=float(np.max(np.ptp(z,axis=0))),
                             token_norm_mean=float(np.linalg.norm(z,axis=-1).mean()),
                             effective_queries=len(z),effective_keys=int(x[2].sum()),evidence=evidence))
    stage_table = pd.DataFrame(rows)
    order=['query_coordinates','query_position_encoding','query_before_attention','pure_attention',
           'attention_weights','residual_sum','residual_normalized']
    for i in range(1,len(model.pga_cross_attention.extra_layers)+1):
        order.extend([f'layer{i}_pure_attention',f'layer{i}_weights',f'layer{i}_readout'])
    order.extend(['pga_head_hidden','mdn_weights','mdn_mu','mdn_sigma','prediction'])
    stage_table=stage_table.set_index('stage').reindex([s for s in order if s in stage_table.stage.values]).reset_index()
    sensitivity = []
    absolute_latitude = x[3][0,:,0].detach()+37
    km_per_degree = torch.stack([torch.full_like(absolute_latitude,110.574),
        111.32*torch.cos(absolute_latitude*math.pi/180)],dim=-1)
    # Elevation is exported unchanged. No km derivative claimed without its unit.
    analytic = mean_grad[0,:,:2]/km_per_degree
    analytic_distribution = distribution_grad[:,:, :, :2]/km_per_degree[None,None,:,:]
    require(torch.isfinite(mean_grad).all() and torch.isfinite(distribution_grad).all(),'Nonfinite query derivative')
    with torch.no_grad():
        for step in fd_km:
            for axis in (0,1):
                plus = [t.detach().clone() for t in x]; minus = [t.detach().clone() for t in x]
                delta = step/km_per_degree[:,axis]
                plus[3][0,:,axis] += delta; minus[3][0,:,axis] -= delta
                p = decoded_tensor(model(*plus)[model.output_layout.index('pga')],norm)
                m = decoded_tensor(model(*minus)[model.output_layout.index('pga')],norm)
                fd = (p[3]-m[3])[0]/(2*step)
                a = analytic[:,axis]
                dist_fd = torch.stack([(p[c]-m[c])[0]/(2*step) for c in range(3)],-1)
                dist_a = analytic_distribution[:,0,:,axis].reshape(mdn.shape[-2],3,-1).permute(2,0,1)
                sensitivity.append(dict(step_km=step,axis=('north','east')[axis],
                    gradient_mean_max_abs=float(a[valid].abs().max()),
                    finite_difference_mean_max_abs=float(fd[valid].abs().max()),
                    mean_fd_gradient_max_error=float((fd[valid]-a[valid]).abs().max()),
                    distribution_gradient_max_abs=float(dist_a[valid].abs().max()),
                    distribution_fd_max_abs=float(dist_fd[valid].abs().max()),
                    distribution_fd_gradient_max_error=float((dist_a[valid]-dist_fd[valid]).abs().max()),evidence=evidence))
    gates = {n:float(p.detach()) for n,p in model.named_parameters()
             if 'gate' in n and p.numel()==1}
    branches = base_branches
    result = dict(evidence=evidence, effective_keys=int(x[2].sum()),effective_queries=int(valid.sum()),
        gates=gates,branches=branches,first_residual=bool(model.pga_cross_attention.first_residual),
        first_residual_attention_multiplier=(float(model.pga_cross_attention.first_residual_gate.detach())
            if model.pga_cross_attention.first_residual_gate is not None else 1.),
        query_gradient_path='eval + enable_grad; no outer no_grad/cache; no optimizer',
        elevation_derivative='NOT_RUN: metadata unit not certified')
    if output is not None:
        output.mkdir(parents=True,exist_ok=True)
        stage_table.to_csv(output/'query_stages.csv',index=False)
        pd.DataFrame(sensitivity).to_csv(output/'query_sensitivity.csv',index=False)
        np.savez_compressed(output/'query_trace_arrays.npz',**arrays,
            mean_coordinate_gradient=mean_grad.detach().cpu().numpy(),
            distribution_coordinate_gradient=distribution_grad.detach().cpu().numpy())
        write_json(output/'query_gates_branches.json',result)
    return result,stage_table,pd.DataFrame(sensitivity)


def permutation_controls(model, inputs):
    model.eval()
    p = torch.arange(inputs[0].shape[1]-1,-1,-1,device=inputs[0].device)
    joint, coords = [v.clone() for v in inputs], [v.clone() for v in inputs]
    for i in (0,1,2,5):
        joint[i] = joint[i][:,p]
    coords[1] = coords[1][:,p]
    with torch.no_grad():
        a=model(*inputs)[model.output_layout.index('pga')]
        b=model(*joint)[model.output_layout.index('pga')]
        c=model(*coords)[model.output_layout.index('pga')]
    error=float((a-b).abs().max())
    require(error<=2e-5,'Joint station permutation is not equivalent')
    return dict(joint_max_abs=error,coords_only_max_abs=float((a-c).abs().max()),
                coords_only_scope='mechanism intervention; NOT a physical scoring population')


def nested_sets(event, mask, cfg):
    from fe01.data import geometry
    selected,queries,roles,_ = geometry(event,mask,cfg,'normal')
    # Stable pick then station-ID order, independent of errors or labels' values.
    selected = sorted(selected,key=lambda i:(float(event['picks'][i]),str(event['ids'][i])))
    maximum = np.array(selected[:8],dtype=int)
    common = queries[~np.isin(queries,maximum)]
    return {k:maximum[:k] for k in (1,3,5,8) if len(maximum)>=k and len(common)}, common, \
        dict(available=len(selected),common_queries=len(common),unsupported=[k for k in (1,3,5,8) if len(maximum)<k or not len(common)],
             maximum_input_ids=event['ids'][maximum].tolist(),query_ids=event['ids'][common].tolist(),
             rule='fixed event/T; stable pick+station ID; queries exclude maximum K; never duplicate/move T')
