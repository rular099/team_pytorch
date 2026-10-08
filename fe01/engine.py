"""Manual experiment engine. No scheduler commands, auto-download, or test fallback."""
import contextlib
import copy
import datetime
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from . import BASE_COMMIT, ROOT
from .config import fingerprint, safe_output, sha256, validate, write_json
from .data import (TrainingDataset, collate, geometry, prepare_sample, read_event,
                   split_catalog)
from .metrics import checkpoint_selection, decode, grouped_metrics, score_rows
from .model import build_model, common_state, model_audit, state_fingerprint
from .sampling import TimeSampler, stable_rng
from .windows import CAPABILITIES, build_window, clock


def runtime(experiment=None):
    versions={}
    for name in ['torch','seisbench','numpy','scipy','h5py']:
        try:
            versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name]='not installed'
    try:
        if not (ROOT/'.git').exists(): raise FileNotFoundError('source archive')
        commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],stderr=subprocess.DEVNULL).decode().strip()
        dirty=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain']).strip())
    except (subprocess.CalledProcessError,FileNotFoundError):
        identity=ROOT/'SOURCE_IDENTITY.json'
        commit=json.loads(identity.read_text())['commit'] if identity.is_file() else 'unknown source identity'
        dirty=None
    return dict(host=socket.gethostname(),job_id=os.environ.get('SLURM_JOB_ID'),
                python=sys.version,versions=versions,rocm=torch.version.hip,
                device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',
                git_commit=commit,git_dirty=dirty,base_commit=experiment.BASE_COMMIT if experiment else BASE_COMMIT,command=sys.argv,
                recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())


def distributed_device():
    world=int(os.environ.get('WORLD_SIZE','1'))
    rank=int(os.environ.get('RANK','0'))
    local=int(os.environ.get('LOCAL_RANK','0'))
    device=torch.device('cuda',local) if torch.cuda.is_available() else torch.device('cpu')
    if device.type=='cuda':
        torch.cuda.set_device(device)
    if world>1:
        dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',init_method='env://')
    return device,rank,world


def synchronize(device):
    if torch.device(device).type=='cuda':
        torch.cuda.synchronize(device)


def inputs_to(sample,device):
    return [x.unsqueeze(0).to(device) for x in sample['inputs']]


def evaluation_rows(model,cfg,row,elapsed,protocol,device,replay=False,reference='first_p_pick'):
    if getattr(model,'_fe01_query_cache',None) is None:
        # Scope is exactly one event/T. Amortize the expensive frozen encoder
        # across station queries while recomputing it at every new cutoff.
        with model.same_cutoff_query_cache():
            return evaluation_rows(model,cfg,row,elapsed,protocol,device,replay,reference)
    event=read_event(row,cfg,elapsed,reference)
    cap=CAPABILITIES[cfg['model_family']]
    _,channel_mask,info=build_window(event['waveform'],event['storage'],event['reference_sample'],
                                     elapsed,cap,cfg['window']['protocol'])
    _,universe,roles,actual_protocol=geometry(event,channel_mask.all(axis=1),cfg,protocol,replay=replay)
    if info['status']=='unsupported':
        return [],dict(event_id=event['event_id'],elapsed_time=elapsed,geometry_protocol=protocol,**info)
    if not len(universe):
        return [],dict(event_id=event['event_id'],elapsed_time=elapsed,status='empty',reason='no_target')
    rows=[]
    # Query independence is tested. One-query inference is the conservative
    # fallback preserving the common downstream even if a future config couples queries.
    chunk_size=cfg.get('query_chunk_size',1)
    if chunk_size>1 and not cfg.get('query_independence_verified',False):
        raise ValueError('Query batching requires an explicit numeric independence audit')
    for start in range(0,len(universe),chunk_size):
        query=universe[start:start+chunk_size]
        sample=prepare_sample(event,cfg,elapsed,protocol,queries=query,replay=replay)
        if sample['inputs'] is None:
            return rows,sample['info']
        info=sample['info']
        synchronize(device);begin=time.perf_counter()
        with torch.no_grad():
            outputs=model(*inputs_to(sample,device))
        synchronize(device);duration=time.perf_counter()-begin
        mdn=outputs[model.output_layout.index('pga')][0,:len(query)].cpu().numpy()
        weights,means,sigmas=decode(mdn,cfg['target_normalization'])
        truth=event['pga'][query]
        finite=np.isfinite(mdn).all(axis=(-1,-2)) & (sigmas>0).all(-1)
        scores=score_rows(truth,np.where(finite[:,None],weights,1/weights.shape[-1]),
                           np.where(finite[:,None],means,0),np.where(finite[:,None],sigmas,1))
        for index,station in enumerate(query):
            record=dict(dataset_id=event['dataset_id'],event_id=event['event_id'],station_id=event['ids'][station],
                latitude=float(event['coords'][station,0]),longitude=float(event['coords'][station,1]),
                event_latitude=float(event['event_location'][0]),event_longitude=float(event['event_location'][1]),
                depth=float(event['event_location'][2]),magnitude=event['magnitude'],split=event['split'],
                seed=cfg['seed'],model_family=cfg['model_family'],elapsed_time=float(elapsed),
                actual_elapsed_time=info['actual_elapsed_time'],geometry_protocol=actual_protocol,
                target_role=roles[station],truth=float(truth[index]),
                status='supported' if finite[index] else 'failure',reason='' if finite[index] else 'numerical_failure',
                window_protocol=cfg['window']['protocol'],time_reference=reference,
                requested_decision_sample=info['requested_decision_sample'],current_sample=info['current_sample'],
                cutout_exclusive=info['cutout_exclusive'],history_start_sample=info['history_start_sample'],
                history_end_sample=info['history_end_sample'],history_seconds=info['history_seconds'],
                absolute_decision_utc=info['absolute_decision_utc'],latest_received_sample=info['latest_received_sample'],
                input_count=info['input_count'],input_ids=json.dumps(info['input_ids']),
                input_coords=json.dumps(info['input_coords']),
                valid_seconds=float(np.mean(info['selected_valid_seconds'])),
                padding_fraction=float(1-np.mean(info['selected_valid_seconds'])*100/cfg['native_n_samples']),
                units='log10(m/s^2)',forward_seconds=duration/len(query),
                encoder_seconds=model._fe01_encoder_seconds/len(query),
                target_valid=True,station_valid_count=info['input_count'],
                storage_mask_rule='valid_n_samples / received prefix / selected input IDs',
                mdn_weights=json.dumps(weights[index].tolist()),mdn_mu=json.dumps(means[index].tolist()),
                mdn_sigma=json.dumps(sigmas[index].tolist()))
            for name,value in scores.items():
                record[name]=float(value[index]) if finite[index] else float('nan')
            rows.append(record)
            if cfg.get('fe02', {}).get('enabled'):
                record['variant_id'] = cfg['variant_id']
                record['event_fusion'] = cfg['event_fusion']
    return rows,info


def evaluate(model,cfg,split='val',device='cpu',allow_test=False,times=None,event_limit=None,random_times=None):
    catalog=split_catalog(cfg,split,allow_test)
    if event_limit:
        catalog=catalog.iloc[:event_limit]
    rows=[];statuses=[]
    model.eval()
    for _,row in catalog.iterrows():
        event_times=times or cfg['realtime']['fixed_times']
        if random_times is not None:
            matched=random_times.event_id.eq(row.event_id)&random_times.dataset_id.eq(row.dataset_id)
            event_times=random_times.loc[matched,'elapsed_time'].tolist()
            if not event_times:
                raise ValueError('Random validation time manifest misses event identity')
        for elapsed in event_times:
            for protocol in ['normal','random']:
                predictions,info=evaluation_rows(model,cfg,row,elapsed,protocol,device)
                rows.extend(predictions)
                statuses.append(dict(dataset_id=row.dataset_id,event_id=row.event_id,elapsed_time=elapsed,
                    geometry_protocol=protocol,**{k:info.get(k) for k in ['status','reason','required_n_samples']}))
    return pd.DataFrame(rows),pd.DataFrame(statuses)


def causal_audit(model,cfg,row,device):
    event=read_event(row,cfg,1,full_for_audit=True)
    sample=prepare_sample(event,cfg,1)
    if sample['inputs'] is None:
        raise ValueError('Causal audit sample has no input/target')
    modified=copy.deepcopy(event)
    cutout=event['cutout_exclusive']
    modified['waveform'][...,cutout:]=np.nan
    modified_sample=prepare_sample(modified,cfg,1)
    for a,b in zip(sample['inputs'],modified_sample['inputs']):
        if not torch.equal(a,b):
            raise AssertionError('Future perturbation changed model input')
    model.eval()
    def shapes(value):
        if torch.is_tensor(value): return list(value.shape)
        if isinstance(value,dict): return {k:shapes(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)): return [shapes(v) for v in value]
        return str(type(value))
    observed=[]
    hook=model.waveform_model.encoder.register_forward_hook(lambda m,x,y:observed.append(shapes(y)))
    with torch.no_grad():
        first=model(*inputs_to(sample,device))
        first_features=model._last_raw_station_emb.detach().clone()
        second=model(*inputs_to(modified_sample,device))
    hook.remove()
    if any(not torch.isfinite(x).all() for x in [*first,*second,first_features,model._last_raw_station_emb]):
        raise FloatingPointError('Non-finite feature/prediction cannot pass causal audit')
    feature_delta=float((first_features-model._last_raw_station_emb).abs().max())
    maximum=max(float((a-b).abs().max()) for a,b in zip(first,second))
    if maximum>1e-6 or feature_delta>1e-6:
        raise AssertionError('Future perturbation changed prediction')
    return dict(status='PASS',event_id=str(row['event_id']),first_forbidden_sample=cutout,
                modified='all raw samples from exclusive cutoff onward → NaN',max_prediction_delta=maximum,
                max_feature_delta=feature_delta,observed_encoder_outputs=observed,
                station_feature_shape=list(first_features.shape),
                upstream_causality='not certified: existing offline resampling/filtering is unchanged')


def query_audit(model,cfg,row,device):
    event=read_event(row,cfg,3)
    query=np.flatnonzero(np.isfinite(event['pga'])&event['query_allowed'])[:min(3,cfg['model_params']['n_pga_targets'])]
    if len(query)<2: raise ValueError('Query audit needs at least two permitted stations')
    model.eval();deltas=[]
    with torch.no_grad(),model.same_cutoff_query_cache():
        batch=prepare_sample(event,cfg,3,queries=query)
        original=model(*inputs_to(batch,device))[-1][0,:len(query)]
        reverse=prepare_sample(event,cfg,3,queries=query[::-1])
        reversed_output=model(*inputs_to(reverse,device))[-1][0,:len(query)].flip(0)
        singles=[]
        for point in query:
            singles.append(model(*inputs_to(prepare_sample(event,cfg,3,queries=[point]),device))[-1][0,0])
        single=torch.stack(singles)
        for output in (reversed_output,single):
            torch.testing.assert_close(original,output,atol=1e-5,rtol=1e-5)
            deltas.append(float((original-output).abs().max()))
    return dict(status='PASS',event_id=row.event_id,queries=len(query),max_mdn_parameter_delta=max(deltas),
                atol=1e-5,rtol=1e-5,rule='batch vs order reversal vs independent single coordinate queries')


def data_identity(cfg,with_hash=True):
    records={}
    for split in ['train','val']:
        catalog=split_catalog(cfg,split)
        for path in sorted(set(catalog.hdf5_path)):
            if path not in records:
                source=Path(path)
                if not source.is_file():
                    raise FileNotFoundError(source)
                records[path]=dict(bytes=source.stat().st_size,mtime_ns=source.stat().st_mtime_ns,
                                   sha256=sha256(source) if with_hash else None)
    return dict(split_manifest_sha256=sha256(cfg['data']['split_manifest']),shards=records,
                spatial_manifest_sha256=sha256(cfg['spatial']['manifest']) if cfg['spatial']['enabled'] else None)


def audit_population(cfg,destination):
    """Pre-model, label-independent support rules; train-only target normalization."""
    cohorts={};exclusions=[];values=[];plans=[];support=[];random_times=[]
    # A family-independent common prefix gives the same cohort/selection keys.
    common=copy.deepcopy(cfg)
    common.update(model_family='diting_pretrained_frozen' if cfg.get('fe02', {}).get('enabled') else 'team_original_scratch',native_n_samples=10000)
    capabilities = {cfg['model_family']: CAPABILITIES[cfg['model_family']]} if cfg.get('fe02', {}).get('enabled') else CAPABILITIES
    for split in ('train','val'):
        catalog=split_catalog(cfg,split)
        limit=cfg.get('audit_limits',{}).get(split+'_events')
        if limit:
            catalog=catalog.iloc[:limit]
        kept=[]
        for _,row in catalog.iterrows():
            try:
                event=read_event(row,common,1)
            except ValueError as exc:
                if 'no KNET stations' not in str(exc):
                    raise
                exclusions.append(dict(split=split,dataset_id=row.dataset_id,event_id=row.event_id,reason=str(exc)))
                continue
            check=prepare_sample(event,common,1,'random')
            if check['inputs'] is None:
                exclusions.append(dict(split=split,dataset_id=row.dataset_id,event_id=row.event_id,reason=check['info']['reason']))
                continue
            kept.append([row.dataset_id,row.event_id])
            if split=='train':
                valid=event['pga'][event['query_allowed']&np.isfinite(event['pga'])]
                values.extend(valid.tolist())
            else:
                for elapsed in cfg['realtime']['common_times']:
                    current=read_event(row,common,elapsed)
                    _,mask,_=build_window(current['waveform'],current['storage'],current['reference_sample'],elapsed,CAPABILITIES['team_original_scratch'])
                    for protocol in ('normal','random'):
                        inputs,queries,roles,_=geometry(current,mask.all(1),common,protocol)
                        for query in queries:
                            plans.append(dict(dataset_id=row.dataset_id,event_id=row.event_id,elapsed_time=float(elapsed),
                                geometry_protocol=protocol,station_id=current['ids'][query],target_role=roles[query],
                                input_ids=json.dumps(current['ids'][inputs].tolist())))
                rng=stable_rng(cfg['sampling_seed'],row.dataset_id,row.event_id,'validation_time_manifest')
                draws=[]
                while len(draws)<3:
                    value=float(rng.integers(100,2001)/100)
                    if value not in cfg['realtime']['fixed_times']: draws.append(value)
                random_times.extend(dict(dataset_id=row.dataset_id,event_id=row.event_id,elapsed_time=float(t),draw=i) for i,t in enumerate(draws))
            for elapsed in cfg['realtime']['fixed_times']:
                _,stop=clock(event['reference_sample'],elapsed,100)
                start=math.floor(event['reference_sample'])-500
                for family,cap in capabilities.items():
                    support.append(dict(split=split,dataset_id=row.dataset_id,event_id=row.event_id,model_family=family,
                        elapsed_time=elapsed,required_n_samples=stop-start,native_n_samples=cap.native_n_samples,
                        status='supported' if stop-start<=cap.native_n_samples else 'unsupported_history'))
        if not kept:
            raise ValueError('No usable '+split+' events under the preregistered support rules')
        cohorts[split]=kept
    target=np.asarray(values,dtype=np.float64)
    if not len(target) or not np.isfinite(target).all() or target.std()<1e-8:
        raise ValueError('Train-only target normalization is degenerate')
    normalization=dict(enabled=True,mean=float(target.mean()),std=float(target.std()),unit='log10(m/s^2)',
        fitted_split='train',targets=len(target),rule='all finite labels at permitted KNET train stations; population std')
    plan=pd.DataFrame(plans)
    # Every primary selection cell must exist before looking at any predictions.
    for protocol in ('normal','random'):
        for elapsed in cfg['realtime']['common_times']:
            if not ((plan.geometry_protocol==protocol)&(plan.elapsed_time==elapsed)&plan.target_role.ne('observed_input')).any():
                raise ValueError('No noninput validation targets in locked common cell; protocol cannot run')
    plan.to_csv(destination/'validation_population.csv.gz',index=False)
    pd.DataFrame(exclusions,columns=['split','dataset_id','event_id','reason']).to_csv(destination/'cohort_exclusions.csv',index=False)
    pd.DataFrame(support).to_csv(destination/'model_time_support.csv',index=False)
    pd.DataFrame(random_times).to_csv(destination/'random_validation_times_manifest.csv',index=False)
    write_json(destination/'cohort_normalization_audit.json',dict(cohorts=cohorts,normalization=normalization,
        exclusion_rule='no KNET support / no causal input or query at 1s; locked before model prediction',excluded_events=len(exclusions)))
    return dict(audited_cohorts=cohorts,target_normalization=normalization),fingerprint(plan.to_dict('records'))


def code_identity(experiment=None):
    if experiment is not None:
        return experiment.code_identity()
    paths=list((ROOT/'fe01').glob('*.py'))+[ROOT/name for name in ('gemini_models.py','gemini_util_light.py','train_light.py')]
    paths+=list((ROOT/'tools').rglob('*.py'))+list((ROOT/'diting/config').glob('*.yml'))
    import dtbench.training.modeling
    dependency=Path(dtbench.training.modeling.__file__).parents[1]
    files={str(p.relative_to(ROOT)):sha256(p) for p in paths}
    files.update({'dtbench/'+str(p.relative_to(dependency)):sha256(p) for p in dependency.rglob('*.py')})
    return fingerprint(files)


def audit(cfg,destination,device='cpu',existing_data_audit=None,experiment=None):
    validate(cfg)
    if experiment is not None:
        experiment.validate(cfg)
    destination=safe_output(cfg['output_root'],destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Audit output is nonempty; choose a new audit directory')
    destination.mkdir(parents=True,exist_ok=True)
    if existing_data_audit:
        data=json.loads(Path(existing_data_audit).read_text())
        if experiment is not None:
            current_data = data_identity(cfg, with_hash=False)
            if set(current_data['shards']) != set(data['shards']) or current_data['spatial_manifest_sha256'] != data['spatial_manifest_sha256']:
                raise ValueError('Reusable data audit shard set/spatial identity mismatch')
        if data_identity(cfg,with_hash=False)['split_manifest_sha256']!=data['split_manifest_sha256']:
            raise ValueError('Reusable data audit split mismatch')
        for path,record in data['shards'].items():
            stat=Path(path).stat()
            if stat.st_size!=record['bytes'] or stat.st_mtime_ns!=record['mtime_ns']:
                raise ValueError('Data audit reuse: file stat changed; redo hash audit')
    else:
        data=data_identity(cfg)
    write_json(destination/'data_split_audit.json',data)
    effective,population_sha=audit_population(cfg,destination)
    resolved=copy.deepcopy(cfg);resolved.update(effective)
    model=(experiment.build_model if experiment else build_model)(resolved,device)
    write_json(destination/'model_interface_audit.json',
               experiment.model_audit(model, resolved) if experiment else model_audit(model))
    train=split_catalog(resolved,'train')
    causal=causal_audit(model,resolved,train.iloc[0],device)
    write_json(destination/'causal_boundary_audit.json',causal)
    query_result=query_audit(model,resolved,train.iloc[0],device)
    write_json(destination/'query_independence_audit.json',query_result)
    if experiment is not None:
        experiment.extra_audit(model, resolved, train.iloc[0], device, destination)
    effective.update(query_independence_verified=True,
                     query_chunk_size=min(cfg.get('query_chunk_size',1),cfg['model_params']['n_pga_targets']))
    sampler=TimeSampler(max_sample=CAPABILITIES[cfg['model_family']].max_elapsed_sample(),
                        bins=cfg['realtime']['bins'],probabilities=cfg['realtime']['probabilities'])
    if cfg['window']['protocol']=='native_rolling_v2':
        sampler=TimeSampler(max_sample=9000,bins=cfg['realtime']['bins'],probabilities=cfg['realtime']['probabilities'])
    sampler_audit=sampler.audit()
    if not sampler_audit['passed']:
        raise AssertionError('Sampler density audit failed')
    write_json(destination/'sampler_distribution_audit.json',sampler_audit)
    weight_manifest=json.loads(Path(cfg['pretrained_manifest']).read_text()) if 'pretrained' in cfg['model_family'] else None
    lock=dict(config_sha256=fingerprint(cfg),code_sha256=code_identity(experiment),effective_config=effective,validation_population_sha256=population_sha,
              data_identity=data,sampler=sampler_audit,
              model_family=cfg['model_family'],common_times=cfg['realtime']['common_times'],
              pretrained_manifest_sha256=sha256(cfg['pretrained_manifest']) if weight_manifest else None,
              upstream_causality='offline preprocessing not certified',status='AUDIT_PASS')
    if experiment is not None:
        lock.update(variant_id=cfg['variant_id'], event_fusion=cfg['event_fusion'],
                    encoder_checkpoint_sha256=weight_manifest['models']['diting']['files']['weights']['sha256'])
    write_json(destination/'protocol.lock.json',lock)
    write_json(destination/'runtime.json',runtime(experiment))
    capabilities={f:c.manifest() for f,c in CAPABILITIES.items() if experiment is None or f == cfg['model_family']}
    capabilities[cfg['model_family']].update(verified_native_forward=True,
        verified_input_lengths=[cfg['native_n_samples']],verification_device=str(device),
        verification='actual registered model forward in this audit; other families not certified here')
    write_json(destination/'capability_manifest.json',capabilities)
    if weight_manifest:
        write_json(destination/'pretrained_manifest.json',weight_manifest)
    print('AUDIT_PASS',destination,flush=True)


def check_audit(cfg,audit_dir,experiment=None):
    lock=json.loads((Path(audit_dir)/'protocol.lock.json').read_text())
    if lock['status']!='AUDIT_PASS' or lock['config_sha256']!=fingerprint(cfg):
        raise ValueError('Audit/config identity mismatch; run matching manual audit')
    if lock['code_sha256']!=code_identity(experiment): raise ValueError('Implementation changed after audit')
    if sha256(cfg['data']['split_manifest'])!=lock['data_identity']['split_manifest_sha256']:
        raise ValueError('Frozen split changed after audit')
    if cfg['spatial']['enabled'] and sha256(cfg['spatial']['manifest'])!=lock['data_identity']['spatial_manifest_sha256']:
        raise ValueError('Spatial holdout changed after audit')
    for path,record in lock['data_identity']['shards'].items():
        stat=Path(path).stat()
        if stat.st_size!=record['bytes'] or stat.st_mtime_ns!=record['mtime_ns']:
            raise ValueError('Data file changed after hash audit')
    if lock['pretrained_manifest_sha256'] and sha256(cfg['pretrained_manifest'])!=lock['pretrained_manifest_sha256']:
        raise ValueError('Pretrained identity changed after audit')
    return lock


def loss(outputs,labels,model,cfg,valid):
    import gemini_models as legacy
    from train_light import distribution_mean_aux_loss
    t=cfg['training']
    selected,true=legacy.select_loss_components(outputs,labels,model.output_layout,t['res_comps'])
    value=legacy.mixture_density_loss_full(selected,true,res_comps=t['res_comps'],
        res_weight=np.asarray(t['res_weight']),pga_target_valid=valid,pga_target_normalization=cfg['target_normalization'])
    auxiliary=distribution_mean_aux_loss(outputs,labels,model.output_layout,t['res_comps'],
        np.asarray(t['res_weight']),valid,t['distribution_mean_loss'],pga_target_normalization=cfg['target_normalization'])
    return value+(auxiliary if auxiliary is not None else 0)


def train(cfg,audit_dir,resume=False,experiment=None):
    validate(cfg)
    if experiment is not None:
        experiment.validate(cfg)
    lock=check_audit(cfg,audit_dir,experiment)
    source_config_sha=fingerprint(cfg)
    cfg=copy.deepcopy(cfg);cfg.update(lock['effective_config'])
    device,rank,world=distributed_device()
    output=safe_output(cfg['output_root'],Path(cfg['output_root'])/cfg['run_id'])
    random.seed(cfg['seed']+rank);np.random.seed(cfg['seed']+rank);torch.manual_seed(cfg['seed']+rank)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(cfg['seed']+rank)
    existing=output.exists() and any(output.iterdir())
    if existing and not resume:
        raise FileExistsError('Nonempty FE01 output; explicit resume required')
    if resume and not (output/'last.pth').is_file():
        raise FileNotFoundError('Resume requires last.pth; never falls back to best')
    if rank==0:
        output.mkdir(parents=True,exist_ok=True)
    if world>1:
        dist.barrier()
    identity=fingerprint(dict(config=cfg,audit_lock=lock,world_size=world))
    dataset=TrainingDataset(cfg)
    if cfg.get('audit_limits',{}).get('train_events'):
        dataset.catalog=dataset.catalog.iloc[:cfg['audit_limits']['train_events']]
    batch=cfg['training']['microbatch']
    global_batch=cfg['training']['global_batch']
    if global_batch%(batch*world):
        raise ValueError('microbatch*world must divide the preregistered global batch')
    accumulation=global_batch//(batch*world)
    sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,
                               seed=cfg['sampling_seed'],drop_last=True)
    loader=DataLoader(dataset,batch_size=batch,sampler=sampler,collate_fn=collate,
                      num_workers=cfg['training']['workers'],drop_last=True)
    updates_per_epoch=len(loader)//accumulation
    if updates_per_epoch<1:
        raise ValueError('Cohort too small for effective batch')
    total_updates=updates_per_epoch*cfg['training']['epochs']
    if cfg['training']['max_updates']:
        total_updates=min(total_updates,cfg['training']['max_updates'])
    core=(experiment.build_model if experiment else build_model)(cfg,device)
    model=DistributedDataParallel(core,device_ids=[device.index] if device.type=='cuda' else None,
                                  find_unused_parameters=True) if world>1 else core
    optimizer=torch.optim.Adam([p for p in core.parameters() if p.requires_grad],
                               lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay'])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=total_updates)
    start_epoch=0;updates=0;best=float('inf');committed_journals=[];curves=[];best_epoch=0
    def payload(epoch):
        return dict(epoch=epoch,updates=updates,model_state_dict=core.state_dict(),
                    optimizer=optimizer.state_dict(),scheduler=schedule.state_dict(),
                    identity=identity,best_selection_score=best,
                    source_config_sha256=source_config_sha,
                    curve_records=curves,best_epoch=best_epoch,
                    rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                    rank_rng=None,config=cfg,common_initial_state_sha256=initial)
    initial=state_fingerprint(common_state(core))
    if resume:
        checkpoint=torch.load(output/'last.pth',map_location=device,weights_only=False)
        if checkpoint['identity']!=identity:
            raise ValueError('Resume identity differs: config/data/weights/protocol/world')
        core.load_state_dict(checkpoint['model_state_dict'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer']);schedule.load_state_dict(checkpoint['scheduler'])
        start_epoch=checkpoint['epoch'];updates=checkpoint['updates'];best=checkpoint['best_selection_score']
        initial=checkpoint['common_initial_state_sha256']
        committed_journals=checkpoint.get('committed_journals',[])
        curves=checkpoint['curve_records'];best_epoch=checkpoint['best_epoch']
        if rank==0 and best_epoch==start_epoch:
            # Recover a crash between last publication and best publication.
            stored_best=torch.load(output/'best.pth',map_location='cpu',weights_only=False) if (output/'best.pth').is_file() else None
            if stored_best is None or stored_best['epoch']!=best_epoch:
                torch.save(checkpoint,output/'best.pth.tmp');os.replace(output/'best.pth.tmp',output/'best.pth')
        rng=checkpoint.get('rank_rng')
        rng= rng[rank] if rng is not None else checkpoint
        torch.set_rng_state(rng['rng'].cpu());random.setstate(rng['python_rng']);np.random.set_state(rng['numpy_rng'])
        if device.type=='cuda' and rng['cuda_rng']:
            torch.cuda.set_rng_state_all(rng['cuda_rng'])
    elif rank==0:
        write_json(output/'model_interface_audit.json',
                   experiment.model_audit(core, cfg) if experiment else model_audit(core))
        torch.save(payload(0),output/'init.pth')
    if rank==0:
        write_json(output/'resolved_config.json',cfg)
        if not (output/'runtime.json').exists(): write_json(output/'runtime.json',runtime(experiment))
        write_json(output/'protocol.lock.json',lock)
    # Each attempt has its own journal. Only a published epoch checkpoint commits
    # its samples; interrupted attempts remain available as forensic evidence.
    attempt=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')+f'_rank{rank}'
    if rank==0: write_json(output/f'runtime_{attempt}.json',runtime(experiment))
    journal=output/'sample_journals';journal.mkdir(exist_ok=True)
    curve_path=output/'training_curves.csv'
    if rank==0 and curves: pd.DataFrame(curves).to_csv(curve_path,index=False)
    for epoch in range(start_epoch,cfg['training']['epochs']):
        if updates>=total_updates:
            break
        dataset.epoch=epoch;sampler.set_epoch(epoch);model.train();optimizer.zero_grad()
        sum_loss=0.;microsteps=0;started=time.perf_counter()
        records_path=journal/f'epoch{epoch+1:04d}_{attempt}.jsonl'
        with records_path.open('w') as stream:
            for micro,part in enumerate(loader):
                if micro>=updates_per_epoch*accumulation or updates>=total_updates:
                    break
                inputs=[x.to(device) for x in part['inputs']];labels=[x.to(device) for x in part['labels']]
                synchronize_grad=(micro+1)%accumulation==0
                context=model.no_sync() if world>1 and not synchronize_grad else contextlib.nullcontext()
                with context:
                    outputs=model(*inputs)
                    value=loss(outputs,labels,core,cfg,inputs[4])
                    if not torch.isfinite(value):
                        raise FloatingPointError('Non-finite FE01 loss')
                    (value/accumulation).backward()
                for info in part['info']:
                    stream.write(json.dumps(dict(epoch=epoch+1,rank=rank,microstep=micro,optimizer_update=updates+1,**info))+'\n')
                sum_loss+=float(value.detach());microsteps+=1
                if synchronize_grad:
                    torch.nn.utils.clip_grad_norm_(core.parameters(),cfg['training']['gradient_clip'])
                    optimizer.step();schedule.step();optimizer.zero_grad();updates+=1
        if world>1:
            dist.barrier()
        selection=None
        if rank==0:
            frame,status=evaluate(core,cfg,device=device,times=cfg['realtime']['common_times'],
                                  event_limit=cfg.get('audit_limits',{}).get('val_events'))
            if not status.status.eq('supported').all():
                raise ValueError('Incomplete common validation support; checkpoint selection refused')
            keys=['dataset_id','event_id','elapsed_time','geometry_protocol','station_id','target_role','input_ids']
            planned=pd.read_csv(Path(audit_dir)/'validation_population.csv.gz',dtype={'event_id':str,'station_id':str})
            if fingerprint(frame[keys].sort_values(keys).to_dict('records'))!=fingerprint(planned[keys].sort_values(keys).to_dict('records')):
                raise ValueError('Validation population differs from preregistered audit')
            selection=checkpoint_selection(frame,cfg['realtime']['common_times'])
            frame.to_csv(output/f'validation_epoch{epoch+1}.csv.gz',index=False)
            improved=selection<best
            if improved:
                best=selection;best_epoch=epoch+1
            row=dict(epoch=epoch+1,updates=updates,train_loss_rank0=sum_loss/max(microsteps,1),
                     validation_selection=selection,lr=optimizer.param_groups[0]['lr'],
                     elapsed_seconds=time.perf_counter()-started,global_batch=global_batch,
                     planned_samples=len(dataset),actual_samples=microsteps*batch*world,
                     dropped_for_equal_update_budget=len(dataset)-microsteps*batch*world)
            curves.append(row)
            write_json(output/f'epoch{epoch+1:04d}_{attempt}.json',row)
        scores=[best,selection]
        if world>1:
            dist.broadcast_object_list(scores,src=0)
        best=scores[0]
        local_rng=dict(rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                       sample_journal=str(records_path.relative_to(output)))
        rank_rng=[None]*world if rank==0 else None
        if world>1:
            dist.gather_object(local_rng,rank_rng,dst=0)
        else:
            rank_rng=[local_rng]
        if rank==0:
            checkpoint=payload(epoch+1);checkpoint['rank_rng']=rank_rng
            checkpoint['committed_journals']=committed_journals+[r['sample_journal'] for r in rank_rng]
            # Atomic publication with existing outputs preserved on interruption.
            torch.save(checkpoint,output/'last.pth.tmp');os.replace(output/'last.pth.tmp',output/'last.pth')
            if improved:
                torch.save(checkpoint,output/'best.pth.tmp');os.replace(output/'best.pth.tmp',output/'best.pth')
            pd.DataFrame(curves).to_csv(curve_path,index=False)
            print(json.dumps(row),flush=True)
            committed_journals=checkpoint['committed_journals']
        if world>1:
            dist.barrier()
    if world>1:
        dist.destroy_process_group()


def load_checkpoint(cfg,path,device,experiment=None):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    allowed={checkpoint.get('source_config_sha256'),fingerprint(checkpoint['config'])}
    if fingerprint(cfg) not in allowed:
        raise ValueError('Checkpoint/config mismatch')
    cfg.clear();cfg.update(copy.deepcopy(checkpoint['config']))
    model=(experiment.build_model if experiment else build_model)(cfg,device)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    model.eval()
    return model,checkpoint


def export_evaluation(cfg,checkpoint_path,destination,device='cpu',split='val',allow_test=False,lock_sha=None,random_manifest=None,test_ledger=None,experiment=None):
    if split=='test' and (not allow_test or lock_sha is None or test_ledger is None):
        raise ValueError('Explicit --allow-test, protocol lock SHA and test exposure ledger required')
    ledger=json.loads(Path(test_ledger).read_text()) if test_ledger else None
    if ledger is not None and not all(k in ledger for k in ['prior_project_exposure','authorized_purpose','recorded_by']):
        raise ValueError('Ledger must describe prior project exposure (including unknown), purpose and recorder')
    destination=safe_output(cfg['output_root'],destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Evaluation output exists')
    if experiment is not None:
        experiment.check_evaluation_contract(cfg, checkpoint_path)
    model,checkpoint=load_checkpoint(cfg,checkpoint_path,device,experiment)
    train_lock=Path(checkpoint_path).parent/'protocol.lock.json'
    training_protocol=json.loads(train_lock.read_text())
    if training_protocol['code_sha256']!=code_identity(experiment): raise ValueError('Inference code differs from training audit')
    if split=='test' and sha256(train_lock)!=lock_sha:
        raise ValueError('Frozen protocol lock hash mismatch')
    random_times=pd.read_csv(random_manifest,dtype={'event_id':str}) if random_manifest else None
    frame,status=evaluate(model,cfg,split,device,allow_test,random_times=random_times)
    destination.mkdir(parents=True,exist_ok=True)
    checkpoint_sha=sha256(checkpoint_path)
    frame['checkpoint_sha256']=checkpoint_sha;frame['checkpoint_epoch']=checkpoint['epoch']
    frame['config_sha256']=fingerprint(cfg)
    frame.to_csv(destination/'predictions.csv.gz',index=False)
    status.to_csv(destination/'support_status.csv',index=False)
    grouped_metrics(frame).to_csv(destination/'metrics_by_time_target.csv',index=False)
    # NPZ strings are non-object arrays and load without allow_pickle.
    np.savez_compressed(destination/'predictions.npz',**{k:frame[k].to_numpy() if frame[k].dtype.kind in 'biuf' else frame[k].fillna('').astype(str).to_numpy(dtype=str) for k in frame})
    write_json(destination/'provenance.json',dict(checkpoint_sha256=checkpoint_sha,checkpoint_epoch=checkpoint['epoch'],
        config_sha256=fingerprint(cfg),split=split,protocol_lock_sha256=sha256(train_lock),
        window_protocol=cfg['window']['protocol'],stage=cfg.get('stage'),
        split_manifest_sha256=sha256(cfg['data']['split_manifest']),
        validation_population_sha256=json.loads(train_lock.read_text())['validation_population_sha256'],
        random_times_manifest_sha256=sha256(random_manifest) if random_manifest else None,runtime=runtime(experiment),
        **(dict(variant_id=cfg['variant_id'], event_fusion=cfg['event_fusion'],
                encoder_checkpoint_sha256=training_protocol['encoder_checkpoint_sha256']) if experiment else {})))
    if ledger is not None:
        write_json(destination/'test_exposure_ledger.json',dict(**ledger,current_evaluation=dict(
            checkpoint_sha256=checkpoint_sha,protocol_lock_sha256=lock_sha,config_sha256=fingerprint(cfg),split=split)))
