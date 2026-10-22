"""Frozen ON inference, arbitrary coordinate queries and same-cutoff-only cache."""
import contextlib
import copy
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp, ndtr
from .contracts import DECISION, EPOCHS, EVENT, ON
from .io import csv_read, csv_write, fingerprint, read_json, require, sha256, write_json
from .sources import checkpoint_spec, inspect, load_export, runtime_config


def load_model(source_session,family,device='cuda'):
    import torch
    if family == 'DiTing':
        from fe01_a01.model import build_model
    else:
        from fe01.model import build_model
    from fe01.model import state_fingerprint
    spec=checkpoint_spec(source_session,family)
    identity,payload=inspect(spec,return_payload=True)
    cfg=runtime_config(payload['config'])
    cfg['absolute_amplitude_mode']=ON
    begin=time.perf_counter()
    model=build_model(cfg,device)
    before_encoder=state_fingerprint(model.waveform_model.encoder.state_dict())
    model.load_state_dict(payload['model_state_dict'],strict=True)
    actual=state_fingerprint(model.waveform_model.encoder.state_dict())
    require(actual==identity['encoder_sha256'],'Loaded encoder mismatch')
    if family!='TEAM':
        require(before_encoder==actual,'Frozen pretrained encoder differs from authenticated asset')
    model.requires_grad_(False).eval()
    del payload
    if torch.device(device).type=='cuda':
        torch.cuda.synchronize(device)
    identity['construct_and_load_seconds']=time.perf_counter()-begin
    return model,cfg,identity


class Forward:
    def __init__(self,model,cfg,checkpoint_sha,device='cuda'):
        from fe01_review.runners import SystemRunner
        self.runner=SystemRunner(model,cfg['target_normalization'],'fe01',device)
        self.cfg,self.checkpoint_sha,self.device=cfg,checkpoint_sha,device
        self.context=None
        self.physical_pin=None

    def token(self,request):
        return fingerprint([self.checkpoint_sha,ON,request.dataset_id,request.event_id,
                            float(request.elapsed_time),int(request.cutout_exclusive),request.input_ids,request.input_coords])

    @contextlib.contextmanager
    def cutoff(self,request):
        require(self.context is None,'Nested cache forbidden')
        token=self.token(request)
        self.context=token
        self.physical_pin=None
        try:
            with self.runner.cutoff_cache(token):
                yield
        finally:
            self.context=None
            self.physical_pin=None

    def predict(self,event,requests,points=None,chunk=None):
        import torch
        from fe01_review.runners import pack_inputs,sync,tensor_pin
        from fe01.metrics import decode,score_rows,quantile
        request=requests.iloc[0]
        require(self.cfg.get('absolute_amplitude_mode',ON)==ON and
                getattr(self.runner.model,'absolute_amplitude_mode',ON)==ON,'Cache/inference path is ON only')
        require(self.context is None or self.context==self.token(request),
                'Cannot reuse cache across cutoff/input coordinates/checkpoint/mode')
        count=len(points) if points is not None else len(requests)
        chunk=chunk or self.cfg['model_params']['n_pga_targets']
        require(0<chunk<=self.cfg['model_params']['n_pga_targets'],'Invalid query chunk')
        rows=[]
        timing=dict(preprocess_seconds=0.,encoder_seconds=0.,forward_seconds=0.,distribution_seconds=0.)
        for start in range(0,count,chunk):
            sub=requests.iloc[start:start+chunk] if points is None else requests
            p=None if points is None else points[start:start+chunk]
            t=time.perf_counter()
            inputs,info=pack_inputs(event,request,self.cfg,self.cfg['native_n_samples'],'fe01',
                                   queries=sub.station_id.tolist() if p is None else None,grid_points=p)
            pin=tensor_pin(inputs[:3]+[inputs[5]])
            if self.context is not None:
                if self.physical_pin is None:
                    self.physical_pin=pin
                require(pin==self.physical_pin,'Cannot reuse across history/input IDs/coords/mask')
            inputs=[v.to(self.device) for v in inputs]
            sync(self.device)
            timing['preprocess_seconds']+=time.perf_counter()-t
            t=time.perf_counter()
            output=self.runner.forward(inputs,self.context)
            sync(self.device)
            timing['forward_seconds']+=time.perf_counter()-t
            timing['encoder_seconds']+=self.runner.model._fe01_encoder_seconds
            raw=output[self.runner.model.output_layout.index('pga')][0,:len(sub) if p is None else len(p)].detach().cpu().numpy().astype(float)
            require(np.isfinite(raw).all() and (raw[...,2]>0).all(),'Invalid full MDN')
            t=time.perf_counter()
            w,mu,sigma=decode(raw,self.cfg['target_normalization'])
            logits=raw[...,0]
            if p is None:
                y=sub.truth.to_numpy()
                scores=score_rows(y,w,mu,sigma)
                scores['nll']=-logsumexp(logits-logsumexp(logits,axis=-1,keepdims=True)-np.log(sigma*np.sqrt(2*np.pi))-(y[:,None]-mu)**2/(2*sigma*sigma),axis=-1)
            else:
                mean=(w*mu).sum(-1)
                scores=dict(prediction=mean,q025=quantile(.025,w,mu,sigma),q975=quantile(.975,w,mu,sigma),
                            predictive_sigma=np.sqrt(np.maximum((w*(sigma**2+mu**2)).sum(-1)-mean**2,0)))
                scores['width95']=scores['q975']-scores['q025']
            scores['probability_ge_minus1p2']=(w*ndtr((mu+1.2)/sigma)).sum(-1)
            scores['probability_ge_zero']=(w*ndtr(mu/sigma)).sum(-1)
            timing['distribution_seconds']+=time.perf_counter()-t
            for i in range(len(raw)):
                row=sub.iloc[i].to_dict() if p is None else dict(dataset_id=request.dataset_id,event_id=request.event_id,
                    elapsed_time=float(request.elapsed_time),geometry_protocol=request.geometry_protocol,grid_point_id=start+i,
                    latitude=float(p[i,0]),longitude=float(p[i,1]),elevation=float(p[i,2]),truth=None,protocol='grid_replay_v4',
                    input_ids=request.input_ids,input_coords=request.input_coords,input_count=int(request.input_count),
                    absolute_decision_utc=request.get('absolute_decision_utc'),origin_elapsed_seconds=request.get('origin_elapsed_seconds'))
                row.update(status='supported',model_family=self.cfg['model_family'],absolute_amplitude_mode=ON,seed=42,
                           checkpoint_sha256=self.checkpoint_sha,units='log10(m/s^2)',**{k:float(v[i]) for k,v in scores.items()})
                for name,a in [('mdn_logits',logits),('mdn_weights',w),('mdn_mu',mu),('mdn_sigma',sigma)]:
                    row[name]=json.dumps(a[i].tolist())
                row['physical_received_history_sha256']=info['physical_received_history_sha256']
                row['model_input_sha256']=info['model_input_sha256']
                rows.append(row)
        timing['query_readout_seconds']=max(0.,timing['forward_seconds']-timing['encoder_seconds'])
        return pd.DataFrame(rows),timing


def sidecars(session):
    from .orchestration import require_stage
    require_stage(session,'covariates')
    path=Path(session)/'stages/covariates'
    req=pd.concat([csv_read(p/'request_covariates.csv.gz') for p in sorted(path.iterdir()) if p.is_dir()],ignore_index=True)
    grids={}
    for p in sorted(path.iterdir()):
        if p.is_dir():
            for g in read_json(p/'grid_manifest.json'):
                grids[(g['dataset_id'],g['event_id'])]=g
    require(not req.duplicated(['request_id']).any(),'Duplicate new physical request')
    return req,grids


def runner_task(session,task,stage,out,device='cuda'):
    import torch
    from fe01_review.runners import sync,buffer_pin
    from .orchestration import verify_plan
    from .covariates import catalog,physical_event
    from .clocks import fifo
    plan=verify_plan(session)
    from .selection import freeze
    selection=freeze(session)
    sync(device)
    begin=time.perf_counter()
    model,cfg,identity=load_model(plan['source_session'],task['family'],device)
    sync(device)
    cold=time.perf_counter()-begin
    forward=Forward(model,cfg,identity['checkpoint_sha256'],device)
    before=buffer_pin(model)
    req,grids=sidecars(session)
    events=task.get('events',selection['pilot_events'][:2] if stage=='model_smoke' else selection['pilot_events'])
    cat=catalog(cfg,pd.DataFrame(events,columns=EVENT)).set_index(EVENT)
    station_files=[]
    latencies=[]
    checks=[]
    family=task['family']
    legacy=load_export(plan['source_session'],family,'selected',include_mdn=True) if stage=='model_smoke' else None
    for dataset,event_id in events:
        row=cat.loc[(dataset,event_id)].copy()
        row['dataset_id'],row['event_id']=dataset,event_id
        t0=time.perf_counter()
        e=physical_event(row,max_elapsed=90 if stage=='long_extension' else 40 if stage=='origin_replay' else 20)
        io_seconds=time.perf_counter()-t0
        geometry_list=[task['geometry']] if 'geometry' in task else ['normal','random']
        for geo in geometry_list:
            original=req[req.dataset_id.eq(dataset)&req.event_id.eq(event_id)&req.geometry_protocol.eq(geo)]
            grid=grids[(dataset,event_id)]
            times=list(range(1,21))
            if stage=='origin_replay':
                offset=e['clock']['first_p_minus_origin_seconds']
                times=[t-offset for t in (5,10,15,20,30)] if offset is not None else [None]
            if stage=='long_extension':
                times=[40,90]
            if stage=='model_smoke':
                times=[1,3,5,10,20]
            station_frames,grid_frames,timing_rows=[],[],[]
            for t in times:
                group=original[original.elapsed_time.eq(t)].copy() if t in range(1,21) else extension_requests(e,original,t,geo,stage)
                require(len(group)>0,'Empty fixed query panel')
                status='unknown_clock' if t is None else 'out_of_domain' if t<1 else 'supported'
                from fe01.windows import build_window,CAPABILITIES
                if status=='supported':
                    _,_,info=build_window(e['waveform'],e['storage'],e['reference_sample'],t,CAPABILITIES[cfg['model_family']])
                    status=info['status'] if info['status']!='unsupported' else 'unsupported_history'
                if status=='supported' and int(group.input_count.iloc[0])==0:
                    status='no_input'
                if status!='supported':
                    station_frames.append(group.assign(status=status,reason=status,family=family,checkpoint_epoch=EPOCHS[family],prediction=np.nan))
                    if stage in ('replay_pilot','replay_grid','latency'):
                        grid_frames.append(pd.DataFrame([dict(dataset_id=dataset,event_id=event_id,geometry_protocol=geo,
                            elapsed_time=t,grid_point_id=i,latitude=p[0],longitude=p[1],elevation=p[2],prediction=np.nan,
                            width95=np.nan,status=status,reason=status,family=family,checkpoint_epoch=EPOCHS[family],
                            grid_spacing_km=grid['spacing_km'],
                            grid_sha256=grid['grid_sha256']) for i,p in enumerate(grid['points'])]))
                    continue
                sync(device)
                start=time.perf_counter()
                if stage=='model_smoke':
                    verify_smoke(forward,e,group,legacy,family,checks)
                need_grid=stage in ('replay_pilot','replay_grid','latency')
                pts=np.asarray([[v if v is not None else np.nan for v in p] for p in grid['points']],float)
                valid=np.isfinite(pts).all(1)
                with forward.cutoff(group.iloc[0]):
                    station,st=forward.predict(e,group)
                    sync(device)
                    station_seconds=time.perf_counter()-start
                    grid_output=pd.DataFrame()
                    gt=dict(preprocess_seconds=0.,encoder_seconds=0.,forward_seconds=0.,distribution_seconds=0.,query_readout_seconds=0.)
                    if need_grid and valid.any():
                        grid_output,gt=forward.predict(e,group,pts[valid])
                        grid_output['grid_point_id']=np.flatnonzero(valid)
                        from fe01.spatial import coverage
                        grid_output['within_100km_input_support']=coverage(pts[valid],np.asarray(json.loads(group.input_coords.iloc[0])),100)
                        grid_output['metadata_status']=grid['third_coordinate_status']
                        grid_output['grid_sha256']=grid['grid_sha256']
                        grid_output['grid_spacing_km']=grid['spacing_km']
                    if need_grid:
                        invalid=pd.DataFrame([dict(dataset_id=dataset,event_id=event_id,geometry_protocol=geo,
                            elapsed_time=t,grid_point_id=int(i),latitude=pts[i,0],longitude=pts[i,1],elevation=None,
                            prediction=np.nan,status='unsupported_grid_metadata',metadata_status='outside_station_height_interpolation_support',
                            grid_sha256=grid['grid_sha256'],family=family) for i in np.flatnonzero(~valid)])
                        invalid['grid_spacing_km']=grid['spacing_km']
                        grid_output=pd.concat([grid_output,invalid],ignore_index=True)
                sync(device)
                total=time.perf_counter()-start
                station['family']=family
                station['checkpoint_epoch']=EPOCHS[family]
                if len(grid_output):
                    grid_output['family']=family
                    grid_output['checkpoint_epoch']=EPOCHS[family]
                station_frames.append(station)
                if need_grid:
                    grid_frames.append(grid_output)
                record=dict(dataset_id=dataset,event_id=event_id,family=family,geometry_protocol=geo,elapsed_time=t,
                            io_seconds=io_seconds if t==times[0] else 0.,raw_prefix_cache='per event; bytes read once; causal slicing each cutoff',
                            station_targets=len(station),grid_points=len(pts) if need_grid else 0,grid_supported_points=int(valid.sum()) if need_grid else 0,
                            numeric_station_service_seconds=station_seconds,numeric_map_service_seconds=total,
                            scope='whole frame synchronized; archive/render outside',device=str(device),cold_worker_seconds=cold,
                            **{k:st[k]+gt[k] for k in st})
                timing_rows.append(record)
                latencies.append(record)
            if not station_frames:
                continue
            station=pd.concat(station_frames,ignore_index=True)
            boundary='numeric_map_ready_seconds' if stage in ('replay_pilot','replay_grid','latency') else 'numeric_station_ready_seconds'
            station[boundary]=np.nan
            # FIFO scenarios share the exact same clock module used by downstream scores.
            if timing_rows:
                ordered=sorted(timing_rows,key=lambda x:x['elapsed_time'])
                queue=fifo([r['elapsed_time'] for r in ordered],[r['numeric_map_service_seconds']+r['io_seconds'] for r in ordered])
                ready={r['elapsed_time']:q['numeric_ready_seconds'] for r,q in zip(ordered,queue)}
                station[boundary]=station.elapsed_time.map(ready)
            stem=fingerprint([dataset,event_id,geo])[:20]
            csv_write(station,out/(stem+'_stations.csv.gz'))
            if grid_frames:
                csv_write(pd.concat(grid_frames,ignore_index=True),out/(stem+'_grid.csv.gz'))
            station_files.append(dict(dataset_id=dataset,event_id=event_id,geometry_protocol=geo,
                                      station_file=stem+'_stations.csv.gz',grid_file=stem+'_grid.csv.gz' if grid_frames else None,
                                      frames=len(times),queries=original.station_id.nunique(),grid_sha256=grid['grid_sha256']))
    require(buffer_pin(model)==before,'Inference changed model buffers')
    write_json(out/'stores.json',station_files)
    write_json(out/'checkpoint_identity.json',identity)
    write_json(out/'smoke_checks.json',checks)
    csv_write(pd.DataFrame(latencies),out/'latency_frames.csv')
    support=[]
    for item in station_files:
        frame=csv_read(out/item['station_file'])
        for (elapsed,status),part in frame.groupby(['elapsed_time','status'],dropna=False):
            support.append(dict(dataset_id=item['dataset_id'],event_id=item['event_id'],geometry_protocol=item['geometry_protocol'],
                                elapsed_time=float(elapsed) if pd.notna(elapsed) else None,status=status,queries=len(part),
                                family=family,protocol=str(part.protocol.iloc[0]),checkpoint_epoch=EPOCHS[family]))
    csv_write(pd.DataFrame(support),out/'replay_support.csv')
    write_json(out/'worker.json',dict(family=family,checkpoint_load_count=1,cold_worker_seconds=cold,
               total_worker_wall_seconds=time.perf_counter()-begin,
               output_bytes_before_worker_json=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
               output_file_count_before_worker_json=sum(1 for p in out.rglob('*') if p.is_file()),
               device=str(device),torch=torch.__version__,rocm=torch.version.hip,
               peak_device_allocated_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None,
               station_support='explicit statuses retained',render_timing='separate CPU render stage',
               source_session_plan_sha256=plan['source_plan_sha256']))
    write_json(out/'metadata_selection.json',selection)


def extension_requests(e,original,t,geo,stage):
    from .query_grid import input_plan,geometry_covariates
    from fe01.windows import clock
    template=original[original.elapsed_time.eq(1)].copy()
    template['elapsed_time']=t if t is not None else np.nan
    template['protocol']=stage+'_v4'
    if t is None or t<1:
        template['input_count']=0
        return template
    selected=input_plan(e,t,geo)
    current,cutoff=clock(e['reference_sample'],t,e['fs'])
    template['current_sample']=current
    template['cutout_exclusive']=cutoff
    template['history_end_sample']=current
    template['requested_decision_sample']=e['reference_sample']+t*100
    template['actual_elapsed_time']=(current-e['reference_sample'])/100
    template['history_seconds']=(cutoff-int(np.floor(e['reference_sample']))+500)/100
    template['latest_received_sample']=min(current,e['waveform'].shape[-1]-1)
    template['input_ids']=json.dumps(e['ids'][selected].tolist())
    template['input_coords']=json.dumps(e['coords'][selected].tolist())
    template['input_count']=len(selected)
    utc=e['record_start_utc']+current/100 if e['record_start_utc'] is not None else np.nan
    template['absolute_decision_utc']=utc
    template['origin_elapsed_seconds']=utc-e['clock']['origin_utc'] if e['clock']['origin_utc'] is not None else np.nan
    template['fixed_never_input']=False  # separate optional protocol; never included in main TTUM panel
    return template


def verify_smoke(forward,e,group,legacy,family,checks):
    """Real allocated model gates; no synthetic stub can certify these."""
    from fe01_review.runners import pack_inputs,sync,tensor_pin
    import torch
    with forward.runner.capture_features() as base_features:
        base,_=forward.predict(e,group)
    perm,_=forward.predict(e,group.iloc[::-1])
    chunk,_=forward.predict(e,group,chunk=1)
    def delta(a,b):
        aa=a.sort_values('station_id')
        bb=b.sort_values('station_id')
        error=float(np.max(np.abs(aa.prediction.to_numpy()-bb.prediction.to_numpy())))
        for field in ('mdn_logits','mdn_mu','mdn_sigma'):
            left=np.asarray([json.loads(v) for v in aa[field]])
            right=np.asarray([json.loads(v) for v in bb[field]])
            error=max(error,float(np.max(np.abs(left-right))))
        return error
    for name,f in [('query_permutation',perm),('query_chunk_size',chunk)]:
        err=delta(base,f)
        require(err<=1e-5,'Query invariance failed: '+name)
        checks.append(dict(check=name,max_abs=err,status='PASS'))
    grid_points=np.vstack([e['coords'][:2],e['coords'][0]+np.array([.02,.01,0])])
    # Direct arbitrary coordinates and append after stations, under one cutoff cache.
    with forward.cutoff(group.iloc[0]):
        cached,_=forward.predict(e,group)
        forward.predict(e,group,grid_points)
        again,_=forward.predict(e,group)
    require(delta(base,cached)<=1e-5 and delta(base,again)<=1e-5,'Cache/grid append changed station output')
    cutoff=int(group.cutout_exclusive.iloc[0])
    from fe01.model import shape_and_scale,state_fingerprint
    first_query=group.station_id.iloc[:1].tolist()
    original_inputs,_=pack_inputs(e,group.iloc[0],forward.cfg,forward.cfg['native_n_samples'],queries=first_query)
    normalized,stats=shape_and_scale(original_inputs[0],original_inputs[5])
    statistics_sha=state_fingerprint({'normalized':normalized,'stats':stats})
    for name,value in [('NaN',np.nan),('Inf',np.inf),('pulse',1e6)]:
        if cutoff >= e['waveform'].shape[-1]:
            checks.append(dict(check='future_'+name,status='NOT_APPLICABLE_NO_FUTURE_SAMPLES'))
            continue
        changed=dict(e,waveform=e['waveform'].copy())
        changed['waveform'][...,cutoff:]=value
        changed_inputs,_=pack_inputs(changed,group.iloc[0],forward.cfg,forward.cfg['native_n_samples'],queries=first_query)
        normal_other,stats_other=shape_and_scale(changed_inputs[0],changed_inputs[5])
        require(tensor_pin(original_inputs)==tensor_pin(changed_inputs),'Future changed actual forward inputs')
        require(state_fingerprint({'normalized':normal_other,'stats':stats_other})==statistics_sha,'Future changed normalization/statistics')
        with forward.runner.capture_features() as features:
            other,_=forward.predict(changed,group)
        require(len(features)==len(base_features),'Future changed station feature count')
        feature_error=max(float((a-b).abs().max()) for a,b in zip(base_features,features))
        require(feature_error<=1e-5,'Future changed station encoder features')
        err=delta(base,other)
        require(err<=1e-5,'Future waveform affected output')
        checks.append(dict(check='future_'+name,status='PASS',max_abs=err,feature_max_abs=feature_error,
                           input_and_normalization_stats_sha256=statistics_sha))
    label=group.copy()
    label['truth']=label.truth+5
    other,_=forward.predict(e,label)
    require(delta(base,other)<=1e-5,'Label-only change entered model')
    checks.append(dict(check='label_only',status='PASS'))
    changed=dict(e,waveform=e['waveform'].copy())
    inputs=json.loads(group.input_ids.iloc[0])
    idx=list(e['ids']).index(inputs[0])
    legal=np.flatnonzero(e['storage'][idx,:cutoff])
    require(len(legal)>0,'No legal positive-control sample')
    changed['waveform'][idx,:,legal[-1]]+=10
    old_inputs,_=pack_inputs(e,group.iloc[0],forward.cfg,forward.cfg['native_n_samples'],queries=group.station_id.iloc[:1].tolist())
    new_inputs,_=pack_inputs(changed,group.iloc[0],forward.cfg,forward.cfg['native_n_samples'],queries=group.station_id.iloc[:1].tolist())
    require(tensor_pin(old_inputs[:1]+old_inputs[5:])!=tensor_pin(new_inputs[:1]+new_inputs[5:]),'Legal endpoint ignored')
    other,_=forward.predict(changed,group)
    checks.append(dict(check='last_legal_positive_control',input_changed=True,output_max_abs=delta(base,other),status='PASS_INPUT_CONTROL'))
    changed_mask=dict(e,storage=e['storage'].copy())
    changed_mask['storage'][idx,legal[-1]]=False
    with forward.cutoff(group.iloc[0]):
        forward.predict(e,group)
        try:
            forward.predict(changed_mask,group)
        except ValueError as exc:
            require('reuse' in str(exc),'Unexpected cache guard failure: '+str(exc))
            checks.append(dict(check='same_cutoff_changed_mask_rejected',status='PASS'))
        else:
            raise ValueError('Changed mask reused cached station features')
    old=legacy[legacy.dataset_id.eq(group.dataset_id.iloc[0])&legacy.event_id.eq(group.event_id.iloc[0])&legacy.elapsed_time.eq(group.elapsed_time.iloc[0])&legacy.geometry_protocol.eq(group.geometry_protocol.iloc[0])]
    if len(old) and old.input_ids.iloc[0]==group.input_ids.iloc[0] and int(old.current_sample.iloc[0])==int(group.current_sample.iloc[0]):
        common=base[base.station_id.isin(old.station_id)]
        err=delta(common,old)
        require(err<=1e-5,'Overlapping legacy snapshot differs')
        checks.append(dict(check='legacy_overlap',status='PASS',max_abs=err,requests=len(common)))
    else:
        checks.append(dict(check='legacy_overlap',status='INPUT_MANIFEST_DIFFERENCE',
                           old_input_ids=old.input_ids.iloc[0] if len(old) else None,new_input_ids=group.input_ids.iloc[0]))
