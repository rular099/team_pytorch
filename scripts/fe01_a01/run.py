#!/usr/bin/env python3
"""Explicit stage runner. No scheduler invocation, test mode or downloads."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def config(index):
    from fe01.config import load
    matrix=json.loads((ROOT/'configs/fe01_a01/matrix.json').read_text())
    if not 0<=index<len(matrix['new_training']): raise ValueError('A01 index must be 0..4')
    return load(ROOT/matrix['new_training'][index]['config'])


def diagnostic(cfg,audit_dir,output,device,cap):
    import numpy as np
    import pandas as pd
    import torch
    from fe01.data import read_event,split_catalog,prepare_sample
    from fe01.engine import inputs_to
    from fe01.windows import build_window,CAPABILITIES
    from fe01.config import sha256
    from fe01_review.requests import read_frame,population_audit
    from fe01_a01.engine import check_audit
    from fe01_a01.identity import restore_initial,old_checkpoint
    from fe01_a01.model import build_model
    from fe01_a01.diagnostics import select_probes,query_trace,permutation_controls,nested_sets
    from fe01_a01.availability import future_audit,scale_audit
    from fe01_a01.provenance import require,new_output,write_json
    lock=check_audit(cfg,audit_dir)
    output=new_output(output)
    resolved=copy.deepcopy(cfg);resolved.update(lock['effective_config'])
    # Frozen old requests; DiTing uses TEAM metadata planning, no substitute encoder.
    old_dir=Path(cfg['a01']['original_run'])
    path=old_dir/f"validation_epoch{cfg['a01']['original_selected_epoch']}.csv.gz"
    require(sha256(path)==cfg['a01']['original_request_csv_sha256'],'Frozen request CSV changed')
    frame=read_frame(path)
    population_audit(frame)
    probes,strata=select_probes(frame,cap)
    probes.to_csv(output/'probe_decisions.csv',index=False);strata.to_csv(output/'probe_strata.csv',index=False)
    identity=frame.merge(probes[['dataset_id','event_id','elapsed_time','geometry_protocol']],how='inner',
                         on=['dataset_id','event_id','elapsed_time','geometry_protocol'],validate='many_to_one')
    identity.to_csv(output/'probe_requests.csv.gz',index=False)
    model=build_model(resolved,device)
    if cfg['model_family']=='diting_pretrained_frozen':
        restore_initial(model,resolved,lock);evidence='matched_initial_checkpoint_NOT_TRAINED'
    else:
        cp,_=old_checkpoint(old_dir/'best.pth',cfg['a01']['original_selected_epoch'],old_dir,resolved)
        model.load_state_dict(cp['model_state_dict'],strict=True)
        del cp
        evidence=('real_ON_checkpoint' if cfg['absolute_amplitude_mode'].startswith('physical')
                  else 'real_ON_checkpoint_inference_intervention')
    catalog=split_catalog(resolved,'val').set_index(['dataset_id','event_id'])
    stages=[];sensitivities=[];controls=[];future=[];scales=[];nested=[];gates=[];branches=[]
    for n,probe in enumerate(probes.itertuples()):
        row=catalog.loc[(probe.dataset_id,probe.event_id)].copy()
        row['dataset_id'],row['event_id']=probe.dataset_id,probe.event_id
        event=read_event(row,resolved,probe.elapsed_time,full_for_audit=True)
        # Target selection is by station identity, not prediction or PGA value.
        request=identity[(identity.dataset_id==probe.dataset_id)&(identity.event_id==probe.event_id)&
            (identity.elapsed_time==probe.elapsed_time)&(identity.geometry_protocol==probe.geometry_protocol)].sort_values('station_id')
        query_ids=request.station_id.head(resolved['model_params']['n_pga_targets']).tolist()
        q=np.array([list(event['ids']).index(v) for v in query_ids])
        sample=prepare_sample(event,resolved,probe.elapsed_time,probe.geometry_protocol,queries=q)
        require(sample['info']['input_count']==probe.input_count,'Probe actual K changed')
        require(sample['info']['input_ids']==json.loads(request.input_ids.iloc[0]),'Probe inputs differ from locked request')
        inputs=inputs_to(sample,device)
        trace_dir=output/'traces'/f'decision{n:04d}'
        result,table,sensitivity=query_trace(model,inputs,resolved['target_normalization'],trace_dir,evidence)
        tag=dict(decision=n,dataset_id=probe.dataset_id,event_id=probe.event_id,elapsed_time=probe.elapsed_time,
                 geometry_protocol=probe.geometry_protocol,input_count=probe.input_count,evidence=evidence)
        stages.append(table.assign(**tag));sensitivities.append(sensitivity.assign(**tag))
        gates.append(dict(**tag,**result['gates']))
        branches.append(dict(**tag,**result['branches']))
        controls.append(dict(**tag,**permutation_controls(model,inputs)))
        scales.extend(dict(**tag,**r) for r in scale_audit(model,inputs,cfg['absolute_amplitude_mode'])['rows'])
        if n<5:
            future.extend({**tag,**r} for r in future_audit(model,resolved,event,probe.elapsed_time,device,'locked_decision')['rows'])
        if n==0:
            from fe01.windows import clock
            cap_native=CAPABILITIES[cfg['model_family']]
            trigger=float(event['picks'][event['picks']>event['reference_sample']].min()-event['reference_sample'])/100
            for case,time in (('trigger_boundary',trigger),('native_capacity',cap_native.max_elapsed_sample()/100),
                              ('tail_missing',probe.elapsed_time),('internal_legal_zero',probe.elapsed_time),
                              ('invalid_station',probe.elapsed_time)):
                changed=copy.deepcopy(event)
                _,cutoff=clock(event['reference_sample'],time,100)
                if case=='tail_missing':changed['storage'][1,...,max(0,cutoff-25):]=False
                elif case=='internal_legal_zero':changed['waveform'][...,max(0,cutoff-25):cutoff-15]=0
                elif case=='invalid_station':changed['storage'][-1]=False
                if changed['waveform'].shape[-1]<=cutoff:
                    future.append({**tag,'case':case,'elapsed_time':time,'status':'UNSUPPORTED_NO_STORED_FUTURE'})
                    continue
                future.extend({**tag,**r} for r in future_audit(model,resolved,changed,time,device,case)['rows'])
            _,_,beyond=build_window(event['waveform'],event['storage'],event['reference_sample'],
                (cap_native.max_elapsed_sample()+1)/100,cap_native)
            require(beyond['status']=='unsupported','Native capacity unexpectedly truncates history')
            future.append({**tag,'case':'capacity_plus_one','status':'PASS_UNSUPPORTED_HISTORY','mutation':'capacity_guard'})
        _,mask,_=build_window(event['waveform'],event['storage'],event['reference_sample'],probe.elapsed_time,CAPABILITIES[cfg['model_family']])
        sets,common,ninfo=nested_sets(event,mask.all(1),resolved)
        nested.append(dict(**tag,**ninfo))
        for k,indices in sets.items():
            changed=copy.deepcopy(event)
            changed['input_allowed']=np.zeros(len(event['ids']),dtype=bool);changed['input_allowed'][indices]=True
            pack=prepare_sample(changed,resolved,probe.elapsed_time,'normal',queries=common[:resolved['model_params']['n_pga_targets']])
            ni=inputs_to(pack,device)
            _,nt,ns=query_trace(model,ni,resolved['target_normalization'],trace_dir/f'nested_K{k}',evidence)
            stages.append(nt.assign(**{**tag,'input_count':k},control='nested_same_queries'))
            sensitivities.append(ns.assign(**{**tag,'input_count':k},control='nested_same_queries'))
        print('DIAGNOSTIC',n+1,len(probes),probe.event_id,probe.elapsed_time,flush=True)
    pd.concat(stages).to_csv(output/'query_stages.csv',index=False)
    pd.concat(sensitivities).to_csv(output/'query_sensitivity.csv',index=False)
    for name,rows in (('gates',gates),('branch_norms',branches),('station_controls',controls),('future_audit',future),('scale_audit',scales)):
        pd.DataFrame(rows).to_csv(output/(name+'.csv'),index=False)
    write_json(output/'nested_support.json',nested)
    write_json(output/'diagnostics.json',dict(status='PASS',evidence=evidence,probe_count=len(probes),cap_per_stratum=cap,
        audit_lock_sha256=sha256(Path(audit_dir)/'protocol.lock.json'),a01_code_sha256=lock['a01_code_sha256'],
        upstream_status='UPSTREAM_CAUSALITY_UNKNOWN',trace_query_rule='first native query-cap station IDs, full population retained in probe manifest'))
    from fe01_a01.plotting import plot_diagnostics
    plot_diagnostics(output,evidence)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['environment','audit','diagnostics','pilot','train','eval','pack'])
    parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--world',type=int,default=16)
    parser.add_argument('--probe-cap',type=int,default=32)
    args=parser.parse_args()
    from fe01_a01.provenance import write_json,new_output,source_identity,require,read_json
    output=Path(args.output)
    if args.stage=='environment':
        from fe01.engine import runtime
        import torch
        out=new_output(output)
        require(torch.cuda.is_available(),'Environment gate must run on an allocated DCU node')
        value=runtime();value.update(status='PASS',source=source_identity())
        write_json(out/'environment.json',value)
        return
    if args.stage=='pack':
        from fe01_a01.pack import review_package
        review_package(output)
        return
    cfg=config(args.index)
    root=Path(cfg['output_root'])
    audit_dir=root/'audits'/cfg['run_id']
    from fe01_a01 import engine
    if args.stage=='audit':
        env=read_json(root/'environment'/'environment.json')
        require(env['status']=='PASS' and env['source']['a01_code_sha256']==source_identity()['a01_code_sha256'],'Matching environment gate required')
        engine.audit(cfg,audit_dir,args.device,args.world)
    elif args.stage=='diagnostics':
        diagnostic(cfg,audit_dir,root/'diagnostics'/cfg['run_id'],args.device,args.probe_cap)
    elif args.stage=='pilot':
        from fe01.config import sha256
        gate=read_json(root/'diagnostics'/cfg['run_id']/'diagnostics.json')
        require(gate['status']=='PASS' and gate['audit_lock_sha256']==sha256(audit_dir/'protocol.lock.json'),'Matching diagnostics required')
        engine.pilot(cfg,audit_dir,root/'pilot'/cfg['run_id'],args.device)
    elif args.stage=='train':
        lock=engine.check_audit(cfg,audit_dir)
        require(int(os.environ.get('WORLD_SIZE','1'))==lock['world_size'],'Training DDP world differs from audited sampler')
        engine.train(cfg,audit_dir)
    elif args.stage=='eval':
        from fe01_a01.evaluation import export,compare,export_old_ON
        from fe01_review.requests import read_frame
        import torch
        run=root/cfg['run_id']
        best=torch.load(run/'best.pth',map_location='cpu',weights_only=False)['epoch']
        for label,checkpoint,epoch in (('selected',run/'best.pth',best),('epoch12',run/'last.pth',12)):
            target=root/'evaluation'/cfg['run_id']/label
            frame=export(cfg,audit_dir,checkpoint,epoch,target,args.device)
            if cfg['model_family']!='diting_pretrained_frozen':
                on=export_old_ON(cfg,audit_dir,label,root/'evaluation'/(cfg['run_id']+'_reused_ON')/label,args.device)
                compare(on,frame,root/'comparisons'/cfg['run_id']/label)
            elif cfg['absolute_amplitude_mode'].startswith('no_absolute'):
                on_cfg=config(3)
                on_path=root/'evaluation'/on_cfg['run_id']/label/'predictions.csv.gz'
                require(on_path.is_file(),'DiTing matching ON evaluation missing; evaluate index3 first')
                compare(read_frame(on_path),frame,root/'comparisons'/cfg['run_id']/label)


if __name__=='__main__':
    main()
