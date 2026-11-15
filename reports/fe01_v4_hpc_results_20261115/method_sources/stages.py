"""Stage workers: compute allocations only; never submit another stage."""
import gzip
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from .contracts import EPOCHS, EVENT, ROOT, contract
from .io import attempt,csv_read,csv_write,fingerprint,read_json,require,sha256,verify_seal,write_json
from .orchestration import prerequisites,stage_identity,verify_plan,require_stage


def common_config(plan):
    # Torch-free relocation for CPU metadata stages.
    old=read_json(Path(plan['source_session'])/'plan.json')
    cfg=json.loads(json.dumps(old['runs'][0]['config']))
    for key,env in [('root','DATA_ROOT'),('split_manifest','FE01_SPLIT_MANIFEST')]:
        if os.environ.get(env):
            cfg['data'][key]=os.environ[env]
    return cfg


def data_gate(plan,cfg):
    old=read_json(Path(plan['source_session'])/'plan.json')
    row=old['runs'][0]
    lock=read_json(Path(row['audit_dir'])/'protocol.lock.json')
    require(sha256(cfg['data']['split_manifest'])==lock['data_identity']['split_manifest_sha256'],'Split byte SHA changed')
    records=[]
    for original,info in lock['data_identity']['shards'].items():
        p=Path(cfg['data']['root'])/Path(original).parent.name/Path(original).name
        require(p.is_file() and p.stat().st_size==info['bytes'],'Shard absent/size mismatch')
        same=p.stat().st_mtime_ns==info['mtime_ns']
        if not same:
            require(sha256(p)==info['sha256'],'Shard byte SHA changed')
        records.append(dict(relative_path=str(p.relative_to(cfg['data']['root'])),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,
                            sha256=info['sha256'],verification='existing SHA at identical size/mtime' if same else 'new complete byte hash'))
    return records


def shared_data_gate(session,plan,cfg):
    """One full shard hash per actual stat identity, shared by all CPU array tasks."""
    old=read_json(Path(plan['source_session'])/'plan.json')
    lock_path=Path(old['runs'][0]['audit_dir'])/'protocol.lock.json'
    lock=read_json(lock_path)
    signature={}
    for original in lock['data_identity']['shards']:
        path=Path(cfg['data']['root'])/Path(original).parent.name/Path(original).name
        stat=path.stat()
        signature[str(path)]=[stat.st_size,stat.st_mtime_ns]
    identity=dict(lock_sha256=sha256(lock_path),data_config_sha256=fingerprint(cfg['data']),
                  split_sha256=sha256(cfg['data']['split_manifest']),stat_signature_sha256=fingerprint(signature))
    key=fingerprint(identity)
    cache=Path(session)/'inputs'
    print('DATA_IDENTITY_SHARED_CACHE',key,flush=True)
    with attempt(cache,'data_identity',key,identity,blocking=True) as out:
        if out is not None:
            write_json(out/'data_identity.json',dict(identity=identity,shards=data_gate(plan,cfg)))
    path=cache/'stages/data_identity'/key
    verify_seal(path,identity)
    return dict(cache_seal_sha256=sha256(path/'SEAL.json'),**read_json(path/'data_identity.json'))


def catalog_rows(plan,events):
    from .covariates import catalog
    cfg=common_config(plan)
    return catalog(cfg,pd.DataFrame(events,columns=EVENT))


def all_stage_directories(session,stage):
    p=verify_plan(session)
    result=[]
    for task in p['tasks'][stage]:
        d=Path(session)/'stages'/stage/str(task['index'])
        verify_seal(d,stage_identity(session,stage,task['index']))
        result.append(d)
    return result


def store_index(session,stage,required=True):
    if not (Path(session)/'stages'/stage).exists() and not required:
        return {}
    if not required:
        p=verify_plan(session)
        if not all((Path(session)/'stages'/stage/str(t['index'])/'SEAL.json').is_file() for t in p['tasks'][stage]):
            return {}  # explicitly incomplete optional scope; completed-but-corrupt seals still fail below
    index={}
    for directory in all_stage_directories(session,stage):
        for r in read_json(directory/'stores.json'):
            family=read_json(directory/'worker.json')['family']
            key=(r['dataset_id'],r['event_id'],r['geometry_protocol'],family)
            require(key not in index,'Duplicate event store')
            index[key]=dict(r,directory=directory)
    return index


def read_event_stores(index,event,geometry,kind='station_file'):
    frames=[]
    for family in EPOCHS:
        key=(*event,geometry,family)
        require(key in index,'Missing event/model numeric store: '+str(key))
        record=index[key]
        if record.get(kind):
            frames.append(csv_read(record['directory']/record[kind]))
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def stream_csv(frame,path):
    """Append only inside this fresh attempt, allowing bounded-memory event processing."""
    exists=Path(path).exists()
    if exists:
        import csv
        opener=gzip.open if str(path).endswith('.gz') else open
        with opener(path,'rt',newline='') as stream:
            columns=next(csv.reader(stream))
        require(set(frame.columns).issubset(set(columns)),'Stream schema changed; preserve attempt rather than corrupt CSV')
        frame=frame.reindex(columns=columns)
    frame.to_csv(path,mode='a' if exists else 'x',header=not exists,index=False,compression='gzip' if str(path).endswith('.gz') else None)


def analyze(session,out):
    from .map_readiness import evaluate as ttum,population_summary,pair_events
    from .shadow import evaluate as shadow
    plan=verify_plan(session)
    # Recompute the old paired protocol with the newly available *shared* quality covariates.
    # These are still legacy-snapshot endpoints, separately named from fixed replay scores.
    decision_quality=pd.concat([csv_read(d/'decision_manifest.csv.gz') for d in all_stage_directories(session,'covariates')],ignore_index=True)
    from .comparison import analyze_existing
    hydrated=out/'legacy_with_physical_covariates'
    hydrated.mkdir()
    analyze_existing(plan['source_session'],hydrated,sidecar=decision_quality)
    from .interactions import analyze as interaction_analysis
    interaction_analysis(plan['source_session'],out/'interaction_contrasts',sidecar=decision_quality)
    from .additional_endpoints import analyze as additional_analysis
    additional_analysis(plan['source_session'],out/'additional_endpoints')
    stations=store_index(session,'replay_stations')
    grids={}
    for geo in ('normal','random'):
        grids.update(store_index(session,'replay_grid_'+geo,False))
    truth=pd.concat([csv_read(d/'crossing_truth.csv.gz') for d in all_stage_directories(session,'crossing_truth')],ignore_index=True)
    times=[]
    support=[]
    event_summaries=[]
    for event in plan['cohort']:
        for geo in ('normal','random'):
            f=read_event_stores(stations,event,geo)
            # Preserve exact common fixed panels for all models, including later inputs and failed frames.
            common=None
            for family in EPOCHS:
                ids=set(f[f.family.eq(family)].station_id)
                require(common is None or ids==common,'Fixed panel differs across models')
                common=ids
            for family in EPOCHS:
                key=(*event,geo,family)
                if key in grids:
                    map_station=csv_read(grids[key]['directory']/grids[key]['station_file'])
                    ready=map_station[['station_id','elapsed_time','numeric_map_ready_seconds']]
                    part=f.family.eq(family)
                    joined=f.loc[part,['station_id','elapsed_time']].merge(ready,on=['station_id','elapsed_time'],how='left',validate='one_to_one')
                    if 'numeric_ready_seconds' not in f:
                        f['numeric_ready_seconds']=np.nan
                    f.loc[part,'numeric_ready_seconds']=joined.numeric_map_ready_seconds.to_numpy()
            frame,tt=ttum(f)
            times.append(tt)
            stream_csv(frame,out/'map_readiness_by_frame.csv.gz')
            tr=truth[truth.dataset_id.eq(event[0])&truth.event_id.eq(event[1])]
            outcomes,summary=shadow(f,tr,contract()['shadow'])
            stream_csv(outcomes,out/'shadow_station_outcomes.csv.gz')
            event_summaries.append(summary)
            support.append(dict(dataset_id=event[0],event_id=event[1],geometry_protocol=geo,targets=len(common),
                                station_rows=len(f),grid_measured_clock_available=all((*event,geo,family) in grids for family in EPOCHS),
                                grid_truth='none',area_status='NA_no_common_independent_spatial_support'))
    t=pd.concat(times,ignore_index=True)
    s=pd.concat(event_summaries,ignore_index=True)
    csv_write(t,out/'event_time_to_usable_map.csv')
    csv_write(t,out/'event_time_to_stable_usable_map.csv')
    csv_write(population_summary(t),out/'ttum_support.csv')
    ttum_pairs=pair_events(t)
    csv_write(ttum_pairs,out/'ttum_pairwise_events.csv')
    from .event_statistics import ttum_summary,shadow_summary
    csv_write(ttum_summary(ttum_pairs),out/'ttum_paired_summary.csv')
    csv_write(s,out/'shadow_event_summary.csv')
    csv_write(pd.DataFrame(support),out/'shadow_support_audit.csv')
    csv_write(pd.DataFrame(support)[['dataset_id','event_id','geometry_protocol','area_status']],out/'shadow_area_support.csv')
    keys=['dataset_id','event_id','geometry_protocol','panel','threshold_logpga','p_alarm','tau','scene']
    pairs=[]
    for other in ('PhaseNet','EQT','TEAM'):
        j=s[s.family.eq('DiTing')].merge(s[s.family.eq(other)],on=keys,how='outer',suffixes=('_diting','_comparator'),validate='one_to_one')
        j['comparator']=other
        j['delta_shadow_lower_bound']=j.failure_fraction_lower_diting-j.failure_fraction_upper_comparator
        j['delta_shadow_upper_bound']=j.failure_fraction_upper_diting-j.failure_fraction_lower_comparator
        pairs.append(j)
    shadow_pairs=pd.concat(pairs,ignore_index=True)
    csv_write(shadow_pairs,out/'shadow_pairwise_events.csv')
    csv_write(shadow_summary(shadow_pairs),out/'shadow_paired_summary.csv')
    write_json(out/'map_readiness_contract.json',contract()['map_readiness'])
    write_json(out/'timing_contract.json',contract()['latency'])
    (out/'limitations.md').write_text('全部为validation、seed42、固定selected。TTUM为回顾性台站支撑研究指标，不是工程认证。\n'
        '没有独立共同面积支撑，面积NA；没有完整网格任务时 measured map clock未知。\n'
        '通信/真实触发与上游离线因果性未认证。未知和未达标在完整分母中保留。\n')


def render_task(session,task,out):
    from .plotting import animate_event,build_index,shadow_map_html
    plan=verify_plan(session)
    stations=store_index(session,'replay_stations')
    stage='replay_grid_'+task['geometry']
    grids=store_index(session,stage)
    analysis=Path(session)/'stages/analyze/0'
    tt=csv_read(analysis/'event_time_to_usable_map.csv') if analysis.exists() else None
    # Stream the large outcome file once per render task, retaining only this task's events.
    wanted={tuple(x) for x in task['events']}
    shadow_parts=[]
    if (analysis/'shadow_station_outcomes.csv.gz').is_file():
        for chunk in csv_read(analysis/'shadow_station_outcomes.csv.gz',chunksize=100000):
            mask=[(a,b) in wanted for a,b in zip(chunk.dataset_id,chunk.event_id)]
            shadow_parts.append(chunk.loc[mask & chunk.geometry_protocol.eq(task['geometry'])])
    shadow=pd.concat(shadow_parts,ignore_index=True) if shadow_parts else pd.DataFrame()
    entries=[]
    for event in task['events']:
        f=read_event_stores(stations,event,task['geometry'])
        g=read_event_stores(grids,event,task['geometry'],'grid_file') if all((*event,task['geometry'],family) in grids for family in EPOCHS) else pd.DataFrame()
        if len(g):
            profiles=read_event_stores(grids,event,task['geometry'])
            timing=profiles[['family','station_id','elapsed_time','numeric_map_ready_seconds']]
            f=f.merge(timing,on=['family','station_id','elapsed_time'],how='left',validate='one_to_one')
        name=fingerprint(event+[task['geometry']])[:20]
        selected=tt[tt.dataset_id.eq(event[0])&tt.event_id.eq(event[1])&tt.geometry_protocol.eq(task['geometry'])] if tt is not None else None
        animate_event(f,g,out/name,selected)
        if len(shadow):
            part=shadow[shadow.dataset_id.eq(event[0])&shadow.event_id.eq(event[1])]
            shadow_map_html(part,out/name/'shadow_station_map.html')
        entries.append(dict(dataset_id=event[0],event_id=event[1],geometry=task['geometry'],href=name+'/index.html',
                            label=event[0]+'/'+event[1]+' '+task['geometry'],status='RENDERED',grid_available=bool(len(g))))
    build_index(entries,out/'map_index.html')
    write_json(out/'visuals_index.json',entries)


def execute(session,key,index,device='cuda'):
    print('START_STAGE',key,'TASK',index,'SESSION',session,flush=True)
    plan=verify_plan(session)
    require(key in plan['tasks'] and 0<=index<len(plan['tasks'][key]),'Unknown task')
    prerequisites(session,key)
    task=plan['tasks'][key][index]
    identity=stage_identity(session,key,index)
    with attempt(session,key,index,identity) as out:
        if out is None:
            print('ALREADY_COMPLETE')
            return
        import sys
        import importlib.metadata
        versions={}
        for name in ('numpy','pandas','scipy','h5py','matplotlib','torch','seisbench'):
            try:
                versions[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name]='NOT_INSTALLED'
        write_json(out/'runtime_environment.json',dict(python=sys.version,versions=versions,
            private_env_file_sha256=sha256(plan['env_file']),loaded_modules=os.environ.get('LOADEDMODULES','UNKNOWN'),
            resources={k:os.environ.get(k) for k in ('SLURM_JOB_ID','SLURM_ARRAY_TASK_ID','SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE')},
            scientific_environment_sha256=fingerprint(plan['scientific_environment'])))
        if key=='inventory':
            from .sources import checkpoint_spec,inspect,verify_export_seals
            write_json(out/'source_exports.json',verify_export_seals(plan['source_session']))
            inventory=[]
            for family in EPOCHS:
                for policy in ('selected','epoch12'):
                    inventory.append(inspect(checkpoint_spec(plan['source_session'],family,policy)))
            write_json(out/'checkpoint_inventory.json',inventory)
        elif key=='compare_existing':
            from .comparison import analyze_existing
            analyze_existing(plan['source_session'],out)
            from .interactions import analyze as interaction_analysis
            interaction_analysis(plan['source_session'],out/'interaction_contrasts')
            from .additional_endpoints import analyze as additional_analysis
            additional_analysis(plan['source_session'],out/'additional_endpoints')
            from .report_plots import render_tables
            render_tables(out,out/'figures')
        elif key in ('covariates','crossing_truth'):
            cfg=common_config(plan)
            write_json(out/'data_identity.json',shared_data_gate(session,plan,cfg))
            rows=catalog_rows(plan,task['events'])
            if key=='covariates':
                from .covariates import hydrate
                hydrate(rows,None,contract(),out)
            else:
                from .crossings import extract
                extract(rows,contract()['shadow']['research_logpga_thresholds'],out)
        elif key=='latency':
            from .latency import benchmark
            benchmark(session,task,out,device)
        elif key in ('model_smoke','replay_pilot','replay_stations','origin_replay','long_extension') or key.startswith('replay_grid_'):
            from .replay import runner_task
            runner_task(session,task,'replay_grid' if key.startswith('replay_grid_') else key,out,device)
        elif key=='analyze':
            analyze(session,out)
        elif key.startswith('render_'):
            render_task(session,task,out)
        elif key=='pack':
            from .packaging import review_bundle
            review_bundle(session,out)
        else:
            raise ValueError('No implementation for stage '+key)
