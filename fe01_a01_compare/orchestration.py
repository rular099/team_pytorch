"""Manual-only scheduler, immutable run plans and append-only job records."""
import argparse
import datetime as dt
import gzip
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import uuid
from .contracts import CONFIG, EPOCHS, ROOT, contract, source_identity, verify_frozen_core
from .io import csv_write, fingerprint, read_json, require, sha256, verify_seal, write_json

CPU={'inventory','compare_existing','covariates','crossing_truth','analyze','pack'}
DCU={'model_smoke','replay_pilot','replay_stations','replay_grid','latency','origin_replay','long_extension'}
STAGES=CPU|DCU|{'render'}


def stage_key(stage,geometry='normal'):
    return stage+'_'+geometry if stage in ('replay_grid','render') else stage


def session_path(run_id):
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',run_id),'Unsafe run ID')
    root=Path(os.environ['ON_EVAL_OUTPUT_ROOT']).resolve()
    return root/run_id


def initialize_env(path,inherit_session):
    """Only explicit prepare initializes a private file; never rewrites one."""
    path=Path(path)
    if path.exists():
        return
    inherited=Path(inherit_session).resolve()
    private=inherited/'runtime.private.env'
    require(private.is_file(),'Existing working A01 session runtime.private.env not found: '+str(private))
    project=inherited.parent.parent
    text='''# Private operational paths. Keep outside Git. Reuses the working A01 settings.\n'''
    text+='source '+shlex.quote(str(private))+'\n'
    values={'ON_EVAL_PROJECT_ROOT':str(project),'ON_EVAL_SOURCE_SESSION':str(inherited),
            'ON_EVAL_RELEASE_ROOT':str(ROOT),'ON_EVAL_OUTPUT_ROOT':str(project/'outputs/fe01_on_eval_v4'),
            'ON_EVAL_BLOCK_EVENTS':'32','ON_EVAL_MAX_CONCURRENT':'8','ON_EVAL_CONDA_ENV':'zb'}
    for k,v in values.items():
        text+='export '+k+'="${'+k+':-'+v+'}"\n' if k in ('ON_EVAL_BLOCK_EVENTS','ON_EVAL_MAX_CONCURRENT','ON_EVAL_CONDA_ENV') else 'export '+k+'='+shlex.quote(v)+'\n'
    text+='export ON_EVAL_GRES="${A01_SINGLE_GRES:?Existing DCU gres is required}"\n'
    text+='export ON_EVAL_PARTITION="${A01_PARTITION:-}"\n'
    text+='export ON_EVAL_INHERITED_CODE_ROOT="$A01_CODE_ROOT"\n'
    text+='export ON_EVAL_PYTHON="${A01_PYTHON:-python}"\n'
    text+='export ON_EVAL_CONFIG_PYTHON="${A01_PYTHON:-${A01_CONFIG_PYTHON:-python3}}"\n'
    text+='export FE01_CODE_ROOT="${FE01_CODE_ROOT:-$A01_CODE_ROOT}"\n'
    path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:
        f.write(text)


def make_tasks(cohort,block):
    chunks=[cohort[i:i+block] for i in range(0,len(cohort),block)]
    tasks={s:[dict(index=0)] for s in ('inventory','compare_existing','analyze','pack')}
    for stage in ('covariates','crossing_truth'):
        tasks[stage]=[dict(index=i,events=c) for i,c in enumerate(chunks)]
    for stage in ('model_smoke','replay_pilot','latency'):
        tasks[stage]=[dict(index=i,family=f) for i,f in enumerate(EPOCHS)]
    for stage in ('replay_stations','origin_replay','long_extension'):
        rows=[]
        for family in EPOCHS:
            for geometry in ('normal','random'):
                for chunk in chunks:
                    rows.append(dict(index=len(rows),family=family,geometry=geometry,events=chunk))
        tasks[stage]=rows
    for geometry in ('normal','random'):
        tasks['replay_grid_'+geometry]=[]
        for family in EPOCHS:
            for chunk in chunks:
                tasks['replay_grid_'+geometry].append(dict(index=len(tasks['replay_grid_'+geometry]),family=family,geometry=geometry,events=chunk))
        tasks['render_'+geometry]=[dict(index=i,geometry=geometry,events=c) for i,c in enumerate(chunks)]
    return tasks


def scientific_environment():
    keys=('DATA_ROOT','FE01_SPLIT_MANIFEST','FE01_WEIGHTS_ROOT','FE01_CODE_ROOT',
          'ON_EVAL_INHERITED_CODE_ROOT','ON_EVAL_CHECKPOINT_MAP')
    result={k:os.environ.get(k) for k in keys}
    mapping=result.get('ON_EVAL_CHECKPOINT_MAP')
    result['checkpoint_map_sha256']=sha256(mapping) if mapping else None
    return result


def prepare(run_id,env_file):
    c=contract()
    source=source_identity()
    local_core=verify_frozen_core()
    session=session_path(run_id)
    if session.exists():
        verify_plan(session)
        print('PREPARED_ALREADY '+str(session))
        return
    old=Path(os.environ['ON_EVAL_SOURCE_SESSION']).resolve()
    require((old/'plan.json').is_file(),'Original private multi-root plan missing')
    # Parse only metadata; no HDF waveform/model/Torch import on the login node.
    from .sources import pins,verify_export_seals
    verify_export_seals(old)
    p=next(x for x in pins() if x['family']=='TEAM' and x['checkpoint_policy']=='selected')
    fields=['dataset_id','event_id','station_id','latitude','longitude','input_count','magnitude']
    import pandas as pd
    from .io import csv_read
    f=csv_read(old/p['source'],usecols=fields)
    cohort=sorted(f[['dataset_id','event_id']].drop_duplicates().values.tolist())
    require(len(cohort)==1310,'Expected original locked validation cohort')
    by_event=f.groupby(['dataset_id','event_id']).agg(queries=('station_id','nunique'),min_K=('input_count','min'),max_K=('input_count','max'),magnitude=('magnitude','first')).reset_index()
    by_event['selection_hash']=[fingerprint(['v4_pilot',r.dataset_id,r.event_id]) for r in by_event.itertuples()]
    from .query_grid import make_grid
    layouts=[]
    for identity,part in f.groupby(['dataset_id','event_id']):
        coords=part[['station_id','latitude','longitude']].drop_duplicates('station_id')
        points=coords[['latitude','longitude']].to_numpy()
        import numpy as np
        layout=make_grid(np.column_stack([points,np.full(len(points),np.nan)]),c['grid'])
        layouts.append(dict(dataset_id=identity[0],event_id=identity[1],grid_points=len(layout['points']),
                            spacing_km=layout['spacing_km'],third_coordinate='NOT_HYDRATED_LAYOUT_ONLY'))
    # Metadata strata include sparse/dense K and query count quantiles; hash breaks ties.
    chosen=[]
    pools=[by_event[by_event.min_K.eq(1)],by_event[by_event.max_K.gt(1)],
           by_event.nsmallest(max(1,len(by_event)//4),'queries'),by_event.nlargest(max(1,len(by_event)//4),'queries'),
           by_event.nsmallest(max(1,len(by_event)//4),'magnitude'),by_event.nlargest(max(1,len(by_event)//4),'magnitude')]
    for pool in pools:
        for r in pool.sort_values('selection_hash').itertuples():
            pair=[r.dataset_id,r.event_id]
            if pair not in chosen:
                chosen.append(pair)
                break
    block=int(os.environ.get('ON_EVAL_BLOCK_EVENTS','32'))
    require(block>0,'Positive event block size required')
    tasks=make_tasks(cohort,block)
    session.mkdir(parents=True,exist_ok=False)
    (session/'slurm').mkdir()
    (session/'jobs').mkdir()
    write_json(session/'analysis_contract.json',c)
    write_json(session/'source_identity.json',source)
    write_json(session/'local_core_verification.json',local_core)
    csv_write(pd.DataFrame(layouts),session/'static_grid_layout.csv')
    # This private absolute plan never enters source or public review bundles unchanged.
    plan=dict(run_id=run_id,source_identity=source,contract_sha256=fingerprint(c),
              scientific_environment=scientific_environment(),
              source_session=str(old),source_plan_sha256=sha256(old/'plan.json'),env_file=str(Path(env_file).resolve()),
              release_root=str(ROOT),cohort=cohort,pilot_events=chosen,block_events=block,tasks=tasks,
              pilot_selection_status='PROVISIONAL; six actual pilot events and twelve report events frozen from CPU physical metadata before first model forward',
              station_frame_upper_bound=len(cohort)*4*20*2,scope='validation seed42 ON selected; no training/test',
              query_count_distribution=by_event.queries.describe().to_dict(),
              grid_point_count_distribution=pd.DataFrame(layouts).grid_points.describe().to_dict(),
              per_model_worker_loads={stage:len(rows) for stage,rows in tasks.items() if stage in DCU or stage.startswith('replay_grid_')},
              expected_files=dict(station_event_stores=len(cohort)*4*2,default_normal_grid_event_stores=len(cohort)*4,
                                  normal_animation_events=len(cohort),normal_low_resolution_frames=len(cohort)*20,
                                  metadata='grid third coordinate hydrated before forward; no grid truth'),
              formal_hpc_status='NOT_SUBMITTED')
    write_json(session/'plan.json',plan)
    write_json(session/'plan_pin.json',dict(sha256=sha256(session/'plan.json')))
    print(json.dumps(dict(session=str(session),cohort_events=len(cohort),station_frame_upper_bound=plan['station_frame_upper_bound'],
                          tasks={k:len(v) for k,v in tasks.items()},queries=plan['query_count_distribution'],
                          grid_points=plan['grid_point_count_distribution'],expected_files=plan['expected_files'],
                          provisional_pilot_events=chosen,pilot_selection_status=plan['pilot_selection_status'],model_loads=plan['per_model_worker_loads'],
                          expected_grid='static grid counts/support frozen by CPU covariates before DCU; no predictions used'),ensure_ascii=False,indent=2))


def verify_plan(session):
    session=Path(session)
    require(sha256(session/'plan.json')==read_json(session/'plan_pin.json')['sha256'],'Run plan changed')
    plan=read_json(session/'plan.json')
    require(source_identity()==plan['source_identity'],'New source differs; use a new release and run ID')
    require(scientific_environment()==plan['scientific_environment'],'Scientific input environment changed; preserve this run, prepare a new run ID')
    require(fingerprint(read_json(session/'analysis_contract.json'))==plan['contract_sha256'],'Frozen contract changed')
    require(sha256(Path(plan['source_session'])/'plan.json')==plan['source_plan_sha256'],'Old private source plan changed')
    return plan


def stage_identity(session,key,index):
    p=verify_plan(session)
    dependencies={}
    for stage in dependency_keys(key):
        for task in p['tasks'][stage]:
            path=Path(session)/'stages'/stage/str(task['index'])/'SEAL.json'
            dependencies[stage+'/'+str(task['index'])]=sha256(path)
    return dict(execution_sha256=p['source_identity']['execution_sha256'],contract_sha256=p['contract_sha256'],
                plan_sha256=sha256(Path(session)/'plan.json'),stage=key,task=index,dependency_seals_sha256=dependencies)


def completed(session,key,index):
    final=Path(session)/'stages'/key/str(index)
    if not final.exists():
        return False
    verify_seal(final,stage_identity(session,key,index))
    return True


def require_stage(session,key):
    p=verify_plan(session)
    for row in p['tasks'][key]:
        require(completed(session,key,row['index']),'Required stage incomplete: '+key+'/'+str(row['index']))


def dependency_keys(key):
    base=key.rsplit('_',1)[0] if key.startswith(('replay_grid_','render_')) else key
    if base=='compare_existing':
        return []  # old prediction seals are independent of checkpoint inventory availability
    if base in ('covariates','crossing_truth'):
        return []  # HDF identity is verified by this stage itself, not a model gate
    needed={'model_smoke':['inventory','covariates'], 'replay_pilot':['model_smoke','covariates'],
            'replay_stations':['replay_pilot'], 'replay_grid':['replay_pilot','covariates'],
            'latency':['replay_pilot'], 'origin_replay':['model_smoke','covariates'],
            'long_extension':['model_smoke','covariates'], 'analyze':['replay_stations','crossing_truth','replay_grid_normal','covariates'],
            'render':['replay_stations','analyze'], 'pack':['compare_existing','analyze','render_normal','latency']}
    if base=='render':
        return needed['render']+['replay_grid_'+key.rsplit('_',1)[1]]
    if base=='pack' and os.environ.get('ON_EVAL_PACK_PARTIAL')=='1':
        require(bool(os.environ.get('ON_EVAL_PACK_PARTIAL_REASON')),'Explicit partial pack reason is required')
        return []
    return needed.get(base,[])


def prerequisites(session,key):
    for dep in dependency_keys(key):
        require_stage(session,dep)


def scheduler_states(session):
    records=[]
    for p in sorted((Path(session)/'jobs').glob('*.json')):
        records.append(read_json(p))
    state={}
    for r in records:
        job=str(r['job_id'])
        raw=[]
        try:
            q=subprocess.run(['squeue','-h','-r','-j',job,'-o','%i|%T'],capture_output=True,text=True,check=False,timeout=20)
            raw+=q.stdout.splitlines() if q.returncode==0 else []
            a=subprocess.run(['sacct','-n','-P','-j',job,'--format=JobID,State,ExitCode'],capture_output=True,text=True,check=False,timeout=20)
            raw+=a.stdout.splitlines() if a.returncode==0 else []
        except (OSError,subprocess.TimeoutExpired):
            pass
        found={}
        for line in raw:
            cols=line.strip().split('|')
            if len(cols)<2:
                continue
            jid,st=cols[:2]
            m=re.fullmatch(re.escape(job)+r'_(\d+)',jid)
            if m:
                found[int(m.group(1))]=st.split()[0].rstrip('+')
        for i in r['indices']:
            state[(r['stage'],i)]=dict(job_id=job,status=found.get(i,'UNKNOWN'))
    return state


def status(session,print_rows=True):
    p=verify_plan(session)
    queue=scheduler_states(session)
    rows=[]
    for key,tasks in p['tasks'].items():
        for task in tasks:
            i=task['index']
            try:
                done=completed(session,key,i)
                s='COMPLETE_SEAL_VERIFIED' if done else queue.get((key,i),{}).get('status','NOT_SUBMITTED')
            except (ValueError,OSError) as e:
                s='INVALID_SEAL:'+str(e)
            rows.append(dict(stage=key,index=i,status=s,job_id=queue.get((key,i),{}).get('job_id',''),
                             log_pattern=str(Path(session)/'slurm'/(key+'_%A_%a.err'))))
    if print_rows:
        from collections import Counter
        for key in p['tasks']:
            print(key,json.dumps(dict(Counter(r['status'] for r in rows if r['stage']==key)),ensure_ascii=False))
            for row in [r for r in rows if r['stage']==key and r['status'] not in ('NOT_SUBMITTED','COMPLETE_SEAL_VERIFIED','RUNNING','PENDING')]:
                task=p['tasks'][key][row['index']]
                label=' / '.join(str(task[k]) for k in ('family','geometry') if k in task)
                if 'events' in task:
                    label+=' / '+str(len(task['events']))+' events'
                print('  task='+str(row['index'])+' '+label+' status='+row['status']+' job='+row['job_id'])
                print('  err='+str(Path(session)/'slurm'/(key+'_'+row['job_id']+'_'+str(row['index'])+'.err')))
                print('  attempts='+str(Path(session)/'attempts'/key/(str(row['index'])+'_*')))
        print('结果目录='+str(session))
        print('日志目录='+str(Path(session)/'slurm'))
        print('UNKNOWN不是完成，也不会自动重提。')
    return rows


def submit(session,stage,geometry='normal',resume=False):
    require(stage in STAGES,'Unknown stage')
    key=stage_key(stage,geometry)
    p=verify_plan(session)
    prerequisites(session,key)
    with (Path(session)/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        states={(r['stage'],r['index']):r for r in status(session,False)}
        pending=[]
        allowed={'NOT_SUBMITTED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL'}
        for task in p['tasks'][key]:
            i=task['index']
            s=states[(key,i)]['status']
            if s=='COMPLETE_SEAL_VERIFIED':
                continue
            if s in {'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','RESIZING'}:
                continue
            require(s in allowed,'Refusing duplicate/unknown/invalid task '+key+'/'+str(i)+': '+s+'; check logs and scheduler')
            if not resume and s!='NOT_SUBMITTED':
                raise ValueError('Use resume_failed.sh for prior failed task '+str(i))
            pending.append(i)
        if not pending:
            print('NO_SUBMISSION: no pending tasks; successful and running tasks preserved')
            return
        resource='inference_array' if stage in DCU else 'render_array' if stage=='render' else 'cpu_stage'
        args=['sbatch','--parsable','--export=ALL','--chdir='+str(ROOT),'--job-name=onv4_'+key,'--nodes=1','--ntasks=1',
              '--array='+','.join(map(str,pending))+'%'+os.environ.get('ON_EVAL_MAX_CONCURRENT','8'),
              '--cpus-per-task='+os.environ.get('ON_EVAL_CPUS','4'),
              '--mem='+os.environ.get('ON_EVAL_MEM','96G' if stage in {'inventory','compare_existing'} else '64G'),
              '--time='+os.environ.get('ON_EVAL_TIME','24:00:00'),
              '--output='+str(Path(session)/'slurm'/(key+'_%A_%a.out')),
              '--error='+str(Path(session)/'slurm'/(key+'_%A_%a.err'))]
        part=os.environ.get('ON_EVAL_PARTITION','') if stage in DCU else os.environ.get('ON_EVAL_CPU_PARTITION',os.environ.get('ON_EVAL_PARTITION',''))
        for option,value in [('partition',part),('account',os.environ.get('ON_EVAL_ACCOUNT','')),('exclude',os.environ.get('ON_EVAL_EXCLUDE_NODES',''))]:
            if value:
                args.append('--'+option+'='+value)
        if stage in DCU:
            args.append('--gres='+os.environ['ON_EVAL_GRES'])
        args += [str(ROOT/'scripts/fe01_a01_compare/jobs'/(resource+'.sbatch')),str(session),key,p['env_file']]
        # This is the only place that invokes sbatch; CLI requires explicit submit/resume.
        r=subprocess.run(args,capture_output=True,text=True,check=True)
        job=r.stdout.strip().split(';')[0]
        require(re.fullmatch(r'\d+',job),'Unrecognized sbatch JobID: '+r.stdout)
        write_json(Path(session)/'jobs'/(dt.datetime.now().strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex+'.json'),
                   dict(stage=key,indices=pending,job_id=job,command=args,status='SUBMITTED_NOT_COMPLETED'))
        print('SUBMITTED stage='+key+' job='+job+' tasks='+str(len(pending)))
        print('Logs='+str(Path(session)/'slurm'))
        print('Results='+str(Path(session)/'stages'/key))
