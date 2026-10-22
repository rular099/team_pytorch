"""Allocated-device measurements: complete station/map, cold/warm, raw samples."""
import time
import numpy as np
import pandas as pd
from .contracts import EVENT
from .io import csv_write,read_json,write_json,require
from .clocks import fifo


def benchmark(session,task,out,device='cuda',warmup=2,repeats=3):
    import torch
    from fe01_review.runners import sync
    from .replay import load_model,Forward,sidecars
    from .covariates import catalog,physical_event
    from .orchestration import verify_plan
    from .plotting import render_frame
    p=verify_plan(session)
    from .selection import freeze
    selection=freeze(session)
    sync(device)
    begin=time.perf_counter()
    model,cfg,identity=load_model(p['source_session'],task['family'],device)
    sync(device)
    cold=time.perf_counter()-begin
    forward=Forward(model,cfg,identity['checkpoint_sha256'],device)
    req,grids=sidecars(session)
    cat=catalog(cfg,pd.DataFrame(selection['pilot_events'],columns=EVENT))
    timings=[]
    for _,r in cat.iterrows():
        t=time.perf_counter()
        e=physical_event(r)
        io=time.perf_counter()-t
        for geometry in ('normal','random'):
            for cutoff in (1,10,20):
                group=req[req.dataset_id.eq(r.dataset_id)&req.event_id.eq(r.event_id)&req.geometry_protocol.eq(geometry)&req.elapsed_time.eq(cutoff)]
                if not len(group) or int(group.input_count.iloc[0])==0:
                    continue
                grid=grids[(r.dataset_id,r.event_id)]
                points=np.array([[v if v is not None else np.nan for v in q] for q in grid['points']],float)
                points=points[np.isfinite(points).all(1)]
                for scope in ('station_only','whole_map'):
                    for rep in range(warmup+repeats):
                        sync(device)
                        start=time.perf_counter()
                        with forward.cutoff(group.iloc[0]):
                            station,a=forward.predict(e,group)
                            grid_out,b=forward.predict(e,group,points) if scope=='whole_map' and len(points) else (pd.DataFrame(),{k:0. for k in a})
                        sync(device)
                        service=time.perf_counter()-start
                        render_seconds=publish_seconds=0.
                        if rep>=warmup and scope=='whole_map':
                            station['family']=task['family']
                            station['checkpoint_epoch']=identity['epoch']
                            if len(grid_out):
                                grid_out['family']=task['family']
                                grid_out['checkpoint_epoch']=identity['epoch']
                            begin=time.perf_counter()
                            image=render_frame(station,grid_out,contract=None,title='PILOT LATENCY; station truth retrospective',dpi=70)
                            render_seconds=time.perf_counter()-begin
                            begin=time.perf_counter()
                            path=out/('latency_preview_'+str(len(timings))+'.png')
                            image.save(path)
                            publish_seconds=time.perf_counter()-begin
                        timings.append(dict(family=task['family'],dataset_id=r.dataset_id,event_id=r.event_id,geometry_protocol=geometry,
                            elapsed_time=cutoff,scope=scope,phase='warmup' if rep<warmup else 'warm_measured',repeat=rep-warmup,
                            K=int(group.input_count.iloc[0]),station_Q=len(group),grid_Q=len(points) if scope=='whole_map' else 0,
                            io_seconds=io,cold_load_seconds=cold,numeric_service_seconds=service,render_seconds=render_seconds,
                            publish_seconds=publish_seconds,synchronized=True,device=str(device),device_name=torch.cuda.get_device_name(device),
                            precision='float32; original model defaults',query_chunk=cfg['model_params']['n_pga_targets'],
                            communication_latency='UNKNOWN',**{k:a[k]+b[k] for k in a}))
    f=pd.DataFrame(timings)
    csv_write(f,out/'latency_frames.csv')
    warm=f[f.phase.eq('warm_measured')]
    summary=warm.groupby(['family','scope']).numeric_service_seconds.agg(['count','median',lambda x:x.quantile(.95),'max']).reset_index()
    summary.columns=['family','scope','count','p50_seconds','p95_seconds','maximum_seconds']
    csv_write(summary,out/'latency_summary.csv')
    write_json(out/'timing_contract.json',dict(warmup=warmup,repeats=repeats,cold_load_separate=True,
        cold_scope='checkpoint authentication/read + construct/load, after Torch import and device initialization',
        synchronization='before/after complete forward and readout',numeric='all station + grid MDN probabilities and quantiles',
        renderer='actual CPU render and PNG publish measured, separate from numerical readiness',
        archive='offline stage I/O, excluded from service',queue_policy='sequential_FIFO_shared_clocks.py',
        hardware_fairness='review recorded device_name/precision/K/Q before comparing; different allocations are not proof of identical devices',
        deployment='communication and true trigger latency UNKNOWN'))
