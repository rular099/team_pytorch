"""Post-render shared FIFO scenario from actual full-map compute and render timings."""
from pathlib import Path
import numpy as np
import pandas as pd
from .clocks import fifo
from .contracts import EPOCHS
from .io import csv_read,csv_write,read_json,write_json


def analyze_rendered(session,out):
    from .stages import store_index,read_event_stores,all_stage_directories,stream_csv
    from .map_readiness import evaluate,population_summary,pair_events
    from .shadow import evaluate as shadow
    from .orchestration import verify_plan
    from .contracts import contract
    plan=verify_plan(session)
    index=store_index(session,'replay_grid_normal')
    truths=pd.concat([csv_read(d/'crossing_truth.csv.gz') for d in all_stage_directories(session,'crossing_truth')],ignore_index=True)
    render={}
    for d in all_stage_directories(session,'render_normal'):
        for item in read_json(d/'visuals_index.json'):
            frame=d/Path(item['href']).parent/'render_latency.csv'
            render[(item['dataset_id'],item['event_id'])]=csv_read(frame)
    queue=[]
    readiness=[]
    warnings=[]
    out.mkdir()
    for event in plan['cohort']:
        frame=read_event_stores(index,event,'normal')
        render_time=render[tuple(event)].set_index('elapsed_time').render_publish_seconds.to_dict()
        for family in EPOCHS:
            mask=frame.family.eq(family)
            times=frame[mask][['elapsed_time','numeric_map_ready_seconds']].drop_duplicates().sort_values('elapsed_time')
            previous=-np.inf
            cutoffs=[]
            service=[]
            for row in times.itertuples():
                if pd.isna(row.numeric_map_ready_seconds):
                    continue
                start=max(row.elapsed_time,previous)
                service.append(max(0.,row.numeric_map_ready_seconds-start))
                previous=row.numeric_map_ready_seconds
                cutoffs.append(row.elapsed_time)
            scenario=fifo(cutoffs,service,[render_time.get(t,0.) for t in cutoffs])
            ready={r['cutoff_seconds']:r['rendered_ready_seconds'] for r in scenario}
            frame.loc[mask,'rendered_ready_seconds']=frame.loc[mask,'elapsed_time'].map(ready)
            frame.loc[mask,'numeric_ready_seconds']=frame.loc[mask,'numeric_map_ready_seconds']
            queue.extend([dict(dataset_id=event[0],event_id=event[1],family=family,**r) for r in scenario])
        _,tt=evaluate(frame)
        readiness.append(tt[tt.scene.eq('measured_rendered_map_replay')])
        tr=truths[truths.dataset_id.eq(event[0])&truths.event_id.eq(event[1])]
        outcomes,summary=shadow(frame,tr,contract()['shadow'])
        stream_csv(outcomes[outcomes.scene.eq('measured_rendered_map_replay')],out/'shadow_station_outcomes_rendered.csv.gz')
        warnings.append(summary[summary.scene.eq('measured_rendered_map_replay')])
    t=pd.concat(readiness,ignore_index=True)
    csv_write(t,out/'event_time_to_usable_map_rendered.csv')
    csv_write(population_summary(t),out/'ttum_rendered_support.csv')
    csv_write(pair_events(t),out/'ttum_rendered_pairwise_events.csv')
    csv_write(pd.concat(warnings,ignore_index=True),out/'shadow_rendered_event_summary.csv')
    csv_write(pd.DataFrame(queue),out/'queue_replay.csv')
    write_json(out/'scenario_contract.json',dict(
        status='MEASURED_COMPUTE_AND_SHARED_COMPARISON_RENDER_SCENARIO',
        compute='actual per-event/model/grid complete-frame timings from replay_grid_normal',
        render='actual four-model comparison frame render+PNG publication time, uniformly charged to each model scenario',
        queue='per-event sequential FIFO; compute + shared comparison rendering on one scenario server',
        communication_latency='UNKNOWN',not_deployment_measurement=True,
        numeric_vs_rendered='numeric-only queue and render-inclusive queue are separate explicit scenarios; no double counting',
        scope='normal only; random rendered scenario requires separately completed grid/random render stages'))
