"""Small, actual end-to-end numeric store -> shadow/TTUM -> dynamic artifacts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fe01.metrics import score_rows
from .contracts import EPOCHS,contract
from .io import csv_read,csv_write,fingerprint,seal,write_json
from .map_readiness import evaluate,population_summary,pair_events
from .shadow import evaluate as shadow
from .plotting import animate_event,shadow_map_html,build_index


def fixture():
    rows=[]
    for model_index,family in enumerate(EPOCHS):
        truth=np.array([-2.2,-1.7,-1.,-.3,.2,.5])
        for t in range(1,21):
            prediction=truth+(21-t)*.014+model_index*.01
            sigma=np.full((6,1),.2)
            scores=score_rows(truth,np.ones((6,1)),prediction[:,None],sigma)
            for i in range(6):
                r=dict(dataset_id='SYNTHETIC',event_id='SYNTHETIC_EVENT',station_id=f'S{i:02d}',family=family,
                       elapsed_time=t,geometry_protocol='normal',latitude=35+i//3*.1,longitude=139+i%3*.1,
                       elevation=.02,truth=float(truth[i]),prediction=float(prediction[i]),status='supported',
                       input_count=min(t//5+1,3),input_ids=json.dumps(['S00'] if t<10 else ['S00','S01']),
                       input_coords=json.dumps([[35,139,.02]] if t<10 else [[35,139,.02],[35,139.1,.02]]),
                       target_role='observed_input' if i==0 or i==1 and t>=10 else 'untriggered_noninput',
                       fixed_never_input=i>=2,first_p_minus_origin_seconds=4.,origin_elapsed_seconds=t+4,
                       checkpoint_epoch=EPOCHS[family],mdn_logits='[0.0]',mdn_mu=json.dumps([float(prediction[i])]),mdn_sigma='[0.2]')
                r.update({k:float(v[i]) for k,v in scores.items()})
                rows.append(r)
    truths=[]
    for i,y in enumerate(truth):
        for threshold in (-1.2,0.):
            truths.append(dict(dataset_id='SYNTHETIC',event_id='SYNTHETIC_EVENT',station_id=f'S{i:02d}',
                               threshold_logpga=threshold,truth=float(y),affected=bool(y>=threshold),
                               crossing_first_p_seconds=12+i if y>=threshold else None,status='exact_within_record' if y>=threshold else 'not_exceeded'))
    return pd.DataFrame(rows),pd.DataFrame(truths)


def run(output):
    out=Path(output)
    out.mkdir(parents=True,exist_ok=False)
    from .contracts import source_identity
    write_json(out/'source_identity.json',source_identity())
    frame,truth=fixture()
    csv_write(frame,out/'station_replay.csv.gz')
    csv_write(truth,out/'crossing_truth.csv')
    # Deliberately read back serialized bytes, as in real offline analysis.
    saved=csv_read(out/'station_replay.csv.gz')
    grid_rows=[]
    for model_index,family in enumerate(EPOCHS):
        for t in range(1,21):
            for i,(lat,lon) in enumerate((lat,lon) for lat in np.linspace(34.97,35.13,7) for lon in np.linspace(138.97,139.23,9)):
                grid_rows.append(dict(family=family,dataset_id='SYNTHETIC',event_id='SYNTHETIC_EVENT',geometry_protocol='normal',
                    elapsed_time=t,latitude=lat,longitude=lon,elevation=.02,grid_point_id=i,
                    prediction=-2.2+7*(lon-139)+8*(lat-35)+(21-t)*.014+model_index*.01,
                    width95=.784,status='supported',metadata_status='SYNTHETIC_declared_height_km'))
    csv_write(pd.DataFrame(grid_rows),out/'grid_replay.csv.gz')
    grid=csv_read(out/'grid_replay.csv.gz')
    m,t=evaluate(saved)
    outcomes,events=shadow(saved,truth,contract()['shadow'])
    csv_write(m,out/'map_readiness_by_frame.csv')
    csv_write(t,out/'event_time_to_usable_map.csv')
    csv_write(population_summary(t),out/'ttum_support.csv')
    csv_write(pair_events(t),out/'ttum_pairwise_events.csv')
    csv_write(outcomes,out/'shadow_station_outcomes.csv.gz')
    csv_write(events,out/'shadow_event_summary.csv')
    animate_event(saved,grid,out/'synthetic_event',t,synthetic=True)
    shadow_map_html(outcomes,out/'synthetic_event/shadow_station_map.html',synthetic=True)
    build_index([dict(href='synthetic_event/index.html',label='SYNTHETIC — NOT JAPAN')],out/'map_index.html',title='SYNTHETIC END-TO-END VERIFICATION')
    write_json(out/'verification.json',dict(status='PASS_SYNTHETIC_E2E',source='deterministic six stations, four synthetic model conditions,20frames',
               real_Japan_result=False,actual_model_forward=False,hpc_submitted=False,frames=20,
               outputs=['saved prediction storage','full research shadow grid with false alarms','fixed-panel TTUM budgets',
                        'dynamic frames/animation/HTML index','station shadow classifications']))
    seal(out,scope='SYNTHETIC_ONLY; not production forward or Japan validation')
