"""Freeze six pilot and twelve report events using physical metadata, never errors."""
import numpy as np
import pandas as pd
from pathlib import Path
from .contracts import EVENT
from .io import attempt,csv_read,fingerprint,read_json,sha256,verify_seal,write_json


def pick(pools,all_events,count):
    result=[]
    for pool in pools:
        for r in pool.sort_values('selection_hash').itertuples():
            key=[r.dataset_id,r.event_id]
            if key not in result:
                result.append(key)
                break
        if len(result)==count:
            return result
    for r in all_events.sort_values('selection_hash').itertuples():
        key=[r.dataset_id,r.event_id]
        if key not in result:
            result.append(key)
        if len(result)==count:
            return result
    return result


def freeze(session):
    from .stages import all_stage_directories
    dirs=all_stage_directories(session,'covariates')
    identity={'covariate_seals_sha256':{d.name:sha256(d/'SEAL.json') for d in dirs}}
    key=fingerprint(identity)
    root=Path(session)/'inputs'
    with attempt(root,'metadata_selection',key,identity,blocking=True) as out:
        if out is not None:
            events=pd.concat([csv_read(d/'event_covariates.csv') for d in dirs],ignore_index=True)
            decisions=pd.concat([csv_read(d/'decision_manifest.csv.gz') for d in dirs],ignore_index=True)
            quality=decisions.groupby(EVENT).agg(min_K=('input_count','min'),max_K=('input_count','max'),
                missing_max=('physical_missing_fraction','max'),snr_known_fraction=('snr_available_fraction','mean')).reset_index()
            events=events.merge(quality,on=EVENT,validate='one_to_one')
            events['selection_hash']=[fingerprint(['v4_metadata',*x]) for x in events[EVENT].values.tolist()]
            pilot=pick([events[events.min_K.eq(1)],events[events.max_K.gt(1)],
                        events.nsmallest(max(1,len(events)//4),'record_samples_min'),
                        events.nlargest(max(1,len(events)//4),'record_samples_max'),
                        events.nsmallest(max(1,len(events)//4),'query_count'),
                        events.nlargest(max(1,len(events)//4),'query_count')],events,6)
            report=[]
            for field in ('magnitude','median_epicentral_distance_km','min_K','missing_max','snr_known_fraction','query_count'):
                known=events[events[field].notna()]
                report.extend([known.nsmallest(max(1,len(known)//4),field),known.nlargest(max(1,len(known)//4),field)])
            cases=pick(report,events,12)
            write_json(out/'selection.json',dict(pilot_events=pilot,report_events=cases,
                pilot_rule='K1/multi-K, short/long actual stored records, small/large query panels; metadata/hash ties',
                report_rule='magnitude/epicentral distance/K/missingness/SNR availability/query count extremes; metadata/hash ties',
                selection_uses_predictions=False,cohort_events=len(events),source_identity=identity))
            from .io import csv_write
            csv_write(events,out/'selection_metadata.csv')
    directory=root/'stages/metadata_selection'/key
    verify_seal(directory,identity)
    return dict(**read_json(directory/'selection.json'),selection_seal_sha256=sha256(directory/'SEAL.json'))
