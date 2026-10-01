"""Every frame rebuilds the received raw prefix; no across-time feature cache."""
import copy
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .config import fingerprint, safe_output, sha256, write_json
from .data import prepare_sample, read_event, split_catalog
from .engine import evaluation_rows, inputs_to, load_checkpoint, synchronize
from .metrics import decode, grouped_metrics, score_rows
from .spatial import coverage, grid, reference_design


def select_cases(cfg,path,count=5):
    catalog=split_catalog(cfg,'val')
    if 'n_station_rows' in catalog:
        catalog=catalog.loc[catalog.n_station_rows>=5]
    if len(catalog)<3:
        raise ValueError('At least three representative validation events required')
    ordered=catalog.sort_values(['Magnitude','event_id'])
    indices=np.unique(np.linspace(0,len(ordered)-1,min(count,len(ordered))).astype(int))
    cases=ordered.iloc[indices][['dataset_id','event_id','Magnitude','Latitude','Longitude']].to_dict('records')
    if Path(path).exists():
        raise FileExistsError('Case selection already frozen')
    write_json(path,dict(split='val',rule='metadata magnitude quantiles; >=5 station rows; before predictions',
                         split_manifest_sha256=sha256(cfg['data']['split_manifest']),cases=cases))


def predict_grid(model,cfg,sample,points,device):
    records=[]
    for coordinate in points:
        inputs=inputs_to(sample,device)
        inputs[3].zero_();inputs[4].zero_()
        inputs[3][0,0]=torch.tensor(coordinate-np.array([37,140,0]),dtype=torch.float32,device=device)
        inputs[4][0,0]=True
        synchronize(device);begin=time.perf_counter()
        with torch.no_grad():
            outputs=model(*inputs)
        synchronize(device);duration=time.perf_counter()-begin
        mdn=outputs[model.output_layout.index('pga')][0,0].cpu().numpy()
        w,mu,sigma=decode(mdn,cfg['target_normalization'])
        if not np.isfinite(mdn).all():
            raise FloatingPointError('Grid query numerical failure')
        score=score_rows(np.asarray(0.),w,mu,sigma)
        records.append(dict(latitude=float(coordinate[0]),longitude=float(coordinate[1]),
            prediction=float(score['prediction']),q025=float(score['q025']),q975=float(score['q975']),
            predictive_sigma=float(score['predictive_sigma']),mdn_weights=json.dumps(w.tolist()),
            mdn_mu=json.dumps(mu.tolist()),mdn_sigma=json.dumps(sigma.tolist()),
            forward_seconds=duration,encoder_seconds=model._fe01_encoder_seconds,
            downstream_and_preprocessing_seconds=max(0,duration-model._fe01_encoder_seconds)))
    return pd.DataFrame(records)


def replay(cfg,checkpoint_path,case_manifest,destination,device='cpu',reference='first_p_pick',
           rolling=False,start=1,end=90,spacing_km=20,include_grid=True):
    cold_start=time.perf_counter()
    model,checkpoint=load_checkpoint(cfg,checkpoint_path,device)
    synchronize(device)
    cold_start_seconds=time.perf_counter()-cold_start
    if torch.device(device).type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    cfg=copy.deepcopy(cfg)
    if rolling:
        cfg['window']['protocol']='native_rolling_v2'
    destination=safe_output(cfg['output_root'],destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Replay output exists; choose a new destination')
    manifest=json.loads(Path(case_manifest).read_text())
    if manifest['split']!='val' or manifest['split_manifest_sha256']!=sha256(cfg['data']['split_manifest']):
        raise ValueError('Replay selection must be locked validation metadata')
    catalog=split_catalog(cfg,'val')
    destination.mkdir(parents=True,exist_ok=True)
    predictions=[];latencies=[];support=[];grids=[];consistency=[]
    for case in manifest['cases']:
        matched=catalog.loc[catalog.event_id.eq(str(case['event_id']))&catalog.dataset_id.eq(case['dataset_id'])]
        if len(matched)!=1:
            raise ValueError('Case identity is not unique')
        row=matched.iloc[0]
        query_points=None
        for elapsed in range(start,end+1):
            synchronize(device);frame_start=time.perf_counter()
            event=read_event(row,cfg,elapsed,reference)
            sample=prepare_sample(event,cfg,elapsed,'normal',replay=True)
            status=sample['info']
            support.append(dict(event_id=row.event_id,elapsed_time=elapsed,status=status['status'],reason=status['reason']))
            if sample['inputs'] is None:
                continue
            if query_points is None and include_grid:
                coords=event['coords']
                bounds=(float(coords[:,0].min()-.2),float(coords[:,0].max()+.2),
                        float(coords[:,1].min()-.2),float(coords[:,1].max()+.2))
                query_points=grid(bounds,spacing_km)
            encoding=downstream=0.
            # Cache scope ends before advancing T. Values are checked exactly.
            with model.same_cutoff_query_cache():
                records,info=evaluation_rows(model,cfg,row,elapsed,'normal',device,replay=True,reference=reference)
                predictions.extend(records)
                encoding=sum(r['encoder_seconds'] for r in records)
                downstream=sum(r['forward_seconds']-r['encoder_seconds'] for r in records)
                if include_grid:
                    table=predict_grid(model,cfg,sample,query_points,device)
                    table['event_id']=row.event_id;table['elapsed_time']=elapsed
                    table['time_reference']=reference;table['window_protocol']=cfg['window']['protocol']
                    table['absolute_decision_utc']=status['absolute_decision_utc']
                    input_coords=event['coords'][np.isin(event['ids'],status['input_ids'])]
                    table['coverage']=coverage(query_points,input_coords)
                    encoding+=float(table.encoder_seconds.sum())
                    downstream+=float(table.downstream_and_preprocessing_seconds.sum())
                    grids.append(table)
            synchronize(device);frame_seconds=time.perf_counter()-frame_start
            latencies.append(dict(event_id=row.event_id,elapsed_time=elapsed,frame_seconds=frame_seconds,
                grid_queries=len(query_points) if include_grid else 0,encoder_seconds=encoding,
                downstream_and_preprocessing_seconds=downstream,
                io_preparation_export_overhead_seconds=max(0,frame_seconds-encoding-downstream),
                input_count=status['input_count'],requested_decision_sample=status['requested_decision_sample'],
                current_sample=status['current_sample'],history_start_sample=status['history_start_sample'],
                history_end_sample=status['history_end_sample'],latest_received_sample=status['latest_received_sample'],
                absolute_decision_utc=status['absolute_decision_utc'],
                result_available_utc=(status['absolute_decision_utc']+frame_seconds if status['absolute_decision_utc'] is not None else None),
                communication_delay_seconds='unknown; ideal zero-delay replay',
                peak_device_memory_bytes=torch.cuda.max_memory_allocated() if torch.device(device).type=='cuda' else None))
            # Persist exact station/sample masks, compact and independently readable.
            mask_dir=destination/'input_masks'/row.event_id;mask_dir.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(mask_dir/f't{elapsed:03d}.npz',sample_mask=sample['inputs'][5].numpy(),
                station_valid=sample['inputs'][2].numpy(),input_ids=np.asarray(status['input_ids'],dtype=str),
                cutoff=np.asarray(status['cutout_exclusive']))
            # Correctness check excluded from measured operational latency.
            if elapsed in cfg['realtime']['fixed_times'] or elapsed==start:
                standalone,_=evaluation_rows(model,cfg,row,elapsed,'normal',device,replay=True,reference=reference)
                delta=max(abs(a['prediction']-b['prediction']) for a,b in zip(records,standalone))
                if len(records)!=len(standalone) or delta>1e-6: raise AssertionError('Replay differs from independent cutoff inference')
                consistency.append(dict(event_id=row.event_id,elapsed_time=elapsed,max_prediction_delta=delta))
    frame=pd.DataFrame(predictions);latency=pd.DataFrame(latencies)
    frame.to_csv(destination/'replay_predictions.csv.gz',index=False)
    latency.to_csv(destination/'replay_latency.csv',index=False)
    pd.DataFrame(support).to_csv(destination/'replay_support.csv',index=False)
    pd.DataFrame(consistency).to_csv(destination/'replay_consistency.csv',index=False)
    if len(frame):
        grouped_metrics(frame).to_csv(destination/'replay_metrics_by_time.csv',index=False)
    if grids:
        pd.concat(grids,ignore_index=True).to_csv(destination/'grid_predictions.csv.gz',index=False)
    write_json(destination/'replay_provenance.json',dict(checkpoint_epoch=checkpoint['epoch'],
        cold_model_load_seconds=cold_start_seconds,
        checkpoint_sha256=sha256(checkpoint_path),case_manifest_sha256=sha256(case_manifest),
        window_protocol=cfg['window']['protocol'],rolling_trained=cfg['window']['rolling_trained'],
        interpretation='rolling matched training' if rolling and cfg['window']['rolling_trained'] else
                       'rolling transfer/OOD diagnostic' if rolling else 'native cumulative prefix',
        replay='each frame is new causal inference for the same final PGA; never interpolated',
        coverage='heuristic <=100 km to a received input; not validated epistemic confidence',
        p50_frame_seconds=float(latency.frame_seconds.median()) if len(latency) else None,
        p95_frame_seconds=float(latency.frame_seconds.quantile(.95)) if len(latency) else None,
        supports_1hz_region_p95=bool(latency.frame_seconds.quantile(.95)<=1) if len(latency) else False))
