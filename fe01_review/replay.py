"""Finite three-case replay with complete frame latency and real station labels."""
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fe01.data import split_catalog,read_event
from fe01.spatial import coverage
from .requests import read_frame,DECISION_KEYS
from .provenance import require,read_json,write_json,sha256,provenance,seal,evaluation_identity
from .runners import (load_runner,relocate,data_gate,catalog_row,predict_decision,decision_token,sync,buffer_pin)


def replay(entry,plan,request_root,source_root,output,verification,device='cuda'):
    require(entry.get('seed')==42 or entry['kind']!='fe01','Replay only fixed seed42 FE01 systems')
    gate=read_json(Path(verification)/'verification.json')
    require(gate['status']=='PASS' and gate['checkpoint_sha256']==sha256(entry['checkpoint']),'Replay verification gate absent/stale')
    require(gate['evaluation_module_sha']==evaluation_identity()['evaluation_module_sha'],'Replay source differs from verified source')
    sync(device);begin=time.perf_counter();runner,cfg,identity=load_runner(entry,device);sync(device)
    cold=time.perf_counter()-begin
    base=Path(source_root)/'formal__team_original_scratch__seed42'
    common=relocate(read_json(base/'resolved_config.json'));data_gate(common,read_json(base/'protocol.lock.json'))
    catalog=split_catalog(common,'val').set_index(['dataset_id','event_id'])
    reqpath=Path(request_root)/'replay_requests.csv.gz';requests=read_frame(reqpath)
    grids={(g['dataset_id'],g['event_id']):g for g in read_json(Path(request_root)/'grid_manifest.json')}
    frames=output/'frames';frames.mkdir();latencies=[];station_frames=[];grid_frames=[];standalone_checks=[];before=buffer_pin(runner.model)
    native=cfg['native_n_samples'] if entry['kind']=='fe01' else 10000
    for key,group in requests.groupby(DECISION_KEYS+['input_mode']):
        sync(device);start=time.perf_counter();io_start=time.perf_counter()
        event=read_event(catalog_row(catalog,key[0],key[1]),common,float(key[2]));read_seconds=time.perf_counter()-io_start
        points=np.asarray(grids[key[:2]]['points'],dtype=np.float32)
        with runner.cutoff_cache(decision_token(group.iloc[0])):
            real,real_time=predict_decision(runner,cfg,event,group,native,reuse_active=True)
            grid,grid_time=predict_decision(runner,cfg,event,group,native,grid_points=points,reuse_active=True)
        require(real_time['status']=='supported' and grid_time['status']=='supported','Replay unsupported frame cannot be dropped')
        ids=json.loads(group.input_ids.iloc[0]);idx={str(s):i for i,s in enumerate(event['ids'])}
        # 100 km mask is a disclosed visualization support convention, not an accuracy threshold.
        grid['within_100km_input_support']=coverage(points,event['coords'][[idx[s] for s in ids]],100)
        grid['display_support_rule']='distance to received input <=100 km; sparse/outside flagged, predictions retained'
        grid['run_id']=real['run_id']=entry['run_id'];grid['model_family']=real['model_family']=entry.get('model_family',entry['run_id'])
        stem=f'{key[0]}_{key[1]}_{key[2]:02.0f}_{key[3]}_{key[4]}'
        io_start=time.perf_counter();real.to_csv(frames/(stem+'_stations.csv.gz'),index=False);grid.to_csv(frames/(stem+'_grid.csv.gz'),index=False)
        write_seconds=time.perf_counter()-io_start;sync(device);elapsed=time.perf_counter()-start
        record=dict(zip(DECISION_KEYS+['input_mode'],key),run_id=entry['run_id'],device=str(device),station_targets=len(real),grid_points=len(grid),
            read_seconds=read_seconds,write_seconds=write_seconds,end_to_end_seconds=elapsed,update_period_seconds=1,over_update_period=elapsed>1,
            encoder_seconds=real_time['encoder_seconds']+grid_time['encoder_seconds'],preprocess_seconds=real_time['preprocess_seconds']+grid_time['preprocess_seconds'],
            forward_seconds=real_time['forward_seconds']+grid_time['forward_seconds'],distribution_seconds=real_time['distribution_seconds']+grid_time['distribution_seconds'])
        latencies.append(record);station_frames.append(real);grid_frames.append(grid)
        if float(key[2]) in (1,10,20):
            standalone,_=predict_decision(runner,cfg,event,group,native,use_cache=False)
            require(np.allclose(real.prediction,standalone.prediction,atol=1e-5,rtol=1e-5),'Replay differs from independent fresh cutoff forward')
            standalone_checks.append(dict(zip(DECISION_KEYS+['input_mode'],key),status='PASS',maximum_prediction_delta=float(np.max(np.abs(real.prediction-standalone.prediction)))))
    require(buffer_pin(runner.model)==before,'Replay changed frozen model buffers')
    station=pd.concat(station_frames,ignore_index=True);grid=pd.concat(grid_frames,ignore_index=True)
    reference_path=Path(request_root).parent/'reference/train_only_reference.json'
    if reference_path.is_file():
        from fe01.spatial import reference_design
        reference=read_json(reference_path)
        for data in (station,grid):
            data['no_site_oracle_reference']=reference_design(data)@np.asarray(reference['no_site_coefficients'])
            data['prediction_minus_no_site']=data.prediction-data.no_site_oracle_reference
    station.to_csv(output/'station_predictions.csv.gz',index=False);grid.to_csv(output/'grid_predictions.csv.gz',index=False)
    latency=pd.DataFrame(latencies);latency.to_csv(output/'replay_latency.csv',index=False)
    write_json(output/'standalone_snapshot_checks.json',standalone_checks)
    write_json(output/'latency_summary.json',dict(cold_start_seconds=cold,frames=len(latency),p50=float(latency.end_to_end_seconds.median()),
        p95=float(latency.end_to_end_seconds.quantile(.95)),maximum=float(latency.end_to_end_seconds.max()),over_1s_fraction=float(latency.over_update_period.mean()),
        synchronized=True,frame_boundary='read prefix to all station/grid distribution results and frame files written; aggregate/render separate',
        replay='ideal trigger causal replay; actual communication latency unmeasured',upstream_offline_causality='uncertified'))
    render_start=time.perf_counter();render_cases(station,grid,grids,output)
    write_json(output/'render_latency.json',dict(render_seconds=time.perf_counter()-render_start,inside_frame_latency=False,generated_animation=False))
    from .diagnostics import fields
    metrics=[]
    for mode,g in station.groupby('input_mode'):
        # Every mode has a fixed query universe excluding S0; natural inputs can change its roles.
        metrics.append(fields(g).assign(input_mode=mode))
    pd.concat(metrics,ignore_index=True).to_csv(output/'case_field_metrics.csv',index=False)
    station.groupby(['dataset_id','event_id','elapsed_time','geometry_protocol','input_mode','target_role']).size().rename('targets').to_csv(output/'case_role_counts.csv')
    from fe01.metrics import summarize
    natural=station[station.input_mode=='natural'];fixed=station[station.input_mode=='fixed_s0']
    keys=DECISION_KEYS+['station_id'];paired=natural[keys+['target_role']].merge(fixed[keys+['target_role']],on=keys,suffixes=('_natural','_fixed'),how='outer',indicator=True,validate='one_to_one')
    require(paired._merge.eq('both').all(),'Case fixed query universes differ')
    common=paired[(paired.target_role_natural!='observed_input')&(paired.target_role_fixed!='observed_input')][keys]
    records=[]
    for mode,data in [('natural',natural),('fixed_s0',fixed)]:
        chosen=data.merge(common,on=keys,validate='one_to_one')
        for key,g in chosen.groupby(DECISION_KEYS):records.append(dict(zip(DECISION_KEYS,key),input_mode=mode,panel='both_paths_noninput',**summarize(g)))
    pd.DataFrame(records).to_csv(output/'case_common_noninput_metrics.csv',index=False)
    write_json(output/'provenance.json',provenance(run_id=entry['run_id'],checkpoint_epoch=entry['epoch'],checkpoint_sha256=identity['checkpoint_sha256'],
        request_manifest_sha256=sha256(reqpath),case_manifest_sha256=sha256(Path(plan)/'case_manifest.json'),grid_manifest_sha256=sha256(Path(request_root)/'grid_manifest.json'),
        verification_sha256=sha256(Path(verification)/'verification.json'),parent_lock_sha256=identity.get('parent_lock_sha256','not applicable: legacy original config'),
        common_request_lock_sha256=sha256(base/'protocol.lock.json'),
        preprocessor_id=runner.preprocessor_id,grid_truth='none',grid_elevation='0m demonstration only',training_source_sha=identity['training_source'],
        oracle_reference_sha256=sha256(reference_path) if reference_path.is_file() else None))
    seal(output)


def render_cases(station,grid,grids,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figs=output/'figures';figs.mkdir()
    for key,g in grid[grid.elapsed_time.isin([1,5,20])].groupby(DECISION_KEYS+['input_mode']):
        r=station[(station.dataset_id==key[0])&(station.event_id==key[1])&(station.elapsed_time==key[2])&(station.geometry_protocol==key[3])&(station.input_mode==key[4])]
        has_ref='prediction_minus_no_site' in g
        fig,axes=plt.subplots(1,5 if has_ref else 4,figsize=(17 if has_ref else 14,3.5),layout='constrained')
        panels=[(axes[0],r,'truth',-4,1),(axes[1],g,'prediction',-4,1),(axes[2],r.assign(error=r.prediction-r.truth),'error',-1,1),(axes[3],g,'width95',0,3)]
        if has_ref:panels.append((axes[4],g,'prediction_minus_no_site',-1,1))
        for ax,part,col,vmin,vmax in panels:
            color=ax.scatter(part.longitude,part.latitude,c=part[col],s=9,vmin=vmin,vmax=vmax,cmap='coolwarm' if col=='error' else 'viridis')
            fig.colorbar(color,ax=ax,label=col+' (dex)');bounds=grids[key[:2]]['bounds'];ax.set(xlim=bounds[2:4],ylim=bounds[:2],xlabel='Longitude',ylabel='Latitude')
            coords=np.asarray(json.loads(r.input_coords.iloc[0]));ax.scatter(coords[:,1],coords[:,0],marker='^',s=15,facecolors='none',edgecolors='red')
        outside=g[~g.within_100km_input_support];axes[1].scatter(outside.longitude,outside.latitude,marker='x',s=3,color='gray')
        axes[1].scatter(r.longitude,r.latitude,s=12,facecolors='none',edgecolors='white')
        fig.suptitle(f'{key[1]} {key[2]}s {key[3]} {key[4]}: final PGA, ideal trigger')
        fig.savefig(figs/f'{key[0]}_{key[1]}_{key[2]}_{key[3]}_{key[4]}.png',dpi=130);plt.close(fig)
    (figs/'CAPTIONS.md').write_text('All panels predict final log10 PGA (m/s²), not the currently observed peak. Triangles: actual received input stations; circles: fixed real queries; gray crosses: grid farther than100km from any input. Same case bounds, 10km grid and fixed shared scales across systems/times. Grid elevation0m is a demonstration assumption; no grid ground truth. Error scored at real stations only. Width95 is mixture quantile width. Sources: station_predictions.csv.gz, grid_predictions.csv.gz and frozen grid/case manifests. Frames1/5/20 illustrated; all1–20 saved. Render outside synchronized end-to-end frame timing. No animation or geology claim.\n')
