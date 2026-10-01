"""Evidence tables and figures, preserving common-domain pairing and failures."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import fingerprint, write_json
from .data import read_event, split_catalog
from .metrics import grouped_metrics, paired_bootstrap
from .spatial import distance_km, field_metrics, fit_reference, residuals, site_summary


def training_reference_rows(cfg):
    rows=[]
    for _,row in split_catalog(cfg,'train').iterrows():
        # Reader rejects other splits before data access. No held-out labels fit reference.
        event=read_event(row,cfg,1)
        for index in np.flatnonzero(np.isfinite(event['pga']) & event['query_allowed']):
            rows.append(dict(event_id=event['event_id'],station_id=event['ids'][index],split='train',
                truth=float(event['pga'][index]),latitude=float(event['coords'][index,0]),
                longitude=float(event['coords'][index,1]),magnitude=event['magnitude'],
                depth=float(event['event_location'][2]),event_latitude=float(event['event_location'][0]),
                event_longitude=float(event['event_location'][1])))
    return pd.DataFrame(rows)


def fit_train_reference(cfg,path):
    if Path(path).exists():
        raise FileExistsError('Reference exists; cannot refit held-out reference in place')
    write_json(path,fit_reference(training_reference_rows(cfg)))


def stratified_metrics(frame):
    result=[]
    frame=frame.copy()
    frame['distance_km']=distance_km(frame.latitude.to_numpy(),frame.longitude.to_numpy(),
                                   frame.event_latitude.to_numpy(),frame.event_longitude.to_numpy())
    frame['depth_km']=frame.depth
    bucket_specs={
        'input_count':([0,1,2,4,8,16,np.inf],['1','2','3-4','5-8','9-16','17+']),
        'magnitude':([-np.inf,4,5,6,np.inf],['<4','4-5','5-6','6+']),
        'truth':([-np.inf,-2,-1.2,0,np.inf],['<-2','-2..-1.2','-1.2..0','0+']),
        'distance_km':([0,25,50,100,200,np.inf],['0-25','25-50','50-100','100-200','200+']),
        'depth_km':([0,10,30,70,300,np.inf],['0-10','10-30','30-70','70-300','300+']),
        'padding_fraction':([-np.inf,.25,.5,.75,1.01],['<.25','.25-.5','.5-.75','.75+'])}
    for field,(bins,labels) in bucket_specs.items():
        group=frame.copy();group['bucket']=pd.cut(group[field],bins,labels=labels,include_lowest=True)
        for bucket,part in group.groupby('bucket',observed=True):
            scores=grouped_metrics(part)
            scores['stratum']=field;scores['bucket']=str(bucket)
            result.append(scores)
    return pd.concat(result,ignore_index=True) if result else pd.DataFrame()


def reliability(frame,bins=10):
    rows=[]
    for protocol,group in frame.groupby('geometry_protocol'):
        bucket=np.minimum((group.exceedance_probability*bins).astype(int),bins-1)
        for i in range(bins):
            sub=group.loc[bucket==i]
            if len(sub):
                rows.append(dict(geometry_protocol=protocol,probability_bin=i,targets=len(sub),
                    events=int(sub.event_id.nunique()),mean_probability=float(sub.exceedance_probability.mean()),
                    observed_exceedance=float((sub.truth>-1.2).mean())))
    return pd.DataFrame(rows)


def summarize_run(evaluation,output,reference_path=None):
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Analysis output exists')
    output.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(Path(evaluation)/'predictions.csv.gz',dtype={'event_id':str,'station_id':str})
    if frame.split.eq('train').any():
        raise ValueError('Evaluation contains train events')
    grouped_metrics(frame).to_csv(output/'metrics_by_time_target.csv',index=False)
    stratified_metrics(frame).to_csv(output/'metrics_stratified.csv',index=False)
    finite=frame.loc[frame.status.eq('supported') & np.isfinite(frame.prediction)]
    reliability(finite).to_csv(output/'reliability.csv',index=False)
    spatial=finite.loc[finite.target_role.ne('observed_input')]
    field_metrics(spatial).to_csv(output/'spatial_fields.csv',index=False)
    if reference_path:
        reference=json.loads(Path(reference_path).read_text())
        site_rows=residuals(spatial,reference)
        site_rows.to_csv(output/'site_residuals_raw.csv.gz',index=False)
        baselines=[]
        for name in ['reference','reference_with_train_site']:
            # Point-only oracle controls have no invented probabilistic scores.
            point=site_rows[['event_id','elapsed_time','geometry_protocol','target_role','status','truth']].copy()
            point['prediction']=site_rows[name]
            scores=grouped_metrics(point);scores['reference_model']=name
            baselines.append(scores)
        pd.concat(baselines).to_csv(output/'reference_point_metrics.csv',index=False)
        site_summary(site_rows).to_csv(output/'site_residuals.csv',index=False)
        stability=site_rows.copy()
        stability['azimuth_quadrant']=np.floor((np.rad2deg(np.arctan2(
            (stability.longitude-stability.event_longitude)*np.cos(np.deg2rad(stability.latitude)),
            stability.latitude-stability.event_latitude))%360)/90).astype(int)
        stability['strength']=pd.cut(stability.truth,[-np.inf,-2,-1.2,0,np.inf])
        counts=stability.groupby(['station_id','azimuth_quadrant','strength'],observed=True).agg(
            targets=('event_id','size'),events=('event_id','nunique'),
            observed_residual=('centered_observed_residual','mean'),predicted_residual=('centered_predicted_residual','mean'))
        counts.to_csv(output/'site_stability.csv')
        recovery=[]
        for key,group in site_summary(site_rows).groupby(['geometry_protocol','elapsed_time']):
            dense=group.loc[group.sufficient_events]
            recovery.append(dict(geometry_protocol=key[0],elapsed_time=key[1],stations=len(group),
                sufficient_stations=len(dense),sparse_stations=int((~group.sufficient_events).sum()),
                correlation=float(dense.observed.corr(dense.predicted)) if len(dense)>2 else None,
                mae=float((dense.observed-dense.predicted).abs().mean()) if len(dense) else None))
        pd.DataFrame(recovery).to_csv(output/'site_recovery.csv',index=False)
    frame.groupby(['station_id','geometry_protocol','elapsed_time']).agg(targets=('event_id','size'),events=('event_id','nunique')).to_csv(output/'station_counts.csv')
    pd.DataFrame([dict(targets=len(frame),events=frame.event_id.nunique(),
        forward_seconds_sum=float(frame.forward_seconds.sum()),encoder_seconds_sum=float(frame.encoder_seconds.sum()),
        forward_p50=float(frame.forward_seconds.median()),forward_p95=float(frame.forward_seconds.quantile(.95)),
        interpretation='per-query forward; region throughput requires replay_latency.csv')]).to_csv(output/'efficiency.csv',index=False)
    figure_data=output/'figure_data';figure_data.mkdir()
    finite[['event_id','elapsed_time','truth','prediction','predictive_sigma','pit','geometry_protocol']].to_csv(figure_data/'point_probability.csv.gz',index=False)
    write_json(output/'analysis_provenance.json',dict(source=str(Path(evaluation).name),
        failed_targets=int(frame.status.ne('supported').sum()),units='log10(m/s^2)',
        site_claim='station-specific repeatable residual; geology is not uniquely identified'))
    render_figures(finite,output)


def compare_runs(evaluations,output,common_times=(1,3,5,10,20)):
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Comparison output exists')
    output.mkdir(parents=True,exist_ok=True)
    frames=[pd.read_csv(Path(p)/'predictions.csv.gz',dtype={'event_id':str,'station_id':str}) for p in evaluations]
    provenance=[json.loads((Path(p)/'provenance.json').read_text()) for p in evaluations]
    for field in ['split','split_manifest_sha256','validation_population_sha256','random_times_manifest_sha256','window_protocol','stage']:
        if len({r.get(field) for r in provenance})!=1: raise ValueError('Unpaired evaluation provenance: '+field)
    for path in evaluations:
        status=pd.read_csv(Path(path)/'support_status.csv')
        if status.loc[status.elapsed_time.isin(common_times),'status'].ne('supported').any():
            raise ValueError('Common-domain missing/failing decisions cannot be omitted')
    main=[]
    for frame in frames:
        if frame['model_family'].nunique()!=1 or frame['seed'].nunique()!=1:
            raise ValueError('Each evaluation must contain one model/seed')
        main.append(frame.loc[frame.elapsed_time.isin(common_times)&frame.target_role.ne('observed_input')])
    keys=['dataset_id','event_id','elapsed_time','station_id','geometry_protocol','target_role','input_ids']
    expected=None
    for common in main:
        identity=fingerprint(common[keys+['truth']].sort_values(keys).to_dict('records'))
        if expected is not None and identity!=expected: raise ValueError('Common-domain target/input population differs')
        expected=identity
    primary=[];support=[]
    for frame,common in zip(frames,main):
        scores=grouped_metrics(common)
        for protocol in ('normal','random'):
            seen=set(scores.loc[(scores.target_group=='noninput')&(scores.geometry_protocol==protocol),'elapsed_time'])
            if seen!=set(common_times):
                raise ValueError('Common ranking cell missing')
        if common.status.ne('supported').any():
            raise ValueError('A model failure cannot be recoded as unsupported / dropped from ranking')
        primary.append(dict(model_family=frame.model_family.iloc[0],seed=int(frame.seed.iloc[0]),
            common_equal_time_protocol_mae=float(scores.loc[scores.target_group.eq('noninput'),'mae'].mean()),
            targets=len(common),events=int(common.event_id.nunique())))
        all_scores=grouped_metrics(frame)
        all_scores['model_family']=frame.model_family.iloc[0];all_scores['seed']=frame.seed.iloc[0]
        support.append(all_scores)
    paired=[]
    for baseline in main:
        if baseline.model_family.iloc[0]!='diting_pretrained_frozen':
            continue
        for competitor in main:
            if competitor.seed.iloc[0]!=baseline.seed.iloc[0] or competitor.model_family.iloc[0]==baseline.model_family.iloc[0]:
                continue
            for protocol in ('normal','random'):
                a=baseline.loc[baseline.geometry_protocol.eq(protocol)]
                b=competitor.loc[competitor.geometry_protocol.eq(protocol)]
                paired.append(dict(baseline='diting_pretrained_frozen',competitor=competitor.model_family.iloc[0],
                    seed=int(baseline.seed.iloc[0]),geometry_protocol=protocol,
                    **paired_bootstrap(a,b)))
            paired.append(dict(baseline='diting_pretrained_frozen',competitor=competitor.model_family.iloc[0],
                seed=int(baseline.seed.iloc[0]),geometry_protocol='equal_normal_random',
                **paired_bootstrap(baseline,competitor)))
    pd.DataFrame(primary).to_csv(output/'common_domain_scores.csv',index=False)
    pd.concat(support).to_csv(output/'full_capability_scores.csv',index=False)
    pd.DataFrame(paired).to_csv(output/'paired_event_bootstrap.csv',index=False)
    write_json(output/'comparison_contract.json',dict(common_times=list(common_times),
        cluster='event ID; all times/stations resampled together',draws=5000,
        multiple_comparisons='predefined DiTing comparisons; exploratory strata not confirmatory',
        support='native_prefix; not pure architecture causality'))


def render_figures(frame,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output)/'figures';output.mkdir(exist_ok=True)
    figure,axes=plt.subplots(1,3,figsize=(11,3.3),constrained_layout=True)
    axes[0].hexbin(frame.truth,frame.prediction,gridsize=45,mincnt=1,cmap='viridis')
    axes[0].set(xlabel='Observed log10 PGA (m/s²)',ylabel='Predicted mixture mean',title='Held-out observed targets')
    axes[1].hist(frame.pit,bins=np.linspace(0,1,21),density=True,color='#4477AA')
    axes[1].axhline(1,color='gray',linestyle='--');axes[1].set(xlabel='Mixture PIT',ylabel='Density')
    calibration=reliability(frame)
    for protocol,part in calibration.groupby('geometry_protocol'):
        axes[2].plot(part.mean_probability,part.observed_exceedance,'o-',label=protocol)
    axes[2].plot([0,1],[0,1],'--',color='gray');axes[2].set(xlabel='Predicted exceedance',ylabel='Observed frequency')
    axes[2].legend()
    figure.savefig(output/'point_probability.png',dpi=180)
    figure.savefig(output/'point_probability.pdf')
    plt.close(figure)


def training_histograms(run_dir,output):
    rows=[]
    import torch
    checkpoint=torch.load(Path(run_dir)/'last.pth',map_location='cpu',weights_only=False)
    for journal in checkpoint['committed_journals']:
        path=Path(run_dir)/journal
        for line in path.open():
            value=json.loads(line)
            rows.append({k:value.get(k) for k in ['epoch','event_id','requested_elapsed_time','input_count','geometry_protocol','rank']})
    frame=pd.DataFrame(rows)
    if not len(frame):
        raise ValueError('No actual training samples; theoretical density is not evidence')
    frame['time_bin']=pd.cut(frame.requested_elapsed_time,[.999,3,5,10,20,40,90.001],right=False)
    result=frame.groupby(['epoch','geometry_protocol','time_bin'],observed=True).agg(samples=('event_id','size'),events=('event_id','nunique')).reset_index()
    result.to_csv(output,index=False)
