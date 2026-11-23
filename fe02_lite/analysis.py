"""Recompute scores, plots and paired/event-cluster comparisons without DiTing."""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fe01.metrics import score_rows, summarize, grouped_metrics, paired_bootstrap, checkpoint_selection
from fe01.spatial import field_metrics, distance_km
from .artifacts import read_table
from .config import fingerprint, sha256, write_json


def scored(path):
    frame = read_table(Path(path)/'predictions')
    w = frame[[f'w{k}' for k in range(3)]].to_numpy()
    mu = frame[[f'mu{k}' for k in range(3)]].to_numpy()
    sigma = frame[[f'sigma{k}' for k in range(3)]].to_numpy()
    for key,value in score_rows(frame.truth.to_numpy(),w,mu,sigma).items():
        frame[key] = value
    return frame


def cluster_ci(frame, draws=1000, seed=42):
    clusters = frame.assign(abs_error=(frame.prediction-frame.truth).abs()).groupby('canonical_event_id').abs_error.agg(['sum','count'])
    if clusters.empty: return dict(events=0,targets=0)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        ix = rng.integers(0,len(clusters),len(clusters))
        values.append(clusters['sum'].to_numpy()[ix].sum()/clusters['count'].to_numpy()[ix].sum())
    return dict(mae_ci025=float(np.quantile(values,.025)),mae_ci975=float(np.quantile(values,.975)),
                events=len(clusters),targets=len(frame),draws=draws,cluster='canonical_event_id',bootstrap_seed=seed)


def plot_table(frame, output, name, x, ys, units, filters):
    """Every figure has exact plotted CSV and machine-readable conditions/counts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame.to_csv(output/(name+'.csv'),index=False)
    write_json(output/(name+'.json'),dict(rows=len(frame),units=units,filters=filters,x=x,y=ys))
    fig,ax = plt.subplots(figsize=(8,4))
    for y in ys:
        ax.plot(frame[x],frame[y],label=y)
    ax.set_xlabel(x)
    ax.set_ylabel(units)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/(name+'.png'),dpi=140)
    plt.close(fig)


def source_errors(sources, frame):
    rows = []
    for record in sources.to_dict('records'):
        w,mu = np.asarray(json.loads(record['location_weights'])),np.asarray(json.loads(record['location_mu']))
        pred,truth = (w[:,None]*mu).sum(0),np.asarray(json.loads(record['truth_location']))
        mag = np.asarray(json.loads(record['magnitude_mdn']))[0,1]
        rows.append(dict(observation_id=record['observation_id'],event_id=record['event_id'],
            source_horizontal_error_km=float(distance_km(pred[0],pred[1],truth[0],truth[1])),
            source_depth_error_km=float(pred[2]-truth[2]),magnitude_error=float(mag-record['truth_magnitude'])))
    result = pd.DataFrame(rows)
    if result.observation_id.duplicated().any():
        raise ValueError('Source output repeated per target/chunk')
    pga = frame.assign(absolute_error=(frame.prediction-frame.truth).abs()).groupby('observation_id').agg(
        pga_mae=('absolute_error','mean'),targets=('absolute_error','size'))
    return result.merge(pga,on='observation_id',validate='one_to_one')


def diagnose(directory, output):
    directory,output = Path(directory),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    index,features = [],[]
    for file in sorted(directory.glob('*.npz')):
        meta = json.loads(file.with_suffix('.json').read_text())
        with np.load(file,allow_pickle=False) as data:
            # Numeric comparison table across init/best/last. Raw arrays remain accessible.
            for key in data.files:
                arr = data[key].astype(np.float64)
                features.append(dict(sample=file.stem,checkpoint=meta['checkpoint_tag'],
                    observation_id=meta['observation_id'],input_count=meta['input_count'],
                    elapsed_time=meta['actual_elapsed_time'],geometry_protocol=meta['geometry_protocol'],
                    count_rule=meta['count_rule'],tensor=key,shape=str(arr.shape),elements=arr.size,
                    mean=float(arr.mean()),std=float(arr.std()),norm=float(np.linalg.norm(arr)),scope='fixed diagnostic samples'))
            for key in ('pool_attention','event_attention','target_attention'):
                arr = data[key]
                name = file.stem+'__'+key
                # Every heatmap exports its entire plotted matrix, with order mappings in JSON.
                matrix = arr.reshape(-1,arr.shape[-1]).astype(float)
                pd.DataFrame(matrix).to_csv(output/(name+'.csv'),index=False)
                write_json(output/(name+'.json'),dict(metadata=meta,shape=list(arr.shape),units='attention weight',
                    samples=1,interpretation='diagnostic attention; no propagation ray or geological inference'))
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig,ax = plt.subplots(figsize=(8,3))
                ax.imshow(matrix,aspect='auto',interpolation='nearest')
                ax.set_xlabel('key index (see saved station/token order)')
                ax.set_ylabel('query index')
                fig.tight_layout()
                fig.savefig(output/(name+'.png'),dpi=100)
                plt.close(fig)
        index.append(dict(path=str(file),sha256=sha256(file),checkpoint=meta['checkpoint_tag']))
    pd.DataFrame(features).to_csv(output/'representation_summary.csv',index=False)
    write_json(output/'sources.json',index)


def summarize_run(evaluation, output, run_dir=None, draws=1000):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    frame = scored(evaluation)
    frame['input_count_group'] = np.where(frame.input_count==1,'single','multi')
    metrics = []
    for keys,part in frame.groupby(['dataset_view','elapsed_time','geometry_protocol','input_count_group','target_role']):
        metrics.append(dict(zip(['dataset_view','elapsed_time','geometry_protocol','input_count_group','target_role'],keys),
                            **summarize(part),**{k:v for k,v in cluster_ci(part,draws).items() if k not in ('events','targets')}))
    pd.DataFrame(metrics).to_csv(output/'stratified_metrics.csv',index=False)
    grouped_metrics(frame).to_csv(output/'metrics_time_role.csv',index=False)
    primary = frame.loc[frame.elapsed_time.isin([1,3,5,10,20]) & frame.target_role.ne('observed_input')]
    write_json(output/'summary.json',dict(overall=summarize(frame),noninput=summarize(primary),
        overall_ci=cluster_ci(frame,draws),selection=checkpoint_selection(frame,[1,3,5,10,20]),
        units='log10(m/s^2)',point='E[log10 PGA], distinct from physical arithmetic mean',
        brier_threshold_log10_mps2=-1.2,intervals='equal-tailed mixture quantiles: .025/.975 and .158655/.841345',
        source_manifest_sha256=sha256(Path(evaluation)/'predictions/manifest.json')))
    fields = field_metrics(primary)
    ranges = []
    for key,part in primary.groupby(['event_id','elapsed_time','geometry_protocol']):
        truth_range = float(part.truth.quantile(.95)-part.truth.quantile(.05))
        ranges.append(dict(event_id=key[0],elapsed_time=key[1],geometry_protocol=key[2],field_targets=len(part),
            truth_p95_p05_dex=truth_range,predicted_p95_p05_dex=float(part.prediction.quantile(.95)-part.prediction.quantile(.05)),
            truth_max_min_dex=float(part.truth.max()-part.truth.min()),
            predicted_max_min_dex=float(part.prediction.max()-part.prediction.min()),
            centered_pairwise_status='supported' if len(part)>=5 else 'fewer_than_5_noninput_targets',
            range_ratio_status='supported' if len(part)>=5 and truth_range>=.05 else 'requires_5_targets_and_truth_range_ge_0.05_dex'))
    ranges = pd.DataFrame(ranges)
    if not fields.empty:
        ranges = ranges.merge(fields,on=['event_id','elapsed_time','geometry_protocol'],how='left',validate='one_to_one')
    ranges.to_csv(output/'spatial_fields.csv',index=False)
    sources = read_table(Path(evaluation)/'sources')
    errors = source_errors(sources,frame)
    errors.to_csv(output/'source_errors_and_pga.csv',index=False)
    corr = errors[['source_horizontal_error_km','source_depth_error_km','magnitude_error','pga_mae']].corr()
    corr.to_csv(output/'source_pga_correlation.csv')
    # Scatter source data with event cluster identities remains available above.
    time = primary.groupby('elapsed_time').agg(mae=('error',lambda s:s.abs().mean()),
        targets=('error','size'),events=('event_id','nunique')).reset_index()
    plot_table(time,output,'pga_by_time','elapsed_time',['mae'],'dex',dict(population='noninput',samples=len(primary)))
    if run_dir:
        run_dir = Path(run_dir)
        curves = pd.read_csv(run_dir/'training_curves.csv')
        plot_table(curves,output,'training','epoch',['train_loss','selection'],'loss / dex',dict(scope='all rank epoch aggregates'))
        logs = []
        checkpoint = torch_load_cpu(run_dir/'last.pth')
        for record in checkpoint['committed_files']:
            path = run_dir/record['path']
            if path.parent.name != 'step_logs': continue
            if sha256(path) != record['sha256']: raise ValueError('Committed training log hash differs')
            import gzip
            with gzip.open(path,'rt') as stream:
                for line in stream:
                    item = json.loads(line)
                    base = {k:v for k,v in item.items() if not isinstance(v,(dict,list))}
                    for section in ('losses','module_gradient_norms','module_update_norms'):
                        base.update({section+'.'+k:v for k,v in (item.get(section) or {}).items()})
                    logs.append(base)
        log_frame = pd.DataFrame(logs)
        log_frame.to_csv(output/'training_step_diagnostics.csv.gz',index=False)
        for prefix in ('module_gradient_norms.','module_update_norms.'):
            columns = [c for c in log_frame if c.startswith(prefix)]
            if columns:
                chosen = log_frame.groupby('step')[columns].mean().dropna().reset_index()
                plot_table(chosen,output,prefix.split('.')[0],'step',columns,'L2 norm',dict(scope='mean recorded ranks; configured update snapshots'))
    return frame


def torch_load_cpu(path):
    import torch
    return torch.load(path,map_location='cpu',weights_only=False)


def compare(evaluations, output, draws=1000):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    frames = [scored(path) for path in evaluations]
    rows = []
    for file,frame in zip(evaluations,frames):
        key = frame.iloc[0]
        rows.append(dict(evaluation=str(file),dataset_view=key.dataset_view,variant=key.variant,seed=int(key.seed),
            run_id=key.run_id,checkpoint_epoch=int(key.checkpoint_epoch),population='all valid view samples',
            **{name:key[name].item() if hasattr(key[name],'item') else key[name] for name in
               ['model_dim','amplitude_dim','station_pool_queries','source_aux_enabled','trainable_task_parameters'] if name in key},
            **summarize(frame),**{k:v for k,v in cluster_ci(frame,draws).items() if k not in ('events','targets')}))
    pd.DataFrame(rows).to_csv(output/'architecture_view_comparison.csv',index=False)
    common_events = set.intersection(*(set(f.canonical_event_id) for f in frames))
    common,paired,pair_support = [],[],[]
    for file,frame in zip(evaluations,frames):
        part = frame.loc[frame.canonical_event_id.isin(common_events)]
        common.append(dict(evaluation=str(file),population='common canonical events; view-specific truth/input/absolute times retained',
                           **summarize(part)))
        part.to_csv(output/(fingerprint(str(file))[:12]+'_common_events.csv.gz'),index=False)
    pd.DataFrame(common).to_csv(output/'common_event_metrics.csv',index=False)
    keys = ['dataset_id','event_id','elapsed_time','geometry_protocol','station_id','target_role','input_ids']
    for (li,left),(ri,right) in itertools.combinations(enumerate(frames),2):
        if left.dataset_view.iloc[0] != right.dataset_view.iloc[0]:
            # Surface/borehole share physical site but different truth. Keep a verified
            # absolute-time intersection; never call this an architecture paired score.
            cross_keys = ['canonical_event_id','physical_station_id','absolute_cutoff_utc','geometry_protocol']
            a,b = left.copy(),right.copy()
            # Exact absolute cutoff values only, no rounding or relative-P proxy.
            joined = a.merge(b,on=cross_keys,suffixes=('_left','_right'),validate='one_to_one')
            joined.to_csv(output/f'cross_view_{li}_{ri}_matched_records.csv.gz',index=False)
            pair_support.append(dict(left=li,right=ri,kind='cross-view same site / exact absolute cutoff',
                matched_records=len(joined),events=joined.canonical_event_id.nunique(),
                note='different layers have different truth; input geometries may differ; no causal architecture claim'))
            continue
        comparable = keys+['truth','absolute_cutoff_utc','input_coords','normalization_sha256','coordinate_sha256']
        provenance = ['data_shards_sha256','split_manifest_sha256','view_manifest_sha256','cohort_sha256',
                      'encoder_checkpoint_sha256','encoder_yaml_sha256']
        if not all(name in f for f in (left,right) for name in provenance):
            pair_support.append(dict(left=li,right=ri,kind='architecture comparison refused',matched_records=0,
                note='missing data/split/cohort/encoder provenance'))
            continue
        comparable += provenance
        if fingerprint(left[comparable].sort_values(keys).to_dict('records')) != fingerprint(right[comparable].sort_values(keys).to_dict('records')):
            pair_support.append(dict(left=li,right=ri,kind='architecture comparison refused',matched_records=0,
                note='data/time/input/target/normalization identities differ'))
            continue
        a = left.loc[left.elapsed_time.isin([1,3,5,10,20]) & left.target_role.ne('observed_input')]
        b = right.loc[right.elapsed_time.isin([1,3,5,10,20]) & right.target_role.ne('observed_input')]
        paired.append(dict(left=li,right=ri,**paired_bootstrap(a,b,draws=draws,seed=42)))
    pd.DataFrame(paired).to_csv(output/'paired_event_bootstrap.csv',index=False)
    pd.DataFrame(pair_support).to_csv(output/'pairing_support.csv',index=False)
    write_json(output/'comparison_contract.json',dict(evaluations=list(map(str,evaluations)),
        common_events=sorted(common_events),bootstrap_cluster='event',draws=draws,
        cross_view='relative first-P times are not physical pairing; only exact absolute cutoffs may match'))
