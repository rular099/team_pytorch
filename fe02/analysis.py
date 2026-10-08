"""Variant-aware paired readout comparisons; model_family remains real DiTing."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fe01.config import fingerprint, write_json
from fe01.metrics import checkpoint_selection, grouped_metrics, paired_bootstrap
from fe01.spatial import field_metrics

CONTRASTS = [('R0','A'),('A','B'),('B','C'),('M','C'),('A','M'),('B','M'),('R0','M')]
KEYS = ['dataset_id','event_id','elapsed_time','geometry_protocol','station_id','target_role','input_ids']


def compare(evaluations, output, seeds=(42,), bootstrap_draws=5000):
    output=Path(output)
    if output.exists(): raise FileExistsError('Comparison must use a new directory')
    frames={};provenances=[];identity=None;headline=[];strata=[];fields=[];paired=[]
    for path in map(Path,evaluations):
        provenance=json.loads((path/'provenance.json').read_text())
        if provenance['split']!='val': raise ValueError('FE02 comparison is validation only')
        provenances.append(provenance)
        frame=pd.read_csv(path/'predictions.csv.gz',dtype={'event_id':str,'station_id':str})
        probability_columns = ['nll','crps','brier','predictive_sigma','pit','covered95','covered68','width95']
        if not set(probability_columns).issubset(frame.columns) or not np.isfinite(frame[probability_columns].to_numpy(dtype=float)).all():
            raise ValueError('Full finite probability evidence is required for FE02 comparison')
        if (frame.model_family.ne('diting_pretrained_frozen').any() or frame.split.ne('val').any()
                or not frame.status.eq('supported').all() or not np.isfinite(frame.prediction).all()):
            raise ValueError('Non-DiTing/test/failing predictions cannot enter FE02 ranking')
        if frame.variant_id.nunique()!=1 or frame.seed.nunique()!=1: raise ValueError('Mixed run identity')
        variant=frame.variant_id.iloc[0];seed=int(frame.seed.iloc[0])
        if provenance['variant_id']!=variant or (seed,variant) in frames: raise ValueError('Duplicate/mismatched variant')
        if frame.duplicated(KEYS).any(): raise ValueError('Duplicated target identity')
        current=fingerprint(frame[KEYS+['truth','input_count']].sort_values(KEYS).to_dict('records'))
        if identity is not None and identity!=current: raise ValueError('Paired target/input labels/population differ')
        identity=current
        status=pd.read_csv(path/'support_status.csv')
        if not status.status.eq('supported').all(): raise ValueError('Missing/failing decisions cannot be dropped')
        frames[seed,variant]=frame
        primary=frame.loc[frame.elapsed_time.isin([1,3,5,10,20]) & frame.target_role.ne('observed_input')]
        headline.append(dict(seed=seed,variant_id=variant,model_family='diting_pretrained_frozen',
            checkpoint_epoch=provenance['checkpoint_epoch'],checkpoint_sha256=provenance['checkpoint_sha256'],
            common_equal_time_protocol_noninput_mae=checkpoint_selection(frame,[1,3,5,10,20]),
            primary_targets=len(primary),primary_events=primary.event_id.nunique()))
        for name,part in [('all_counts',frame),('single',frame.loc[frame.input_count.eq(1)]),
                          ('multi',frame.loc[frame.input_count.ge(2)])]:
            if part.empty: continue
            metrics=grouped_metrics(part);metrics['seed']=seed;metrics['variant_id']=variant;metrics['input_count_stratum']=name
            strata.append(metrics)
            remote=part.loc[part.target_role.ne('observed_input')]
            if not remote.empty:
                spatial=field_metrics(remote)
                spatial['seed']=seed;spatial['variant_id']=variant;spatial['input_count_stratum']=name
                fields.append(spatial)
    expected={(s,v) for s in seeds for v in ('R0','A','B','C','M')}
    if set(frames)!=expected: raise ValueError('Incomplete or extra FE02 seed/variant matrix')
    for field in ('split_manifest_sha256','validation_population_sha256','window_protocol','stage',
                  'encoder_checkpoint_sha256','random_times_manifest_sha256'):
        if len({p.get(field) for p in provenances})!=1: raise ValueError('Unpaired provenance: '+field)
    if provenances[0].get('random_times_manifest_sha256') is not None:
        raise ValueError('Use fixed-time evaluations for primary FE02 comparison')
    for seed in seeds:
        for left,right in CONTRASTS:
            a=frames[seed,left];b=frames[seed,right]
            for count_name,count_mask in [('all_counts',a.input_count.ge(1)),('single',a.input_count.eq(1)),('multi',a.input_count.ge(2))]:
                for protocol in ('normal','random','equal_normal_random'):
                    selected=count_mask & a.elapsed_time.isin([1,3,5,10,20]) & a.target_role.ne('observed_input')
                    if protocol!='equal_normal_random': selected &= a.geometry_protocol.eq(protocol)
                    # Population/order are already certified identical; select by exact keys.
                    sub=a.loc[selected];other=b.merge(sub[KEYS],on=KEYS,validate='one_to_one')
                    if sub.empty: continue
                    result=paired_bootstrap(sub,other,draws=bootstrap_draws,seed=20261008)
                    result['bootstrap_seed'] = result.pop('seed')
                    paired.append(dict(seed=seed,baseline=left,competitor=right,
                        input_count_stratum=count_name,geometry_protocol=protocol,
                        endpoint_domain='existing cells within common 1/3/5/10/20s; strata exploratory',**result))
    output.mkdir(parents=True)
    pd.DataFrame(headline).to_csv(output/'headline_primary.csv',index=False)
    pd.concat(strata,ignore_index=True).to_csv(output/'metrics_time_geometry_count_target.csv',index=False)
    pd.concat(fields,ignore_index=True).to_csv(output/'spatial_fields.csv',index=False)
    pd.DataFrame(paired).to_csv(output/'paired_event_bootstrap.csv',index=False)
    write_json(output/'comparison_contract.json',dict(status='PASS',split='val',units='log10(m/s^2)',
        model_family='diting_pretrained_frozen',paired_population_sha256=identity,
        contrasts=CONTRASTS,seeds=list(seeds),bootstrap_draws=bootstrap_draws,bootstrap_seed=20261008,
        selection='equal normal/random and common 1/3/5/10/20s, noninput MAE; earlier epoch on tie',
        spatial='field spread and pairwise delta accompany point/probability scores; not physical-geology proof',
        capacity='C bypasses PGA cross-attention; not strict equal-capacity ablation',provenance=provenances))
