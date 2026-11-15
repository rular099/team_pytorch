"""Real saved-prediction analysis, independent of Torch/checkpoint availability."""
import numpy as np
import pandas as pd
from .contracts import EPOCHS, EVENT, TIMES, contract
from .io import csv_write, fingerprint, write_json
from .sources import load_export, load_spatial_export, verify_export_seals
from .statistics import ENDPOINTS, bootstrap_pair, bootstrap_mae, groups, paired, robustness, summarize
from .stratification import assign, covariates, support


def analyze_existing(source,out,draws=2000,sidecar=None):
    evidence=verify_export_seals(source)
    write_json(out/'source_export_verification.json',dict(status='PASS',exports=evidence,protocol='legacy_snapshot_v1'))
    metrics,effects,eventstats,contributions,field_effects=[],[],[],[],[]
    strata_results, interaction_results, supports=[],[],[]
    for policy in ('selected','epoch12'):
        data={family:load_export(source,family,policy) for family in EPOCHS}
        d=data['DiTing']
        spatial={family:load_spatial_export(source,family,policy) for family in EPOCHS}
        print('POLICY_LOADED',policy,flush=True)
        for family in ('PhaseNet','EQT','TEAM'):
            paired(d,data[family])
        for family,f in data.items():
            for group,mask in groups(f).items():
                for endpoint,times in ENDPOINTS.items():
                    metrics.append(dict(family=family,checkpoint_policy=policy,epoch=EPOCHS[family] if policy=='selected' else 12,
                                        target_group=group,endpoint=endpoint,**summarize(f.loc[mask],times)))
        # Covariates depend only on the common physical request/metadata, never on a model output.
        enriched=covariates(d,sidecar)
        strata=assign(enriched)
        non=enriched.target_role.ne('observed_input')
        table,factors=support(enriched[non],strata[non],contract()['strata']['interactions'])
        table['checkpoint_policy']=policy
        supports.append(table)
        csv_write(table,out/('strata_support_frozen_'+policy+'.csv'))
        write_json(out/('strata_support_frozen_'+policy+'.json'),dict(
            support_sha256=fingerprint(table.to_dict('records')),rule='support and bins frozen before effect calculation',
            exploratory=True, multiple_testing='uncorrected; <20 events descriptive only'))
        masks=groups(d)
        for comparator in ('PhaseNet','EQT','TEAM'):
            b=data[comparator]
            for group,mask in masks.items():
                for endpoint,times in ENDPOINTS.items():
                    a_sub,b_sub=d.loc[mask],b.loc[mask]
                    for r in bootstrap_pair(a_sub,b_sub,times,draws):
                        effects.append(dict(comparator=comparator,checkpoint_policy=policy,target_group=group,endpoint=endpoint,**r))
                r,s=robustness(d.loc[mask],b.loc[mask])
                tag=dict(comparator=comparator,checkpoint_policy=policy,target_group=group)
                event_file=out/'event_effects.csv.gz'
                r.assign(**tag).to_csv(event_file,mode='a' if event_file.exists() else 'x',
                    header=not event_file.exists(),index=False,compression='gzip')
                eventstats.append(dict(**tag,**s))
            mask=masks['noninput']
            primary=next(x['estimate'] for x in effects if x['comparator']==comparator and x['checkpoint_policy']==policy and x['target_group']=='noninput' and x['endpoint']=='primary_10cell' and x['metric']=='mae')
            for geo in ('normal','random'):
                for t in TIMES:
                    m=mask & d.geometry_protocol.eq(geo).to_numpy() & d.elapsed_time.eq(t).to_numpy()
                    delta=float((np.abs(d.loc[m,'prediction']-d.loc[m,'truth'])-np.abs(b.loc[m,'prediction']-b.loc[m,'truth'])).mean())
                    contributions.append(dict(comparator=comparator,checkpoint_policy=policy,geometry_protocol=geo,elapsed_time=t,
                                              cell_delta=delta,contribution_to_primary=delta/10,
                                              signed_fraction_of_primary=delta/10/primary if abs(primary)>1e-12 else None,
                                              interpretation='algebraic weighted difference; not percent MAE reduction or causal attribution'))
            # All fixed factors plus exactly the four prespecified interactions, noninput.
            cache={}
            for factor,labels in factors.items():
                for label in sorted(labels.unique()):
                    idx=labels.index[labels.eq(label)]
                    signature=tuple(idx)
                    if signature not in cache:
                        cache[signature]=bootstrap_mae(d.loc[idx],b.loc[idx],draws,TIMES)
                    dest=interaction_results if ' X ' in factor else strata_results
                    for r in cache[signature]:
                        dest.append(dict(comparator=comparator,checkpoint_policy=policy,factor=factor,stratum=label,**r))
            print('PAIRED_AND_STRATA_COMPLETE',policy,comparator,flush=True)
            # Reuse already authenticated original field exports rather than repeating field construction.
            for group,mask in masks.items():
                af=spatial['DiTing'];bf=spatial[comparator]
                af=af[af.a01_target_group.eq(group)].sort_values(EVENT+['geometry_protocol','elapsed_time']).reset_index(drop=True)
                bf=bf[bf.a01_target_group.eq(group)].sort_values(EVENT+['geometry_protocol','elapsed_time']).reset_index(drop=True)
                if len(af):
                    from .spatial_statistics import field_intervals
                    for r in field_intervals(af,bf,draws):
                        field_effects.append(dict(comparator=comparator,checkpoint_policy=policy,target_group=group,**r))
                    field_file=out/'spatial_fields.csv.gz'
                    for part,family in ((af,'DiTing'),(bf,comparator)):
                        part.assign(family=family,comparator=comparator,checkpoint_policy=policy,target_panel=group).to_csv(
                            field_file,mode='a' if field_file.exists() else 'x',header=not field_file.exists(),index=False,compression='gzip')
        del data
    tables={'cross_encoder_metrics.csv':pd.DataFrame(metrics),'paired_effects.csv':pd.DataFrame(effects),
            'event_robustness.csv':pd.DataFrame(eventstats),
            'first_second_contributions.csv':pd.DataFrame(contributions),'stratified_effects.csv':pd.DataFrame(strata_results),
            'interaction_effects.csv':pd.DataFrame(interaction_results),'strata_support.csv':pd.concat(supports,ignore_index=True),
            'spatial_paired_effects.csv':pd.DataFrame(field_effects)}
    for name,f in tables.items():
        csv_write(f,out/name)
    csv_write(pd.DataFrame(effects)[['comparator','checkpoint_policy','target_group','endpoint','metric','events','requests','supported_cells','missing_cells','effective_replicates','empty_cell_fraction','status']],out/'bootstrap_support.csv')
    write_json(out/'analysis_scope.json',dict(split='validation',seed=42,protocol='legacy_snapshot_v1',draws=draws,
        result_type='REAL_JAPAN_SAVED_PREDICTION_ANALYSIS',effect='DiTing minus comparator',
        tails='P90/P95 pooled descriptive endpoint; no averaging cell quantiles; CIs cover additive metrics and transformed RMSE',
        limitations=['single seed; fixed selected models; no training/selection uncertainty',
                     'exploratory endpoints and strata uncorrected','SNR/missing/origin Unknown until matching HDF sidecar',
                     'shared downstream comparison does not isolate pretraining causal effect','UPSTREAM_CAUSALITY_UNKNOWN']))
