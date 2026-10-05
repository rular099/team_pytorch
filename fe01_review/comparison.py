"""Independent system comparisons; no hard-coded DiTing family or seed clones."""
from pathlib import Path
import numpy as np
import pandas as pd
from .requests import read_frame,population_audit,PAIR_KEYS,IDENTITY_KEYS
from .provenance import require,read_json,fingerprint,sha256,evaluation_identity
from .diagnostics import group_scores,target_groups,fields,paired_fields,field_summary,station_recovery
from .statistics import event_losses,interval


def assert_same_population(left,right,physical=True):
    keys=IDENTITY_KEYS+['latitude','longitude','input_coords']
    if physical:keys+=['physical_received_history_sha256','query_elevation']
    require(not left.duplicated(PAIR_KEYS).any() and not right.duplicated(PAIR_KEYS).any(),'Repeated physical query')
    require(len(left)==len(right),'Target populations differ; no silent intersection')
    require(fingerprint(left[keys].sort_values(PAIR_KEYS).to_dict('records'))==fingerprint(right[keys].sort_values(PAIR_KEYS).to_dict('records')),'Physical request identity differs')


def random_loss_table(frame,mapping,role):
    group=target_groups(frame)[role]
    data=group.assign(abs_error=(group.prediction-group.truth).abs(),nll=group.nll)
    scores=data.groupby(['dataset_id','event_id','elapsed_time','geometry_protocol'])[['abs_error','nll']].mean().reset_index()
    expected=mapping[['dataset_id','event_id','elapsed_time','draw']].merge(pd.DataFrame({'geometry_protocol':['normal','random']}),how='cross')
    joined=expected.merge(scores,on=['dataset_id','event_id','elapsed_time','geometry_protocol'],how='left',validate='many_to_one')
    require(joined.abs_error.notna().all(),'Random subgroup missing original draws; no conditional mean')
    out=joined.groupby(['dataset_id','event_id','geometry_protocol'])[['abs_error','nll']].sum().add_suffix('_sum').reset_index()
    out['elapsed_time']=0;out['targets']=3;out['target_group']=role
    return out


def compare_available(root,source_root,plan,output):
    root,output=Path(root),Path(output);frames={};gates=[];scope_scores=[];paired=[];metrics=[];field_stats=[];field_ci=[];sites=[]
    reference_path=root/'reference/train_only_reference.json'
    reference=read_json(reference_path) if reference_path.is_file() else None
    published=read_frame(Path(__file__).resolve().parents[1]/'reports/fe01_hpc_results_20261004/run_summary.csv')
    for scope in ('fixed','random','long'):
        current={}
        if scope=='fixed':
            for r in published.itertuples():
                v=root/'verification'/r.run_id/'verification.json'
                if not v.is_file() or read_json(v).get('status')!='PASS':continue
                if read_json(v).get('evaluation_module_sha')!=evaluation_identity()['evaluation_module_sha']:
                    gates.append(dict(scope=scope,run_id=r.run_id,status='BLOCKED',reason='Stale verification module SHA'));continue
                p=Path(source_root)/r.selected_source;require(sha256(p)==r.selected_source_sha256,'Reused CSV SHA changed')
                require(read_json(v).get('reused_csv_sha256')==r.selected_source_sha256,'Reused CSV forward gate differs')
                f=read_frame(p);population_audit(f);current[r.run_id]=f
        for p in sorted((root/'evaluations').glob('*/'+scope+'/predictions.csv.gz')):
            identity=read_json(p.parent/'provenance.json');require(identity['scope']==scope,'Evaluation scope mismatch')
            f=read_frame(p);require(f.status.eq('supported').all(),'Failed targets cannot be omitted')
            if scope=='fixed':population_audit(f)
            current[p.parent.parent.name]=f
        names=sorted(current)
        fixed_fields={}
        for name,f in current.items():
            from fe01.metrics import grouped_metrics
            metrics.append(grouped_metrics(f).assign(scope=scope,run_id=name))
            if scope=='fixed':
                fixed_fields[name]=fields(f)
                field_stats.append(field_summary(fixed_fields[name]).assign(scope=scope,run_id=name))
                if reference and 'strict_prefix_replay' in name:
                    s,t,u=station_recovery(f,reference)
                    s.assign(run_id=name).to_csv(output/(name+'_station_recovery.csv.gz'),index=False)
                    u.assign(run_id=name).to_csv(output/(name+'_station_stability.csv.gz'),index=False)
                    sites.append(t.assign(run_id=name))
            if scope=='random':
                mapping=read_frame(Path(plan)/'random_draw_mapping.csv')
                for role in ('all','noninput'):
                    try:
                        loss=random_loss_table(f,mapping,role)
                        scope_scores.append(dict(scope=scope,run_id=name,target_group=role,endpoint='original_draw_event_geometry_equal_MAE',**interval(loss)))
                    except ValueError as exc:
                        gates.append(dict(scope=scope,run_id=name,target_group=role,status='INCOMPLETE',reason=str(exc)))
            else:
                for role,t in event_losses(f).groupby('target_group'):
                    for protocol,g in t.groupby('geometry_protocol'):
                        scope_scores.append(dict(scope=scope,run_id=name,target_group=role,geometry_protocol=protocol,evaluated_times=','.join(map(str,sorted(g.elapsed_time.unique()))),**interval(g)))
        for i,left in enumerate(names):
            for right in names[i+1:]:
                # All legacy references paired independently with every FE01 seed; never clone references.
                legacy='strict_prefix_replay' in left or 'strict_prefix_replay' in right
                same_seed=left.split('seed')[-1]==right.split('seed')[-1]
                if not legacy and not same_seed:continue
                a,b=current[left],current[right]
                if scope=='long':
                    shared=sorted(set(a.elapsed_time)&set(b.elapsed_time));require(shared,'No common native capacity')
                    a,b=a[a.elapsed_time.isin(shared)],b[b.elapsed_time.isin(shared)]
                try:
                    assert_same_population(a,b,physical=scope!='fixed')
                    if scope=='fixed' and legacy:
                        field_ci.append(paired_fields(fixed_fields[left],fixed_fields[right]).assign(left=left,right=right))
                    for role in ('all','noninput','untriggered_noninput','actual_single_input_noninput'):
                        try:
                            if scope=='random':
                                la,lb=random_loss_table(a,mapping,role),random_loss_table(b,mapping,role)
                            else:
                                la=event_losses(a);lb=event_losses(b);la,lb=la[la.target_group==role],lb[lb.target_group==role]
                            for metric in ('abs_error','nll'):
                                paired.append(dict(scope=scope,left=left,right=right,target_group=role,metric=metric,**interval(la,metric,right=lb)))
                        except ValueError as exc:gates.append(dict(scope=scope,left=left,right=right,target_group=role,status='INCOMPLETE',reason=str(exc)))
                    gates.append(dict(scope=scope,left=left,right=right,status='PASS',reason='exact physical query/input/label/clock population'))
                except ValueError as exc:gates.append(dict(scope=scope,left=left,right=right,status='BLOCKED',reason=str(exc)))
        expected=8 if scope=='long' else 11
        if len(current)<expected:gates.append(dict(scope=scope,status='INCOMPLETE',reason=f'{len(current)} available certified systems of{expected}; PhaseNet long N/A; EQT90 N/A'))
    pd.DataFrame(scope_scores).to_csv(output/'new_scope_scores.csv',index=False)
    pd.DataFrame(paired).to_csv(output/'new_system_paired_event_ci.csv',index=False)
    pd.DataFrame(gates).to_csv(output/'new_system_comparison_gates.csv',index=False)
    if metrics:pd.concat(metrics,ignore_index=True).to_csv(output/'new_metrics_by_scope_time_geometry_role.csv',index=False)
    if field_stats:pd.concat(field_stats,ignore_index=True).to_csv(output/'certified_fixed_field_summary.csv',index=False)
    if field_ci:pd.concat(field_ci,ignore_index=True).to_csv(output/'legacy_paired_field_event_ci.csv',index=False)
    if sites:pd.concat(sites,ignore_index=True).to_csv(output/'legacy_station_recovery_summary.csv',index=False)
    return gates
