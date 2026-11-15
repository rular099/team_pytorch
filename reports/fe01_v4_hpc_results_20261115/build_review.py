"""Build a neutral, reproducible review from verified saved data; never load models."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import warnings

import numpy as np
import pandas as pd

FAMILIES = ['DiTing', 'PhaseNet', 'EQT', 'TEAM']
EVENT = ['dataset_id', 'event_id']
TT_KEYS = ['geometry_protocol', 'panel', 'budget_dex', 'gate', 'scene']
REPO = Path(__file__).resolve().parents[2]


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4194304), b''):
            h.update(b)
    return h.hexdigest()


def read(p, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message="The 'float_precision' argument is deprecated.*")
        return pd.read_csv(p, float_precision='round_trip', dtype={k: str for k in EVENT}, **kwargs)


def close(a, b, label, tolerance=2e-10):
    if not np.isclose(a, b, atol=tolerance, rtol=0, equal_nan=True):
        raise ValueError((label, a, b))


def interval(values):
    """Independent fixed-event bootstrap, same declared seed/draws as the HPC protocol."""
    x = np.asarray(values, float)
    if not np.isfinite(x).any():
        return dict(estimate=np.nan, ci_low=np.nan, ci_high=np.nan, events=len(x), known_events=0, effective_replicates=0)
    rng = np.random.default_rng(20261007)
    dist = []
    for start in range(0, 2000, 100):
        sampled = x[rng.integers(0, len(x), size=(100, len(x)))]
        keep = np.isfinite(sampled).any(axis=1)
        dist.extend(np.nanmean(sampled[keep], axis=1))
    return dict(estimate=float(np.nanmean(x)), ci_low=float(np.quantile(dist, .025)),
                ci_high=float(np.quantile(dist, .975)), events=len(x),
                known_events=int(np.isfinite(x).sum()), effective_replicates=len(dist))


def build(raw, out):
    raw, out = Path(raw), Path(out)
    if out.exists():
        raise ValueError('Preserve existing output: ' + str(out))
    out.mkdir(parents=True)
    sources, derived = {}, {}

    def copy(relative, target=None):
        src, dst = raw / relative, out / (target or ('source_tables/' + relative))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        sources[str(dst.relative_to(out))] = dict(archive_member='fe01_on_eval_v4_review/' + relative,
                                                 sha256=digest(src), bytes=src.stat().st_size)

    def save(frame, name, inputs):
        p = out / 'derived' / name
        p.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(p, index=False, float_format='%.17g',
                     compression={'method':'gzip', 'mtime':0} if name.endswith('.gz') else None)
        derived[str(p.relative_to(out))] = dict(rows=len(frame), sha256=digest(p), inputs=inputs)

    def store_json(name, value):
        p = out / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('x') as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')

    # Small original tables remain unchanged. Duplicate pre-hydration legacy tables are omitted.
    for directory in ['legacy_with_physical_covariates', 'interaction_contrasts', 'additional_endpoints']:
        for p in sorted((raw / 'analyze/0' / directory).glob('*')):
            if p.is_file() and p.suffix in ['.csv', '.json']:
                copy(str(p.relative_to(raw)))
    for name in ['ttum_support.csv', 'ttum_paired_summary.csv', 'shadow_paired_summary.csv',
                 'shadow_area_support.csv', 'shadow_support_audit.csv', 'map_readiness_contract.json', 'timing_contract.json']:
        copy('analyze/0/' + name)
    for name in ['ttum_rendered_support.csv', 'scenario_contract.json']:
        copy('render_delivery/' + name)
    for name in ['analysis_contract.json', 'inventory/0/checkpoint_inventory.json', 'full_visuals_index.json', 'selected_report_cases.json']:
        copy(name, 'metadata/' + Path(name).name)
    copy('REVIEW_MANIFEST.json', 'metadata/ORIGINAL_REVIEW_MANIFEST.json')
    verification = json.loads((raw.parent / 'verification.json').read_text())
    store_json('verification.json', verification)
    store_json('metadata/review_software.json',dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,
        script_sha256={n:digest(Path(__file__).with_name(n)) for n in ['review_bundle.py','build_review.py']}))
    plan = json.loads((raw / 'plan.json').read_text())
    store_json('metadata/run_identity.json', dict(run_id=plan['run_id'], scope=plan['scope'],
        source_identity=plan['source_identity'], contract_sha256=plan['contract_sha256'],
        cohort_events=len(plan['cohort']), block_events=plan['block_events'],
        scientific_environment=plan['scientific_environment'], task_counts={k:len(v) for k,v in plan['tasks'].items()}))
    status = pd.DataFrame(json.loads((raw / 'status_summary.json').read_text()))
    save(status.groupby(['stage','status'], as_index=False).size(), 'stage_completion_snapshot.csv', ['status_summary.json'])
    checks = dict(review_manifest='PASS', ttum_support_rows=0, original_ttum_intervals_checked=0,
                  legacy_primary_intervals_checked=0, queue_frames_verified=0, latency_summaries_checked=0)

    # Recompute every support count from event rows, keeping eligibility and unknowns explicit.
    ev = pd.concat([read(raw / 'analyze/0/event_time_to_usable_map.csv'),
                    read(raw / 'render_delivery/event_time_to_usable_map_rendered.csv')], ignore_index=True)
    assert not ev.duplicated(EVENT + ['family'] + TT_KEYS).any()
    ev['confirmation_clock_seconds'] = np.select(
        [ev.scene.eq('ideal_cutoff_replay'), ev.scene.eq('measured_compute_replay')],
        [ev.stable_confirmation, ev.stable_confirmation_numeric_ready_seconds],
        default=ev.stable_confirmation_rendered_ready_seconds)
    support, degradation = [], []
    for key, full in ev.groupby(['family'] + TT_KEYS, sort=True):
        g = full[full.targets.ge(5)]
        clock = g.confirmation_clock_seconds
        unknown = g.stable_status.isin(['unknown_missing_frames','tail_right_censored'])
        if key[-1] != 'ideal_cutoff_replay':
            unknown |= clock.isna() & g.stable_confirmation.notna()
        tag = dict(zip(['family'] + TT_KEYS, key))
        for horizon in [5,10,20]:
            hit = clock.notna() & clock.le(horizon)
            support.append(dict(**tag, horizon=horizon, events=len(g), total_cohort_events=len(full),
                ineligible_fewer_than_5=len(full)-len(g), achieved=int(hit.sum()), unknown=int(unknown.sum()),
                fraction_lower=float(hit.mean()) if len(g) else np.nan,
                fraction_upper=float((hit | unknown).mean()) if len(g) else np.nan))
        degradation.append(dict(**tag, events=len(g), any_degradation_after_first=int(g.degradations_after_first.gt(0).sum()),
                                stable_confirmed=int(g.stable_confirmation.notna().sum()),
                                post_confirmation_known_nonpassing_events=int((g.stable_confirmation.notna() & g.last_nonpassing_frame.gt(g.stable_confirmation)).sum())))
    support = pd.DataFrame(support)
    original_support = pd.concat([read(raw / 'analyze/0/ttum_support.csv'), read(raw / 'render_delivery/ttum_rendered_support.csv')])
    sk = ['family'] + TT_KEYS + ['horizon']
    joined = support.merge(original_support, on=sk, suffixes=('_local','_hpc'), validate='one_to_one')
    assert len(joined)==len(support)==len(original_support)
    for c in ['events','total_cohort_events','ineligible_fewer_than_5','achieved','unknown','fraction_lower','fraction_upper']:
        assert np.allclose(joined[c+'_local'], joined[c+'_hpc'], atol=2e-14, rtol=0, equal_nan=True), c
    checks['ttum_support_rows'] = len(support)
    save(support, 'ttum_support_recomputed.csv', ['analyze/0/event_time_to_usable_map.csv','render_delivery/event_time_to_usable_map_rendered.csv'])
    save(pd.DataFrame(degradation), 'post_first_pass_degradation.csv', ['analyze/0/event_time_to_usable_map.csv','render_delivery/event_time_to_usable_map_rendered.csv'])
    columns = EVENT + ['family'] + TT_KEYS + ['targets','stable_confirmation','stable_status','confirmation_clock_seconds',
        'first_observed_pass','stable_onset_retrospective','stable_confirmation_origin_seconds','missing_frames',
        'degradations_after_first','last_nonpassing_frame','stable_confirmation_numeric_ready_seconds','stable_confirmation_rendered_ready_seconds']
    save(ev[columns], 'ttum_event_records.csv.gz', ['analyze/0/event_time_to_usable_map.csv','render_delivery/event_time_to_usable_map_rendered.csv'])

    # Recompute all central-budget paired CIs, including new rendered-scene CIs.
    pair_results = []
    original_summary = read(raw / 'analyze/0/ttum_paired_summary.csv')
    d = ev[ev.family.eq('DiTing') & ev.budget_dex.eq(.2)]
    for family in FAMILIES[1:]:
        b = ev[ev.family.eq(family) & ev.budget_dex.eq(.2)]
        pair = d.merge(b, on=EVENT+TT_KEYS, suffixes=('_a','_b'), validate='one_to_one')
        assert (pair.targets_a==pair.targets_b).all()
        for key, full in pair.groupby(TT_KEYS, sort=True):
            g = full[full.targets_a.ge(5) & full.targets_b.ge(5)]
            a, b = g.confirmation_clock_seconds_a.to_numpy(), g.confirmation_clock_seconds_b.to_numpy()
            both = np.isfinite(a) & np.isfinite(b)
            measures = {'conditional_both_confirmed_delta_seconds':np.where(both,a-b,np.nan)}
            for horizon in [5,10,20]:
                measures['stable_confirmed_fraction_difference_by_'+str(horizon)] = ((np.isfinite(a)&(a<=horizon)).astype(float)-(np.isfinite(b)&(b<=horizon)).astype(float))
            outcomes = Counter(np.select([both,np.isfinite(a),np.isfinite(b)], ['both_confirmed','only_DiTing_confirmed','only_comparator_confirmed'], default='neither_or_unknown'))
            for metric, values in measures.items():
                row = dict(comparator=family, **dict(zip(TT_KEYS,key)), metric=metric,
                           total_cohort_events=len(full), ineligible_events=len(full)-len(g),
                           outcome_counts=json.dumps(dict(outcomes),sort_keys=True), draws=2000, rng_seed=20261007, **interval(values))
                pair_results.append(row)
                match = original_summary[original_summary.comparator.eq(family) & original_summary.metric.eq(metric)]
                for k,v in zip(TT_KEYS,key):
                    match = match[match[k].eq(v)]
                if len(match):
                    assert len(match)==1
                    for col in ['estimate','ci_low','ci_high','events','known_events','effective_replicates']:
                        close(row[col],match.iloc[0][col],('ttum',family,key,metric,col))
                    checks['original_ttum_intervals_checked'] += 1
    save(pd.DataFrame(pair_results), 'ttum_paired_b020_recomputed.csv', ['derived/ttum_event_records.csv.gz'])

    # Complete rendered event summaries are retained compressed, with both event-equal and target-weighted summaries.
    shadow = read(raw / 'render_delivery/shadow_rendered_event_summary.csv')
    assert (shadow.timely+shadow.failures+shadow.unknown==shadow.affected_targets).all()
    assert shadow.area_km2.isna().all()
    save(shadow, 'shadow_rendered_events.csv.gz', ['render_delivery/shadow_rendered_event_summary.csv'])
    keys = ['family','geometry_protocol','panel','threshold_logpga','p_alarm','tau','scene']
    rows = []
    for key,g in shadow.groupby(keys, sort=True):
        affected = int(g.affected_targets.sum()); negatives = int(g.false_positive_denominator.sum())
        alarmed = int(g.alarmed_targets.sum())
        affected_alarmed = np.nansum(g.alarm_precision*g.alarmed_targets)
        rows.append(dict(**dict(zip(keys,key)), events=len(g), events_with_affected_targets=int(g.affected_targets.gt(0).sum()),
            affected_targets=affected, timely=int(g.timely.sum()), failures=int(g.failures.sum()), unknown=int(g.unknown.sum()),
            event_equal_failure_lower=g.failure_fraction_lower.mean(), event_equal_failure_upper=g.failure_fraction_upper.mean(),
            pooled_failure_lower=g.failures.sum()/affected if affected else np.nan,
            pooled_failure_upper=(g.failures.sum()+g.unknown.sum())/affected if affected else np.nan,
            complete_negative_targets=negatives, false_positives=int(g.false_positives.sum()),
            pooled_false_positive_rate=g.false_positives.sum()/negatives if negatives else np.nan,
            incomplete_negative_targets=int(g.unaffected_incomplete_observation.sum()),
            known_false_alarms_in_incomplete_observation=int(g.known_false_alarms_in_incomplete_observation.sum()),
            event_equal_false_positive_rate=g.false_positive_rate.mean(),
            pooled_alarm_precision=affected_alarmed/alarmed if alarmed else np.nan,
            event_equal_brier_at_20s=g.mean_station_brier_at_20s.mean()))
    save(pd.DataFrame(rows), 'shadow_rendered_summary.csv', ['render_delivery/shadow_rendered_event_summary.csv'])
    shadow_pairs = []
    focus = shadow[shadow.threshold_logpga.eq(-1.2)&shadow.p_alarm.eq(.5)&shadow.tau.eq(3)]
    for family in FAMILIES[1:]:
        pair = focus[focus.family.eq('DiTing')].merge(focus[focus.family.eq(family)], on=EVENT+keys[1:], suffixes=('_a','_b'),validate='one_to_one')
        assert (pair.affected_targets_a==pair.affected_targets_b).all()
        for panel,g in pair.groupby('panel',sort=True):
            for bound in ['lower','upper']:
                values = g['failure_fraction_'+bound+'_a']-g['failure_fraction_'+('upper' if bound=='lower' else 'lower')+'_b']
                shadow_pairs.append(dict(comparator=family,panel=panel,threshold_logpga=-1.2,p_alarm=.5,tau=3,
                    metric='delta_shadow_'+bound+'_bound',**interval(values)))
    save(pd.DataFrame(shadow_pairs), 'shadow_rendered_example_pairwise.csv', ['render_delivery/shadow_rendered_event_summary.csv'])

    # Validate the shared-render FIFO algebra for every available frame, then summarize recorded clocks.
    queue = read(raw / 'render_delivery/queue_replay.csv')
    for _,g in queue.groupby(EVENT+['family'],sort=False):
        g = g.sort_values('cutoff_seconds')
        previous = np.r_[-np.inf,g.rendered_ready_seconds.to_numpy()[:-1]]
        assert np.allclose(g.compute_start_seconds, np.maximum(g.cutoff_seconds,previous),atol=3e-12,rtol=0)
        assert np.allclose(g.numeric_ready_seconds,g.compute_start_seconds+g.service_seconds,atol=3e-12,rtol=0)
        assert np.allclose(g.rendered_ready_seconds,g.numeric_ready_seconds+g.render_seconds,atol=3e-12,rtol=0)
    assert queue.groupby(EVENT+['cutoff_seconds']).render_seconds.nunique().eq(1).all()
    checks['queue_frames_verified'] = len(queue)
    save(queue, 'queue_event_frames.csv.gz', ['render_delivery/queue_replay.csv'])
    qs = []
    for family,g in queue.groupby('family',sort=True):
        for cohort,part in [('all_frames',g),('cutoff_20s',g[g.cutoff_seconds.eq(20)])]:
            for metric in ['service_seconds','render_seconds','queue_wait_seconds','numeric_ready_seconds','rendered_ready_seconds']:
                qs.append(dict(family=family,cohort=cohort,metric=metric,frames=len(part),events=part[EVENT].drop_duplicates().shape[0],
                               p50=part[metric].median(),p95=part[metric].quantile(.95),maximum=part[metric].max()))
    save(pd.DataFrame(qs), 'queue_quantiles.csv', ['render_delivery/queue_replay.csv'])

    # Independently check all eight saved latency medians/P95s from raw synchronized repeats.
    latency = []
    for i in range(4):
        prefix = 'latency/'+str(i)+'/'
        for name in ['latency_frames.csv','latency_summary.csv','runtime_environment.json','timing_contract.json']:
            copy(prefix+name)
        f = read(raw / (prefix+'latency_frames.csv'))
        latency.append(f)
        saved = read(raw / (prefix+'latency_summary.csv'))
        for scope,g in f[f.phase.eq('warm_measured')].groupby('scope'):
            v = saved[saved.scope.eq(scope)].iloc[0]
            assert len(g)==v['count']==108
            close(g.numeric_service_seconds.median(),v.p50_seconds,'latency median')
            close(g.numeric_service_seconds.quantile(.95),v.p95_seconds,'latency P95')
            close(g.numeric_service_seconds.max(),v.maximum_seconds,'latency maximum')
            checks['latency_summaries_checked'] += 1
    latency = pd.concat(latency,ignore_index=True)
    assert latency.synchronized.all() and set(latency.family)==set(FAMILIES)
    pivot = latency.pivot(index=EVENT+['geometry_protocol','elapsed_time','scope','phase','repeat'],columns='family',values=['K','station_Q','grid_Q'])
    assert not pivot.isna().any().any()
    for c in ['K','station_Q','grid_Q']:
        assert pivot[c].nunique(axis=1).eq(1).all()
    latency_rows = []
    for (family,scope),g in latency[latency.phase.eq('warm_measured')].groupby(['family','scope']):
        latency_rows.append(dict(family=family,scope=scope,repeats=len(g),events=g[EVENT].drop_duplicates().shape[0],
            frame_conditions=len(g.drop_duplicates(EVENT+['geometry_protocol','elapsed_time'])),warmup=2,repeats_per_condition=3,
            p50_seconds=g.numeric_service_seconds.median(),p95_seconds=g.numeric_service_seconds.quantile(.95),
            maximum_seconds=g.numeric_service_seconds.max(),cold_load_seconds=g.cold_load_seconds.iloc[0],
            render_p50_seconds=g.render_seconds.median(),device_names=';'.join(sorted(g.device_name.unique())),
            K_min=int(g.K.min()),K_max=int(g.K.max()),station_Q_min=int(g.station_Q.min()),station_Q_max=int(g.station_Q.max()),
            grid_Q_min=int(g.grid_Q.min()),grid_Q_max=int(g.grid_Q.max())))
    save(pd.DataFrame(latency_rows),'latency_recomputed.csv',['latency/0-3/latency_frames.csv'])

    # A previously verified historical event-cell artifact provides an independent legacy ON MAE bootstrap check.
    historic = REPO / 'reports/fe01_a01_hpc_results_20261021/event_cells_noninput.csv.gz'
    historical_manifest=REPO/'reports/fe01_a01_hpc_results_20261021/source_metadata/A01_REVIEW_MANIFEST.json'
    old_pins=json.loads(historical_manifest.read_text())['files_sha256']
    current_exports=json.loads((raw/'analyze/0/legacy_with_physical_covariates/source_export_verification.json').read_text())['exports']
    matches=[]
    for record in current_exports:
        paths=[k for k,v in old_pins.items() if v==record['prediction_sha256']]
        assert len(paths)==1
        matches.append(dict(**record,historical_export_path=paths[0]))
    store_json('metadata/legacy_export_identity_crosscheck.json',dict(status='PASS',matches=matches,
        historical_review_manifest_sha256=digest(historical_manifest)))
    f = read(historic)
    f = f[f['mode'].eq('ON')].copy()
    assert f.seed.eq(42).all() and f.split.eq('validation').all()
    cols = EVENT+['family','checkpoint_policy','epoch','geometry_protocol','elapsed_time','targets','abs_error_sum','squared_error_sum','nll_sum','crps_sum','brier_sum','covered95_sum']
    save(f[cols], 'legacy_on_event_cells.csv.gz', ['historical:reports/fe01_a01_hpc_results_20261021/event_cells_noninput.csv.gz'])
    store_json('metadata/historical_event_cells_source.json',dict(path=str(historic.relative_to(REPO)),sha256=digest(historic),
        origin='Prior verified A01 saved ON exports; not new fixed-query replay.',
        crosscheck='Both policies/4 families reproduce fresh V4 legacy noninput primary MAE; six paired MAE CIs checked below.'))
    metric = read(raw/'analyze/0/legacy_with_physical_covariates/cross_encoder_metrics.csv')
    saved_pairs = read(raw/'analyze/0/legacy_with_physical_covariates/paired_effects.csv')
    legacy_checks = []
    for policy in ['selected','epoch12']:
        arrays = {}
        for family in FAMILIES:
            g = f[f.family.eq(family)&f.checkpoint_policy.eq(policy)]
            pivot = g.pivot(index=EVENT,columns=['geometry_protocol','elapsed_time'],values=['targets','abs_error_sum']).fillna(0).sort_index()
            columns = pd.MultiIndex.from_product([['normal','random'],[1.,3.,5.,10.,20.]])
            den = pivot['targets'].reindex(columns=columns,fill_value=0).to_numpy()
            num = pivot['abs_error_sum'].reindex(columns=columns,fill_value=0).to_numpy()
            arrays[family] = (num,den)
            saved = metric[metric.family.eq(family)&metric.checkpoint_policy.eq(policy)&metric.target_group.eq('noninput')&metric.endpoint.eq('primary_10cell')].iloc[0]
            assert int(den.sum())==saved.requests==139440 and len(den)==1310
            close((num.sum(0)/den.sum(0)).mean(),saved.equal_cell_mae,'historical point estimate')
        a,den = arrays['DiTing']
        for family in FAMILIES[1:]:
            b,other = arrays[family];assert np.array_equal(den,other)
            delta=a-b;rng=np.random.default_rng(20261007);dist=[]
            for start in range(0,2000,100):
                ids=rng.integers(0,len(delta),size=(100,len(delta)))
                n=den[ids].sum(axis=1);v=delta[ids].sum(axis=1)
                valid=(n>0).all(axis=1);dist.extend((v[valid]/n[valid]).mean(axis=1))
            row=dict(checkpoint_policy=policy,comparator=family,metric='mae',target_group='noninput',endpoint='primary_10cell',
                events=1310,requests=139440,estimate=(delta.sum(0)/den.sum(0)).mean(),ci_low=np.quantile(dist,.025),ci_high=np.quantile(dist,.975),effective_replicates=len(dist))
            ref=saved_pairs[saved_pairs.comparator.eq(family)&saved_pairs.checkpoint_policy.eq(policy)&saved_pairs.metric.eq('mae')&saved_pairs.target_group.eq('noninput')&saved_pairs.endpoint.eq('primary_10cell')].iloc[0]
            for col in ['estimate','ci_low','ci_high','effective_replicates']:close(row[col],ref[col],('legacy CI',policy,family,col))
            legacy_checks.append(row);checks['legacy_primary_intervals_checked']+=1
    save(pd.DataFrame(legacy_checks), 'legacy_primary_pairwise_recheck.csv', ['derived/legacy_on_event_cells.csv.gz'])

    # All twelve original metadata-selected cases, same two predeclared keyframes; no error-based selection.
    selected = json.loads((raw/'selected_report_cases.json').read_text())
    assert len(selected)==12
    for case in selected:
        directory = Path(case['href']).parent
        for name in ['keyframe_05.png','keyframe_20.png','frame_sources.json','render_latency.csv']:
            copy('cases/'+str(directory/name),'cases/'+str(directory/name))
    methods = ['map_readiness.py','shadow.py','event_statistics.py','clocks.py','latency.py','delivery_analysis.py',
               'comparison.py','statistics.py','stratification.py','interactions.py','stages.py','plotting.py']
    pins = json.loads((raw/'source_identity.json').read_text())['files_sha256']
    method_pins = {}
    for name in methods:
        source = REPO/'fe01_a01_compare'/name
        assert digest(source)==pins['fe01_a01_compare/'+name]
        target=out/'method_sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        method_pins[name]=digest(source)
    store_json('method_sources/SOURCE_SHA256.json',method_pins)
    store_json('local_numerical_checks.json',checks)
    store_json('SOURCE_MANIFEST.json',dict(original_archive_sha256=verification['archive']['sha256'],
        copied_sources=sources,derived_tables=derived,scope='Neutral evidence; Codex independent interpretation excluded.'))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    build(args.raw,args.output)
