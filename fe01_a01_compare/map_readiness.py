"""Retrospective station-supported TTUM; full fixed panels, no success filtering."""
import numpy as np
import pandas as pd
from .io import require


def frame_metrics(frame, expected_ids):
    expected_ids = set(expected_ids)
    f = frame[frame.station_id.isin(expected_ids)]
    out = dict(targets=len(expected_ids), status='Unknown', station_pass=None, field_pass=None,
               reason='',mae=None,bias=None,centered_rmse=None,truth_range=None,prediction_range=None,
               range_error=None,range_ratio=None)
    if len(expected_ids) < 5:
        return dict(out, reason='fewer_than_5_fixed_targets')
    if len(f) != len(expected_ids) or f.station_id.duplicated().any() or set(f.station_id) != expected_ids:
        return dict(out, reason='missing_or_duplicate_fixed_targets')
    if f.status.eq('no_input').all():
        return dict(out,status='NO_SERVICE',station_pass=False,field_pass=False,reason='known_no_input; map not delivered')
    if not f.status.eq('supported').all() or not np.isfinite(f[['prediction','truth']]).all().all():
        return dict(out, reason='missing_prediction_or_truth')
    e = f.prediction.to_numpy()-f.truth.to_numpy()
    tr = float(np.quantile(f.truth,.95)-np.quantile(f.truth,.05))
    pr = float(np.quantile(f.prediction,.95)-np.quantile(f.prediction,.05))
    return dict(out, status='OK', reason='', mae=float(np.abs(e).mean()), bias=float(e.mean()),
                centered_rmse=float(np.sqrt(np.mean((e-e.mean())**2))),
                truth_range=tr, prediction_range=pr, range_error=abs(pr-tr),
                range_ratio=pr/tr if tr >= .05 else None, errors=e)


def gate(metrics, budget):
    m = {k:v for k,v in metrics.items() if k != 'errors'}
    m['budget_dex'] = budget
    m['fraction_abs_error_within_2b']=None
    if metrics['status'] != 'OK':
        return m
    fraction = float(np.mean(np.abs(metrics['errors']) <= 2*budget))
    passed = metrics['mae'] <= budget and abs(metrics['bias']) <= budget/2 and fraction >= .9
    m.update(fraction_abs_error_within_2b=fraction, station_pass=bool(passed),
             field_pass=bool(passed and metrics['centered_rmse'] <= budget and metrics['range_error'] <= budget))
    return m


def times_to_ready(frames, kind='station_pass', times=tuple(range(1,21))):
    lookup = {float(f['elapsed_time']): f for f in frames}
    first = onset = confirm = None
    run, previous, degrades = [], None, 0
    unknown = 0
    last_bad = None
    for t in times:
        f = lookup.get(float(t))
        raw = f.get(kind) if f else None
        passed = raw if isinstance(raw, (bool, np.bool_)) else None
        if passed is None:
            unknown += 1
            run = []
            previous = None
            continue
        if passed:
            if first is None:
                first = t
            run.append(t)
            if len(run) >= 3 and run[-1]-run[-3] == 2 and onset is None:
                onset, confirm = run[-3], run[-1]
        else:
            if first is not None and previous is True:
                degrades += 1
            last_bad = t
            run = []
        previous = bool(passed)
    tail = bool(run) and confirm is None
    status = 'stable_confirmed' if confirm is not None else 'tail_right_censored' if tail else 'unknown_missing_frames' if unknown else 'not_reached_within_20s'
    return dict(first_observed_pass=first, stable_onset_retrospective=onset, stable_confirmation=confirm,
                first_status='first_observed_at_1s_left_censored' if first == 1 else 'observed' if first is not None else 'Unknown' if unknown else 'not_reached',
                stable_status=status, missing_frames=unknown, degradations_after_first=degrades,
                last_nonpassing_frame=last_bad, stable_by_5=confirm is not None and confirm <= 5,
                stable_by_10=confirm is not None and confirm <= 10, stable_by_20=confirm is not None and confirm <= 20,
                retrospective_truth=True, engineering_acceptance=None)


def evaluate(frame, budgets=(.15,.2,.25)):
    keys = ['family','dataset_id','event_id','geometry_protocol']
    metrics, ttum = [], []
    for key, event in frame.groupby(keys, dropna=False):
        base = dict(zip(keys,key))
        panels = {'fixed_all_targets': set(event.station_id),
                  'fixed_never_input_targets': set(event.loc[event.fixed_never_input.astype(str).str.lower().eq('true'),'station_id']),
                  'fixed_high_PGA_ge_minus1p2': set(event.loc[event.truth.ge(-1.2),'station_id'])}
        for panel, ids in panels.items():
            for b in budgets:
                rows = []
                for t in range(1,21):
                    f = event[event.elapsed_time.eq(t)]
                    m = gate(frame_metrics(f, ids), b)
                    m.update(**base, panel=panel, elapsed_time=t)
                    rows.append(m)
                    metrics.append(m)
                for kind in ('station_pass','field_pass'):
                    result = times_to_ready(rows,kind)
                    result.update(**base, panel=panel, budget_dex=b, gate=kind, targets=len(ids), scene='ideal_cutoff_replay')
                    offset = event.get('first_p_minus_origin_seconds', pd.Series(dtype=float)).dropna()
                    result['stable_confirmation_origin_seconds'] = float(result['stable_confirmation']+offset.iloc[0]) if result['stable_confirmation'] is not None and len(offset) else None
                    ttum.append(result)
                    if 'numeric_ready_seconds' in event:
                        t = result['stable_confirmation']
                        delivered = event[event.elapsed_time.eq(t)].numeric_ready_seconds.dropna() if t is not None else pd.Series(dtype=float)
                        measured = dict(result, scene='measured_compute_replay',
                                        stable_confirmation_numeric_ready_seconds=float(delivered.max()) if len(delivered) else None,
                                        communication_latency='UNKNOWN')
                        rendered = event[event.elapsed_time.eq(t)].get('rendered_ready_seconds', pd.Series(dtype=float)).dropna() if t is not None else pd.Series(dtype=float)
                        measured['stable_confirmation_rendered_ready_seconds'] = float(rendered.max()) if len(rendered) else None
                        measured['delivery_clock_status'] = 'measured_numeric_map' if len(delivered) else 'UNKNOWN_no_complete_map_clock'
                        ttum.append(measured)
                    if 'rendered_ready_seconds' in event:
                        t=result['stable_confirmation']
                        rendered=event[event.elapsed_time.eq(t)].rendered_ready_seconds.dropna() if t is not None else pd.Series(dtype=float)
                        ttum.append(dict(result,scene='measured_rendered_map_replay',
                            stable_confirmation_rendered_ready_seconds=float(rendered.max()) if len(rendered) else None,
                            delivery_clock_status='measured_compute_plus_shared_render_FIFO_scenario' if len(rendered) else 'UNKNOWN',
                            communication_latency='UNKNOWN'))
    return pd.DataFrame(metrics), pd.DataFrame(ttum)


def population_summary(ttum):
    keys = ['family','geometry_protocol','panel','budget_dex','gate','scene']
    result=[]
    for k,g in ttum.groupby(keys):
        total_events=len(g)
        g=g[g.targets.ge(5)]
        complete=g.stable_status.eq('stable_confirmed')
        unknown=g.stable_status.isin(['unknown_missing_frames','tail_right_censored'])
        for horizon in (5,10,20):
            timing='stable_confirmation_rendered_ready_seconds' if k[-1]=='measured_rendered_map_replay' else 'stable_confirmation_numeric_ready_seconds' if k[-1]=='measured_compute_replay' else 'stable_confirmation'
            times=g[timing] if timing in g else pd.Series(np.nan,index=g.index)
            hit=times.notna() & times.le(horizon)
            if k[-1]!='ideal_cutoff_replay':
                unknown=unknown | times.isna() & g.stable_confirmation.notna()
            result.append(dict(zip(keys,k), horizon=horizon, events=len(g),total_cohort_events=total_events,
                               ineligible_fewer_than_5=total_events-len(g),achieved=int(hit.sum()),
                               unknown=int(unknown.sum()), fraction_lower=float(hit.mean()) if len(g) else None,
                               fraction_upper=float((hit|unknown).mean()) if len(g) else None,
                               median_confirmation_all_events=float(times.dropna().sort_values().iloc[(len(g)-1)//2])
                               if len(g) and hit.sum() >= (len(g)+1)//2 else None,
                               denominator='all fixed eligible events; missing/unsupported retained'))
    return pd.DataFrame(result)


def pair_events(ttum):
    keys=['dataset_id','event_id','geometry_protocol','panel','budget_dex','gate','scene']
    out=[]
    d=ttum[ttum.family.eq('DiTing')]
    for other in ('PhaseNet','EQT','TEAM'):
        j=d.merge(ttum[ttum.family.eq(other)], on=keys, suffixes=('_diting','_comparator'), how='outer', validate='one_to_one')
        for r in j.to_dict('records'):
            timing='stable_confirmation_rendered_ready_seconds' if r['scene']=='measured_rendered_map_replay' else 'stable_confirmation_numeric_ready_seconds' if r['scene']=='measured_compute_replay' else 'stable_confirmation'
            a,b=r.get(timing+'_diting'),r.get(timing+'_comparator')
            both=pd.notna(a) and pd.notna(b)
            r.update(comparator=other, delta_confirmation_seconds=a-b if both else None,
                     paired_eligible=r.get('targets_diting',0)>=5 and r.get('targets_comparator',0)>=5,
                     outcome='both_confirmed' if both else 'only_DiTing_confirmed' if pd.notna(a) else 'only_comparator_confirmed' if pd.notna(b) else 'neither_or_unknown',
                     comparison_scope='time difference conditional on both determinable; all outcomes retained')
            out.append(r)
    return pd.DataFrame(out)
