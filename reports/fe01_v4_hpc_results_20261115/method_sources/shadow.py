"""Latched MDN alarms, common affected population, and false alarm costs."""
import json
import numpy as np
import pandas as pd
from scipy.special import logsumexp, ndtr
from .io import require


def probabilities(frame, threshold):
    if 'mdn_logits' not in frame:
        if not frame.status.eq('supported').any():
            return np.full(len(frame),np.nan)
        raise ValueError('Raw MDN logits required; mean crossing is not an alarm probability')
    result = np.full(len(frame), np.nan)
    good = frame.status.eq('supported').to_numpy()
    if good.any():
        part=frame.loc[good]
        logits=np.asarray([json.loads(x) for x in part.mdn_logits])
        mu=np.asarray([json.loads(x) for x in part.mdn_mu])
        sigma=np.asarray([json.loads(x) for x in part.mdn_sigma])
        require(np.isfinite(logits).all() and np.isfinite(mu).all() and np.isfinite(sigma).all() and (sigma>0).all(), 'Invalid full MDN')
        w=np.exp(logits-logsumexp(logits,axis=-1,keepdims=True))
        result[good]=(w*ndtr((mu-threshold)/sigma)).sum(-1)
    return result


def classify(affected, crossing, crossing_status, predictions, p_alarm, tau, scene='ideal_cutoff_replay', horizon=20):
    """t in the same first-P clock; missing exports never mean 'not alarmed'."""
    f=predictions.sort_values('elapsed_time')
    arrival='elapsed_time' if scene=='ideal_cutoff_replay' else 'rendered_ready_seconds' if scene=='measured_rendered_map_replay' else 'numeric_ready_seconds'
    if arrival not in f:
        return dict(status='unknown_clock', alert_time=None, lead=None, alarmed=None,complete_observation=False,scene=scene,communication_latency='UNKNOWN')
    valid=f.status.eq('supported') & f.probability.notna() & f[arrival].notna()
    alerts=f[valid & f.probability.ge(p_alarm)]
    alert=float(alerts[arrival].min()) if len(alerts) else None  # delivered order, not cutoff order
    complete=set(f.elapsed_time)==set(range(1,horizon+1)) and not f.elapsed_time.duplicated().any() and f.status.isin(['supported','no_input']).all()
    if scene!='ideal_cutoff_replay' and f.loc[f.status.eq('supported'),arrival].isna().any():
        complete=False
    base=dict(alert_time=alert, alarmed=alert is not None, lead=None, complete_observation=bool(complete),
              scene=scene, communication_latency='UNKNOWN' if scene!='ideal_cutoff_replay' else 'zero_assumption')
    if not affected:
        return dict(base,status='false_alarm' if alert is not None else 'true_negative' if complete else 'unknown_negative')
    if crossing_status!='exact_within_record' or pd.isna(crossing):
        return dict(base,status='unknown_crossing')
    unknown_alert_clock=f.status.eq('supported') & f.probability.ge(p_alarm) & f[arrival].isna()
    if alert is not None:
        lead=float(crossing-alert)
        if unknown_alert_clock.any() and not (lead>0 and lead>=tau):
            return dict(base,status='unknown_clock')
        return dict(base,lead=lead,status='timely' if lead>0 and lead>=tau else 'positive_but_short' if lead>0 else 'late_or_zero')
    if f.loc[f.status.eq('supported'),arrival].isna().any():
        return dict(base,status='unknown_clock')
    deadline=float(crossing-tau)
    maxdel=float(f.loc[f[arrival].notna(),arrival].max()) if f[arrival].notna().any() else -np.inf
    if deadline>maxdel or deadline>horizon:
        return dict(base,status='unknown_prediction_horizon')
    if deadline<1:
        return dict(base,status='no_service_before_start', policy='no service before firstP+1s; no physical impossibility claim')
    needed=set(range(1, min(horizon,int(np.floor(deadline)))+1))
    covered=f[f.elapsed_time.isin(needed)]
    if len(covered)!=len(needed):
        return dict(base,status='missing_export')
    if covered.status.eq('unsupported_history').any():
        return dict(base,status='unsupported_history')
    if not covered.status.isin(['supported','no_input']).all():
        return dict(base,status='missing_export')
    if len(covered) and covered.status.eq('no_input').all():
        return dict(base,status='no_input')
    return dict(base,status='not_alerted_by_deadline')


def evaluate(frame, truth, contract):
    outcomes=[]
    keys=['family','dataset_id','event_id','geometry_protocol','station_id']
    require(not truth.duplicated(['dataset_id','event_id','station_id','threshold_logpga']).any(),'Duplicate crossing truth')
    for threshold in contract['research_logpga_thresholds']:
        f=frame.copy()
        f['probability']=probabilities(f,threshold)
        tr=truth[truth.threshold_logpga.eq(threshold)]
        for key,g in f.groupby(keys):
            base=dict(zip(keys,key))
            match=tr[tr.dataset_id.eq(base['dataset_id']) & tr.event_id.eq(base['event_id']) & tr.station_id.eq(base['station_id'])]
            require(len(match)==1,'Missing common threshold truth station')
            r=match.iloc[0]
            valid_probability=g.probability.dropna()
            score_count=len(valid_probability)
            brier_mean=float(np.mean((valid_probability.to_numpy()-float(bool(r.affected)))**2)) if score_count else None
            last=g[g.elapsed_time.eq(20)].probability.dropna()
            brier_last=float((last.iloc[0]-float(bool(r.affected)))**2) if len(last) else None
            for panel in ('fixed_all_targets','fixed_never_input_targets'):
                if panel=='fixed_never_input_targets' and not g.fixed_never_input.astype(str).str.lower().eq('true').all():
                    continue
                scenes=['ideal_cutoff_replay']
                if 'numeric_ready_seconds' in g:scenes.append('measured_compute_replay')
                if 'rendered_ready_seconds' in g:scenes.append('measured_rendered_map_replay')
                for scene in scenes:
                    for p in contract['research_probability_thresholds']:
                        for tau in contract['minimum_lead_seconds']:
                            outcome=classify(bool(r.affected),r.crossing_first_p_seconds,r.status,g,p,tau,scene)
                            outcome={key:outcome.get(key) for key in ('alert_time','alarmed','lead','complete_observation','scene','communication_latency','status','policy')}
                            outcomes.append(dict(**base,panel=panel,threshold_logpga=threshold,p_alarm=p,tau=tau,
                                                 affected=bool(r.affected),truth_status=r.status,
                                                 brier_available_frame_mean=brier_mean,brier_at_20s=brier_last,
                                                 probability_scored_frames=score_count,
                                                 latitude=float(g.latitude.iloc[0]),longitude=float(g.longitude.iloc[0]),**outcome))
    o=pd.DataFrame(outcomes)
    summaries=[]
    group=['family','dataset_id','event_id','geometry_protocol','panel','threshold_logpga','p_alarm','tau','scene']
    bad={'positive_but_short','late_or_zero','not_alerted_by_deadline','no_input','no_service_before_start'}
    for k,g in o.groupby(group,dropna=False):
        h=g[g.affected]
        n=len(h)
        nbad=int(h.status.isin(bad).sum())
        ntimely=int(h.status.eq('timely').sum())
        unknown=n-nbad-ntimely
        negative=g[~g.affected & g.complete_observation.eq(True)]
        fp=int(negative.alarmed.eq(True).sum())
        alarm=g[g.alarmed.eq(True)]
        summaries.append(dict(zip(group,k), affected_targets=n, timely=ntimely, failures=nbad, unknown=unknown,
                              failure_fraction_lower=nbad/n if n else None,
                              failure_fraction_upper=(nbad+unknown)/n if n else None,
                              false_positive_denominator=len(negative), false_positives=fp,
                              unaffected_targets=int((~g.affected).sum()),
                              unaffected_incomplete_observation=int((~g.affected).sum())-len(negative),
                              known_false_alarms_in_incomplete_observation=int((~g.affected & ~g.complete_observation & g.alarmed.eq(True)).sum()),
                              false_positive_rate=fp/len(negative) if len(negative) else None,
                              timely_fraction_lower=ntimely/n if n else None,
                              timely_fraction_upper=(ntimely+unknown)/n if n else None,
                              alarm_precision=float(alarm.affected.mean()) if len(alarm) else None,
                              mean_station_brier_available_frames=float(g.brier_available_frame_mean.mean()) if g.brier_available_frame_mean.notna().any() else None,
                              mean_station_brier_at_20s=float(g.brier_at_20s.mean()) if g.brier_at_20s.notna().any() else None,
                              probability_targets_at_20s=int(g.brier_at_20s.notna().sum()),
                              alarmed_targets=len(alarm), event_scope='within catalog spatial false alarm; not daily continuous rate',
                              area_km2=None, area_status='NA_no_common_independent_spatial_support'))
    return o,pd.DataFrame(summaries)
