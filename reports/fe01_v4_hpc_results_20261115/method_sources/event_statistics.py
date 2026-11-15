"""Event-equal warning/readiness comparisons, distinct from target/cell endpoints."""
import numpy as np
import pandas as pd


def scalar_interval(values,draws=2000):
    x=np.asarray(values,float)
    good=np.isfinite(x)
    if not good.any():
        return dict(estimate=None,ci_low=None,ci_high=None,events=len(x),known_events=0,effective_replicates=0,draws=draws)
    rng=np.random.default_rng(20261007)
    draws_out=[]
    for start in range(0,draws,100):
        ids=rng.integers(0,len(x),size=(min(100,draws-start),len(x)))
        sampled=x[ids]
        valid=np.isfinite(sampled).any(1)
        draws_out.extend(np.nanmean(sampled[valid],axis=1).tolist())
    return dict(estimate=float(np.nanmean(x)),ci_low=float(np.quantile(draws_out,.025)),ci_high=float(np.quantile(draws_out,.975)),
                events=len(x),known_events=int(good.sum()),effective_replicates=len(draws_out),draws=draws,
                conditioning='reported available event subset; all outcomes/unknown counts retained',
                interval='fixed model event cluster; exploratory uncorrected')


def ttum_summary(pairs):
    keys=['comparator','geometry_protocol','panel','budget_dex','gate','scene']
    out=[]
    for key,g in pairs.groupby(keys):
        base=dict(zip(keys,key))
        total=len(g)
        g=g[g.paired_eligible]
        base.update(total_cohort_events=total,ineligible_events=total-len(g))
        counts=g.outcome.value_counts().to_dict()
        out.append(dict(**base,metric='conditional_both_confirmed_delta_seconds',
                        outcome_counts=str(counts),**scalar_interval(g.delta_confirmation_seconds)))
        timing='stable_confirmation_rendered_ready_seconds' if key[-1]=='measured_rendered_map_replay' else 'stable_confirmation_numeric_ready_seconds' if key[-1]=='measured_compute_replay' else 'stable_confirmation'
        a=g.get(timing+'_diting',pd.Series(np.nan,index=g.index))
        b=g.get(timing+'_comparator',pd.Series(np.nan,index=g.index))
        for t in (5,10,20):
            # Determinate lower CDF: no missing/unsupported case is counted as a success.
            diff=(a.notna()&a.le(t)).astype(float)-(b.notna()&b.le(t)).astype(float)
            out.append(dict(**base,metric='stable_confirmed_fraction_difference_by_'+str(t),
                            outcome_counts=str(counts),**scalar_interval(diff)))
    return pd.DataFrame(out)


def shadow_summary(pairs):
    keys=['comparator','geometry_protocol','panel','threshold_logpga','p_alarm','tau','scene']
    out=[]
    for key,g in pairs.groupby(keys):
        base=dict(zip(keys,key))
        for metric in ('delta_shadow_lower_bound','delta_shadow_upper_bound'):
            out.append(dict(**base,metric=metric,**scalar_interval(g[metric])))
    return pd.DataFrame(out)
