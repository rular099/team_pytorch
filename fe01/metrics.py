"""Exact Gaussian mixture scoring in log10(m/s²), not a single-Gaussian proxy."""
import numpy as np
import pandas as pd
from scipy.special import logsumexp, ndtr


def decode(mdn, normalization):
    mdn = np.asarray(mdn,dtype=np.float64)
    weights = np.exp(mdn[...,0] - logsumexp(mdn[...,0],axis=-1,keepdims=True))
    mean = mdn[...,1]*normalization['std']+normalization['mean']
    sigma = mdn[...,2]*normalization['std']
    return weights,mean,sigma


def mixture_cdf(x,w,mu,sigma):
    return np.sum(w*ndtr((np.asarray(x)[...,None]-mu)/sigma),axis=-1)


def quantile(p,w,mu,sigma):
    if not 0 < p < 1:
        raise ValueError('Quantile probability must be between zero and one')
    lo = np.min(mu-12*sigma,axis=-1)
    hi = np.max(mu+12*sigma,axis=-1)
    for _ in range(70):
        mid = (lo+hi)/2
        below = mixture_cdf(mid,w,mu,sigma)<p
        lo = np.where(below,mid,lo)
        hi = np.where(below,hi,mid)
    return (lo+hi)/2


def normal_absolute(mu,sigma):
    z=mu/sigma
    return 2*sigma/np.sqrt(2*np.pi)*np.exp(-z*z/2)+mu*(2*ndtr(z)-1)


def score_rows(truth,w,mu,sigma,threshold=-1.2):
    truth=np.asarray(truth,dtype=np.float64)
    mean=(w*mu).sum(-1)
    variance=(w*(sigma*sigma+mu*mu)).sum(-1)-mean*mean
    log_density=-np.log(sigma*np.sqrt(2*np.pi))-(truth[...,None]-mu)**2/(2*sigma*sigma)
    nll=-logsumexp(np.log(w.clip(1e-300))+log_density,axis=-1)
    crps=(w*normal_absolute(truth[...,None]-mu,sigma)).sum(-1)
    difference=mu[..., :,None]-mu[...,None,:]
    pair_scale=np.sqrt(sigma[..., :,None]**2+sigma[...,None,:]**2)
    crps-=0.5*(w[..., :,None]*w[...,None,:]*normal_absolute(difference,pair_scale)).sum(axis=(-2,-1))
    exceed=1-mixture_cdf(threshold,w,mu,sigma)
    low,high=quantile(.025,w,mu,sigma),quantile(.975,w,mu,sigma)
    q16,q84=quantile(.158655,w,mu,sigma),quantile(.841345,w,mu,sigma)
    # E[linear PGA] differs from 10**E[log PGA].
    linear_mean=(w*np.exp(np.log(10)*mu+0.5*np.log(10)**2*sigma*sigma)).sum(-1)
    return dict(prediction=mean,error=mean-truth,nll=nll,crps=crps,
                brier=(exceed-(truth>threshold))**2,exceedance_probability=exceed,
                predictive_sigma=np.sqrt(np.maximum(variance,0)),pit=mixture_cdf(truth,w,mu,sigma),
                q025=low,q975=high,q16=q16,q84=q84,
                covered95=(truth>=low)&(truth<=high),width95=high-low,
                covered68=(truth>=q16)&(truth<=q84),
                covered_mean_sigma=np.abs(mean-truth)<=np.sqrt(np.maximum(variance,0)),
                covered_mean_2sigma=np.abs(mean-truth)<=2*np.sqrt(np.maximum(variance,0)),
                linear_mixture_mean_mps2=linear_mean,linear_mixture_median_mps2=10**quantile(.5,w,mu,sigma))


def summarize(frame):
    valid=frame['status'].eq('supported') if 'status' in frame else np.ones(len(frame),dtype=bool)
    valid &= np.isfinite(frame.get('prediction',pd.Series(np.nan,index=frame.index)))
    successful=frame.loc[valid]
    result=dict(attempted_targets=int(len(frame)),valid_targets=int(len(successful)),
                failed_targets=int((~valid).sum()),events=int(frame['event_id'].nunique()),
                decision_rows=int(frame[['event_id','elapsed_time','geometry_protocol']].drop_duplicates().shape[0]))
    if not len(successful):
        return result
    y,p=successful['truth'].to_numpy(),successful['prediction'].to_numpy()
    error=p-y
    result.update(mae=float(np.abs(error).mean()),rmse=float(np.sqrt(np.mean(error**2))),bias=float(error.mean()),
                  p90_abs=float(np.quantile(np.abs(error),.90)),p95_abs=float(np.quantile(np.abs(error),.95)),
                  within01=float(np.mean(np.abs(error)<=.1)),within02=float(np.mean(np.abs(error)<=.2)),
                  r2=float(1-np.sum(error**2)/np.sum((y-y.mean())**2)) if np.var(y)>0 else None,
                  prediction_on_truth_slope=float(np.cov(y,p,ddof=0)[0,1]/np.var(y)) if np.var(y)>0 else None,
                  truth_on_prediction_slope=float(np.cov(y,p,ddof=0)[0,1]/np.var(p)) if np.var(p)>0 else None,
                  correlation=float(np.corrcoef(y,p)[0,1]) if np.var(y)>0 and np.var(p)>0 else None)
    for name in ('nll','crps','brier','covered95','covered68','covered_mean_sigma','covered_mean_2sigma','width95','predictive_sigma'):
        if name in successful:
            result[name]=float(successful[name].mean())
    return result


def grouped_metrics(frame):
    rows=[]
    for (protocol,time),group in frame.groupby(['geometry_protocol','elapsed_time'],dropna=False):
        groups={'all':group,'noninput':group.loc[group['target_role']!='observed_input']}
        for role,sub in group.groupby('target_role'):
            groups[role]=sub
        for name,sub in groups.items():
            rows.append(dict(geometry_protocol=protocol,elapsed_time=time,target_group=name,**summarize(sub)))
    return pd.DataFrame(rows)


def checkpoint_selection(frame,common_times):
    selected=frame.loc[frame['elapsed_time'].isin(common_times)&frame['target_role'].ne('observed_input')]
    values=[]
    for protocol in ('normal','random'):
        for time in common_times:
            group=selected.loc[selected['geometry_protocol'].eq(protocol)&selected['elapsed_time'].eq(time)]
            if not len(group) or not group['status'].eq('supported').all() or not np.isfinite(group['prediction']).all():
                raise ValueError('Incomplete/failing common support checkpoint selection cell')
            values.append(float(np.abs(group['prediction']-group['truth']).mean()))
    return float(np.mean(values))


def paired_bootstrap(left,right,draws=5000,seed=20261001):
    keys=['dataset_id','event_id','elapsed_time','station_id','geometry_protocol']
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError('Duplicate paired evaluation identity')
    joined=left.merge(right,on=keys,suffixes=('_left','_right'),validate='one_to_one')
    if len(joined)!=len(left) or len(joined)!=len(right):
        raise ValueError('Paired population differs; export intersection/coverage audit first')
    if not np.array_equal(joined.truth_left.to_numpy(),joined.truth_right.to_numpy()):
        raise ValueError('Paired query labels differ')
    if not joined.status_left.eq('supported').all() or not joined.status_right.eq('supported').all():
        raise ValueError('Numerical failures cannot be omitted from paired ranking')
    joined['delta']=np.abs(joined.prediction_right-joined.truth_right)-np.abs(joined.prediction_left-joined.truth_left)
    clusters=joined.groupby('event_id')['delta'].agg(['sum','count','mean'])
    rng=np.random.default_rng(seed)
    sampled=np.empty(draws)
    # Keep all station/time rows of an event together and compute the primary
    # equal-time/protocol endpoint inside each replicate, rather than pooling
    # target counts across times with a different weighting scheme.
    cells=joined.groupby(['event_id','geometry_protocol','elapsed_time']).delta.agg(['sum','count'])
    cell_sums=cells['sum'].unstack(['geometry_protocol','elapsed_time']).reindex(clusters.index).fillna(0).to_numpy()
    cell_counts=cells['count'].unstack(['geometry_protocol','elapsed_time']).reindex(clusters.index).fillna(0).to_numpy()
    primary_sampled=np.empty(draws)
    for i in range(draws):
        ids=rng.integers(0,len(clusters),len(clusters))
        sampled[i]=clusters['sum'].to_numpy()[ids].sum()/clusters['count'].to_numpy()[ids].sum()
        denominators=cell_counts[ids].sum(0)
        primary_sampled[i]=np.mean(cell_sums[ids].sum(0)/denominators) if (denominators>0).all() else np.nan
    primary_delta=float(np.mean(cell_sums.sum(0)/cell_counts.sum(0)))
    valid_primary=primary_sampled[np.isfinite(primary_sampled)]
    return dict(delta_mae=float(joined.delta.mean()),event_macro_delta=float(clusters['mean'].mean()),
                equal_time_protocol_delta_mae=primary_delta,
                equal_time_protocol_ci_low=float(np.quantile(valid_primary,.025)),
                equal_time_protocol_ci_high=float(np.quantile(valid_primary,.975)),
                empty_cell_bootstrap_replicates=int((~np.isfinite(primary_sampled)).sum()),
                ci_low=float(np.quantile(sampled,.025)),ci_high=float(np.quantile(sampled,.975)),
                events=len(clusters),targets=len(joined),draws=draws,seed=seed,
                cluster='event_id; all station/time rows move together')
