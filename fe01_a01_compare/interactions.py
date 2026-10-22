"""Prespecified difference-of-differences with common time/K support standardization."""
from itertools import product
import numpy as np
import pandas as pd
from .contracts import EPOCHS,EVENT,contract
from .io import csv_write,fingerprint,write_json
from .sources import load_export
from .statistics import paired
from .stratification import covariates,assign

LEVELS={
    'time':['1.0','3.0','5.0','10.0','20.0'],
    'input_count':['1','2','3-4','5-8','9-16','17-25'],
    'snr':['<0','[0,10)','[10,20)','>=20'],
    'truth_logpga':['y<=-2','-2<y<=-1.2','-1.2<y<=0','y>0'],
    'nearest_input_distance':['0-5','5-20','20-50','50-100','>100'],
    'magnitude':['M<=4','4<M<=5','5<M<=6','M>6'],
    'epicentral_distance':['0-25','25-50','50-100','100-200','>200'],
}


def contrast_arrays(frame,strata,first,second,a1,b1):
    # References and all contrasts are specified without looking at outcomes.
    a0,b0=LEVELS[first][0],LEVELS[second][0]
    pairs=[(a0,b0),(a0,b1),(a1,b0),(a1,b1)]
    domain=['geometry_protocol']
    if 'time' not in (first,second):domain.append('elapsed_time')
    if 'input_count' not in (first,second):domain.append('_K')
    f=frame[EVENT+['geometry_protocol','elapsed_time','delta']].copy()
    f['_K']=strata.input_count
    f['_first']=strata[first]
    f['_second']=strata[second]
    group_counts=[]
    parts=[]
    domains=[]
    for i,(a,b) in enumerate(pairs):
        sub=f[f._first.eq(a)&f._second.eq(b)].copy()
        group_counts.append(sub[EVENT].drop_duplicates().shape[0])
        sub['_contrast_cell']=i
        parts.append(sub)
        domains.append(set(map(tuple,sub[domain].drop_duplicates().to_numpy())))
    common=set.intersection(*domains) if all(domains) else set()
    if not common:
        return None,dict(status='NOT_IDENTIFIABLE_NO_COMMON_SUPPORT',group_events=group_counts,common_domains=0,
                         unknown_policy='Unknown retained in main support/stratum tables; no invented SNR contrast')
    use=pd.concat(parts,ignore_index=True)
    use=use[[tuple(v) in common for v in use[domain].to_numpy()]]
    use['_domain']=[fingerprint(list(v)) for v in use[domain].to_numpy()]
    table=use.groupby(EVENT+['_domain','_contrast_cell']).delta.agg(['sum','count'])
    columns=pd.MultiIndex.from_product([sorted(set(use._domain)),range(4)],names=['_domain','_contrast_cell'])
    den=table['count'].unstack(['_domain','_contrast_cell']).reindex(columns=columns).fillna(0)
    num=table['sum'].unstack(['_domain','_contrast_cell']).reindex(index=den.index,columns=columns).fillna(0)
    support=dict(status='SUPPORTED',group_events=group_counts,events=len(den),common_domains=len(common),
                 common_domain_sha256=fingerprint(sorted([list(x) for x in common])),
                 excluded_noncommon_domains=len(set.union(*domains)-common),
                 requests=int(den.to_numpy().sum()),descriptive_only=min(group_counts)<20,
                 standardization='targets within common domain; equal geometry/time/K domains excluding the interacting factors')
    return (num.to_numpy(),den.to_numpy(),len(common)),support


def interval(arrays,draws=2000):
    num,den,domains=arrays
    signs=np.tile([1.,-1.,-1.,1.],domains)/domains
    estimate=float((num.sum(0)/den.sum(0))@signs)
    rng=np.random.default_rng(20261007)
    samples=[]
    for start in range(0,draws,100):
        idx=rng.integers(0,len(num),size=(min(100,draws-start),len(num)))
        weights=np.asarray([np.bincount(r,minlength=len(num)) for r in idx],float)
        nn,dd=weights@num,weights@den
        valid=(dd>0).all(1)
        samples.extend(((nn[valid]/dd[valid])@signs).tolist())
    return dict(estimate=estimate,ci_low=float(np.quantile(samples,.025)) if samples else None,
                ci_high=float(np.quantile(samples,.975)) if samples else None,draws=draws,effective_replicates=len(samples),
                empty_cell_fraction=1-len(samples)/draws,
                interval_status='CONDITIONAL_UNSTABLE' if len(samples)<.95*draws else 'SUPPORTED',
                effect='(DiTing-comparator at A1,B1 - at A1,B0) - (at A0,B1 - at A0,B0)',
                uncertainty='joint event clusters across all four contrast cells; fixed model; exploratory uncorrected')


def analyze(source,out,sidecar=None,draws=2000):
    out.mkdir(parents=True,exist_ok=False)
    supports=[]
    outputs=[]
    for policy in ('selected','epoch12'):
        d=load_export(source,'DiTing',policy)
        non=d.target_role.ne('observed_input')
        enriched=covariates(d,sidecar)
        strata=assign(enriched)
        masks=[]
        # Freeze support independently of comparator predictions, before any contrast effect.
        base=enriched[non].copy();base['delta']=0.
        for first,second in contract()['strata']['interactions']:
            for a1,b1 in product(LEVELS[first][1:],LEVELS[second][1:]):
                array,s=contrast_arrays(base,strata.loc[non],first,second,a1,b1)
                tag=dict(checkpoint_policy=policy,first=first,second=second,A0=LEVELS[first][0],B0=LEVELS[second][0],A1=a1,B1=b1)
                supports.append(dict(**tag,**s))
                masks.append((tag,s))
        csv_write(pd.DataFrame([x for x in supports if x['checkpoint_policy']==policy]),out/('interaction_support_frozen_'+policy+'.csv'))
        for comparator in ('PhaseNet','EQT','TEAM'):
            b=load_export(source,comparator,policy)
            paired(d,b)
            f=enriched[non].copy()
            f['delta']=np.abs(d.loc[non,'prediction']-d.loc[non,'truth'])-np.abs(b.loc[non,'prediction']-b.loc[non,'truth'])
            for tag,s in masks:
                if s['status']=='SUPPORTED':
                    arr,_=contrast_arrays(f,strata.loc[non],tag['first'],tag['second'],tag['A1'],tag['B1'])
                    r=interval(arr,draws)
                else:
                    r=dict(estimate=None,ci_low=None,ci_high=None,effective_replicates=0,draws=draws)
                outputs.append(dict(comparator=comparator,**tag,**s,**r))
    csv_write(pd.DataFrame(outputs),out/'interaction_contrasts.csv')
    csv_write(pd.DataFrame(supports),out/'interaction_contrast_support.csv')
    write_json(out/'contract.json',dict(reference_levels={k:v[0] for k,v in LEVELS.items()},all_levels=LEVELS,
        interactions=contract()['strata']['interactions'],support_freeze='before comparator effect calculation',
        bins='fixed V4; missing/Unknown retained and non-identifiable contrasts NA',
        weighting='common time/K domain standardized difference-of-differences; distinct from original ten-cell main endpoint',
        inference='exploratory uncorrected; event clusters do not cover seed or selection uncertainty'))
