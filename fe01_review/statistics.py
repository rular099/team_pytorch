"""Sufficient statistics and paired event-cluster intervals, never ensembles."""
import numpy as np
import pandas as pd
from .provenance import require

EVENT_CELL=['dataset_id','event_id','geometry_protocol','elapsed_time']


def event_losses(frame):
    from .diagnostics import target_groups
    rows=[]
    for role,group in target_groups(frame).items():
        data=group.assign(abs_error=(group.prediction-group.truth).abs(),squared_error=(group.prediction-group.truth)**2)
        measures=[c for c in ('abs_error','squared_error','nll','crps','brier','covered95','covered68','width95','predictive_sigma') if c in data]
        table=data.groupby(EVENT_CELL)[measures].sum().add_suffix('_sum')
        table['targets']=data.groupby(EVENT_CELL).size()
        table=table.reset_index();table['target_group']=role;rows.append(table)
    return pd.concat(rows,ignore_index=True)


def loss_arrays(table,metric='abs_error',times=None,protocol=None):
    table=table if times is None else table[table.elapsed_time.isin(times)]
    table=table if protocol is None else table[table.geometry_protocol.eq(protocol)]
    require(len(table)>0,'Empty loss population')
    numerator=table.pivot(index=['dataset_id','event_id'],columns=['geometry_protocol','elapsed_time'],values=metric+'_sum')
    denominator=table.pivot(index=['dataset_id','event_id'],columns=['geometry_protocol','elapsed_time'],values='targets')
    require((denominator.sum()>0).all(),'Empty scoring cell')
    return numerator.fillna(0).to_numpy(),denominator.fillna(0).to_numpy(),numerator.index


def interval(table,metric='abs_error',times=None,protocol=None,draws=5000,right=None):
    a,count,index=loss_arrays(table,metric,times,protocol)
    if right is not None:
        b,other,other_index=loss_arrays(right,metric,times,protocol)
        require(index.equals(other_index) and np.array_equal(count,other),'Unpaired event-cell denominators')
        a=b-a
    point=float(np.mean(a.sum(0)/count.sum(0)))
    rng=np.random.default_rng(20261001);samples=[]
    for start in range(0,draws,100):
        idx=rng.integers(0,len(index),(min(100,draws-start),len(index)))
        sums,counts=a[idx].sum(1),count[idx].sum(1)
        good=(counts>0).all(1)
        samples.extend((sums[good]/counts[good]).mean(1).tolist())
    require(len(samples)>0,'No complete event-bootstrap replicate')
    return dict(estimate=point,ci_low=float(np.quantile(samples,.025)),ci_high=float(np.quantile(samples,.975)),
        events=len(index),targets=int(count.sum()),draws=draws,rng_seed=20261001,empty_cell_replicates=draws-len(samples),
        weighting='target weighted within cell; cells equal',cluster='dataset_id,event_id; all times/protocols/stations move together')


def fixed_model_mean_loss(tables):
    keys=EVENT_CELL+['target_group']
    ordered=[t.sort_values(keys).reset_index(drop=True) for t in tables]
    base=ordered[0].copy()
    for t in ordered[1:]:
        require(base[keys+['targets']].equals(t[keys+['targets']]),'Seed loss populations differ')
    for c in base.columns:
        if c.endswith('_sum'): base[c]=np.mean([t[c].to_numpy() for t in ordered],axis=0)
    return base
