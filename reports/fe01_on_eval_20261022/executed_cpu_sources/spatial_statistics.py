"""Field-equal original spatial metrics; explicit sparse cell conditioning."""
import numpy as np
import pandas as pd
from .contracts import EVENT, TIMES
from .io import require


def field_intervals(a,b,draws=2000):
    keys=EVENT+['geometry_protocol','elapsed_time']
    require(a[keys+['targets']].equals(b[keys+['targets']]),'Field population mismatch')
    rows=[]
    for metric in ('level_mse','shape_mse','field_mse','range_error','equal_distance_delta_mae'):
        good=a[metric].notna() & b[metric].notna()
        x=a.loc[good,keys].copy()
        x['delta']=a.loc[good,metric]-b.loc[good,metric]
        if not len(x):
            continue
        cells=pd.MultiIndex.from_product([('normal','random'),TIMES],names=['geometry_protocol','elapsed_time'])
        p=x.pivot(index=EVENT,columns=['geometry_protocol','elapsed_time'],values='delta').reindex(columns=cells)
        counts=p.notna().astype(int).to_numpy()
        num=p.fillna(0).to_numpy()
        supported=counts.sum(0)>0
        num,counts=num[:,supported],counts[:,supported]
        rng=np.random.default_rng(20261007)
        sampled=[]
        for start in range(0,draws,100):
            idx=rng.integers(0,len(p),size=(min(100,draws-start),len(p)))
            n,d=num[idx].sum(1),counts[idx].sum(1)
            valid=(d>0).all(1)
            sampled.extend((n[valid]/d[valid]).mean(1).tolist())
        rows.append(dict(metric=metric,estimate=float((num.sum(0)/counts.sum(0)).mean()),
                         ci_low=float(np.quantile(sampled,.025)) if sampled else None,ci_high=float(np.quantile(sampled,.975)) if sampled else None,
                         events=len(p),fields=len(x),supported_cells=int(supported.sum()),missing_cells=int((~supported).sum()),
                         draws=draws,effective_replicates=len(sampled),empty_cell_fraction=1-len(sampled)/draws,
                         weighting='eligible field equal within cell; supported cells equal',
                         status='CONDITIONAL_UNSTABLE' if len(sampled)<.95*draws else 'SPARSE' if not supported.all() else 'FULL_SUPPORT'))
    return rows
