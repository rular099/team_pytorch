"""68% mixture widths, kept separate from the saved original 95% score definition."""
import pandas as pd
from .contracts import EPOCHS
from .statistics import ENDPOINTS,groups,bootstrap_mae,paired
from .sources import load_export
from .io import csv_write,write_json


def analyze(source,out,draws=2000):
    out.mkdir(parents=True,exist_ok=False)
    points=[];effects=[]
    for policy in ('selected','epoch12'):
        data={family:load_export(source,family,policy) for family in EPOCHS}
        for family,f in data.items():
            for group,mask in groups(f).items():
                for endpoint,times in ENDPOINTS.items():
                    s=f[mask & f.elapsed_time.isin(times)]
                    width=s.q84-s.q16
                    cells=width.groupby([s.geometry_protocol,s.elapsed_time]).mean()
                    points.append(dict(family=family,checkpoint_policy=policy,target_group=group,endpoint=endpoint,
                                       metric='mixture_width68_dex',equal_cell_width68=float(cells.mean()) if len(cells) else None,
                                       supported_cells=len(cells),missing_cells=2*len(times)-len(cells),requests=len(s),
                                       quantiles='0.158655 to 0.841345; exact Gaussian mixture'))
        d=data['DiTing']
        for other in ('PhaseNet','EQT','TEAM'):
            b=data[other]
            paired(d,b)
            aa=d.copy();bb=b.copy()
            aa['prediction']=aa.q84-aa.q16;bb['prediction']=bb.q84-bb.q16
            aa['truth']=0.;bb['truth']=0.
            for group,mask in groups(d).items():
                for endpoint,times in ENDPOINTS.items():
                    for r in bootstrap_mae(aa.loc[mask],bb.loc[mask],draws,times):
                        r['metric']='mixture_width68_dex'
                        effects.append(dict(comparator=other,checkpoint_policy=policy,target_group=group,endpoint=endpoint,**r))
    csv_write(pd.DataFrame(points),out/'width68_metrics.csv')
    csv_write(pd.DataFrame(effects),out/'width68_paired_effects.csv')
    write_json(out/'interpretation.json',dict(definition='q84-q16; source exact mixture quantiles',
        width_direction='narrower is not automatically better; interpret with coverage and proper scores',
        effect='DiTing minus comparator',uncertainty='event clusters; fixed models, seed42 validation'))
