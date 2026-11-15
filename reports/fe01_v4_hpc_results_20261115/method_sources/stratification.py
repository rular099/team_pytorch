"""Fixed bins and four fixed interactions, including explicit Unknown support."""
import json
import numpy as np
import pandas as pd
from fe01.spatial import distance_km
from .contracts import DECISION, EVENT
from .io import fingerprint
from .query_grid import geometry_covariates


def bins(values, boundaries, labels, right=True):
    s=pd.Series(values)
    return pd.cut(s,[-np.inf]+list(boundaries)+[np.inf],labels=labels,right=right).astype(object).where(s.notna(),'Unknown').astype(str)


def covariates(frame, sidecar=None):
    f=frame.copy()
    rows=[]
    for key,g in f.groupby(DECISION,sort=False):
        inputs=np.asarray(json.loads(g.input_coords.iloc[0]))
        cov=geometry_covariates(g[['latitude','longitude']].to_numpy(),inputs)
        for index, r in zip(g.index,cov):
            rows.append(dict(_index=index,**r))
    geometry=pd.DataFrame(rows).set_index('_index').reindex(f.index)
    for c in geometry:
        f[c]=geometry[c]
    f['epicentral_distance_km']=distance_km(f.latitude,f.longitude,f.event_latitude,f.event_longitude)
    for c in ('snr_db','physical_missing_fraction','origin_elapsed_seconds','post_p_valid_seconds_min','post_p_valid_seconds_median','inputs_post_p_at_least_1s'):
        f[c]=np.nan
    f['physical_metadata_status']='Unknown_requires_CPU_HDF_sidecar'
    if sidecar is not None:
        # Legacy geometry is joined only when the actual input IDs AND cutoff match.
        shared=['snr_db','physical_missing_fraction','origin_elapsed_seconds','post_p_valid_seconds_min','post_p_valid_seconds_median','inputs_post_p_at_least_1s']
        keys=DECISION+['input_ids','current_sample']
        s=sidecar[keys+shared].drop_duplicates()
        joined=f.drop(columns=shared).merge(s,on=keys,how='left',validate='many_to_one',sort=False)
        f=joined
        f['physical_metadata_status']=np.where(f.physical_missing_fraction.notna(),'verified_shared_input_sidecar','Unknown_input_plan_or_metadata')
    return f


def assign(f):
    out={}
    out['magnitude']=bins(f.magnitude,[4,5,6],['M<=4','4<M<=5','5<M<=6','M>6'])
    out['truth_logpga']=bins(f.truth,[-2,-1.2,0],['y<=-2','-2<y<=-1.2','-1.2<y<=0','y>0'])
    out['epicentral_distance']=bins(f.epicentral_distance_km,[25,50,100,200],['0-25','25-50','50-100','100-200','>200'])
    out['depth']=bins(f.depth,[10,30,70,300],['0-10','10-30','30-70','70-300','>300'])
    out['nearest_input_distance']=bins(f.nearest_input_distance_km,[5,20,50,100],['0-5','5-20','20-50','50-100','>100'])
    out['input_count']=bins(f.input_count,[1,2,4,8,16,25],['1','2','3-4','5-8','9-16','17-25','Unknown'])
    out['missing_fraction']=bins(f.physical_missing_fraction,[0,.25,.5],['0','(0,.25]','(.25,.5]','>.5'])
    out['snr']=bins(f.snr_db,[0,10,20],['<0','[0,10)','[10,20)','>=20'],right=False)
    out['origin_time']=bins(f.origin_elapsed_seconds,[0,5,10,20,40],['Unknown_abnormal','[0,5)','[5,10)','[10,20)','[20,40)','>=40'],right=False)
    out['time']=f.elapsed_time.astype(str)
    out['hull_status']=f.hull_status.fillna('Unknown').astype(str)
    out['post_p_seconds_min']=bins(f.post_p_valid_seconds_min,[0,.5,1,3,5],['0','(0,.5]','(.5,1]','(1,3]','(3,5]','>5'])
    out['input_span']=bins(f.input_span_km,[5,20,50,100],['<=5','5-20','20-50','50-100','>100'])
    out['azimuth_gap']=bins(f.azimuth_gap_degrees,[90,180,270],['<=90','90-180','180-270','>270'])
    for name in ('depth','epicentral_distance','nearest_input_distance'):
        col={'depth':'depth','epicentral_distance':'epicentral_distance_km','nearest_input_distance':'nearest_input_distance_km'}[name]
        out[name]=out[name].where(f[col]>=0,'Unknown_abnormal')
    return pd.DataFrame(out,index=f.index)


def support(f, strata, interactions):
    records=[]
    factors={name:strata[name] for name in strata}
    for left,right in interactions:
        factors[left+' X '+right]=strata[left]+' | '+strata[right]
    for factor,v in factors.items():
        for label in sorted(v.unique()):
            sub=f.loc[v.eq(label)]
            records.append(dict(factor=factor,stratum=label,events=sub[EVENT].drop_duplicates().shape[0],
                                stations=sub.station_id.nunique(),requests=len(sub),decisions=sub[DECISION].drop_duplicates().shape[0],
                                supported_cells=sub[['geometry_protocol','elapsed_time']].drop_duplicates().shape[0],
                                fields=int((sub.groupby(DECISION).size()>=5).sum()),
                                missing_metadata_fraction=float(('Unknown' in label)),descriptive_only=sub[EVENT].drop_duplicates().shape[0]<20,
                                mask_sha256=fingerprint(sub.index.tolist()),
                                cohort_rule='shared before model effect; fixed bins; Unknown retained'))
    return pd.DataFrame(records),factors
