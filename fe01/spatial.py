"""Retrospective train-only references, site residuals and geographic holdouts."""
import json

import numpy as np
import pandas as pd

from .sampling import stable_rng


def distance_km(lat1,lon1,lat2,lon2):
    lat1,lon1,lat2,lon2=map(np.deg2rad,[lat1,lon1,lat2,lon2])
    a=np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 6371.0088*2*np.arcsin(np.sqrt(np.clip(a,0,1)))


def reference_design(frame):
    horizontal=distance_km(frame['latitude'].to_numpy(),frame['longitude'].to_numpy(),
                           frame['event_latitude'].to_numpy(),frame['event_longitude'].to_numpy())
    depth=frame['depth'].to_numpy()
    radius=np.sqrt(horizontal**2+depth**2)
    magnitude=frame['magnitude'].to_numpy()-4
    return np.column_stack([np.ones(len(frame)),magnitude,magnitude**2,
                            np.log10(np.maximum(radius,1)),radius/100,depth/100])


def fit_reference(frame,penalty=.1,shrinkage=10):
    if not len(frame) or not frame['split'].eq('train').all():
        raise ValueError('Reference fitting is train-only')
    keys=['event_id','station_id']
    if frame.duplicated(keys).any():
        raise ValueError('Fit each final PGA once, not repeated decision rows')
    x=reference_design(frame)
    y=frame['truth'].to_numpy()
    regularizer=penalty*np.eye(x.shape[1]);regularizer[0,0]=0
    beta=np.linalg.solve(x.T@x+regularizer,x.T@y)
    # Fit additive event/site effects on train, with fixed shrinkage.
    station=np.zeros(len(frame));event=np.zeros(len(frame))
    site_map={}
    event_map={}
    for _ in range(20):
        residual=y-x@beta-station
        event_map=pd.Series(residual).groupby(frame.event_id.to_numpy()).mean().to_dict()
        event=np.asarray([event_map[e] for e in frame.event_id])
        residual=y-x@beta-event
        grouped=pd.Series(residual).groupby(frame.station_id.to_numpy()).agg(['sum','count'])
        site_map=(grouped['sum']/(grouped['count']+shrinkage)).to_dict()
        station=np.asarray([site_map[s] for s in frame.station_id])
        beta=np.linalg.solve(x.T@x+regularizer,x.T@(y-event-station))
    # A separate no-site reference is the primary residual anchor.
    no_site=np.linalg.solve(x.T@x+regularizer,x.T@y)
    return dict(no_site_coefficients=no_site.tolist(),site_reference_coefficients=beta.tolist(),
                train_site_effects=site_map,train_event_effects=event_map,
                train_events=int(frame.event_id.nunique()),train_stations=int(frame.station_id.nunique()),
                penalty=penalty,shrinkage=shrinkage,units='log10(m/s^2)',
                interpretation='retrospective/oracle M, hypocenter, depth; not a realtime competitor',
                distance='hypocentral sqrt(haversine_epicentral_km^2 + depth_km^2)',
                terms=['intercept','M-4','(M-4)^2','log10(max(R,1 km))','R/100','depth/100'])


def residuals(frame,reference):
    if frame['split'].eq('train').any():
        raise ValueError('Site evaluation requires held-out events')
    result=frame.copy()
    result['reference']=reference_design(frame)@np.asarray(reference['no_site_coefficients'])
    site=np.asarray([reference['train_site_effects'].get(str(s),0) for s in frame.station_id])
    result['reference_with_train_site']=reference_design(frame)@np.asarray(reference['site_reference_coefficients'])+site
    result['site_seen_in_reference_train']=frame.station_id.astype(str).isin(reference['train_site_effects'])
    result['observed_residual']=result['truth']-result['reference']
    result['predicted_residual']=result['prediction']-result['reference']
    keys=['event_id','elapsed_time','geometry_protocol']
    for name in ['observed_residual','predicted_residual']:
        result['centered_'+name]=result[name]-result.groupby(keys)[name].transform('mean')
    return result


def site_summary(frame,draws=5000,seed=20261001,min_events=10):
    rows=[]
    # Time/protocol are kept separate rather than multiplying independent events.
    for key,group in frame.groupby(['station_id','geometry_protocol','elapsed_time']):
        obs=group['centered_observed_residual'].to_numpy()
        pred=group['centered_predicted_residual'].to_numpy()
        if group.event_id.duplicated().any():
            raise ValueError('Duplicate station/event/time scoring row')
        rng=stable_rng(seed,key,'site-ci')
        idx=rng.integers(0,len(group),(draws,len(group)))
        delta=(pred-obs)[idx].mean(-1)
        rows.append(dict(station_id=key[0],geometry_protocol=key[1],elapsed_time=key[2],
            events=len(group),sufficient_events=len(group)>=min_events,
            observed=float(obs.mean()),predicted=float(pred.mean()),
            mae=float(np.abs(pred-obs).mean()),sign_accuracy=float(np.mean(np.sign(obs)==np.sign(pred))),
            correlation=float(np.corrcoef(obs,pred)[0,1]) if np.std(obs)>0 and np.std(pred)>0 else None,
            delta_ci_low=float(np.quantile(delta,.025)),delta_ci_high=float(np.quantile(delta,.975))))
    return pd.DataFrame(rows)


def field_metrics(frame,min_targets=5,minimum_range=.05,equal_distance_tolerance_km=5):
    rows=[]
    for key,group in frame.groupby(['event_id','elapsed_time','geometry_protocol']):
        if len(group)<min_targets:
            continue
        y,p=group.truth.to_numpy(),group.prediction.to_numpy()
        centered=(p-p.mean())-(y-y.mean())
        truth_range=np.quantile(y,.95)-np.quantile(y,.05)
        pred_range=np.quantile(p,.95)-np.quantile(p,.05)
        i,j=np.triu_indices(len(group),1)
        pair_error=np.abs((p[i]-p[j])-(y[i]-y[j]))
        distances=distance_km(group.latitude.to_numpy(),group.longitude.to_numpy(),
                             group.event_latitude.to_numpy(),group.event_longitude.to_numpy())
        close=np.abs(distances[i]-distances[j])<=equal_distance_tolerance_km
        rows.append(dict(event_id=key[0],elapsed_time=key[1],geometry_protocol=key[2],targets=len(group),
            input_count=int(group.input_count.iloc[0]),field_mean_error=float(p.mean()-y.mean()),
            centered_mae=float(np.abs(centered).mean()),pairwise_delta_mae=float(pair_error.mean()),
            range_error=float(abs(pred_range-truth_range)),
            range_ratio=float(pred_range/truth_range) if truth_range>=minimum_range else None,
            range_ratio_excluded=truth_range<minimum_range,
            std_ratio=float(np.std(p)/np.std(y)) if np.std(y)>=minimum_range/4 else None,
            equal_distance_pairs=int(close.sum()),
            equal_distance_delta_mae=float(pair_error[close].mean()) if close.any() else None,
            equal_distance_sign_accuracy=float(np.mean(np.sign(p[i][close]-p[j][close])==np.sign(y[i][close]-y[j][close]))) if close.any() else None))
    return pd.DataFrame(rows)


def make_holdout(stations,seed=42,buffer_km=20):
    if stations.station_id.duplicated().any():
        raise ValueError('Station catalog must be unique')
    table=stations.copy()
    table['block_lat']=np.floor(table.latitude).astype(int)
    table['block_lon']=np.floor(table.longitude).astype(int)
    def assign(row):
        u=stable_rng(seed,int(row.block_lat),int(row.block_lon),'geographic-block').random()
        return 'test' if u<.10 else 'val' if u<.20 else 'train'
    table['spatial_split']=table.apply(assign,axis=1)
    held=table.loc[table.spatial_split.ne('train')]
    table['buffer_excluded']=False
    for index,row in table.loc[table.spatial_split.eq('train')].iterrows():
        distances=distance_km(row.latitude,row.longitude,held.latitude.to_numpy(),held.longitude.to_numpy())
        if len(distances) and distances.min()<buffer_km:
            table.loc[index,'buffer_excluded']=True
    # Remove close val/test interfaces, preserving both core assignments.
    test=table.loc[table.spatial_split.eq('test')]
    for index,row in table.loc[table.spatial_split.eq('val')].iterrows():
        d=distance_km(row.latitude,row.longitude,test.latitude.to_numpy(),test.longitude.to_numpy())
        if len(d) and d.min()<buffer_km:
            table.loc[index,'buffer_excluded']=True
    if any(not ((table.spatial_split==split)&~table.buffer_excluded).any() for split in ('train','val','test')):
        raise ValueError('Empty spatial core; audit geographic coverage before training')
    return table


def grid(bounds,spacing_km=10):
    lat_min,lat_max,lon_min,lon_max=bounds
    latitude=np.arange(lat_min,lat_max+1e-9,np.rad2deg(spacing_km/6371.0088))
    points=[]
    for lat in latitude:
        step=np.rad2deg(spacing_km/(6371.0088*np.cos(np.deg2rad(lat))))
        points.extend((lat,lon,0.) for lon in np.arange(lon_min,lon_max+1e-9,step))
    return np.asarray(points,dtype=np.float32)


def coverage(points,input_coords,maximum_distance_km=100):
    if not len(input_coords):
        return np.zeros(len(points),dtype=bool)
    distances=distance_km(points[:,None,0],points[:,None,1],input_coords[None,:,0],input_coords[None,:,1])
    return distances.min(-1)<=maximum_distance_km
