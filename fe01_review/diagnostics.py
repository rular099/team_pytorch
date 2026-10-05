"""Validation diagnostics; event fields and target populations stay explicit."""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from fe01.metrics import grouped_metrics, summarize
from fe01.spatial import distance_km, reference_design, fit_reference
from .requests import TIMES, PAIR_KEYS, DECISION_KEYS
from .provenance import require, fingerprint, write_json, sha256


def fields(frame, min_targets=5, minimum_range=.05):
    rows = []
    require(frame.status.eq('supported').all(), 'Failures cannot be discarded from field comparison')
    require(np.isfinite(frame[['truth', 'prediction']]).all().all(), 'Nonfinite field prediction')
    for role, subset in target_groups(frame).items():
        for key, group in subset.groupby(DECISION_KEYS):
            base = dict(zip(DECISION_KEYS, key), target_group=role, targets=len(group),
                        input_count=int(group.input_count.iloc[0]), eligible_field=len(group) >= min_targets)
            if len(group) < min_targets:
                rows.append(dict(**base, status='N/A', reason='fewer than 5 targets'))
                continue
            y, p = group.truth.to_numpy(), group.prediction.to_numpy()
            error = p - y
            centered = error - error.mean()
            tr, pr = np.quantile(y, .95) - np.quantile(y, .05), np.quantile(p, .95) - np.quantile(p, .05)
            i, j = np.triu_indices(len(y), 1)
            yd, pdiff = y[i] - y[j], p[i] - p[j]
            radius = distance_km(group.latitude.to_numpy(), group.longitude.to_numpy(), group.event_latitude.to_numpy(), group.event_longitude.to_numpy())
            close = np.abs(radius[i] - radius[j]) <= 5
            signs = close & (np.abs(yd) > 1e-8)
            rows.append(dict(**base, status='supported', reason='', field_mean_error=float(error.mean()),
                level_mse=float(error.mean() ** 2), shape_mse=float(np.mean(centered ** 2)), field_mse=float(np.mean(error ** 2)),
                centered_mae=float(np.abs(centered).mean()), centered_rmse=float(np.sqrt(np.mean(centered ** 2))),
                pairwise_pairs=len(i), pairwise_delta_mae=float(np.abs(pdiff - yd).mean()),
                truth_range=float(tr), prediction_range=float(pr), range_error=float(abs(pr - tr)),
                range_ratio=float(pr / tr) if tr >= minimum_range else None, range_ratio_excluded=tr < minimum_range,
                equal_distance_pairs=int(close.sum()), equal_distance_delta_mae=float(np.abs(pdiff[close] - yd[close]).mean()) if close.any() else None,
                equal_distance_sign_denominator=int(signs.sum()), equal_distance_sign_accuracy=float(np.mean(np.sign(pdiff[signs]) == np.sign(yd[signs]))) if signs.any() else None))
    return pd.DataFrame(rows)


def target_groups(frame):
    return {'all': frame, 'noninput': frame.loc[frame.target_role.ne('observed_input')],
            'observed_input': frame.loc[frame.target_role.eq('observed_input')],
            'triggered_noninput': frame.loc[frame.target_role.eq('triggered_noninput')],
            'untriggered_noninput': frame.loc[frame.target_role.eq('untriggered_noninput')],
            'actual_single_input_noninput': frame.loc[frame.input_count.eq(1) & frame.target_role.ne('observed_input')]}


def group_scores(frame):
    rows = []
    for role, subset in target_groups(frame).items():
        for protocol in ('normal', 'random'):
            for time in TIMES:
                group = subset.loc[subset.geometry_protocol.eq(protocol) & subset.elapsed_time.eq(time)]
                if len(group):
                    record = summarize(group)
                    record['mse'] = float(np.mean((group.prediction - group.truth) ** 2))
                    for name in ('width68',):
                        record[name] = float(group[name].mean()) if name in group else float((group.q84 - group.q16).mean())
                else:
                    record = dict(attempted_targets=0, valid_targets=0, failed_targets=0, events=0, status='N/A', reason='empty subgroup')
                rows.append(dict(geometry_protocol=protocol, elapsed_time=time, target_group=role, **record))
    return pd.DataFrame(rows)


def fixed_panels(frame):
    rows, audits, migrations = [], [], []
    for protocol, group in frame.groupby('geometry_protocol'):
        keys = ['dataset_id', 'event_id', 'station_id']
        counts = group.groupby(keys).elapsed_time.nunique()
        intersection = counts[counts == len(TIMES)].index
        fixed = group.set_index(keys).loc[intersection].reset_index()
        always = fixed.groupby(keys).target_role.apply(lambda roles: roles.ne('observed_input').all())
        always_keys = always[always].index
        all_noninput = fixed.set_index(keys).loc[always_keys].reset_index()
        for panel, subset in (('full_changing_population', group), ('fixed_targets_natural_inputs', fixed), ('fixed_always_noninput_natural_inputs', all_noninput)):
            for time in TIMES:
                part = subset.loc[subset.elapsed_time.eq(time)]
                for role, chosen in target_groups(part).items():
                    if len(chosen):
                        rows.append(dict(geometry_protocol=protocol, elapsed_time=time, panel=panel, target_group=role, **summarize(chosen)))
                whole = group[group.elapsed_time.eq(time)]
                audits.append(dict(geometry_protocol=protocol, elapsed_time=time, panel=panel, original_targets=len(whole), retained_targets=len(part),
                    excluded_targets=len(whole) - len(part), original_events=whole[['dataset_id', 'event_id']].drop_duplicates().shape[0],
                    retained_events=part[['dataset_id', 'event_id']].drop_duplicates().shape[0]))
        pivot = fixed.pivot(index=keys, columns='elapsed_time', values='target_role')
        for previous, current in zip(TIMES[:-1], TIMES[1:]):
            moves = pivot.groupby([previous, current]).size()
            migrations.extend(dict(geometry_protocol=protocol, from_time=previous, to_time=current,
                                   from_role=a, to_role=b, event_station_trajectories=int(n)) for (a, b), n in moves.items())
    return pd.DataFrame(rows), pd.DataFrame(audits), pd.DataFrame(migrations)


def stratified(frame, native_length):
    frame = frame.copy()
    frame['distance_km'] = distance_km(frame.latitude.to_numpy(), frame.longitude.to_numpy(), frame.event_latitude.to_numpy(), frame.event_longitude.to_numpy())
    frame['native_left_padding_fraction'] = np.maximum(0, 1 - frame.history_seconds * 100 / native_length)
    frame['physical_history_missing_fraction'] = np.clip(1 - frame.valid_seconds / frame.history_seconds, 0, 1)
    specs = {
        'input_count': ([0, 1, 2, 4, 8, 16, np.inf], ['1', '2', '3–4', '5–8', '9–16', '17+']),
        'magnitude': ([-np.inf, 4, 5, 6, np.inf], ['M≤4', '4<M≤5', '5<M≤6', 'M>6']),
        'truth': ([-np.inf, -2, -1.2, 0, np.inf], ['y≤−2', '−2<y≤−1.2', '−1.2<y≤0', 'y>0']),
        'distance_km': ([0, 25, 50, 100, 200, np.inf], ['0≤R≤25', '25<R≤50', '50<R≤100', '100<R≤200', 'R>200']),
        'depth': ([0, 10, 30, 70, 300, np.inf], ['0≤D≤10', '10<D≤30', '30<D≤70', '70<D≤300', 'D>300']),
        'native_left_padding_fraction': ([-np.inf, .25, .5, .75, 1], ['p≤.25', '.25<p≤.5', '.5<p≤.75', '.75<p≤1']),
        'physical_history_missing_fraction': ([-np.inf, .25, .5, .75, 1], ['p≤.25', '.25<p≤.5', '.5<p≤.75', '.75<p≤1'])}
    tables = []
    for name, (edges, labels) in specs.items():
        codes = pd.cut(frame[name], edges, labels=labels, right=True, include_lowest=True)
        for label in labels:
            subset = frame.loc[codes.eq(label)]
            if not len(subset):
                continue
            scores = grouped_metrics(subset)
            scores['stratum'], scores['bucket'], scores['closed_side'] = name, label, 'right (first interval includes lower bound)'
            tables.append(scores)
    return pd.concat(tables, ignore_index=True)


def calibration_data(frame):
    reliability, pits, densities = [], [], []
    for role, subset in target_groups(frame).items():
        for (protocol, time), group in subset.groupby(['geometry_protocol', 'elapsed_time']):
            base = dict(target_group=role, geometry_protocol=protocol, elapsed_time=time)
            code = np.minimum((group.exceedance_probability.to_numpy() * 10).astype(int), 9)
            for bin_id in range(10):
                part = group.iloc[np.flatnonzero(code == bin_id)]
                reliability.append(dict(**base, probability_bin=bin_id, targets=len(part), events=part[['dataset_id', 'event_id']].drop_duplicates().shape[0],
                    probability_sum=float(part.exceedance_probability.sum()), exceedance_count=int(part.truth.gt(-1.2).sum())))
            counts, edges = np.histogram(group.pit, bins=np.linspace(0, 1, 21))
            pits.extend(dict(**base, lower=edges[i], upper=edges[i+1], targets=int(n), denominator=len(group)) for i, n in enumerate(counts))
            if role == 'noninput':
                bins = np.linspace(-4, 1, 51)
                hist, xb, yb = np.histogram2d(group.truth, group.prediction, bins=(bins, bins))
                densities.extend(dict(**base, truth_left=xb[i], prediction_left=yb[j], width=.1, targets=int(hist[i,j]),
                                      total_denominator=len(group), outside_plot=int(len(group)-hist.sum()))
                                 for i, j in zip(*np.nonzero(hist)))
    return pd.DataFrame(reliability), pd.DataFrame(pits), pd.DataFrame(densities)


def random_endpoint(frame, mapping):
    require(not frame.duplicated(PAIR_KEYS).any(), 'Random physical snapshots must not be duplicated')
    scores = frame.assign(abs_error=(frame.prediction-frame.truth).abs()).groupby(
        ['dataset_id', 'event_id', 'elapsed_time', 'geometry_protocol']).abs_error.mean().reset_index()
    expected = mapping[['dataset_id', 'event_id', 'elapsed_time', 'draw']].merge(pd.DataFrame({'geometry_protocol': ['normal','random']}), how='cross')
    joined = expected.merge(scores, on=['dataset_id', 'event_id', 'elapsed_time', 'geometry_protocol'], how='left', validate='many_to_one')
    require(joined.abs_error.notna().all(), 'Random draw missing; no intersection scoring')
    events = joined.groupby(['dataset_id', 'event_id', 'geometry_protocol']).abs_error.mean()
    return float(events.groupby('geometry_protocol').mean().mean())


def field_summary(table):
    columns = ['field_mean_error', 'level_mse', 'shape_mse', 'field_mse', 'centered_mae', 'centered_rmse',
               'pairwise_delta_mae', 'range_error', 'range_ratio', 'equal_distance_delta_mae', 'equal_distance_sign_accuracy']
    rows = []
    for key, group in table.groupby(['target_group', 'geometry_protocol', 'elapsed_time']):
        valid = group[group.status.eq('supported')]
        row = dict(target_group=key[0], geometry_protocol=key[1], elapsed_time=key[2], fields_attempted=len(group),
                   fields_eligible=len(valid), fields_excluded=len(group)-len(valid),
                   eligible_targets=int(valid.targets.sum()), events=valid[['dataset_id','event_id']].drop_duplicates().shape[0])
        for col in columns:
            row[col + '_field_mean'] = float(valid[col].mean()) if len(valid) and valid[col].notna().any() else None
            row[col + '_field_denominator'] = int(valid[col].notna().sum()) if len(valid) else 0
        row['field_equal_weight_rmse'] = float(np.sqrt(valid.field_mse.mean())) if len(valid) else None
        row['equal_distance_pair_count'] = int(valid.equal_distance_pairs.sum()) if len(valid) else 0
        row['sign_pair_denominator'] = int(valid.equal_distance_sign_denominator.sum()) if len(valid) else 0
        rows.append(row)
    return pd.DataFrame(rows)


def paired_fields(left, right, draws=5000):
    keys = DECISION_KEYS + ['target_group']
    columns = ['level_mse', 'shape_mse', 'field_mse', 'centered_mae', 'pairwise_delta_mae', 'range_error', 'equal_distance_delta_mae']
    a, b = left[left.status.eq('supported')], right[right.status.eq('supported')]
    joined = a.merge(b, on=keys, how='outer', suffixes=('_left', '_right'), indicator=True, validate='one_to_one')
    require(joined._merge.eq('both').all(), 'Field populations differ')
    require(joined.targets_left.eq(joined.targets_right).all(), 'Field target counts differ')
    rows = []
    for role, group in joined.groupby('target_group'):
        for metric in columns:
            good = group[metric+'_left'].notna() & group[metric+'_right'].notna()
            subset = group.loc[good].copy()
            if not len(subset):
                continue
            subset['delta'] = subset[metric+'_right'] - subset[metric+'_left']
            # Each event-field gets one weight; then equal available time/protocol cells.
            pivot = subset.pivot(index=['dataset_id','event_id'], columns=['geometry_protocol','elapsed_time'], values='delta')
            values, counts = pivot.fillna(0).to_numpy(), pivot.notna().to_numpy().astype(int)
            rng = np.random.default_rng(20261001)
            sampled = []
            for start in range(0, draws, 100):
                indices = rng.integers(0, len(pivot), size=(min(100, draws-start),len(pivot)))
                numerator, denominator = values[indices].sum(1), counts[indices].sum(1)
                valid = (denominator > 0).all(1)
                sampled.extend((numerator[valid]/denominator[valid]).mean(1).tolist())
            rows.append(dict(target_group=role, metric=metric, events=len(pivot), paired_fields=len(subset),
                delta=float((values.sum(0)/counts.sum(0)).mean()), ci_low=float(np.quantile(sampled,.025)), ci_high=float(np.quantile(sampled,.975)),
                empty_cell_replicates=draws-len(sampled), draws=draws, rng_seed=20261001,
                weighting='field equal within cell, then equal available cells; paired exact field population',
                cluster='dataset_id,event_id; all station/time/protocol rows move together'))
    return pd.DataFrame(rows)


def station_recovery(frame, reference, draws=5000, min_events=10):
    """Station-equal recovery; bootstrap whole events then recompute station means."""
    rows, summary, stability = [], [], []
    for role in ('noninput', 'untriggered_noninput'):
        subset = target_groups(frame)[role].copy()
        x = reference_design(subset)
        subset['no_site'] = x @ np.asarray(reference['no_site_coefficients'])
        subset['train_site'] = x @ np.asarray(reference['site_reference_coefficients']) + subset.station_id.map(reference['train_site_effects']).fillna(0)
        for anchor in ('no_site', 'train_site'):
            data = subset.copy()
            for name, original in (('observed','truth'), ('predicted','prediction')):
                residual = data[original] - data[anchor]
                data[name] = residual - residual.groupby([data[k] for k in DECISION_KEYS]).transform('mean')
            for (protocol,time), group in data.groupby(['geometry_protocol','elapsed_time']):
                require(not group.duplicated(['dataset_id','event_id','station_id']).any(), 'Station event identity repeated')
                aggregates = group.groupby('station_id').agg(events=('event_id','nunique'), observed=('observed','mean'), predicted=('predicted','mean')).reset_index()
                aggregates['sparse'] = aggregates.events.lt(min_events)
                aggregates['error_of_event_mean'] = aggregates.predicted-aggregates.observed
                aggregates['station_in_train_label_cohort'] = aggregates.station_id.isin(reference['train_site_effects'])
                for column,value in [('target_group',role),('anchor',anchor),('geometry_protocol',protocol),('elapsed_time',time)]: aggregates[column]=value
                rows.append(aggregates)
                dense = aggregates.loc[~aggregates['sparse']]
                station_ids = dense.station_id.tolist()
                if len(station_ids) < 2:
                    summary.append(dict(target_group=role,anchor=anchor,geometry_protocol=protocol,elapsed_time=time,status='INCOMPLETE',dense_stations=len(dense),sparse_stations=int(aggregates['sparse'].sum())))
                    continue
                matrices=[]
                for name in ('observed','predicted'):
                    matrices.append(group.pivot(index=['dataset_id','event_id'],columns='station_id',values=name).reindex(columns=station_ids))
                obs,pred=matrices;support=obs.notna().to_numpy().astype(float)
                delta=pred.fillna(0).to_numpy()-obs.fillna(0).to_numpy()
                from scipy.sparse import csr_matrix
                support,delta=csr_matrix(support),csr_matrix(delta)
                rng=np.random.default_rng(20261001);sampled=[];station_samples=[]
                for start in range(0,draws,100):
                    indices=rng.integers(0,len(obs),size=(min(100,draws-start),len(obs)))
                    weights=np.zeros((len(indices),len(obs)),dtype=np.float64)
                    for j,ids in enumerate(indices): weights[j]=np.bincount(ids,minlength=len(obs))
                    counts=(support.T@weights.T).T
                    numerators=(delta.T@weights.T).T
                    estimates=np.divide(numerators,counts,out=np.full(counts.shape,np.nan),where=counts>0)
                    complete=np.isfinite(estimates).all(1)
                    sampled.extend(np.abs(estimates[complete]).mean(1).tolist());station_samples.append(estimates)
                station_samples=np.concatenate(station_samples)
                require(len(sampled)>0,'No complete event-bootstrap station replicate')
                # Preserve sparse stations; their recovery is exploratory and has no fabricated CI.
                low,high=np.nanquantile(station_samples,[.025,.975],axis=0)
                for station,l,h in zip(station_ids,low,high):
                    aggregates.loc[aggregates.station_id.eq(station),'event_mean_error_ci_low']=l
                    aggregates.loc[aggregates.station_id.eq(station),'event_mean_error_ci_high']=h
                y,p=dense.observed.to_numpy(),dense.predicted.to_numpy()
                summary.append(dict(target_group=role,anchor=anchor,geometry_protocol=protocol,elapsed_time=time,status='supported',
                    dense_stations=len(dense),sparse_stations=int(aggregates['sparse'].sum()),station_equal_recovery_mae=float(np.abs(p-y).mean()),
                    correlation=float(np.corrcoef(y,p)[0,1]) if np.std(y)>0 and np.std(p)>0 else None,
                    prediction_on_observed_slope=float(np.cov(y,p,ddof=0)[0,1]/np.var(y)) if np.var(y)>0 else None,
                    sign_denominator=int((np.abs(y)>1e-8).sum()), sign_accuracy=float(np.mean(np.sign(y[np.abs(y)>1e-8])==np.sign(p[np.abs(y)>1e-8]))) if (np.abs(y)>1e-8).any() else None,
                    ci_low=float(np.quantile(sampled,.025)),ci_high=float(np.quantile(sampled,.975)),draws=draws,
                    incomplete_station_replicates=draws-len(sampled),bootstrap='event clusters recompute station means; fixed original >=10-unique-event station cohort'))
            data['azimuth_quadrant']=np.floor((np.rad2deg(np.arctan2((data.longitude-data.event_longitude)*np.cos(np.deg2rad(data.latitude)),data.latitude-data.event_latitude))%360)/90).astype(int)
            data['strength']=pd.cut(data.truth,[-np.inf,-2,-1.2,0,np.inf],right=True)
            stable=data.groupby(['station_id','geometry_protocol','elapsed_time','azimuth_quadrant','strength'],observed=True).agg(
                events=('event_id','nunique'),rows=('event_id','size'),observed=('observed','mean'),predicted=('predicted','mean')).reset_index()
            stable['anchor'],stable['target_group']=anchor,role;stability.append(stable)
    return pd.concat(rows,ignore_index=True),pd.DataFrame(summary),pd.concat(stability,ignore_index=True)
