#!/usr/bin/env python3
"""Versioned V01 validation closure report; preserve all historical outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from tools.audit_v01_validation_contract import checkpoint_contract, npz_arrays
from tools.summarize_v01_results import point_metrics, field_metrics
from tools.v01_validation_contract import PROTOCOL, sha256, write_json

KEYS = ['event_id', 'query_sensor_id', 'time_s', 'absolute_cutoff_utc']
WINDOWS = {'all7': None, 'early135': [1, 3, 5], 't1': [1], 't3': [3], 't5': [5]}
STRATA = ('all', 'above_threshold', 'below_threshold', 'untriggered', 'single_source')


def read_records(path):
    with Path(path).open(encoding='utf8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_rows(path, identities):
    arrays = npz_arrays(path)
    if 'pga_target_valid' not in arrays:
        return pd.DataFrame(columns=KEYS + ['truth', 'prediction', 'nll', 'prob', 'sigma',
                                          'target_type', 'input_count', 'abs_error', 'remote_sensor_exclusion'])
    valid = arrays['pga_target_valid'].astype(bool)
    truth = arrays['pga_label'].squeeze(-1)
    pred = arrays['pga_mu_best']
    if np.any(valid & (~np.isfinite(truth) | ~np.isfinite(pred))):
        raise ValueError('Nonfinite valid prediction/label')
    records = {int(r['global_index']): r for r in identities}
    if len(records) != len(identities):
        raise ValueError('Duplicate ledger request index')
    row_ids = arrays['event_index'].astype(int)
    if len(np.unique(row_ids)) != len(row_ids):
        raise ValueError('Duplicate NPZ requested index')
    if set(row_ids) != {i for i,r in records.items() if r['outcome'] == 'predicted'}:
        raise ValueError('Ledger predicted requests differ from NPZ indices')
    query_ids = np.full(valid.shape, '', dtype=object)
    cutoff = np.empty(len(row_ids), float)
    remote = np.zeros(valid.shape, bool)
    for row, index in enumerate(row_ids):
        record = records[int(index)]
        info = record.get('identity')
        if info is None:
            qids = record['query_sensor_ids']
            source_pair_ids = record['source_paired_acc_ids']
            cutoff[row] = record['absolute_cutoff_utc']
        else:
            qids = info['v01_query_sensor_id']
            source_pair_ids = info['v01_source_paired_acc_id']
            cutoff[row] = info['v01_absolute_cutoff_utc']
            if cutoff[row] != record['absolute_cutoff_utc']:
                raise ValueError('Planned vs actual UTC cutoff mismatch')
        if str(arrays['event_id'][row]) != record['event_id']:
            raise ValueError('Neighbour event substitution in closure evidence')
        if float(arrays['realtime_requested_elapsed_time'][row]) != record['time_s']:
            raise ValueError('Neighbour time substitution in closure evidence')
        query_ids[row] = qids
        if np.any(valid[row] & (query_ids[row] == '')):
            raise ValueError('Missing query sensor identity')
        selected_acc = {str(x) for x in source_pair_ids if x}
        # Sensor exclusion is NOT certification of different physical sites.
        remote[row] = [bool(x) and str(x) not in selected_acc for x in qids]
    row, slot = np.where(valid)
    frame = pd.DataFrame({
        'event_id': arrays['event_id'][row].astype(str), 'query_sensor_id': query_ids[row,slot],
        'time_s': arrays['realtime_requested_elapsed_time'][row],
        'absolute_cutoff_utc': cutoff[row], 'truth': truth[row,slot],
        'prediction': pred[row,slot], 'target_type': arrays['realtime_target_type'][row,slot].astype(int),
        'input_count': arrays['station_valid'].astype(bool).sum(axis=1)[row],
        'sigma': arrays['pga_sigma'][row,slot], 'nll': arrays['pga_nll_log10_mps2'][row,slot],
        'prob': arrays['pga_prob_ge_threshold'][row,slot], 'remote_sensor_exclusion': remote[row,slot],
    })
    frame['abs_error'] = abs(frame.prediction-frame.truth)
    if frame.duplicated(KEYS).any():
        raise ValueError('Nonunique physical sensor/cutoff/request-time keys')
    return frame


def pairing(left, right):
    for frame in (left, right):
        if frame.duplicated(KEYS).any():
            raise ValueError('Pairing must be one-to-one')
    outer = left.merge(right, on=KEYS, how='outer', suffixes=('_left', '_right'),
                       validate='one_to_one', indicator=True)
    counts = outer['_merge'].value_counts()
    matched = outer.loc[outer['_merge'] == 'both'].drop(columns='_merge').copy()
    if not np.array_equal(matched.truth_left, matched.truth_right):
        raise ValueError('Paired label mismatch')
    audit = {'left_targets': len(left), 'right_targets': len(right), 'matched': len(matched),
             'left_only': int(counts.get('left_only', 0)), 'right_only': int(counts.get('right_only', 0)),
             'matched_events': int(matched.event_id.nunique()), 'keys': KEYS}
    unmatched = outer.loc[outer['_merge'] != 'both', KEYS + ['_merge']]
    return matched, audit, unmatched


def subset(frame, stratum, window, side=''):
    suffix = '_' + side if side else ''
    selected = np.ones(len(frame), bool)
    if WINDOWS[window] is not None:
        selected &= frame.time_s.isin(WINDOWS[window])
    truth = frame['truth' + suffix]
    if stratum == 'above_threshold':
        selected &= truth >= -1.2
    elif stratum == 'below_threshold':
        selected &= truth < -1.2
    elif stratum == 'untriggered':
        selected &= frame['target_type' + suffix] == 2
    elif stratum == 'single_source':
        selected &= frame['input_count' + suffix] == 1
    return np.asarray(selected, bool)


def cluster_statistics(paired, draws=5000, seed=20260915):
    """Target-micro and event-macro CIs + uncorrected MSE decomposition.

    An event is the resampling unit across all times and targets. Macro RMSE
    means mean(event RMSE), not sqrt(mean(event MSE)). No bias correction.
    """
    if paired.empty:
        return [], pd.DataFrame()
    pieces = {'event_id': paired.event_id, 'count': np.ones(len(paired))}
    for side in ('left', 'right'):
        error = paired['prediction_' + side].to_numpy() - paired['truth_' + side].to_numpy()
        pieces.update({side + '_error': error, side + '_mae': abs(error), side + '_mse': error**2,
                       side + '_nll': paired['nll_' + side].to_numpy(),
                       side + '_brier': (paired['prob_' + side].to_numpy() - (paired['truth_' + side] >= -1.2))**2})
    sufficient = pd.DataFrame(pieces).groupby('event_id', sort=True).sum()
    if not np.isfinite(sufficient.to_numpy()).all():
        raise ValueError('Nonfinite probability/error sufficient statistics')
    count = sufficient['count'].to_numpy()
    n = len(count)
    rng = np.random.default_rng(seed)
    dist = {key: np.empty(draws) for key in
            [averaging+'_'+metric for averaging in ('micro', 'macro') for metric in ('mae', 'rmse', 'nll', 'brier')]
            + [side+'_'+metric for side in ('left', 'right', 'delta')
               for metric in ('mse', 'bias', 'bias_squared', 'centered_variance')]}
    for start in range(0, draws, 128):
        size = min(128, draws-start)
        sample = rng.integers(0, n, size=(size, n))
        denominator = count[sample].sum(axis=1)
        evaluated = {}
        for side in ('left', 'right'):
            for metric in ('mae', 'mse', 'nll', 'brier', 'error'):
                totals = sufficient[side+'_'+metric].to_numpy()
                evaluated[side+'_micro_'+metric] = totals[sample].sum(axis=1)/denominator
                means = totals/count
                if metric == 'mse':
                    means = np.sqrt(means)
                evaluated[side+'_macro_'+metric] = means[sample].mean(axis=1)
            for metric in ('mse', 'bias', 'bias_squared', 'centered_variance'):
                mse = evaluated[side+'_micro_mse']; bias = evaluated[side+'_micro_error']
                value = {'mse': mse, 'bias': bias, 'bias_squared': bias**2,
                         'centered_variance': mse-bias**2}[metric]
                dist[side+'_'+metric][start:start+size] = value
        for metric in ('mae', 'rmse', 'nll', 'brier'):
            raw = 'mse' if metric == 'rmse' else metric
            for averaging in ('micro', 'macro'):
                a = evaluated['left_'+averaging+'_'+raw]
                b = evaluated['right_'+averaging+'_'+raw]
                if metric == 'rmse' and averaging == 'micro':
                    a, b = np.sqrt(a), np.sqrt(b)
                dist[averaging+'_'+metric][start:start+size] = b-a
        for metric in ('mse', 'bias', 'bias_squared', 'centered_variance'):
            dist['delta_'+metric][start:start+size] = (dist['right_'+metric][start:start+size]
                                                      - dist['left_'+metric][start:start+size])
    rows = []
    point = {}
    for side in ('left', 'right'):
        bias = sufficient[side+'_error'].sum()/count.sum()
        mse = sufficient[side+'_mse'].sum()/count.sum()
        point.update({side+'_mse': mse, side+'_bias': bias,
                      side+'_bias_squared': bias**2, side+'_centered_variance': mse-bias**2})
    for metric in ('mse', 'bias', 'bias_squared', 'centered_variance'):
        point['delta_'+metric] = point['right_'+metric] - point['left_'+metric]
    for averaging in ('micro', 'macro'):
        for metric in ('mae', 'rmse', 'nll', 'brier'):
            raw = 'mse' if metric == 'rmse' else metric
            values = []
            for side in ('left', 'right'):
                totals = sufficient[side+'_'+raw].to_numpy()
                value = totals.sum()/count.sum() if averaging == 'micro' else np.mean(
                    np.sqrt(totals/count) if metric == 'rmse' else totals/count)
                if metric == 'rmse' and averaging == 'micro':
                    value = np.sqrt(value)
                values.append(value)
            point[averaging+'_'+metric] = values[1]-values[0]
    for key, samples in dist.items():
        rows.append({'statistic': key, 'estimate': float(point[key]),
                     'ci_low': float(np.quantile(samples, .025)), 'ci_high': float(np.quantile(samples, .975)),
                     'paired_targets': len(paired), 'paired_events': n, 'bootstrap_draws': draws,
                     'bootstrap_seed': seed, 'ci_type': 'event-cluster percentile, paired',
                     'bias_corrected_predictions': False})
    return rows, sufficient.reset_index()


def additional_field_metrics(frame):
    fields = []
    for (event, time), group in frame.groupby(['event_id', 'time_s']):
        if len(group) < 5:
            continue
        true_span = np.quantile(group.truth, .95) - np.quantile(group.truth, .05)
        pred_span = np.quantile(group.prediction, .95) - np.quantile(group.prediction, .05)
        if true_span > 1e-8:
            fields.append({'event_id': event, 'time_s': time, 'input_count': group.input_count.iloc[0],
                           'p95_p05_ratio': pred_span/true_span})
    fields = pd.DataFrame(fields, columns=['event_id', 'time_s', 'input_count', 'p95_p05_ratio'])
    rows = []
    for window in ('all7', 'early135'):
        for stratum in ('all_inputs', 'single_input'):
            mask = np.ones(len(fields), bool)
            if window == 'early135':
                mask &= fields.time_s.isin([1,3,5])
            if stratum == 'single_input':
                mask &= fields.input_count == 1
            selected = fields[mask]
            rows.append({'time_window': window, 'group': stratum,
                         'eligible_fields': len(selected), 'events': selected.event_id.nunique(),
                         'p95_p05_ratio_mean': selected.p95_p05_ratio.mean() if len(selected) else None,
                         'definition': 'mean across eligible event/time fields; >=5 queries; nonzero truth span'})
    return rows


def figures(frames, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    plt.rcParams.update({'font.size': 9, 'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False})
    edges = np.linspace(-3, 1, 121)
    residual_edges = np.linspace(-2, 2, 161)
    geometries = [g for g in ('normal','random') if any(a+'__'+a+'__'+g in frames for a in ('vfull','vmissing'))]
    panels = [(arm+'__'+arm+'__'+geometry, geometry, arm)
              for geometry in geometries for arm in ('vfull','vmissing')]
    counts = []; histograms = []
    for cell, geometry, arm in panels:
        if cell not in frames:
            continue
        frame = frames[cell]
        hist, _, _ = np.histogram2d(frame.truth, frame.prediction, bins=(edges, edges))
        histograms.append((cell, geometry, arm, frame, hist))
        for x,y in np.ndindex(hist.shape):
            counts.append({'cell': cell, 'truth_left': edges[x], 'prediction_left': edges[y],
                           'bin_width': edges[1]-edges[0], 'count': int(hist[x,y])})
    pd.DataFrame(counts).to_csv(output/'density_bin_counts.csv', index=False)
    max_count = max((h.max() for *_,h in histograms), default=1)
    fig, axes = plt.subplots(len(geometries), 2, figsize=(7.2, 3.2*len(geometries)),
                             sharex=True, sharey=True, constrained_layout=True, squeeze=False)
    outside = []
    for ax, (cell, geometry, arm, frame, hist) in zip(axes.ravel(), histograms):
        image = ax.pcolor(edges, edges, np.ma.masked_less(hist.T, 1), cmap='viridis',
                             norm=LogNorm(vmin=1, vmax=max(2,max_count)), rasterized=False)
        ax.plot([-3,1], [-3,1], '--', color='black', lw=.8)
        ax.set(xlim=(-3,1), ylim=(-3,1), aspect='equal', xlabel='True PGA [log10(m/s^2)]',
               ylabel='Predicted PGA [log10(m/s^2)]', title=cell.replace('__',' / '))
        ax.text(.03, .97, 'N={}\nMAE={:.3f}'.format(len(frame), frame.abs_error.mean()),
                transform=ax.transAxes, va='top', bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
        outside.append({'cell': cell, 'targets': len(frame), 'binned_targets': int(hist.sum()),
                        'outside_fixed_axes': len(frame)-int(hist.sum())})
    if histograms:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.7, label='Target count per bin (log scale)')
    fig.savefig(output/'pga_density.svg'); fig.savefig(output/'pga_density.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(len(geometries), 2, figsize=(7.2, 2.4*len(geometries)),
                             sharex=True, constrained_layout=True, squeeze=False)
    residual_counts = []
    for row, geometry in enumerate(geometries):
        for col, positive in enumerate((False, True)):
            ax = axes[row,col]
            for arm, color, style in (('vfull','#0072B2','-'), ('vmissing','#D55E00','--')):
                cell = arm+'__'+arm+'__'+geometry
                if cell not in frames:
                    continue
                frame = frames[cell]
                selected = frame[(frame.truth >= -1.2) == positive]
                err = selected.prediction-selected.truth
                hist, _ = np.histogram(err, bins=residual_edges)
                for i,count in enumerate(hist):
                    residual_counts.append({'cell': cell, 'stratum': 'above' if positive else 'below',
                                            'residual_left': residual_edges[i], 'bin_width': .025,
                                            'count': int(count), 'total_targets': len(selected)})
                density = hist / max(1,len(selected)) / np.diff(residual_edges)
                ax.step(residual_edges[:-1], density, where='post', color=color, ls=style,
                        label=arm+' (N='+str(len(selected))+')')
                outside.append({'cell': cell, 'stratum': 'above' if positive else 'below',
                                'targets': len(selected), 'binned_targets': int(hist.sum()),
                                'outside_fixed_axes': len(selected)-int(hist.sum())})
            ax.axvline(0, color='black', lw=.7, ls=':')
            ax.set(xlim=(-2,2), xlabel='Prediction - truth [log10(m/s^2)]', ylabel='Density',
                   title=geometry+' / '+('truth >= -1.2' if positive else 'truth < -1.2'))
            ax.legend(fontsize=8, loc='upper left')
    pd.DataFrame(residual_counts).to_csv(output/'residual_bin_counts.csv', index=False)
    write_json(output/'figure_count_audit.json', outside)
    fig.savefig(output/'residual_density.svg'); fig.savefig(output/'residual_density.png', dpi=150)
    plt.close(fig)


def offline_random_review(run_root, output):
    """P1 offline addendum, explicitly NOT sensor-certified closure or A/V."""
    from tools.summarize_v01_results import load_cell
    output.mkdir(parents=True, exist_ok=False)
    cells = ['vfull__vfull__random','vfull__vmissing__random',
             'vmissing__vfull__random','vmissing__vmissing__random']
    frames = {}; sources = []; metric_rows = []
    for cell in cells:
        path = run_root/'eval_retry1'/(cell+'.npz')
        frame, _ = load_cell(path)
        frames[cell] = frame
        sources.append({'cell':cell, 'file':str(path), 'sha256':sha256(path)})
        for window in WINDOWS:
            for stratum in STRATA:
                selected = frame[subset(frame,stratum,window)]
                metric_rows.append({'cell':cell,'stratum':stratum,'time_window':window,**point_metrics(selected)})
    pd.DataFrame(metric_rows).to_csv(output/'metrics_by_stratum.csv',index=False)
    ci_rows = []; audit_rows = []; sufficient_rows = []
    coordinate_keys = ['event_id','time_s','query_lat','query_lon','query_depth']
    ff,fm,mf,mm = cells
    for left,right in ((ff,fm),(ff,mf),(ff,mm),(mf,mm),(fm,mm)):
        for frame in (frames[left],frames[right]):
            if frame.duplicated(coordinate_keys).any():
                raise ValueError('Nonunique coordinate keys; offline comparison aborted')
        outer = frames[left].merge(frames[right],on=coordinate_keys,how='outer',
                                   suffixes=('_left','_right'),validate='one_to_one',indicator=True)
        matched = outer[outer['_merge'] == 'both'].copy()
        for key in ('truth','current_sample','first_pick_sample'):
            if not np.array_equal(matched[key+'_left'],matched[key+'_right']):
                raise ValueError('Legacy coordinate-paired evidence mismatch: '+key)
        name = left+'--'+right
        counts = outer['_merge'].value_counts()
        audit_rows.append({'comparison':name,'keys':coordinate_keys,'matched':len(matched),
                           'left_only':int(counts.get('left_only',0)), 'right_only':int(counts.get('right_only',0)),
                           'sensor_id_certified':False, 'absolute_utc_certified':False})
        for window in WINDOWS:
            for stratum in STRATA:
                selected = matched[subset(matched,stratum,window,'left') & subset(matched,stratum,window,'right')]
                rows, sufficient = cluster_statistics(selected)
                tags = {'comparison':name,'stratum':stratum,'time_window':window,
                        'pair_protocol':'legacy coordinate + requested time; label/current/first pick verified; sensor/UTC pending'}
                ci_rows.extend({**tags,**r} for r in rows)
                if len(sufficient):
                    sufficient_rows.append(sufficient.assign(**tags))
    pd.DataFrame(ci_rows).to_csv(output/'paired_event_cluster_ci_and_mse_decomposition.csv',index=False)
    sufficient = pd.concat(sufficient_rows,ignore_index=True)
    sufficient.to_csv(output/'paired_event_sufficient_statistics.csv',index=False)
    # Lightweight, reproducible principal comparison for Git/ChatGPT review.
    primary = sufficient[(sufficient.comparison == ff+'--'+mm)
                         & sufficient.time_window.isin(['all7','early135'])]
    primary.to_csv(output/'primary_ff_mm_event_sufficient_statistics.csv',index=False)
    write_json(output/'one_to_one_coordinate_outer_match_audit.json',audit_rows)
    write_json(output/'input_provenance.json',sources)
    write_json(output/'status.json',{'task_id':'20261004-v01-validation-closure',
               'base_commit':'74c55aa5442b4200961c88ceee3d11af5033275d',
               'scope':'pure offline P1 interpretation of four existing velocity random exports',
               'sensor_id_certified':False,'normal_results_available':False,'aa_included':False,
               'closure_complete':False,'bootstrap_draws':5000,'bootstrap_seed':20260915,
               'predictions_bias_corrected':False,'pga_coordinate':'log10(m/s^2)',
               'threshold':-1.2,'full_record_centering_unchanged':True})
    figures(frames,output)
    print('Offline P1 done; sensor/absolute UTC certification and normal/A/V closure remain pending.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--closure-root', type=Path, required=True)
    parser.add_argument('--offline-random-only', action='store_true',
                        help='Pure NPZ P1 addendum; no checkpoint/cache/test reads or closure claims')
    args = parser.parse_args()
    if args.offline_random_only:
        offline_random_review(args.run_root,args.closure_root)
        return
    root, old = args.closure_root, args.run_root/'eval_retry1'
    output = root/'report'
    output.mkdir(exist_ok=False)
    write_json(output/'checkpoints_after.json', checkpoint_contract(args.run_root))
    audited_old = json.loads((root/'audit/reused_random_artifacts.json').read_text())
    for record in audited_old:
        if sha256(record['file']) != record['sha256']:
            raise ValueError('Old random artifact changed since audit: '+record['cell'])
    before_weights = json.loads((root/'audit/checkpoints_before.json').read_text())
    after_weights = json.loads((output/'checkpoints_after.json').read_text())
    if before_weights != after_weights:
        raise ValueError('Checkpoint contract differs before vs after validation')
    plan = json.loads((root/'submission_plan.json').read_text())
    cells = list(plan['cells']) + [a+'__'+b+'__random' for a,b in
                                   (('vfull','vfull'),('vfull','vmissing'),('vmissing','vfull'),('vmissing','vmissing'))]
    frames = {}; inventory = []; coverage = []; metrics = []; fields = []; p95 = []
    for cell in cells:
        reuse = cell not in plan['cells']
        path = (old if reuse else root/'eval')/(cell+'.npz')
        ledger = root/'audit'/(cell+'.plan.jsonl') if reuse else Path(str(path)+'.val.requests.jsonl')
        records = read_records(ledger)
        if not reuse:
            summary = json.loads(Path(str(ledger)+'.summary.json').read_text())
            if not summary['complete'] or summary['outcomes']['implementation_error']:
                raise ValueError('Incomplete/error request ledger: ' + cell)
            if summary['requested'] != len(records) or len(records) != sum(summary['outcomes'].values()):
                raise ValueError('Ledger summary accounting mismatch')
        frame = load_rows(path, records)
        frames[cell] = frame
        requested_events = {r['event_id'] for r in records}
        no_prediction = [r for r in records if r['outcome'] != 'predicted']
        missing_events = requested_events - set(frame.event_id)
        requested_queries = sum(r['expected_query_count'] for r in records)
        predicted = sum(r['outcome'] == 'predicted' for r in records)
        coverage.append({'cell': cell, 'requested_events': len(requested_events),
                         'requested_event_times': len(records), 'predicted_event_times': predicted,
                         'prediction_rate': predicted/len(records) if records else None,
                         'no_prediction_rate': len(no_prediction)/len(records) if records else None,
                         'available_events': frame.event_id.nunique(), 'selected_predicted_targets': len(frame),
                         'missing_events': sorted(missing_events), 'no_prediction_requests': no_prediction,
                         'candidate_query_opportunities_before_sampling': requested_queries,
                         'candidate_opportunities_in_abstained_requests': sum(r['expected_query_count'] for r in no_prediction),
                         'opportunities_are_not_selected_target_denominator': True,
                         'true_physical_site_common_remote': 'NOT_AVAILABLE: cache sensor pairing is not site identity certification'})
        inventory.append({'cell': cell, 'npz': str(path), 'sha256': sha256(path), 'ledger_sha256': sha256(ledger),
                          'origin': 'legacy random, identities certified and numeric signature audit' if reuse else PROTOCOL})
        for window in WINDOWS:
            for stratum in STRATA:
                selected = frame[subset(frame, stratum, window)]
                metrics.append({'cell': cell, 'time_window': window, 'stratum': stratum,
                                **point_metrics(selected),
                                'event_macro_rmse': float(selected.groupby('event_id').apply(
                                    lambda g: np.sqrt(np.mean((g.prediction-g.truth)**2))).mean()) if len(selected) else None,
                                'event_macro_nll': float(selected.groupby('event_id').nll.mean().mean()) if len(selected) else None,
                                'event_macro_brier': float(selected.assign(brier=(selected.prob-(selected.truth >= -1.2))**2)
                                                          .groupby('event_id').brier.mean().mean()) if len(selected) else None})
        if len(frame) and frame.groupby(['event_id','time_s']).size().ge(5).any():
            fields.extend({'cell': cell, 'definition': 'np.ptp ratio median, historical definition retained', **r}
                          for r in field_metrics(frame))
            p95.extend({'cell': cell, **r} for r in additional_field_metrics(frame))
    pd.DataFrame(metrics).to_csv(output/'metrics_by_stratum.csv', index=False)
    pd.DataFrame(fields).to_csv(output/'field_ptp_ratio_median.csv', index=False)
    pd.DataFrame(p95).to_csv(output/'field_p95_p05_ratio_mean.csv', index=False)
    write_json(output/'coverage_and_denominators.json', coverage)
    write_json(output/'input_provenance.json', inventory)
    match_audits = []; ci_rows = []; sufficient_rows = []; avmetrics = []
    for geometry in ('normal','random'):
        suffix = '__'+geometry
        ff, fm, mf, mm, aa = [s+suffix for s in ('vfull__vfull','vfull__vmissing','vmissing__vfull','vmissing__vmissing','apair__apair')]
        comparisons = [(ff,fm), (ff,mf), (ff,mm), (mf,mm), (fm,mm)]
        if aa in frames:
            comparisons += [(aa,ff), (aa,mm)]
        for left,right in comparisons:
            paired, audit, unmatched = pairing(frames[left], frames[right])
            name = left+'--'+right
            match_audits.append({'comparison': name, **audit})
            unmatched.to_csv(output/(name+'.unmatched.csv'), index=False)
            if left == aa:
                common = paired.copy()
                common_remote = paired[paired.remote_sensor_exclusion_left & paired.remote_sensor_exclusion_right]
                for label, values in (('common_available_queries', common),
                                      ('common_remote_sensor_exclusion_proxy', common_remote)):
                    for side, cell in (('left',left),('right',right)):
                        renamed = values.rename(columns={k+'_'+side:k for k in
                                                         ('truth','prediction','nll','prob','sigma','abs_error')})
                        avmetrics.append({'comparison': name, 'cell': cell, 'population': label,
                                          **point_metrics(renamed),
                                          'selection_bias_warning': 'conditional on both systems predicting; full request coverage separately reported'})
            for window in WINDOWS:
                for stratum in STRATA:
                    eligible = subset(paired,stratum,window,'left') & subset(paired,stratum,window,'right')
                    selected = paired[eligible]
                    rows, sufficient = cluster_statistics(selected)
                    common_fields = {'comparison': name, 'time_window': window, 'stratum': stratum,
                                     'stratum_policy': 'both arms eligible; unpaired/eligibility counts explicit',
                                     'left_stratum_matched_targets': int(subset(paired,stratum,window,'left').sum()),
                                     'right_stratum_matched_targets': int(subset(paired,stratum,window,'right').sum())}
                    ci_rows.extend({**common_fields, **r} for r in rows)
                    if len(sufficient):
                        sufficient_rows.append(sufficient.assign(**common_fields))
    pd.DataFrame(ci_rows).to_csv(output/'paired_event_cluster_ci_and_mse_decomposition.csv', index=False)
    if sufficient_rows:
        pd.concat(sufficient_rows, ignore_index=True).to_csv(output/'paired_event_sufficient_statistics.csv', index=False)
    pd.DataFrame(avmetrics).to_csv(output/'acc_velocity_common_population_metrics.csv', index=False)
    write_json(output/'one_to_one_outer_match_audit.json', match_audits)
    figures(frames, output)
    complete = 'apair__apair__random' in frames
    write_json(output/'report_status.json', {'protocol': PROTOCOL, 'normal_cells_complete': True,
               'aa_random_complete': complete, 'closure_complete': complete,
               'train_or_test_executed': False, 'threshold': -1.2, 'bootstrap_draws': 5000,
               'bootstrap_seed': 20260915, 'pair_keys': KEYS,
               'macro_rmse_definition': 'mean event RMSE',
               'legacy_random_reuse_scope': 'clock-copy route unchanged + production/real-cache input-label signatures; no model rerun',
               'limitations': ['validation reused for development', 'one seed and eight epochs',
                               'different A/V instruments/depths/sites/response, not a causal padding contrast',
                               'full-record centering and legacy cutout+1 inherited, not changed here',
                               'historical encoder/source hashes may remain unknown',
                               'sensor-ID exclusion is not certified true physical-site remote']})
    report = ('# V01 validation closure\n\nValidation only; fixed epoch8/step1496 weights. '
              'No training, preflight, or held-out test.\n\n'
              'Read metrics_by_stratum.csv with coverage_and_denominators.json and the outer-match audit. '
              'A/V common-available comparisons are conditional and carry selection bias; '
              'common remote is a paired-sensor exclusion proxy, not certified physical-site separation.\n\n'
              'MSE = bias squared + centered error variance is recomputed from saved predictions with event-cluster CIs; '
              'no bias correction is applied. Above-threshold means target PGA >= -1.2 log10(m/s^2), not event magnitude.\n\n'
              'Historical np.ptp ratio median and new P95-P05 ratio mean are different quantities in separate files. '
              'Density/residual SVGs use fixed axes and audited bin counts; out-of-view points are reported.\n\n'
              'AA random closure: '+('complete' if complete else 'PENDING; old duplicate export excluded')+'.\n')
    (output/'RESULT_REVIEW.md').write_text(report, encoding='utf8')
    print(report)


if __name__ == '__main__':
    main()
