#!/usr/bin/env python3
"""Analyze the paired RT59-v3 random/normal validation NPZ files.

The frozen gamma=1.66 baseline and RT59 candidate come from the same forward,
so target alignment is exact and no second model evaluation is required.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _stack(npz, key):
    if key not in npz:
        raise KeyError(f'missing required NPZ key: {key}')
    value = npz[key]
    if value.dtype == object:
        value = np.stack([np.asarray(item) for item in value])
    return np.asarray(value)


def _optional(npz, key):
    try:
        return _stack(npz, key)
    except KeyError:
        return None


def _target_matrix(value):
    value = np.asarray(value, dtype=np.float64)
    return value.reshape(value.shape[0], value.shape[1], -1)[..., 0]


def _event_ids(npz, rows):
    values = _optional(npz, 'val_event_id')
    if values is None:
        return np.arange(rows).astype(str)
    return np.asarray(values).reshape(rows, -1)[:, 0].astype(str)


def _mixture_stats(mixture):
    mixture = np.asarray(mixture, dtype=np.float64)
    logits = mixture[..., 0]
    logits = logits - logits.max(axis=-1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=-1, keepdims=True)
    means = mixture[..., 1]
    sigmas = np.maximum(mixture[..., 2], 1e-8)
    mean = (weights * means).sum(axis=-1)
    second = (weights * (sigmas ** 2 + means ** 2)).sum(axis=-1)
    return weights, means, sigmas, mean, np.sqrt(np.maximum(second - mean ** 2, 0.0))


def _mixture_nll(weights, means, sigmas, target):
    log_weights = np.log(np.maximum(weights, np.finfo(np.float64).tiny))
    log_component = (
        log_weights
        - np.log(sigmas)
        - 0.5 * math.log(2.0 * math.pi)
        - 0.5 * ((target[..., None] - means) / sigmas) ** 2
    )
    maximum = log_component.max(axis=-1, keepdims=True)
    return -(maximum[..., 0] + np.log(np.exp(log_component - maximum).sum(axis=-1)))


def _tail_probability(weights, means, sigmas, threshold):
    z = (float(threshold) - means) / sigmas
    erfc = np.vectorize(math.erfc)
    return (weights * 0.5 * erfc(z / math.sqrt(2.0))).sum(axis=-1)


def _point_metrics(target, prediction, mask):
    target = target[mask]
    prediction = prediction[mask]
    error = prediction - target
    if not error.size:
        return {'targets': 0}
    truth_var = np.sum((target - target.mean()) ** 2)
    if target.size >= 2 and truth_var > 0:
        slope, intercept = np.polyfit(target, prediction, 1)
        r2 = 1.0 - np.sum(error ** 2) / truth_var
    else:
        slope = intercept = r2 = float('nan')
    absolute = np.abs(error)
    return {
        'targets': int(error.size),
        'mae': float(absolute.mean()),
        'rmse': float(np.sqrt(np.mean(error ** 2))),
        'bias': float(error.mean()),
        'r2': float(r2),
        'slope': float(slope),
        'intercept': float(intercept),
        'p90_abs_error': float(np.quantile(absolute, 0.90)),
        'p95_abs_error': float(np.quantile(absolute, 0.95)),
        'within_0p1': float(np.mean(absolute <= 0.1)),
        'within_0p2': float(np.mean(absolute <= 0.2)),
    }


def _paired_event_ci(target, base, final, mask, event_ids, metric, seed, draws):
    unique, inverse = np.unique(event_ids, return_inverse=True)
    count = np.zeros(unique.size, dtype=np.float64)
    base_sum = np.zeros(unique.size, dtype=np.float64)
    final_sum = np.zeros(unique.size, dtype=np.float64)
    for index in range(unique.size):
        selected = mask & (inverse[:, None] == index)
        count[index] = selected.sum()
        if metric == 'mae':
            base_sum[index] = np.abs(base[selected] - target[selected]).sum()
            final_sum[index] = np.abs(final[selected] - target[selected]).sum()
        elif metric == 'rmse':
            base_sum[index] = ((base[selected] - target[selected]) ** 2).sum()
            final_sum[index] = ((final[selected] - target[selected]) ** 2).sum()
        else:
            base_sum[index] = base[selected].sum()
            final_sum[index] = final[selected].sum()
    valid_events = count > 0
    count, base_sum, final_sum = count[valid_events], base_sum[valid_events], final_sum[valid_events]
    if count.size == 0:
        return {'events': 0, 'lower': None, 'upper': None, 'delta': None}
    def difference(weights):
        denominator = np.dot(weights, count)
        base_value = np.dot(weights, base_sum) / denominator
        final_value = np.dot(weights, final_sum) / denominator
        if metric == 'rmse':
            base_value = math.sqrt(base_value)
            final_value = math.sqrt(final_value)
        return final_value - base_value
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.integers(0, count.size, size=count.size)
        weights = np.bincount(sampled, minlength=count.size)
        samples[draw] = difference(weights)
    point = difference(np.ones(count.size))
    return {
        'events': int(count.size),
        'draws': int(draws),
        'seed': int(seed),
        'delta': float(point),
        'lower': float(np.quantile(samples, 0.025)),
        'upper': float(np.quantile(samples, 0.975)),
    }


def _field_metrics(target, prediction, mask, station_count):
    range_ratios = []
    range_abs_errors = []
    pairwise_maes = []
    single_ratios = []
    single_range_errors = []
    for row in range(target.shape[0]):
        valid = np.flatnonzero(mask[row])
        if valid.size < 5:
            continue
        truth = target[row, valid]
        pred = prediction[row, valid]
        truth_range = float(np.ptp(truth))
        pred_range = float(np.ptp(pred))
        if truth_range > 0:
            range_ratios.append(pred_range / truth_range)
            if station_count[row] == 1:
                single_ratios.append(pred_range / truth_range)
        range_error = abs(pred_range - truth_range)
        range_abs_errors.append(range_error)
        pairs = np.triu_indices(valid.size, 1)
        pair_error = np.abs(
            (pred[pairs[0]] - pred[pairs[1]])
            - (truth[pairs[0]] - truth[pairs[1]])
        )
        if station_count[row] == 1:
            single_range_errors.append(range_error)
            pairwise_maes.append(float(pair_error.mean()))
    def mean_or_none(values):
        return float(np.mean(values)) if values else None
    return {
        'fields_ge5': len(range_abs_errors),
        'range_ratio_mean': mean_or_none(range_ratios),
        'range_abs_error_mean': mean_or_none(range_abs_errors),
        'one_station_fields_ge5': len(single_range_errors),
        'one_station_range_ratio_mean': mean_or_none(single_ratios),
        'one_station_range_abs_error_mean': mean_or_none(single_range_errors),
        'one_station_pairwise_delta_mae': mean_or_none(pairwise_maes),
    }


def _load_protocol(path):
    npz = np.load(path, allow_pickle=True)
    target = _target_matrix(_stack(npz, 'val_pga_label'))
    final = _target_matrix(_stack(npz, 'val_pga_mu_best'))
    base = _target_matrix(_stack(npz, 'val_rt59_base_mean'))
    valid = _stack(npz, 'val_pga_target_valid').astype(bool).reshape(target.shape)
    observed = _stack(npz, 'val_rt59_route_observed').astype(bool).reshape(target.shape)
    ambiguous = _stack(npz, 'val_rt59_route_ambiguous').astype(bool).reshape(target.shape)
    base_mdn = _stack(npz, 'val_rt59_base_mdn').astype(np.float64)
    base_weights, base_components, base_component_sigma, _, base_sigma = _mixture_stats(base_mdn)
    base_nll = _mixture_nll(base_weights, base_components, base_component_sigma, target)
    final_nll = _optional(npz, 'val_pga_nll_log10_mps2')
    if final_nll is not None:
        final_nll = np.asarray(final_nll, dtype=np.float64).reshape(target.shape)
    final_sigma = _optional(npz, 'val_pga_sigma')
    if final_sigma is not None:
        final_sigma = np.asarray(final_sigma, dtype=np.float64).reshape(target.shape)
    final_probability = _optional(npz, 'val_pga_prob_ge_threshold')
    if final_probability is not None:
        final_probability = np.asarray(final_probability, dtype=np.float64).reshape(target.shape)
    target_type = _optional(npz, 'val_realtime_target_type')
    if target_type is not None:
        target_type = np.asarray(target_type).reshape(target.shape)
    # Gate 5 and the station-count strata use the formal realtime count when
    # available.  ``station_valid_count`` is only a compatibility fallback for
    # old archives which predate the explicit actual-count export.
    station_count = _optional(npz, 'val_actual_station_count')
    if station_count is None:
        station_count = _optional(npz, 'val_station_valid_count')
    if station_count is None:
        station_count = _stack(npz, 'val_station_valid').astype(bool).sum(axis=1)
    station_count = np.asarray(station_count).reshape(target.shape[0], -1)[:, 0]
    requested_time = _optional(npz, 'val_realtime_requested_elapsed_time')
    if requested_time is not None:
        requested_time = np.asarray(requested_time, dtype=np.float64).reshape(target.shape[0], -1)[:, 0]
    event_ids = _event_ids(npz, target.shape[0])
    if not (
        target.shape == final.shape == base.shape == valid.shape
        == observed.shape == ambiguous.shape
    ):
        raise ValueError('public target/base/final/route shapes are not aligned.')
    if target_type is None:
        raise KeyError(
            'missing required NPZ key: val_realtime_target_type; formal input/'
            'non-input populations must not be inferred from the RT59 route.'
        )
    target_type = np.asarray(target_type).reshape(target.shape)
    if not np.isfinite(target[valid]).all():
        raise ValueError('valid PGA labels contain non-finite values.')
    if not np.isfinite(final[valid]).all() or not np.isfinite(base[valid]).all():
        raise ValueError('valid base/final predictions contain non-finite values.')
    _, _, _, base_from_mdn, _ = _mixture_stats(base_mdn)
    if not np.allclose(base_from_mdn[valid], base[valid], rtol=1e-6, atol=1e-6):
        raise ValueError('exported RT59 base mean disagrees with its MDN mixture mean.')
    return {
        'path': str(Path(path).resolve()),
        'sha256': _sha256_file(path),
        'target': target,
        'final': final,
        'base': base,
        'valid': valid,
        'observed': observed,
        'ambiguous': ambiguous,
        'base_sigma': base_sigma,
        'final_sigma': final_sigma,
        'base_nll': base_nll,
        'final_nll': final_nll,
        'base_probability': _tail_probability(base_weights, base_components, base_component_sigma, -1.2),
        'final_probability': final_probability,
        'target_type': target_type,
        'station_count': station_count,
        'requested_time': requested_time,
        'event_ids': event_ids,
    }


def _mask_summary(data, mask):
    """Compact paired point/field summary for descriptive protocol strata."""
    mask = np.asarray(mask, dtype=bool) & data['valid']
    row_mask = mask.any(axis=1)
    base = _point_metrics(data['target'], data['base'], mask)
    final = _point_metrics(data['target'], data['final'], mask)
    deltas = {}
    for metric in (
        'mae', 'rmse', 'bias', 'r2', 'slope', 'intercept',
        'p90_abs_error', 'p95_abs_error', 'within_0p1', 'within_0p2',
    ):
        if metric in base and metric in final:
            deltas[metric] = final[metric] - base[metric]
    return {
        'events': int(np.unique(data['event_ids'][row_mask]).size),
        'realtime_rows': int(row_mask.sum()),
        'targets': int(mask.sum()),
        'base': base,
        'final': final,
        'delta_final_minus_base': deltas,
        'fields': {
            'base': _field_metrics(
                data['target'], data['base'], mask, data['station_count']
            ),
            'final': _field_metrics(
                data['target'], data['final'], mask, data['station_count']
            ),
        },
    }


def _protocol_strata(data):
    """Export real counts/metrics without hard-coding expected populations."""
    result = {}
    type_names = {0: 'input', 1: 'triggered_noninput', 2: 'untriggered'}
    result['target_type'] = {
        type_names.get(int(value), f'type_{int(value)}'): _mask_summary(
            data, data['target_type'] == value
        )
        for value in sorted(np.unique(data['target_type'][data['valid']]).tolist())
    }
    result['observable_route'] = {
        'observed_unique': _mask_summary(data, data['observed']),
        'transport': _mask_summary(data, ~data['observed']),
        'ambiguous_transport': _mask_summary(data, data['ambiguous']),
    }
    result['actual_station_count'] = {
        str(int(value)): _mask_summary(
            data, np.broadcast_to(data['station_count'][:, None] == value, data['valid'].shape)
        )
        for value in sorted(np.unique(data['station_count']).tolist())
    }
    if data['requested_time'] is not None:
        result['requested_elapsed_time_seconds'] = {
            f'{float(value):g}': _mask_summary(
                data,
                np.broadcast_to(
                    np.isclose(data['requested_time'][:, None], value),
                    data['valid'].shape,
                ),
            )
            for value in sorted(np.unique(data['requested_time']).tolist())
        }
    return result


def _population(data, mask, seed, draws):
    base_metrics = _point_metrics(data['target'], data['base'], mask)
    final_metrics = _point_metrics(data['target'], data['final'], mask)
    ci = {
        metric: _paired_event_ci(
            data['target'], data['base'], data['final'], mask,
            data['event_ids'], metric, seed, draws,
        )
        for metric in ('mae', 'rmse')
    }
    result = {'base': base_metrics, 'final': final_metrics, 'paired_ci': ci}
    for name in ('base_nll', 'final_nll'):
        value = data[name]
        result[name] = float(np.mean(value[mask])) if value is not None and mask.any() else None
    truth_binary = data['target'] >= -1.2
    for prefix in ('base', 'final'):
        probability = data[f'{prefix}_probability']
        sigma = data[f'{prefix}_sigma']
        result[f'{prefix}_brier'] = (
            float(np.mean((probability[mask] - truth_binary[mask]) ** 2))
            if probability is not None and mask.any() else None
        )
        if sigma is not None and mask.any():
            absolute = np.abs(data[prefix][mask] - data['target'][mask])
            result[f'{prefix}_coverage1'] = float(np.mean(absolute <= sigma[mask]))
            result[f'{prefix}_coverage2'] = float(np.mean(absolute <= 2 * sigma[mask]))
        else:
            result[f'{prefix}_coverage1'] = None
            result[f'{prefix}_coverage2'] = None
    if data['base_probability'] is not None and data['final_probability'] is not None:
        base_brier = (data['base_probability'] - truth_binary) ** 2
        final_brier = (data['final_probability'] - truth_binary) ** 2
        result['brier_paired_ci'] = _paired_event_ci(
            np.zeros_like(base_brier), base_brier, final_brier, mask,
            data['event_ids'], 'mean', seed, draws,
        )
    return result


def _gate(name, value, threshold, relation, required=True):
    if value is None or not np.isfinite(value):
        return {'gate': name, 'value': value, 'rule': f'{relation} {threshold}', 'pass': None, 'required': required}
    passed = value <= threshold if relation == '<=' else value >= threshold if relation == '>=' else value < threshold
    return {'gate': name, 'value': float(value), 'rule': f'{relation} {threshold}', 'pass': bool(passed), 'required': required}


def _write_truth_prediction_figure(output_path, populations):
    """Use identical axes, bins, and density scale without dropping outliers."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        return False
    all_values = []
    selected = []
    for title, data, mask in populations:
        truth = data['target'][mask]
        prediction = data['final'][mask]
        selected.append((title, truth, prediction))
        all_values.extend([truth, prediction])
    finite_values = np.concatenate(all_values)
    finite_values = finite_values[np.isfinite(finite_values)]
    if not finite_values.size:
        return False
    lower, upper = float(finite_values.min()), float(finite_values.max())
    if lower == upper:
        lower -= 0.5
        upper += 0.5
    edges = np.linspace(lower, upper, 121)
    histograms = [np.histogram2d(truth, prediction, bins=(edges, edges))[0] for _, truth, prediction in selected]
    maximum = max(float(histogram.max()) for histogram in histograms)
    norm = LogNorm(vmin=1.0, vmax=max(maximum, 1.0001))
    figure, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    image = None
    for axis, (title, _truth, _prediction), histogram in zip(axes.flat, selected, histograms):
        masked = np.ma.masked_less_equal(histogram.T, 0)
        image = axis.pcolormesh(edges, edges, masked, cmap='viridis', norm=norm, shading='auto')
        axis.plot([lower, upper], [lower, upper], color='white', linewidth=1.0, linestyle='--')
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_aspect('equal', adjustable='box')
        axis.set_title(title)
        axis.set_xlabel('Truth log10(m/s²)')
        axis.set_ylabel('Prediction log10(m/s²)')
    if image is not None:
        figure.colorbar(image, ax=axes, label='Targets per bin')
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_npz', required=True)
    parser.add_argument('--normal_npz', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--bootstrap_draws', type=int, default=5000)
    parser.add_argument('--bootstrap_seed', type=int, default=20260915)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random_data = _load_protocol(args.random_npz)
    normal_data = _load_protocol(args.normal_npz)
    random_mask = random_data['valid']
    normal_all = normal_data['valid']
    normal_input = normal_all & (normal_data['target_type'] == 0)
    normal_noninput = normal_all & (
        (normal_data['target_type'] == 1) | (normal_data['target_type'] == 2)
    )
    if (random_mask & (random_data['target_type'] == 0)).any():
        raise ValueError(
            'random-geometry validation contains formal input targets; the fixed '
            'RT59 protocol requires non-input random queries.'
        )
    if not np.array_equal(normal_input | normal_noninput, normal_all):
        unexpected = sorted(np.unique(normal_data['target_type'][
            normal_all & ~(normal_input | normal_noninput)
        ]).tolist())
        raise ValueError(f'unexpected valid normal target_type values: {unexpected}')
    groups = {
        'random': _population(random_data, random_mask, args.bootstrap_seed, args.bootstrap_draws),
        'normal_all': _population(normal_data, normal_all, args.bootstrap_seed, args.bootstrap_draws),
        'normal_input': _population(normal_data, normal_input, args.bootstrap_seed, args.bootstrap_draws),
        'normal_noninput': _population(normal_data, normal_noninput, args.bootstrap_seed, args.bootstrap_draws),
    }
    random_fields_final = _field_metrics(
        random_data['target'], random_data['final'], random_mask, random_data['station_count']
    )
    random_fields_base = _field_metrics(
        random_data['target'], random_data['base'], random_mask, random_data['station_count']
    )
    requested1_mask = None
    if random_data['requested_time'] is not None:
        requested1_mask = random_mask & np.isclose(random_data['requested_time'][:, None], 1.0)
    requested1 = _population(
        random_data,
        requested1_mask,
        args.bootstrap_seed,
        args.bootstrap_draws,
    ) if requested1_mask is not None else None

    random_group = groups['random']
    gates = [
        _gate('random_mae', random_group['final']['mae'], 0.242, '<='),
        _gate('random_slope', random_group['final']['slope'], 0.45, '>='),
        _gate('random_r2', random_group['final']['r2'], 0.39, '>='),
        _gate('random_range_ratio_ge5', random_fields_final['range_ratio_mean'], 0.55, '>='),
        _gate('random_one_station_range_ratio_ge5', random_fields_final['one_station_range_ratio_mean'], 0.25, '>='),
        _gate('random_one_station_pairwise_delta_mae', random_fields_final['one_station_pairwise_delta_mae'], 0.330, '<='),
        _gate('random_requested1_mae', requested1['final']['mae'] if requested1 else None, 0.253, '<='),
        _gate('random_nll', random_group['final_nll'], 0.22, '<='),
        _gate('random_delta_mae_ci_upper', random_group['paired_ci']['mae']['upper'], 0.0, '<'),
        _gate('random_delta_brier_ci_upper', (random_group.get('brier_paired_ci') or {}).get('upper'), 0.0, '<='),
    ]
    for coverage in ('coverage1', 'coverage2'):
        base_value = random_group[f'base_{coverage}']
        final_value = random_group[f'final_{coverage}']
        change = abs(final_value - base_value) if base_value is not None and final_value is not None else None
        gates.append(_gate(f'random_{coverage}_absolute_change', change, 0.01, '<='))
    for population in ('normal_all', 'normal_noninput'):
        for metric in ('mae', 'rmse'):
            gates.append(_gate(
                f'{population}_delta_{metric}_ci_upper',
                groups[population]['paired_ci'][metric]['upper'], 0.0, '<',
            ))
    for metric in ('mae', 'rmse'):
        final_value = groups['normal_input']['final'][metric]
        base_value = groups['normal_input']['base'][metric]
        gates.append(_gate(f'normal_input_{metric}_delta', final_value - base_value, 0.0, '<='))
    gates.extend([
        _gate('normal_all_mae_historical_redline', groups['normal_all']['final']['mae'], 0.136, '<='),
        _gate('normal_noninput_mae_historical_redline', groups['normal_noninput']['final']['mae'], 0.218, '<='),
    ])
    gates.extend([
        _gate('normal_nll_delta', (
            groups['normal_all']['final_nll'] - groups['normal_all']['base_nll']
            if groups['normal_all']['final_nll'] is not None else None
        ), 0.0, '<='),
        _gate('normal_brier_delta', (
            groups['normal_all']['final_brier'] - groups['normal_all']['base_brier']
            if groups['normal_all']['final_brier'] is not None else None
        ), 0.0, '<='),
        _gate('one_station_range_abs_error_delta', (
            random_fields_final['one_station_range_abs_error_mean']
            - random_fields_base['one_station_range_abs_error_mean']
            if random_fields_final['one_station_range_abs_error_mean'] is not None
            and random_fields_base['one_station_range_abs_error_mean'] is not None else None
        ), 0.0, '<='),
    ])
    for coverage in ('coverage1', 'coverage2'):
        base_value = groups['normal_all'][f'base_{coverage}']
        final_value = groups['normal_all'][f'final_{coverage}']
        change = abs(final_value - base_value) if base_value is not None and final_value is not None else None
        gates.append(_gate(f'normal_{coverage}_absolute_change', change, 0.01, '<='))
    for population in ('normal_all', 'normal_noninput'):
        bias_change = (
            abs(groups[population]['final']['bias'])
            - abs(groups[population]['base']['bias'])
        )
        slope_error_change = (
            abs(groups[population]['final']['slope'] - 1.0)
            - abs(groups[population]['base']['slope'] - 1.0)
        )
        gates.extend([
            _gate(f'{population}_absolute_bias_change', bias_change, 0.0, '<='),
            _gate(
                f'{population}_absolute_slope_error_change',
                slope_error_change,
                0.0,
                '<=',
            ),
        ])
    required = [item['pass'] for item in gates if item['required']]
    decision = 'GO_FOR_DEVELOPMENT' if required and all(value is True for value in required) else 'NO_GO'
    if any(value is None for value in required):
        decision = 'INCOMPLETE_EVIDENCE'
    summary = {
        'schema_version': 1,
        'decision': decision,
        'sources': {
            'random': {'path': random_data['path'], 'sha256': random_data['sha256']},
            'normal': {'path': normal_data['path'], 'sha256': normal_data['sha256']},
        },
        'bootstrap': {'draws': args.bootstrap_draws, 'seed': args.bootstrap_seed, 'unit': 'event_id'},
        'counts': {
            'random_rows': int(random_data['target'].shape[0]),
            'random_targets': int(random_mask.sum()),
            'normal_rows': int(normal_data['target'].shape[0]),
            'normal_targets': int(normal_all.sum()),
            'normal_input_targets': int(normal_input.sum()),
            'normal_noninput_targets': int(normal_noninput.sum()),
            'normal_observed_route_targets': int((normal_all & normal_data['observed']).sum()),
            'normal_ambiguous_route_targets': int((normal_all & normal_data['ambiguous']).sum()),
        },
        'groups': groups,
        'random_fields': {'base': random_fields_base, 'final': random_fields_final},
        'random_requested1': requested1,
        'strata': {
            'random': _protocol_strata(random_data),
            'normal': _protocol_strata(normal_data),
        },
        'gates': gates,
        'diagnostic_only': {
            population: {
                'absolute_bias_change': abs(groups[population]['final']['bias']) - abs(groups[population]['base']['bias']),
                'absolute_slope_error_change': abs(groups[population]['final']['slope'] - 1.0) - abs(groups[population]['base']['slope'] - 1.0),
            }
            for population in ('normal_all', 'normal_noninput')
        },
    }
    figure_written = _write_truth_prediction_figure(
        output_dir / 'truth_prediction_density.png',
        [
            ('Random', random_data, random_mask),
            ('Normal all', normal_data, normal_all),
            ('Normal input', normal_data, normal_input),
            ('Normal non-input', normal_data, normal_noninput),
        ],
    )
    summary['truth_prediction_figure_written'] = figure_written
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    with (output_dir / 'gates.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['gate', 'value', 'rule', 'pass', 'required'])
        writer.writeheader()
        writer.writerows(gates)
    with (output_dir / 'group_metrics.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['population', 'model', 'targets', 'mae', 'rmse', 'bias', 'r2', 'slope', 'intercept'])
        for population, values in groups.items():
            for model_name in ('base', 'final'):
                metrics = values[model_name]
                writer.writerow([population, model_name] + [metrics.get(name) for name in ('targets', 'mae', 'rmse', 'bias', 'r2', 'slope', 'intercept')])
    with (output_dir / 'paired_ci.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['population', 'metric', 'events', 'delta_final_minus_base', 'ci_lower', 'ci_upper'])
        for population, values in groups.items():
            for metric, ci in values['paired_ci'].items():
                writer.writerow([population, metric, ci['events'], ci['delta'], ci['lower'], ci['upper']])
    with (output_dir / 'strata_counts.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            'protocol', 'stratum', 'value', 'events', 'realtime_rows', 'targets',
            'base_mae', 'final_mae', 'delta_mae', 'base_rmse', 'final_rmse',
            'delta_rmse', 'fields_ge5', 'one_station_fields_ge5',
        ])
        for protocol, protocol_values in summary['strata'].items():
            for stratum, stratum_values in protocol_values.items():
                for value, metrics in stratum_values.items():
                    writer.writerow([
                        protocol,
                        stratum,
                        value,
                        metrics['events'],
                        metrics['realtime_rows'],
                        metrics['targets'],
                        metrics['base'].get('mae'),
                        metrics['final'].get('mae'),
                        metrics['delta_final_minus_base'].get('mae'),
                        metrics['base'].get('rmse'),
                        metrics['final'].get('rmse'),
                        metrics['delta_final_minus_base'].get('rmse'),
                        metrics['fields']['final']['fields_ge5'],
                        metrics['fields']['final']['one_station_fields_ge5'],
                    ])
    readme = [
        '# RT59-v3 paired validation analysis', '',
        f'- Decision: **{decision}**',
        f'- Random targets: {int(random_mask.sum()):,}',
        f'- Normal targets: {int(normal_all.sum()):,} '
        f'(input {int(normal_input.sum()):,}, non-input {int(normal_noninput.sum()):,})',
        f'- Bootstrap: event_id paired, {args.bootstrap_draws} draws, seed {args.bootstrap_seed}', '',
        'The decision is conjunctive. `INCOMPLETE_EVIDENCE` means at least one required '
        'quantity was absent; it must not be treated as a pass.', '',
        '| Gate | Value | Rule | Pass |', '|---|---:|---:|:---:|',
    ]
    for item in gates:
        readme.append(f"| {item['gate']} | {item['value']} | {item['rule']} | {item['pass']} |")
    (output_dir / 'README.md').write_text('\n'.join(readme) + '\n')
    print(json.dumps({'decision': decision, 'output_dir': str(output_dir.resolve())}, indent=2))


if __name__ == '__main__':
    main()
