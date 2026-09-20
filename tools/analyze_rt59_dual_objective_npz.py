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
import subprocess
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
    # Event-paired bootstrap is invalid without the exported event identity.
    # Deliberately fail closed instead of silently using row indices.
    values = _stack(npz, 'val_event_id')
    values = np.asarray(values).reshape(rows, -1)[:, 0].astype(str)
    if np.any(np.char.str_len(values) == 0):
        raise ValueError('val_event_id contains empty values.')
    return values


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
    # Chunked vectorization keeps the exact event-cluster estimator while
    # avoiding thousands of Python-level bincount calls.
    chunk_size = 256
    for start in range(0, draws, chunk_size):
        stop = min(start + chunk_size, draws)
        sampled = rng.integers(0, count.size, size=(stop - start, count.size))
        denominator = count[sampled].sum(axis=1)
        base_value = base_sum[sampled].sum(axis=1) / denominator
        final_value = final_sum[sampled].sum(axis=1) / denominator
        if metric == 'rmse':
            base_value = np.sqrt(base_value)
            final_value = np.sqrt(final_value)
        samples[start:stop] = final_value - base_value
    point = difference(np.ones(count.size))
    return {
        'events': int(count.size),
        'draws': int(draws),
        'seed': int(seed),
        'delta': float(point),
        'lower': float(np.quantile(samples, 0.025)),
        'upper': float(np.quantile(samples, 0.975)),
    }


def _canonical_field_range(values):
    """Historical query-diagnostic field range: NumPy P95 minus P05."""
    values = np.asarray(values, dtype=np.float64)
    return float(np.percentile(values, 95) - np.percentile(values, 5))


def _field_metrics(target, prediction, mask, station_count):
    range_ratios = []
    range_abs_errors = []
    ptp_range_ratios = []
    ptp_range_abs_errors = []
    pairwise_maes = []
    pairwise_rmses = []
    single_ratios = []
    single_range_errors = []
    single_ptp_ratios = []
    single_ptp_range_errors = []
    for row in range(target.shape[0]):
        valid = np.flatnonzero(mask[row])
        if valid.size < 5:
            continue
        truth = target[row, valid]
        pred = prediction[row, valid]
        truth_range = _canonical_field_range(truth)
        pred_range = _canonical_field_range(pred)
        truth_ptp = float(np.ptp(truth))
        pred_ptp = float(np.ptp(pred))
        if truth_range > 0:
            range_ratios.append(pred_range / truth_range)
            if station_count[row] == 1:
                single_ratios.append(pred_range / truth_range)
        if truth_ptp > 0:
            ptp_range_ratios.append(pred_ptp / truth_ptp)
            if station_count[row] == 1:
                single_ptp_ratios.append(pred_ptp / truth_ptp)
        range_error = abs(pred_range - truth_range)
        ptp_range_error = abs(pred_ptp - truth_ptp)
        range_abs_errors.append(range_error)
        ptp_range_abs_errors.append(ptp_range_error)
        pairs = np.triu_indices(valid.size, 1)
        pair_error = (
            (pred[pairs[0]] - pred[pairs[1]])
            - (truth[pairs[0]] - truth[pairs[1]])
        )
        if station_count[row] == 1:
            single_range_errors.append(range_error)
            single_ptp_range_errors.append(ptp_range_error)
            pairwise_maes.append(float(np.mean(np.abs(pair_error))))
            pairwise_rmses.append(float(np.sqrt(np.mean(pair_error ** 2))))
    def mean_or_none(values):
        return float(np.mean(values)) if values else None
    return {
        'fields_ge5': len(range_abs_errors),
        'range_ratio_mean': mean_or_none(range_ratios),
        'range_abs_error_mean': mean_or_none(range_abs_errors),
        'ptp_range_ratio_mean_diagnostic': mean_or_none(ptp_range_ratios),
        'ptp_range_abs_error_mean_diagnostic': mean_or_none(ptp_range_abs_errors),
        'one_station_fields_ge5': len(single_range_errors),
        'one_station_range_ratio_mean': mean_or_none(single_ratios),
        'one_station_range_abs_error_mean': mean_or_none(single_range_errors),
        'one_station_pairwise_delta_mae': mean_or_none(pairwise_maes),
        'one_station_pairwise_delta_rmse': mean_or_none(pairwise_rmses),
        'one_station_ptp_range_ratio_mean_diagnostic': mean_or_none(single_ptp_ratios),
        'one_station_ptp_range_abs_error_mean_diagnostic': mean_or_none(single_ptp_range_errors),
    }


def _field_records(data, mask, protocol, population):
    """Return one lightweight row per field, retaining n>=2 fields."""
    records = []
    mask = np.asarray(mask, dtype=bool) & data['valid']
    for row in range(data['target'].shape[0]):
        valid = np.flatnonzero(mask[row])
        if valid.size < 2:
            continue
        truth = data['target'][row, valid]
        base = data['base'][row, valid]
        final = data['final'][row, valid]
        base_error = base - truth
        final_error = final - truth
        pairs = np.triu_indices(valid.size, 1)
        truth_delta = truth[pairs[0]] - truth[pairs[1]]
        base_delta_error = base[pairs[0]] - base[pairs[1]] - truth_delta
        final_delta_error = final[pairs[0]] - final[pairs[1]] - truth_delta
        truth_range = _canonical_field_range(truth)
        base_range = _canonical_field_range(base)
        final_range = _canonical_field_range(final)
        records.append({
            'protocol': protocol,
            'population': population,
            'row_index': row,
            'event_id': data['event_ids'][row],
            'requested_elapsed_time_seconds': (
                None if data['requested_time'] is None
                else float(data['requested_time'][row])
            ),
            'actual_station_count': int(data['station_count'][row]),
            'n_targets': int(valid.size),
            'eligible_ge5': bool(valid.size >= 5),
            'truth_p95_p05_range': truth_range,
            'base_p95_p05_range': base_range,
            'final_p95_p05_range': final_range,
            'base_range_ratio': base_range / truth_range if truth_range > 0 else None,
            'final_range_ratio': final_range / truth_range if truth_range > 0 else None,
            'base_range_abs_error': abs(base_range - truth_range),
            'final_range_abs_error': abs(final_range - truth_range),
            'base_pairwise_delta_mae': float(np.mean(np.abs(base_delta_error))),
            'final_pairwise_delta_mae': float(np.mean(np.abs(final_delta_error))),
            'base_pairwise_delta_rmse': float(np.sqrt(np.mean(base_delta_error ** 2))),
            'final_pairwise_delta_rmse': float(np.sqrt(np.mean(final_delta_error ** 2))),
            'base_field_mean_error': float(np.mean(base_error)),
            'final_field_mean_error': float(np.mean(final_error)),
            'base_centered_error_mae': float(np.mean(np.abs(base_error - np.mean(base_error)))),
            'final_centered_error_mae': float(np.mean(np.abs(final_error - np.mean(final_error)))),
            'base_centered_error_rmse': float(np.sqrt(np.mean((base_error - np.mean(base_error)) ** 2))),
            'final_centered_error_rmse': float(np.sqrt(np.mean((final_error - np.mean(final_error)) ** 2))),
            'truth_ptp_range_diagnostic': float(np.ptp(truth)),
            'base_ptp_range_diagnostic': float(np.ptp(base)),
            'final_ptp_range_diagnostic': float(np.ptp(final)),
        })
    return records


def _rolled_delta_summary(data, mask):
    mask = np.asarray(mask, dtype=bool) & data['valid']
    correct = data['base'] + data['applied_delta']
    rolled = data['base'] + data['fixed_context_rolled_delta']
    correct_metrics = _point_metrics(data['target'], correct, mask)
    rolled_metrics = _point_metrics(data['target'], rolled, mask)
    return {
        'correct_route': correct_metrics,
        'fixed_context_rolled': rolled_metrics,
        'delta_rolled_minus_correct': {
            metric: rolled_metrics[metric] - correct_metrics[metric]
            for metric in ('mae', 'rmse')
        },
    }


def _event_manifest(data):
    unique = np.unique(data['event_ids'])
    year_counts = {}
    for event_id in unique:
        year = str(event_id)[:4]
        if len(year) != 4 or not year.isdigit():
            raise ValueError(f'event_id does not begin with a four-digit year: {event_id!r}')
        year_counts[year] = year_counts.get(year, 0) + 1
    canonical = '\n'.join(sorted(unique.tolist())) + '\n'
    return {
        'source': 'val_event_id in formal validation NPZ',
        'unique_events': int(unique.size),
        'first_event_id': str(unique[0]),
        'last_event_id': str(unique[-1]),
        'year_counts': year_counts,
        'event_id_sha256': hashlib.sha256(canonical.encode('utf-8')).hexdigest(),
    }


def _sanitize_config_value(value):
    if isinstance(value, dict):
        return {str(key): _sanitize_config_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_config_value(item) for item in value]
    if isinstance(value, str) and value.startswith('/'):
        return f'<ABSOLUTE_PATH_REDACTED>/{Path(value).name}'
    return value


def _write_sanitized_config(source, destination):
    source = Path(source)
    payload = json.loads(source.read_text())
    destination.write_text(
        json.dumps(_sanitize_config_value(payload), indent=2, sort_keys=True) + '\n'
    )
    return {
        'name': source.name,
        'original_sha256': _sha256_file(source),
        'sanitized_file': destination.name,
        'sanitized_sha256': _sha256_file(destination),
    }


def _git_value(args):
    try:
        return subprocess.check_output(
            ['git', *args], cwd=Path(__file__).resolve().parents[1], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_metrics_identity(path):
    path = Path(path)
    payload = json.loads(path.read_text())
    protocol = payload.get('metric_protocol') or {}
    if protocol.get('pga_coordinate') != 'log10(m/s^2)':
        raise ValueError(
            'formal metrics do not unambiguously declare PGA coordinate '
            'log10(m/s^2).'
        )
    if protocol.get('point_estimate') != 'predictive_mixture_mean':
        raise ValueError('formal metrics point estimate is not predictive_mixture_mean.')
    if payload.get('splits') != ['val']:
        raise ValueError(f'formal metrics must contain only split val: {payload.get("splits")!r}')
    metadata = payload.get('checkpoint_metadata') or {}
    if int(metadata.get('epoch', -1)) != 8:
        raise ValueError('formal metrics checkpoint metadata is not epoch 8.')
    return {
        'file': path.name,
        'sha256': _sha256_file(path),
        'pga_coordinate': protocol['pga_coordinate'],
        'point_estimate': protocol['point_estimate'],
        'splits': payload['splits'],
        'checkpoint_metadata': _sanitize_config_value(metadata),
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
    # All station-count representations must agree.  Silently preferring one
    # of them would make the single/multi-field gates non-auditable.
    station_valid = _stack(npz, 'val_station_valid').astype(bool)
    station_valid = station_valid.reshape(target.shape[0], -1)
    station_mask_count = station_valid.sum(axis=1).astype(np.int64)
    exported_counts = {}
    for key in ('val_actual_station_count', 'val_station_valid_count'):
        value = _optional(npz, key)
        if value is not None:
            value = np.asarray(value).reshape(target.shape[0], -1)[:, 0]
            if not np.isfinite(value.astype(np.float64)).all():
                raise ValueError(f'{key} contains non-finite values.')
            if not np.array_equal(value.astype(np.int64), value):
                raise ValueError(f'{key} contains non-integer station counts.')
            exported_counts[key] = value.astype(np.int64)
    if not exported_counts:
        raise KeyError(
            'missing both val_actual_station_count and val_station_valid_count; '
            'station strata must not be inferred without an exported count.'
        )
    for key, value in exported_counts.items():
        if not np.array_equal(value, station_mask_count):
            mismatch = np.flatnonzero(value != station_mask_count)[:10].tolist()
            raise ValueError(
                f'{key} disagrees with val_station_valid sum at rows {mismatch}.'
            )
    count_values = list(exported_counts.values())
    for other in count_values[1:]:
        if not np.array_equal(count_values[0], other):
            raise ValueError('exported station-count fields disagree.')
    station_count = count_values[0]
    requested_time = _optional(npz, 'val_realtime_requested_elapsed_time')
    if requested_time is not None:
        requested_time = np.asarray(requested_time, dtype=np.float64).reshape(target.shape[0], -1)[:, 0]
    event_ids = _event_ids(npz, target.shape[0])
    applied_delta = _target_matrix(_stack(npz, 'val_rt59_applied_delta'))
    fixed_context_rolled_delta = _target_matrix(
        _stack(npz, 'val_rt59_fixed_context_rolled_delta')
    )
    if not (
        target.shape == final.shape == base.shape == valid.shape
        == observed.shape == ambiguous.shape == applied_delta.shape
        == fixed_context_rolled_delta.shape
    ):
        raise ValueError('public target/base/final/route shapes are not aligned.')
    if target_type is None:
        raise KeyError(
            'missing required NPZ key: val_realtime_target_type; formal input/'
            'non-input populations must not be inferred from the RT59 route.'
        )
    target_type = np.asarray(target_type).reshape(target.shape)
    formal_input = target_type == 0
    if not np.array_equal(formal_input, observed):
        mismatch = np.argwhere(formal_input != observed)[:10].tolist()
        raise ValueError(
            'formal input membership and observable-route membership disagree '
            f'elementwise at indices {mismatch}.'
        )
    if not np.isfinite(target[valid]).all():
        raise ValueError('valid PGA labels contain non-finite values.')
    if not np.isfinite(final[valid]).all() or not np.isfinite(base[valid]).all():
        raise ValueError('valid base/final predictions contain non-finite values.')
    if not np.isfinite(applied_delta[valid]).all():
        raise ValueError('valid applied deltas contain non-finite values.')
    if not np.isfinite(fixed_context_rolled_delta[valid]).all():
        raise ValueError('valid fixed-context rolled deltas contain non-finite values.')
    if not np.allclose(base[valid] + applied_delta[valid], final[valid], rtol=1e-6, atol=1e-6):
        maximum = float(np.max(np.abs(base[valid] + applied_delta[valid] - final[valid])))
        raise ValueError(
            'base + val_rt59_applied_delta disagrees with final mean; '
            f'max absolute error={maximum:.9g}.'
        )
    _, _, _, base_from_mdn, _ = _mixture_stats(base_mdn)
    if not np.allclose(base_from_mdn[valid], base[valid], rtol=1e-6, atol=1e-6):
        raise ValueError('exported RT59 base mean disagrees with its MDN mixture mean.')
    return {
        'path': f'<LOCAL_ARTIFACT>/{Path(path).name}',
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
        'station_valid': station_valid,
        'requested_time': requested_time,
        'event_ids': event_ids,
        'applied_delta': applied_delta,
        'fixed_context_rolled_delta': fixed_context_rolled_delta,
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
    parser.add_argument('--random_metrics', required=True)
    parser.add_argument('--normal_metrics', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--bootstrap_draws', type=int, default=5000)
    parser.add_argument('--bootstrap_seed', type=int, default=20260915)
    parser.add_argument('--training_config')
    parser.add_argument('--random_config')
    parser.add_argument('--normal_config')
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random_data = _load_protocol(args.random_npz)
    normal_data = _load_protocol(args.normal_npz)
    random_metrics_identity = _load_metrics_identity(args.random_metrics)
    normal_metrics_identity = _load_metrics_identity(args.normal_metrics)
    if (
        random_metrics_identity['checkpoint_metadata']
        != normal_metrics_identity['checkpoint_metadata']
    ):
        raise ValueError('random/normal formal metrics checkpoint metadata disagree.')
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

    rolled_controls = {}
    for protocol_name, data, protocol_mask in (
        ('random', random_data, random_mask),
        ('normal', normal_data, normal_all),
    ):
        rolled_controls[protocol_name] = {
            'all': _rolled_delta_summary(data, protocol_mask),
            'single_station': _rolled_delta_summary(
                data,
                protocol_mask & np.broadcast_to(
                    data['station_count'][:, None] == 1, data['valid'].shape
                ),
            ),
            'multi_station': _rolled_delta_summary(
                data,
                protocol_mask & np.broadcast_to(
                    data['station_count'][:, None] > 1, data['valid'].shape
                ),
            ),
        }

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
            _gate(f'{population}_absolute_bias_change', bias_change, 0.0, '<=', required=False),
            _gate(
                f'{population}_absolute_slope_error_change',
                slope_error_change,
                0.0,
                '<=',
                required=False,
            ),
        ])
    required = [item['pass'] for item in gates if item['required']]
    if len(gates) != 29 or len(required) != 25:
        raise AssertionError(
            f'protocol gate inventory changed unexpectedly: {len(gates)} total, '
            f'{len(required)} required (expected 29/25).'
        )
    decision = 'GO_FOR_DEVELOPMENT' if required and all(value is True for value in required) else 'NO_GO'
    if any(value is None for value in required):
        decision = 'INCOMPLETE_EVIDENCE'
    summary = {
        'schema_version': 2,
        'decision': decision,
        'metric_correction': {
            'canonical_field_range': 'per-field NumPy percentile(95) - percentile(5), default linear method',
            'minimum_valid_targets_for_range_gate': 5,
            'aggregation': 'equal weight per eligible field',
            'zero_truth_range': 'range ratio is undefined when true P95-P05 range <= 0',
            'ptp_values': 'retained only under explicitly diagnostic ptp_* keys',
            'gate_inventory': {'required': 25, 'diagnostic_only': 4, 'total': 29},
        },
        'sources': {
            'random': {
                'path': random_data['path'], 'sha256': random_data['sha256'],
                'formal_metrics': random_metrics_identity,
            },
            'normal': {
                'path': normal_data['path'], 'sha256': normal_data['sha256'],
                'formal_metrics': normal_metrics_identity,
            },
        },
        'provenance': {
            'analysis_git_commit': _git_value(['rev-parse', 'HEAD']),
            'analysis_git_branch': _git_value(['branch', '--show-current']),
            'analysis_worktree_dirty': bool(_git_value(['status', '--porcelain'])),
            'rt59_implementation_commit': '54fd63d623ac5837e16a5d175d34041b7488661e',
            'rt59_counter_fix_commit': '7e00824b3aab7a15286dfd9bf9a264b03080c1cd',
            'rt59_submitted_source_manifest_sha256': '3e1164bc4fb0441fb33e5d8cd03aa1708416708d930b01e71c4a82fd3779715e',
            'rt59_evidence_commit': 'b709825b78e7579dbf05c8d4eacda5b29b9e58ae',
            'rt59_review_commit': 'd43c7d63528b3ade71c3506eeb2cb71f90fb3ffd',
        },
        'split_identity': {
            'declared_split': 'val',
            'declared_dataset_period': 'Japan full 2000-2024',
            'random': _event_manifest(random_data),
            'normal': _event_manifest(normal_data),
            'note': (
                'The evaluated-event manifest is reconstructed directly from each '
                'formal NPZ val_event_id export; no external split manifest was bundled.'
            ),
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
        'fixed_context_rolled_control': rolled_controls,
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

    config_sources = {}
    for label, source in (
        ('training', args.training_config),
        ('random_validation', args.random_config),
        ('normal_validation', args.normal_config),
    ):
        if source:
            config_sources[label] = _write_sanitized_config(
                source, output_dir / f'{label}_config.sanitized.json'
            )
    summary['sanitized_config_sources'] = config_sources
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    with (output_dir / 'gates.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['gate', 'value', 'rule', 'pass', 'required'],
            lineterminator='\n',
        )
        writer.writeheader()
        writer.writerows(gates)
    with (output_dir / 'group_metrics.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['population', 'model', 'targets', 'mae', 'rmse', 'bias', 'r2', 'slope', 'intercept'])
        for population, values in groups.items():
            for model_name in ('base', 'final'):
                metrics = values[model_name]
                writer.writerow([population, model_name] + [metrics.get(name) for name in ('targets', 'mae', 'rmse', 'bias', 'r2', 'slope', 'intercept')])
    with (output_dir / 'paired_ci.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['population', 'metric', 'events', 'delta_final_minus_base', 'ci_lower', 'ci_upper'])
        for population, values in groups.items():
            for metric, ci in values['paired_ci'].items():
                writer.writerow([population, metric, ci['events'], ci['delta'], ci['lower'], ci['upper']])
    with (output_dir / 'strata_counts.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
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
    field_records = []
    field_records.extend(_field_records(random_data, random_mask, 'random', 'all'))
    field_records.extend(_field_records(normal_data, normal_all, 'normal', 'all'))
    field_records.extend(_field_records(
        normal_data, normal_noninput, 'normal', 'noninput'
    ))
    with (output_dir / 'field_metrics.csv').open('w', newline='') as handle:
        if field_records:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(field_records[0]),
                lineterminator='\n',
            )
            writer.writeheader()
            writer.writerows(field_records)
    readme = [
        '# RT59-v3 paired validation analysis', '',
        f'- Decision: **{decision}**',
        f'- Random targets: {int(random_mask.sum()):,}',
        f'- Normal targets: {int(normal_all.sum()):,} '
        f'(input {int(normal_input.sum()):,}, non-input {int(normal_noninput.sum()):,})',
        f'- Bootstrap: event_id paired, {args.bootstrap_draws} draws, seed {args.bootstrap_seed}', '',
        '- Corrected field range: within-field P95-P05 (NumPy default linear percentile), '
        'equal field weighting, at least five valid targets for range gates.',
        '- Gate inventory: 25 required decision gates + 4 diagnostic bias/slope gates.',
        '- Dataset identity: formal `val` exports from Japan full 2000-2024, not Japan 2018.', '',
        'The decision is conjunctive. `INCOMPLETE_EVIDENCE` means at least one required '
        'quantity was absent; it must not be treated as a pass.', '',
        '| Gate | Value | Rule | Pass | Required |', '|---|---:|---:|:---:|:---:|',
    ]
    for item in gates:
        readme.append(
            f"| {item['gate']} | {item['value']} | {item['rule']} | "
            f"{item['pass']} | {item['required']} |"
        )
    (output_dir / 'README.md').write_text('\n'.join(readme) + '\n')
    print(json.dumps({'decision': decision, 'output_dir': str(output_dir.resolve())}, indent=2))


if __name__ == '__main__':
    main()
