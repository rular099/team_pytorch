#!/usr/bin/env python3
"""Reproduce the formal RT60 random/normal validation decision.

The RT60 candidate, its same-forward immutable RT59 reference, and the
historical RT57 gamma=1.66 prediction are all exported in the same NPZ.  This
analyzer deliberately keeps those three roles separate and evaluates the
pre-registered RT60 mechanism gates independently from the legacy gates.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

try:
    from tools.analyze_rt59_dual_objective_npz import (
        _canonical_field_range,
        _event_manifest,
        _gate,
        _git_value,
        _load_protocol,
        _mixture_nll,
        _mixture_stats,
        _optional,
        _paired_event_ci,
        _point_metrics,
        _sha256_file,
        _stack,
        _tail_probability,
        _target_matrix,
        _write_sanitized_config,
        _write_truth_prediction_figure,
    )
except ModuleNotFoundError:
    from analyze_rt59_dual_objective_npz import (
        _canonical_field_range,
        _event_manifest,
        _gate,
        _git_value,
        _load_protocol,
        _mixture_nll,
        _mixture_stats,
        _optional,
        _paired_event_ci,
        _point_metrics,
        _sha256_file,
        _stack,
        _tail_probability,
        _target_matrix,
        _write_sanitized_config,
        _write_truth_prediction_figure,
    )


TASK_ID = '20260920-rt60-final-contrast-readout'
PARENT_TASK_ID = '20260915-rt59-dual-objective-transport-v3'
SOURCE_MANIFEST_SHA256 = (
    '405d533f7613adb4a6e52ac23096ae41a1c50d173417265b500c943bb7a27a38'
)
BOOTSTRAP_SEED = 20260915
EXPECTED_COUNTS = {
    'random_rows': 9681,
    'random_events': 1310,
    'random_targets': 75654,
    'normal_rows': 9681,
    'normal_events': 1383,
    'normal_targets': 89770,
    'normal_input_targets': 63651,
    'normal_noninput_targets': 26119,
    'random_fields_ge5': 5762,
    'random_one_station_fields_ge5': 1494,
}


def _load_metrics_identity(path):
    path = Path(path)
    payload = json.loads(path.read_text())
    protocol = payload.get('metric_protocol') or {}
    metadata = payload.get('checkpoint_metadata') or {}
    reference = metadata.get('rt60_reference_identity') or {}
    source = reference.get('source_identity') or {}
    checks = {
        'coordinate': protocol.get('pga_coordinate') == 'log10(m/s^2)',
        'point_estimate': protocol.get('point_estimate') == 'predictive_mixture_mean',
        'validation_only': payload.get('splits') == ['val'],
        'epoch8': metadata.get('epoch') == 8,
        'task_id': metadata.get('task_id') == TASK_ID,
        'parent_epoch8': reference.get('parent_epoch') == 8,
        'parent_task_id': reference.get('parent_task_id') == PARENT_TASK_ID,
        'source_manifest': (
            source.get('rt60_uploaded_source_manifest_sha256')
            == SOURCE_MANIFEST_SHA256
        ),
        'no_heldout_test_contract': source.get('split_contract')
        == 'inherited full 2000-2024 train/dev; no held-out test',
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f'formal metrics identity checks failed: {failed}')
    normalization = reference.get('normalization') or {}
    if not normalization.get('enabled'):
        raise ValueError('RT60 normalization metadata is absent or disabled.')
    mean = float(normalization['mean'])
    std = float(normalization['std'])
    if not np.isfinite([mean, std]).all() or std <= 0:
        raise ValueError('invalid RT60 normalization metadata.')
    return {
        'file': path.name,
        'sha256': _sha256_file(path),
        'checks': checks,
        'checkpoint': {
            'epoch': metadata['epoch'],
            'loss': metadata.get('loss'),
            'checkpoint_format': metadata.get('checkpoint_format'),
            'task_id': metadata['task_id'],
        },
        'reference_identity': reference,
        'normalization': {'mean': mean, 'std': std},
    }


def _raw_mdn(value, normalization):
    value = np.asarray(value, dtype=np.float64).copy()
    value[..., 1] = value[..., 1] * normalization['std'] + normalization['mean']
    value[..., 2] = value[..., 2] * normalization['std']
    return value


def _load_rt60_protocol(npz_path, metrics_identity):
    # This performs RT59's strict target/route/station-count/alignment checks and
    # retains the historical RT57 gamma=1.66 prediction under its old meaning.
    legacy = _load_protocol(npz_path)
    npz = np.load(npz_path, allow_pickle=True)
    valid = legacy['valid']
    reference = _target_matrix(_stack(npz, 'val_rt60_reference_mean'))
    increment = _target_matrix(_stack(npz, 'val_rt60_increment'))
    reference_mdn = np.asarray(_stack(npz, 'val_rt60_reference_mdn'), dtype=np.float64)
    candidate_mdn = _raw_mdn(
        _stack(npz, 'val_pga_pred'), metrics_identity['normalization']
    )
    historical_mdn = np.asarray(_stack(npz, 'val_rt59_base_mdn'), dtype=np.float64)

    models = {}
    for name, mdn, exported_mean in (
        ('historical_rt57', historical_mdn, legacy['base']),
        ('reference_rt59', reference_mdn, reference),
        ('candidate_rt60', candidate_mdn, legacy['final']),
    ):
        weights, components, component_sigma, mean, sigma = _mixture_stats(mdn)
        maximum_mean_error = float(np.max(np.abs(mean[valid] - exported_mean[valid])))
        if maximum_mean_error > 1e-6:
            raise ValueError(
                f'{name} MDN mean disagrees with exported mean: {maximum_mean_error:.9g}'
            )
        models[name] = {
            'mean': exported_mean,
            'weights': weights,
            'components': components,
            'component_sigma': component_sigma,
            'sigma': sigma,
            'nll': _mixture_nll(weights, components, component_sigma, legacy['target']),
            'probability': _tail_probability(
                weights, components, component_sigma, -1.2
            ),
            'mdn_mean_max_abs_error': maximum_mean_error,
        }

    reconstruction_error = float(np.max(np.abs(
        reference[valid] + increment[valid] - legacy['final'][valid]
    )))
    if reconstruction_error > 1e-6:
        raise ValueError(
            'RT60 reference + increment does not reconstruct candidate: '
            f'{reconstruction_error:.9g}'
        )
    final_nll = _optional(npz, 'val_pga_nll_log10_mps2')
    final_sigma = _optional(npz, 'val_pga_sigma')
    final_probability = _optional(npz, 'val_pga_prob_ge_threshold')
    exported = {
        'nll': np.asarray(final_nll, dtype=np.float64).reshape(valid.shape),
        'sigma': np.asarray(final_sigma, dtype=np.float64).reshape(valid.shape),
        'probability': np.asarray(final_probability, dtype=np.float64).reshape(valid.shape),
    }
    export_errors = {}
    for name, value in exported.items():
        error = float(np.max(np.abs(value[valid] - models['candidate_rt60'][name][valid])))
        export_errors[name] = error
        # Predictive sigma is reconstructed from a difference of second
        # moments; float32 serialization can amplify cancellation for nearly
        # deterministic mixtures.  The component-level identity check below
        # remains at 1e-6 and is the stronger architectural assertion.
        tolerance = 1e-5 if name == 'sigma' else 1e-6
        if error > tolerance:
            raise ValueError(f'candidate exported {name} mismatch: {error:.9g}')

    component_identity = {
        'logit_max_abs_difference': float(np.max(np.abs(
            candidate_mdn[..., 0][valid] - reference_mdn[..., 0][valid]
        ))),
        'component_sigma_max_abs_difference': float(np.max(np.abs(
            candidate_mdn[..., 2][valid] - reference_mdn[..., 2][valid]
        ))),
        'predictive_sigma_max_abs_difference': float(np.max(np.abs(
            models['candidate_rt60']['sigma'][valid]
            - models['reference_rt59']['sigma'][valid]
        ))),
    }
    if max(component_identity.values()) > 1e-6:
        raise ValueError(f'RT60 changed non-mean MDN quantities: {component_identity}')

    return {
        **legacy,
        'models': models,
        'increment': increment,
        'identity': {
            'candidate_reconstruction_max_abs_error': reconstruction_error,
            'candidate_export_max_abs_error': export_errors,
            'mdn_component_identity': component_identity,
        },
    }


def _population(data, mask, seed, draws):
    mask = np.asarray(mask, dtype=bool) & data['valid']
    truth_binary = data['target'] >= -1.2
    result = {'targets': int(mask.sum()), 'models': {}, 'paired': {}}
    for name, model in data['models'].items():
        metrics = _point_metrics(data['target'], model['mean'], mask)
        absolute = np.abs(model['mean'][mask] - data['target'][mask])
        metrics.update({
            'nll': float(np.mean(model['nll'][mask])),
            'brier': float(np.mean(
                (model['probability'][mask] - truth_binary[mask]) ** 2
            )),
            'coverage1': float(np.mean(absolute <= model['sigma'][mask])),
            'coverage2': float(np.mean(absolute <= 2.0 * model['sigma'][mask])),
            'predictive_sigma_mean': float(np.mean(model['sigma'][mask])),
        })
        result['models'][name] = metrics

    for baseline in ('reference_rt59', 'historical_rt57'):
        pair_name = f'candidate_rt60_minus_{baseline}'
        result['paired'][pair_name] = {}
        for metric in ('mae', 'rmse'):
            result['paired'][pair_name][metric] = _paired_event_ci(
                data['target'],
                data['models'][baseline]['mean'],
                data['models']['candidate_rt60']['mean'],
                mask,
                data['event_ids'],
                metric,
                seed,
                draws,
            )
        baseline_brier = (
            data['models'][baseline]['probability'] - truth_binary
        ) ** 2
        candidate_brier = (
            data['models']['candidate_rt60']['probability'] - truth_binary
        ) ** 2
        result['paired'][pair_name]['brier'] = _paired_event_ci(
            np.zeros_like(candidate_brier), baseline_brier, candidate_brier,
            mask, data['event_ids'], 'mean', seed, draws,
        )
    return result


def _field_records(data, mask):
    mask = np.asarray(mask, dtype=bool) & data['valid']
    records = []
    for row in range(mask.shape[0]):
        valid = np.flatnonzero(mask[row])
        if valid.size < 2:
            continue
        truth = data['target'][row, valid]
        pairs = np.triu_indices(valid.size, 1)
        truth_delta = truth[pairs[0]] - truth[pairs[1]]
        truth_range = _canonical_field_range(truth)
        record = {
            'row_index': row,
            'event_id': str(data['event_ids'][row]),
            'requested_elapsed_time_seconds': (
                None if data['requested_time'] is None
                else float(data['requested_time'][row])
            ),
            'actual_station_count': int(data['station_count'][row]),
            'n_targets': int(valid.size),
            'eligible_ge5': bool(valid.size >= 5),
            'truth_p95_p05_range': truth_range,
        }
        for name, model in data['models'].items():
            prediction = model['mean'][row, valid]
            pred_delta = prediction[pairs[0]] - prediction[pairs[1]]
            delta_error = pred_delta - truth_delta
            pred_range = _canonical_field_range(prediction)
            prefix = name.replace('_rt57', '').replace('_rt59', '').replace('_rt60', '')
            record.update({
                f'{prefix}_p95_p05_range': pred_range,
                f'{prefix}_range_ratio': (
                    pred_range / truth_range if truth_range > 0 else None
                ),
                f'{prefix}_range_abs_error': abs(pred_range - truth_range),
                f'{prefix}_pairwise_delta_mae': float(np.mean(np.abs(delta_error))),
                f'{prefix}_pairwise_delta_rmse': float(np.sqrt(np.mean(delta_error ** 2))),
            })
        records.append(record)
    return records


def _mean(records, key):
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return float(np.mean(values)) if values else None


def _field_summary(records):
    eligible = [record for record in records if record['eligible_ge5']]
    single = [record for record in eligible if record['actual_station_count'] == 1]
    result = {
        'fields_ge5': len(eligible),
        'one_station_fields_ge5': len(single),
        'all': {},
        'one_station': {},
    }
    for group, selected in (('all', eligible), ('one_station', single)):
        for model in ('historical', 'reference', 'candidate'):
            result[group][model] = {
                'range_ratio_mean': _mean(selected, f'{model}_range_ratio'),
                'range_abs_error_mean': _mean(selected, f'{model}_range_abs_error'),
                'pairwise_delta_mae': _mean(selected, f'{model}_pairwise_delta_mae'),
                'pairwise_delta_rmse': _mean(selected, f'{model}_pairwise_delta_rmse'),
            }
    return result


def _paired_field_ci(records, baseline_key, candidate_key, seed, draws):
    if not records:
        return {'events': 0, 'fields': 0, 'delta': None, 'lower': None, 'upper': None}
    event_ids = np.asarray([record['event_id'] for record in records])
    base = np.asarray([record[baseline_key] for record in records], dtype=np.float64)
    candidate = np.asarray([record[candidate_key] for record in records], dtype=np.float64)
    unique, inverse = np.unique(event_ids, return_inverse=True)
    counts = np.bincount(inverse, minlength=unique.size).astype(np.float64)
    base_sum = np.bincount(inverse, weights=base, minlength=unique.size)
    candidate_sum = np.bincount(inverse, weights=candidate, minlength=unique.size)
    point = float((candidate_sum.sum() - base_sum.sum()) / counts.sum())
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    chunk = 256
    for start in range(0, draws, chunk):
        stop = min(draws, start + chunk)
        sampled = rng.integers(0, unique.size, size=(stop - start, unique.size))
        denominator = counts[sampled].sum(axis=1)
        samples[start:stop] = (
            candidate_sum[sampled].sum(axis=1)
            - base_sum[sampled].sum(axis=1)
        ) / denominator
    return {
        'events': int(unique.size),
        'fields': int(len(records)),
        'draws': int(draws),
        'seed': int(seed),
        'delta': point,
        'lower': float(np.quantile(samples, 0.025)),
        'upper': float(np.quantile(samples, 0.975)),
    }


def _strata_rows(protocol, data):
    definitions = []
    type_names = {0: 'input', 1: 'triggered_noninput', 2: 'untriggered'}
    for value in sorted(np.unique(data['target_type'][data['valid']]).tolist()):
        definitions.append(('target_type', type_names.get(int(value), str(value)), data['target_type'] == value))
    for value in sorted(np.unique(data['station_count']).tolist()):
        definitions.append((
            'actual_station_count', str(int(value)),
            np.broadcast_to(data['station_count'][:, None] == value, data['valid'].shape),
        ))
    if data['requested_time'] is not None:
        for value in sorted(np.unique(data['requested_time']).tolist()):
            definitions.append((
                'requested_elapsed_time_seconds', f'{float(value):g}',
                np.broadcast_to(
                    np.isclose(data['requested_time'][:, None], value), data['valid'].shape
                ),
            ))
    rows = []
    for stratum, value, mask in definitions:
        mask = mask & data['valid']
        active_rows = mask.any(axis=1)
        row = {
            'protocol': protocol,
            'stratum': stratum,
            'value': value,
            'events': int(np.unique(data['event_ids'][active_rows]).size),
            'realtime_rows': int(active_rows.sum()),
            'targets': int(mask.sum()),
        }
        for model_name, model in data['models'].items():
            prefix = model_name.replace('_rt57', '').replace('_rt59', '').replace('_rt60', '')
            metrics = _point_metrics(data['target'], model['mean'], mask)
            row[f'{prefix}_mae'] = metrics.get('mae')
            row[f'{prefix}_rmse'] = metrics.get('rmse')
        rows.append(row)
    return rows


def _read_epoch_csv(directory, filename):
    path = Path(directory) / filename
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    steps = [int(row['step']) for row in rows]
    values = [float(row['value']) for row in rows]
    if steps != list(range(8)):
        raise ValueError(f'{filename} must contain exactly epoch steps 0..7: {steps}')
    if not np.isfinite(values).all():
        raise ValueError(f'{filename} contains non-finite values.')
    return values


def _training_summary(directory):
    filenames = {
        'train_loss': 'train_epoch_loss.csv',
        'val_loss': 'val_epoch_loss.csv',
        'rt60_total': 'rt60_epoch_loss_total.csv',
        'point': 'rt60_epoch_loss_point.csv',
        'contrast': 'rt60_epoch_loss_contrast.csv',
        'regret': 'rt60_epoch_loss_regret.csv',
        'nll': 'rt60_epoch_loss_nll.csv',
        'smooth': 'rt60_epoch_loss_smooth.csv',
        'mse': 'rt60_epoch_loss_mse.csv',
        'lr_readout': 'rt60_epoch_lr_readout.csv',
        'grad_pre_clip': 'rt60_epoch_readout_pre_clip_norm.csv',
        'grad_post_clip': 'rt60_epoch_readout_post_clip_norm.csv',
        'targets_random': 'rt60_epoch_targets_random.csv',
        'targets_normal_observed': 'rt60_epoch_targets_normal_observed.csv',
        'targets_normal_remote': 'rt60_epoch_targets_normal_remote.csv',
        'fields_random_single': 'rt60_epoch_fields_random_single.csv',
        'fields_random_multi': 'rt60_epoch_fields_random_multi.csv',
        'fields_normal_remote_single': 'rt60_epoch_fields_normal_remote_single.csv',
        'fields_normal_remote_multi': 'rt60_epoch_fields_normal_remote_multi.csv',
    }
    series = {name: _read_epoch_csv(directory, filename) for name, filename in filenames.items()}
    expected_lr = [1e-4] * 4 + [5e-5] * 2 + [2.5e-5] * 2
    if not np.allclose(series['lr_readout'], expected_lr, rtol=0.0, atol=1e-12):
        raise ValueError(f'RT60 LR schedule mismatch: {series["lr_readout"]}')
    rows = []
    for epoch in range(8):
        row = {'epoch': epoch + 1}
        row.update({name: values[epoch] for name, values in series.items()})
        rows.append(row)
    return {
        'epochs': rows,
        'selected_epoch': 8,
        'selection_rule': 'pre-registered final epoch; no validation checkpoint selection',
        'minimum_val_loss_epoch_diagnostic': int(np.argmin(series['val_loss']) + 1),
        'minimum_val_loss_diagnostic': float(np.min(series['val_loss'])),
        'final_val_loss': series['val_loss'][-1],
    }


def _validate_training_config(path):
    config = json.loads(Path(path).read_text())
    model = config['model_params']
    training = config['training_params']
    data_paths = training['data_path']
    years = [int(Path(item).stem.rsplit('_', 1)[-1]) for item in data_paths]
    checks = {
        'seed42': config.get('seed') == 42,
        'rt60_model_enabled': model.get('use_rt60_contrast_readout') is True,
        'freeze_mode': training.get('freeze_mode') == 'rt60_contrast_readout_only',
        'task_id': training.get('task_id') == TASK_ID,
        'eight_epochs': training.get('epochs_full_model') == 8,
        'full_25_shards_2000_2024': len(data_paths) == 25 and years == list(range(2000, 2025)),
        'rt59_objective_disabled': not training.get('rt59_dual_objective', {}).get('enabled'),
        'rt60_objective_enabled': training.get('rt60_contrast_objective', {}).get('enabled') is True,
        'parent_task': training.get('rt60_parent', {}).get('task_id') == PARENT_TASK_ID,
        'parent_epoch8': training.get('rt60_parent', {}).get('epoch') == 8,
        'source_manifest': training.get('rt60_parent', {}).get('source_identity', {}).get(
            'rt60_uploaded_source_manifest_sha256'
        ) == SOURCE_MANIFEST_SHA256,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f'training config contract checks failed: {failed}')
    return {'checks': checks, 'years': years, 'shards': len(data_paths)}


def _write_rows(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_npz', required=True)
    parser.add_argument('--normal_npz', required=True)
    parser.add_argument('--random_metrics', required=True)
    parser.add_argument('--normal_metrics', required=True)
    parser.add_argument('--training_config', required=True)
    parser.add_argument('--random_config', required=True)
    parser.add_argument('--normal_config', required=True)
    parser.add_argument('--training_log_dir', required=True)
    parser.add_argument('--source_zip')
    parser.add_argument('--source_tar_gz')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--bootstrap_draws', type=int, default=5000)
    parser.add_argument('--bootstrap_seed', type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.bootstrap_draws != 5000 or args.bootstrap_seed != BOOTSTRAP_SEED:
        raise ValueError('formal RT60 decision requires 5000 draws and seed 20260915.')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random_identity = _load_metrics_identity(args.random_metrics)
    normal_identity = _load_metrics_identity(args.normal_metrics)
    if random_identity['checkpoint'] != normal_identity['checkpoint']:
        raise ValueError('random/normal checkpoint identities disagree.')
    if random_identity['reference_identity'] != normal_identity['reference_identity']:
        raise ValueError('random/normal RT60 reference identities disagree.')
    random_data = _load_rt60_protocol(args.random_npz, random_identity)
    normal_data = _load_rt60_protocol(args.normal_npz, normal_identity)

    random_mask = random_data['valid']
    normal_all = normal_data['valid']
    normal_input = normal_all & (normal_data['target_type'] == 0)
    normal_noninput = normal_all & np.isin(normal_data['target_type'], [1, 2])
    if (random_mask & (random_data['target_type'] == 0)).any():
        raise ValueError('formal random validation contains input targets.')
    if not np.array_equal(normal_input | normal_noninput, normal_all):
        raise ValueError('normal target-type masks do not partition all valid targets.')

    groups = {
        'random': _population(random_data, random_mask, args.bootstrap_seed, args.bootstrap_draws),
        'normal_all': _population(normal_data, normal_all, args.bootstrap_seed, args.bootstrap_draws),
        'normal_input': _population(normal_data, normal_input, args.bootstrap_seed, args.bootstrap_draws),
        'normal_noninput': _population(normal_data, normal_noninput, args.bootstrap_seed, args.bootstrap_draws),
    }
    random_records = _field_records(random_data, random_mask)
    field_summary = _field_summary(random_records)
    eligible = [record for record in random_records if record['eligible_ge5']]
    single = [record for record in eligible if record['actual_station_count'] == 1]
    field_ci = {
        'candidate_minus_reference': {
            'all_range_abs_error': _paired_field_ci(
                eligible, 'reference_range_abs_error', 'candidate_range_abs_error',
                args.bootstrap_seed, args.bootstrap_draws,
            ),
            'one_station_range_abs_error': _paired_field_ci(
                single, 'reference_range_abs_error', 'candidate_range_abs_error',
                args.bootstrap_seed, args.bootstrap_draws,
            ),
            'one_station_pairwise_delta_mae': _paired_field_ci(
                single, 'reference_pairwise_delta_mae', 'candidate_pairwise_delta_mae',
                args.bootstrap_seed, args.bootstrap_draws,
            ),
        },
        'candidate_minus_historical': {
            'one_station_range_abs_error': _paired_field_ci(
                single, 'historical_range_abs_error', 'candidate_range_abs_error',
                args.bootstrap_seed, args.bootstrap_draws,
            ),
        },
    }

    actual_counts = {
        'random_rows': int(random_data['target'].shape[0]),
        'random_events': int(np.unique(random_data['event_ids'][random_mask.any(axis=1)]).size),
        'random_targets': int(random_mask.sum()),
        'normal_rows': int(normal_data['target'].shape[0]),
        'normal_events': int(np.unique(normal_data['event_ids'][normal_all.any(axis=1)]).size),
        'normal_targets': int(normal_all.sum()),
        'normal_input_targets': int(normal_input.sum()),
        'normal_noninput_targets': int(normal_noninput.sum()),
        'random_fields_ge5': field_summary['fields_ge5'],
        'random_one_station_fields_ge5': field_summary['one_station_fields_ge5'],
    }
    if actual_counts != EXPECTED_COUNTS:
        raise ValueError(f'formal count contract mismatch: {actual_counts}')

    def point_delta(population, metric, baseline='reference_rt59'):
        models = groups[population]['models']
        return models['candidate_rt60'][metric] - models[baseline][metric]

    normal_input_increment = normal_data['increment'][normal_input]
    observed_input_max_abs_change = float(np.max(np.abs(normal_input_increment)))
    mechanism_gates = [
        _gate('one_station_pairwise_delta_mae_ci_upper', field_ci['candidate_minus_reference']['one_station_pairwise_delta_mae']['upper'], 0.0, '<'),
        _gate('one_station_range_abs_error_ci_upper', field_ci['candidate_minus_reference']['one_station_range_abs_error']['upper'], 0.0, '<'),
        _gate('random_all_range_abs_error_delta', field_ci['candidate_minus_reference']['all_range_abs_error']['delta'], 0.0, '<='),
        _gate('random_mae_delta', point_delta('random', 'mae'), 0.0, '<='),
        _gate('random_rmse_delta', point_delta('random', 'rmse'), 0.0, '<='),
        _gate('normal_noninput_mae_delta', point_delta('normal_noninput', 'mae'), 0.0, '<='),
        _gate('normal_noninput_rmse_delta', point_delta('normal_noninput', 'rmse'), 0.0, '<='),
        _gate('normal_input_max_abs_change', observed_input_max_abs_change, 1e-7, '<='),
    ]
    for population in ('random', 'normal_all', 'normal_noninput'):
        for metric in ('nll', 'brier'):
            mechanism_gates.append(_gate(
                f'{population}_{metric}_delta', point_delta(population, metric), 0.0, '<='
            ))
    if len(mechanism_gates) != 14:
        raise AssertionError('RT60 mechanism gate inventory must contain 14 gates.')
    mechanism_pass = all(item['pass'] is True for item in mechanism_gates)

    requested1_mask = random_mask & np.isclose(
        random_data['requested_time'][:, None], 1.0
    )
    requested1 = _population(
        random_data, requested1_mask, args.bootstrap_seed, args.bootstrap_draws
    )
    legacy_pair = 'candidate_rt60_minus_historical_rt57'
    legacy_gates = [
        _gate('random_mae', groups['random']['models']['candidate_rt60']['mae'], 0.242, '<='),
        _gate('random_slope', groups['random']['models']['candidate_rt60']['slope'], 0.45, '>='),
        _gate('random_r2', groups['random']['models']['candidate_rt60']['r2'], 0.39, '>='),
        _gate('random_range_ratio_ge5', field_summary['all']['candidate']['range_ratio_mean'], 0.55, '>='),
        _gate('random_one_station_range_ratio_ge5', field_summary['one_station']['candidate']['range_ratio_mean'], 0.25, '>='),
        _gate('random_one_station_pairwise_delta_mae', field_summary['one_station']['candidate']['pairwise_delta_mae'], 0.330, '<='),
        _gate('random_requested1_mae', requested1['models']['candidate_rt60']['mae'], 0.253, '<='),
        _gate('random_nll', groups['random']['models']['candidate_rt60']['nll'], 0.22, '<='),
        _gate('random_delta_mae_ci_upper', groups['random']['paired'][legacy_pair]['mae']['upper'], 0.0, '<'),
        _gate('random_delta_brier_ci_upper', groups['random']['paired'][legacy_pair]['brier']['upper'], 0.0, '<='),
    ]
    for coverage in ('coverage1', 'coverage2'):
        legacy_gates.append(_gate(
            f'random_{coverage}_absolute_change',
            abs(point_delta('random', coverage, baseline='historical_rt57')),
            0.01, '<=',
        ))
    for population in ('normal_all', 'normal_noninput'):
        for metric in ('mae', 'rmse'):
            legacy_gates.append(_gate(
                f'{population}_delta_{metric}_ci_upper',
                groups[population]['paired'][legacy_pair][metric]['upper'], 0.0, '<',
            ))
    for metric in ('mae', 'rmse'):
        legacy_gates.append(_gate(
            f'normal_input_{metric}_delta',
            point_delta('normal_input', metric, baseline='historical_rt57'), 0.0, '<=',
        ))
    legacy_gates.extend([
        _gate('normal_all_mae_historical_redline', groups['normal_all']['models']['candidate_rt60']['mae'], 0.136, '<='),
        _gate('normal_noninput_mae_historical_redline', groups['normal_noninput']['models']['candidate_rt60']['mae'], 0.218, '<='),
        _gate('normal_nll_delta', point_delta('normal_all', 'nll', baseline='historical_rt57'), 0.0, '<='),
        _gate('normal_brier_delta', point_delta('normal_all', 'brier', baseline='historical_rt57'), 0.0, '<='),
        _gate('one_station_range_abs_error_delta', field_ci['candidate_minus_historical']['one_station_range_abs_error']['delta'], 0.0, '<='),
    ])
    for coverage in ('coverage1', 'coverage2'):
        legacy_gates.append(_gate(
            f'normal_{coverage}_absolute_change',
            abs(point_delta('normal_all', coverage, baseline='historical_rt57')),
            0.01, '<=',
        ))
    diagnostics = []
    for population in ('normal_all', 'normal_noninput'):
        candidate = groups[population]['models']['candidate_rt60']
        historical = groups[population]['models']['historical_rt57']
        diagnostics.extend([
            _gate(f'{population}_absolute_bias_change', abs(candidate['bias']) - abs(historical['bias']), 0.0, '<=', required=False),
            _gate(f'{population}_absolute_slope_error_change', abs(candidate['slope'] - 1.0) - abs(historical['slope'] - 1.0), 0.0, '<=', required=False),
        ])
    if len(legacy_gates) != 25 or len(diagnostics) != 4:
        raise AssertionError('legacy inventory must contain 25 required + 4 diagnostic gates.')
    legacy_full_go = all(item['pass'] is True for item in legacy_gates)

    training = _training_summary(args.training_log_dir)
    training_config_contract = _validate_training_config(args.training_config)
    config_sources = {}
    for label, source in (
        ('training', args.training_config),
        ('random_validation', args.random_config),
        ('normal_validation', args.normal_config),
    ):
        config_sources[label] = _write_sanitized_config(
            source, output_dir / f'{label}_config.sanitized.json'
        )
    archive_sources = {}
    for label, source in (('zip', args.source_zip), ('tar_gz', args.source_tar_gz)):
        if source:
            archive_sources[label] = {
                'file': Path(source).name,
                'sha256': _sha256_file(source),
                'size_bytes': Path(source).stat().st_size,
            }

    summary = {
        'schema_version': 1,
        'task_id': TASK_ID,
        'evaluation_status': 'validation_only_no_heldout_test',
        'rt60_mechanism_pass': mechanism_pass,
        'legacy_full_go': legacy_full_go,
        'recommendation': 'retain_rt59_reference' if not mechanism_pass else 'rt60_mechanism_supported',
        'counts': actual_counts,
        'expected_counts_verified': True,
        'bootstrap': {'draws': args.bootstrap_draws, 'seed': args.bootstrap_seed, 'unit': 'event_id cluster'},
        'sources': {
            'archives': archive_sources,
            'random_npz': {'file': Path(args.random_npz).name, 'sha256': _sha256_file(args.random_npz)},
            'normal_npz': {'file': Path(args.normal_npz).name, 'sha256': _sha256_file(args.normal_npz)},
            'random_metrics': random_identity,
            'normal_metrics': normal_identity,
            'sanitized_configs': config_sources,
        },
        'provenance': {
            'analysis_git_commit': _git_value(['rev-parse', 'HEAD']),
            'analysis_git_branch': _git_value(['branch', '--show-current']),
            'analysis_worktree_dirty': bool(_git_value(['status', '--porcelain'])),
            'rt60_implementation_commit': '033f01a481fdb7499aff836abe6aaae35b72e5ca',
            'rt60_submitted_source_manifest_sha256': SOURCE_MANIFEST_SHA256,
        },
        'split_identity': {
            'declared_dataset_period': 'Japan full 2000-2024',
            'declared_split': 'val',
            'random': _event_manifest(random_data),
            'normal': _event_manifest(normal_data),
        },
        'training_config_contract': training_config_contract,
        'training': training,
        'identity_checks': {
            'random': random_data['identity'],
            'normal': normal_data['identity'],
            'normal_input_increment_max_abs': observed_input_max_abs_change,
        },
        'groups': groups,
        'random_fields': field_summary,
        'field_paired_ci': field_ci,
        'requested_1s': requested1,
        'mechanism_gates': mechanism_gates,
        'legacy_gates': legacy_gates,
        'legacy_diagnostics': diagnostics,
    }
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n'
    )
    _write_rows(output_dir / 'mechanism_gates.csv', mechanism_gates)
    _write_rows(output_dir / 'legacy_gates.csv', legacy_gates + diagnostics)
    _write_rows(output_dir / 'field_metrics.csv', random_records)
    _write_rows(output_dir / 'strata_counts.csv', _strata_rows('random', random_data) + _strata_rows('normal', normal_data))
    _write_rows(output_dir / 'training_summary.csv', training['epochs'])

    group_rows = []
    for population, values in groups.items():
        for model, metrics in values['models'].items():
            group_rows.append({'population': population, 'model': model, **metrics})
    _write_rows(output_dir / 'group_metrics.csv', group_rows)
    ci_rows = []
    for population, values in groups.items():
        for comparison, metrics in values['paired'].items():
            for metric, ci in metrics.items():
                ci_rows.append({'population': population, 'comparison': comparison, 'metric': metric, **ci})
    for comparison, metrics in field_ci.items():
        for metric, ci in metrics.items():
            ci_rows.append({'population': 'random_fields', 'comparison': comparison, 'metric': metric, **ci})
    _write_rows(output_dir / 'paired_ci.csv', ci_rows)

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
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n'
    )

    mechanism_passes = sum(item['pass'] is True for item in mechanism_gates)
    legacy_passes = sum(item['pass'] is True for item in legacy_gates)
    readme = [
        '# RT60 final-contrast readout validation', '',
        f'- `rt60_mechanism_pass`: **{mechanism_pass}** ({mechanism_passes}/14 gates pass)',
        f'- `legacy_full_go`: **{legacy_full_go}** ({legacy_passes}/25 required gates pass)',
        '- Recommendation: retain the RT59 same-forward reference as the development parent.',
        '- Evidence: fixed epoch-8 Japan 2000-2024 validation only; no held-out test.',
        f'- Bootstrap: {args.bootstrap_draws} paired event-cluster draws, seed {args.bootstrap_seed}.', '',
        '## RT60 mechanism gates', '',
        '| Gate | Value | Rule | Pass |', '|---|---:|---:|:---:|',
    ]
    for item in mechanism_gates:
        readme.append(f"| {item['gate']} | {item['value']} | {item['rule']} | {item['pass']} |")
    readme.extend(['', '## Legacy gates', '', '| Gate | Value | Rule | Pass | Required |', '|---|---:|---:|:---:|:---:|'])
    for item in legacy_gates + diagnostics:
        readme.append(f"| {item['gate']} | {item['value']} | {item['rule']} | {item['pass']} | {item['required']} |")
    (output_dir / 'README.md').write_text('\n'.join(readme) + '\n')
    print(json.dumps({
        'rt60_mechanism_pass': mechanism_pass,
        'legacy_full_go': legacy_full_go,
        'mechanism_passes': mechanism_passes,
        'legacy_passes': legacy_passes,
        'output_dir': str(output_dir.resolve()),
    }, indent=2))


if __name__ == '__main__':
    main()
