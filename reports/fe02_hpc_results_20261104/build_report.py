#!/usr/bin/env python
"""Read existing FE02 artifacts; export portable evidence without model execution.

Run the unmodified fe02.analysis.compare first into --comparison. This companion
checks its inputs and exports compact tables. It never writes to --root.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from fe01.config import fingerprint, sha256, write_json
from fe01.metrics import checkpoint_selection, grouped_metrics, score_rows, summarize
from fe02.analysis import KEYS

VARIANTS = ('R0', 'A', 'B', 'C', 'M')
TIMES = (1, 3, 5, 10, 20)
PROB = ('nll', 'crps', 'brier', 'covered95', 'covered68', 'covered_mean_sigma',
        'covered_mean_2sigma', 'width95', 'predictive_sigma')
COORDS = ['latitude', 'longitude', 'event_latitude', 'event_longitude']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def portable(value):
    """Replace deployment roots, including paths used as JSON dictionary keys."""
    if isinstance(value, dict):
        return {portable(k): portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, str):
        if not value.startswith('/public/'):
            return re.sub(r'/public/[^\s"\x27]+', lambda m: portable(m.group()), value)
        if value.startswith('/public/'):
            if '/origin_corrected_diting_vel_acc_vs30/' in value:
                return '${FE02_DATA_ROOT}/' + value.split('/origin_corrected_diting_vel_acc_vs30/', 1)[1]
            if '/fe02_runs_20261008' in value:
                return '${FE02_OUTPUT_ROOT}' + value.split('/fe02_runs_20261008', 1)[1]
            if '/team_pytorch_fe02/' in value:
                return '${FE02_CODE_ROOT}/' + value.split('/team_pytorch_fe02/', 1)[1]
            return '${HPC_EXTERNAL_ROOT}/' + Path(value).name
        return value
    return value


def save_json(path, value):
    write_json(path, portable(value))


def save_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format='%.12g')


def read_frame(path, columns=None):
    frame = pd.read_csv(path, usecols=columns, dtype={'dataset_id': str, 'event_id': str, 'station_id': str})
    return frame[columns] if columns is not None else frame


def source_check(bundle):
    with tarfile.open(bundle) as archive:
        identity = json.load(archive.extractfile('SOURCE_IDENTITY.json'))
    files = identity['files_sha256']
    backend = {k: v for k, v in files.items() if
               (k.startswith('fe01/') and k.endswith('.py')) or
               k in ('gemini_models.py', 'gemini_util_light.py', 'train_light.py') or
               (k.startswith('tools/') and k.endswith('.py')) or
               (k.startswith('diting/config/') and k.endswith('.yml'))}
    backend.update({k.removeprefix('vendor/'): v for k, v in files.items()
                    if k.startswith('vendor/dtbench/') and k.endswith('.py')})
    extra = {k: v for k, v in files.items() if
             (k.startswith('fe02/') and k.endswith('.py')) or
             (k.startswith('scripts/fe02/') and '/' not in k[len('scripts/fe02/'):]) or
             (k.startswith('configs/fe02/') and k.endswith('.json')) or
             k == 'configs/fe01/formal/formal__diting_pretrained_frozen__seed42.json'}
    with tarfile.open(bundle) as archive:
        for name, expected in files.items():
            require(hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected,
                    'Shipped archive member hash mismatch: ' + name)
    local = {k: v for k, v in {**backend, **extra}.items() if not k.startswith('dtbench/')}
    require(all(sha256(REPO / k) == v for k, v in local.items()), 'Local scientific source differs from shipped source')
    return dict(source_commit=identity['commit'], bundle_sha256=sha256(bundle),
                backend_files=len(backend), experiment_files=len(extra),
                checked_local_files=len(local), checked_archive_members=len(files), local_scientific_files_match=True,
                code_sha256=fingerprint(dict(backend=fingerprint(backend), files=extra)),
                source_file_sha256={**backend, **extra})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--comparison', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-bundle', default='artifacts/fe02/fe02_source_20261009_conda_fix.tar.gz')
    args = parser.parse_args()
    root, comparison, out = map(Path, (args.root, args.comparison, args.output))
    require(root.resolve() not in out.resolve().parents, 'Output must be outside experiment root')
    require(not out.exists(), 'Use a new output directory')
    out.mkdir(parents=True)
    source = source_check(REPO / args.source_bundle)
    save_json(out / 'source_verification.json', source)
    print('Hashing all source artifacts', flush=True)
    inventory = []
    for path in sorted(root.rglob('*')):
        if path.is_file():
            inventory.append(dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=sha256(path)))
    save_csv(out / 'source_artifacts.csv', pd.DataFrame(inventory))

    reference = None
    shared = None
    checks, runs, curves_all, endpoints, event_cells, samples, epoch_scores, exclusions = [], [], [], [], [], [], [], []
    selection_columns = ['truth', 'prediction', 'status', 'elapsed_time', 'geometry_protocol', 'target_role']
    id_columns = KEYS + ['truth', 'input_count'] + COORDS + [
        'actual_elapsed_time', 'current_sample', 'cutout_exclusive', 'history_start_sample',
        'history_end_sample', 'input_coords', 'station_valid_count', 'storage_mask_rule']
    for variant in VARIANTS:
        print('Checking', variant, flush=True)
        run_id = f'fe02__formal__{variant}__seed42'
        run, audit = root / run_id, root / 'audits' / run_id
        cfg = json.loads((run / 'resolved_config.json').read_text())
        lock = json.loads((run / 'protocol.lock.json').read_text())
        provenance = json.loads((run / 'evaluation_fixed_val/provenance.json').read_text())
        interface = json.loads((run / 'model_interface_audit.json').read_text())
        freeze = json.loads((audit / 'freeze_gradient_audit.json').read_text())
        require(cfg['variant_id'] == variant and cfg['seed'] == 42 and cfg['sampling_seed'] == 42, 'Run identity')
        require(lock['status'] == 'AUDIT_PASS' and freeze['status'] == 'PASS', 'Audit not PASS')
        require(lock['code_sha256'] == source['code_sha256'], 'Shipped source does not reproduce lock identity')
        require(sha256(run / 'protocol.lock.json') == provenance['protocol_lock_sha256'], 'Evaluation lock hash mismatch')
        require(sha256(run / 'protocol.lock.json') == sha256(audit / 'protocol.lock.json'), 'Run/audit lock mismatch')
        require(sha256(run / 'model_interface_audit.json') == sha256(audit / 'model_interface_audit.json'), 'Run/audit interface mismatch')
        require(provenance['config_sha256'] == fingerprint(cfg), 'Resolved config hash mismatch')
        require(provenance['split'] == 'val' and provenance['stage'] == 'formal', 'Split/stage drift')
        require(provenance['random_times_manifest_sha256'] is None, 'Nonfixed evaluation')
        require(cfg['target_normalization']['fitted_split'] == 'train', 'Normalization leakage')
        require(all(freeze[k] is True for k in ['encoder_not_in_optimizer', 'parent_train_encoder_eval',
                    'encoder_parameters_and_buffers_unchanged']), 'Freeze gate not satisfied')
        for name in ('causal_boundary_audit.json', 'query_independence_audit.json'):
            value = json.loads((audit / name).read_text())
            require(value['status'] == 'PASS', name + ' not PASS')
        curves = pd.read_csv(run / 'training_curves.csv')
        require(curves.epoch.tolist() == list(range(1, 13)), 'Incomplete epoch curve')
        selected_epoch = int(curves.loc[curves.validation_selection.idxmin(), 'epoch'])
        require(selected_epoch == provenance['checkpoint_epoch'], 'Exported epoch differs from locked selection')
        for row in curves.to_dict('records'):
            records = list(run.glob(f"epoch{int(row['epoch']):04d}_*.json"))
            require(len(records) == 1, 'Missing or duplicate epoch record')
            recorded = json.loads(records[0].read_text())
            for k, v in row.items():
                require(np.isclose(v, recorded[k], rtol=1e-12, atol=1e-12), 'Curve/epoch mismatch: ' + k)
            frame_epoch = read_frame(run / f"validation_epoch{int(row['epoch'])}.csv.gz", selection_columns)
            selection = checkpoint_selection(frame_epoch, TIMES)
            require(np.isclose(selection, row['validation_selection'], rtol=0, atol=1e-12), 'Epoch selection not reproducible')
            epoch_scores.append(dict(variant_id=variant, epoch=int(row['epoch']),
                                     recorded_selection=row['validation_selection'], recomputed_selection=selection,
                                     targets=len(frame_epoch), events=None))
            del frame_epoch
        budget = fingerprint(curves[['epoch', 'updates', 'global_batch', 'planned_samples',
                                     'actual_samples', 'dropped_for_equal_update_budget']].to_dict('records'))
        current_shared = (budget, interface['unchanged_downstream_initial_sha256'], interface['adapter_initial_sha256'],
                          lock['encoder_checkpoint_sha256'], lock['data_identity'], cfg['audited_cohorts'],
                          cfg['target_normalization'], cfg['training'], cfg['geometry'], cfg['realtime'])
        if shared is None:
            shared = current_shared
            for split, population in cfg['audited_cohorts'].items():
                save_csv(out / f'cohort_{split}.csv', pd.DataFrame(population, columns=['dataset_id', 'event_id']))
            save_json(out / 'data_identity.json', lock['data_identity'])
        require(current_shared == shared, 'Unpaired budget/initialization/data/cohort/config')
        portable_cfg = dict(cfg)
        portable_cfg['audited_cohorts'] = {k: dict(events=len(v), fingerprint=fingerprint(v), file=f'cohort_{k}.csv')
                                            for k, v in cfg['audited_cohorts'].items()}
        portable_lock = dict(lock)
        portable_lock['effective_config'] = {k: v for k, v in lock['effective_config'].items() if k != 'audited_cohorts'}
        portable_lock['effective_config']['audited_cohorts'] = portable_cfg['audited_cohorts']
        save_json(out / 'evidence' / variant / 'resolved_config.portable.json', portable_cfg)
        save_json(out / 'evidence' / variant / 'protocol.lock.portable.json', portable_lock)
        for name in ('model_interface_audit.json', 'freeze_gradient_audit.json', 'causal_boundary_audit.json',
                     'query_independence_audit.json', 'sampler_distribution_audit.json', 'pretrained_manifest.json'):
            save_json(out / 'evidence' / variant / name, json.loads((audit / name).read_text()))
        save_json(out / 'evidence' / variant / 'provenance.portable.json', provenance)
        histories = []
        for path in sorted(run.glob('runtime_*.json')):
            value = json.loads(path.read_text())
            require(value['git_commit'] == source['source_commit'], 'Runtime source drift')
            histories.append(value)
        save_json(out / 'evidence' / variant / 'runtime_attempts.portable.json', histories)
        excluded = pd.read_csv(audit / 'cohort_exclusions.csv', dtype={'event_id': str})
        excluded['variant_id'] = variant
        exclusions.append(excluded)
        population = read_frame(audit / 'validation_population.csv.gz')
        require(fingerprint(population.to_dict('records')) == lock['validation_population_sha256'], 'Audit population hash mismatch')
        cols = list(dict.fromkeys(id_columns + ['prediction', 'split', 'variant_id', 'event_fusion', 'model_family',
              'seed', 'status', 'units', 'checkpoint_epoch', 'checkpoint_sha256', 'config_sha256', 'pit'] + list(PROB)))
        frame = read_frame(run / 'evaluation_fixed_val/predictions.csv.gz', cols).sort_values(KEYS).reset_index(drop=True)
        require(not frame.duplicated(KEYS).any(), 'Duplicate prediction keys')
        primary = frame.loc[frame.elapsed_time.isin(TIMES)]
        pd.testing.assert_frame_equal(primary[KEYS].reset_index(drop=True),
                                     population.sort_values(KEYS)[KEYS].reset_index(drop=True), check_dtype=False)
        current_identity = fingerprint(frame[id_columns].to_dict('records'))
        if reference is None:
            reference = current_identity
        require(current_identity == reference, 'Coordinates/masks/inputs/labels/order differ')
        for column, value in [('variant_id', variant), ('seed', 42), ('split', 'val'),
                              ('checkpoint_epoch', selected_epoch), ('checkpoint_sha256', provenance['checkpoint_sha256']),
                              ('config_sha256', provenance['config_sha256']), ('units', 'log10(m/s^2)'),
                              ('model_family', 'diting_pretrained_frozen'), ('status', 'supported')]:
            require(frame[column].eq(value).all(), 'Prediction identity/value mismatch: ' + column)
        require(np.isfinite(frame[['truth', 'prediction', 'pit'] + list(PROB)].to_numpy(float)).all(), 'Nonfinite metrics')
        status = pd.read_csv(run / 'evaluation_fixed_val/support_status.csv', dtype={'event_id': str})
        require(status.status.eq('supported').all() and len(status) == 1310 * 14, 'Missing/failed decision')
        exported_metrics = pd.read_csv(run / 'evaluation_fixed_val/metrics_by_time_target.csv')
        rebuilt_metrics = grouped_metrics(frame)
        keys = ['geometry_protocol', 'elapsed_time', 'target_group']
        pd.testing.assert_frame_equal(exported_metrics.sort_values(keys).reset_index(drop=True),
                                     rebuilt_metrics.sort_values(keys).reset_index(drop=True),
                                     check_dtype=False, rtol=1e-10, atol=1e-12)
        save_csv(out / 'evidence' / variant / 'metrics_by_time_target.csv', rebuilt_metrics)
        selected_frame = read_frame(run / f'validation_epoch{selected_epoch}.csv.gz', KEYS + ['truth', 'prediction']).sort_values(KEYS).reset_index(drop=True)
        pd.testing.assert_frame_equal(primary[KEYS + ['truth']].reset_index(drop=True),
                                     selected_frame[KEYS + ['truth']], check_dtype=False)
        prediction_difference = np.abs(primary.prediction.to_numpy() - selected_frame.prediction.to_numpy())
        # Separate inference is not bitwise reproducible. Record the observed
        # discrepancy and use the original query-audit atol=rtol=1e-5.
        require(np.allclose(primary.prediction, selected_frame.prediction, atol=1e-5, rtol=1e-5),
                'Export/selected-epoch prediction difference exceeds query-audit tolerance')
        # Independently score a deterministic MDN sample across the whole export.
        sample = read_frame(run / 'evaluation_fixed_val/predictions.csv.gz',
                            KEYS + ['truth', 'mdn_weights', 'mdn_mu', 'mdn_sigma', 'prediction'] + list(PROB))
        sample = sample.iloc[np.linspace(0, len(sample)-1, 512, dtype=int)].copy()
        arrays = [np.asarray(sample[k].map(json.loads).tolist()) for k in ('mdn_weights', 'mdn_mu', 'mdn_sigma')]
        require(np.allclose(arrays[0].sum(-1), 1) and (arrays[2] > 0).all(), 'Invalid sampled MDN')
        scores = score_rows(sample.truth.to_numpy(), *arrays, threshold=cfg['selection']['threshold_log10_pga'])
        for name in ['prediction'] + list(PROB):
            require(np.allclose(scores[name], sample[name].to_numpy(), atol=1e-9, rtol=1e-9), 'MDN sample mismatch: ' + name)
        sample.insert(0, 'variant_id', variant)
        samples.append(sample)
        for domain, part in [('primary_1_3_5_10_20', primary), ('diagnostic_40_90', frame.loc[~frame.elapsed_time.isin(TIMES)])]:
            for count, condition in [('all_counts', part.input_count.ge(1)), ('single', part.input_count.eq(1)), ('multi', part.input_count.ge(2))]:
                current = part.loc[condition]
                for target, mask in [('all', current.target_role.notna()), ('noninput', current.target_role.ne('observed_input')),
                                     ('observed_input', current.target_role.eq('observed_input')),
                                     ('triggered_noninput', current.target_role.eq('triggered_noninput')),
                                     ('untriggered_noninput', current.target_role.eq('untriggered_noninput'))]:
                    selected = current.loc[mask]
                    if selected.empty:
                        continue
                    for protocol in ('equal_normal_random', 'normal', 'random'):
                        selected_protocol = selected if protocol == 'equal_normal_random' else selected.loc[selected.geometry_protocol.eq(protocol)]
                        if selected_protocol.empty:
                            continue
                        metric = summarize(selected_protocol)
                        cells = grouped_metrics(selected_protocol).loc[lambda x: x.target_group.eq('all')]
                        endpoint = dict(variant_id=variant, seed=42, checkpoint_epoch=selected_epoch, split='val', domain=domain,
                                        input_count_stratum=count, target_group=target, geometry_protocol=protocol,
                                        existing_cells=len(cells), **metric)
                        for name in ['mae', 'bias'] + list(PROB):
                            endpoint['equal_cell_' + name] = float(cells[name].mean())
                        endpoint['equal_cell_rmse'] = float(np.sqrt(np.mean(cells.rmse.to_numpy() ** 2)))
                        endpoints.append(endpoint)
        remote = primary.loc[primary.target_role.ne('observed_input')].copy()
        remote['input_count_stratum'] = np.where(remote.input_count.eq(1), 'single', 'multi')
        remote['abs_error'] = np.abs(remote.prediction - remote.truth)
        grouped = remote.groupby(['dataset_id', 'event_id', 'geometry_protocol', 'elapsed_time', 'input_count_stratum']).abs_error.agg(['sum', 'count']).reset_index()
        grouped.insert(0, 'variant_id', variant)
        event_cells.append(grouped.rename(columns={'sum': 'sum_abs_error', 'count': 'targets'}))
        curve = curves.copy(); curve.insert(0, 'variant_id', variant); curves_all.append(curve)
        runs.append(dict(variant_id=variant, seed=42, split='val', run_id=run_id, epochs=12,
                         updates=int(curves.updates.iloc[-1]), checkpoint_epoch=selected_epoch,
                         selected_mae=float(curves.validation_selection.min()),
                         checkpoint_sha256=provenance['checkpoint_sha256'], checkpoint_bytes_locally_verified=False,
                         primary_all_targets=len(primary), primary_noninput_targets=len(remote),
                         primary_events=primary.event_id.nunique(), full_fixed_targets=len(frame),
                         full_decisions=len(status), train_events=len(cfg['audited_cohorts']['train']),
                         validation_events=len(cfg['audited_cohorts']['val']),
                         total_parameters=interface['total_parameters'], trainable_parameters=interface['trainable_parameters'],
                         active_gradient_parameters=freeze['active_gradient_parameters'],
                         nonzero_gradient_parameters=freeze['nonzero_gradient_parameters'],
                         epoch_wall_seconds=curves.elapsed_seconds.sum(), world_size_from_receipts=16,
                         final_job_id=provenance['runtime']['job_id'], encoder_checkpoint_sha256=lock['encoder_checkpoint_sha256']))
        checks.append(dict(variant_id=variant, status='PASS', epoch_rows=12,
                           epoch_selection_recomputed=12, sampled_mdn_rows=512,
                           max_abs_epoch_export_prediction_delta=float(prediction_difference.max()),
                           mean_abs_epoch_export_prediction_delta=float(prediction_difference.mean()),
                           export_selection_minus_recorded=checkpoint_selection(frame, TIMES)-float(curves.validation_selection.min()),
                           max_selection_difference=max(abs(r['recorded_selection']-r['recomputed_selection']) for r in epoch_scores if r['variant_id']==variant),
                           complete_prediction_identity_sha256=current_identity))
        print('PASS', variant, 'best epoch', selected_epoch, 'score', curves.validation_selection.min(), flush=True)

    save_csv(out / 'runs_manifest.csv', pd.DataFrame(runs))
    save_csv(out / 'training_curves.csv', pd.concat(curves_all, ignore_index=True))
    save_csv(out / 'epoch_selection_recomputed.csv', pd.DataFrame(epoch_scores).drop(columns='events'))
    save_csv(out / 'endpoints.csv', pd.DataFrame(endpoints))
    save_csv(out / 'primary_event_cells_noninput.csv', pd.concat(event_cells, ignore_index=True))
    save_csv(out / 'mdn_scoring_sample.csv', pd.concat(samples, ignore_index=True))
    # Identical cohort exclusions need only one portable table.
    require(all(x.drop(columns='variant_id').equals(exclusions[0].drop(columns='variant_id')) for x in exclusions), 'Exclusions differ')
    save_csv(out / 'cohort_exclusions.csv', exclusions[0].drop(columns='variant_id'))

    print('Summarizing spatial fields and device receipts', flush=True)
    contract = json.loads((comparison / 'comparison_contract.json').read_text())
    require(contract['status'] == 'PASS' and contract['bootstrap_draws'] == 5000, 'Original comparison not PASS/5000')
    require(contract['seeds'] == [42], 'Unexpected comparison seeds')
    save_json(out / 'comparison_contract.json', contract)
    for name in ('headline_primary.csv', 'metrics_time_geometry_count_target.csv', 'paired_event_bootstrap.csv'):
        save_csv(out / name, pd.read_csv(comparison / name))
    fields = pd.read_csv(comparison / 'spatial_fields.csv', dtype={'event_id': str})
    spatial_rows = []
    for key, part in fields.groupby(['variant_id', 'input_count_stratum', 'geometry_protocol', 'elapsed_time']):
        row = dict(zip(['variant_id', 'input_count_stratum', 'geometry_protocol', 'elapsed_time'], key))
        row.update(fields=len(part), events=part.event_id.nunique(), targets=int(part.targets.sum()),
                   range_ratio_eligible=int(part.range_ratio.notna().sum()),
                   range_ratio_excluded=int(part.range_ratio.isna().sum()))
        for name in ['field_mean_error', 'centered_mae', 'pairwise_delta_mae', 'range_error', 'range_ratio',
                     'std_ratio', 'equal_distance_delta_mae', 'equal_distance_sign_accuracy']:
            row[name + '_mean'] = part[name].mean()
            row[name + '_median'] = part[name].median()
        spatial_rows.append(row)
    save_csv(out / 'spatial_summary_by_cell.csv', pd.DataFrame(spatial_rows))
    # One record per field; single/multi can be reconstructed using input_count.
    primary_fields = fields.loc[fields.input_count_stratum.eq('all_counts') & fields.elapsed_time.isin(TIMES)]
    keep = ['variant_id', 'event_id', 'geometry_protocol', 'elapsed_time', 'targets', 'input_count',
            'field_mean_error', 'centered_mae', 'pairwise_delta_mae', 'range_error', 'range_ratio', 'range_ratio_excluded']
    save_csv(out / 'primary_spatial_fields.csv', primary_fields[keep])
    receipts = []
    for path in sorted((root / 'compatibility_receipts').glob('*.json')):
        value = json.loads(path.read_text())
        require(value['state_bytes_preserved'] is True and value['restore_device'] == 'cpu', 'RNG receipt failed')
        require(value['shim_sha256'] == sha256(REPO / 'compat/fe02_rng_cpu/sitecustomize.py'), 'Shim hash mismatch')
        require(value['original_source_commit'] == source['source_commit'], 'Receipt source mismatch')
        receipts.append(dict(file=path.name, launcher_matches_repository=
                             value['launcher_sha256'] == sha256(REPO / 'resume_fe02_rng_cpu.sh'), **value))
    save_json(out / 'rng_receipts.portable.json', receipts)
    receipt_groups = []
    for (job, index), group in pd.DataFrame(receipts).groupby(['job_id', 'array_index']):
        require(sorted(group['rank']) == list(range(16)), 'Missing/duplicate rank receipts')
        receipt_groups.append(dict(job_id=job, array_index=index, variant_id=VARIANTS[int(index)],
                                   ranks=len(group), min_rank=int(group['rank'].min()), max_rank=int(group['rank'].max()),
                                   launcher_sha256=group.launcher_sha256.iloc[0],
                                   launcher_matches_repository=bool(group.launcher_matches_repository.all()),
                                   status='CPU_RESTORE_RECEIPTS_PASS'))
    save_csv(out / 'rng_receipt_jobs.csv', pd.DataFrame(receipt_groups))
    log_rows, excerpts = [], []
    for path in sorted((root / 'slurm').iterdir()):
        lines = path.read_text(errors='replace').splitlines()
        selected = []
        for number, line in enumerate(lines, 1):
            if line.startswith('{"epoch"') or any(x in line for x in ['FE02 completed', 'DUE TO TIME LIMIT',
                'PS1: unbound', 'TypeError: RNG state', '[FE02-RNG-CPU]', 'AUDIT_PASS']):
                selected.append(dict(file=path.name, line=number, text=line))
        excerpts.extend(selected)
        log_rows.append(dict(file=path.name, bytes=path.stat().st_size,
                             training_completion_message=any('FE02 completed training and normal/random validation' in x for x in lines),
                             time_limit=any('DUE TO TIME LIMIT' in x for x in lines),
                             rng_byte_tensor_error=any('TypeError: RNG state must be a torch.ByteTensor' in x for x in lines),
                             epoch_records=sum(x.startswith('{"epoch"') for x in lines)))
    save_csv(out / 'slurm_log_summary.csv', pd.DataFrame(log_rows))
    save_json(out / 'slurm_evidence_excerpts.json', excerpts)
    save_json(out / 'verification.json', dict(status='NUMERIC_AND_PAIRING_PASS_WITH_PROVENANCE_GAPS', runs=checks,
        repository='rular099/team_pytorch', branch='exp/fe02-event-fusion-readout',
        analysis_base_commit='42b3d5405d9fe9672f0a0925275ca4f7e9e6a9e7',
        runtime_source_commit=source['source_commit'], code_sha256=source['code_sha256'],
        artifact_files=len(inventory), artifact_bytes=sum(x['bytes'] for x in inventory),
        matched_scientific_identity_sha256=reference, rng_receipts=len(receipts), rng_jobs=len(receipt_groups),
        numpy=np.__version__, pandas=pd.__version__,
        checkpoint_payloads='NOT_AVAILABLE; epoch and SHA are recorded export claims checked against rows/selection',
        training_journals='NOT_AVAILABLE; no actual sample-frequency histogram can be certified',
        scheduler_state_exit_code='NOT_AVAILABLE; no sacct in supplied artifacts',
        launcher_identity='Final R0/A/B/C receipt launcher hashes differ; executable bytes unavailable. Shim and scientific source match.',
        probability_check='all exported probability columns finite; 512 deterministic MDN rows/run independently rescored',
        compared_npz='NOT_OPENED; csv is the explicit analysis source; NPZ SHA retained',
        hpc_actions='none; read-only local artifact analysis, no model forward/training'))
    print('Evidence export PASS', flush=True)


if __name__ == '__main__':
    main()
