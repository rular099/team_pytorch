#!/usr/bin/env python3
"""Summarize saved FE01 validation and training journals without model/HDF access.

Run from the FE01 repository. The destination must not already exist. Raw source
files are never modified. This is a result reader, not a training/evaluation job.
"""
import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from fe01.config import fingerprint, load, sha256
from fe01.metrics import checkpoint_selection, grouped_metrics, paired_bootstrap, score_rows, summarize

FAMILIES = ('team_original_scratch', 'phasenet_pretrained_frozen', 'eqt_pretrained_frozen')
POPULATION_KEYS = ['dataset_id', 'event_id', 'elapsed_time', 'geometry_protocol',
                   'station_id', 'target_role', 'input_ids']
PAIR_KEYS = ['dataset_id', 'event_id', 'elapsed_time', 'station_id', 'geometry_protocol']
METRIC_COLUMNS = ['prediction', 'truth', 'status', 'nll', 'crps', 'brier', 'covered95',
                  'covered68', 'covered_mean_sigma', 'covered_mean_2sigma', 'width95',
                  'predictive_sigma']
READ_COLUMNS = list(dict.fromkeys(POPULATION_KEYS + METRIC_COLUMNS + [
    'split', 'seed', 'model_family', 'units', 'requested_decision_sample',
    'current_sample', 'cutout_exclusive', 'history_start_sample', 'history_end_sample',
    'latest_received_sample', 'input_count', 'mdn_weights', 'mdn_mu', 'mdn_sigma']))


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expand(value, replacements):
    if isinstance(value, dict):
        return {k: expand(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v, replacements) for v in value]
    return Template(value).substitute(replacements) if isinstance(value, str) else value


def roots(cfg):
    return dict(FE01_CODE_ROOT=str(Path(cfg['diting_config']).parents[2]),
                FE01_WEIGHTS_ROOT=str(Path(cfg['pretrained_manifest']).parent),
                FE01_OUTPUT_ROOT=cfg['output_root'], DATA_ROOT=cfg['data']['root'],
                FE01_SPLIT_MANIFEST=cfg['data']['split_manifest'],
                FE01_SPATIAL_MANIFEST=cfg['spatial']['manifest'])


def portable(value, replacements):
    """Publish a readable snapshot with root placeholders, not runnable config."""
    if isinstance(value, dict):
        return {portable(k, replacements): portable(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v, replacements) for v in value]
    if isinstance(value, str):
        for name, path in sorted(replacements.items(), key=lambda item: -len(item[1])):
            value = value.replace(path, '${' + name + '}')
    return value


def audited_source_identity():
    """Same source-file scope as engine.code_identity, without importing torch."""
    spec = importlib.util.find_spec('dtbench')
    require(spec is not None, 'dtbench source snapshot is needed for code identity verification')
    dependency = Path(next(iter(spec.submodule_search_locations)))
    paths = list((REPO / 'fe01').glob('*.py'))
    paths += [REPO / name for name in ('gemini_models.py', 'gemini_util_light.py', 'train_light.py')]
    paths += list((REPO / 'tools').rglob('*.py')) + list((REPO / 'diting/config').glob('*.yml'))
    files = {str(p.relative_to(REPO)): sha256(p) for p in paths}
    files.update({'dtbench/' + str(p.relative_to(dependency)): sha256(p)
                  for p in dependency.rglob('*.py')})
    return files, fingerprint(files)


def journal_summary(run_dir):
    """Read every available formal journal, checking its epoch/rank and timing."""
    run_dir = Path(run_dir)
    epochs, bins, geometry, inputs, statuses = Counter(), Counter(), Counter(), Counter(), Counter()
    rows, seen, file_records = 0, set(), []
    time_min, time_max = float('inf'), float('-inf')
    boundary_violations = 0
    edges = [(1, 3), (3, 5), (5, 10), (10, 20), (20, 40), (40, 90)]
    for path in sorted((run_dir / 'sample_journals').glob('*.jsonl')):
        match = re.fullmatch(r'epoch(\d+)_\d+T\d+_rank(\d+)\.jsonl', path.name)
        require(match is not None, 'Unexpected journal filename: ' + path.name)
        epoch, rank = map(int, match.groups())
        require((epoch, rank) not in seen, 'Multiple journal attempts for epoch/rank: ' + str(path))
        seen.add((epoch, rank))
        digest, count = hashlib.sha256(), 0
        with path.open('rb') as stream:
            for line in stream:
                digest.update(line)
                item = json.loads(line)
                require(item['epoch'] == epoch and item['rank'] == rank, 'Journal identity mismatch')
                require(item['split'] == 'train' and item['units'] == 'log10(m/s^2)', 'Journal contract mismatch')
                time = float(item['actual_elapsed_time'])
                require(np.isfinite(time), 'Nonfinite journal time')
                index = next((i for i, (a, b) in enumerate(edges)
                              if a <= time < b or (i == 5 and time == b)), None)
                require(index is not None, 'Journal time outside registered bins')
                bins[index] += 1
                geometry[item['geometry_protocol']] += 1
                inputs[item['input_count']] += 1
                statuses[item['status']] += 1
                # The public reader checks saved sample boundaries; it does not
                # independently certify the offline waveform preprocessing.
                cutoff, current = item['cutout_exclusive'], item['current_sample']
                violation = cutoff != current + 1 or item['history_end_sample'] > current
                latest = item.get('latest_received_sample')
                violation |= latest is not None and latest > current
                boundary_violations += int(violation)
                count += 1
                time_min, time_max = min(time_min, time), max(time_max, time)
        epochs[epoch] += count
        rows += count
        file_records.append(dict(relative_path=str(path.relative_to(run_dir.parent)),
                                 bytes=path.stat().st_size, sha256=digest.hexdigest(),
                                 journal_rows=count, epoch=epoch, rank=rank))
    ranks = sorted({rank for _, rank in seen})
    require(ranks == list(range(16)), 'Expected ranks 0..15: ' + run_dir.name)
    require(seen == {(epoch, rank) for epoch in range(1, 13) for rank in ranks}, 'Incomplete journal grid')
    require(boundary_violations == 0, 'Saved journal boundary violations')
    summary = dict(run_id=run_dir.name, journal_files=len(seen), rows=rows,
                   inferred_world_size=len(ranks), rows_by_epoch=dict(sorted(epochs.items())),
                   min_actual_elapsed_time=time_min, max_actual_elapsed_time=time_max,
                   geometry_counts=dict(geometry), input_count_counts=dict(sorted(inputs.items())),
                   status_counts=dict(statuses), saved_boundary_violations=boundary_violations,
                   checkpoint_committed_journal_membership='unverified: checkpoint files not supplied')
    bin_rows = [dict(run_id=run_dir.name, bin_index=i, lower_seconds=a, upper_seconds=b,
                     samples=bins[i], fraction=bins[i] / rows,
                     boundaries='left inclusive, right exclusive; 90 included in final bin')
                for i, (a, b) in enumerate(edges)]
    return summary, bin_rows, file_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--journal-workers', type=int, default=3)
    parser.add_argument('--bootstrap-draws', type=int, default=5000)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    require(source.is_dir(), 'Source directory missing')
    require(not output.exists(), 'Destination already exists; use a new directory')
    require(source != output and source not in output.parents and output not in source.parents,
            'Source and output must be separate directory trees')
    require(args.journal_workers > 0 and args.bootstrap_draws > 0, 'Worker/draw count must be positive')
    formal = sorted(source.glob('formal__*'))
    expected = {f'formal__{family}__seed{seed}' for family in FAMILIES for seed in (42, 43, 44)}
    require({p.name for p in formal} == expected, 'Expected nine formal runs')
    output.mkdir(parents=True)
    run_rows, audit_rows, curves, grouped, event_cells, pilot_rows = [], [], [], [], [], []
    pair_frames, cohorts_reference, population_reference = {}, None, None
    interface_reference, truth_reference = {}, None
    for run in formal:
        cfg = read_json(run / 'resolved_config.json')
        lock = read_json(run / 'protocol.lock.json')
        audit = source / 'audits' / run.name
        require(lock == read_json(audit / 'protocol.lock.json'), 'Training/audit lock differs')
        require(lock['status'] == 'AUDIT_PASS', 'Audit did not pass')
        template = load(REPO / 'configs' / 'fe01' / 'formal' / (run.name + '.json'), expand=False)
        original_cfg = expand(template, roots(cfg))
        require(fingerprint(original_cfg) == lock['config_sha256'], 'Source config lock mismatch')
        original_cfg.update(lock['effective_config'])
        require(original_cfg == cfg, 'Resolved config does not match audit-effective config')
        cohort = cfg['audited_cohorts']
        if cohorts_reference is None:
            cohorts_reference = cohort
        require(cohort == cohorts_reference, 'Cohort differs across models or seeds')
        interface = read_json(run / 'model_interface_audit.json')
        require(interface == read_json(audit / 'model_interface_audit.json'), 'Model interface snapshot differs')
        interface_reference.setdefault(cfg['seed'], interface['common_initial_state_sha256'])
        require(interface_reference[cfg['seed']] == interface['common_initial_state_sha256'],
                'Common downstream initialization differs within seed')
        causal = read_json(audit / 'causal_boundary_audit.json')
        query = read_json(audit / 'query_independence_audit.json')
        sampler = read_json(audit / 'sampler_distribution_audit.json')
        require(causal['status'] == query['status'] == 'PASS' and sampler['passed'], 'Audit evidence failure')
        curve = pd.read_csv(run / 'training_curves.csv')
        require(curve.epoch.tolist() == list(range(1, 13)), 'Incomplete epoch curve')
        require(curve.updates.tolist() == [212 * e for e in range(1, 13)], 'Update budget differs')
        require((curve.actual_samples == 27136).all() and (curve.global_batch == 128).all(), 'Sample budget differs')
        for record in curve.to_dict('records'):
            candidates = list(run.glob(f"epoch{int(record['epoch']):04d}_*_rank0.json"))
            require(len(candidates) == 1, 'Ambiguous epoch metadata')
            saved = read_json(candidates[0])
            for key, value in record.items():
                require(np.isclose(value, saved[key], rtol=1e-12, atol=1e-12), 'Curve/epoch metadata differs')
        best = curve.sort_values(['validation_selection', 'epoch']).iloc[0]
        epoch = int(best.epoch)
        path = run / f'validation_epoch{epoch}.csv.gz'
        frame = pd.read_csv(path, usecols=READ_COLUMNS, dtype={'event_id': str, 'station_id': str})
        require(frame['split'].eq('val').all() and frame['units'].eq('log10(m/s^2)').all(), 'Validation contract mismatch')
        require(frame.seed.eq(cfg['seed']).all() and frame.model_family.eq(cfg['model_family']).all(), 'Prediction identity differs')
        require(frame.status.eq('supported').all(), 'Validation failures must not be silently omitted')
        require(np.isfinite(frame[['prediction', 'truth', 'nll', 'crps', 'brier']]).all().all(), 'Nonfinite validation score')
        require(not frame.duplicated(PAIR_KEYS).any(), 'Duplicate query population')
        require(set(frame.elapsed_time) == set(cfg['realtime']['common_times']), 'Unexpected validation times')
        require(frame.cutout_exclusive.eq(frame.current_sample + 1).all(), 'Invalid validation exclusive cutoff')
        require(frame.history_end_sample.le(frame.current_sample).all(), 'Validation uses future history')
        require(frame.latest_received_sample.le(frame.current_sample).all(), 'Validation uses future received sample')
        pop = fingerprint(frame[POPULATION_KEYS].sort_values(POPULATION_KEYS).to_dict('records'))
        require(pop == lock['validation_population_sha256'], 'Validation population differs from audit lock')
        population_reference = population_reference or pop
        require(pop == population_reference, 'Validation population differs across runs')
        identity = PAIR_KEYS + ['target_role', 'input_ids', 'truth', 'requested_decision_sample',
                                'current_sample', 'cutout_exclusive', 'history_start_sample',
                                'history_end_sample', 'latest_received_sample', 'input_count']
        truth_hash = fingerprint(frame[identity].sort_values(PAIR_KEYS).to_dict('records'))
        truth_reference = truth_reference or truth_hash
        require(truth_hash == truth_reference, 'Truth, input selection or causal decision boundaries differ')
        selected = checkpoint_selection(frame, cfg['realtime']['common_times'])
        require(np.isclose(selected, best.validation_selection, rtol=0, atol=1e-12), 'Selection score differs from curve')
        probe = frame.iloc[np.linspace(0, len(frame) - 1, 32, dtype=int)]
        arrays = [np.array([json.loads(v) for v in probe[name]])
                  for name in ('mdn_weights', 'mdn_mu', 'mdn_sigma')]
        scored = score_rows(probe.truth.to_numpy(), *arrays, threshold=cfg['selection']['threshold_log10_pga'])
        for name in METRIC_COLUMNS:
            if name in scored:
                require(np.allclose(probe[name], scored[name], atol=1e-7, rtol=1e-7), 'Mixture score check differs: ' + name)
        noninput = frame.loc[frame.target_role.ne('observed_input')].copy()
        noninput['abs_error'] = (noninput.prediction - noninput.truth).abs()
        noninput['squared_error'] = (noninput.prediction - noninput.truth) ** 2
        noninput['signed_error'] = noninput.prediction - noninput.truth
        cells = noninput.groupby(['dataset_id', 'event_id', 'geometry_protocol', 'elapsed_time']).agg(
            targets=('abs_error', 'size'), sum_abs_error=('abs_error', 'sum'),
            sum_squared_error=('squared_error', 'sum'), sum_error=('signed_error', 'sum'),
            sum_nll=('nll', 'sum'), sum_crps=('crps', 'sum'), sum_brier=('brier', 'sum'),
            count_covered95=('covered95', 'sum'), count_covered68=('covered68', 'sum'),
            sum_width95=('width95', 'sum')).reset_index()
        cells.insert(0, 'run_id', run.name)
        event_cells.append(cells)
        pair_frames[(cfg['model_family'], cfg['seed'])] = noninput[PAIR_KEYS + ['truth', 'prediction', 'status']]
        group = grouped_metrics(frame)
        group.insert(0, 'run_id', run.name)
        grouped.append(group)
        cellmetrics = group.loc[group.target_group.eq('noninput')]
        runtime = read_json(run / 'runtime.json')
        metrics = summarize(noninput)
        row = dict(run_id=run.name, model_family=cfg['model_family'], seed=cfg['seed'],
                   selected_epoch=epoch, validation_equal_time_protocol_noninput_mae=selected,
                   final_epoch_selection=float(curve.iloc[-1].validation_selection),
                   validation_events=int(frame.event_id.nunique()), validation_rows=len(frame),
                   noninput_rows=len(noninput), epochs_recorded=len(curve), final_updates=int(curve.iloc[-1].updates),
                   epoch_seconds_sum=float(curve.elapsed_seconds.sum()),
                   trainable_parameters=interface['trainable_parameters'], encoder_parameters=interface['encoder_parameters'],
                   native_n_samples=cfg['native_n_samples'], job_id=runtime.get('job_id'),
                   training_git_commit=runtime.get('git_commit'), training_git_dirty=runtime.get('git_dirty'),
                   selected_source=str(path.relative_to(source)), selected_source_sha256=sha256(path),
                   pooled_noninput_mae=metrics['mae'], pooled_noninput_rmse=metrics['rmse'],
                   pooled_noninput_bias=metrics['bias'])
        for metric in ('nll', 'crps', 'brier', 'covered95', 'covered68', 'width95'):
            row['equal_time_protocol_noninput_' + metric] = float(cellmetrics[metric].mean())
        run_rows.append(row)
        curve.insert(0, 'run_id', run.name)
        curves.append(curve)
        audit_rows.append(dict(run_id=run.name, audit_status=lock['status'],
                               code_sha256=lock['code_sha256'], config_sha256=lock['config_sha256'],
                               validation_population_sha256=pop, truth_and_decision_sha256=truth_hash,
                               common_initial_state_sha256=interface['common_initial_state_sha256'],
                               common_structure_sha256=interface['common_structure_sha256'],
                               pretrained_manifest_sha256=lock['pretrained_manifest_sha256'],
                               causal_audit_status=causal['status'], query_audit_status=query['status'],
                               sampler_audit_passed=sampler['passed'],
                               train_events=len(cohort['train']), val_events=len(cohort['val'])))
        compact_cfg = dict(cfg)
        compact_cfg['audited_cohorts'] = dict(source='shared_cohorts.csv', fingerprint=fingerprint(cohort),
                                             counts={k: len(v) for k, v in cohort.items()})
        compact_lock = {k: v for k, v in lock.items() if k != 'effective_config'}
        compact_lock['effective_config'] = {k: v for k, v in lock['effective_config'].items() if k != 'audited_cohorts'}
        write_json(output / 'metadata' / (run.name + '.json'), portable(dict(
            note='Portable derived snapshot; original bytes/hashes indexed in source_inventory.csv',
            resolved_config=compact_cfg, protocol_lock=compact_lock, runtime=runtime,
            model_interface_audit=interface, causal_boundary_audit=causal,
            query_independence_audit=query, sampler_distribution_audit=sampler,
            capability_manifest=read_json(audit / 'capability_manifest.json'),
            cohort_normalization_audit={k: v for k, v in read_json(
                audit / 'cohort_normalization_audit.json').items() if k != 'cohorts'}), roots(cfg)))
        print(f'VALIDATED {run.name}: epoch={epoch} score={selected:.9f} rows={len(frame)}', flush=True)

    require(len({r['common_structure_sha256'] for r in audit_rows}) == 1, 'Common structure differs')
    require(len({r['code_sha256'] for r in audit_rows}) == 1, 'Audited code differs')
    source_files, source_hash = audited_source_identity()
    require(source_hash == audit_rows[0]['code_sha256'], 'Reader implementation/source differs from HPC audit')
    write_json(output / 'audited_source_files.json', dict(code_sha256=source_hash, files_sha256=source_files))
    for pilot in sorted(source.glob('pilot__*')):
        cfg = read_json(pilot / 'resolved_config.json')
        curve = pd.read_csv(pilot / 'training_curves.csv')
        pilot_rows.append(dict(run_id=pilot.name, model_family=cfg['model_family'], seed=cfg['seed'],
                               epochs_recorded=len(curve), updates=int(curve.iloc[-1].updates),
                               samples_recorded=int(curve.actual_samples.sum()),
                               final_validation_selection=float(curve.iloc[-1].validation_selection),
                               note='Separate smoke run; excluded from formal comparison'))
    run_table = pd.DataFrame(run_rows)
    run_table.to_csv(output / 'run_summary.csv', index=False)
    pd.DataFrame(audit_rows).to_csv(output / 'audit_summary.csv', index=False)
    pd.concat(curves, ignore_index=True).to_csv(output / 'training_curves.csv', index=False)
    pd.concat(grouped, ignore_index=True).to_csv(output / 'validation_by_time_geometry_role.csv', index=False)
    pd.concat(event_cells, ignore_index=True).to_csv(output / 'validation_noninput_event_cells.csv', index=False)
    pd.DataFrame(pilot_rows).to_csv(output / 'pilot_summary.csv', index=False)
    pd.DataFrame([dict(split=split, dataset_id=dataset, event_id=event)
                  for split, entries in cohorts_reference.items() for dataset, event in entries]).to_csv(
                      output / 'shared_cohorts.csv', index=False)
    summary_rows = []
    endpoints = ['validation_equal_time_protocol_noninput_mae', 'equal_time_protocol_noninput_nll',
                 'equal_time_protocol_noninput_crps', 'equal_time_protocol_noninput_brier',
                 'equal_time_protocol_noninput_covered95', 'equal_time_protocol_noninput_covered68',
                 'equal_time_protocol_noninput_width95', 'epoch_seconds_sum']
    for family, group in run_table.groupby('model_family'):
        row = dict(model_family=family, seeds=len(group), selected_epochs=','.join(map(str, group.selected_epoch)))
        for endpoint in endpoints:
            row[endpoint + '_mean'] = float(group[endpoint].mean())
            row[endpoint + '_sample_std'] = float(group[endpoint].std(ddof=1))
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output / 'seed_summary.csv', index=False)
    paired = []
    for left, right in ((FAMILIES[0], FAMILIES[1]), (FAMILIES[0], FAMILIES[2]), (FAMILIES[2], FAMILIES[1])):
        for seed in (42, 43, 44):
            result = paired_bootstrap(pair_frames[(left, seed)], pair_frames[(right, seed)],
                                      draws=args.bootstrap_draws, seed=20261001)
            expected_delta = (run_table.loc[(run_table.model_family == right) & (run_table.seed == seed),
                                            'validation_equal_time_protocol_noninput_mae'].iloc[0] -
                              run_table.loc[(run_table.model_family == left) & (run_table.seed == seed),
                                            'validation_equal_time_protocol_noninput_mae'].iloc[0])
            require(np.isclose(result['equal_time_protocol_delta_mae'], expected_delta, atol=1e-12),
                    'Paired endpoint differs from primary endpoint')
            paired.append(dict(left_model=left, right_model=right, model_seed=seed,
                               interpretation='right minus left; negative favors right; validation only', **result))
            print(f'BOOTSTRAP {left} -> {right} seed={seed}', flush=True)
    pd.DataFrame(paired).to_csv(output / 'validation_paired_event_bootstrap.csv', index=False)
    journal_rows, exposure_rows, known_hashes = [], [], {}
    with ProcessPoolExecutor(max_workers=args.journal_workers) as pool:
        pending = {pool.submit(journal_summary, run): run.name for run in formal}
        for future in as_completed(pending):
            result, exposure, records = future.result()
            curve = next(c for c in curves if c.run_id.iloc[0] == result['run_id'])
            require(result['rows_by_epoch'] == {int(r.epoch): int(r.actual_samples)
                                               for r in curve.itertuples()}, 'Journal/curve sample counts differ')
            journal_rows.append(result)
            exposure_rows.extend(exposure)
            known_hashes.update({r['relative_path']: r['sha256'] for r in records})
            print(f"JOURNALS {result['run_id']}: rows={result['rows']} time={result['min_actual_elapsed_time']}..{result['max_actual_elapsed_time']}", flush=True)
    write_json(output / 'journal_summary.json', sorted(journal_rows, key=lambda r: r['run_id']))
    pd.DataFrame(exposure_rows).sort_values(['run_id', 'bin_index']).to_csv(output / 'training_time_exposure.csv', index=False)
    files = sorted(p for p in source.rglob('*') if p.is_file())
    def inventory_item(path):
        relative = str(path.relative_to(source))
        cache = '.cache' in path.relative_to(source).parts
        digest = None if cache else known_hashes.get(relative) or sha256(path)
        return dict(relative_path=relative, bytes=path.stat().st_size, sha256=digest,
                    hash_status='not hashed: derived cache' if cache else 'SHA256_READ',
                    category=('sample_journal' if 'sample_journals' in path.parts else
                              'validation_prediction' if path.name.startswith('validation_epoch') else
                              'cache' if cache else 'metadata_or_audit'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        inventory = list(pool.map(inventory_item, files))
    pd.DataFrame(inventory).to_csv(output / 'source_inventory.csv', index=False)
    evidence = dict(source_label=source.name, source_files=len(files), source_bytes=sum(p.stat().st_size for p in files),
                    formal_runs=len(formal), pilot_runs=len(pilot_rows),
                    selected_validation_rows=sum(r['validation_rows'] for r in run_rows),
                    formal_journal_rows=sum(r['rows'] for r in journal_rows),
                    training_git_commits=sorted(set(run_table.training_git_commit)),
                    common_validation_population_sha256=population_reference,
                    common_truth_input_and_decision_sha256=truth_reference,
                    audited_source_sha256=source_hash, audited_source_files=len(source_files),
                    bootstrap_draws=args.bootstrap_draws, bootstrap_rng_seed=20261001,
                    reader_versions=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__),
                    checks=dict(source_configuration_matches_lock=True, training_and_audit_lock_identical=True,
                                shared_cohort_identical=True, shared_validation_population_identical=True,
                                shared_labels_inputs_and_decisions_identical=True, common_initialization_matches_within_seed=True,
                                all_nine_selection_scores_recomputed=True, mixture_score_32_rows_per_run_verified=True,
                                all_formal_journal_counts_match_curves=True, saved_journal_boundaries_verified=True,
                                reader_audited_source_matches_hpc_code_hash=True),
                    missing=dict(checkpoint_files=not any(source.rglob('*.pth')),
                                 final_test_evaluation=True, rt55_paired_reevaluation=True,
                                 random_validation_time_predictions=True,
                                 spatial_evaluation=True, formal_slurm_completion_records=True),
                    limitations=['Validation checkpoint selection and validation reporting share the same data.',
                                 'Event bootstrap is descriptive within a selected model/seed and does not include checkpoint-selection uncertainty.',
                                 'Training-time support is model-specific; common fixed-time validation is 1/3/5/10/20s.',
                                 'Journal commit membership and best.pth identity cannot be verified without checkpoints.',
                                 'AUDIT_PASS is saved audit evidence; raw HDF5 and pretrained weights were not reloaded.',
                                 'Epoch timers are recorded rank0 durations, not independently measured whole-job wall time.',
                                 'Offline resampling/filtering causality remains uncertified.'])
    write_json(output / 'evidence_checks.json', evidence)
    print('SUMMARY_COMPLETE ' + str(output), flush=True)


if __name__ == '__main__':
    main()
