#!/usr/bin/env python3
"""Read-only V01 artifact verification and lightweight Git review packaging.

Never executes inference, opens waveform caches, or rewrites returned artifacts.
Run from the repository with --closure-root, --old-eval-root and --output.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.analyze_v01_validation_closure import (
    KEYS, STRATA, cluster_statistics, load_rows, pairing, read_records, subset,
)
from tools.audit_v01_validation_contract import npz_arrays
from tools.summarize_v01_results import point_metrics
from tools.v01_validation_contract import CHECKPOINTS, sha256


def read_json(path):
    return json.loads(path.read_text())


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def close(actual, expected, label):
    if not np.isclose(actual, expected, rtol=1e-9, atol=1e-9, equal_nan=True):
        raise AssertionError((label, actual, expected))


def verify_and_package(root, old, output):
    if root == output or root in output.parents or output in root.parents:
        raise ValueError('Output must be separate from original artifacts')
    if (output/'verification.json').exists():
        raise FileExistsError('Completed review exists; choose a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    plan = read_json(root/'submission_plan.json')
    assert sha256(root/'source_manifest.json') == plan['source_manifest_sha256']
    manifest = read_json(root/'source_manifest.json')['files']
    assert all(sha256(REPO/name) == expected for name, expected in manifest.items())
    for cell, expected in plan['effective_config_sha256'].items():
        assert sha256(root/'configs'/(cell+'.json')) == expected
    before = read_json(root/'audit/checkpoints_before.json')
    assert before == read_json(root/'report/checkpoints_after.json')
    for item in before:
        assert item['sha256'] == CHECKPOINTS[item['arm']][1]
        assert item['epoch'] == 8 and item['step_min'] == item['step_max'] == 1496
    source_evidence = read_json(root/'audit/source_provenance.json')
    for record in source_evidence:
        if 'sha256' in record:
            assert sha256(root/'source_evidence'/record['file']) == record['sha256']
    frozen_path = root/'source_evidence/frozen_split_events.csv'
    lock = read_json(root/'source_evidence/derived_cache/protocol_lock.json')
    assert sha256(frozen_path) == lock['split_manifest_sha256']
    frozen = pd.read_csv(frozen_path, dtype={'EVENT': str})
    dev = set(frozen.loc[frozen.split == 'dev', 'EVENT'])
    train = set(frozen.loc[frozen.split == 'train', 'EVENT'])
    assert not (dev & train)
    frozen.loc[frozen.split.isin(['train', 'dev']), ['dataset_index', 'EVENT', 'split']].to_csv(
        output/'frozen_train_dev_membership.csv', index=False)
    gate = read_json(root/'audit/gate_passed.json')
    assert all(gate[key] for key in ('real_idx7_confirmed', 'random_numeric_signatures_equal',
                                    'checkpoints_unchanged'))
    for path in (root/'audit').glob('*.idx7_trace.json'):
        trace = read_json(path)
        assert trace['confirmed'] and 'Found event without PGA' in trace['before_error']
        assert trace['before_clock'][0]['changed_slots'] > 0
        assert trace['after_clock'][0]['changed_slots'] == 0
        assert trace['before_clock'][0]['clock'] == trace['after_clock'][0]['clock']
        assert trace['after']['query_count'] > 0

    report = root/'report'
    supplied_metrics = pd.read_csv(report/'metrics_by_stratum.csv')
    supplied_ci = pd.read_csv(report/'paired_event_cluster_ci_and_mse_decomposition.csv')
    provenance = read_json(report/'input_provenance.json')
    coverage = {row['cell']: row for row in read_json(report/'coverage_and_denominators.json')}
    match_report = {row['comparison']: row for row in read_json(report/'one_to_one_outer_match_audit.json')}
    frames, cell_checks, abstentions, inventory = {}, [], [], []
    metric_checks = 0
    for record in provenance:
        cell = record['cell']
        new = cell in plan['cells']
        path = (root/'eval' if new else old)/(cell+'.npz')
        ledger = Path(str(path)+'.val.requests.jsonl') if new else root/'audit'/(cell+'.plan.jsonl')
        assert sha256(path) == record['sha256']
        assert sha256(ledger) == record['ledger_sha256']
        records = read_records(ledger)
        counts = Counter(r['outcome'] for r in records)
        assert len(records) == 8358
        assert len({r['global_index'] for r in records}) == len(records)
        assert len({(r['event_id'], r['time_s']) for r in records}) == len(records)
        assert {r['event_id'] for r in records} <= dev
        assert not ({r['event_id'] for r in records} & train)
        assert not counts['implementation_error'] and not counts['invalid_label_or_metadata']
        if new:
            summary = read_json(Path(str(ledger)+'.summary.json'))
            assert summary['complete'] and summary['requested'] == len(records)
            assert dict(counts) == {k: v for k, v in summary['outcomes'].items() if v}
            config = read_json(root/'configs'/(cell+'.json'))
            assert config['training_params']['v01_validation_closure'] is True
            formal = read_json(root/'eval'/(cell+'.metrics.json'))
            assert formal['splits'] == ['val'] and formal['checkpoint_metadata']['epoch'] == 8
        frame = load_rows(path, records)
        frames[cell] = frame
        assert len(frame) == coverage[cell]['selected_predicted_targets']
        assert counts['predicted'] == coverage[cell]['predicted_event_times']
        arrays = npz_arrays(path)
        valid = arrays['pga_target_valid'].astype(bool)
        station = arrays['station_valid'].astype(bool)
        role = arrays['v01_source_role'].astype(bool)
        assert not np.any(station & ~role)
        assert not pd.DataFrame({'event': arrays['event_id'],
                                 'time': arrays['realtime_requested_elapsed_time']}).duplicated().any()
        types, type_counts = np.unique(arrays['realtime_target_type'][valid], return_counts=True)
        row = {'cell': cell, 'requested': len(records), 'outcomes': dict(counts),
               'predicted_targets': len(frame), 'predicted_events': int(frame.event_id.nunique()),
               'unique_event_time_rows': len(arrays['event_id']),
               'target_type_counts': {str(int(k)): int(v) for k, v in zip(types, type_counts)},
               'source_observations': int(station.sum()), 'query_only_rows_in_input': 0,
               'npz_sha256': record['sha256'], 'ledger_sha256': record['ledger_sha256']}
        cell_checks.append(row)
        for r in records:
            if r['outcome'] != 'predicted':
                abstentions.append({'cell': cell, **r})
        for window in ('all7', 'early135', 't1', 't3', 't5'):
            for stratum in STRATA:
                selected = frame[subset(frame, stratum, window)]
                actual = point_metrics(selected)
                reference = supplied_metrics[(supplied_metrics.cell == cell)
                    & (supplied_metrics.time_window == window) & (supplied_metrics.stratum == stratum)].iloc[0]
                for key, value in actual.items():
                    if value is not None:
                        close(value, reference[key], (cell, window, stratum, key))
                        metric_checks += 1
                if new and window == 'all7' and stratum == 'all':
                    for key in ('mae', 'rmse', 'r2', 'slope', 'bias', 'nll', 'brier',
                                'coverage_1sigma', 'coverage_2sigma', 'targets', 'events'):
                        close(actual[key], formal['metrics']['val']['pga'][key], (cell, 'formal', key))
        del records, arrays
        print('Verified cell:', cell, len(frame), 'targets', flush=True)

    ci_checks, sufficient = [], []
    paired_cache = {}
    for name, original in match_report.items():
        left, right = name.split('--')
        matched, computed, _ = pairing(frames[left], frames[right])
        assert all(computed[key] == original[key] for key in computed)
        paired_cache[name] = matched
    for geometry in ('normal', 'random'):
        name = 'vfull__vfull__'+geometry+'--vmissing__vmissing__'+geometry
        matched = paired_cache[name]
        for window in ('all7', 'early135'):
            for stratum in STRATA:
                eligible = subset(matched, stratum, window, 'left') & subset(matched, stratum, window, 'right')
                computed, stats = cluster_statistics(matched[eligible], draws=5000, seed=20260915)
                tags = {'comparison': name, 'time_window': window, 'stratum': stratum}
                expected = supplied_ci[(supplied_ci.comparison == name)
                    & (supplied_ci.time_window == window) & (supplied_ci.stratum == stratum)].set_index('statistic')
                for row in computed:
                    for key in ('estimate', 'ci_low', 'ci_high', 'paired_targets', 'paired_events'):
                        close(row[key], expected.loc[row['statistic'], key], (tags, row['statistic'], key))
                    ci_checks.append({**tags, **row})
                sufficient.append(stats.assign(**tags))
                print('Recomputed 5000-draw CI:', geometry, window, stratum, flush=True)
    pd.concat(sufficient, ignore_index=True).to_csv(output/'primary_ff_mm_event_sufficient_statistics.csv', index=False)
    pd.DataFrame(ci_checks).to_csv(output/'primary_ff_mm_recomputed_ci.csv', index=False)
    # Small tables for a reviewer; the full tables are preserved below as well.
    supplied_metrics[supplied_metrics.stratum == 'all'].to_csv(output/'headline_metrics.csv', index=False)
    supplied_ci[supplied_ci.statistic.isin(['micro_mae', 'macro_mae', 'micro_rmse',
        'micro_nll', 'micro_brier'])].to_csv(output/'paired_effects.csv', index=False)

    # Verify real figure source counts without regenerating or modifying plots.
    density = pd.read_csv(report/'density_bin_counts.csv')
    residual = pd.read_csv(report/'residual_bin_counts.csv')
    for cell, frame in frames.items():
        if cell not in set(density.cell):
            continue
        edges = np.linspace(-3, 1, 121)
        hist, _, _ = np.histogram2d(frame.truth, frame.prediction, bins=(edges, edges))
        np.testing.assert_array_equal(hist.ravel(), density[density.cell == cell]['count'].to_numpy())
        for group in ('above', 'below'):
            selected = frame[(frame.truth >= -1.2) == (group == 'above')]
            hist, _ = np.histogram(selected.prediction-selected.truth, bins=np.linspace(-2, 2, 161))
            np.testing.assert_array_equal(hist, residual[(residual.cell == cell)
                & (residual.stratum == group)]['count'].to_numpy())
    for record in read_json(report/'figure_count_audit.json'):
        assert record['targets'] == record['binned_targets'] + record['outside_fixed_axes']

    # An explicit whitelist keeps large/raw artifacts and test metadata out of Git.
    copies = [root/'source_manifest.json', root/'submission_plan.json', root/'submitted_jobs.tsv']
    copies += list((root/'audit').glob('*.json'))
    copies += list((root/'configs').glob('*.json'))
    copies += list((root/'source_evidence/derived_cache').glob('*.json'))
    copies += list((root/'source_evidence/derived_cache').glob('*.csv'))
    for directory in ('weights_vfull_retry1', 'weights_vmissing', 'weights_apair'):
        copies += list((root/'source_evidence'/directory).glob('*.json'))
    copies += list((root/'source_evidence').glob('*.config.json'))
    copies += [p for p in report.glob('*.csv') if p.name != 'paired_event_sufficient_statistics.csv']
    copies += list(report.glob('*.json')) + list(report.glob('*.png')) + list(report.glob('*.md'))
    copies += list((root/'eval').glob('*.metrics.json'))
    copies += list((root/'eval').glob('*.summary.json'))
    for path in copies:
        target = output/'evidence'/path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        inventory.append({'original_relative_path': str(path.relative_to(root)),
                          'packaged_path': str(target.relative_to(output)),
                          'bytes': path.stat().st_size, 'sha256': sha256(path)})
        assert sha256(target) == inventory[-1]['sha256']
    save_json(output/'artifact_copy_manifest.json', inventory)
    save_json(output/'no_prediction_requests.json', abstentions)
    save_json(output/'verification.json', {
        'analysis_base_commit': '7d87c4007b78405bb901c67ff0b78dc5d0e85a8e',
        'protocol': plan['protocol'], 'source_manifest_sha256': plan['source_manifest_sha256'],
        'runtime_files_matched': len(manifest), 'config_hashes_matched': len(plan['cells']),
        'checkpoint_before_after_equal': True,
        'checkpoint_verification_scope': 'returned HPC audit; checkpoint body not re-read locally',
        'frozen_manifest_hash_matched': True, 'train_dev_disjoint': True,
        'all_request_events_in_frozen_dev': True, 'metric_values_checked': metric_checks,
        'recomputed_primary_ci_rows': len(ci_checks), 'primary_bootstrap_draws': 5000,
        'primary_bootstrap_seed': 20260915, 'outer_joins_rechecked': len(match_report),
        'figure_bin_counts_recomputed': True, 'cells': cell_checks,
        'sacct_available': False, 'hpc_jobs_from_submitted_tsv': [29318960, 29318961,
            29318962, 29318963, 29318964, 29318965, 29318966, 29318967],
        'no_inference_training_or_test': True,
        'excluded_from_git': ['raw NPZ', 'full ledgers/plans', 'waveform caches', 'checkpoints',
                              '118MB full sufficient statistics', 'full frozen split including test rows'],
    })
    print('Verification and packaging complete:', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--closure-root', type=Path, required=True)
    parser.add_argument('--old-eval-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify_and_package(args.closure_root.resolve(), args.old_eval_root.resolve(), args.output.resolve())
