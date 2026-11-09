"""Read-only review of the downloaded FE01 V4 smoke/pilot evidence.

Run from this repository with --input-dir (extracted archive), --archive,
and a new --output-dir. No model, scheduler, production data, or test split is
accessed. Full input CSVs and private path-bearing plans stay outside Git.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.special import ndtr, softmax

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fe01_a01_compare.contracts import EPOCHS, contract, source_identity
from fe01_a01_compare.io import fingerprint, read_json, require, sha256, verify_seal

ARCHIVE_SHA = '40764a1cb0cdb34e2d5e080c621da78a7d223b9c5a6512288f976e1b62307395'
MAIN = '50672e131d876a604432b283162d5b0189d40cf0'
REVISION = 'c66db619e99a60751ba02d24931d4570ead464a3'
EXECUTION = 'b9dd0368adc7041bb0c1e36a6899d49014ac9545ae7f44850274099bab6dc15c'
EVENT = ['dataset_id', 'event_id']
FRAME = EVENT + ['geometry_protocol', 'elapsed_time']
STATION = FRAME + ['station_id']
GRID = FRAME + ['grid_point_id']


def csv(path):
    # Keep the original C round-trip parser even on pandas versions deprecating
    # the explicit keyword. Only that deprecation warning is suppressed here.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message="The 'float_precision' argument is deprecated.*")
        return pd.read_csv(path, engine='c', float_precision='round_trip',
                           dtype={'dataset_id': str, 'event_id': str, 'station_id': str})


def same_frame(left, right, columns, keys):
    a = left.sort_values(keys)[columns].reset_index(drop=True)
    b = right.sort_values(keys)[columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=True)


def mdn_audit(frame):
    require(len(frame) > 0, 'Empty supported prediction population')
    numeric = ['prediction', 'predictive_sigma', 'q025', 'q975', 'width95',
               'probability_ge_minus1p2', 'probability_ge_zero']
    require(np.isfinite(frame[numeric].to_numpy(float)).all(), 'Nonfinite supported prediction')
    arrays = {k: np.asarray([json.loads(v) for v in frame[k]], float)
              for k in ('mdn_logits', 'mdn_weights', 'mdn_mu', 'mdn_sigma')}
    logits, weights, mu, sigma = (arrays[k] for k in arrays)
    require(logits.ndim == 2 and logits.shape[1] > 0, 'Invalid MDN component shape')
    require(all(a.shape == logits.shape and np.isfinite(a).all() for a in arrays.values()), 'Invalid full MDN')
    require((sigma > 0).all() and (weights >= 0).all(), 'Invalid MDN scales/weights')
    require(np.allclose(weights, softmax(logits, axis=-1), rtol=0, atol=1e-10), 'Weights/logits differ')
    require(np.allclose(weights.sum(-1), 1., rtol=0, atol=1e-10), 'Unnormalized mixture')
    require(np.allclose(frame.prediction, (weights * mu).sum(-1), rtol=0, atol=1e-9), 'Mixture mean differs')
    require(np.allclose(frame.width95, frame.q975 - frame.q025, rtol=0, atol=1e-9), 'Interval width differs')
    require((frame.q975 >= frame.q025).all() and (frame.predictive_sigma > 0).all(), 'Invalid interval/sigma')
    for threshold, name in ((-1.2, 'probability_ge_minus1p2'), (0., 'probability_ge_zero')):
        expected = (weights * ndtr((mu - threshold) / sigma)).sum(-1)
        require(np.allclose(frame[name], expected, rtol=0, atol=1e-9), 'Mixture tail probability differs: ' + name)
    return dict(rows=len(frame), components=logits.shape[1], finite=True,
                mixture_mean_weights_width_and_probabilities='PASS')


def review(raw, archive, output):
    raw, archive, output = Path(raw).resolve(), Path(archive).resolve(), Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(raw), 'Use a new output outside input evidence')
    require(sha256(archive) == ARCHIVE_SHA, 'Wrong archive byte identity')
    require(Path(str(archive) + '.sha256').read_text().split()[0] == ARCHIVE_SHA, 'Checksum sidecar differs')
    plan = read_json(raw / 'plan.json')
    plan_sha = sha256(raw / 'plan.json')
    require(plan_sha == read_json(raw / 'plan_pin.json')['sha256'], 'Plan byte pin differs')
    source = read_json(raw / 'source_identity.json')
    require(source == plan['source_identity'], 'Source identity differs from plan')
    require(source['result_commit'] == MAIN and source['execution_sha256'] == EXECUTION, 'Wrong frozen release')
    require(source['files_sha256'] == source_identity()['files_sha256'], 'Current execution files differ')
    require(read_json(raw / 'analysis_contract.json') == contract(), 'Scientific contract differs')
    require(plan['contract_sha256'] == fingerprint(contract()), 'Contract pin differs')
    require(len(plan['cohort']) == 1310 and plan['block_events'] == 32, 'Wrong full evaluation population/block')
    resources, seals, checkpoint_rows, support_rows, mdn_rows = [], [], [], [], []
    covariate_parents = None
    for stage in ('model_smoke', 'replay_pilot'):
        needed = ('inventory', 'covariates') if stage == 'model_smoke' else ('model_smoke', 'covariates')
        expected_parents = {dep + '/' + str(t['index']) for dep in needed for t in plan['tasks'][dep]}
        for i, (family, epoch) in enumerate(EPOCHS.items()):
            directory = raw / 'stages' / stage / str(i)
            seal = verify_seal(directory, dict(execution_sha256=EXECUTION, contract_sha256=plan['contract_sha256'],
                                              plan_sha256=plan_sha, stage=stage, task=i))
            parents = seal['dependency_seals_sha256']
            require(set(parents) == expected_parents, 'Incomplete parent seal identity')
            cov = {k: v for k, v in parents.items() if k.startswith('covariates/')}
            if covariate_parents is None:
                covariate_parents = cov
            require(cov == covariate_parents, 'Covariate parents differ across models/stages')
            available = 0
            for key, digest in parents.items():
                parent = raw / 'stages' / key / 'SEAL.json'
                if parent.is_file():
                    require(sha256(parent) == digest, 'Parent seal bytes differ: ' + key)
                    available += 1
            attempt = read_json(directory / 'ATTEMPT.json')
            require(all(attempt['identity'][k] == v for k, v in seal.items()
                        if k not in ('status', 'files_sha256')), 'Attempt identity differs')
            ci = read_json(directory / 'checkpoint_identity.json')
            require(ci['family'] == family and ci['epoch'] == epoch and ci['status'] == 'IDENTITY_PASS', 'Wrong checkpoint')
            checkpoint_rows.append(dict(stage=stage, task=i, family=family, epoch=epoch,
                                        checkpoint_sha256=ci['checkpoint_sha256']))
            seals.append(dict(stage=stage, task=i, family=family, files_verified=len(seal['files_sha256']),
                              seal_sha256=sha256(directory / 'SEAL.json'), job_id=attempt['job_id'],
                              parent_seals_recorded=len(parents), parent_seals_checked_locally=available,
                              runtime_revision_commit=seal.get('runtime_revision_commit')))
            if stage == 'model_smoke':
                checks = read_json(directory / 'smoke_checks.json')
                require(all(c['status'] in {'PASS', 'PASS_INPUT_CONTROL', 'NOT_APPLICABLE_NO_FUTURE_SAMPLES'}
                            for c in checks), 'Smoke contains non-passing/ambiguous result')
                require(sum(c['check'] == 'legacy_overlap' and c['status'] == 'PASS' for c in checks) == 20,
                        'Legacy snapshot overlap gate incomplete')

    smoke_team = raw / 'stages/model_smoke/3'
    operational = read_json(smoke_team / 'operational_execution.json')
    revision = operational['revision']
    require(revision['commit'] == REVISION and revision['original_commit'] == MAIN, 'Wrong TEAM revision')
    declared = {k: v for k, v in revision.items() if k not in ('revision_sha256', 'marker_sha256')}
    require(fingerprint(declared) == revision['revision_sha256'], 'Operational identity digest differs')
    require(read_json(smoke_team / 'SEAL.json')['runtime_revision_sha256'] == revision['revision_sha256'],
            'TEAM seal does not bind operational evidence')
    for name, digest in revision['files_sha256'].items():
        payload = subprocess.check_output(['git', '-C', str(ROOT), 'show', REVISION + ':' + name])
        require(hashlib.sha256(payload).hexdigest() == digest, 'Operational file differs from exact commit')
    audits = list((smoke_team / 'smoke_diagnostics').iterdir())
    require(len(audits) == 20, 'TEAM diagnostic request count differs')
    for d in audits:
        before = read_json(d / '00_scope.json')['backend_before']
        during = read_json(d / '01_backend.json')
        after = read_json(d / '99_pass.json')
        require(during['deterministic_algorithms_enabled'] and not during['deterministic_algorithms_warn_only'],
                'Strict deterministic audit not active')
        require(after['backend_restored'] and after['backend_after'] == before, 'Backend not restored')

    first_station = first_grid = selection = None
    for i, (family, epoch) in enumerate(EPOCHS.items()):
        d = raw / 'stages/replay_pilot' / str(i)
        ci = read_json(d / 'checkpoint_identity.json')
        smoke_ci = read_json(raw / 'stages/model_smoke' / str(i) / 'checkpoint_identity.json')
        require(ci['checkpoint_sha256'] == smoke_ci['checkpoint_sha256'], 'Pilot checkpoint changed after smoke')
        meta = read_json(d / 'metadata_selection.json')
        if selection is None:
            selection = meta
        require(meta == selection and not meta['selection_uses_predictions'], 'Pilot event selection differs')
        require(meta['cohort_events'] == 1310 and len(meta['pilot_events']) == 6, 'Wrong selected pilot cohort')
        require({('covariates/' + k): v for k, v in meta['source_identity']['covariate_seals_sha256'].items()}
                == covariate_parents, 'Selection uses different covariates')
        stores = read_json(d / 'stores.json')
        require(len(stores) == 12 and all(s['frames'] == 20 for s in stores), 'Incomplete pilot stores')
        require({(s['dataset_id'], s['event_id']) for s in stores} == {tuple(e) for e in meta['pilot_events']},
                'Pilot events differ from frozen selection')
        station = pd.concat([csv(d / s['station_file']) for s in stores], ignore_index=True)
        grid = pd.concat([csv(d / s['grid_file']) for s in stores], ignore_index=True)
        require(not station.duplicated(STATION).any() and not grid.duplicated(GRID).any(), 'Duplicated requests')
        # covariates.catalog normalizes the validation alias to the literal val.
        require(set(station['split']) == {'val'} and set(station.seed) == {42}, 'Wrong split/seed')
        for f, protocol in ((station, 'fixed_query_replay_v4'), (grid, 'grid_replay_v4')):
            supported = f[f.status.eq('supported')]
            require(set(supported.protocol) == {protocol} and set(f.family) == {family}
                    and set(f.checkpoint_epoch) == {epoch}, 'Wrong output protocol/checkpoint')
            require(set(supported.checkpoint_sha256) == {ci['checkpoint_sha256']}
                    and set(supported.units) == {'log10(m/s^2)'}
                    and set(supported.absolute_amplitude_mode) == {'physical_scale_keep_duration'}, 'Wrong ON identity/units')
            mdn_rows.append(dict(family=family, population=protocol, **mdn_audit(supported)))
        require(set(station.status) == {'supported'}, 'Station has unsupported or missing requests')
        require(set(grid.status) <= {'supported', 'unsupported_grid_metadata'}, 'Unexpected grid support failure')
        unsupported = grid[grid.status.ne('supported')]
        require(unsupported.prediction.isna().all(), 'Unsupported grid contains fabricated predictions')
        require(grid.truth.isna().all(), 'Grid contains purported ground truth')
        require(station.groupby(FRAME).ngroups == 240 and grid.groupby(FRAME).ngroups == 240, 'Missing frames')
        for frame in (station, grid):
            require(set(map(tuple, frame[EVENT].drop_duplicates().values.tolist()))
                    == {tuple(e) for e in meta['pilot_events']}, 'Exported pilot events differ')
            require(set(frame.geometry_protocol) == {'normal', 'random'}
                    and set(frame.elapsed_time) == set(range(1, 21)), 'Wrong geometry/time domain')
        require(station.groupby(EVENT + ['station_id']).truth.nunique().eq(1).all(), 'Final PGA label changes across frames')
        for frame, part in station.groupby(FRAME):
            require(part.physical_received_history_sha256.nunique() == 1, 'History differs inside frame')
        population = STATION + ['request_id', 'truth', 'latitude', 'longitude', 'elevation',
                                'input_ids', 'input_coords', 'input_count',
                                'cutout_exclusive', 'target_role', 'fixed_never_input', 'physical_received_history_sha256']
        grid_population = GRID + ['status', 'latitude', 'longitude', 'elevation', 'grid_sha256', 'grid_spacing_km']
        if first_station is None:
            first_station, first_grid = station, grid
        else:
            same_frame(first_station, station, population, STATION)
            same_frame(first_grid, grid, grid_population, GRID)
        lat = csv(d / 'latency_frames.csv')
        require(len(lat) == 240 and not lat.duplicated(FRAME).any(), 'Timing frame population differs')
        worker = read_json(d / 'worker.json')
        require(worker['family'] == family and worker['checkpoint_load_count'] == 1, 'Worker load identity differs')
        total_bytes = sum(p.stat().st_size for p in d.rglob('*') if p.is_file())
        resources.append(dict(family=family, epoch=epoch, events=6, frame_count=240, station_targets=len(station),
                              noninput_targets=int(station.target_role.ne('observed_input').sum()),
                              observed_input_targets=int(station.target_role.eq('observed_input').sum()),
                              grid_targets=len(grid), grid_supported_targets=int(grid.status.eq('supported').sum()),
                              total_worker_seconds=worker['total_worker_wall_seconds'],
                              cold_worker_seconds=worker['cold_worker_seconds'],
                              peak_allocated_bytes=worker['peak_device_allocated_bytes'],
                              peak_allocated_GiB=worker['peak_device_allocated_bytes'] / 2**30,
                              complete_stage_bytes=total_bytes,
                              numeric_map_service_p50_seconds=float(lat.numeric_map_service_seconds.median()),
                              numeric_map_service_p95_seconds=float(lat.numeric_map_service_seconds.quantile(.95)),
                              source='stages/replay_pilot/' + str(i) + '/worker.json; latency_frames.csv; full stage files'))
        for population_name, f in (('station', station), ('grid', grid)):
            for status, count in f.status.value_counts().items():
                support_rows.append(dict(family=family, population=population_name, status=status, targets=int(count)))

    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(resources).to_csv(output / 'resource_summary.csv', index=False)
    pd.DataFrame(support_rows).to_csv(output / 'support_summary.csv', index=False)
    pd.DataFrame(checkpoint_rows).to_csv(output / 'checkpoint_identity.csv', index=False)
    report = dict(
        task_id='20261109-fe01-v4-pilot-evidence-review', repository='rular099/team_pytorch',
        review_base_commit='17ba43eea495e8df6f1e6c6825f6b99f41f67f69',
        review_branch='handoff/fe01-on-20261108', frozen_main_commit=MAIN, team_smoke_revision=REVISION,
        archive=dict(name=archive.name, sha256=ARCHIVE_SHA, bytes=archive.stat().st_size),
        source_plan_sha256=plan_sha, execution_sha256=EXECUTION, contract_sha256=plan['contract_sha256'],
        split='validation', seed=42, mode='physical_scale_keep_duration',
        protocol='fixed_query_replay_v4/grid_replay_v4; each second 1-20; normal/random; six metadata-selected pilot events',
        actual_pilot_events=selection['pilot_events'], provisional_plan_events=plan['pilot_events'],
        station_target_roles=dict(Counter(first_station.target_role)),
        selection_note='The plan explicitly labels its pilot provisional. Actual events are frozen by selection.freeze from CPU physical metadata, before model predictions; all four models share the same sealed selection receipt.',
        seals=seals, sealed_payload_files_verified=sum(s['files_verified'] for s in seals),
        evidence_level='Eight supplied smoke/pilot seals and their complete payloads verified locally; pilot links to all four supplied smoke seals verified.',
        missing_dependency_scope='Inventory/covariate payloads and seals were not included. Their recorded hashes are consistent across supplied artifacts; the supplied production status reports them verified. No claim of independent local verification of those absent files.',
        team_operational_identity_and_backend_restore='PASS; exact c66db61 file digests and 20 restored strict audits checked',
        station_grid_population_pairing='PASS; complete identity and physical input populations matched across four models',
        mdn_checks=mdn_rows, resources=resources, support=support_rows,
        resource_caveats=['One worker per family, six metadata-selected events, both geometries, 240 frames. These are pilot resource observations, not formal repeated warm latency benchmarks.',
                          'peak_device_allocated_bytes is PyTorch allocated memory, not total device or host RAM. worker time includes cold loading and full pilot work; no render time or queue wait.',
                          'Do not extrapolate a full-run completion time or compare hardware speed without the dedicated latency stage.'],
        scientific_scope='No new model selection, accuracy ranking, calibration tuning, retraining, or held-out test access. Grid has no ground truth; unsupported heights remain NA.',
        next_step=dict(manual_stages=['replay_stations', 'replay_grid --geometry normal', 'latency'],
                       preserve_existing_run=True, change_frozen_block_partition=False,
                       concurrency_proposal='6 station tasks + 6 normal-grid tasks + up to 4 latency tasks: at most 16 requested one-node/one-DCU workers across these arrays.',
                       resources='Keep existing per-worker memory/time/GRES; no host/device memory reduction justified by this package alone.'),
        hpc_status='Downloaded smoke/pilot complete. Packaged status snapshot lists the new full stages NOT_SUBMITTED; this review submits none and does not query their subsequent live state.',
        verification='Actual sealed artifacts and CSV distributions checked; no production model reexecution or new scheduler submission.')
    require('torch' not in sys.modules, 'Review unexpectedly imported model libraries')
    (output / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='PASS', seals=len(seals), sealed_files=report['sealed_payload_files_verified'],
                          resources=resources, support=support_rows), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    review(args.input_dir, args.archive, args.output_dir)
