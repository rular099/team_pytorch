"""Lock physical requests from original metadata before any new inference."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fe01.config import fingerprint, sha256
from .provenance import require, read_json, write_json

TIMES = (1, 3, 5, 10, 20)
POP_KEYS = ['dataset_id', 'event_id', 'elapsed_time', 'geometry_protocol', 'station_id', 'target_role', 'input_ids']
PAIR_KEYS = ['dataset_id', 'event_id', 'elapsed_time', 'station_id', 'geometry_protocol']
DECISION_KEYS = ['dataset_id', 'event_id', 'elapsed_time', 'geometry_protocol']
EXPECTED_POP = 'e96284780cb5f72b952c1697d4d264d307e7139a153201fa222566015aec7f69'
EXPECTED_TRUTH = 'f204deca06725af5513895416d5b0691c75c12221127854c32d1d497855155bb'
IDENTITY_KEYS = PAIR_KEYS + ['target_role', 'input_ids', 'truth', 'requested_decision_sample',
    'current_sample', 'cutout_exclusive', 'history_start_sample', 'history_end_sample',
    'latest_received_sample', 'input_count']
REQUEST_COLUMNS = list(dict.fromkeys(IDENTITY_KEYS + ['latitude', 'longitude', 'input_coords',
    'absolute_decision_utc', 'actual_elapsed_time', 'units', 'split', 'time_reference',
    'magnitude', 'depth', 'event_latitude', 'event_longitude', 'history_seconds',
    'valid_seconds', 'target_valid', 'status']))


def read_frame(path, columns=None):
    return pd.read_csv(path, usecols=columns, dtype={'dataset_id': str, 'event_id': str, 'station_id': str})


def population_audit(frame, production=True):
    require(frame.split.eq('val').all(), 'Test/train results cannot enter EVAL1 validation')
    require(frame.units.eq('log10(m/s^2)').all(), 'Physical prediction coordinate mismatch')
    require(not frame.duplicated(PAIR_KEYS).any(), 'Duplicate physical query identity')
    require(frame.status.eq('supported').all(), 'Missing/failing population cannot be omitted')
    require(frame.cutout_exclusive.eq(frame.current_sample + 1).all(), 'Exclusive cutoff mismatch')
    require(frame.history_end_sample.le(frame.current_sample).all(), 'Future history in request')
    pop = fingerprint(frame[POP_KEYS].sort_values(POP_KEYS).to_dict('records'))
    truth = fingerprint(frame[IDENTITY_KEYS].sort_values(PAIR_KEYS).to_dict('records'))
    counts = frame.groupby('target_role').size().to_dict()
    if production:
        require(pop == EXPECTED_POP and truth == EXPECTED_TRUTH, 'Locked population/labels/input/clock mismatch')
        require(len(frame) == 194265 and frame[['dataset_id', 'event_id']].drop_duplicates().shape[0] == 1310, 'Locked fixed population size mismatch')
        require(counts == dict(observed_input=54825, triggered_noninput=43850, untriggered_noninput=95590), 'Locked target-role counts mismatch')
    return dict(population_sha256=pop, truth_input_clock_sha256=truth, rows=len(frame), roles=counts,
                events=frame[['dataset_id', 'event_id']].drop_duplicates().shape[0])


def verification_decisions(frame):
    decisions = frame[DECISION_KEYS + ['input_count']].drop_duplicates().copy()
    require(not decisions.duplicated(DECISION_KEYS).any(), 'Decision input_count inconsistent')
    decisions['selection_hash'] = decisions.apply(lambda r: fingerprint(['FE01-EVAL1-probe', *[r[k] for k in DECISION_KEYS]]), axis=1)
    chosen = []
    for protocol in ('normal', 'random'):
        for time in (1, 20):
            group = decisions.loc[decisions.geometry_protocol.eq(protocol) & decisions.elapsed_time.eq(time)]
            require(len(group) >= 8, 'Verification stratum has fewer than eight decisions')
            chosen.append(group.sort_values('selection_hash').head(8))
    selected = pd.concat(chosen).drop_duplicates(DECISION_KEYS)
    for single in (True, False):
        if not (selected.input_count.eq(1) if single else selected.input_count.gt(1)).any():
            candidates = decisions.loc[decisions.input_count.eq(1) if single else decisions.input_count.gt(1)]
            require(len(candidates) > 0, 'Actual single/multi-input probe unavailable')
            selected = pd.concat([selected, candidates.sort_values('selection_hash').head(1)]).drop_duplicates(DECISION_KEYS)
    require(32 <= len(selected) <= 40, 'Verification budget exceeds contract')
    return selected.reset_index(drop=True)


def select_cases(frame):
    # Metadata only: population counts and catalog magnitude, never predictions.
    common = frame.groupby(['dataset_id', 'event_id', 'geometry_protocol', 'elapsed_time']).station_id.nunique()
    allowed = common.groupby(['dataset_id', 'event_id']).min()
    allowed = allowed[allowed >= 5].index
    events = frame[['dataset_id', 'event_id', 'magnitude', 'event_latitude', 'event_longitude', 'depth']].drop_duplicates()
    require(not events.duplicated(['dataset_id', 'event_id']).any(), 'Catalog metadata varies inside event')
    eligible = events.set_index(['dataset_id', 'event_id']).loc[allowed].reset_index()
    ordered = events.sort_values(['magnitude', 'dataset_id', 'event_id'])
    selected, used = [], set()
    for quantile in (.25, .5, .9):
        target = float(ordered.magnitude.quantile(quantile))
        candidates = eligible.copy()
        candidates['distance_to_quantile'] = (candidates.magnitude - target).abs()
        candidates = candidates.sort_values(['distance_to_quantile', 'dataset_id', 'event_id'])
        pick = next(row for row in candidates.to_dict('records') if (row['dataset_id'], row['event_id']) not in used)
        used.add((pick['dataset_id'], pick['event_id']))
        selected.append(dict(quantile=quantile, target_magnitude=target, **pick))
    return dict(split='val', rule='magnitude quantiles .25/.50/.90 of locked cohort; nearest eligible metadata; >=5 queries at every common decision; ties dataset/event',
                cases=selected, selection_uses_predictions=False)


def freeze_plan(source_root, output):
    source_root, output = Path(source_root), Path(output)
    run_id = 'formal__team_original_scratch__seed42'
    source_csv = source_root / run_id / 'validation_epoch11.csv.gz'
    published = read_frame(Path(__file__).resolve().parents[1] / 'reports/fe01_hpc_results_20261004/run_summary.csv')
    expected = published.loc[published.run_id.eq(run_id)].iloc[0].selected_source_sha256
    require(sha256(source_csv) == expected, 'Original selected CSV byte SHA mismatch')
    frame = read_frame(source_csv, REQUEST_COLUMNS)
    audit = population_audit(frame)
    frame['request_id'] = [fingerprint([r.dataset_id, r.event_id, r.elapsed_time, r.geometry_protocol, r.station_id]) for r in frame.itertuples()]
    frame.to_csv(output / 'fixed_requests.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    verification_decisions(frame).to_csv(output / 'verification_decisions.csv', index=False)
    cases = select_cases(frame)
    cases['source_csv_sha256'] = expected
    cases['cohort_sha256'] = fingerprint(read_json(source_root / run_id / 'resolved_config.json')['audited_cohorts']['val'])
    write_json(output / 'case_manifest.json', cases)
    random_source = source_root / 'audits' / run_id / 'random_validation_times_manifest.csv'
    random = read_frame(random_source)
    require(not random.duplicated(['dataset_id', 'event_id', 'draw']).any(), 'Random draw identity repeated')
    require(random.elapsed_time.between(1, 20).all() and not random.elapsed_time.isin(TIMES).any(), 'Random times outside original domain or fixed ticks')
    require(set(map(tuple, random[['dataset_id', 'event_id']].to_numpy())) == set(map(tuple, frame[['dataset_id', 'event_id']].to_numpy())), 'Random cohort differs')
    require((random.groupby(['dataset_id', 'event_id']).size() == 3).all(), 'Expected original three draws per event')
    (output / 'random_time_draws.csv').write_bytes(random_source.read_bytes())
    snapshots = random[['dataset_id', 'event_id', 'elapsed_time']].drop_duplicates().reset_index(drop=True)
    snapshots['snapshot_id'] = [fingerprint([r.dataset_id, r.event_id, r.elapsed_time]) for r in snapshots.itertuples()]
    mapping = random.merge(snapshots, on=['dataset_id', 'event_id', 'elapsed_time'], validate='many_to_one')
    mapping.to_csv(output / 'random_draw_mapping.csv', index=False)
    hashes = {p.parent.name: sha256(p) for p in sorted((source_root / 'audits').glob('formal__*/random_validation_times_manifest.csv'))}
    audit.update(source_csv_sha256=expected, random_manifest_source_sha256=sha256(random_source),
                 random_manifest_hashes=hashes, random_draws=len(random), unique_random_snapshots=len(snapshots),
                 duplicate_draws=len(random) - len(snapshots), random_time_domain_seconds=[1, 20],
                 draw_weight='mean targets in event/draw/geometry, then mean original draws within event, then events, then geometries')
    write_json(output / 'request_population_audit.json', audit)
    return frame


def lock_new_requests(cfg, plan, output):
    """One shared HDF planning pass for random/long/replay, before model outputs."""
    from fe01.data import read_event, split_catalog, geometry
    from fe01.windows import CAPABILITIES, build_window
    from fe01.spatial import grid
    plan, output = Path(plan), Path(output)
    catalog = split_catalog(cfg, 'val').set_index(['dataset_id', 'event_id'])
    # Original CSV does not carry query elevation. Lock its real station
    # coordinates/storage metadata from the already authenticated HDF shards
    # before any model is loaded. Never invent zero elevation for real stations.
    import h5py
    from fe01.data import strings
    station_metadata=[]
    fixed=read_frame(plan/'fixed_requests.csv.gz')
    for path,entries in catalog.reset_index().groupby('hdf5_path'):
        with h5py.File(path,'r') as handle:
            for entry in entries.itertuples():
                group=handle['data'][str(entry.event_id)];ids=strings(group['station_codes'][()]);net=strings(group['source_network'][()])
                coords=group['coords'][()].astype(np.float32);starts=group['record_start_sample'][()];lengths=group['valid_n_samples'][()]
                for i in np.flatnonzero(net=='knt'):
                    station_metadata.append(dict(dataset_id=entry.dataset_id,event_id=str(entry.event_id),station_id=ids[i],latitude=float(coords[i,0]),
                        longitude=float(coords[i,1]),query_elevation=float(coords[i,2]),storage_start_sample=int(starts[i]),storage_valid_n_samples=int(lengths[i])))
    station_metadata=pd.DataFrame(station_metadata)
    joined=fixed.merge(station_metadata,on=['dataset_id','event_id','station_id'],how='left',suffixes=('_original',''),validate='many_to_one')
    require(joined.query_elevation.notna().all(),'Frozen query station absent from original Japan metadata')
    for coordinate in ('latitude','longitude'):
        require(np.array_equal(joined[coordinate+'_original'].to_numpy(dtype=np.float32),joined[coordinate].to_numpy(dtype=np.float32)),'Original frozen query coordinate differs from HDF')
    station_metadata.to_csv(output/'fixed_station_metadata.csv.gz',index=False)
    random = read_frame(plan / 'random_draw_mapping.csv')
    cases = read_json(plan / 'case_manifest.json')['cases']
    times = [('random', r.dataset_id, r.event_id, float(r.elapsed_time), None) for r in random.drop_duplicates(['snapshot_id']).itertuples()]
    times += [('long', ds, event, t, None) for ds, event in catalog.index for t in (40, 90)]
    times += [('replay', case['dataset_id'], case['event_id'], t, mode) for case in cases for t in range(1, 21) for mode in ('natural', 'fixed_s0')]
    rows, support, grids, fixed_inputs = [], [], {}, {}
    for scope, dataset, event_id, elapsed, input_mode in times:
        row = catalog.loc[(dataset, event_id)].copy()
        row['dataset_id'], row['event_id'] = dataset, event_id
        event = read_event(row, cfg, elapsed)
        _, masks, info = build_window(event['waveform'], event['storage'], event['reference_sample'], elapsed, CAPABILITIES['team_original_scratch'])
        protocols = ('normal', 'random')
        for protocol in protocols:
            selected, queries, roles, _ = geometry(event, masks.all(1), cfg, protocol)
            case_key = (dataset, event_id, protocol)
            if scope == 'replay':
                if case_key not in fixed_inputs:
                    initial = read_event(row, cfg, 1)
                    _, imask, _ = build_window(initial['waveform'], initial['storage'], initial['reference_sample'], 1, CAPABILITIES['team_original_scratch'])
                    s0, q0, _, _ = geometry(initial, imask.all(1), cfg, protocol)
                    fixed_inputs[case_key] = (s0, q0)
                s0, q0 = fixed_inputs[case_key]
                queries = q0[~np.isin(q0, s0)]
                if input_mode == 'fixed_s0':
                    selected = s0
                    roles = np.where(np.isin(np.arange(len(event['ids'])), selected), 'observed_input',
                                     np.where((event['picks'] > 0) & (event['picks'] <= event['current_sample']), 'triggered_noninput', 'untriggered_noninput'))
                gkey = (dataset, event_id)
                if gkey not in grids:
                    coords = event['coords']
                    padlat, padlon = 50 / 111.195, 50 / (111.195 * np.cos(np.deg2rad(float(coords[:, 0].mean()))))
                    bounds = [float(coords[:, 0].min() - padlat), float(coords[:, 0].max() + padlat), float(coords[:, 1].min() - padlon), float(coords[:, 1].max() + padlon)]
                    grids[gkey] = dict(dataset_id=dataset, event_id=event_id, bounds=bounds, spacing_km=10,
                                      elevation_assumption='0m demonstration only; no DEM', points=grid(bounds, 10).tolist())
            status = 'supported' if len(selected) and len(queries) and info['status'] == 'supported' else 'empty'
            support.append(dict(scope=scope, dataset_id=dataset, event_id=event_id, elapsed_time=elapsed, geometry_protocol=protocol, input_mode=input_mode or 'natural', status=status, queries=len(queries), inputs=len(selected)))
            for index in queries:
                record = dict(scope=scope, input_mode=input_mode or 'natural', dataset_id=dataset, event_id=event_id,
                    elapsed_time=float(elapsed), geometry_protocol=protocol, station_id=event['ids'][index],
                    latitude=float(event['coords'][index, 0]), longitude=float(event['coords'][index, 1]), elevation=float(event['coords'][index, 2]),
                    target_role=roles[index], truth=float(event['pga'][index]), split='val', units='log10(m/s^2)', status=status,
                    input_ids=json.dumps(event['ids'][selected].tolist()), input_coords=json.dumps(event['coords'][selected].tolist()), input_count=len(selected),
                    magnitude=event['magnitude'], depth=float(event['event_location'][2]), event_latitude=float(event['event_location'][0]), event_longitude=float(event['event_location'][1]),
                    time_reference='first_p_pick', absolute_decision_utc=(event['record_start_utc'] + info['current_sample'] / 100 if event['record_start_utc'] is not None else None),
                    **{k: info[k] for k in ('current_sample', 'cutout_exclusive', 'history_start_sample', 'history_end_sample', 'actual_elapsed_time', 'history_seconds', 'requested_decision_sample', 'latest_received_sample')})
                record['request_id'] = fingerprint([scope, input_mode, dataset, event_id, elapsed, protocol, str(event['ids'][index])])
                rows.append(record)
    frame = pd.DataFrame(rows)
    for scope in ('random', 'long', 'replay'):
        frame[frame.scope.eq(scope)].to_csv(output / (scope + '_requests.csv.gz'), index=False, compression={'method': 'gzip', 'mtime': 0})
    pd.DataFrame(support).to_csv(output / 'planning_support.csv', index=False)
    write_json(output / 'grid_manifest.json', list(grids.values()))
    write_json(output / 'new_requests_identity.json', dict(shared_geometry_sampling_seed=cfg['sampling_seed'],
        source_random_draws_sha256=sha256(plan / 'random_time_draws.csv'), case_manifest_sha256=sha256(plan / 'case_manifest.json'),
        request_sha256={s: sha256(output / (s + '_requests.csv.gz')) for s in ('random', 'long', 'replay')},
        population_rows=frame.groupby('scope').size().to_dict(), planning_uses_predictions=False,
        fixed_station_metadata_sha256=sha256(output/'fixed_station_metadata.csv.gz')))
