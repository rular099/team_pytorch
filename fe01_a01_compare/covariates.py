"""Read-only HDF5 sidecars. Full records are available only to offline truth stage."""
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from fe01.windows import clock
from fe01.spatial import distance_km
from .clocks import audit_clock
from .contracts import EVENT
from .io import csv_read, csv_write, fingerprint, read_json, require, sha256, write_json
from .quality import decision_quality
from .query_grid import geometry_covariates, input_plan, make_grid


def strings(a):
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in a])


def catalog(config, cohort):
    f = csv_read(config['data']['split_manifest'])
    f = f.rename(columns={'EVENT': 'event_id'}) if 'EVENT' in f else f
    f['event_id'] = f.event_id.astype(str)
    f['split'] = f.split.replace({'validation': 'val', 'dev': 'val'})
    # Only the frozen validation IDs are touched. No test waveform path is opened.
    f = f[f.split.eq('val')].copy()
    if 'source_data_path' in f:
        f['dataset_id'] = f.source_data_path.map(lambda p: Path(p).name)
    elif 'year' in f:
        f['dataset_id'] = 'japan_'+f.year.astype(str)+'.hdf5'
    else:
        f['dataset_id'] = f.event_id.str[:4].map(lambda y: 'japan_'+y+'.hdf5')
    f = f.merge(cohort[EVENT].drop_duplicates(), on=EVENT, how='inner', validate='one_to_one')
    require(len(f) == len(cohort[EVENT].drop_duplicates()), 'Validation cohort missing catalog events')
    f['hdf5_path'] = f.dataset_id.map(lambda name: str(Path(config['data']['root'])/name.split('_')[1][:4]/name))
    return f.sort_values(EVENT).reset_index(drop=True)


def physical_event(row, full=False, max_elapsed=20):
    with h5py.File(row['hdf5_path'], 'r') as h:
        fs = float(np.asarray(h['metadata/sampling_rate'][()]).ravel()[0])
        require(fs == 100, 'No implicit resampling')
        g = h['data'][str(row['event_id'])]
        keep = np.flatnonzero(strings(g['source_network'][()]) == 'knt')
        ids = strings(g['station_codes'][keep])
        require(len(set(ids)) == len(ids), 'Duplicate station identity')
        coords = g['coords'][keep].astype(np.float32)
        picks = g['p_picks'][keep].astype(np.float64)
        starts = g['record_start_sample'][keep].astype(np.int64)
        lengths = g['valid_n_samples'][keep].astype(np.int64)
        require(len(keep)>0 and np.isfinite(coords).all() and (starts>=0).all() and (lengths>0).all(),
                'Invalid station coordinates/storage metadata')
        good = np.isfinite(picks) & (picks > 0)
        require(good.any(), 'No reference pick')
        ref = float(picks[good].min())
        stop = g['waveforms'].shape[1] if full else min(g['waveforms'].shape[1], max(0, clock(ref, max_elapsed, fs)[1]))
        wave = g['waveforms'][keep, :stop, :].transpose(0, 2, 1).astype(np.float32)
        samples = np.arange(stop)
        mask = (samples[None, :] >= starts[:, None]) & (samples[None, :] < (starts+lengths)[:, None])
        if 'waveform_sample_valid' in g:
            explicit = g['waveform_sample_valid'][keep, :stop]
            if explicit.ndim == 3:
                explicit = explicit.all(-1)
            mask &= explicit
        if 'component_valid' in g:
            mask &= g['component_valid'][keep].all(-1)[:, None]
        pga = g['pga'][keep].astype(np.float32)
        times = strings(g['record_start_time_jst'][keep]) if 'record_start_time_jst' in g else []
        clock_info = audit_clock(times, starts, fs, row.get('Origin_Time(JST)'), ref)
        # No clipping is inferred from repeated values without instrument limits.
        clipping = g['clipping'][keep].astype(bool) if 'clipping' in g else np.full(len(ids), False)
        if clipping.ndim>1:
            clipping=clipping.any(axis=tuple(range(1,clipping.ndim)))
        clipping_known = 'clipping' in g
        attrs = {str(k): str(v) for k, v in h.attrs.items()}
        attrs.update({'group_'+str(k): str(v) for k, v in g.attrs.items()})
    return dict(dataset_id=row['dataset_id'], event_id=str(row['event_id']), fs=fs, waveform=wave,
                storage=mask, coords=coords, picks=picks, ids=ids, starts=starts, lengths=lengths,
                pga=pga, reference_sample=ref, queries=np.flatnonzero(np.isfinite(pga)),
                clock=clock_info, record_start_utc=clock_info['record_start_utc'],
                clipping=clipping, clipping_known=clipping_known, source_attributes=attrs,
                magnitude=float(row['Magnitude']), depth=float(row['DEPTH']),
                p_pick_source=str(row.get('P_Pick_Source','UNKNOWN_catalog_or_repaired_pick; ideal_trigger_replay')),
                event_latitude=float(row['Latitude']), event_longitude=float(row['Longitude']))


def event_summary(event):
    return dict(dataset_id=event['dataset_id'], event_id=event['event_id'], magnitude=event['magnitude'],
                magnitude_scale='catalog_native_UNKNOWN_scale', depth=event['depth'],
                query_count=len(event['queries']), stations=len(event['ids']),
                median_epicentral_distance_km=float(np.median(distance_km(event['coords'][event['queries'],0],event['coords'][event['queries'],1],event['event_latitude'],event['event_longitude']))),
                record_samples_min=int(event['lengths'].min()), record_samples_max=int(event['lengths'].max()),
                third_coordinate='station_height_km', metadata_role='retrospective_catalog_scoring_only',
                p_pick_source=event['p_pick_source'],trigger_scope='ideal_trigger_replay_not_continuous_detection',
                **event['clock'])


def hydrate(rows, legacy, contract, out):
    """Each event prefix is read once; all models share the exact same physical sidecars."""
    request_rows, input_rows, event_rows, decisions, grids = [], [], [], [], []
    for _, row in rows.iterrows():
        e = physical_event(row)
        base = dict(zip(EVENT, [e['dataset_id'], e['event_id']]))
        event_rows.append(event_summary(e))
        grid = make_grid(e['coords'][e['queries']], contract['grid'])
        grids.append(dict(**base, **grid))
        ever = {'normal': set(), 'random': set()}
        event_decisions = []
        for geo in ('normal', 'random'):
            for t in contract['replay']['times_first_p_seconds']:
                selected = input_plan(e, t, geo)
                ever[geo].update(e['ids'][selected].tolist())
                station_quality, quality = decision_quality(e, selected, t)
                current, cutoff = clock(e['reference_sample'], t, e['fs'])
                utc = e['record_start_utc']+current/e['fs'] if e['record_start_utc'] is not None else None
                d = dict(**base, elapsed_time=float(t), geometry_protocol=geo,
                         input_ids=json.dumps(e['ids'][selected].tolist()), input_coords=json.dumps(e['coords'][selected].tolist()),
                         current_sample=current, cutout_exclusive=cutoff, requested_decision_sample=e['reference_sample']+t*e['fs'],
                         history_start_sample=int(np.floor(e['reference_sample']))-500, history_end_sample=current,
                         actual_elapsed_time=(current-e['reference_sample'])/e['fs'],
                         history_seconds=(cutoff-int(np.floor(e['reference_sample']))+500)/e['fs'],
                         latest_received_sample=min(current,e['waveform'].shape[-1]-1),
                         absolute_decision_utc=utc, first_p_minus_origin_seconds=e['clock']['first_p_minus_origin_seconds'],
                         origin_elapsed_seconds=utc-e['clock']['origin_utc'] if utc is not None and e['clock']['origin_utc'] is not None else None,
                         protocol='fixed_query_replay_v4', status='supported' if len(selected) else 'no_input', **quality)
                event_decisions.append(d)
                for q in station_quality:
                    input_rows.append(dict(**base, elapsed_time=float(t), geometry_protocol=geo, **q))
        for d in event_decisions:
            geo = d['geometry_protocol']
            input_ids = set(json.loads(d['input_ids']))
            selected = [i for i, s in enumerate(e['ids']) if s in input_ids]
            gc = geometry_covariates(e['coords'][e['queries']], e['coords'][selected])
            for i, cov in zip(e['queries'], gc):
                r = dict(**d, station_id=str(e['ids'][i]), latitude=float(e['coords'][i, 0]),
                         longitude=float(e['coords'][i, 1]), elevation=float(e['coords'][i, 2]), truth=float(e['pga'][i]),
                         fixed_never_input=str(e['ids'][i]) not in ever[geo],
                         target_role='observed_input' if str(e['ids'][i]) in input_ids else 'triggered_noninput'
                                     if 0 < e['picks'][i] <= d['current_sample'] else 'untriggered_noninput',
                         epicentral_distance_km=float(distance_km(e['coords'][i, 0], e['coords'][i, 1], e['event_latitude'], e['event_longitude'])),
                         magnitude=e['magnitude'], depth=e['depth'], event_latitude=e['event_latitude'], event_longitude=e['event_longitude'],
                         split='val', units='log10(m/s^2)', seed=42, time_reference='first_p_pick', **cov)
                r['request_id'] = fingerprint(['fixed_query_replay_v4', *[r[k] for k in EVENT], geo, d['elapsed_time'], r['station_id']])
                request_rows.append(r)
        decisions.extend(event_decisions)
    events_frame=pd.DataFrame(event_rows)
    input_frame=pd.DataFrame(input_rows)
    decisions_frame=pd.DataFrame(decisions)
    requests_frame=pd.DataFrame(request_rows)
    csv_write(events_frame, out/'event_covariates.csv')
    csv_write(input_frame, out/'input_station_covariates.csv.gz')
    csv_write(decisions_frame, out/'decision_manifest.csv.gz')
    csv_write(requests_frame, out/'request_covariates.csv.gz')
    csv_write(events_frame[EVENT+['status','record_start_utc','origin_utc','first_p_utc','first_p_minus_origin_seconds']],out/'clock_audit.csv')
    csv_write(input_frame.groupby(['status','reason'],dropna=False).size().rename('input_station_decisions').reset_index(),out/'snr_quality_audit.csv')
    coverage=[]
    for name in ('snr_db','physical_missing_fraction','post_p_valid_seconds_min','origin_elapsed_seconds'):
        coverage.append(dict(field=name,known=int(decisions_frame[name].notna().sum()),total=len(decisions_frame),
                             missing_fraction=float(decisions_frame[name].isna().mean()),models='shared',
                             available_at_decision=name!='origin_elapsed_seconds',retrospective_only=name=='origin_elapsed_seconds'))
    csv_write(pd.DataFrame(coverage),out/'metadata_coverage.csv')
    write_json(out/'grid_manifest.json', grids)
    write_json(out/'request_manifest.json',dict(protocol='fixed_query_replay_v4',events=len(events_frame),
        target_requests=len(requests_frame),decisions=len(decisions_frame),
        request_sha256=sha256(out/'request_covariates.csv.gz'),decision_sha256=sha256(out/'decision_manifest.csv.gz'),
        grid_sha256=sha256(out/'grid_manifest.json'),
        fixed_panels='all legal station targets and retrospective never-input subset; no role-based target deletion',
        history='5s before original first P through inclusive current integer sample; native_prefix only'))
    write_json(out/'sidecar_scope.json', dict(protocol='fixed_query_replay_v4', shared_by_all_models=True,
               data_reads='one event prefix per worker; only through firstP+20s',
               reference='original first-P prefix start retained', upstream='UPSTREAM_CAUSALITY_UNKNOWN',
               truth_role='retrospective scoring only; never passed to forward'))
