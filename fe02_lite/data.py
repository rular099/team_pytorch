"""Sensor-row views of existing Japan shards. No waveform copies or guessed channels."""
import copy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from fe01.data import (strings, jst_timestamp, split_catalog as base_catalog,
                       prepare_sample as base_prepare, geometry, collate)
from fe01.sampling import TimeSampler
from fe01.windows import build_window, CAPABILITIES, clock
from .config import VIEWS, sampling_config, fingerprint, sha256, write_json


class NoViewRecords(ValueError):
    pass


def catalog(cfg, split):
    if split not in ('train', 'val'):
        raise ValueError('This release exposes train/validation only; held-out test is not authorized')
    result = base_catalog(cfg, split)
    # A canonical identity is the builder EVENT key, common across networks/layers.
    # Missing keys are excluded in the audit; no per-view randomized split.
    result['canonical_event_id'] = result.event_id
    return result


def transform_coordinates(values, cfg, source=False, inverse=False):
    values = np.asarray(values, dtype=np.float32).copy()
    origin = np.asarray(cfg['coordinates']['origin'], np.float32)
    scale = np.asarray(cfg['coordinates']['scale'], np.float32)
    # Unified third axis is positive UP, km; source catalogue depth is positive DOWN.
    if inverse:
        values = values*scale + origin
        if source:
            values[..., 2] *= -1
    else:
        if source:
            values[..., 2] *= -1
        values = (values-origin)/scale
    return values


def metadata(group, view):
    """Every output row refers to exactly the same sensor in every field."""
    if group['waveforms'].ndim != 3 or group['waveforms'].shape[-1] != 3:
        raise ValueError('Only verified station,time,NEZ sensor-row schema is supported; no six-channel guessing')
    network, layer, suffix = VIEWS[view]
    all_network = strings(group['source_network'][()])
    all_layer = strings(group['sensor_class'][()])
    keep = np.flatnonzero((all_network == network) & (all_layer == layer))
    if not len(keep):
        raise NoViewRecords('no_records_for_view')
    required = ['sensor_suffix', 'station_codes', 'coords', 'p_picks', 'record_start_sample',
                'valid_n_samples', 'pga', 'pga_norm_resampled_mps2', 'record_start_time_jst']
    for name in required:
        if name not in group:
            raise ValueError('Verified Japan sensor-row schema missing '+name)
    actual_suffix = strings(group['sensor_suffix'][keep])
    # The raw archive builder stores KNET suffix ''; the flat CSV builder
    # explicitly stores '0'. Both mappings are confirmed in repository sources.
    permitted_suffixes = ('','0') if view == 'knet' else (suffix,)
    if not np.isin(actual_suffix,permitted_suffixes).all():
        raise ValueError('Layer/sensor_suffix conflict')
    station = strings(group['station_codes'][keep])
    ids = np.asarray([f'{network}:{s}:{layer}' for s in station])
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate sensor rows in event')
    coords = group['coords'][keep].astype(np.float32)
    picks = group['p_picks'][keep].astype(np.float64)
    starts = group['record_start_sample'][keep].astype(np.int64)
    lengths = group['valid_n_samples'][keep].astype(np.int64)
    pga = group['pga'][keep].astype(np.float32)
    physical_pga = group['pga_norm_resampled_mps2'][keep].astype(np.float64)
    finite_label = np.isfinite(pga) & np.isfinite(physical_pga) & (physical_pga > 0)
    if not np.allclose(pga[finite_label], np.log10(physical_pga[finite_label]+1e-10), atol=2e-6):
        raise ValueError('pga is not log10(selected sensor vector-norm resampled PGA + 1e-10)')
    pga[~finite_label] = np.nan
    reasons = [[] for _ in keep]
    good = np.isfinite(coords).all(-1) & (starts >= 0) & (lengths > 0)
    good &= starts+lengths <= group['waveforms'].shape[1]
    component = np.ones(len(keep), bool)
    if 'component_valid' in group:
        flags = group['component_valid'][keep]
        if flags.shape != (len(keep), 3):
            raise ValueError('component_valid must describe NEZ for each sensor row')
        component = flags.astype(bool).all(-1)
        good &= component
    for i in range(len(keep)):
        if not np.isfinite(coords[i]).all(): reasons[i].append('invalid_coordinates')
        if starts[i] < 0 or lengths[i] <= 0 or starts[i]+lengths[i] > group['waveforms'].shape[1]:
            reasons[i].append('invalid_storage_interval')
        if not component[i]: reasons[i].append('missing_component')
        if not finite_label[i]: reasons[i].append('missing_pga_target')
        if not np.isfinite(picks[i]) or picks[i] <= 0: reasons[i].append('no_trigger_pick')
    timestamps = np.asarray([jst_timestamp(v).timestamp() for v in strings(group['record_start_time_jst'][keep])])
    axes = timestamps-starts/100
    if not np.isfinite(axes).all() or np.ptp(axes) > .01001:
        raise ValueError('Sensor record start timestamps disagree with shared sample axis')
    pick_source = strings(group['p_pick_refined_source'][keep]) if 'p_pick_refined_source' in group else np.full(len(keep),'not recorded')
    if 'p_pick_refined_aligned' in group and not np.array_equal(group['p_pick_refined_aligned'][keep],picks,equal_nan=True):
        raise ValueError('p_picks differs from stored final refined pick; ambiguous timing semantics')
    return dict(indices=keep, ids=ids, physical_ids=np.asarray([network+':'+s for s in station]),
                coords=coords, picks=picks, starts=starts, lengths=lengths, pga=pga,
                record_start_utc=float(axes[0]), good=good, reasons=reasons,
                sensor_class=layer, network=network,pick_source=pick_source)


def read_event(row, cfg, elapsed):
    with h5py.File(row['hdf5_path'], 'r') as handle:
        fs = float(np.asarray(handle['metadata/sampling_rate'][()]).reshape(-1)[0])
        if fs != 100:
            raise ValueError('Unexpected sampling rate; no implicit resampling')
        group = handle['data'][str(row['event_id'])]
        meta = metadata(group, cfg['dataset_view'])
        usable = np.flatnonzero(meta['good'])
        if not len(usable):
            raise NoViewRecords('no_valid_sensor_rows')
        picks = meta['picks'][usable]
        good_picks = np.isfinite(picks) & (picks > 0)
        if not good_picks.any():
            raise NoViewRecords('no_reference_pick')
        reference_sample = float(picks[good_picks].min())
        current, cutout = clock(reference_sample, elapsed, 100)
        # HDF5 read stops at cutoff; no full-future encoding followed by a mask.
        stop = max(0, min(cutout, group['waveforms'].shape[1]))
        keep = meta['indices'][usable]
        wave = group['waveforms'][keep, :stop, :].astype(np.float32).transpose(0,2,1)
        indices = np.arange(stop)[None]
        starts, lengths = meta['starts'][usable], meta['lengths'][usable]
        storage = (indices >= starts[:, None]) & (indices < (starts+lengths)[:, None])
        if 'waveform_sample_valid' in group:
            explicit = group['waveform_sample_valid'][keep, :stop]
            if explicit.ndim == 3:
                if explicit.shape[-1] != 3:
                    raise ValueError('Explicit mask component order mismatch')
                explicit = explicit.all(-1)
            if explicit.shape != storage.shape:
                raise ValueError('Explicit storage mask shape mismatch')
            storage &= explicit.astype(bool)
        total = group['waveforms'].shape[1]
    return dict(event_id=str(row['event_id']), canonical_event_id=str(row['canonical_event_id']),
        dataset_id=row['dataset_id'], split=row['split'], waveform=wave, storage=storage,
        coords=meta['coords'][usable], picks=picks, ids=meta['ids'][usable],
        physical_ids=meta['physical_ids'][usable], pga=meta['pga'][usable],
        reference_sample=reference_sample, current_sample=current, cutout_exclusive=cutout,
        total_record_samples=total, record_start_utc=meta['record_start_utc'], reference='first_p_pick',
        magnitude=float(row['Magnitude']), dataset_view=cfg['dataset_view'],
        sensor_class=meta['sensor_class'], network=meta['network'],
        event_location=np.asarray([row['Latitude'],row['Longitude'],row['DEPTH']],np.float32),
        input_allowed=np.ones(len(usable), bool), query_allowed=np.ones(len(usable), bool))


def prepare(event, cfg, elapsed, protocol='normal', epoch=0, draw=0, queries=None):
    sample = base_prepare(event, sampling_config(cfg), elapsed, protocol, epoch, draw, queries=queries)
    info = sample['info']
    info.update(dataset_view=cfg['dataset_view'], canonical_event_id=event['canonical_event_id'],
                sensor_class=event['sensor_class'], network=event['network'],
                time_reference='view_first_valid_P_on_shared_shard_axis',
                reference_sample=event['reference_sample'], record_axis_start_utc=event['record_start_utc'])
    if sample['inputs'] is None:
        return sample
    n, q = len(info['input_ids']), len(info['query_ids'])
    sample['inputs'][1][:n] = torch.from_numpy(transform_coordinates(info['input_coords'], cfg))
    sample['inputs'][3][:q] = torch.from_numpy(transform_coordinates(info['query_coords'], cfg))
    sample['labels'][1] = torch.from_numpy(transform_coordinates(event['event_location'], cfg, source=True))
    info['input_set_id'] = fingerprint(sorted(info['input_ids']))
    info['observation_id'] = fingerprint(dict(view=cfg['dataset_view'], event=event['canonical_event_id'],
        dataset_id=event['dataset_id'],
        cutoff=info['cutout_exclusive'], axis=event['record_start_utc'], inputs=sorted(info['input_ids']),
        geometry=info['geometry_protocol']))
    info['target_physical_ids'] = event['physical_ids'][info['query_indices']].tolist()
    info['record_ids'] = [f"{event['dataset_id']}:{event['event_id']}:{s}" for s in info['query_ids']]
    info['mask_sha256'] = hashlib.sha256(sample['inputs'][5].numpy().tobytes()).hexdigest()
    return sample


class TrainingDataset(Dataset):
    def __init__(self, cfg):
        self.cfg, self.epoch = cfg, 0
        self.catalog = catalog(cfg, 'train')
        self.sampler = TimeSampler(max_sample=9499, bins=cfg['realtime']['bins'],
                                   probabilities=cfg['realtime']['probabilities'])

    def __len__(self):
        return len(self.catalog)*self.cfg['realtime']['draws_per_event']

    def __getitem__(self, index):
        i, draw = divmod(index, self.cfg['realtime']['draws_per_event'])
        row = self.catalog.iloc[i]
        elapsed = self.sampler.sample(row.dataset_id, row.event_id, self.epoch, draw, self.cfg['sampling_seed'])
        event = read_event(row, self.cfg, elapsed)
        sample = prepare(event, self.cfg, elapsed, 'mixed', self.epoch, draw)
        sample['info'].update(sampling_epoch=self.epoch, draw_index=draw, cohort_event_index=i)
        if sample['inputs'] is None:
            raise ValueError('Audited training support failed: '+json.dumps(sample['info']))
        return sample


def audit_data(cfg, destination):
    """Compute-node only. Freeze cohorts, per-view statistics and diagnostic plan."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    records, exclusions, cohorts, values, validation, diagnostic = [], [], {}, [], [], []
    filters = []
    waveform_label_checks = []
    shards = {}
    shard_semantics = {}
    reuse_path = cfg['data'].get('shard_identity_manifest')
    reusable = json.loads(Path(reuse_path).read_text()).get('shards',{}) if reuse_path else {}
    def shard_identity(path):
        path = Path(path).resolve()
        stat = path.stat()
        record = reusable.get(str(path))
        if record is not None:
            if (record['bytes'],record['mtime_ns']) != (stat.st_size,stat.st_mtime_ns) or not record.get('sha256'):
                raise ValueError('Reusable shard identity stat/hash mismatch; recompute hashes')
            return dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=record['sha256'])
        return dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=sha256(path))
    split_manifest = pd.read_csv(cfg['data']['split_manifest'], dtype={'EVENT':str,'event_id':str})
    event_key = 'EVENT' if 'EVENT' in split_manifest else 'event_id'
    known = set(split_manifest[event_key])
    for path in sorted(Path(cfg['data']['root']).glob('*/japan_*.hdf5')):
        shards[str(path.resolve())] = shard_identity(path)
        with h5py.File(path, 'r') as handle:
            shard_semantics[str(path.resolve())] = {name:strings(np.asarray(handle['metadata'][name][()]).reshape(-1)).tolist()
                for name in ('sampling_rate','pretrigger_seconds','alignment_mode','source_format','final_pick','year')
                if name in handle['metadata']}
            for event_id in handle['data']:
                if event_id not in known:
                    g = handle['data'][event_id]
                    network, layer, _ = VIEWS[cfg['dataset_view']]
                    count = int(((strings(g['source_network'][()]) == network) &
                                 (strings(g['sensor_class'][()]) == layer)).sum())
                    exclusions.append(dict(dataset_id=path.name, event_id=event_id, split='unassigned',
                                           records=count, reason='absent_from_frozen_split; no new split invented'))
    for split in ('train', 'val'):
        table = catalog(cfg, split)
        limit = cfg.get('audit_limits', {}).get(split)
        if limit:
            table = table.iloc[:limit]
        kept = []
        for _, row in table.iterrows():
            if not Path(row.hdf5_path).is_file():
                raise FileNotFoundError(row.hdf5_path)
            if str(Path(row.hdf5_path).resolve()) not in shards:
                path = Path(row.hdf5_path)
                shards[str(path.resolve())] = shard_identity(path)
            with h5py.File(row.hdf5_path, 'r') as handle:
                group = handle['data'][row.event_id]
                total_rows = group['waveforms'].shape[0]
                try:
                    meta = metadata(group, cfg['dataset_view'])
                except NoViewRecords:
                    filters.append(dict(split=split,event_id=row.event_id,dataset_id=row.dataset_id,
                        total_sensor_rows=total_rows,selected_view_rows=0,excluded_other_view_rows=total_rows))
                    exclusions.append(dict(dataset_id=row.dataset_id,event_id=row.event_id,split=split,records=0,reason='no_records_for_view'))
                    continue
                filters.append(dict(split=split,event_id=row.event_id,dataset_id=row.dataset_id,
                    total_sensor_rows=total_rows,selected_view_rows=len(meta['ids']),excluded_other_view_rows=total_rows-len(meta['ids'])))
                # Full-record label correspondence is an audit-only read, never a
                # model feature. Bounded to three valid records per data view.
                for i in np.flatnonzero(meta['good'] & np.isfinite(meta['pga'])):
                    if len(waveform_label_checks)>=3: break
                    j = int(meta['indices'][i])
                    start,stop = int(meta['starts'][i]),int(meta['starts'][i]+meta['lengths'][i])
                    wave = group['waveforms'][j,start:stop,:].astype(np.float64)
                    valid = np.ones(stop-start,bool)
                    if 'waveform_sample_valid' in group:
                        explicit = group['waveform_sample_valid'][j,start:stop]
                        valid &= explicit.all(-1) if explicit.ndim==2 else explicit.astype(bool)
                    if not valid.any() or not np.isfinite(wave[valid]).all():
                        raise ValueError('Audit waveform has no finite storage-valid samples')
                    recomputed = float(np.linalg.norm(wave[valid],axis=-1).max())
                    declared = float(group['pga_norm_resampled_mps2'][j])
                    if not np.isclose(recomputed,declared,rtol=1e-4,atol=1e-8):
                        raise ValueError('Sensor waveform/PGA magnitude mismatch; check units, masks, layer and builder provenance')
                    waveform_label_checks.append(dict(dataset_id=row.dataset_id,event_id=row.event_id,
                        sensor_id=meta['ids'][i],recomputed_vector_peak_mps2=recomputed,declared_mps2=declared,
                        status='verified_full_storage_interval; audit-only'))
            try:
                event = read_event(row, cfg, 1)
                check = prepare(event, cfg, 1, 'random')
                supported = check['inputs'] is not None
                reason = check['info']['reason']
            except NoViewRecords as exc:
                supported,reason = False,str(exc)
            for i, sensor in enumerate(meta['ids']):
                    records.append(dict(dataset_view=cfg['dataset_view'], dataset_id=row.dataset_id,
                        event_id=row.event_id, canonical_event_id=row.canonical_event_id, split=split,
                        sensor_id=sensor, physical_station_id=meta['physical_ids'][i],
                        row_index=int(meta['indices'][i]), sensor_class=meta['sensor_class'],
                        valid_input=bool(meta['good'][i]), valid_target=bool(meta['good'][i] and np.isfinite(meta['pga'][i])),
                        cohort_included=supported, reasons=';'.join(meta['reasons'][i]),cohort_exclusion_reason='' if supported else reason,
                        receiver_depth_status='not provided; source station height is proxy' if cfg['dataset_view']=='kiknet_borehole' else 'surface station height',
                        latitude=float(meta['coords'][i,0]),longitude=float(meta['coords'][i,1]),
                        height_km=float(meta['coords'][i,2]), pga_log10_mps2=float(meta['pga'][i])))
                    records[-1].update(record_start_sample=int(meta['starts'][i]),valid_n_samples=int(meta['lengths'][i]),
                        p_pick_sample=float(meta['picks'][i]),p_pick_source=meta['pick_source'][i],
                        record_axis_start_utc=meta['record_start_utc'])
            if not supported:
                exclusions.append(dict(dataset_id=row.dataset_id,event_id=row.event_id,split=split,
                    records=len(meta['ids']),reason=reason))
                continue
            kept.append([row.dataset_id, row.event_id])
            if split == 'train':
                values.extend(event['pga'][np.isfinite(event['pga'])].tolist())
            else:
                for time in cfg['realtime']['common_times']:
                    current = read_event(row, cfg, time)
                    for protocol in ('normal','random'):
                        _, mask, _ = build_window(current['waveform'], current['storage'], current['reference_sample'],
                                                   time, CAPABILITIES['diting_pretrained_frozen'])
                        selected, queries, roles, actual = geometry(current,mask.all(1),sampling_config(cfg),protocol)
                        if not len(selected) or not len(queries):
                            raise ValueError('Fixed validation support differs across times')
                        validation.extend(dict(event_id=row.event_id,dataset_id=row.dataset_id,elapsed_time=time,
                            geometry_protocol=actual,station_id=current['ids'][j],target_role=roles[j],
                            input_ids=json.dumps(current['ids'][selected].tolist())) for j in queries)
                diagnostic.append(dict(dataset_id=row.dataset_id,event_id=row.event_id,
                    priority=fingerprint([cfg['diagnostics']['seed'],row.event_id])))
        if not kept:
            pd.DataFrame(records).to_csv(destination/'view_records.csv.gz',index=False)
            pd.DataFrame(exclusions).to_csv(destination/'exclusions.csv',index=False)
            pd.DataFrame(filters).to_csv(destination/'view_filter_counts.csv',index=False)
            write_json(destination/'data_audit.json',dict(status='DATA_AUDIT_FAILED',dataset_view=cfg['dataset_view'],
                reason='Empty audited '+split+' cohort',model_not_loaded=True))
            raise ValueError('Empty audited '+split+' cohort')
        cohorts[split] = kept
    target = np.asarray(values, np.float64)
    if len(target) < 2 or not np.isfinite(target).all() or target.std() < 1e-8:
        raise ValueError('Degenerate train-only target normalization')
    normal = dict(enabled=True, mean=float(target.mean()), std=float(target.std()), count=len(target),
        unit='log10(m/s^2)', fitted_split='train', population_std=True,
        rule='all finite selected sensor labels in included training events', dataset_view=cfg['dataset_view'])
    frame = pd.DataFrame(records)
    frame.to_csv(destination/'view_records.csv.gz', index=False)
    pd.DataFrame(filters).to_csv(destination/'view_filter_counts.csv',index=False)
    pd.DataFrame(exclusions,columns=['dataset_id','event_id','split','records','reason']).to_csv(destination/'exclusions.csv',index=False)
    plan = pd.DataFrame(validation)
    plan.to_csv(destination/'validation_population.csv.gz', index=False)
    chosen = sorted(diagnostic, key=lambda x:x['priority'])[:cfg['diagnostics']['max_events']]
    write_json(destination/'diagnostic_plan.json', dict(seed=cfg['diagnostics']['seed'],
        rule='smallest SHA(seed,event); all configured times x protocols, first and full input sets',
        events=chosen, times=[1,20], protocols=['normal','random'], input_counts=['single','full']))
    summary = frame.groupby('split').agg(records=('sensor_id','size'),
        events=('event_id','nunique'),stations=('sensor_id','nunique'),valid_input_records=('valid_input','sum'),
        valid_target_records=('valid_target','sum'),cohort_records=('cohort_included','sum')).reset_index().to_dict('records')
    for part in summary:
        split = part['split']
        part['included_events'] = len(cohorts[split])
        selected_frame = frame.loc[frame.split.eq(split)&frame.cohort_included&frame.valid_input]
        part['included_input_records'] = int(len(selected_frame))
        part['included_stations'] = int(selected_frame.sensor_id.nunique())
        part['excluded_sensor_rows'] = int(part['records']-len(selected_frame))
    reason_counts = frame.reasons.str.split(';').explode()
    reason_counts = reason_counts[reason_counts.ne('')].value_counts().to_dict()
    effective = dict(audited_cohorts=cohorts, target_normalization=normal)
    lock = dict(status='DATA_AUDIT_PASS',dataset_view=cfg['dataset_view'],
        split_manifest=dict(path=cfg['data']['split_manifest'],sha256=sha256(cfg['data']['split_manifest']),
            rule='builder EVENT identity; shared split across all networks/layers; no remapping/random split',
            covered_events=len(known)), shards=shards,summary=summary,effective_config=effective,
        view_manifest_sha256=fingerprint(json.loads(frame.to_json(orient='records',double_precision=15))),
        view_manifest_file_sha256=sha256(destination/'view_records.csv.gz'),
        validation_population_sha256=fingerprint(plan.to_dict('records')),
        coordinates=cfg['coordinates'], normalization=normal,
        pga='log10(max_t sqrt(NS²+EW²+UD²) of resampled selected sensor acceleration [m/s²] + 1e-10)',
        upstream='offline demeaning/resampling/filtering; online causality not certified',
        receiver_depth='no verified downhole depth field; use station height km as explicit proxy',
        exclusions=len(exclusions),sensor_exclusion_reason_counts={str(k):int(v) for k,v in reason_counts.items()},
        filter_counts_sha256=sha256(destination/'view_filter_counts.csv'),
        cohort_sha256=fingerprint(cohorts),coordinate_sha256=fingerprint(cfg['coordinates']))
    lock['waveform_label_checks'] = waveform_label_checks
    lock['shard_metadata_observed'] = shard_semantics
    lock['hash_provenance'] = dict(mode='reuse identical stat-pinned records; hash additional shards' if reuse_path else 'fresh full shard SHA256',
        source_manifest=reuse_path or None,source_manifest_sha256=sha256(reuse_path) if reuse_path else None)
    write_json(destination/'data_audit.json', lock)
    return lock
