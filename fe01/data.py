"""Read-only Japan HDF5 adapter with frozen event splits and causal selection."""
import copy
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import sha256
from .sampling import TimeSampler, stable_rng
from .windows import CAPABILITIES, build_window, clock


def strings(values):
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in values])


def jst_timestamp(value):
    timestamp=pd.Timestamp(str(value))
    # The explicitly named *_jst fields define the zone of naive strings.
    if timestamp.tz is None: timestamp=timestamp.tz_localize('Asia/Tokyo')
    return timestamp.tz_convert('UTC')


def split_catalog(cfg, split, allow_test=False):
    if split == 'test' and not allow_test:
        raise ValueError('Held-out test requires explicit permission and protocol lock')
    table = pd.read_csv(cfg['data']['split_manifest'], dtype={'EVENT': str, 'event_id': str})
    event_key = 'EVENT' if 'EVENT' in table else 'event_id'
    table = table.rename(columns={event_key: 'event_id'})
    table['split'] = table['split'].replace({'dev': 'val', 'validation': 'val'})
    if not set(table['split']).issubset({'train','val','test'}):
        raise ValueError('Unexpected frozen split label')
    group_keys = ['dataset_index', 'event_id'] if 'dataset_index' in table else ['event_id']
    if table.duplicated(group_keys).any():
        raise ValueError('Repeated event identity / split leakage')
    if table.groupby('event_id')['split'].nunique().max() > 1:
        raise ValueError('Event crosses frozen splits')
    subset = table.loc[table['split'].eq(split)].copy()
    if subset.empty:
        raise ValueError('No events in requested split')
    def source(row):
        if 'source_data_path' in row and pd.notna(row['source_data_path']):
            name = Path(row['source_data_path']).name
            year = name.split('_')[1].split('.')[0]
        else:
            year = str(row.get('year', row['event_id'][:4]))
            name = f'japan_{year}.hdf5'
        return str(Path(cfg['data']['root']) / year / name)
    subset['hdf5_path'] = subset.apply(source, axis=1)
    subset['dataset_id'] = subset['hdf5_path'].map(lambda p: Path(p).name)
    if cfg.get('audited_cohorts', {}).get(split) is not None:
        allowed = {tuple(key) for key in cfg['audited_cohorts'][split]}
        subset = subset.loc[[ (r.dataset_id, r.event_id) in allowed for r in subset.itertuples() ]]
        if subset.empty:
            raise ValueError('Audited cohort is empty')
    return subset.sort_values(['dataset_id','event_id']).reset_index(drop=True)


def read_event(row, cfg, elapsed, reference='first_p_pick', full_for_audit=False):
    """Load only allowed waveform prefix; final PGA labels are scoring targets."""
    path = row['hdf5_path']
    with h5py.File(path, 'r') as handle:
        fs = float(np.asarray(handle['metadata/sampling_rate'][()]).reshape(-1)[0])
        if fs != cfg['data']['sampling_rate']:
            raise ValueError('Unexpected sampling rate; no implicit resampling')
        group = handle['data'][str(row['event_id'])]
        if group['waveforms'].ndim != 3 or group['waveforms'].shape[-1] != 3:
            raise ValueError('Expected stored station,time,NEZ acceleration')
        network = strings(group['source_network'][()])
        keep = np.flatnonzero(network == 'knt')
        spatial = None
        if cfg.get('spatial', {}).get('enabled'):
            spatial = pd.read_csv(cfg['spatial']['manifest'], dtype={'station_id': str})
            if spatial.station_id.duplicated().any():
                raise ValueError('Duplicate spatial holdout station')
            spatial = spatial.set_index('station_id')
            all_ids = strings(group['station_codes'][keep])
            if not set(all_ids).issubset(set(spatial.index)):
                raise ValueError('Unregistered station in frozen spatial holdout')
            if row['split'] == 'train':
                registered = spatial.loc[all_ids]
                permitted = registered.spatial_split.eq('train').to_numpy()
                permitted &= ~registered.buffer_excluded.astype(str).str.lower().eq('true').to_numpy()
                # Remove before reading labels, waveforms or reference picks.
                keep = keep[permitted]
        if not len(keep):
            raise ValueError('Frozen event has no KNET stations')
        coords = group['coords'][keep].astype(np.float32)
        picks = group['p_picks'][keep].astype(np.float64)
        starts = group['record_start_sample'][keep].astype(np.int64)
        lengths = group['valid_n_samples'][keep].astype(np.int64)
        ids = strings(group['station_codes'][keep])
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate sensor identity in KNET event')
        if not np.isfinite(coords).all() or np.any(starts < 0) or np.any(lengths <= 0):
            raise ValueError('Invalid coordinates/storage support')
        pick_valid = np.isfinite(picks) & (picks > 0)
        if not pick_valid.any():
            raise ValueError('No event reference P pick')
        reference_sample = float(np.min(picks[pick_valid]))
        record_start = None
        if 'record_start_time_jst' in group:
            timestamps = [jst_timestamp(v).timestamp() for v in strings(group['record_start_time_jst'][keep])]
            candidates = np.asarray(timestamps) - starts / fs
            if np.ptp(candidates) > 1 / fs + 1e-5:
                raise ValueError('Global record time axes disagree')
            record_start = float(candidates[0])
        if reference == 'origin_time':
            if record_start is None or 'Origin_Time(JST)' not in row:
                raise ValueError('Origin replay requires verified absolute record and origin timestamps')
            origin = jst_timestamp(row['Origin_Time(JST)'])
            reference_sample = (origin.timestamp() - record_start) * fs
        current, cutout = clock(reference_sample, elapsed, fs)
        total_samples = group['waveforms'].shape[1]
        stop = total_samples if full_for_audit else max(0, min(total_samples, cutout))
        waveform = group['waveforms'][keep, :stop, :].transpose(0,2,1).astype(np.float32)
        indices = np.arange(stop)
        storage = (indices[None,:] >= starts[:,None]) & (indices[None,:] < (starts+lengths)[:,None])
        if 'waveform_sample_valid' in group:
            explicit = group['waveform_sample_valid'][keep, :stop]
            if explicit.ndim == 3:
                explicit = explicit.all(axis=-1)
            storage &= explicit
        if 'component_valid' in group:
            storage &= group['component_valid'][keep].all(axis=-1)[:,None]
        pga = group['pga'][keep].astype(np.float32)
    input_allowed = np.ones(len(ids), dtype=bool)
    query_allowed = np.ones(len(ids), dtype=bool)
    if spatial is not None:
        selected_spatial = spatial.loc[ids]
        buffer = selected_spatial.buffer_excluded.astype(str).str.lower().eq('true').to_numpy()
        spatial_split = selected_spatial.spatial_split.to_numpy()
        input_allowed = (spatial_split == 'train') & ~buffer
        query_allowed = (spatial_split == row['split']) & ~buffer
    return dict(event_id=str(row['event_id']), dataset_id=row['dataset_id'], split=row['split'],
                waveform=waveform, storage=storage, coords=coords, picks=picks, ids=ids,
                pga=pga, reference_sample=reference_sample, current_sample=current,
                cutout_exclusive=cutout, total_record_samples=total_samples,
                record_start_utc=record_start, reference=reference,
                magnitude=float(row['Magnitude']),
                event_location=np.asarray([row['Latitude'],row['Longitude'],row['DEPTH']],dtype=np.float32),
                input_allowed=input_allowed,query_allowed=query_allowed)


def geometry(event, mask, cfg, protocol, epoch=0, draw=0, replay=False):
    current = event['current_sample']
    eligible = np.flatnonzero((event['picks'] > 0) & (event['picks'] <= current) & mask.any(-1)
                             & event.get('input_allowed', np.ones(len(mask),dtype=bool)))
    query_universe = np.flatnonzero(np.isfinite(event['pga']) & event.get('query_allowed',np.ones(len(mask),dtype=bool)))
    rng = stable_rng(cfg['sampling_seed'], event['dataset_id'],event['event_id'],epoch,
                     0 if replay else draw, 'geometry')
    is_random = protocol == 'random' or (protocol == 'mixed' and rng.random() < 0.5)
    if is_random:
        count = int(rng.choice(cfg['geometry']['random_station_counts']))
        # One stable ordering per replay case, including not-yet-arrived stations.
        priority = rng.permutation(len(event['ids']))
        available = set(eligible.tolist())
        selected = np.asarray([i for i in priority if i in available][:count], dtype=int)
        if len(query_universe) and set(query_universe).issubset(set(selected)):
            selected = selected[selected != int(query_universe[-1])]
    else:
        selected = eligible[np.argsort(event['picks'][eligible],kind='stable')][:cfg['model_params']['max_stations']]
    selected = selected[np.argsort(event['picks'][selected],kind='stable')]
    selected = selected[:cfg['model_params']['max_stations']]
    roles = np.where(np.isin(np.arange(len(event['ids'])),selected),'observed_input',
                     np.where((event['picks'] > 0)&(event['picks'] <= current), 'triggered_noninput','untriggered_noninput'))
    if is_random:
        query_universe = query_universe[~np.isin(query_universe,selected)]
    return selected, query_universe, roles, 'random' if is_random else 'normal'


def balanced_targets(universe, roles, count, rng, random=False):
    weights = [0.0 if random else 0.3, 0.2, 0.5]
    names = ['observed_input','triggered_noninput','untriggered_noninput']
    selected = []
    for name, weight in zip(names,weights):
        pool = universe[roles[universe] == name]
        take = min(len(pool), int(round(count*weight/sum(weights))))
        if take:
            selected.extend(rng.choice(pool,take,replace=False).tolist())
    remaining = np.setdiff1d(universe, selected)
    if len(selected) < count and len(remaining):
        selected.extend(rng.choice(remaining,min(count-len(selected),len(remaining)),replace=False).tolist())
    return np.asarray(selected[:count],dtype=int)


def prepare_sample(event, cfg, elapsed, protocol='normal', epoch=0, draw=0,
                   queries=None, replay=False):
    capability = replace(CAPABILITIES[cfg['model_family']], native_n_samples=cfg['native_n_samples'],
                         pre_p_seconds=cfg['window']['pre_p_seconds'])
    raw, channel_mask, info = build_window(event['waveform'],event['storage'],event['reference_sample'],
                                          elapsed,capability,cfg['window']['protocol'])
    info.update(event_id=event['event_id'], dataset_id=event['dataset_id'], split=event['split'],
                time_reference=event['reference'], units='log10(m/s^2)')
    if info['status'] != 'supported':
        return dict(inputs=None, labels=None, info=info)
    mask = channel_mask.all(axis=1)
    raw = np.where(mask[:,None,:],raw,0)
    selected, universe, roles, actual_protocol = geometry(event,mask,cfg,protocol,epoch,draw,replay)
    info['geometry_protocol'] = actual_protocol
    info['input_ids'] = event['ids'][selected].tolist()
    info['input_coords'] = event['coords'][selected].tolist()
    if not len(selected):
        info.update(status='empty',reason='no_input')
        return dict(inputs=None,labels=None,info=info)
    if queries is None:
        rng = stable_rng(cfg['sampling_seed'],event['dataset_id'],event['event_id'],epoch,draw,'targets')
        queries = balanced_targets(universe,roles,cfg['model_params']['n_pga_targets'],rng,actual_protocol=='random')
    queries = np.asarray(queries,dtype=int)
    if not len(queries):
        info.update(status='empty',reason='no_target')
        return dict(inputs=None,labels=None,info=info)
    s,q = cfg['model_params']['max_stations'], cfg['model_params']['n_pga_targets']
    if len(queries)>q:
        raise ValueError('Query chunk exceeds common downstream configured capacity')
    wave = np.zeros((s,3,cfg['native_n_samples']),dtype=np.float32)
    sample_mask = np.zeros((s,cfg['native_n_samples']),dtype=bool)
    coords = np.zeros((s,3),dtype=np.float32)
    valid = np.zeros(s,dtype=bool)
    target_coords = np.zeros((q,3),dtype=np.float32)
    target_valid = np.zeros(q,dtype=bool)
    target = np.zeros(q,dtype=np.float32)
    offset = np.asarray([37,140,0],dtype=np.float32)
    wave[:len(selected)] = raw[selected]
    sample_mask[:len(selected)] = mask[selected]
    coords[:len(selected)] = event['coords'][selected]-offset
    valid[:len(selected)] = True
    target_coords[:len(queries)] = event['coords'][queries]-offset
    target_valid[:len(queries)] = np.isfinite(event['pga'][queries])
    target[:len(queries)] = np.where(target_valid[:len(queries)],event['pga'][queries],0)
    inputs = [torch.from_numpy(x) for x in (wave,coords,valid,target_coords,target_valid,sample_mask)]
    labels = [torch.tensor([event['magnitude']],dtype=torch.float32),
              torch.from_numpy(event['event_location']-offset),torch.from_numpy(target[:,None])]
    info.update(query_ids=event['ids'][queries].tolist(), query_indices=queries.tolist(),
                target_roles=roles[queries].tolist(),query_coords=event['coords'][queries].tolist(),
                input_count=len(selected),event_location=event['event_location'].tolist(),magnitude=event['magnitude'],
                selected_valid_seconds=[info['valid_seconds_by_station'][i] for i in selected],
                absolute_decision_utc=(event['record_start_utc']+info['current_sample']/100 if event['record_start_utc'] is not None else None))
    return dict(inputs=inputs, labels=labels, info=info)


class TrainingDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.catalog = split_catalog(cfg,'train')
        self.epoch = 0
        cap = CAPABILITIES[cfg['model_family']]
        maximum=9000 if cfg['window']['protocol']=='native_rolling_v2' else cap.max_elapsed_sample()
        self.sampler = TimeSampler(max_sample=maximum,
                    bins=cfg['realtime']['bins'],probabilities=cfg['realtime']['probabilities'])

    def __len__(self):
        return len(self.catalog)*self.cfg['realtime']['draws_per_event']

    def __getitem__(self,index):
        event_index,draw = divmod(index,self.cfg['realtime']['draws_per_event'])
        row = self.catalog.iloc[event_index]
        elapsed = self.sampler.sample(row['dataset_id'],row['event_id'],self.epoch,draw,self.cfg['sampling_seed'])
        event = read_event(row,self.cfg,elapsed)
        result = prepare_sample(event,self.cfg,elapsed,'mixed',self.epoch,draw)
        result['info'].update(sampling_epoch=self.epoch,draw_index=draw,cohort_event_index=event_index)
        # Never silently advance to another event or change effective updates.
        if result['inputs'] is None:
            raise ValueError(f'Training plan failure {row["event_id"]}: {result["info"]["reason"]}; audit cohort before training')
        return result


def collate(samples):
    return dict(inputs=[torch.stack([s['inputs'][i] for s in samples]) for i in range(6)],
                labels=[torch.stack([s['labels'][i] for s in samples]) for i in range(3)],
                info=[s['info'] for s in samples])
