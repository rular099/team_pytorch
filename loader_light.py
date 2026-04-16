import numpy as np
import pandas as pd
import h5py
import os
import time

EVENT_KEY_CANDIDATES = ['KiK_File', '#EventID', 'EVENT']
MAGNITUDE_KEY_ALIASES = {
    'M_J': ['Magnitude'],
    'MA': ['Magnitude'],
    'Magnitude': ['M_J', 'MA'],
}


class TrainDevTestSplitter:
    @staticmethod
    def run_method(event_metadata, name, shuffle_train_dev, parts):
        mask = np.zeros(len(event_metadata), dtype=bool)

        split_methods = {'test_2016': TrainDevTestSplitter.test_2016,
                         'test_2011': TrainDevTestSplitter.test_2011,
                         'no_test': TrainDevTestSplitter.no_test}

        if name is None or name == '':
            test_set = TrainDevTestSplitter.default(event_metadata)
        elif name in split_methods:
            test_set = split_methods[name](event_metadata)
        else:
            raise ValueError(f'Unknown split function: {name}')

        b1 = int(0.7 / 0.8 * np.sum(~test_set))
        train_set = np.zeros(np.sum(~test_set), dtype=bool)
        train_set[:b1] = True

        if shuffle_train_dev:
            np.random.seed(len(event_metadata))  # The same length of data always gets split the same way
            np.random.shuffle(train_set)

        if parts[0] and parts[1]:
            mask[~test_set] = True
        elif parts[0]:
            mask[~test_set] = train_set
        elif parts[1]:
            mask[~test_set] = ~train_set
        if parts[2]:
            mask[test_set] = True

        return mask

    @staticmethod
    def default(event_metadata):
        test_set = np.zeros(len(event_metadata), dtype=bool)
        b2 = int(0.8 * len(event_metadata))
        test_set[b2:] = True
        return test_set

    @staticmethod
    def test_2016(event_metadata):
        test_set = np.array([x[:4] == '2016' for x in event_metadata['Time']])
        return test_set

    @staticmethod
    def test_2011(event_metadata):
        test_set = np.array([x[:4] == '2011' for x in event_metadata['Origin_Time(JST)']])
        return test_set


    @staticmethod
    def no_test(event_metadata):
        return np.zeros(len(event_metadata), dtype=bool)

def _read_event_metadata_h5(data_path):
    """Read event metadata from HDF5, supporting both pytables and raw h5py formats."""
    with h5py.File(data_path, 'r') as f:
        em = f['metadata/event_metadata']
        if isinstance(em, h5py.Group):
            # Raw h5py format: each column is a separate dataset
            cols = {}
            for col_name in em.keys():
                vals = em[col_name][()]
                if vals.dtype.kind == 'S':  # byte strings → str
                    vals = np.array([v.decode() for v in vals])
                cols[col_name] = vals
            return pd.DataFrame(cols)
    # Fallback: pytables format
    return pd.read_hdf(data_path, 'metadata/event_metadata')


def _read_metadata_table_h5(data_path, table_name):
    with h5py.File(data_path, 'r') as f:
        table = f[f'metadata/{table_name}']
        if isinstance(table, h5py.Group):
            cols = {}
            for col_name in table.keys():
                vals = table[col_name][()]
                if vals.dtype.kind == 'S':
                    vals = np.array([v.decode() for v in vals])
                cols[col_name] = vals
            return pd.DataFrame(cols)
    return pd.read_hdf(data_path, f'metadata/{table_name}')


def detect_event_key(columns):
    for event_key in EVENT_KEY_CANDIDATES:
        if event_key in columns:
            return event_key
    raise ValueError(f'Unable to find event key column in {list(columns)}')


def resolve_target_key(columns, requested_key):
    if requested_key is None or requested_key in columns:
        return requested_key
    for alias in MAGNITUDE_KEY_ALIASES.get(requested_key, []):
        if alias in columns:
            return alias
    return requested_key


def _filter_rows_present_in_hdf5(event_metadata, data_path, event_key, decimate=1):
    with h5py.File(data_path, 'r') as f:
        counts = {}
        for event_name, g_event in f['data'].items():
            if 'waveforms' in g_event:
                counts[str(event_name)] = int(g_event['waveforms'].shape[0])

    event_names = event_metadata[event_key].astype(str)
    present_counts = event_names.map(counts)
    mask = present_counts.notna()
    if 'wave_idx' in event_metadata.columns:
        wave_idx = pd.to_numeric(event_metadata['wave_idx'], errors='coerce')
        mask &= (wave_idx.isna() | (wave_idx < present_counts))
    return event_metadata.loc[mask].reset_index(drop=True)


def build_event_metadata(data_path, event_metadata_path, overwrite_sampling_rate=None, min_stalta_ratio_at_pick=None):
    t0 = time.time()
    metadata = {}
    with h5py.File(data_path, 'r') as f:
        for key in f['metadata'].keys():
            if key in ('event_metadata', 'station_metadata'):
                continue
            metadata[key] = f['metadata'][key][()]

        if overwrite_sampling_rate is not None:
            if metadata['sampling_rate'] % overwrite_sampling_rate != 0:
                raise ValueError(f'Overwrite sampling ({overwrite_sampling_rate}) rate must be true divisor of sampling'
                                 f' rate ({metadata["sampling_rate"]})')
            decimate = metadata['sampling_rate'] // overwrite_sampling_rate
            metadata['sampling_rate'] = overwrite_sampling_rate
        else:
            decimate = 1

        if 'station_metadata' in f['metadata']:
            event_metadata = _read_metadata_table_h5(data_path, 'station_metadata')
        else:
            event_metadata = _read_event_metadata_h5(data_path)
            event_key = detect_event_key(event_metadata.columns)
            contained = []
            n_rec_per_event = []
            for _, event in event_metadata.iterrows():
                event_name = str(event[event_key])
                if event_name not in f['data']:
                    contained += [False]
                    continue
                contained += [True]
                g_event = f['data'][event_name]
                if 'waveforms' in g_event:
                    cur_waveform = g_event['waveforms'][:, ::decimate, :]
                    n_rec_per_event.append(cur_waveform.shape[0])
            event_metadata = event_metadata[contained].reset_index(drop=True)
            event_metadata = event_metadata.loc[event_metadata.index.repeat(n_rec_per_event)].reset_index(drop=True)
            wave_idx_per_event = np.concatenate([np.arange(i) for i in n_rec_per_event]) if n_rec_per_event else np.array([], dtype=int)
            event_metadata['wave_idx'] = wave_idx_per_event

    event_key = detect_event_key(event_metadata.columns)
    before_rows = len(event_metadata)
    event_metadata = _filter_rows_present_in_hdf5(event_metadata, data_path, event_key, decimate=decimate)
    if min_stalta_ratio_at_pick is not None and 'stalta_ratio_at_pick' in event_metadata.columns:
        event_metadata = event_metadata[event_metadata['stalta_ratio_at_pick'] >= min_stalta_ratio_at_pick].reset_index(drop=True)
    event_metadata.to_csv(event_metadata_path, index=False)
    dt = time.time() - t0
    print(f'Built metadata cache {event_metadata_path} with {len(event_metadata)} rows '
          f'(from {before_rows}) in {dt:.1f}s')
    return event_metadata

def load_events(data_paths, event_metadata_path='./event_metadata.csv', limit=None, parts=None, shuffle_train_dev=False, custom_split=None, data_keys=None,
                overwrite_sampling_rate=None, min_mag=None, mag_key=None, decimate_events=None,
                min_stalta_ratio_at_pick=0.1):
    if min_mag is not None and mag_key is None:
        raise ValueError('mag_key needs to be set to enforce magnitude threshold')
    if isinstance(data_paths, str):
        data_paths = [data_paths]
    if len(data_paths) > 1:
        raise NotImplementedError('Loading partitioned data is currently not supported')
    data_path = data_paths[0]

    # Derive cache path from data_path to avoid stale CSV when switching datasets
    import hashlib
    cache_token = f'{os.path.abspath(data_path)}|stationmeta_v3|{overwrite_sampling_rate}|{min_stalta_ratio_at_pick}'
    data_hash = hashlib.md5(cache_token.encode()).hexdigest()[:8]
    base_dir = os.path.dirname(os.path.abspath(event_metadata_path)) or os.getcwd()
    event_metadata_path = os.path.join(base_dir, f'station_cache_{data_hash}.csv')

    if not os.path.exists(event_metadata_path):
        print(f'Building station metadata cache for {os.path.basename(data_path)} ...')
        build_event_metadata(data_path, event_metadata_path, overwrite_sampling_rate,
                             min_stalta_ratio_at_pick=min_stalta_ratio_at_pick)
    else:
        print(f'Using station metadata cache {event_metadata_path}')
    event_metadata = pd.read_csv(event_metadata_path)
    resolved_mag_key = resolve_target_key(event_metadata.columns, mag_key)
    if min_mag is not None:
        event_metadata = event_metadata[event_metadata[resolved_mag_key] >= min_mag]
    event_key = detect_event_key(event_metadata.columns)

    if limit:
        event_metadata = event_metadata.iloc[:limit]

    # Split at event level to avoid data leakage (same event in train+dev/test).
    # event_metadata is station-level (one row per station per event), so we
    # deduplicate to event level, run the splitter, then map back.
    if parts:
        unique_events = event_metadata.drop_duplicates(subset=event_key, keep='first')
        event_mask = TrainDevTestSplitter.run_method(unique_events, custom_split, shuffle_train_dev, parts=parts)
        keep_events = set(unique_events.loc[event_mask, event_key])
        event_metadata = event_metadata[event_metadata[event_key].isin(keep_events)]

    if decimate_events is not None:
        event_metadata = event_metadata.iloc[::decimate_events]

    metadata = {}
    data = {}

    with h5py.File(data_path, 'r') as f:
        for key in f['metadata'].keys():
            if key in ('event_metadata', 'station_metadata'):
                continue
            metadata[key] = f['metadata'][key][()]

        if overwrite_sampling_rate is not None:
            if metadata['sampling_rate'] % overwrite_sampling_rate != 0:
                raise ValueError(f'Overwrite sampling ({overwrite_sampling_rate}) rate must be true divisor of sampling'
                                 f' rate ({metadata["sampling_rate"]})')
            decimate = metadata['sampling_rate'] // overwrite_sampling_rate
            metadata['sampling_rate'] = overwrite_sampling_rate
        else:
            decimate = 1

    event_metadata = _filter_rows_present_in_hdf5(event_metadata, data_path, event_key, decimate=decimate)
    if min_stalta_ratio_at_pick is not None and 'stalta_ratio_at_pick' in event_metadata.columns:
        event_metadata = event_metadata[event_metadata['stalta_ratio_at_pick'] >= min_stalta_ratio_at_pick].reset_index(drop=True)

    metadata['resolved_mag_key'] = resolved_mag_key
    metadata['event_key'] = event_key
    metadata['min_stalta_ratio_at_pick'] = min_stalta_ratio_at_pick

    return event_metadata, data, metadata
