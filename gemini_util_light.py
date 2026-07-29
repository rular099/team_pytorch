import glob
import h5py
import numpy as np
import pandas as pd
import obspy
import os
from obspy import UTCDateTime
import warnings
import time
from tqdm import tqdm
from scipy.stats import norm

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import loader_light

# warnings.simplefilter("ignore", UserWarning)

D2KM = 111.19492664455874


class _EmptySample(Exception):
    """Internal signal: this index has no valid station after cutout; skip it."""
    pass


class DPKPriorCache:
    """Row-aligned reader for precomputed DPK token priors."""

    LEVEL_ORDER = ("f2", "f3", "f4", "x")

    def __init__(self, path, source="dpk_finetuned", mode="event",
                 token_floor=1e-4, missing_policy="error"):
        self.path = os.fspath(path)
        self.source = str(source)
        self.mode = str(mode)
        self.token_floor = float(token_floor)
        self.missing_policy = str(missing_policy or "error").lower()
        if self.missing_policy not in ("error", "ones"):
            raise ValueError(
                "dpk_prior_cache missing_policy must be 'error' or 'ones', "
                f"got {self.missing_policy!r}."
            )
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"DPK prior cache not found: {self.path}")
        self._h5 = None
        with h5py.File(self.path, "r") as h5:
            required = [
                f"priors/{self.source}/{self.mode}/{level}"
                for level in self.LEVEL_ORDER
            ]
            missing = [name for name in required if name not in h5]
            if missing:
                raise KeyError(
                    f"DPK prior cache {self.path} is missing datasets: {missing}"
                )
            sample_station_keys = h5["index/sample_station_key"][:]
            if "index/event_station_time_key" in h5:
                event_station_time_keys = h5["index/event_station_time_key"][:]
            else:
                event_station_time_keys = []
        self.row_by_sample_station_key = self._build_first_row_index(
            sample_station_keys,
            normalizer=self._normalize_sample_station_key,
        )
        self.row_by_event_station_time_key = self._build_first_row_index(
            event_station_time_keys,
            normalizer=self._normalize_event_station_time_key,
        )
        (
            self.current_samples_by_event_key,
            self.original_stations_by_event_time_key,
        ) = self._build_event_time_indexes(event_station_time_keys)
        self.row_by_key = self.row_by_sample_station_key
        self.level_lengths = None
        self.max_level_len = None

    @staticmethod
    def _decode_key(key):
        return key.decode("utf-8") if isinstance(key, bytes) else str(key)

    @classmethod
    def _build_first_row_index(cls, keys, normalizer=None):
        row_by_key = {}
        for i, key in enumerate(keys):
            decoded = normalizer(key) if normalizer is not None else cls._decode_key(key)
            if decoded and decoded not in row_by_key:
                row_by_key[decoded] = int(i)
        return row_by_key

    @classmethod
    def _normalize_sample_station_key(cls, key):
        decoded = cls._decode_key(key)
        parts = decoded.split("|")
        if len(parts) < 4:
            return decoded
        split, dataset_id, sample_index, station_slot = parts[:4]
        try:
            dataset_id = int(round(float(dataset_id)))
            sample_index = int(round(float(sample_index)))
            station_slot = int(round(float(station_slot)))
        except (TypeError, ValueError):
            return decoded
        return f"{split}|{dataset_id}|{sample_index}|{station_slot}"

    @classmethod
    def _normalize_event_station_time_key(cls, key):
        decoded = cls._decode_key(key)
        parts = decoded.split("|")
        if len(parts) < 5:
            return decoded
        split, dataset_id, event_id, current_sample, original_station = parts[:5]
        try:
            dataset_id = int(round(float(dataset_id)))
            current_sample = int(round(float(current_sample)))
            original_station = int(round(float(original_station)))
        except (TypeError, ValueError):
            return decoded
        return f"{split}|{dataset_id}|{event_id}|{current_sample}|{original_station}"

    @classmethod
    def _build_event_time_indexes(cls, keys):
        samples = {}
        stations = {}
        for key in keys:
            parts = cls._decode_key(key).split("|")
            if len(parts) < 5:
                continue
            split, dataset_id, event_id, current_sample, original_station = parts[:5]
            try:
                dataset_id = int(round(float(dataset_id)))
                current_sample = int(round(float(current_sample)))
                original_station = int(round(float(original_station)))
            except (TypeError, ValueError):
                continue
            event_key = (str(split), dataset_id, str(event_id))
            time_key = (str(split), dataset_id, str(event_id), current_sample)
            samples.setdefault(event_key, set()).add(current_sample)
            stations.setdefault(time_key, set()).add(original_station)
        sample_index = {
            key: np.asarray(sorted(value), dtype=np.int64)
            for key, value in samples.items()
            if value
        }
        station_index = {
            key: frozenset(value)
            for key, value in stations.items()
            if value
        }
        return sample_index, station_index

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def _ensure_shapes(self):
        if self.level_lengths is not None:
            return
        h5 = self._file()
        self.level_lengths = [
            int(h5[f"priors/{self.source}/{self.mode}/{level}"].shape[1])
            for level in self.LEVEL_ORDER
        ]
        self.max_level_len = max(self.level_lengths)

    @staticmethod
    def _optional_int(value):
        if value is None:
            return None
        try:
            as_float = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(as_float):
            return None
        return int(round(as_float))

    def nearest_current_sample(self, split, dataset_id, event_id, current_sample):
        current_sample = self._optional_int(current_sample)
        if current_sample is None:
            return None
        samples = self.current_samples_by_event_key.get(
            (str(split), int(dataset_id), str(event_id))
        )
        if samples is None or samples.size == 0:
            return None
        pos = int(np.searchsorted(samples, current_sample))
        candidates = []
        if pos < samples.size:
            candidates.append(int(samples[pos]))
        if pos > 0:
            candidates.append(int(samples[pos - 1]))
        return min(candidates, key=lambda value: abs(value - current_sample))

    def station_available_mask(self, split, dataset_id, event_id, current_sample,
                               original_station_indices, station_valid=None):
        current_sample = self._optional_int(current_sample)
        original_station_indices = np.asarray(original_station_indices)
        if station_valid is None:
            station_valid = np.ones(original_station_indices.shape, dtype=bool)
        else:
            station_valid = np.asarray(station_valid, dtype=bool)
        out = np.zeros(original_station_indices.shape, dtype=bool)
        if current_sample is None:
            return out
        available = self.original_stations_by_event_time_key.get(
            (str(split), int(dataset_id), str(event_id), current_sample),
            frozenset(),
        )
        if not available:
            return out
        for station_slot, is_valid in enumerate(station_valid):
            if not is_valid or station_slot >= len(original_station_indices):
                continue
            original_idx = self._optional_int(original_station_indices[station_slot])
            if original_idx is not None and original_idx >= 0 and original_idx in available:
                out[station_slot] = True
        return out

    def lookup_sample(self, split, dataset_id, sample_index, station_valid,
                      event_id=None, realtime_current_sample=None,
                      original_station_indices=None):
        self._ensure_shapes()
        station_valid = np.asarray(station_valid, dtype=bool)
        original_station_indices = (
            None if original_station_indices is None
            else np.asarray(original_station_indices)
        )
        current_sample = self._optional_int(realtime_current_sample)
        event_station_time_lookup_requested = (
            event_id is not None
            and current_sample is not None
            and original_station_indices is not None
        )
        if event_station_time_lookup_requested and not self.row_by_event_station_time_key:
            raise KeyError(
                f"DPK prior cache {self.path} has no index/event_station_time_key "
                "dataset; realtime cached-prior lookup cannot safely fall back to "
                "sample_index/station_slot."
            )
        use_event_station_time_key = event_station_time_lookup_requested
        out = np.ones(
            (station_valid.shape[0], len(self.LEVEL_ORDER), self.max_level_len),
            dtype=np.float32,
        )
        h5 = self._file()
        misses = []
        for station_slot, is_valid in enumerate(station_valid):
            if not is_valid:
                continue
            if use_event_station_time_key:
                if station_slot >= len(original_station_indices):
                    misses.append(
                        f"{split}|{int(dataset_id)}|{event_id}|"
                        f"{current_sample}|<missing_original_station_index:{station_slot}>"
                    )
                    continue
                original_idx = self._optional_int(original_station_indices[station_slot])
                if original_idx is None or original_idx < 0:
                    misses.append(
                        f"{split}|{int(dataset_id)}|{event_id}|"
                        f"{current_sample}|<invalid_original_station_index:{station_slot}>"
                    )
                    continue
                key = (
                    f"{split}|{int(dataset_id)}|{event_id}|"
                    f"{current_sample}|{original_idx}"
                )
                row = self.row_by_event_station_time_key.get(key)
            else:
                key = f"{split}|{int(dataset_id)}|{int(sample_index)}|{int(station_slot)}"
                row = self.row_by_sample_station_key.get(key)
            if row is None:
                misses.append(key)
                continue
            for level_idx, (level, length) in enumerate(zip(self.LEVEL_ORDER, self.level_lengths)):
                values = h5[f"priors/{self.source}/{self.mode}/{level}"][row]
                out[station_slot, level_idx, :length] = np.asarray(values, dtype=np.float32)
        if misses and self.missing_policy == "error":
            preview = ", ".join(misses[:3])
            suffix = "" if len(misses) <= 3 else f", ... ({len(misses)} misses)"
            raise KeyError(
                f"DPK prior cache miss in {self.path}: {preview}{suffix}. "
                "Check split/dataset_id/event_id/realtime_current_sample/"
                "original_station_index alignment and deterministic realtime sampling."
            )
        return out

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass

def setup_distributed(backend='nccl', init_method='env://'):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method=init_method)
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0

def cleanup():
    dist.destroy_process_group()

def distributed_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0

def load_checkpoint(model, optimizer, scheduler, device, is_ddp, checkpoint_path):
    """加载 checkpoint 并返回下一个 epoch"""
    if not os.path.exists(checkpoint_path):
        return 0  # 从 epoch 0 开始

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 模型参数
    if is_ddp:
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    # 优化器
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # scheduler
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming from checkpoint '{checkpoint_path}', starting at epoch {start_epoch}")
    return start_epoch

def resample_trace(trace, sampling_rate):
    if trace.stats.sampling_rate == sampling_rate:
        return
    if trace.stats.sampling_rate % sampling_rate == 0:
        trace.decimate(int(trace.stats.sampling_rate / sampling_rate))
    else:
        trace.resample(sampling_rate)


def _safe_int_array(values):
    arr = np.asarray(values)
    if arr.size == 0:
        return np.zeros(0, dtype=int)
    return np.rint(arr).astype(int)


def _select_wave_idx_rows(event_df, g_event):
    if 'wave_idx' not in event_df.columns:
        return slice(None)
    wave_idx = _safe_int_array(event_df['wave_idx'].values)
    if wave_idx.size == 0:
        return wave_idx
    max_idx = g_event['waveforms'].shape[0]
    return wave_idx[wave_idx < max_idx]


def _crop_aligned_event_window(waveforms, p_picks, target_length, sampling_rate, noise_seconds, rng=None):
    if waveforms.shape[1] <= target_length:
        if waveforms.shape[1] == target_length:
            return waveforms, p_picks, 0
        padded = np.zeros((waveforms.shape[0], target_length, waveforms.shape[2]), dtype=waveforms.dtype)
        padded[:, :waveforms.shape[1], :] = waveforms
        return padded, p_picks, 0

    valid_picks = p_picks[np.logical_and(np.isfinite(p_picks), p_picks > 0)]
    if valid_picks.size == 0:
        start = 0
    else:
        pre_samples = int(round(noise_seconds * sampling_rate))
#        start = int(np.floor(np.min(valid_picks))) - pre_samples
        if rng is None:
            anchor_pick = np.random.choice(valid_picks)
        else:
            anchor_pick = rng.choice(valid_picks)
        start = int(np.floor(anchor_pick)) - pre_samples
        start = max(0, start)
    max_start = waveforms.shape[1] - target_length
    start = min(start, max_start)
    end = start + target_length
    return waveforms[:, start:end, :], p_picks - start, start


class DataGenerator(Dataset):
    def __init__(self, event_metadata, metadata, data_path, generator_params, cutout=None, sliding_window=False, windowlen=10000, data_keys=None, overwrite_sampling_rate=None,
                 shuffle=True, label_smoothing=False, oversample=1):
        self.event_metadata = event_metadata
        self.metadata = metadata
        self.data_path = data_path
        requested_key = generator_params.get('key', 'Magnitude')
        self.target_key = loader_light.resolve_target_key(event_metadata.columns, requested_key)
        self.data_keys = data_keys
        self.overwrite_sampling_rate = overwrite_sampling_rate
        self.noise_seconds = generator_params.get('noise_seconds', 5)

        self.dist = None
        if cutout is None:
            self.cutout = None
        else:
            self.cutout = tuple(int(round(x)) for x in cutout)
        self.sliding_window = sliding_window  # If true, selects sliding windows instead of cutout. Uses cutout as values for end of window.
        self.windowlen = int(round(windowlen))  # Length of window for sliding window
        self.shuffle = shuffle
        self.oversample = oversample
        self.indexes = np.arange(len(self.event_metadata))
        self.label_smoothing = label_smoothing
        self.event_key = loader_light.detect_event_key(event_metadata.columns)
        if self.overwrite_sampling_rate is not None:
            if self.metadata['sampling_rate'] % self.overwrite_sampling_rate != 0:
                raise ValueError(f'Overwrite sampling ({self.overwrite_sampling_rate}) rate must be true divisor of sampling'
                                 f' rate ({self.metadata["sampling_rate"]})')
            self.decimate = self.metadata['sampling_rate'] // self.overwrite_sampling_rate
            self.metadata['sampling_rate'] = self.overwrite_sampling_rate
        else:
            self.decimate = 1
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.oversample * len(self.event_metadata)))

    def __getitem__(self, index):
        index = self.indexes[index]
        with h5py.File(self.data_path, 'r') as f:
            event = self.event_metadata.iloc[index]
            event_name = str(event[self.event_key])
            g_event = f['data'][event_name]
            data = {}
            wave_idx = int(event['wave_idx']) if 'wave_idx' in event else 0
            for key in g_event:
                if self.data_keys is not None and key not in self.data_keys:
                    continue
                if key not in data:
                    data[key] = []
                if key == 'waveforms':
                    cur_waveform = g_event[key][wave_idx:wave_idx+1, ::self.decimate, :]
                    cur_waveform -= np.mean(cur_waveform, axis=1, keepdims=True)
                    data[key] += [cur_waveform]
                else:
                    values = g_event[key][()]
                    if np.ndim(values) > 0 and values.shape[0] > wave_idx:
                        values = values[wave_idx:wave_idx+1]
                    data[key] += [values]
                if key == 'p_picks':
                    data[key][-1] //= self.decimate

        X = data['waveforms'][0]
        picks = data['p_picks'][0] if 'p_picks' in data else np.zeros((1,), dtype=int)
        X, picks, _ = _crop_aligned_event_window(X, _safe_int_array(picks), 10000,
                                                 self.metadata['sampling_rate'], self.noise_seconds)
        data['p_picks'] = [picks]
        y = np.array([self.event_metadata.iloc[index][self.target_key]])

        if self.cutout: # 可调整
            if self.sliding_window:
                windowlen = self.windowlen
                window_end = np.random.randint(max(windowlen, self.cutout[0]), min(X.shape[1], self.cutout[1]) + 1)
                X = X[:, window_end - windowlen: window_end]
            else:
                X[:, np.random.randint(*self.cutout):] = 0

        if self.label_smoothing:
            y += (y > 4) * np.random.randn(y.shape[0]).reshape(y.shape) * (y - 4) * 0.05

        X = torch.from_numpy(X[0].T).float()  # Convert to torch tensor
        y = torch.from_numpy(y).float()  # Convert to torch tensor
#        y = torch.from_numpy(y[0]).float()  # Convert to torch tensor
        metadata = 1

        return X, y, metadata

    def on_epoch_end(self):
        self.indexes = np.repeat(self.indexes, self.oversample)
        if self.shuffle:
            np.random.shuffle(self.indexes)


class PreloadedEventGenerator(Dataset):
    def __init__(self, event_metadata, metadata, data_path, generator_params, data_keys=None, overwrite_sampling_rate=None, key='MA', cutout=None,
                 sliding_window=False, windowlen=10000, shuffle=True,
                 coords_target=True, oversample=1, pos_offset=(-21, -69),
                 label_smoothing=False, station_blinding=False, magnitude_resampling=3,
                 pga_targets=None, adjust_mean=True, transform_target_only=False,
                 max_stations=None, trigger_based=None, min_upsample_magnitude=2,
                 disable_station_foreshadowing=False, selection_skew=None, pga_from_inactive=False,
                 integrate=False, sampling_rate=100.,
                 select_first=False, select_first_inputs=None, select_first_pga_targets=None,
                 fake_borehole=False, scale_metadata=True, pga_key='pga',
                 pga_mode=False, p_pick_limit=5000, coord_keys=None, upsample_high_station_events=None,
                 no_event_token=False, pga_selection_skew=None,
                 random_input_station_count=None, pga_target_sampling=None, pga_distance_bins=None,
                 pga_label_stratified_threshold=None, pga_label_strong_fraction=0.5,
                 dump_debug_snapshot=False,
                 use_coords_rel=False, use_coords_abs=True,
                 use_coords_rel_abs_fusion=False, station_experiment=None,
                 input_station_selection=None, deterministic_sampling_seed=None,
                 use_vs30=False, realtime_training=None, realtime_target_sampling=None,
                 dpk_prior_cache=None, dpk_prior_cache_split=None,
                 dpk_prior_cache_dataset_id=0, dpk_prior_cache_align_realtime=True,
                 dpk_prior_cache_filter_missing_stations=True, **kwargs):
        if kwargs:
            print(f'Unused parameters: {", ".join(kwargs.keys())}')
        self.shuffle = shuffle
        self.wave_eps = 1e-5
#        self.waveforms = data['waveforms']
#        self.metadata = data['coords']
        self.event_metadata = event_metadata
        self.event_key = loader_light.detect_event_key(event_metadata.columns)
        self.event_metadata = self.event_metadata.groupby(self.event_key)
        self.metadata = metadata
        self.overwrite_sampling_rate = overwrite_sampling_rate
        if self.overwrite_sampling_rate is not None:
            if self.metadata['sampling_rate'] % self.overwrite_sampling_rate != 0:
                raise ValueError(f'Overwrite sampling ({self.overwrite_sampling_rate}) rate must be true divisor of sampling'
                                 f' rate ({self.metadata["sampling_rate"]})')
            self.decimate = self.metadata['sampling_rate'] // self.overwrite_sampling_rate
            self.metadata['sampling_rate'] = self.overwrite_sampling_rate
        else:
            self.decimate = 1
        self.data_path = data_path
        requested_key = generator_params.get('key', key)
        resolved_key = loader_light.resolve_target_key(event_metadata.columns, requested_key)
        self.target_key = resolved_key
        self.data_keys = data_keys
        self.pga_key = pga_key
#        if pga_key in data:
#            self.pga = data[pga_key]
#        else:
#            print('Found no PGA values')
#            self.pga = [np.zeros(x.shape[0]) for x in self.waveforms]
        self.key = loader_light.resolve_target_key(event_metadata.columns, key)
        if cutout is None:
            self.cutout = None
        else:
            self.cutout = tuple(int(round(x)) for x in cutout)
        self.sliding_window = sliding_window  # If true, selects sliding windows instead of cutout. Uses cutout as values for end of window.
        self.windowlen = int(round(windowlen))  # Length of window for sliding window
        self.trace_length = windowlen
        self.noise_seconds = generator_params.get('noise_seconds', 5)
        self.coords_target = coords_target
        self.oversample = oversample
        self.pos_offset = pos_offset
        self.label_smoothing = label_smoothing
        self.station_blinding = station_blinding
        self.magnitude_resampling = magnitude_resampling
        self.pga_targets = pga_targets
        self.adjust_mean = adjust_mean
        self.transform_target_only = transform_target_only
        if max_stations is None:
            max_stations = self.waveforms.shape[1]
        self.max_stations = max_stations
        self.dpk_prior_cache = dpk_prior_cache
        self.dpk_prior_cache_split = dpk_prior_cache_split
        self.dpk_prior_cache_dataset_id = int(dpk_prior_cache_dataset_id)
        self.dpk_prior_cache_align_realtime = bool(dpk_prior_cache_align_realtime)
        self.dpk_prior_cache_filter_missing_stations = bool(dpk_prior_cache_filter_missing_stations)
        self.trigger_based = trigger_based
        self.disable_station_foreshadowing = disable_station_foreshadowing
        self.selection_skew = selection_skew
        self.pga_from_inactive = pga_from_inactive
        self.pga_selection_skew = pga_selection_skew
        self.random_input_station_count = self._normalize_station_count_choices(random_input_station_count)
        self.pga_target_sampling = pga_target_sampling
        valid_pga_target_sampling = {None, 'random', 'distance_stratified', 'distance_coverage', 'label_stratified'}
        if self.pga_target_sampling not in valid_pga_target_sampling:
            raise ValueError(
                f'pga_target_sampling must be one of {sorted(x for x in valid_pga_target_sampling if x)}, '
                f'got {self.pga_target_sampling!r}'
            )
        self.pga_distance_bins = self._normalize_distance_bins(pga_distance_bins)
        self.pga_label_stratified_threshold = (
            None if pga_label_stratified_threshold is None else float(pga_label_stratified_threshold)
        )
        self.pga_label_strong_fraction = float(pga_label_strong_fraction)
        if not (0.0 <= self.pga_label_strong_fraction <= 1.0):
            raise ValueError('pga_label_strong_fraction must be between 0 and 1.')
        self.integrate = integrate
        self.sampling_rate = sampling_rate
        self.select_first = select_first
        self.select_first_inputs = select_first if select_first_inputs is None else select_first_inputs
        self.select_first_pga_targets = (
            select_first if select_first_pga_targets is None else select_first_pga_targets
        )
        self.input_station_selection = self._normalize_input_station_selection(input_station_selection)
        self.deterministic_sampling_seed = deterministic_sampling_seed
        self.fake_borehole = fake_borehole
        self.scale_metadata = scale_metadata
        self.upsample_high_station_events = upsample_high_station_events
        self.no_event_token = no_event_token
        self.dump_debug_snapshot = dump_debug_snapshot
        self.use_coords_rel = use_coords_rel
        self.use_coords_abs = use_coords_abs
        self.use_coords_rel_abs_fusion = use_coords_rel_abs_fusion
        self.station_experiment = self._normalize_station_experiment(station_experiment)
        self.use_vs30 = bool(use_vs30)
        active_coord_modes = sum(bool(flag) for flag in (
            self.use_coords_rel, self.use_coords_abs, self.use_coords_rel_abs_fusion
        ))
        if active_coord_modes != 1:
            raise ValueError(
                'Exactly one of use_coords_rel / use_coords_abs / '
                'use_coords_rel_abs_fusion must be True.'
            )
        self.loc_target_mode = 'abs' if self.use_coords_abs else 'rel'

#        if 'p_picks' in data:
#            self.triggers = data['p_picks']
#        else:
#            print('Found no picks')
#            self.triggers = [np.zeros(x.shape[0]) for x in self.waveforms]

        # Extend samples to include all pga targets in each epoch
        # PGA mode is only for evaluation, as it adds zero padding to the input/pga target!
        self.pga_mode = pga_mode
        self.p_pick_limit = p_pick_limit
        self.realtime_training = self._normalize_realtime_training(realtime_training, shuffle=shuffle)
        self.realtime_enabled = bool(self.realtime_training.get('enabled', False))
        self.realtime_target_sampling = self._normalize_realtime_target_sampling(
            realtime_target_sampling,
            realtime_enabled=self.realtime_enabled,
        )
        if self.realtime_enabled and self.pga_mode:
            raise ValueError('realtime_training is not compatible with pga_mode evaluation.')

        self._realtime_epoch = 0
        self.base_indexes = np.arange(len(self.event_metadata))
        self.event_keys = list(self.event_metadata.groups.keys())
        self.reverse_index = None
        if magnitude_resampling > 1:
            magnitude = self.event_metadata[self.key].mean().values
            for i in np.arange(min_upsample_magnitude, 9):
                ind = np.where(np.logical_and(i < magnitude, magnitude <= i + 1))[0]
                self.base_indexes = np.concatenate(
                    (self.base_indexes, np.repeat(ind, int(magnitude_resampling ** (i - 1) - 1))))

        if self.upsample_high_station_events is not None:
            new_indexes = []
            for ind in self.base_indexes:
                n_stations = self.waveforms[ind].shape[0]
                new_indexes += [ind for _ in range(n_stations // self.upsample_high_station_events + 1)]
            self.base_indexes = np.array(new_indexes) # ?zb

        if pga_mode:
            new_base_indexes = []
            self.reverse_index = []
            c = 0
            for idx in self.base_indexes:
                # This produces an issue if there are 0 pga targets for an event.
                # As all input stations are always pga targets as well, this should not occur.
                #
                num_samples = (len(self.pga[idx]) - 1) // pga_targets + 1
                new_base_indexes += [(idx, i) for i in range(num_samples)]
                self.reverse_index += [c]
                c += num_samples
            self.reverse_index += [c]
            self.base_indexes = new_base_indexes

        if coord_keys is None:
            self.coord_keys = detect_location_keys(self.event_metadata.obj.columns)
        else:
            self.coord_keys = coord_keys

        self.on_epoch_end()

    def __len__(self):
#        return len(self.event_metadata)
        return len(self.indexes)

    @staticmethod
    def _normalize_time_bins(value):
        if value is None:
            value = [(0, 1), (1, 3), (3, 5), (5, 10), (10, 20), (20, 40), (40, 90)]
        bins = []
        for item in value:
            if len(item) != 2:
                raise ValueError(f'realtime time bin must contain [start, end], got {item!r}')
            start, end = float(item[0]), float(item[1])
            if not (np.isfinite(start) and np.isfinite(end)) or start < 0 or end < start:
                raise ValueError(f'invalid realtime time bin: {item!r}')
            bins.append((start, end))
        if not bins:
            raise ValueError('realtime_training.train_time_bins must not be empty.')
        return bins

    @staticmethod
    def _normalize_realtime_training(value, shuffle=True):
        if not value:
            return {'enabled': False}
        cfg = dict(value)
        cfg['enabled'] = bool(cfg.get('enabled', False))
        if not cfg['enabled']:
            return cfg
        mode = str(cfg.get('mode', 'train' if shuffle else 'val')).strip().lower()
        if mode not in ('train', 'val'):
            raise ValueError(f'realtime_training.mode must be train or val, got {mode!r}')
        cfg['mode'] = mode
        reference = str(cfg.get('reference', 'first_p_pick')).strip().lower()
        if reference not in ('first_p_pick', 'first_valid_p_pick'):
            raise ValueError(
                'realtime_training.reference currently supports only first_p_pick / first_valid_p_pick.'
            )
        cfg['reference'] = 'first_p_pick'
        cfg['train_time_bins'] = PreloadedEventGenerator._normalize_time_bins(cfg.get('train_time_bins'))
        val_times = cfg.get('val_times', [1, 3, 5, 10, 20, 40, 90])
        cfg['val_times'] = [float(x) for x in val_times]
        if not cfg['val_times'] or any((not np.isfinite(x) or x < 0) for x in cfg['val_times']):
            raise ValueError('realtime_training.val_times must contain non-negative finite values.')
        train_times = cfg.get('train_times', None)
        train_time_mode = str(cfg.get('train_time_mode', '')).strip().lower()
        if train_times is None and train_time_mode in ('fixed', 'fixed_times', 'discrete'):
            train_times = cfg['val_times']
        if train_times is not None:
            cfg['train_times'] = [float(x) for x in train_times]
            if not cfg['train_times'] or any((not np.isfinite(x) or x < 0) for x in cfg['train_times']):
                raise ValueError('realtime_training.train_times must contain non-negative finite values.')
            cfg['train_time_mode'] = 'fixed'
        else:
            cfg['train_times'] = None
            cfg['train_time_mode'] = 'random_bins'
        cfg['bins_per_event_per_epoch'] = max(1, int(cfg.get('bins_per_event_per_epoch', 1)))
        bin_sampling = str(cfg.get('bin_sampling', 'without_replacement')).strip().lower()
        if bin_sampling not in ('without_replacement', 'with_replacement'):
            raise ValueError(
                'realtime_training.bin_sampling must be without_replacement or with_replacement.'
            )
        cfg['bin_sampling'] = bin_sampling
        return cfg

    @staticmethod
    def _normalize_realtime_target_sampling(value, realtime_enabled=False):
        if value is None:
            cfg = {'enabled': bool(realtime_enabled)}
        else:
            cfg = dict(value)
            cfg['enabled'] = bool(cfg.get('enabled', False))
        if not cfg.get('enabled', False):
            return {'enabled': False}
        cfg['input_ratio'] = float(cfg.get('input_ratio', 0.3))
        cfg['triggered_noninput_ratio'] = float(cfg.get('triggered_noninput_ratio', 0.2))
        cfg['untriggered_ratio'] = float(cfg.get('untriggered_ratio', 0.5))
        ratios = [cfg['input_ratio'], cfg['triggered_noninput_ratio'], cfg['untriggered_ratio']]
        if any((not np.isfinite(x) or x < 0) for x in ratios) or sum(ratios) <= 0:
            raise ValueError('realtime_target_sampling ratios must be non-negative and sum to > 0.')
        cfg['fill_missing'] = bool(cfg.get('fill_missing', True))
        return cfg

    @staticmethod
    def _ratio_quotas(ratios, total):
        ratios = np.asarray(ratios, dtype=np.float64)
        ratios = np.maximum(ratios, 0.0)
        if total <= 0 or ratios.sum() <= 0:
            return np.zeros_like(ratios, dtype=np.int64)
        raw = ratios / ratios.sum() * int(total)
        quotas = np.floor(raw).astype(np.int64)
        remainder = int(total - quotas.sum())
        if remainder > 0:
            order = np.argsort(-(raw - quotas))
            for pos in order[:remainder]:
                quotas[pos] += 1
        return quotas

    def _realtime_index_context(self, index_entry):
        if not isinstance(index_entry, tuple):
            return int(index_entry), {'mode': self.realtime_training['mode'], 'slot': 0}
        event_index = int(index_entry[0])
        slot = int(index_entry[1])
        mode = self.realtime_training['mode']
        if mode == 'val':
            time_index = -slot - 1 if slot < 0 else slot
            return event_index, {'mode': mode, 'time_index': int(time_index), 'slot': slot}
        if slot < 0:
            time_index = -slot - 1
            return event_index, {'mode': mode, 'time_index': int(time_index), 'slot': slot}
        return event_index, {'mode': mode, 'bin_index': int(slot), 'slot': slot}

    def _realtime_pick_valid(self, picks):
        picks = np.asarray(picks, dtype=float)
        return np.isfinite(picks) & (picks > 0) & (picks < self.trace_length)

    def _select_realtime_cutout(self, full_p_picks, station_valid_full, rng, context, waveform_length):
        picks = np.asarray(full_p_picks[0], dtype=float)
        valid = np.asarray(station_valid_full[0], dtype=bool)
        valid &= self._realtime_pick_valid(picks)
        if not valid.any():
            raise _EmptySample()
        first_pick = int(round(float(np.min(picks[valid]))))
        if 'time_index' in context:
            time_index = int(context.get('time_index', 0))
            times_key = 'val_times' if context['mode'] == 'val' else 'train_times'
            times = self.realtime_training.get(times_key) or []
            if time_index < 0 or time_index >= len(times):
                raise ValueError(f'invalid realtime {context["mode"]} time index: {time_index}')
            elapsed_requested = float(times[time_index])
            time_bin = -time_index - 1
        else:
            bins = self.realtime_training['train_time_bins']
            bin_index = int(context.get('bin_index', 0))
            if bin_index < 0 or bin_index >= len(bins):
                raise ValueError(f'invalid realtime train bin index: {bin_index}')
            start, end = bins[bin_index]
            u = float(rng.random() if rng is not None else np.random.random())
            elapsed_requested = start + (end - start) * u
            time_bin = bin_index

        current_sample = first_pick + int(round(elapsed_requested * self.sampling_rate))
        # Existing cutout code treats cutout as an exclusive endpoint. Add one
        # sample so a station whose P pick is exactly at current_sample remains
        # a valid triggered input at t=0.
        cutout = int(np.clip(current_sample + 1, 1, waveform_length))
        current_sample = cutout - 1
        elapsed_actual = max(0.0, (current_sample - first_pick) / float(self.sampling_rate))
        return {
            'cutout': cutout,
            'current_sample': current_sample,
            'first_p_pick_sample': first_pick,
            'elapsed_time': elapsed_actual,
            'requested_elapsed_time': elapsed_requested,
            'time_bin': time_bin,
        }

    def _normalize_station_experiment(self, station_experiment):
        if not station_experiment:
            return {'enabled': False}
        cfg = dict(station_experiment)
        cfg['enabled'] = bool(cfg.get('enabled', False))
        if not cfg['enabled']:
            return cfg
        mode = cfg.get('mode', None)
        valid_modes = {
            'single_input_same_station_pga',
            'single_input_multi_target_pga',
            'snr_filtered_input_holdout_pga',
        }
        if mode not in valid_modes:
            raise ValueError(f'station_experiment.mode must be one of {sorted(valid_modes)}, got {mode!r}')
        if mode in ('single_input_same_station_pga', 'single_input_multi_target_pga'):
            cfg['input_station_count'] = int(cfg.get('input_station_count', 1))
            if cfg['input_station_count'] != 1:
                raise ValueError(f'{mode} requires input_station_count=1')
        cfg['target_station_count'] = cfg.get('target_station_count', self.pga_targets)
        if cfg['target_station_count'] is not None:
            cfg['target_station_count'] = int(cfg['target_station_count'])
        cfg['exclude_input_from_targets'] = bool(cfg.get(
            'exclude_input_from_targets',
            mode != 'single_input_same_station_pga',
        ))
        cfg['max_input_stations'] = int(cfg.get('max_input_stations', self.max_stations))
        cfg['snr_threshold'] = cfg.get('snr_threshold', None)
        if cfg['snr_threshold'] is not None:
            cfg['snr_threshold'] = float(cfg['snr_threshold'])
        cfg['snr_noise_window_sec'] = tuple(cfg.get('snr_noise_window_sec', (-5.0, -1.0)))
        cfg['snr_signal_window_sec'] = tuple(cfg.get('snr_signal_window_sec', (0.0, 5.0)))
        return cfg

    def _station_snr(self, waveforms, p_picks):
        cfg = self.station_experiment
        noise_sec = cfg.get('snr_noise_window_sec', (-5.0, -1.0))
        signal_sec = cfg.get('snr_signal_window_sec', (0.0, 5.0))
        noise_offsets = tuple(int(round(x * self.sampling_rate)) for x in noise_sec)
        signal_offsets = tuple(int(round(x * self.sampling_rate)) for x in signal_sec)
        snr = np.zeros(waveforms.shape[:2], dtype=np.float32)
        eps = self.wave_eps
        n_samples = waveforms.shape[2]
        for i in range(waveforms.shape[0]):
            for j in range(waveforms.shape[1]):
                pick = int(round(p_picks[i, j]))
                if pick <= 0 or pick >= n_samples:
                    continue
                n0 = max(0, pick + noise_offsets[0])
                n1 = min(n_samples, pick + noise_offsets[1])
                s0 = max(0, pick + signal_offsets[0])
                s1 = min(n_samples, pick + signal_offsets[1])
                if n1 <= n0 or s1 <= s0:
                    continue
                noise = waveforms[i, j, n0:n1, :]
                signal = waveforms[i, j, s0:s1, :]
                noise_rms = np.sqrt(np.mean(noise ** 2))
                signal_rms = np.sqrt(np.mean(signal ** 2))
                snr[i, j] = signal_rms / (noise_rms + eps)
        return snr

    @staticmethod
    def _rng_int(rng, *args):
        return rng.integers(*args) if rng is not None else np.random.randint(*args)

    @staticmethod
    def _rng_choice(rng, values):
        return rng.choice(values) if rng is not None else np.random.choice(values)

    @staticmethod
    def _rng_random(rng, shape):
        return rng.random(shape) if rng is not None else np.random.random(shape)

    @staticmethod
    def _rng_shuffle(rng, values):
        if rng is None:
            np.random.shuffle(values)
        else:
            rng.shuffle(values)

    def _sample_rng(self, index):
        if self.deterministic_sampling_seed is None:
            return None
        if isinstance(index, tuple):
            sample_id = 0
            for item in index:
                sample_id = sample_id * 1009 + int(item)
        else:
            sample_id = int(index)
        seed = (int(self.deterministic_sampling_seed) + sample_id) % (2 ** 63 - 1)
        return np.random.default_rng(seed)

    @staticmethod
    def _pick_random_subset(candidates, n, rng=None):
        candidates = np.asarray(candidates, dtype=np.int64)
        if candidates.size <= n:
            return candidates
        picked = candidates.copy()
        if rng is None:
            np.random.shuffle(picked)
        else:
            rng.shuffle(picked)
        return picked[:n]

    @staticmethod
    def _normalize_station_count_choices(value):
        if value is None:
            return None
        if isinstance(value, str):
            value = [x.strip() for x in value.split(',') if x.strip()]
        elif np.isscalar(value):
            value = [value]
        choices = [int(x) for x in value]
        choices = sorted({x for x in choices if x > 0})
        if not choices:
            raise ValueError('random_input_station_count must contain at least one positive count')
        return choices

    @staticmethod
    def _normalize_distance_bins(value):
        if value is None:
            return None
        if isinstance(value, str):
            value = [x.strip() for x in value.split(',') if x.strip()]
        bins = np.asarray([float(x) for x in value], dtype=np.float32)
        bins = np.sort(bins[np.isfinite(bins)])
        return bins if bins.size > 0 else None

    @staticmethod
    def _normalize_input_station_selection(value):
        if value is None:
            return None
        value = str(value).strip().lower()
        if value in ('', 'config', 'default'):
            return None
        aliases = {
            'pick': 'p_pick',
            'ppick': 'p_pick',
            'picks': 'p_pick',
            'distance': 'epidist',
            'nearest': 'epidist',
            'nearest_epidist': 'epidist',
            'epicentral_distance': 'epidist',
            'random': 'random',
            'p_pick': 'p_pick',
            'epidist': 'epidist',
        }
        if value not in aliases:
            raise ValueError(
                'input_station_selection must be one of config/default, random, '
                f'p_pick, epidist; got {value!r}'
            )
        return aliases[value]

    @staticmethod
    def _first_event_coord(target):
        if target is None:
            return None
        arr = np.asarray(target, dtype=float)
        if arr.size == 0:
            return None
        arr = arr.reshape(-1, arr.shape[-1])
        if arr.shape[1] < 2:
            return None
        return arr[0]

    @staticmethod
    def _horizontal_distance_order(candidates, station_coords, event_coord):
        candidates = np.asarray(candidates, dtype=np.int64)
        if candidates.size == 0:
            return candidates
        if station_coords is None or event_coord is None:
            return candidates
        coords = np.asarray(station_coords, dtype=float)
        event_coord = np.asarray(event_coord, dtype=float).reshape(-1)
        if coords.ndim != 2 or coords.shape[1] < 2 or event_coord.size < 2:
            return candidates
        diffs = coords[candidates, :2] - event_coord[None, :2]
        dist2 = np.sum(diffs * diffs, axis=1)
        dist2[~np.isfinite(dist2)] = np.inf
        return candidates[np.lexsort((candidates, dist2))]

    def _input_station_order(self, candidates, station_coords=None, event_coord=None, p_picks=None):
        candidates = np.asarray(candidates, dtype=np.int64)
        if candidates.size == 0:
            return candidates
        if self.input_station_selection == 'epidist':
            return self._horizontal_distance_order(candidates, station_coords, event_coord)
        if self.input_station_selection == 'p_pick':
            if p_picks is None:
                return candidates
            picks = np.asarray(p_picks, dtype=float)
            order_values = picks[candidates].copy()
            bad = np.logical_or(order_values <= 0, order_values > self.p_pick_limit)
            order_values[bad] = np.inf
            return candidates[np.lexsort((candidates, order_values))]
        return candidates

    def _apply_random_input_station_count(self, station_valid, metadata=None, target=None, p_picks=None, rng=None):
        if self.random_input_station_count is None:
            return station_valid

        sampled_valid = np.zeros_like(station_valid, dtype=bool)
        for i in range(station_valid.shape[0]):
            active = np.where(station_valid[i])[0]
            if active.size == 0:
                raise _EmptySample()
            n_inputs = int(self._rng_choice(rng, self.random_input_station_count))
            n_inputs = min(n_inputs, active.size)
            if self.input_station_selection in ('epidist', 'p_pick'):
                event_coord = self._first_event_coord(None if target is None else target[i])
                coords = None if metadata is None else metadata[i]
                picks = None if p_picks is None else p_picks[i]
                selected = self._input_station_order(active, coords, event_coord, picks)[:n_inputs]
            else:
                selected = self._pick_random_subset(active, n_inputs, rng=rng)
            sampled_valid[i, selected] = True
        return sampled_valid

    @staticmethod
    def _coords3_from_rows(rows):
        if rows.shape[-1] == 4:
            return rows[:, (0, 1, 3)]
        return rows[:, :3]

    def _stratified_by_nearest_input_distance(self, active, target_coords, input_coords, n_targets, rng=None):
        active = np.asarray(active, dtype=np.int64)
        if active.size <= n_targets or input_coords.size == 0:
            selected = active.copy()
            self._rng_shuffle(rng, selected)
            return selected[:n_targets]

        target_xyz = self._coords3_from_rows(target_coords[active])
        input_xyz = self._coords3_from_rows(input_coords)
        nearest_dist = np.linalg.norm(
            target_xyz[:, None, :] - input_xyz[None, :, :],
            axis=-1,
        ).min(axis=1)

        selected_positions = []
        if self.pga_distance_bins is None:
            ordered = np.argsort(nearest_dist)
            groups = np.array_split(ordered, min(n_targets, active.size))
            bin_positions = [g.copy() for g in groups if g.size > 0]
        else:
            bin_ids = np.digitize(nearest_dist, self.pga_distance_bins)
            bin_positions = [np.where(bin_ids == b)[0] for b in np.unique(bin_ids)]

        bin_queues = []
        for positions in bin_positions:
            shuffled = positions.copy()
            self._rng_shuffle(rng, shuffled)
            bin_queues.append(list(shuffled))

        while len(selected_positions) < n_targets:
            progressed = False
            for queue in bin_queues:
                if len(selected_positions) >= n_targets:
                    break
                if not queue:
                    continue
                selected_positions.append(int(queue.pop(0)))
                progressed = True
            if not progressed:
                break

        if len(selected_positions) < n_targets:
            used = set(selected_positions)
            remaining = np.array([j for j in range(active.size) if j not in used], dtype=np.int64)
            self._rng_shuffle(rng, remaining)
            selected_positions.extend(int(j) for j in remaining[:n_targets - len(selected_positions)])

        return active[np.asarray(selected_positions, dtype=np.int64)]

    def _label_stratified_pga_targets(self, active, pga_values, n_targets, rng=None):
        active = np.asarray(active, dtype=np.int64)
        if active.size <= n_targets:
            selected = active.copy()
            self._rng_shuffle(rng, selected)
            return selected[:n_targets]

        labels = np.asarray(pga_values[active], dtype=float)
        finite = np.isfinite(labels)
        if not finite.any():
            selected = active.copy()
            self._rng_shuffle(rng, selected)
            return selected[:n_targets]

        if self.pga_label_stratified_threshold is None:
            threshold = float(np.quantile(labels[finite], 0.8))
        else:
            threshold = self.pga_label_stratified_threshold

        strong = active[finite & (labels >= threshold)]
        weak = active[~(finite & (labels >= threshold))]
        self._rng_shuffle(rng, strong)
        self._rng_shuffle(rng, weak)

        n_strong = int(np.ceil(n_targets * self.pga_label_strong_fraction))
        if strong.size > 0 and self.pga_label_strong_fraction > 0:
            n_strong = max(1, n_strong)
        n_strong = min(n_strong, strong.size, n_targets)

        selected = list(strong[:n_strong])
        remaining_slots = n_targets - len(selected)
        if remaining_slots > 0:
            selected.extend(weak[:remaining_slots])
        remaining_slots = n_targets - len(selected)
        if remaining_slots > 0:
            selected.extend(strong[n_strong:n_strong + remaining_slots])

        selected = np.asarray(selected, dtype=np.int64)
        self._rng_shuffle(rng, selected)
        return selected[:n_targets]

    def _classify_realtime_targets(self, samples, input_station_valid, full_p_picks, current_sample):
        samples = np.asarray(samples, dtype=np.int64)
        picks = np.asarray(full_p_picks, dtype=float)
        input_mask = np.zeros_like(picks, dtype=bool)
        n_input = min(input_station_valid.shape[0], input_mask.shape[0])
        input_mask[:n_input] = input_station_valid[:n_input]
        pick_valid = self._realtime_pick_valid(picks)
        triggered = pick_valid & (picks <= current_sample)
        target_type = np.full(samples.shape, 2, dtype=np.int64)
        target_type[triggered[samples]] = 1
        target_type[input_mask[samples]] = 0
        return target_type

    def _sample_realtime_pga_targets(self, active, input_station_valid, full_p_picks,
                                     current_sample, n_targets, rng=None):
        active = np.asarray(active, dtype=np.int64)
        if active.size == 0:
            return active, np.zeros((0,), dtype=np.int64)

        picks = np.asarray(full_p_picks, dtype=float)
        input_mask = np.zeros_like(picks, dtype=bool)
        n_input = min(input_station_valid.shape[0], input_mask.shape[0])
        input_mask[:n_input] = input_station_valid[:n_input]
        pick_valid = self._realtime_pick_valid(picks)
        triggered = pick_valid & (picks <= current_sample)

        active_mask = np.zeros_like(picks, dtype=bool)
        active_mask[active] = True
        category_candidates = [
            np.where(active_mask & input_mask)[0],
            np.where(active_mask & triggered & ~input_mask)[0],
            np.where(active_mask & ~triggered & ~input_mask)[0],
        ]
        ratios = [
            self.realtime_target_sampling['input_ratio'],
            self.realtime_target_sampling['triggered_noninput_ratio'],
            self.realtime_target_sampling['untriggered_ratio'],
        ]
        quotas = self._ratio_quotas(ratios, n_targets)

        selected = []
        selected_types = []
        used = set()
        for target_type, candidates, quota in zip((0, 1, 2), category_candidates, quotas):
            if quota <= 0:
                continue
            candidates = np.asarray([int(x) for x in candidates if int(x) not in used], dtype=np.int64)
            picked = self._pick_random_subset(candidates, int(quota), rng=rng)
            for station_idx in picked:
                station_idx = int(station_idx)
                selected.append(station_idx)
                selected_types.append(target_type)
                used.add(station_idx)

        remaining_slots = min(n_targets, active.size) - len(selected)
        if remaining_slots > 0 and self.realtime_target_sampling.get('fill_missing', True):
            remaining = np.asarray([int(x) for x in active if int(x) not in used], dtype=np.int64)
            picked = self._pick_random_subset(remaining, remaining_slots, rng=rng)
            picked_types = self._classify_realtime_targets(
                picked,
                input_station_valid,
                full_p_picks,
                current_sample,
            )
            for station_idx, target_type in zip(picked, picked_types):
                selected.append(int(station_idx))
                selected_types.append(int(target_type))

        samples = np.asarray(selected, dtype=np.int64)
        target_types = np.asarray(selected_types, dtype=np.int64)
        if samples.size > n_targets:
            samples = samples[:n_targets]
            target_types = target_types[:n_targets]
        return samples, target_types

    def _apply_station_experiment_inputs(self, waveforms, p_picks, station_valid, pga_valid_input):
        cfg = self.station_experiment
        if not cfg.get('enabled', False):
            return station_valid, None

        mode = cfg['mode']
        input_mask = np.zeros_like(station_valid, dtype=bool)
        snr = None
        if mode == 'snr_filtered_input_holdout_pga':
            snr = self._station_snr(waveforms, p_picks)

        for i in range(station_valid.shape[0]):
            candidates = station_valid[i].copy()
            if mode == 'single_input_same_station_pga':
                candidates &= pga_valid_input[i]
                n_inputs = 1
            elif mode == 'single_input_multi_target_pga':
                n_inputs = 1
            elif mode == 'snr_filtered_input_holdout_pga':
                threshold = cfg.get('snr_threshold', None)
                if threshold is not None:
                    candidates &= snr[i] >= threshold
                n_inputs = cfg.get('max_input_stations', self.max_stations)
            else:
                raise ValueError(f'Unsupported station experiment mode: {mode}')

            active = np.where(candidates)[0]
            if active.size == 0:
                raise _EmptySample()
            selected = self._pick_random_subset(active, n_inputs)
            input_mask[i, selected] = True

        waveforms[~input_mask] = 0
        return station_valid & input_mask, snr

    def _build_station_experiment_pga_targets(self, metadata, pga, pga_valid, station_valid_full,
                                              input_station_valid, selected_input_indices,
                                              full_selected_indices, return_indices=False):
        cfg = self.station_experiment
        if not cfg.get('enabled', False) or not self.pga_targets:
            return None

        mode = cfg['mode']
        n_targets = min(cfg.get('target_station_count', self.pga_targets) or self.pga_targets, self.pga_targets)
        pga_values = np.zeros((input_station_valid.shape[0], self.pga_targets), dtype=np.float32)
        pga_targets = np.zeros((input_station_valid.shape[0], self.pga_targets, 3), dtype=np.float32)
        pga_target_valid = np.zeros((input_station_valid.shape[0], self.pga_targets), dtype=bool)
        pga_target_indices = -np.ones((input_station_valid.shape[0], self.pga_targets), dtype=np.int64)

        for i in range(input_station_valid.shape[0]):
            input_slots = np.where(input_station_valid[i])[0]
            if input_slots.size == 0:
                raise _EmptySample()

            if mode == 'single_input_same_station_pga':
                candidates = input_slots[pga_valid[i, input_slots]]
            else:
                candidates = np.where(pga_valid[i] & station_valid_full[i])[0]
                if cfg.get('exclude_input_from_targets', True):
                    input_orig = set(
                        int(selected_input_indices[i, slot])
                        for slot in input_slots
                        if selected_input_indices[i, slot] >= 0
                    )
                    candidates = np.array(
                        [
                            idx for idx in candidates
                            if int(full_selected_indices[i, idx]) not in input_orig
                        ],
                        dtype=np.int64,
                    )

            if candidates.size == 0:
                raise _EmptySample()
            selected_targets = self._pick_random_subset(candidates, n_targets)
            n = len(selected_targets)
            if metadata.shape[-1] == 3:
                pga_targets[i, :n, :] = metadata[i, selected_targets, :]
            else:
                full_targets = metadata[i, selected_targets]
                pga_targets[i, :n, :] = full_targets[:, (0, 1, 3)]
            pga_values[i, :n] = pga[i, selected_targets]
            pga_target_valid[i, :n] = True
            pga_target_indices[i, :n] = selected_targets

        result = (pga_targets, pga_values.reshape((input_station_valid.shape[0], self.pga_targets, 1)), pga_target_valid)
        if return_indices:
            result = result + (pga_target_indices,)
        return result

    def __getitem__(self, index):
        # Iteratively skip empty samples (no station with signal after cutout).
        # Previous implementation recursed into __getitem__, which could blow
        # the Python stack if many consecutive samples happen to be empty.
        n = len(self.indexes)
        for _ in range(n):
            try:
                return self._get_one(index)
            except _EmptySample:
                index = (index + 1) % n
        raise RuntimeError(
            'All samples in dataset produced empty waveforms after cutout; '
            'check data quality or cutout configuration.'
        )

    def _get_one(self, index):
        # Generate indexes of the batch
        sample_index_for_cache = int(index)
        index_entry = self.indexes[index]
        indexes = self.indexes[index:(index + 1)]
        rng = self._sample_rng(index_entry)
        realtime_context = None
        if self.realtime_enabled:
            event_index, realtime_context = self._realtime_index_context(index_entry)
            index = event_index
            indexes = [event_index]
        else:
            index = index_entry
        if self.pga_mode:
            pga_indexes = [x[1] for x in indexes]
            indexes = [x[0] for x in indexes]
            index = indexes[0]
        ith_event = self.event_keys[index]
        with h5py.File(self.data_path, 'r') as f:
            event = self.event_metadata.get_group(ith_event)
            event_name = str(event[self.event_key].iloc[0]) # ith_event? zb
            g_event = f['data'][event_name]
            row_selector = _select_wave_idx_rows(event, g_event)
            if isinstance(row_selector, np.ndarray) and row_selector.size == 0:
                raise _EmptySample()
            if isinstance(row_selector, slice):
                original_wave_idx_for_loaded = np.arange(g_event['waveforms'].shape[0], dtype=np.int64)[row_selector]
            else:
                original_wave_idx_for_loaded = np.asarray(row_selector, dtype=np.int64)
            data = {}
            for key in g_event:
                extra_vs30_key = self.use_vs30 and key in (
                    'vs30',
                    'vs30_valid',
                    'vs30_query_distance_km',
                    'vs30_match_method',
                )
                if self.data_keys is not None and key not in self.data_keys and not extra_vs30_key:
                    continue
                if key not in data:
                    data[key] = []
                if key == 'waveforms':
                    cur_waveform = g_event[key][row_selector, ::self.decimate, :]
                    cur_waveform -= np.mean(cur_waveform, axis=1, keepdims=True)
                    data[key] += [cur_waveform]
                else:
                    values = g_event[key][()]
                    if np.ndim(values) > 0 and values.shape[0] == g_event['waveforms'].shape[0]:
                        values = values[row_selector]
                    data[key] += [values]
                if key == 'p_picks':
                    data[key][-1] //= self.decimate

        X = np.concatenate(data['waveforms'], axis=0)
        self.metadata = np.concatenate(data['coords'], axis=0) # coords of stations (lat, lon, elev)
        self.waveforms = X
        self.original_wave_idx = original_wave_idx_for_loaded

        has_pga_values = self.pga_key in data
        if has_pga_values:
            self.pga = np.concatenate(data[self.pga_key], axis=0)
        else:
            print('Found no PGA values')
            self.pga = np.zeros(X.shape[0])

        if 'p_picks' in data:
            self.triggers = np.concatenate(data['p_picks'], axis=0)
        else:
            print('Found no picks')
            self.triggers = np.zeros(X.shape[0])

        if self.use_vs30:
            if 'vs30' in data:
                self.vs30 = np.concatenate(data['vs30'], axis=0).astype(np.float32, copy=False)
            else:
                self.vs30 = np.full(X.shape[0], np.nan, dtype=np.float32)
            if 'vs30_valid' in data:
                self.vs30_valid = np.concatenate(data['vs30_valid'], axis=0).astype(bool, copy=False)
            else:
                self.vs30_valid = np.isfinite(self.vs30) & (self.vs30 > 0)
        else:
            self.vs30 = None
            self.vs30_valid = None

        self.waveforms, self.triggers, crop_start = _crop_aligned_event_window(
            self.waveforms,
            _safe_int_array(self.triggers),
            self.trace_length,
            self.sampling_rate,
            self.noise_seconds,
            rng=rng,
        )
        X = self.waveforms
        if self.pga_key in data:
            self.pga = np.asarray(self.pga)
        self.crop_start = crop_start

        y = np.array([self.event_metadata.get_group(ith_event)[self.target_key]]) # magnitude
        event_target_for_input_selection = None
        if self.coords_target:
            event_target_for_input_selection = self.event_metadata.get_group(ith_event)[self.coord_keys].values
        true_batch_size = 1

        waveforms = np.zeros((true_batch_size, self.max_stations) + self.waveforms.shape[1:])  # shape (1, 25, 10000, 3)
        true_max_stations_in_batch = max(max([self.metadata.shape[0] for idx in indexes]), self.max_stations) # max(n_stations,25) = tms
        metadata = np.zeros((true_batch_size, true_max_stations_in_batch) + self.metadata.shape[1:]) # shape (1,tms, 3), coords
        # Use NaN for PGA so that "no measurement" is unambiguous and the legal
        # log-PGA value 0 is not confused with padding.
        pga = np.full((true_batch_size, true_max_stations_in_batch), np.nan)  # shape (1, tms)
        if self.use_vs30:
            vs30 = np.full((true_batch_size, true_max_stations_in_batch), np.nan, dtype=np.float32)
            vs30_valid = np.zeros((true_batch_size, true_max_stations_in_batch), dtype=bool)
        else:
            vs30 = None
            vs30_valid = None
        full_p_picks = np.zeros((true_batch_size, true_max_stations_in_batch)) # shape (1, tms)
        p_picks = np.zeros((true_batch_size, self.max_stations)) # shape (1, 25)
        selected_input_indices = -np.ones((true_batch_size, self.max_stations), dtype=np.int64)
        selected_original_input_indices = -np.ones((true_batch_size, self.max_stations), dtype=np.int64)
        full_selected_indices = -np.ones((true_batch_size, true_max_stations_in_batch), dtype=np.int64)
        # station_valid: True for slots that hold a real station (not padding).
        # Will be tightened later by cutout/blinding/etc.
        station_valid_full = np.zeros((true_batch_size, true_max_stations_in_batch), dtype=bool) # shape (1, tms)
        reverse_selections = []

        # Find list of IDs
        for i, idx in enumerate(indexes):
            if len(X) <= self.max_stations:
                waveforms[i, :len(X)] = X
                metadata[i, :len(self.metadata)] = self.metadata
                pga[i, :len(self.pga)] = self.pga
                if self.use_vs30:
                    vs30[i, :len(self.vs30)] = self.vs30
                    vs30_valid[i, :len(self.vs30_valid)] = self.vs30_valid
                p_picks[i, :len(self.triggers)] = self.triggers
                full_p_picks[i, :len(self.triggers)] = self.triggers
                selected_input_indices[i, :len(X)] = np.arange(len(X), dtype=np.int64)
                selected_original_input_indices[i, :len(X)] = self.original_wave_idx[:len(X)]
                full_selected_indices[i, :len(X)] = np.arange(len(X), dtype=np.int64)
                station_valid_full[i, :len(self.metadata)] = True # all stations init to True
                reverse_selections += [[]]
            else:
                if self.selection_skew is None or self.selection_skew <= 0:  # random select
                    selection = np.arange(0, len(self.waveforms)) # all stations
                    self._rng_shuffle(rng, selection)
                else:  # pick_time + randomness
                    tmp_p_picks = self.triggers.copy()
                    mask = np.logical_or(tmp_p_picks <= 0, tmp_p_picks > self.p_pick_limit)
                    tmp_p_picks[mask] = min(np.max(tmp_p_picks), self.p_pick_limit)
                    coeffs = np.exp(-tmp_p_picks / self.selection_skew)
                    coeffs *= self._rng_random(rng, coeffs.shape)
                    coeffs[self.triggers == 0] = 0
                    coeffs[self.triggers > self.waveforms.shape[1]] = 0
                    selection = np.argsort(-coeffs)

                if self.input_station_selection in ('epidist', 'p_pick'):
                    event_coord = self._first_event_coord(event_target_for_input_selection)
                    selection = self._input_station_order(
                        selection,
                        station_coords=self.metadata,
                        event_coord=event_coord,
                        p_picks=self.triggers,
                    )
                elif self.select_first_inputs: # pick_time
                    selection = np.argsort(self.triggers)

                selection = selection[:true_max_stations_in_batch] # len tms
                metadata[i, :len(selection)] = self.metadata[selection]
                pga[i, :len(selection)] = self.pga[selection]
                if self.use_vs30:
                    vs30[i, :len(selection)] = self.vs30[selection]
                    vs30_valid[i, :len(selection)] = self.vs30_valid[selection]
                full_p_picks[i, :len(selection)] = self.triggers[selection]
                full_selected_indices[i, :len(selection)] = selection
                station_valid_full[i, :len(selection)] = True # tms set to True

                tmp_reverse_selection = [0 for _ in selection]
                for j, s in enumerate(selection):
                    tmp_reverse_selection[s] = j
                reverse_selections += [tmp_reverse_selection]

                selection = selection[:self.max_stations]
                selected_input_indices[i, :len(selection)] = selection
                selected_original_input_indices[i, :len(selection)] = self.original_wave_idx[selection]
                waveforms[i] = self.waveforms[selection]
                p_picks[i] = self.triggers[selection]

        if self.dump_debug_snapshot:
            debug_raw_waveforms = waveforms.copy()
            debug_raw_metadata = metadata[:, :self.max_stations].copy()
            debug_raw_station_valid = station_valid_full[:, :self.max_stations].copy()
            debug_raw_pga = pga[:, :self.max_stations].copy()
            debug_raw_full_p_picks = full_p_picks[:, :self.max_stations].copy()

        # Defensive: mark stations with NaN/Inf coordinates as invalid. Current
        # KiK-Net data is clean, but this guards the model against upstream
        # corruption (NaN coord → NaN position embedding → NaN loss).
        coord_valid = ~(np.isnan(metadata).any(axis=-1) | np.isinf(metadata).any(axis=-1))
        station_valid_full &= coord_valid
        metadata_for_input_selection = metadata.copy()

        magnitude = self.event_metadata.get_group(ith_event)[self.key].values.copy()

        target = None
        if self.coords_target:
            target = self.event_metadata.get_group(ith_event)[self.coord_keys].values  # event location
            if np.isnan(target).any() or np.isinf(target).any():
                raise ValueError(
                    f'Event {ith_event} has NaN/Inf hypocenter coordinates: {target}. '
                    f'Filter such events out of event_metadata before training.'
                )

        org_waveform_length = waveforms.shape[2]
        raw_p_picks = p_picks.copy()
        realtime_info = None
        cache_split_for_sample = None
        if self.dpk_prior_cache is not None:
            cache_split_for_sample = self.dpk_prior_cache_split
            if cache_split_for_sample is None:
                realtime_mode = self.realtime_training.get('mode') if self.realtime_training else 'dev'
                cache_split_for_sample = 'train' if realtime_mode == 'train' else 'dev'
        if self.realtime_enabled:
            if self.sliding_window:
                raise ValueError('realtime_training does not support sliding_window=True.')
            realtime_info = self._select_realtime_cutout(
                full_p_picks,
                station_valid_full,
                rng,
                realtime_context,
                org_waveform_length,
            )
            if self.dpk_prior_cache is not None and self.dpk_prior_cache_align_realtime:
                requested_current_sample = int(realtime_info['current_sample'])
                aligned_current_sample = self.dpk_prior_cache.nearest_current_sample(
                    cache_split_for_sample,
                    self.dpk_prior_cache_dataset_id,
                    str(ith_event),
                    requested_current_sample,
                )
                if aligned_current_sample is not None:
                    cutout = int(np.clip(int(aligned_current_sample) + 1, 1, org_waveform_length))
                    aligned_current_sample = cutout - 1
                    realtime_info['cache_requested_current_sample'] = requested_current_sample
                    realtime_info['cache_aligned_current_sample'] = int(aligned_current_sample)
                    realtime_info['cache_current_sample_delta'] = int(aligned_current_sample - requested_current_sample)
                    realtime_info['cutout'] = cutout
                    realtime_info['current_sample'] = int(aligned_current_sample)
                    first_pick = int(realtime_info['first_p_pick_sample'])
                    realtime_info['elapsed_time'] = max(
                        0.0,
                        (int(aligned_current_sample) - first_pick) / float(self.sampling_rate),
                    )

        if self.cutout or realtime_info is not None:
            if self.sliding_window:
                windowlen = self.windowlen
                window_end = self._rng_int(rng, max(windowlen, self.cutout[0]),
                                           min(waveforms.shape[2], self.cutout[1]) + 1)
                waveforms = waveforms[:, :, window_end - windowlen: window_end]

                cutout = window_end
                if self.adjust_mean:
                    waveforms -= np.mean(waveforms, axis=2, keepdims=True)
            else:
                if realtime_info is not None:
                    cutout = realtime_info['cutout']
                elif self.cutout[0] == self.cutout[1]:
                    cutout = self.cutout[0]
                else:
                    cutout = self._rng_int(rng, *self.cutout)
                if self.adjust_mean:
                    # Mean only over non-zero samples so that leading zero-padding
                    # is neither diluting the mean nor getting offset by it.
                    region = waveforms[:, :, :cutout + 1]                    # (B, S, T, C)
                    has_data = np.any(np.abs(region) > self.wave_eps, axis=-1)                  # (B, S, T)
                    n = has_data.sum(axis=2, keepdims=True).clip(min=1)      # (B, S, 1)
                    mu = (region * has_data[..., None]).sum(axis=2, keepdims=True) / n[..., None]
                    waveforms[:, :, :cutout + 1] -= mu * has_data[..., None]
                # Right-align: shift valid signal [0:cutout] to end of window,
                # matching real-time EEW where signal arrives at the tail
                shift = waveforms.shape[2] - cutout
                waveforms = np.roll(waveforms, shift, axis=2)
                waveforms[:, :, :shift] = 0
                p_picks = p_picks + shift
        else:
            cutout = waveforms.shape[2]
            shift = 0

        if self.trigger_based or self.realtime_enabled:
            # Remove waveforms for all stations that did not trigger yet to avoid knowledge leakage
            p_picks[p_picks <= 0] = org_waveform_length  # Ensure that stations without P picks do not show data
            waveforms[cutout + shift <= p_picks, :, :] = 0

        if self.integrate: # always False. acc to vel should be done on the data source.
            waveforms = np.cumsum(waveforms, axis=2) / self.sampling_rate

        # Reshape magnitude to match dimension number of MDN output
        magnitude = np.expand_dims(np.expand_dims(magnitude, axis=-1), axis=-1)
        # Center location on mean of locations
        if self.coords_target:
            metadata, target = self.location_transformation(metadata, target, station_valid=station_valid_full)
        else:
            metadata = self.location_transformation(metadata, station_valid=station_valid_full)

        if self.label_smoothing:
            magnitude += (magnitude > 4) * np.random.randn(magnitude.shape[0]).reshape(magnitude.shape) * (
                    magnitude - 4) * 0.05

        if not self.pga_from_inactive and not self.pga_mode: # select 1st 25 zb.
            metadata = metadata[:, :self.max_stations]
            pga = pga[:, :self.max_stations]
            if self.use_vs30:
                vs30 = vs30[:, :self.max_stations]
                vs30_valid = vs30_valid[:, :self.max_stations]
            station_valid_full = station_valid_full[:, :self.max_stations]
            full_selected_indices = full_selected_indices[:, :self.max_stations]

        # PGA validity is detected via NaN/Inf so that the legal value 0 is preserved.
        pga_valid_full = ~(np.isnan(pga) | np.isinf(pga))
        metadata_for_pga = metadata.copy()
        pga_for_targets = pga.copy()
        pga_valid_for_targets = pga_valid_full.copy()
        station_valid_for_targets = station_valid_full.copy()
        full_selected_indices_for_targets = full_selected_indices.copy()
        input_station_valid_for_model = station_valid_full[:, :self.max_stations].copy()
        has_signal_for_input = (np.abs(waveforms) > self.wave_eps).any(axis=(2, 3))
        pick_valid_for_input = (p_picks > 0) & (p_picks < waveforms.shape[2])
        input_station_valid_for_model &= has_signal_for_input
        input_station_valid_for_model &= pick_valid_for_input
        input_station_valid_for_model = self._apply_random_input_station_count(
            input_station_valid_for_model,
            metadata=metadata_for_input_selection,
            target=event_target_for_input_selection,
            p_picks=p_picks,
            rng=rng,
        )

        if self.pga_targets:
            pga_values = np.zeros(
                (true_batch_size, self.pga_targets))
            pga_targets = np.zeros((true_batch_size, self.pga_targets, 3))
            pga_target_valid = np.zeros((true_batch_size, self.pga_targets), dtype=bool)
            pga_target_indices = -np.ones((true_batch_size, self.pga_targets), dtype=np.int64)
            realtime_target_type = -np.ones((true_batch_size, self.pga_targets), dtype=np.int64)
            realtime_target_lead_time = np.full((true_batch_size, self.pga_targets), np.nan, dtype=np.float32)
            if self.use_vs30:
                pga_target_vs30 = np.zeros((true_batch_size, self.pga_targets, 1), dtype=np.float32)
                pga_target_vs30_valid = np.zeros((true_batch_size, self.pga_targets), dtype=bool)
            if self.pga_mode:
                for i in range(waveforms.shape[0]):
                    pga_index = pga_indexes[i]
                    if len(reverse_selections[i]) > 0:
                        sorted_pga = pga[i, reverse_selections[i]]
                        sorted_metadata = metadata[i, reverse_selections[i]]
                        sorted_valid = pga_valid_full[i, reverse_selections[i]]
                        if self.use_vs30:
                            sorted_vs30 = vs30[i, reverse_selections[i]]
                            sorted_vs30_valid = vs30_valid[i, reverse_selections[i]]
                    else:
                        sorted_pga = pga[i]
                        sorted_metadata = metadata[i]
                        sorted_valid = pga_valid_full[i]
                        if self.use_vs30:
                            sorted_vs30 = vs30[i]
                            sorted_vs30_valid = vs30_valid[i]
                    sl = slice(pga_index * self.pga_targets, (pga_index + 1) * self.pga_targets)
                    pga_values_pre = sorted_pga[sl]
                    pga_valid_pre = sorted_valid[sl]
                    pga_targets_pre = sorted_metadata[sl, :]
                    if self.use_vs30:
                        pga_vs30_pre = sorted_vs30[sl]
                        pga_vs30_valid_pre = sorted_vs30_valid[sl]
                    if pga_targets_pre.shape[-1] == 4:
                        pga_targets_pre = pga_targets_pre[:, (0, 1, 3)]
                    n = len(pga_values_pre)
                    # Replace NaN/Inf with 0 in label tensor; loss will mask via pga_target_valid.
                    pga_values[i, :n] = np.where(pga_valid_pre, pga_values_pre, 0.0)
                    pga_targets[i, :n, :] = pga_targets_pre
                    pga_target_valid[i, :n] = pga_valid_pre
                    pga_target_indices[i, :n] = np.arange(sl.start, sl.start + n, dtype=np.int64)
                    if self.use_vs30:
                        pga_target_vs30[i, :n, 0] = np.where(pga_vs30_valid_pre, pga_vs30_pre, 0.0)
                        pga_target_vs30_valid[i, :n] = pga_vs30_valid_pre
            else:
                # Slice station_valid to match metadata/pga slicing above (462-464).
                if not self.pga_from_inactive:
                    sv_for_pga = station_valid_full[:, :self.max_stations]
                else:
                    sv_for_pga = station_valid_full
                for i in range(waveforms.shape[0]): # ith event ,zb
                    valid_pos = pga_valid_full[i] & sv_for_pga[i]
                    active = np.where(valid_pos)[0]
                    active_target_types = None
                    if len(active) == 0:
                        raise ValueError(f'Found event without PGA idx={indexes[i]}')
                    if self.select_first_pga_targets:
                        active_p_picks = full_p_picks[i, active].copy()
                        bad = np.logical_or(active_p_picks <= 0, active_p_picks > self.p_pick_limit)
                        active_p_picks[bad] = min(np.max(active_p_picks), self.p_pick_limit)
                        active = active[np.argsort(active_p_picks)]
                    elif self.pga_selection_skew is not None and self.pga_selection_skew > 0:
                        active_p_picks = full_p_picks[i, active]
                        bad = np.logical_or(active_p_picks <= 0, active_p_picks > self.p_pick_limit)
                        active_p_picks[bad] = min(np.max(active_p_picks), self.p_pick_limit)
                        coeffs = np.exp(-active_p_picks / self.pga_selection_skew)
                        coeffs *= self._rng_random(rng, coeffs.shape)
                        active = active[np.argsort(-coeffs)]
                    elif self.pga_target_sampling in ('distance_stratified', 'distance_coverage'):
                        input_slots = np.where(input_station_valid_for_model[i])[0]
                        input_coords = metadata[i, input_slots]
                        active = self._stratified_by_nearest_input_distance(
                            active,
                            metadata[i],
                            input_coords,
                            self.pga_targets,
                            rng=rng,
                        )
                    elif self.pga_target_sampling == 'label_stratified':
                        active = self._label_stratified_pga_targets(
                            active,
                            pga[i],
                            self.pga_targets,
                            rng=rng,
                        )
                    elif self.realtime_enabled and self.realtime_target_sampling.get('enabled', False):
                        active, active_target_types = self._sample_realtime_pga_targets(
                            active,
                            input_station_valid_for_model[i],
                            full_p_picks[i],
                            realtime_info['current_sample'],
                            self.pga_targets,
                            rng=rng,
                        )
                    else:
                        self._rng_shuffle(rng, active)

                    samples = active[:self.pga_targets]
                    n = len(samples)
                    if self.realtime_enabled:
                        if active_target_types is None:
                            target_types = self._classify_realtime_targets(
                                samples,
                                input_station_valid_for_model[i],
                                full_p_picks[i],
                                realtime_info['current_sample'],
                            )
                        else:
                            target_types = active_target_types[:n]
                        picked_picks = full_p_picks[i, samples]
                        valid_picked_picks = self._realtime_pick_valid(picked_picks)
                        lead_time = np.full((n,), np.nan, dtype=np.float32)
                        lead_time[valid_picked_picks] = (
                            picked_picks[valid_picked_picks] - realtime_info['current_sample']
                        ) / float(self.sampling_rate)
                        realtime_target_type[i, :n] = target_types
                        realtime_target_lead_time[i, :n] = lead_time
                    if metadata.shape[-1] == 3:
                        pga_targets[i, :n, :] = metadata[i, samples, :]
                    else:
                        full_targets = metadata[i, samples]
                        pga_targets[i, :n, :] = full_targets[:, (0, 1, 3)]
                    pga_values[i, :n] = pga[i, samples]
                    pga_target_valid[i, :n] = True
                    pga_target_indices[i, :n] = samples
                    if self.use_vs30:
                        pga_target_vs30[i, :n, 0] = np.where(vs30_valid[i, samples], vs30[i, samples], 0.0)
                        pga_target_vs30_valid[i, :n] = vs30_valid[i, samples]
                    # Unfilled slots [n:] keep value=0 and valid=False; the loss masks them.
            pga_values = pga_values.reshape((true_batch_size, self.pga_targets, 1))
        elif self.use_vs30:
            pga_target_vs30 = np.zeros((true_batch_size, 0, 1), dtype=np.float32)
            pga_target_vs30_valid = np.zeros((true_batch_size, 0), dtype=bool)

        metadata = metadata[:, :self.max_stations]
        station_valid = input_station_valid_for_model.copy()

        # Mark stations whose waveform was zeroed by cutout / trigger_based as
        # invalid for the encoder. Waveform "all zero" is a safe sentinel here:
        # real seismic data is mean-subtracted but never identically zero across
        # all samples and channels; only explicit zeroing produces this state.
        has_signal = (np.abs(waveforms) > self.wave_eps).any(axis=(2, 3))
        station_valid &= has_signal
        pick_valid = (p_picks > 0) & (p_picks < waveforms.shape[2])
        station_valid &= pick_valid

        if self.station_blinding:
            blind_mask = np.zeros(waveforms.shape[:2], dtype=bool)
            for i in range(waveforms.shape[0]):
                active = np.where(station_valid[i])[0]
                if len(active) == 0:
                    continue
                blind_length = self._rng_int(rng, 0, len(active))
                self._rng_shuffle(rng, active)
                blind_mask[i, active[:blind_length]] = True
            waveforms[blind_mask] = 0
            station_valid &= ~blind_mask
            # Note: metadata is intentionally NOT zeroed at blinded positions —
            # the model uses station_valid for masking, not the value.

        station_snr = None
        if self.station_experiment.get('enabled', False):
            pga_valid_input = ~(np.isnan(pga[:, :self.max_stations]) | np.isinf(pga[:, :self.max_stations]))
            station_valid, station_snr = self._apply_station_experiment_inputs(
                waveforms, p_picks, station_valid, pga_valid_input
            )
            experiment_targets = self._build_station_experiment_pga_targets(
                metadata_for_pga,
                pga_for_targets,
                pga_valid_for_targets,
                station_valid_for_targets,
                station_valid,
                selected_input_indices,
                full_selected_indices_for_targets,
                return_indices=self.use_vs30,
            )
            if experiment_targets is not None:
                if self.use_vs30:
                    pga_targets, pga_values, pga_target_valid, experiment_target_indices = experiment_targets
                    pga_target_vs30 = np.zeros((true_batch_size, self.pga_targets, 1), dtype=np.float32)
                    pga_target_vs30_valid = np.zeros((true_batch_size, self.pga_targets), dtype=bool)
                    for i in range(true_batch_size):
                        valid_idx = experiment_target_indices[i] >= 0
                        target_idx = experiment_target_indices[i, valid_idx]
                        pga_target_vs30[i, valid_idx, 0] = np.where(
                            vs30_valid[i, target_idx],
                            vs30[i, target_idx],
                            0.0,
                        )
                        pga_target_vs30_valid[i, valid_idx] = vs30_valid[i, target_idx]
                else:
                    pga_targets, pga_values, pga_target_valid = experiment_targets

        dpk_prior_cache_missing_count = 0
        dpk_prior_cache_available_count = 0
        if (
            self.dpk_prior_cache is not None
            and realtime_info is not None
            and self.dpk_prior_cache_filter_missing_stations
        ):
            cache_available = self.dpk_prior_cache.station_available_mask(
                cache_split_for_sample,
                self.dpk_prior_cache_dataset_id,
                str(ith_event),
                realtime_info['current_sample'],
                selected_original_input_indices[0],
                station_valid[0],
            )
            valid_before_cache = station_valid[0].copy()
            missing_cache = valid_before_cache & ~cache_available
            dpk_prior_cache_missing_count = int(missing_cache.sum())
            dpk_prior_cache_available_count = int((valid_before_cache & cache_available).sum())
            if dpk_prior_cache_missing_count:
                station_valid[0, missing_cache] = False
                waveforms[0, missing_cache, :, :] = 0.0
                if self.use_vs30:
                    vs30_valid[0, missing_cache] = False

        input_pga_values = None
        input_pga_valid = None
        if has_pga_values:
            input_pga = pga[:, :self.max_stations]
            input_pga_valid = ~(np.isnan(input_pga) | np.isinf(input_pga))
            input_pga_valid &= station_valid
            input_pga_values = np.where(input_pga_valid, input_pga, 0.0)

        # Sanity check: at least one station must have a real waveform to avoid
        # degenerate forward passes (all-zero input → NaN in energy loss).
        # If violated, skip to the next sample.
        for i in range(waveforms.shape[0]):
            if not station_valid[i].any():
                import warnings
                warnings.warn(
                    f'Event {ith_event} has no station with nonzero waveform '
                    f'after cutout — skipping sample.'
                )
                raise _EmptySample()

        if self.fake_borehole and waveforms.shape[3] == 3:
            waveforms = np.concatenate([np.zeros_like(waveforms), waveforms], axis=3)
            metadata_new = np.zeros(metadata.shape[:-1] + (4,))
            metadata_new[:, :, 0] = metadata[:, :, 0]
            metadata_new[:, :, 1] = metadata[:, :, 1]
            metadata_new[:, :, 3] = metadata[:, :, 2]
            metadata = metadata_new

        if self.use_vs30:
            station_vs30_np = vs30[:, :self.max_stations]
            station_vs30_valid_np = vs30_valid[:, :self.max_stations]
            station_vs30_valid_np &= np.isfinite(station_vs30_np) & (station_vs30_np > 0)
            station_vs30_valid_np &= station_valid
            station_vs30_np = np.where(station_vs30_valid_np, station_vs30_np, 0.0).astype(np.float32, copy=False)

        loc_target_abs = None
        loc_center = None
        if self.coords_target and target is not None:
            coords3 = self._coords3_from_metadata(metadata)
            loc_center = self._masked_coord_center(coords3, station_valid)
            loc_target_abs = target.copy()
            target = self._finalize_loc_target(target, metadata, station_valid)

        # Convert to Torch tensors
        waveforms = torch.from_numpy(np.swapaxes(waveforms[0],1,2)).float() # shape (nstation, channel, length)
        metadata = torch.from_numpy(metadata[0]).float()
        magnitude = torch.from_numpy(magnitude[0]).float()
        station_valid_t = torch.from_numpy(station_valid[0]).bool()

        inputs = [waveforms, metadata, station_valid_t]
        outputs = []
        if not self.no_event_token:
            outputs += [magnitude[0]]

            if self.coords_target:
#                target = np.expand_dims(target, axis=-1)
                target = torch.from_numpy(target[0]).float()
                outputs += [target]

        if self.pga_targets:
            pga_targets = torch.from_numpy(pga_targets[0]).float()
            pga_values = torch.from_numpy(pga_values[0]).float()
            pga_target_valid_t = torch.from_numpy(pga_target_valid[0]).bool()
            inputs += [pga_targets, pga_target_valid_t]
            outputs += [pga_values]
        elif self.use_vs30:
            inputs += [
                torch.zeros((0, 3), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.bool),
            ]

        if self.use_vs30:
            station_vs30_t = torch.from_numpy(station_vs30_np[0, :, None]).float()
            station_vs30_valid_t = torch.from_numpy(station_vs30_valid_np[0]).bool()
            pga_target_vs30_t = torch.from_numpy(pga_target_vs30[0]).float()
            pga_target_vs30_valid_t = torch.from_numpy(pga_target_vs30_valid[0]).bool()
            inputs += [station_vs30_t, station_vs30_valid_t, pga_target_vs30_t, pga_target_vs30_valid_t]

        if self.dpk_prior_cache is not None:
            cache_split = cache_split_for_sample
            cached_weights = self.dpk_prior_cache.lookup_sample(
                cache_split,
                self.dpk_prior_cache_dataset_id,
                sample_index_for_cache,
                station_valid[0],
                event_id=str(ith_event),
                realtime_current_sample=(
                    realtime_info['current_sample'] if realtime_info is not None else None
                ),
                original_station_indices=selected_original_input_indices[0],
            )
            inputs += [torch.from_numpy(cached_weights).float()]

        p_pick_info = {
            'shifted': torch.from_numpy(p_picks[0]).float(),
            'raw': torch.from_numpy(raw_p_picks[0]).float(),
            'shift': torch.tensor(float(shift), dtype=torch.float32),
            'event_id': str(ith_event),
            'selected_input_indices': torch.from_numpy(selected_input_indices[0]).long(),
            'selected_original_input_indices': torch.from_numpy(selected_original_input_indices[0]).long(),
            'original_station_indices': torch.from_numpy(selected_original_input_indices[0]).long(),
        }
        if self.dpk_prior_cache is not None:
            p_pick_info['dpk_prior_cache_path'] = self.dpk_prior_cache.path
            p_pick_info['dpk_prior_cache_split'] = str(cache_split)
            p_pick_info['dpk_prior_cache_sample_index'] = torch.tensor(
                sample_index_for_cache,
                dtype=torch.long,
            )
            if realtime_info is not None:
                p_pick_info['dpk_prior_cache_realtime_current_sample'] = torch.tensor(
                    int(realtime_info['current_sample']),
                    dtype=torch.long,
                )
                p_pick_info['dpk_prior_cache_requested_current_sample'] = torch.tensor(
                    int(realtime_info.get(
                        'cache_requested_current_sample',
                        realtime_info['current_sample'],
                    )),
                    dtype=torch.long,
                )
                p_pick_info['dpk_prior_cache_current_sample_delta'] = torch.tensor(
                    int(realtime_info.get('cache_current_sample_delta', 0)),
                    dtype=torch.long,
                )
                p_pick_info['dpk_prior_cache_missing_station_count'] = torch.tensor(
                    dpk_prior_cache_missing_count,
                    dtype=torch.long,
                )
                p_pick_info['dpk_prior_cache_available_station_count'] = torch.tensor(
                    dpk_prior_cache_available_count,
                    dtype=torch.long,
                )
        if self.pga_targets:
            p_pick_info['pga_target_indices'] = torch.from_numpy(pga_target_indices[0]).long()
        if self.realtime_enabled:
            p_pick_info['realtime_elapsed_time'] = torch.tensor(
                float(realtime_info['elapsed_time']),
                dtype=torch.float32,
            )
            p_pick_info['realtime_requested_elapsed_time'] = torch.tensor(
                float(realtime_info['requested_elapsed_time']),
                dtype=torch.float32,
            )
            p_pick_info['realtime_current_sample'] = torch.tensor(
                int(realtime_info['current_sample']),
                dtype=torch.long,
            )
            p_pick_info['realtime_first_p_pick_sample'] = torch.tensor(
                int(realtime_info['first_p_pick_sample']),
                dtype=torch.long,
            )
            p_pick_info['realtime_time_bin'] = torch.tensor(
                int(realtime_info['time_bin']),
                dtype=torch.long,
            )
            if self.pga_targets:
                p_pick_info['realtime_target_type'] = torch.from_numpy(realtime_target_type[0]).long()
                p_pick_info['realtime_target_lead_time'] = torch.from_numpy(realtime_target_lead_time[0]).float()
        if station_snr is not None:
            p_pick_info['station_snr'] = torch.from_numpy(station_snr[0]).float()
        if input_pga_values is not None:
            p_pick_info['input_pga_values'] = torch.from_numpy(input_pga_values[0]).float()
            p_pick_info['input_pga_valid'] = torch.from_numpy(input_pga_valid[0]).bool()
        if self.dump_debug_snapshot:
            raw_pga_valid = ~(np.isnan(debug_raw_pga[0]) | np.isinf(debug_raw_pga[0]))
            p_pick_info['debug_raw_waveforms'] = torch.from_numpy(np.swapaxes(debug_raw_waveforms[0], 1, 2)).float()
            p_pick_info['debug_raw_metadata'] = torch.from_numpy(debug_raw_metadata[0]).float()
            p_pick_info['debug_raw_station_valid'] = torch.from_numpy(debug_raw_station_valid[0]).bool()
            p_pick_info['debug_raw_pga'] = torch.from_numpy(np.where(raw_pga_valid, debug_raw_pga[0], 0.0)).float()
            p_pick_info['debug_raw_pga_valid'] = torch.from_numpy(raw_pga_valid).bool()
            p_pick_info['debug_raw_full_p_picks'] = torch.from_numpy(debug_raw_full_p_picks[0]).float()
        if loc_target_abs is not None:
            p_pick_info['loc_target_abs'] = torch.from_numpy(loc_target_abs[0]).float()
            p_pick_info['loc_center'] = torch.from_numpy(loc_center[0, 0]).float()
            p_pick_info['loc_target_mode'] = self.loc_target_mode
        return inputs, outputs, p_pick_info

    @staticmethod
    def _coords3_from_metadata(metadata):
        if metadata.shape[-1] == 4:
            return metadata[:, :, (0, 1, 3)]
        return metadata[:, :, :3]

    @staticmethod
    def _masked_coord_center(coords, station_valid):
        weights = station_valid[..., None].astype(coords.dtype)
        denom = np.clip(weights.sum(axis=1, keepdims=True), a_min=1.0, a_max=None)
        return (coords * weights).sum(axis=1, keepdims=True) / denom

    def _finalize_loc_target(self, target_abs, metadata, station_valid):
        if self.loc_target_mode == 'abs':
            return target_abs
        coords3 = self._coords3_from_metadata(metadata)
        center = self._masked_coord_center(coords3, station_valid)
        return target_abs - center[:, 0, :]

    def on_epoch_end(self):
        if self.realtime_enabled:
            repeated_indexes = np.repeat(self.base_indexes.copy(), self.oversample, axis=0)
            realtime_indexes = []
            if self.realtime_training['mode'] == 'val':
                for idx in repeated_indexes:
                    for time_index in range(len(self.realtime_training['val_times'])):
                        realtime_indexes.append((int(idx), -int(time_index) - 1))
            elif self.realtime_training.get('train_time_mode') == 'fixed':
                for idx in repeated_indexes:
                    for time_index in range(len(self.realtime_training['train_times'])):
                        realtime_indexes.append((int(idx), -int(time_index) - 1))
            else:
                n_bins = len(self.realtime_training['train_time_bins'])
                n_draw = int(self.realtime_training['bins_per_event_per_epoch'])
                replace = (
                    self.realtime_training['bin_sampling'] == 'with_replacement'
                    or n_draw > n_bins
                )
                for repeat_pos, idx in enumerate(repeated_indexes):
                    bin_rng = self._sample_rng((
                        int(idx),
                        int(self._realtime_epoch),
                        int(repeat_pos),
                        104729,
                    ))
                    if bin_rng is None:
                        chosen_bins = np.random.choice(n_bins, size=n_draw, replace=replace)
                    else:
                        chosen_bins = bin_rng.choice(n_bins, size=n_draw, replace=replace)
                    for draw_id, bin_index in enumerate(chosen_bins):
                        realtime_indexes.append((
                            int(idx),
                            int(bin_index),
                            int(self._realtime_epoch),
                            int(draw_id),
                        ))
            if self.shuffle:
                shuffle_rng = self._sample_rng((
                    int(self._realtime_epoch),
                    int(len(realtime_indexes)),
                    13007,
                ))
                self._rng_shuffle(shuffle_rng, realtime_indexes)
            self.indexes = realtime_indexes
            if self.realtime_training['mode'] == 'train':
                self._realtime_epoch += 1
            return
        self.indexes = np.repeat(self.base_indexes.copy(), self.oversample, axis=0)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def location_transformation(self, metadata, target=None, station_valid=None):
        transform_target_only = self.transform_target_only
        metadata = metadata.copy()

        metadata_old = metadata
        metadata = metadata.copy()
        # Padding mask now comes from the explicit station_valid array.
        # This avoids treating legitimate (lat=0, lon=0, depth=0) coords as padding.
        if station_valid is not None:
            mask = ~station_valid
        else:
            mask = np.zeros(metadata.shape[:2], dtype=bool)
        if target is not None:
            target[:, 0] -= self.pos_offset[0]
            target[:, 1] -= self.pos_offset[1]
        metadata[:, :, 0] -= self.pos_offset[0]
        metadata[:, :, 1] -= self.pos_offset[1]

        # Coordinates to kilometers (assuming a flat earth, which is okay close to equator)
        if self.scale_metadata:
            metadata[:, :, :2] *= D2KM
            if target is not None:
                target[:, :2] *= D2KM

        metadata[mask] = 0

        if self.scale_metadata:
            metadata /= 100
            if target is not None:
                target /= 100

        if transform_target_only:
            metadata = metadata_old

        if target is None:
            return metadata
        else:
            return metadata, target


class JointGenerator(Dataset):
    def __init__(self, generators=(), shuffle=True, dataset_id=False, fake_id=None):
        assert len(generators)
        self.generators = generators
        self.indexes = None
        self.shuffle = shuffle
        self.dataset_id = dataset_id
        self.fake_id = fake_id
        self.on_epoch_end()

    def __len__(self):
        return sum(len(generator) for generator in self.generators)

    def __getitem__(self, index):
        generator_id, batch_id = self.indexes[index]
        batch_inp, batch_out, batch_info = self.generators[generator_id][batch_id]

        if self.dataset_id:
            # Per-sample scalar (collate will stack into shape (B,)) — matches
            # FullModel.dataset_embedding which expects a 1D index tensor.
            id_value = generator_id if self.fake_id is None else self.fake_id
            dataset_id = torch.tensor(id_value, dtype=torch.long)
            batch_inp += [dataset_id]
        return batch_inp, batch_out, batch_info

    def on_epoch_end(self):
        self.indexes = []
        for i, generator in enumerate(self.generators):
            self.indexes += [(i, j) for j in range(len(generator))]
        if self.shuffle:
            np.random.shuffle(self.indexes)


class CutoutGenerator(Dataset):
    def __init__(self, generator, times, sampling_rate):
        self.generator = generator
        self.times = times
        self.sampling_rate = sampling_rate
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        return len(self.generator) * len(self.times)

    def __getitem__(self, index):
        time, batch_id = self.indexes[index]
        cutout = int(self.sampling_rate * (time + 5))
        self.generator.cutout = (cutout, cutout + 1)
        return self.generator[batch_id]

    def on_epoch_end(self):
        self.indexes = []
        for time in self.times:
            self.indexes += [(time, i) for i in range(len(self.generator))]


def gaussian_mixture(x, params, eps=1e-6, memory_save=True, fortran=True):
    if fortran and not np.isfortran(params):
        params = np.array(params, order='F')
    if not memory_save:
        return gaussian_mixture_fully_vectorized(x, params, eps)
    if params.ndim == 2:
        alpha = params[:, 0]
        mu = params[:, 1]
        sigma = np.maximum(params[:, 2], eps)
        density = np.zeros_like(x)
        for i in range(alpha.shape[0]):
            density += alpha[i] * 1 / (np.sqrt(2 * np.pi) * sigma[i]) * np.exp(-(x - mu[i]) ** 2 / (2 * sigma[i] ** 2))
    elif params.ndim == 3:
        alpha = np.expand_dims(params[:, :, 0], axis=0)
        mu = np.expand_dims(params[:, :, 1], axis=0)
        sigma = np.expand_dims(np.maximum(params[:, :, 2], eps), axis=0)
        x = np.reshape(x, (-1, 1))
        density = np.zeros((x.shape[0], alpha.shape[1]))
        for i in range(alpha.shape[2]):
            density += alpha[:, :, i] * 1 / (np.sqrt(2 * np.pi) * sigma[:, :, i]) * np.exp(
                -(x - mu[:, :, i]) ** 2 / (2 * sigma[:, :, i] ** 2))
        density = density.T
    else:
        raise ValueError('Params ndim must be 2 or 3')

    return density


def gaussian_mixture_fully_vectorized(x, params, eps=1e-6):
    if params.ndim == 2:
        alpha = np.reshape(params[:, 0], (1, -1))
        mu = np.reshape(params[:, 1], (1, -1))
        sigma = np.reshape(np.maximum(params[:, 2], eps), (1, -1))
        x = np.reshape(x, (-1, 1))
        density = alpha * 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
        density = np.sum(density, axis=1)
    elif params.ndim == 3:
        alpha = np.expand_dims(params[:, :, 0], axis=0)
        mu = np.expand_dims(params[:, :, 1], axis=0)
        sigma = np.expand_dims(np.maximum(params[:, :, 2], eps), axis=0)
        x = np.reshape(x, (-1, 1, 1))
        density = alpha * 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
        density = np.sum(density, axis=2).T
    else:
        raise ValueError('Params ndim must be 2 or 3')

    return density


def filter_shard(events, shard_id, shards):
    if not shards:
        return events
    if isinstance(events, pd.DataFrame):
        events = events.iloc[shard_id::shards]
    else:
        return events[shard_id::shards]
    return events


def merge_hdf5(inputs, output, event_id, sort_key=None):
    inputs = sorted(glob.glob(inputs))
    print(inputs)
    metadata = {}
    delete_keys = []
    existing_keys = []
    catalog = None
    with h5py.File(output, 'w') as fout:
        gout_data = fout.create_group('data')
        gout_meta = fout.create_group('metadata')
        for inp in tqdm(inputs):
            with h5py.File(inp, 'r') as fin:
                for key in fin['metadata'].keys():
                    if key == 'event_metadata':
                        continue
                    tmp = fin['metadata'][key].value
                    if key in metadata:
                        if isinstance(metadata[key] == tmp, (bool, np.bool_, np.bool)):
                            assert metadata[key] == tmp
                        else:
                            assert all(metadata[key] == tmp)
                    metadata[key] = tmp

                for event in fin['data']:
                    existing_keys += [event]
                    if event in gout_data:
                        delete_keys += [event]
                        del gout_data[event]
                        continue
                    gout_event = gout_data.create_group(event)
                    for ds in fin['data'][event]:
                        gout_event.create_dataset(ds, data=fin['data'][event][ds].value)

            if catalog is None:
                catalog = pd.read_hdf(inp, 'metadata/event_metadata')
            else:
                catalog = pd.concat([catalog, pd.read_hdf(inp, 'metadata/event_metadata')])

        for key, val in metadata.items():
            gout_meta.create_dataset(key, data=val)

    if sort_key is None:
        sort_key = event_id

    catalog = catalog.sort_values(by=sort_key)
    catalog = catalog[~catalog[event_id].isin(delete_keys)]
    catalog = catalog[catalog[event_id].isin(existing_keys)]
    print(f'Removed {len(delete_keys)} redundant events')
    catalog.to_hdf(output, key='metadata/event_metadata', mode='a', encoding='utf-8', format='table')


def detect_location_keys(columns):
    candidates = [['LAT', 'Latitude(°)', 'Latitude'],
                  ['LON', 'Longitude(°)', 'Longitude'],
                  ['DEPTH', 'JMA_Depth(km)', 'Depth(km)', 'Depth/Km']]

    coord_keys = []
    for keyset in candidates:
        for key in keyset:
            if key in columns:
                coord_keys += [key]
                break

    if len(coord_keys) != len(candidates):
        raise ValueError('Unknown location key format')

    return coord_keys


def wait_for_file(path, sleep_seconds=600, silent=False):
    while not os.path.exists(path):
        if not silent:
            print(f'File {path} for weight transfer missing. Sleeping for {sleep_seconds} seconds.')
        time.sleep(sleep_seconds)


def normalize_gain(stream, inv, key='ACC'):
    delete_traces = []
    for trace in stream:
        try:
            resp = inv.get_response(trace.id, trace.stats.starttime)
        except:
            # get_response throws plain Exception, therefore bare except is required
            print(f'Missing response for {trace.id}')
            delete_traces += [trace]
            continue
        try:
            sensitivity = np.abs(resp.get_evalresp_response_for_frequencies([1.], key))[0]
            reported_sensitivity = resp.instrument_sensitivity.value
            if np.abs(sensitivity - reported_sensitivity) / max(sensitivity, reported_sensitivity) > 0.05:
                print(f'Sensitivity mismatch for station {trace.stats.network}.{trace.stats.station}. '
                      f'Reported: {reported_sensitivity}\tComputed: {sensitivity}\tFactor: {reported_sensitivity / sensitivity}')
                sensitivity = reported_sensitivity
            trace.data = trace.data / sensitivity
        except ValueError:
            # Illegal RESP format
            trace.data = trace.data * 0.0

    for trace in delete_traces:
        stream.remove(trace)


def generator_from_config(config, data, event_metadata, time, batch_size=64, sampling_rate=100, pga=False,
                          dataset_id=None):
    training_params = config['training_params']

    if dataset_id is not None:
        generator_params = training_params.get('generator_params', [training_params.copy()])[dataset_id]
    else:
        generator_params = training_params.get('generator_params', [training_params.copy()])[0]

    noise_seconds = generator_params.get('noise_seconds', 5)
    cutout = int(sampling_rate * (noise_seconds + time))
    cutout = (cutout, cutout + 1)

    n_pga_targets = config['model_params'].get('n_pga_targets', 0)
    max_stations = config['model_params']['max_stations']
    generator_params['magnitude_resampling'] = 1
    generator_params['batch_size'] = batch_size
    generator_params['transform_target_only'] = generator_params.get('transform_target_only', True)
    generator_params['upsample_high_station_events'] = None
    generator_params['use_coords_rel'] = config['model_params'].get('use_coords_rel', False)
    generator_params['use_coords_abs'] = config['model_params'].get('use_coords_abs', True)
    generator_params['use_coords_rel_abs_fusion'] = config['model_params'].get('use_coords_rel_abs_fusion', False)
    generator_params['use_vs30'] = config['model_params'].get('use_vs30', False)
    generator_params['select_first_inputs'] = generator_params.get(
        'select_first_inputs', generator_params.get('select_first', True)
    )
    generator_params['select_first_pga_targets'] = generator_params.get(
        'select_first_pga_targets', generator_params.get('select_first', True)
    )
    if generator_params.get('coord_keys', None) is not None:
        raise NotImplementedError('Fixed coordinate keys are not implemented in location evaluation')
    generator_params['translate'] = False
    dpk_prior_cache = None
    dpk_prior_cache_cfg = training_params.get('dpk_prior_cache', None)
    if dpk_prior_cache_cfg and dpk_prior_cache_cfg.get('enabled', False):
        paths = dpk_prior_cache_cfg.get('paths', {})
        if isinstance(paths, dict):
            cache_path = paths.get('dev') or paths.get('val') or paths.get('validation')
        else:
            cache_path = paths
        cache_path = cache_path or dpk_prior_cache_cfg.get('dev_path') or dpk_prior_cache_cfg.get('val_path')
        if not cache_path:
            raise ValueError(
                'dpk_prior_cache is enabled but no dev/val cache path is configured for generator_from_config.'
            )
        dpk_prior_cache = DPKPriorCache(
            cache_path,
            source=dpk_prior_cache_cfg.get('source', 'dpk_finetuned'),
            mode=dpk_prior_cache_cfg.get('mode', 'event'),
            token_floor=dpk_prior_cache_cfg.get('token_floor', dpk_prior_cache_cfg.get('floor', 1e-4)),
            missing_policy=dpk_prior_cache_cfg.get('missing_policy', 'error'),
        )
    generator = PreloadedEventGenerator(data=data,
                                        event_metadata=event_metadata,
                                        coords_target=True,
                                        cutout=cutout,
                                        pga_targets=n_pga_targets,
                                        max_stations=max_stations,
                                        sampling_rate=sampling_rate,
                                        shuffle=False,
                                        pga_mode=pga,
                                        dpk_prior_cache=dpk_prior_cache,
                                        dpk_prior_cache_split='dev',
                                        dpk_prior_cache_dataset_id=0 if dataset_id is None else dataset_id,
                                        dpk_prior_cache_align_realtime=(
                                            (dpk_prior_cache_cfg or {}).get('align_realtime_to_cache', True)
                                        ),
                                        dpk_prior_cache_filter_missing_stations=(
                                            (dpk_prior_cache_cfg or {}).get('filter_missing_stations', True)
                                        ),
                                        **generator_params)
    if dataset_id is not None and config['model_params'].get('dataset_bias', False):
        generator = JointGenerator([generator], shuffle=False, dataset_id=True, fake_id=dataset_id)
    return generator

def bin_search_quantile(pred, quantile, vmin=0, vmax=10):
    min_val = vmin * np.ones((pred.shape[0], 1))
    max_val = vmax * np.ones((pred.shape[0], 1))

    for _ in range(14):
        # 14 iterations mean approximation error is below 0.001
        mean_val = (min_val + max_val) / 2
        prob = np.sum(
            pred[:, :, 0] * (1 - norm.cdf((mean_val - pred[:, :, 1]) / pred[:, :, 2])),
            axis=-1, keepdims=True)
        min_val = np.where(prob > quantile, mean_val, min_val)
        max_val = np.where(prob <= quantile, mean_val, max_val)

    pred = np.squeeze(min_val, axis=-1)

    return pred
