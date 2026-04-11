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

# warnings.simplefilter("ignore", UserWarning)

D2KM = 111.19492664455874


class _EmptySample(Exception):
    """Internal signal: this index has no valid station after cutout; skip it."""
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


class DataGenerator(Dataset):
    def __init__(self, event_metadata, metadata, data_path, generator_params, cutout=None, sliding_window=False, windowlen=10000, data_keys=None, overwrite_sampling_rate=None,
                 shuffle=True, label_smoothing=False, oversample=1):
        self.event_metadata = event_metadata
        self.metadata = metadata
        self.data_path = data_path
        self.target_key = generator_params['key']
        self.data_keys = data_keys
        self.overwrite_sampling_rate = overwrite_sampling_rate

        self.dist = None
        self.cutout = cutout
        self.sliding_window = sliding_window  # If true, selects sliding windows instead of cutout. Uses cutout as values for end of window.
        self.windowlen = windowlen  # Length of window for sliding window
        self.shuffle = shuffle
        self.oversample = oversample
        self.indexes = np.arange(len(self.event_metadata))
        self.label_smoothing = label_smoothing
        for event_key in ['KiK_File', '#EventID', 'EVENT']:
            if event_key in event_metadata.columns:
                self.event_key = event_key
                break
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
            for key in g_event:
                if self.data_keys is not None and key not in self.data_keys:
                    continue
                if key not in data:
                    data[key] = []
                if key == 'waveforms':
                    # pad or truncate to 10000 points
                    cur_waveform = g_event[key][event['wave_idx']:event['wave_idx']+1, ::self.decimate, :]
                    cur_waveform -= np.mean(cur_waveform, axis=1, keepdims=True)
                    if cur_waveform.shape[1] < 10000:
                        pad_arr = np.zeros((cur_waveform.shape[0],
                                            10000 - cur_waveform.shape[1],
                                            cur_waveform.shape[2]))
                        data[key] += [np.concatenate((cur_waveform, pad_arr),
                                                     axis=1)]
                    else:
                        data[key] += [cur_waveform[:, :10000, :]]
                else:
                    data[key] += [g_event[key][()]]
                if key == 'p_picks':
                    data[key][-1] //= self.decimate

        X = data['waveforms'][0]
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
                 select_first=False, fake_borehole=False, scale_metadata=True, pga_key='pga',
                 pga_mode=False, p_pick_limit=5000, coord_keys=None, upsample_high_station_events=None,
                 no_event_token=False, pga_selection_skew=None, **kwargs):
        if kwargs:
            print(f'Unused parameters: {", ".join(kwargs.keys())}')
        self.shuffle = shuffle
#        self.waveforms = data['waveforms']
#        self.metadata = data['coords']
        self.event_metadata = event_metadata
        for event_key in ['KiK_File', '#EventID', 'EVENT']:
            if event_key in event_metadata.columns:
                self.event_key = event_key
                break
        self.event_metadata = self.event_metadata.groupby(event_key)
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
        self.target_key = generator_params['key']
        self.data_keys = data_keys
        self.pga_key = pga_key
#        if pga_key in data:
#            self.pga = data[pga_key]
#        else:
#            print('Found no PGA values')
#            self.pga = [np.zeros(x.shape[0]) for x in self.waveforms]
        self.key = key
        self.cutout = cutout
        self.sliding_window = sliding_window  # If true, selects sliding windows instead of cutout. Uses cutout as values for end of window.
        self.windowlen = windowlen  # Length of window for sliding window
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
        self.trigger_based = trigger_based
        self.disable_station_foreshadowing = disable_station_foreshadowing
        self.selection_skew = selection_skew
        self.pga_from_inactive = pga_from_inactive
        self.pga_selection_skew = pga_selection_skew
        self.integrate = integrate
        self.sampling_rate = sampling_rate
        self.select_first = select_first
        self.fake_borehole = fake_borehole
        self.scale_metadata = scale_metadata
        self.upsample_high_station_events = upsample_high_station_events
        self.no_event_token = no_event_token

#        if 'p_picks' in data:
#            self.triggers = data['p_picks']
#        else:
#            print('Found no picks')
#            self.triggers = [np.zeros(x.shape[0]) for x in self.waveforms]

        # Extend samples to include all pga targets in each epoch
        # PGA mode is only for evaluation, as it adds zero padding to the input/pga target!
        self.pga_mode = pga_mode
        self.p_pick_limit = p_pick_limit

        self.base_indexes = np.arange(len(self.event_metadata))
        self.event_keys = list(self.event_metadata.groups.keys())
        self.reverse_index = None
        if magnitude_resampling > 1:
            magnitude = self.event_metadata[key].mean().values
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
        return self.indexes.shape[0]

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
        indexes = self.indexes[index:(index + 1)]
        index = self.indexes[index]
        ith_event = self.event_keys[index]
        if self.pga_mode:
            pga_indexes = [x[1] for x in indexes]
            indexes = [x[0] for x in indexes]
            index = indexes[0]
        with h5py.File(self.data_path, 'r') as f:
            event = self.event_metadata.get_group(ith_event)
            event_name = str(event[self.event_key].iloc[0]) # ith_event? zb
            g_event = f['data'][event_name]
            data = {}
            for key in g_event:
                if self.data_keys is not None and key not in self.data_keys:
                    continue
                if key not in data:
                    data[key] = []
                if key == 'waveforms':
                    # pad to 10000 points
                    cur_waveform = g_event[key][:, ::self.decimate, :]
                    cur_waveform -= np.mean(cur_waveform, axis=1, keepdims=True)
                    if cur_waveform.shape[1] < 10000:
                        pad_arr = np.zeros((cur_waveform.shape[0],
                                            10000 - cur_waveform.shape[1],
                                            cur_waveform.shape[2]))
                        data[key] += [np.concatenate((cur_waveform, pad_arr),
                                                     axis=1)]
                    else:
                        data[key] += [cur_waveform]
                else:
                    data[key] += [g_event[key][()]]
                if key == 'p_picks':
                    data[key][-1] //= self.decimate

        X = np.concatenate(data['waveforms'], axis=0)
        self.metadata = np.concatenate(data['coords'], axis=0) # coords of stations (lat, lon, elev)
        self.waveforms = X

        if self.pga_key in data:
            self.pga = np.concatenate(data[self.pga_key], axis=0)
        else:
            print('Found no PGA values')
            self.pga = np.zeros(X.shape[0])

        if 'p_picks' in data:
            self.triggers = np.concatenate(data['p_picks'], axis=0)
        else:
            print('Found no picks')
            self.triggers = np.zeros(X.shape[0])

        y = np.array([self.event_metadata.get_group(ith_event)[self.target_key]]) # magnitude
        true_batch_size = 1

        waveforms = np.zeros((true_batch_size, self.max_stations) + X.shape[1:])  # shape (1, 25, 10000, 3)
        true_max_stations_in_batch = max(max([self.metadata.shape[0] for idx in indexes]), self.max_stations) # max(n_stations,25) = tms
        metadata = np.zeros((true_batch_size, true_max_stations_in_batch) + self.metadata.shape[1:]) # shape (1,tms, 3), coords
        # Use NaN for PGA so that "no measurement" is unambiguous and the legal
        # log-PGA value 0 is not confused with padding.
        pga = np.full((true_batch_size, true_max_stations_in_batch), np.nan)  # shape (1, tms)
        full_p_picks = np.zeros((true_batch_size, true_max_stations_in_batch)) # shape (1, tms)
        p_picks = np.zeros((true_batch_size, self.max_stations)) # shape (1, 25)
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
                p_picks[i, :len(self.triggers)] = self.triggers
                full_p_picks[i, :len(self.triggers)] = self.triggers
                station_valid_full[i, :len(self.metadata)] = True # all stations init to True
                reverse_selections += [[]]
            else:
                if self.selection_skew is None or self.selection_skew <= 0:  # random select
                    selection = np.arange(0, len(self.waveforms)) # all stations
                    np.random.shuffle(selection)
                else:  # pick_time + randomness
                    tmp_p_picks = self.triggers.copy()
                    mask = np.logical_or(tmp_p_picks <= 0, tmp_p_picks > self.p_pick_limit)
                    tmp_p_picks[mask] = min(np.max(tmp_p_picks), self.p_pick_limit)
                    coeffs = np.exp(-tmp_p_picks / self.selection_skew)
                    coeffs *= np.random.random(coeffs.shape)
                    coeffs[self.triggers == 0] = 0
                    coeffs[self.triggers > self.waveforms.shape[1]] = 0
                    selection = np.argsort(-coeffs)

                if self.select_first: # pick_time
                    selection = np.argsort(self.triggers)

                selection = selection[:true_max_stations_in_batch] # len tms
                metadata[i, :len(selection)] = self.metadata[selection]
                pga[i, :len(selection)] = self.pga[selection]
                full_p_picks[i, :len(selection)] = self.triggers[selection]
                station_valid_full[i, :len(selection)] = True # tms set to True

                tmp_reverse_selection = [0 for _ in selection]
                for j, s in enumerate(selection):
                    tmp_reverse_selection[s] = j
                reverse_selections += [tmp_reverse_selection]

                selection = selection[:self.max_stations]
                waveforms[i] = self.waveforms[selection]
                p_picks[i] = self.triggers[selection]

        # Defensive: mark stations with NaN/Inf coordinates as invalid. Current
        # KiK-Net data is clean, but this guards the model against upstream
        # corruption (NaN coord → NaN position embedding → NaN loss).
        coord_valid = ~(np.isnan(metadata).any(axis=-1) | np.isinf(metadata).any(axis=-1))
        station_valid_full &= coord_valid

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
        if self.cutout:
            if self.sliding_window:
                windowlen = self.windowlen
                window_end = np.random.randint(max(windowlen, self.cutout[0]),
                                              min(waveforms.shape[2], self.cutout[1]) + 1)
                waveforms = waveforms[:, :, window_end - windowlen: window_end]

                cutout = window_end
                if self.adjust_mean:
                    waveforms -= np.mean(waveforms, axis=2, keepdims=True)
            else:
                if self.cutout[0] == self.cutout[1]:
                    cutout = self.cutout[0]
                else:
                    cutout = np.random.randint(*self.cutout)
                if self.adjust_mean:
                    # Mean only over non-zero samples so that leading zero-padding
                    # is neither diluting the mean nor getting offset by it.
                    region = waveforms[:, :, :cutout + 1]                    # (B, S, T, C)
                    has_data = np.any(region != 0, axis=-1)                  # (B, S, T)
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

        if self.trigger_based:
            # Remove waveforms for all stations that did not trigger yet to avoid knowledge leakage
            p_picks[p_picks <= 0] = org_waveform_length  # Ensure that stations without P picks do not show data
            waveforms[cutout + shift < p_picks, :, :] = 0

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

        # PGA validity is detected via NaN/Inf so that the legal value 0 is preserved.
        pga_valid_full = ~(np.isnan(pga) | np.isinf(pga))

        if self.pga_targets:
            pga_values = np.zeros(
                (true_batch_size, self.pga_targets))
            pga_targets = np.zeros((true_batch_size, self.pga_targets, 3))
            pga_target_valid = np.zeros((true_batch_size, self.pga_targets), dtype=bool)
            if self.pga_mode:
                for i in range(waveforms.shape[0]):
                    pga_index = pga_indexes[i]
                    if len(reverse_selections[i]) > 0:
                        sorted_pga = pga[i, reverse_selections[i]]
                        sorted_metadata = metadata[i, reverse_selections[i]]
                        sorted_valid = pga_valid_full[i, reverse_selections[i]]
                    else:
                        sorted_pga = pga[i]
                        sorted_metadata = metadata[i]
                        sorted_valid = pga_valid_full[i]
                    sl = slice(pga_index * self.pga_targets, (pga_index + 1) * self.pga_targets)
                    pga_values_pre = sorted_pga[sl]
                    pga_valid_pre = sorted_valid[sl]
                    pga_targets_pre = sorted_metadata[sl, :]
                    if pga_targets_pre.shape[-1] == 4:
                        pga_targets_pre = pga_targets_pre[:, (0, 1, 3)]
                    n = len(pga_values_pre)
                    # Replace NaN/Inf with 0 in label tensor; loss will mask via pga_target_valid.
                    pga_values[i, :n] = np.where(pga_valid_pre, pga_values_pre, 0.0)
                    pga_targets[i, :n, :] = pga_targets_pre
                    pga_target_valid[i, :n] = pga_valid_pre
            else:
                # Slice station_valid to match metadata/pga slicing above (462-464).
                if not self.pga_from_inactive:
                    sv_for_pga = station_valid_full[:, :self.max_stations]
                else:
                    sv_for_pga = station_valid_full
                for i in range(waveforms.shape[0]): # ith event ,zb
                    valid_pos = pga_valid_full[i] & sv_for_pga[i]
                    active = np.where(valid_pos)[0]
                    if len(active) == 0:
                        raise ValueError(f'Found event without PGA idx={indexes[i]}')
                    if self.select_first:
                        active_p_picks = full_p_picks[i, active].copy()
                        bad = np.logical_or(active_p_picks <= 0, active_p_picks > self.p_pick_limit)
                        active_p_picks[bad] = min(np.max(active_p_picks), self.p_pick_limit)
                        active = active[np.argsort(active_p_picks)]
                    elif self.pga_selection_skew is not None and self.pga_selection_skew > 0:
                        active_p_picks = full_p_picks[i, active]
                        bad = np.logical_or(active_p_picks <= 0, active_p_picks > self.p_pick_limit)
                        active_p_picks[bad] = min(np.max(active_p_picks), self.p_pick_limit)
                        coeffs = np.exp(-active_p_picks / self.pga_selection_skew)
                        coeffs *= np.random.random(coeffs.shape)
                        active = active[np.argsort(-coeffs)]
                    else:
                        np.random.shuffle(active)

                    samples = active[:self.pga_targets]
                    n = len(samples)
                    if metadata.shape[-1] == 3:
                        pga_targets[i, :n, :] = metadata[i, samples, :]
                    else:
                        full_targets = metadata[i, samples]
                        pga_targets[i, :n, :] = full_targets[:, (0, 1, 3)]
                    pga_values[i, :n] = pga[i, samples]
                    pga_target_valid[i, :n] = True
                    # Unfilled slots [n:] keep value=0 and valid=False; the loss masks them.
            pga_values = pga_values.reshape((true_batch_size, self.pga_targets, 1))

        metadata = metadata[:, :self.max_stations]
        station_valid = station_valid_full[:, :self.max_stations].copy()

        # Mark stations whose waveform was zeroed by cutout / trigger_based as
        # invalid for the encoder. Waveform "all zero" is a safe sentinel here:
        # real seismic data is mean-subtracted but never identically zero across
        # all samples and channels; only explicit zeroing produces this state.
        has_signal = (waveforms != 0).any(axis=(2, 3))
        station_valid &= has_signal

        if self.station_blinding:
            blind_mask = np.zeros(waveforms.shape[:2], dtype=bool)
            for i in range(waveforms.shape[0]):
                active = np.where(station_valid[i])[0]
                if len(active) == 0:
                    continue
                blind_length = np.random.randint(0, len(active))
                np.random.shuffle(active)
                blind_mask[i, active[:blind_length]] = True
            waveforms[blind_mask] = 0
            station_valid &= ~blind_mask
            # Note: metadata is intentionally NOT zeroed at blinded positions —
            # the model uses station_valid for masking, not the value.

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

        return inputs, outputs, p_picks

    def on_epoch_end(self):
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
    if generator_params.get('coord_keys', None) is not None:
        raise NotImplementedError('Fixed coordinate keys are not implemented in location evaluation')
    generator_params['translate'] = False
    generator = PreloadedEventGenerator(data=data,
                                        event_metadata=event_metadata,
                                        coords_target=True,
                                        cutout=cutout,
                                        pga_targets=n_pga_targets,
                                        max_stations=max_stations,
                                        sampling_rate=sampling_rate,
                                        select_first=True,
                                        shuffle=False,
                                        pga_mode=pga,
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
