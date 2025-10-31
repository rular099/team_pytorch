import argparse
import copy
from operator import itemgetter
from typing import Any, List, Tuple, Union, Union
import numpy as np

from finetuneing.config import Config
from finetuneing.datasets import build_dataset
from finetuneing.datasets.lsd import LSD
from finetuneing.datasets.SFTData import SFTData,SFTDataSplitPS
from torch.utils.data import Dataset

import random
import json
import os

__all__ = ["Preprocessor", "SeismicDataset"]


def _pad_phases(
    ppks: list, spks: list, padding_idx: int, num_samples: int
) -> Tuple[list, list]:
    """
    Pad phase-P/S to ensure the two list have the same length.
    """
    padding_idx = abs(padding_idx)
    ppks, spks = sorted(ppks), sorted(spks)
    ppks_, spks_ = ppks.copy(), spks.copy()
    ppk_arr, spk_arr = np.array(ppks), np.array(sorted(spks))
    idx = 0
    while idx < min(len(ppks), len(spks)) and all(
        ppk_arr[: idx + 1] < spk_arr[-idx - 1 :]
    ):
        idx += 1
    ppks = len(spk_arr[: len(spk_arr) - idx]) * [-padding_idx] + ppks
    spks = spks + len(ppk_arr[idx:]) * [num_samples + padding_idx]

    assert len(ppks) == len(spks), f"Error:{ppks_} -> {ppks},{spks_} -> {spks}"
    return ppks, spks


def _pad_array(s: list, length: int, padding_value: Union[int, float]) -> np.ndarray:
    """
    Pad array with `padding_value`
    """
    padding_size = int(length - len(s))
    if padding_size >= 0:
        padded = np.pad(
            s, (0, padding_size), mode="constant", constant_values=padding_value
        )
        return padded
    else:
        raise Exception(f"`length < len(s)` . Array:{len(s)},Target:{length},s:{s}")


class DataPreprocessor:
    """
    Data preprocessor.

    Preprocess input data, perform data augmentation and generate labels.

    Reference:
        Some of the data augmentation methods, such as `_normalize`, `_adjust_amplitude`, `_scale_amplitude` and `_pre_emphasis`,
        are modified from: https://github.com/smousavi05/EQTransformer/blob/master/EQTransformer/core/EqT_uilts.py
    """

    def __init__(
        self,
        data_channels: int,
        sampling_rate: int,
        in_samples: int,
        min_snr: float,
        p_position_ratio: float,
        coda_ratio: float,
        norm_mode: str,
        add_event_rate: float,
        add_noise_rate: float,
        add_gap_rate: float,
        drop_channel_rate: float,
        scale_amplitude_rate: float,
        pre_emphasis_rate: float,
        pre_emphasis_ratio: float,
        max_event_num: float,
        generate_noise_rate: float,
        shift_event_rate: float,
        mask_percent: float,
        noise_percent: float,
        min_event_gap_sec: float,
        soft_label_shape: str,
        soft_label_width: int,
        operations: str,
        add_mask_window_rate: float,
        add_noise_window_rate: float,
        noise_type: str,
        whether_fixed_mask_percent: bool,
        lower_bound_mask_percent: float,
        upper_bound_mask_percent: float,
        whether_fixed_noise_percent: bool,
        lower_bound_noise_percent: float,
        upper_bound_noise_percent: float,
        shift_event_distance_percent: float,
        dtype,
        p_position_ratio_type,
        p_position_ratio_range_or_sigma,
    ):
        self.sampling_rate = sampling_rate
        self.data_channels = data_channels
        self.in_samples = in_samples
        self.coda_ratio = coda_ratio
        self.norm_mode = norm_mode
        self.min_snr = min_snr
        self.p_position_ratio = p_position_ratio
        self.p_position_ratio_type = p_position_ratio_type
        self.p_position_ratio_range_or_sigma = p_position_ratio_range_or_sigma
        self.add_event_rate = add_event_rate
        self.add_noise_rate = add_noise_rate
        self.add_gap_rate = add_gap_rate
        self.drop_channel_rate = drop_channel_rate
        self.scale_amplitude_rate = scale_amplitude_rate
        self.pre_emphasis_rate = pre_emphasis_rate
        self.pre_emphasis_ratio = pre_emphasis_ratio
        self._max_event_num = max_event_num
        self.generate_noise_rate = generate_noise_rate
        self.shift_event_rate = shift_event_rate
        self.mask_percent = mask_percent
        self.noise_percent = noise_percent
        self.min_event_gap = int(min_event_gap_sec * self.sampling_rate)
        # [Modified] data augmentation
        self.operations = operations#.split(",")
        self.add_mask_window_rate = add_mask_window_rate
        self.add_noise_window_rate = add_noise_window_rate
        self.whether_fixed_mask_percent = whether_fixed_mask_percent
        self.lower_bound_mask_percent = lower_bound_mask_percent
        self.upper_bound_mask_percent = upper_bound_mask_percent
        self.whether_fixed_noise_percent = whether_fixed_noise_percent
        self.lower_bound_noise_percent = lower_bound_noise_percent
        self.upper_bound_noise_percent = upper_bound_noise_percent
        self.noise_type = noise_type
        self.shift_event_distance_percent = shift_event_distance_percent

        if 0 <= self.p_position_ratio <= 1:
            if self.add_event_rate > 0:
                self.add_event_rate = 0.0
                print(
                    f"`p_position_ratio` is {p_position_ratio}, `add_event_rate` -> `0.0`"
                )

            # if self.shift_event_rate > 0:
            #     self.shift_event_rate = 0.0
            #     print(
            #         f"`p_position_ratio` is {p_position_ratio}, `shift_event_rate` -> `0.0`"
            #     )

            if self.generate_noise_rate > 0:
                self.generate_noise_rate = 0.0
                print(
                    f"`p_position_ratio` is {p_position_ratio}, `generate_noise_rate` -> `0.0`"
                )
        self.soft_label_shape = soft_label_shape
        self.soft_label_width = soft_label_width
        self.dtype = dtype

    def _clear_dict_except(self, d: dict, *args) -> None:
        if len(args) > 0:
            for arg in args:
                assert isinstance(
                    arg, str
                ), f"Input arguments must be str, got `{arg}`({type(arg)})"
        for k in set(d) - set(args):
            if isinstance(d[k], (list, dict)):
                d[k].clear()
            elif isinstance(d[k], np.ndarray):
                d[k] = np.array([])
            elif isinstance(d[k], (int, float)):
                d[k] = 0
            elif isinstance(d[k], str):
                d[k] = ""
            else:
                raise TypeError(f"Got `{d[k]}`({type(d[k])})")

    def _is_noise(
        self, data: np.ndarray, ppks: List[int], spks: List[int]
    ) -> bool:
        """
        Determine noise data
        """
        is_noise = (
            (len(ppks) != len(spks))
            or len(ppks) < 1
            or len(spks) < 1
            or min(ppks + spks) < 0
            or max(ppks + spks) >= data.shape[-1]
        )
        for i in range(min(len(ppks), len(spks))):
            is_noise |= ppks[i] >= spks[i]
        return is_noise

    def _cut_window(
        self, data: np.ndarray, ppks: list, spks: list, window_size: int
    ) -> Tuple[np.ndarray, list, list]:
        """
        Slice the ndarray to `window_size`
        """
        input_len = data.shape[-1]

        if 0 <= self.p_position_ratio <= 1: # limit p on certain place
            if self.p_position_ratio_type == 'uniform':
                start = self.p_position_ratio - self.p_position_ratio_range_or_sigma
                end = self.p_position_ratio + self.p_position_ratio_range_or_sigma
                p_position_ratio = np.random.uniform(
                    start, end
                )
            elif self.p_position_ratio_type == 'gaussian':
                while True:
                    p_position_ratio = np.random.normal(
                        self.p_position_ratio, self.p_position_ratio_range_or_sigma
                    )
                    if 0 <= p_position_ratio <= 0.5:
                        break
            else:
                raise NotImplementedError

            new_data = np.zeros((data.shape[0], window_size), dtype=np.float32)
            tgt_l, tgt_r = 0, window_size

            p_idx = ppks[0]
            c_l = p_idx - int(window_size * p_position_ratio)
            c_r = c_l + window_size
            offset = -c_l

            if c_l < 0:
                tgt_l += abs(c_l)
                offset += c_l
                c_l = 0

            if c_r > data.shape[-1]:
                tgt_r -= c_r - data.shape[-1]
                c_r = data.shape[-1]

            new_data[:, tgt_l:tgt_r] = data[:, c_l:c_r]
            offset += tgt_l
            data = new_data

            assert len(ppks) <= 1 and len(spks) <= 1, f"ppks:{ppks},spks:{spks}"
            ppks = [t + offset for t in ppks if 0 <= t + offset < window_size]
            spks = [t + offset for t in spks if 0 <= t + offset < window_size]

        elif self.p_position_ratio == -1: # no limit
            # ori_ppks, ori_spks = ppks.copy(), spks.copy()
            if input_len > window_size:
                c_l = np.random.randint(
                    0,
                    max(min([input_len - window_size]) - self.min_event_gap, 1),
                )
                c_r = c_l + window_size

                data = data[:, c_l:c_r]
                ppks = [t - c_l for t in ppks if c_l <= t < c_r]
                spks = [t - c_l for t in spks if c_l <= t < c_r]

            elif input_len < window_size:
                data = np.concatenate(
                    [data, np.zeros((data.shape[0], window_size - input_len))], axis=1
                )
                # c_l, c_r = 0, input_len
            # print(f'init ppk&spk:[{ori_ppks},{ori_spks}], cl&cr:[{c_l},{c_r}]. after cut windows ppk&spk:[{ppks},{spks}]')
        elif self.p_position_ratio == -2: # before p arrival
            if min(ppks) > window_size:
                c_r = np.random.randint(window_size, min(ppks))
                c_l = c_r - window_size

                data = data[:, c_l:c_r]
                ppks = [t - c_l for t in ppks if c_l <= t < c_r]
                spks = [t - c_l for t in spks if c_l <= t < c_r]

            elif min(ppks) < window_size:
                # 对data[:,:min(ppks)]，self-padding，直到长度大于window_size,再裁切出window_size
                tgt_data = data[:,:min(ppks)]
                while tgt_data.shape[1] < window_size:
                    tgt_data = np.concatenate(
                        [tgt_data, data[:,:min(ppks)]], axis=1
                    )
                data = tgt_data[:,:window_size]

        return data, ppks, spks

    def _normalize(self, data, mode):
        """
        Normalize waveform of each sample. (inplace)
        """
        data -= np.mean(data, axis=1, keepdims=True)
        if mode == "max":
            max_data = np.max(data, axis=1, keepdims=True)
            max_data[max_data == 0] = 1
            data /= max_data # + 1e-6

        elif mode == "std":
            std_data = np.std(data, axis=1, keepdims=True)
            std_data[std_data == 0] = 1
            data /= std_data # + 1e-6
        elif mode == "":
            return data
        else:
            raise ValueError(f"Supported mode: 'max','std', got '{mode}'")
        return data

    def _generate_noise_data(self, data: np.ndarray, ppks: list, spks: list):
        """
        Remove all phases.(inplace)
        """
        if len(ppks) > 0 and len(spks) > 0:
            for i in range(len(ppks)):
                ppk = ppks[i]
                spk = spks[i]
                coda_end = np.clip(
                    int(spk + self.coda_ratio * (spk - ppk)),
                    0,
                    data.shape[-1],
                    dtype=int,
                )
                if ppk < coda_end:
                    data[:, ppk:coda_end] = np.random.randn(
                        data.shape[0], coda_end - ppk
                    )

        return data, [], []

    def _add_event(self, data: np.ndarray, ppks: list, spks: list, min_gap: int):
        """
        Add seismic event.(inplace) note: use the method before `_shift_event`
        """
        target_idx = np.random.randint(0, len(ppks))

        ppk = ppks[target_idx]
        spk = spks[target_idx]
        coda_end = int(spk + (self.coda_ratio * (spk - ppk)))

        left = coda_end + min_gap
        right = data.shape[-1] - (spk - ppk) - min_gap

        if left < right:
            ppk_add = np.random.randint(left, right)
            spk_add = ppk_add + spk - ppk
            space = min(data.shape[-1] - ppk_add, coda_end - ppk)

            scale = np.random.random()

            data[:, ppk_add : ppk_add + space] += data[:, ppk : ppk + space] * scale

            ppks.append(ppk_add)
            spks.append(spk_add)

        ppks.sort()
        spks.sort()
        return data, ppks, spks

    def _shift_event(self, data, ppks, spks):
        """
        Shift event.
        """
        shift = np.random.randint(0, data.shape[-1])
        data = np.concatenate((data[:, -shift:], data[:, :-shift]), axis=1)
        ppks = [(p + shift) % data.shape[-1] for p in ppks]
        spks = [(s + shift) % data.shape[-1] for s in spks]

        ppks.sort()
        spks.sort()
        return data, ppks, spks

    def _drop_channel(self, data):
        """
        Drop channels. (inplace)
        """
        if data.shape[0] < 2:
            return data
        else:
            drop_num = np.random.choice(range(1, data.shape[0]))
            candidates = list(range(data.shape[0]))
            for _ in range(drop_num):
                c = np.random.choice(candidates)
                candidates.remove(c)
                data[c, :] = 0.0

        return data

    def _adjust_amplitude(self, data):
        """
        Adjust amplitude after dropping channels.(inplace)
        """
        max_amp = np.max(np.abs(data), axis=1)

        if np.count_nonzero(max_amp) > 0:
            data *= data.shape[0] / np.count_nonzero(max_amp)

        return data

    def _scale_amplitude(self, data):
        """
        Scale amplitude.(inplace)
        """
        if np.random.uniform(0, 1) < 0.5:
            data *= np.random.uniform(1, 3)
        else:
            data /= np.random.uniform(1, 3)

        return data

    def _pre_emphasis(self, data: np.ndarray, pre_emphasis: float) -> np.ndarray:
        """
        Pre-emphasis.(inplace)
        """
        for c in range(data.shape[0]):
            bpf = data[c, :]
            data[c, :] = np.append(bpf[0], bpf[1:] - pre_emphasis * bpf[:-1])
        return data

    def _add_noise(self, data):
        """
        Add gaussian or uniform noise.(inplace)
        """
        for c in range(data.shape[0]):
            x = data[c, :]
            snr = np.random.randint(10, 50)
            px = np.sum(x**2) / len(x)
            pn = px * 10 ** (-snr / 10.0)
            if self.noise_type == "gaussian":
                noise = np.random.randn(len(x)) * np.sqrt(pn)
            elif self.noise_type == "uniform":
                noise = np.random.uniform(-1, 1, len(x)) * np.sqrt(pn)
            data[c, :] += noise
        return data

    def _add_gaps(self, data: np.ndarray, ppks: list, spks: list):
        """
        Add gaps.(inplace)
        """
        phases = sorted(ppks + spks)

        if len(phases) > 0:
            phases.append(data.shape[-1] - 1)
            phases = sorted(set(phases))

            insert_pos = np.random.randint(0, len(phases) - 1)

            sgt = np.random.randint(phases[insert_pos], phases[insert_pos + 1])
            egt = np.random.randint(sgt, phases[insert_pos + 1])
        else:
            sgt = np.random.randint(0, data.shape[-1] - 1)
            egt = np.random.randint(sgt + 1, data.shape[-1])

        data[:, sgt:egt] = 0

        return data

    def _add_mask_windows(
        self,
        data: np.ndarray,
        window_size: int = 20,
        mask_value: float = 1.0
    ):
        """
        Add mask windows.(inplace)
        """
        if self.whether_fixed_mask_percent:
            p = np.clip(self.mask_percent, 0, 100)
        else:
            p = np.random.uniform(self.lower_bound_mask_percent, self.upper_bound_mask_percent)
        num_windows = data.shape[-1] // window_size
        num_mask = int(num_windows * p // 100)
        selected = np.random.choice(range(num_windows), num_mask, replace=False)
        for i in selected:
            st = i * window_size
            et = st + window_size
            data[:, st:et] = mask_value

        return data

    def _add_noise_windows(
        self, 
        data: np.ndarray, 
        window_size: int = 20,
    ):
        """
        Add noise windows.(inplace)
        """
        if self.whether_fixed_noise_percent:
            p = np.clip(self.noise_percent, 0, 100)
        else:
            p = np.random.uniform(self.lower_bound_noise_percent, self.upper_bound_noise_percent)
        num_windows = data.shape[-1] // window_size
        num_block = int(num_windows * p // 100)
        selected = np.random.choice(range(num_windows), num_block, replace=False)
        for i in selected:
            st = i * window_size
            et = st + window_size
            data[:, st:et] = np.random.randn(data.shape[0], window_size) # Gaussian noise
        return data

    def _ori_data_augmentation_combine(self, event: dict) -> dict:
        data, ppks, spks = itemgetter("data", "ppks", "spks")(event)

        # Generate noise data
        if np.random.random() < self.generate_noise_rate:
            # Noise data
            data, ppks, spks = self._generate_noise_data(data, ppks, spks)
            self._clear_dict_except(event, "data")

            # Drop channel
            if np.random.random() < self.drop_channel_rate:
                data = self._drop_channel(data)
                data = self._adjust_amplitude(data)

            # Scale
            if np.random.random() < self.scale_amplitude_rate:
                data = self._scale_amplitude(data)

        else:
            # Add event
            for _ in range(self._max_event_num - len(ppks)):
                if np.random.random() < self.add_event_rate and ppks:
                    data, ppks, spks = self._add_event(
                        data, ppks, spks, self.min_event_gap
                    )

            # Shift event
            if np.random.random() < self.shift_event_rate:
                data, ppks, spks = self._shift_event(data, ppks, spks)

            # Drop channel
            if np.random.random() < self.drop_channel_rate:
                data = self._drop_channel(data)
                data = self._adjust_amplitude(data)

            # Scale
            if np.random.random() < self.scale_amplitude_rate:
                data = self._scale_amplitude(data)

            # Pre-emphasis
            if np.random.random() < self.pre_emphasis_rate:
                data = self._pre_emphasis(data, self.pre_emphasis_ratio)

            # Add noise
            if np.random.random() < self.add_noise_rate:
                data = self._add_noise(data)

            # Add gaps
            if np.random.random() < self.add_gap_rate:
                data = self._add_gaps(data, ppks, spks)

        if self.mask_percent > 0:
            data = self._add_mask_windows(
                data=data,
                window_size=self.sampling_rate // 2,
            )

        if self.noise_percent > 0:
            data = self._add_noise_windows(
                data=data,
                window_size=self.sampling_rate // 2,
            )

        event.update({"data": data, "ppks": ppks, "spks": spks})
        return event

    def _data_augmentation_seperate(self, event: dict, operation) -> dict:
        data, ppks, spks = itemgetter("data", "ppks", "spks")(event)
        # Generate noise data
        if operation == 'generate_noise_data':
            if np.random.random() < self.generate_noise_rate:
                data, ppks, spks = self._generate_noise_data(data, ppks, spks)
                self._clear_dict_except(event, "data")

        if operation == 'drop_channel':
            # Drop channel
            if np.random.random() < self.drop_channel_rate:
                data = self._drop_channel(data)
                data = self._adjust_amplitude(data)

        if operation == 'scale_amplitude':
            # Scale
            if np.random.random() < self.scale_amplitude_rate:
                data = self._scale_amplitude(data)

        if operation == 'add_event':
            # Add event
            for _ in range(self._max_event_num - len(ppks)):
                if np.random.random() < self.add_event_rate and ppks:
                    data, ppks, spks = self._add_event(
                        data, ppks, spks, self.min_event_gap
                    )

        # if operation == 'shift_event':
        #     # Shift event
        #     if np.random.random() < self.shift_event_rate:
        #         data, ppks, spks = self._shift_event(data, ppks, spks)

        if operation == 'pre_emphasis':
            # Pre-emphasis
            if np.random.random() < self.pre_emphasis_rate:
                data = self._pre_emphasis(data, self.pre_emphasis_ratio)
                
        if  operation == 'add_noise':
            # Add noise
            if np.random.random() < self.add_noise_rate:
                data = self._add_noise(data)

        if operation == 'add_gaps':
            # Add gaps
            if np.random.random() < self.add_gap_rate:
                data = self._add_gaps(data, ppks, spks)

        if operation == 'add_noise_window':
            # Add noise window
            if np.random.random() < self.add_noise_window_rate:
                data = self._add_noise_windows(
                    data=data,
                    window_size=self.sampling_rate // 2,
                )

        if operation == 'add_mask_window':
            # Add mask window
            if np.random.random() < self.add_mask_window_rate:
                data = self._add_mask_windows(
                    data=data,
                    window_size=self.sampling_rate // 2,
                )

        event.update({"data": data, "ppks": ppks, "spks": spks})
        return event

    def _data_augmentation(self, event: dict) -> dict:
        """
        Data augmentation.
        """
        for operation in self.operations:
            event = self._data_augmentation_seperate(event, operation)
        return event

    def process(self, event: dict, augmentation: bool, inplace: bool = True) -> dict:
        """Process raw data.

        Args:
            event (dict): Event dict.
            augmentation (bool): Whether to use data augmentation.
            inplace (bool): Whether to modify the event dict rather than create a new one.

        Returns:
            dict: Processed event data.
        """
        if not inplace:
            event = copy.deepcopy(event)

        if augmentation:
            event = self._data_augmentation(event=event)

        # Cut window
        event["data"], event["ppks"], event["spks"] = self._cut_window(
            data=event["data"],
            ppks=event["ppks"],
            spks=event["spks"],
            window_size=self.in_samples,
        )

        # shift event
        if 'shift_event' in self.operations:
            if np.random.random() < self.shift_event_rate:
                event["data"], event["ppks"], event["spks"] = self._shift_event(
                    data=event["data"], ppks=event["ppks"], spks=event["spks"]
                )

        # Instance Norm
        event["data"] = self._normalize(event["data"], self.norm_mode)

        return event

    def _generate_soft_label(
        self, name: str, event: dict, soft_label_width: int, soft_label_shape: str
    ) -> np.ndarray:
        """Generate soft io-item

        Args:
            name (str): Item name. See :class:`~SeisT.config.Config._avl_io_items`.
            event (dict): Event dict.
            soft_label_width (int): Label width.
            soft_label_shape (str): Label shape.

        Raises:
            NotImplementedError: Unsupported label shape.
            NotImplementedError: Unsupported label name.

        Returns:
            np.ndarray: label.
        """
        length = event["data"].shape[-1]

        def _clip(x: int) -> int:
            return min(max(x, 0), length)

        def _get_soft_label(idxs, length):
            """Soft label"""
            slabel = np.zeros(length)

            if len(idxs) > 0:
                left = int(soft_label_width / 2)
                right = soft_label_width - left

                if soft_label_shape == "gaussian":
                    window = np.exp(
                        -((np.arange(-left, right + 1)) ** 2) / (2 * 10**2)
                    )
                elif soft_label_shape == "triangle":
                    window = 1 - np.abs(
                        2 / soft_label_width * (np.arange(-left, right + 1))
                    )
                elif soft_label_shape == "box":
                    window = np.ones(soft_label_width + 1)

                elif soft_label_shape == "sigmoid":

                    def _sigmoid(x):
                        return 1 / (1 + np.exp(x))

                    l_l, l_r = -int(left / 2), left - int(left / 2)
                    r_l, r_r = -int(right / 2), right - int(right / 2)
                    x_l, x_r = -10 / left * np.arange(l_l, l_r), -10 / right * (
                        -1
                    ) * np.arange(r_l, r_r)
                    w_l, w_r = _sigmoid(x_l), _sigmoid(x_r)
                    window = np.concatenate((w_l, [1.0], w_r), axis=0)
                else:
                    raise NotImplementedError(
                        f"Unsupported label shape: '{soft_label_shape}'"
                    )

                for idx in idxs:
                    idx = int(idx)
                    if idx < 0:
                        pass  # Out of range
                    elif idx - left < 0:
                        slabel[: idx + right + 1] += window[
                            soft_label_width + 1 - (idx + right + 1) :
                        ]
                    elif idx + right <= length - 1:
                        slabel[idx - left : idx + right + 1] += window
                    elif idx <= length - 1:
                        slabel[-(length - (idx - left)) :] += window[
                            : length - (idx - left)
                        ]
                    else:
                        pass  # Out of range

            return slabel

        ppks, spks = _pad_phases(
            ppks=event["ppks"],
            spks=event["spks"],
            padding_idx=soft_label_width,
            num_samples=length,
        )

        # Phase-P/S
        if name in ["ppk", "spk"]:
            key = {"ppk":"ppks", "spk":"spks"}.get(name)
            label = _get_soft_label(idxs=event[key], length=length)

        # None (=1-P(p)-P(s))
        elif name == "non":
            label = (
                np.ones(length)
                - _get_soft_label(idxs=ppks, length=length)
                - _get_soft_label(idxs=spks, length=length)
            )
            label[label < 0] = 0

        # Detection
        elif name == "det":
            label = np.zeros(length)

            assert len(ppks) == len(spks)

            for i in range(len(ppks)):
                ppk = ppks[i]
                spk = spks[i]
                dst = int(ppk)
                det = int(spk + (self.coda_ratio * (spk - ppk)))
                label_i = _get_soft_label(idxs=[dst, det], length=length)
                label_i[_clip(dst) : _clip(det)] = 1.0
                label += label_i
            label[label > 1] = 1.0

        # Phase-P/S (plus)
        elif name in ["ppk+", "spk+"]:
            label = np.zeros(length)
            key = {"ppk+":"ppks", "spk+":"spks"}.get(name)
            phases = event[key]
            for i in range(len(phases)):
                st = phases[i]
                label_i = _get_soft_label(idxs=[st], length=length)
                label_i[_clip(st) :] = 1.0
                label += label_i / len(phases)

        # Waveform
        elif name in self.data_channels:
            ch_idx = self.data_channels.index(name)
            label = event["data"][ch_idx]

        # Diff
        elif name in [f"d{c}" for c in self.data_channels]:
            channel_data = event["data"][self.data_channels.index(name[-1])]
            label = np.zeros_like(channel_data)
            label[1:] = np.diff(channel_data)

        else:
            raise NotImplementedError(f"Unsupported label name: '{name}'")

        return label.astype(self.dtype)

    def _get_io_item(
        self,
        name: Union[str, tuple, list],
        event: dict,
        soft_label_width: int = None,
        soft_label_shape: str = None,
    ) -> Union[tuple, list, np.ndarray]:
        """Get IO item
        
        In order to adapt to the input and output data of different models, we have weakened 
        the difference between input and output, and collectively refer to them as `io_item`.

        Args:
            name (Union[str,tuple,list]): Item name
            event (dict): Event.
            soft_label_width (int, optional): Label width (only applicable to soft label). Defaults to None.
            soft_label_shape (str, optional): Label shape (only applicable to soft label). Defaults to None.

        Raises:
            ValueError: No value to generate one-hot vetor.
            NotImplementedError: Unknow item type

        Returns:
            Union[tuple,list,np.ndarray]: Item.
        """
        if isinstance(name, (tuple, list)):
            children = [self._get_io_item(sub_name, event) for sub_name in name]
            item = np.array(children)
            return item

        else:
            if Config.get_type(name) == "soft":
                item = self._generate_soft_label(
                    name=name,
                    event=event,
                    soft_label_width=(soft_label_width or self.soft_label_width),
                    soft_label_shape=(soft_label_shape or self.soft_label_shape),
                )

            elif Config.get_type(name) == "value":
                value = event[name]
                item = np.array(value).astype(self.dtype)

            elif Config.get_type(name) == "onehot":
                cidx = event[name]
                if not len(cidx) > 0:
                    raise ValueError(f"Item:{name}, Value:{cidx}")
                nc = Config.get_num_classes(name=name)
                item = np.eye(nc)[int(cidx[0])].astype(np.int64)

            else:
                raise NotImplementedError(f"Unknown item: {name}")

            return item

    def get_targets_for_loss(self, event: dict, label_names: list) -> Any:
        """Get targets which are used to calculate loss

        Args:
            event (dict): Event dict.
            label_names (list): label names.
        Returns:
            Any: Targets.
        """

        targets = [self._get_io_item(name=name, event=event) for name in label_names]

        if len(targets) > 1:
            return tuple(targets)
        else:
            return targets.pop()

    def get_targets_for_metrics(
        self,
        event: dict,
        max_event_num: int,
        task_names: list,
    ) -> dict:
        """Get labels which are used to calculate metrics

        Args:
            event (dict): Event dict.
            max_event_num (int): Used for padding phase list to the same length.
            task_names (list): Names of tasks.

        Returns:
            dict: Labels.
        """
        targets = {}

        for name in task_names:
            if name in ["ppk", "spk"]:
                key = {"ppk":"ppks", "spk":"spks"}.get(name)
                tgt = self._get_io_item(name=key, event=event)
                tgt = _pad_array(tgt, length=max_event_num, padding_value=int(-1e7)).astype(np.int64)
            elif name == "det":
                padded_ppks, padded_spks = _pad_phases(
                    event["ppks"], event["spks"], self.soft_label_width, self.in_samples
                )
                # detections = []
                # for ppk, spk in zip(padded_ppks, padded_spks):
                #     st = np.clip(ppk,0,self.in_samples)
                #     et = int(spk + (self.coda_ratio * (spk - ppk)))
                #     detections.extend([st,et])
                # expected_num = self._max_event_num
                # if len(detections)//2< expected_num:
                #     detections = detections + [1,0] * (expected_num-len(detections)//2)
                
                if len(padded_ppks) > 0:
                    ppk, spk = padded_ppks[0],padded_spks[0]
                    st = np.clip(ppk,0,self.in_samples)
                    et = int(spk + (self.coda_ratio * (spk - ppk)))
                    detections = [st,et]
                else:
                    detections = [1,0]
                tgt = np.array(detections).astype(np.int64)
            else:
                tgt = self._get_io_item(name=name, event=event)

            targets[name] = tgt

        return targets

    def get_inputs(self, event: dict, input_names: list) -> Union[np.ndarray, tuple]:
        """Get inputs data

        Args:
            event (dict): Event dict.
            linput_names (list): input names.

        Returns:
            Any: Inputs.
        """

        inputs = [self._get_io_item(name=name, event=event) for name in input_names]
        if len(inputs) > 1:
            return tuple(inputs)
        else:
            return inputs.pop()


class SeismicDataset(Dataset):
    """
    Read and preprocess data.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        input_names: list,
        label_names: list,
        task_names: list,
        mode: str,
    ) -> None:
        """
        Args:
            args:argparse.Namespace
                Input arguments.
            input_names: list
                Input names. See :class:`~SeisT.config.Config` for more details.
            label_names: list
                Label names. See :class:`~SeisT.config.Config` for more details.
            task_names: list
                Task names. See :class:`~SeisT.config.Config` for more details.
            mode: str
                train/val/test.
        """

        self._seed = int(args.seed)
        self._mode = mode.lower()
        self._input_names = input_names
        self._label_names = label_names
        self._task_names = task_names
        self._max_event_num = args.max_event_num

        self._augmentation = args.augmentation and self._mode == "train"
        if self._augmentation != args.augmentation:
            print(f"[{self._mode}]Augmentation -> {self._augmentation}")

        # Dataset
        self._dataset = LSD(
            dataset_name=args.dataset_name,
            seed=self._seed,
            mode=self._mode,
            data_dir=args.data,
            shuffle=args.shuffle,
            data_split=args.data_split,
            train_size=args.train_size,
            val_size=args.val_size,
            downstream_task=args.downstream_task,
            subset_names=args.subset_names,
        )
        print(self._dataset)

        self._dataset_size = len(self._dataset)

        if self._augmentation:
            print(
                f"Data augmentation: Dataset size -> {self._dataset_size *2}"
            )

        # Preprocessor
        self._preprocessor = DataPreprocessor(
            data_channels=self._dataset.channels(),
            sampling_rate=self._dataset.sampling_rate(),
            in_samples=args.in_samples,
            min_snr=args.min_snr,
            coda_ratio=args.coda_ratio,
            norm_mode=args.norm_mode,
            p_position_ratio=args.p_position_ratio,
            add_event_rate=args.add_event_rate,
            add_noise_rate=args.add_noise_rate,
            add_gap_rate=args.add_gap_rate,
            drop_channel_rate=args.drop_channel_rate,
            scale_amplitude_rate=args.scale_amplitude_rate,
            pre_emphasis_rate=args.pre_emphasis_rate,
            pre_emphasis_ratio=args.pre_emphasis_ratio,
            max_event_num=args.max_event_num,
            generate_noise_rate=args.generate_noise_rate,
            shift_event_rate=args.shift_event_rate,
            mask_percent=args.mask_percent,
            noise_percent=args.noise_percent,
            min_event_gap_sec=args.min_event_gap,
            soft_label_shape=args.label_shape,
            soft_label_width=int(args.label_width * self._dataset.sampling_rate()),
            operations=args.operations,
            add_mask_window_rate=args.add_mask_window_rate,
            add_noise_window_rate=args.add_noise_window_rate,
            noise_type=args.noise_type,
            whether_fixed_mask_percent=args.whether_fixed_mask_percent,
            lower_bound_mask_percent=args.lower_bound_mask_percent,
            upper_bound_mask_percent=args.upper_bound_mask_percent,
            whether_fixed_noise_percent=args.whether_fixed_noise_percent,
            lower_bound_noise_percent=args.lower_bound_noise_percent,
            upper_bound_noise_percent=args.upper_bound_noise_percent,
            shift_event_distance_percent=args.shift_event_distance_percent,
            dtype=np.float32,
            p_position_ratio_type=args.p_position_ratio_type,
            p_position_ratio_range_or_sigma=args.p_position_ratio_range_or_sigma,
        )

    def sampling_rate(self):
        return self._dataset.sampling_rate()

    def data_channels(self):
        return self._dataset.channels()
    
    def name(self):
        return f"{self._dataset.name()}_{self._mode}"

    def __len__(self) -> int:
        if self._augmentation:
            return 2 * self._dataset_size
        else:
            return self._dataset_size

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, Any]:
        """
        Args:
            idx (int): Index
        Returns:
            tuple: inputs, loss_targets, metrics_targets, meta_data
        """

        # Load data
        event = self._dataset[idx % self._dataset_size]

        # Preprocess
        event = self._preprocessor.process(
            event=event, augmentation=(self._augmentation and idx >= self._dataset_size)
        )

        # Generate inputs
        inputs = self._preprocessor.get_inputs(
            event=event, input_names=self._input_names
        )

        # ##########
        # ppks = event['ppks']
        # spks = event['spks']
        # crop_start = 0
        # crop_end = 200  # crop is left closed and right open
        # if len(ppks) != 0:  # divide by 50 to obtain the arrival time on the token level
        #     crop_start = int(ppks[0] / 50)
        #
        # if 'dis' in self._task_names:  # difference in the crop length between different tasks is the end time
        #     if len(ppks) != 0 and len(spks) != 0:  # divide by 50 to obtain the arrival time on the token level
        #         crop_end = int(((spks[0] - ppks[0]) * 3 + ppks[0]) / 50) + 1  # Exceeding the maximum length, the
        #         # tensor will automatically truncate
        # elif 'mag_full' in self._task_names:
        #     if len(ppks) != 0 and len(spks) != 0:
        #         crop_end = int(((spks[0] - ppks[0]) * 3 + ppks[0]) / 50) + 1
        # elif 'mag_P_only' in self._task_names: # use 5 seconds of data to predict the magnitude of the P wave
        #     if len(ppks) != 0:
        #         crop_end = int((ppks[0] + 500) / 50) + 1
        # elif 'baz' in self._task_names: # use 1 second before and 2 seconds after the P wave to predict the back azimuth
        #     if len(ppks) != 0:
        #         crop_start = min(int((ppks[0] - 100) / 50), 0) # if the start time is less than 0, set it to 0
        #         crop_end = int((ppks[0] + 200) / 50) + 1
        # elif 'fmp' in self._task_names:  # use 1 second before and 1 seconds after the P wave to predict the first motion polarity
        #     if len(ppks) != 0:
        #         crop_start = min(int((ppks[0] - 100) / 50), 0)
        #         crop_end = int((ppks[0] + 100) / 50) + 1
        # elif 'dep' in self._task_names:  # use P to 3*(S-P) for predicting earthquake depth
        #     if len(ppks) != 0 and len(spks) != 0:
        #         crop_end = int(((spks[0] - ppks[0]) * 3 + ppks[0]) / 50) + 1
        # elif 'cls' in self._task_names:  # use  P to 3*(S-P) to predict the class of the earthquake
        #     if len(ppks) != 0 and len(spks) != 0:
        #         crop_end = int(((spks[0] - ppks[0]) * 3 + ppks[0]) / 50) + 1
        # elif 'det' in self._task_names:  # dpk task does not require a crop
        #     pass
        # else:
        #     raise NotImplementedError(f"Downstream task {self._task_names} not implemented")
        # new_inputs = [inputs, crop_start, crop_end]
        # # modify the original inputs to the new_inputs, which is a list:[inputs, crop_start, crop_end]

        # Generate labels
        loss_targets = self._preprocessor.get_targets_for_loss(
            event=event, label_names=self._label_names
        )
        metrics_targets = self._preprocessor.get_targets_for_metrics(
            event=event, task_names=self._task_names, max_event_num=self._max_event_num
        )
        # print(inputs, loss_targets, metrics_targets, meta_data_json)
        return inputs, loss_targets, metrics_targets

class SFTDataset(Dataset):
    """
    Read and preprocess data.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        input_names: list,
        label_names: list,
        task_names: list,
        mode: str,
    ) -> None:
        """
        Args:
            args:argparse.Namespace
                Input arguments.
            input_names: list
                Input names. See :class:`~SeisT.config.Config` for more details.
            label_names: list
                Label names. See :class:`~SeisT.config.Config` for more details.
            task_names: list
                Task names. See :class:`~SeisT.config.Config` for more details.
            mode: str
                train/val/test.
        """

        self._seed = int(args.seed)
        self._mode = mode.lower()
        self._input_names = input_names
        self._label_names = label_names
        self._task_names = task_names
        self._max_event_num = args.max_event_num

        self._augmentation = args.augmentation and self._mode == "train"
        if self._augmentation != args.augmentation:
            print(f"[{self._mode}]Augmentation -> {self._augmentation}")

        # Dataset
        if self._mode == "train" or self._mode == 'val':
            meta_data_path = args.train_meta_data_path
            data_dir = args.train_data_dir
            sample_num = args.train_sample_num
            test_p_postion_ratio = None
            target_column = args.train_target_column
        elif self._mode == 'test':
            meta_data_path = args.test_meta_data_path
            data_dir = args.test_data_dir
            sample_num = None
            test_p_postion_ratio = args.test_p_postion_ratio
            target_column = args.test_target_column
        else:
            raise NotImplementedError
        
        if args.splitPS:
            self._dataset = SFTDataSplitPS(
                seed=self._seed,
                mode=self._mode,
                data_dir=data_dir,
                meta_data_path=meta_data_path,
                shuffle=args.shuffle,
                data_split=args.data_split,
                train_size=args.train_size,
                val_size=args.val_size,
                downstream_task=args.downstream_task,
                subset_names=args.subset_names,
                sample_num=sample_num,
            )
        else:
            self._dataset = SFTData(
                seed=self._seed,
                mode=self._mode,
                data_dir=data_dir,
                meta_data_path=meta_data_path,
                shuffle=args.shuffle,
                data_split=args.data_split,
                train_size=args.train_size,
                val_size=args.val_size,
                downstream_task=args.downstream_task,
                target_column=target_column,
                subset_names=args.subset_names,
                sample_num=sample_num,
                test_p_postion_ratio=test_p_postion_ratio,
            )
        print(self._dataset)

        self._dataset_size = len(self._dataset)

        if self._augmentation:
            print(
                f"Data augmentation: Dataset size -> {self._dataset_size}"
            )

        # Preprocessor
        self._preprocessor = DataPreprocessor(
            data_channels=self._dataset.channels(),
            sampling_rate=self._dataset.sampling_rate(),
            in_samples=args.in_samples,
            min_snr=args.min_snr,
            coda_ratio=args.coda_ratio,
            norm_mode=args.norm_mode,
            p_position_ratio=args.p_position_ratio,
            add_event_rate=args.add_event_rate,
            add_noise_rate=args.add_noise_rate,
            add_gap_rate=args.add_gap_rate,
            drop_channel_rate=args.drop_channel_rate,
            scale_amplitude_rate=args.scale_amplitude_rate,
            pre_emphasis_rate=args.pre_emphasis_rate,
            pre_emphasis_ratio=args.pre_emphasis_ratio,
            max_event_num=args.max_event_num,
            generate_noise_rate=args.generate_noise_rate,
            shift_event_rate=args.shift_event_rate,
            mask_percent=args.mask_percent,
            noise_percent=args.noise_percent,
            min_event_gap_sec=args.min_event_gap,
            soft_label_shape=args.label_shape,
            soft_label_width=int(args.label_width * self._dataset.sampling_rate()),
            operations=args.operations,
            add_mask_window_rate=args.add_mask_window_rate,
            add_noise_window_rate=args.add_noise_window_rate,
            noise_type=args.noise_type,
            whether_fixed_mask_percent=args.whether_fixed_mask_percent,
            lower_bound_mask_percent=args.lower_bound_mask_percent,
            upper_bound_mask_percent=args.upper_bound_mask_percent,
            whether_fixed_noise_percent=args.whether_fixed_noise_percent,
            lower_bound_noise_percent=args.lower_bound_noise_percent,
            upper_bound_noise_percent=args.upper_bound_noise_percent,
            shift_event_distance_percent=args.shift_event_distance_percent,
            dtype=np.float32,
            p_position_ratio_type=args.p_position_ratio_type,
            p_position_ratio_range_or_sigma=args.p_position_ratio_range_or_sigma,
        )
        self.no_event_p=args.no_event_p
        self.random_crop_p=args.random_crop_p
        self.no_event_label=args.no_event_label
        self.default_label_dis=args.default_label_dis

    def sampling_rate(self):
        return self._dataset.sampling_rate()

    def data_channels(self):
        return self._dataset.channels()
    
    def name(self):
        return f"{self._dataset.name()}_{self._mode}"

    def __len__(self) -> int:
        return self._dataset_size

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, Any]:
        """
        Args:
            idx (int): Index
        Returns:
            tuple: inputs, loss_targets, metrics_targets, meta_data
        """

        # Load data
        event,meta_data = self._dataset[idx % self._dataset_size]

        # Sample window
        if self._mode == 'test':
            self._preprocessor.p_position_ratio = event['test_p_postion_ratio']
            self._preprocessor.p_position_ratio_type = 'uniform'
            self._preprocessor.p_position_ratio_range_or_sigma = 0
        else:
            if random.random() <= self.random_crop_p or event['p_exits'] == [0]:
                self._preprocessor.p_position_ratio = -1
            #else:
            #    self._preprocessor.p_position_ratio = 0.5
            #    self._preprocessor.p_position_ratio_type = 'uniform'
            #    self._preprocessor.p_position_ratio_range_or_sigma = 0.5

        # Preprocess
        event = self._preprocessor.process(
            event=event, augmentation=self._augmentation
        )

        if len(event['ppks']) < 1:
            if self.no_event_label == 'default':
                event['emg'] = [-3.0]
                event['dis'] = [self.default_label_dis]
            else:
                event['emg'] = [-1]
                event['dis'] = [-1]

        # add by zb, use 1 second before and 1 seconds after the P wave to predict the first motion polarity
        sampling_rate=self._dataset.sampling_rate(),
        if len(self._task_names) == 1 and self._task_names[0] == "fmp":
            if len(event["ppks"]) != 0:
                crop_start = max(event["ppks"][0] - int(1.0 * self._dataset.sampling_rate()), 0)
                crop_end = event["ppks"][0] + int(1.0 * self._dataset.sampling_rate()) + 1
                event["data"][:,:crop_start] = 0.
                event["data"][:,crop_end:] = 0.
        # end add by zb

        # Generate inputs
        inputs = self._preprocessor.get_inputs(
            event=event, input_names=self._input_names
        )


        # Generate labels
        if len(self._label_names) > 1:
            loss_targets = [
                self._preprocessor.get_targets_for_loss(event=event, label_names=label_name) 
                for label_name in self._label_names
            ]
            metrics_targets = [
                self._preprocessor.get_targets_for_metrics(event=event, task_names=task_name, max_event_num=self._max_event_num)
                for task_name in self._task_names
            ]
            if 'ppks_type' in event.keys() and 'spks_type' in event.keys():
                print('hello')
                metrics_targets[0]['ppks_type'] = np.array([event['ppks_type']])
                metrics_targets[0]['spks_type'] = np.array([event['spks_type']])
        # add by lhl
        elif isinstance(self._label_names, list) and len(self._label_names) == 1 and isinstance(self._label_names[0], list): # for multi det_ppk_spk or det_ppk_spk_dis_emg
            loss_targets = {
                label_name:self._preprocessor.get_targets_for_loss(event=event, label_names=[label_name]) 
                for label_name in self._label_names[0]
            }
            metrics_targets = self._preprocessor.get_targets_for_metrics(event=event, task_names=self._task_names, max_event_num=self._max_event_num)

            '''
            metrics_targets['ppk'] = [-1e7]: p exits but shift out of window
                                   = [-1]: p not exits
            metrics_targets['det'] = [1,0]: no event
                                   = [-1,-1]: no label
            '''
            if event['p_exits'] == [0]:
                metrics_targets['ppk'] = np.array([-1])
                metrics_targets['det'] = np.array([-1,-1])
            if event['s_exits'] == [0]:
                metrics_targets['spk'] = np.array([-1])
                metrics_targets['det'] = np.array([-1,-1])
        
        elif isinstance(self._label_names, list) and len(self._label_names) == 1 : # for single dis/emg task
            loss_targets = {self._label_names[0]:
                self._preprocessor.get_targets_for_loss(
                event=event, label_names=self._label_names)
            }
            metrics_targets = self._preprocessor.get_targets_for_metrics(
                event=event, task_names=self._task_names, max_event_num=self._max_event_num
            )
            if 'ppks_type' in event.keys() and 'spks_type' in event.keys():
                metrics_targets['ppks_type'] = np.array([event['ppks_type']])
                metrics_targets['spks_type'] = np.array([event['spks_type']])
        
        # add by lhl         
        # # task_mask
        # # note: task order in `task_mask` can be found in help_builder.py `get_labels_tasks` function
        # # here will filter some labels which is affected by the shift of ppk and spk & invalid labels
        # task_mask = np.zeros(len(self._task_names), dtype=bool)
        # p_exist = (metrics_targets["ppk"] != int(-1e7)).any()
        # s_exist = (metrics_targets["spk"] != int(-1e7)).any()
        # for i, task in enumerate(self._task_names):
        #     if task == "ppk" and p_exist:
        #         task_mask[i] = 1
            
        #     elif task == "spk" and s_exist:
        #         task_mask[i] = 1

        #     elif task == "det":
        #         assert len(metrics_targets["det"]) == 2, f"only support 1 event, but det:{metrics_targets['det']}"
        #         if metrics_targets[task][0] != 1 and metrics_targets[task][1] != 0 and p_exist and s_exist:
        #             task_mask[i] = 1

        #     elif task == "dis":
        #         if metrics_targets[task].size > 0 and p_exist and metrics_targets[task][0] > 0 and metrics_targets[task][0] < 500:
        #             task_mask[i] = 1

        #     elif task == "emg":
        #         if metrics_targets[task].size > 0 and p_exist and metrics_targets[task][0] >= 0 and metrics_targets[task][0] < 9:
        #             task_mask[i] = 1

        # metrics_targets["task_mask"] = task_mask

        ## visualize
        #linewidth=0.3
        #fontsize=5
        #import matplotlib.pyplot as plt
        #fig = plt.figure()
        #num_row = 3
        #pidx = event["ppks"][0]
        #print("shape of inputs: ",inputs.shape)
        #for idx, wave in enumerate(inputs):
        #    plt.subplot(num_row, 1, idx + 1)
        #    plt.plot(wave, "-", color="k", linewidth=linewidth)
        #    plt.text(
        #        0.001,
        #        0.95,
        #        f"Channel-{idx}",
        #        horizontalalignment="left",
        #        verticalalignment="top",
        #        transform=plt.gca().transAxes,
        #        fontsize="small",
        #        fontweight="normal",
        #    )
        #    if np.isnan(pidx) == False:
        #        plt.axvline(x=pidx, color='b', linestyle='-', linewidth=linewidth)

        #    plt.xlim(0, len(wave))

        #plt.tight_layout()
        #plt.subplots_adjust(wspace=0.1, hspace=1)

        #save_dir = os.path.join("/public/home/zhangbei/work_dir/DiTing/single_task_custom","plots")
        #if not os.path.exists(save_dir):
        #    os.makedirs(save_dir,exist_ok=True)

        #save_count = np.random.randint(100000)
        #plt.savefig(
        #    os.path.join(save_dir,f"{save_count:>06}.png"),
        #    dpi=400,
        #)
        #plt.close()
        ## end visualize
        meta_data_json = json.dumps(meta_data)
        # print(inputs,loss_targets,metrics_targets,meta_data_json)
        return inputs, loss_targets, metrics_targets,meta_data_json
