"""Integer-clock windows. No preprocessing reads beyond the exclusive cutoff."""
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_FLOOR

import numpy as np


@dataclass(frozen=True)
class Capability:
    family: str
    native_n_samples: int
    sampling_rate: int = 100
    pre_p_seconds: float = 5.0
    feature_layer: str = ''
    verified_native_forward: bool = False

    def max_elapsed_sample(self):
        return self.native_n_samples - int(self.pre_p_seconds * self.sampling_rate) - 1

    def manifest(self):
        return dict(asdict(self), max_elapsed_seconds=self.max_elapsed_sample() / self.sampling_rate,
                    endpoint='inclusive current sample; exclusive cutout=current+1',
                    left_padding=True, only_native_length=True,
                    normalization='masked per-component demeaning / joint component peak',
                    capability_claim='declared native shape; verification status recorded separately')


CAPABILITIES = {
    'diting_pretrained_frozen': Capability('diting', 10000, feature_layer='0.backbone + BackboneAttentionPoolAdapter'),
    'diting_random_frozen': Capability('diting', 10000, feature_layer='0.backbone + BackboneAttentionPoolAdapter'),
    'team_original_scratch': Capability('team', 10000, feature_layer='original CNN / flatten / MLP port'),
    'phasenet_pretrained_frozen': Capability('phasenet', 3001, feature_layer='down_branch.4 before decoder'),
    'eqt_pretrained_frozen': Capability('eqtransformer', 6000, feature_layer='transformer_d shared trunk'),
    'amplitude_only': Capability('amplitude_only', 10000),
    'coords_only': Capability('coords_only', 10000),
}


def floor_samples(seconds, rate):
    return int((Decimal(str(seconds)) * Decimal(str(rate))).to_integral_value(rounding=ROUND_FLOOR))


def clock(reference_sample, elapsed_seconds, sampling_rate):
    current = floor_samples(Decimal(str(reference_sample)) / Decimal(str(sampling_rate))
                            + Decimal(str(elapsed_seconds)), sampling_rate)
    return current, current + 1


def build_window(raw, storage_valid, reference_sample, elapsed_seconds, capability,
                 protocol='native_prefix_v2'):
    """raw: station,component,time; storage_valid: same or station,time.

    Requested history is independent of storage gaps. A short record never
    licenses truncation of the scientific prefix. Future NaN/Inf never enter it.
    """
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[1] != 3:
        raise ValueError('Expected raw shape (stations,3,time)')
    valid = np.asarray(storage_valid, dtype=bool)
    if valid.ndim == 2:
        valid = np.broadcast_to(valid[:, None, :], raw.shape)
    if valid.shape != raw.shape:
        raise ValueError('Storage/component mask does not match waveform')
    rate = capability.sampling_rate
    current, cutout = clock(reference_sample, elapsed_seconds, rate)
    if protocol == 'native_prefix_v2':
        start = floor_samples(reference_sample, 1) - floor_samples(capability.pre_p_seconds, rate)
    elif protocol == 'native_rolling_v2':
        start = cutout - capability.native_n_samples
    else:
        raise ValueError('Unknown window protocol')
    required = cutout - start
    info = dict(requested_elapsed_time=float(elapsed_seconds),
                actual_elapsed_time=(current - float(reference_sample)) / rate,
                requested_decision_sample=float(reference_sample) + float(elapsed_seconds) * rate,
                current_sample=current, cutout_exclusive=cutout,
                history_start_sample=start, history_end_sample=current,
                required_n_samples=required, history_seconds=required / rate,
                window_protocol=protocol, status='supported', reason='')
    shape = (*raw.shape[:2], capability.native_n_samples)
    out = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=bool)
    if required <= 0:
        raise ValueError('History interval is empty')
    if required > capability.native_n_samples:
        info.update(status='unsupported', reason='unsupported_history')
        return out, mask, info
    lo, hi = max(0, start), min(raw.shape[-1], cutout)
    if hi > lo:
        destination = capability.native_n_samples - (cutout - lo)
        support = valid[..., lo:hi]
        region = raw[..., lo:hi]
        if not np.isfinite(region[support]).all():
            raise ValueError('Non-finite received waveform (numerical/data failure)')
        out[..., destination:destination + hi - lo] = np.where(support, region, 0)
        mask[..., destination:destination + hi - lo] = support
    info['latest_received_sample'] = min(current, raw.shape[-1] - 1)
    info['valid_seconds_by_station'] = mask.all(axis=1).sum(axis=-1).tolist()
    info['valid_seconds_by_station'] = [v / rate for v in info['valid_seconds_by_station']]
    sample_indices = np.arange(cutout - capability.native_n_samples, cutout)
    for label, selector in [('pre_p_seconds', sample_indices < float(reference_sample)),
                            ('post_p_seconds', sample_indices >= float(reference_sample))]:
        info[label] = (mask.all(axis=1)[:, selector].sum(axis=-1) / rate).tolist()
    info['storage_missing_samples'] = int((~mask).sum())
    info['history_padding_samples'] = max(0, capability.native_n_samples - required)
    return out, mask, info
