"""FE01 only overrides preprocessing/frontend; RT55 downstream is reused."""
import contextlib
import hashlib
import time

import numpy as np
import torch
from torch import nn

import gemini_models as legacy

from .extractors import (EQTSharedTrunk, OriginalTEAM, PhaseNetBottleneck,
                         StationFrontend, checked_asset, load_seisbench)


def state_fingerprint(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def common_state(model):
    return {name: tensor for name, tensor in model.state_dict().items() if not name.startswith('waveform_model.')}


def shape_and_scale(raw, mask):
    """Shared valid-prefix statistics, including duration, in SI/log10 units."""
    if mask is None:
        raise ValueError('FE01 requires an explicit received/storage mask')
    mask = mask.bool()
    support = mask.unsqueeze(-2)
    clean = torch.where(support, raw, torch.zeros_like(raw))
    if not torch.isfinite(clean).all():
        raise ValueError('Non-finite valid waveform')
    count = support.sum(-1, keepdim=True).clamp_min(1)
    mean = clean.sum(-1, keepdim=True) / count
    centered = torch.where(support, clean - mean, torch.zeros_like(clean))
    joint_peak = centered.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    normalized = centered / joint_peak
    # Side channel is computed before normalization from valid physical samples.
    std = (centered.square().sum(-1) / count.squeeze(-1)).sqrt()
    rms = (clean.square().sum(-1) / count.squeeze(-1)).sqrt()
    peak = clean.abs().amax(-1)
    log = lambda value: torch.log10(value.clamp_min(1e-12))
    duration = mask.sum(-1).to(raw.dtype) / 100
    stats = torch.cat([log(std), log(rms), log(peak),
                       log(peak.amax(-1, keepdim=True)), torch.log1p(duration).unsqueeze(-1)], dim=-1)
    stats = stats * mask.any(-1, keepdim=True).to(raw.dtype)
    return normalized, stats


class FE01FullModel(legacy.FullModel):
    @contextlib.contextmanager
    def same_cutoff_query_cache(self):
        if self.training:
            raise ValueError('Query cache is evaluation-only')
        if getattr(self, '_fe01_query_cache', None) is not None:
            raise ValueError('Nested query cache')
        self._fe01_query_cache = {}
        try:
            yield
        finally:
            self._fe01_query_cache = None

    def forward(self, *inputs, **kwargs):
        if len(inputs) < 6 or not inputs[2].any(-1).all() or not inputs[5].any(-1).any(-1).all():
            raise ValueError('FE01 no_input: every batch member needs received station samples')
        self._fe01_station_index = 0
        self._fe01_encoder_seconds = 0.0
        return super().forward(*inputs, **kwargs)

    def _normalize(self, waveform, mode='std', axis=3, sample_mask=None):
        normalized, stats = shape_and_scale(waveform, sample_mask)
        self._fe01_scale = stats
        return normalized

    def _extract_scale_features(self, waveform):
        if self.waveform_model.constant and self._fe01_coords_only:
            return torch.zeros_like(self._fe01_scale)
        return self._fe01_scale

    def _encode_station_waveform(self, waveform, raw_waveform=None, collect_tokens=False,
                                 station_context=None, cached_token_weights=None, sample_valid_mask=None):
        if collect_tokens or cached_token_weights is not None:
            raise ValueError('FE01 base disables temporal/DPK enhancements')
        if not sample_valid_mask.any():
            return waveform.new_zeros((len(waveform), self.waveform_model.output_dim)), None, None
        index = self._fe01_station_index
        self._fe01_station_index += 1
        cache = getattr(self, '_fe01_query_cache', None)
        if cache is not None and index in cache:
            old_wave, old_mask, result = cache[index]
            if not torch.equal(old_wave, waveform) or not torch.equal(old_mask, sample_valid_mask):
                raise ValueError('Attempted query cache reuse across different received history')
            return result, None, None
        if waveform.is_cuda:
            torch.cuda.synchronize(waveform.device)
        begin = time.perf_counter()
        result = self.waveform_model(waveform, sample_valid_mask)
        if waveform.is_cuda:
            torch.cuda.synchronize(waveform.device)
        self._fe01_encoder_seconds += time.perf_counter() - begin
        if cache is not None:
            cache[index] = (waveform.detach().clone(), sample_valid_mask.detach().clone(), result.detach())
        return result, None, None


def build_model(cfg, device='cpu', injected_encoder=None):
    from .config import validate
    validate(cfg)
    family = cfg['model_family']
    dim = cfg['model_params']['waveform_model_dims'][-1]
    seed = cfg['seed']
    # Frontend and common initialization use independent streams. Constructing
    # a 1200M backbone cannot consume the common downstream's random stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 19001)
        if family in ('phasenet_pretrained_frozen', 'eqt_pretrained_frozen'):
            name = 'phasenet' if family.startswith('phasenet') else 'eqtransformer'
            picker, entry = load_seisbench(cfg['pretrained_manifest'], name) if injected_encoder is None else (injected_encoder, {'component_order': 'ZNE'})
            encoder = PhaseNetBottleneck(picker) if name == 'phasenet' else EQTSharedTrunk(picker)
            frontend = StationFrontend(encoder, dim, encoder.output_channels, frozen=True,
                                       component_order=entry['component_order'])
        elif family == 'team_original_scratch':
            frontend = StationFrontend(OriginalTEAM(cfg['native_n_samples'], (dim, dim, dim)), dim)
        elif family in ('amplitude_only', 'coords_only'):
            frontend = StationFrontend(nn.Identity(), dim, constant=True)
        elif family.startswith('diting_'):
            files=None
            if family=='diting_pretrained_frozen':
                _,files=checked_asset(cfg['pretrained_manifest'],'diting')
            from train_light import build_diting_args
            args = build_diting_args(cfg['diting_config'], device='cpu', pretrained_override='')
            sequence = legacy.get_diting_model(args, dim)
            if family == 'diting_pretrained_frozen':
                from dtbench.training.modeling import _extract_pretrained_state_dict, _filter_backbone_state_dict
                checkpoint = torch.load(files['weights'], map_location='cpu', weights_only=False)
                state = _filter_backbone_state_dict(_extract_pretrained_state_dict(checkpoint, args))
                expected = {k: v for k, v in sequence.state_dict().items() if k.startswith('0.')}
                if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in expected):
                    raise ValueError('DiTing encoder keys/shapes do not exactly match declared checkpoint')
                sequence[0].load_state_dict({k[2:]: v for k, v in state.items()}, strict=True)
            frontend = StationFrontend(sequence[0], dim, frozen=True, diting_adapter=sequence[1])
        else:
            raise ValueError('Unknown FE01 model family: ' + family)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        params = dict(cfg['model_params'])
        model = legacy.build_transformer_model(**params, trace_length=cfg['native_n_samples'],
                    station_waveform_model=frontend, full_model_class=FE01FullModel)
    model._fe01_coords_only = family == 'coords_only'
    return model.to(device)


def model_audit(model):
    params = list(model.named_parameters())
    return dict(common_initial_state_sha256=state_fingerprint(common_state(model)),
                common_structure_sha256=hashlib.sha256(str([(n,tuple(t.shape)) for n,t in common_state(model).items()]).encode()).hexdigest(),
                total_parameters=sum(p.numel() for _,p in params),
                trainable_parameters=sum(p.numel() for _,p in params if p.requires_grad),
                encoder_parameters=sum(p.numel() for n,p in params if n.startswith('waveform_model.encoder.')),
                trainable_names=[n for n,p in params if p.requires_grad],
                normalization='masked demeaning + joint peak; scale log10/std/rms/peak/duration',
                input_component_order='NEZ', outputs=model.output_layout)
