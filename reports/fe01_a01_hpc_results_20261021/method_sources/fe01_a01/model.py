"""Explicit class injection; unchanged state names/shapes and FE01 seed streams."""
import torch
from torch import nn
import gemini_models as legacy
from fe01.model import FE01FullModel, shape_and_scale, common_state, state_fingerprint, model_audit
from fe01.extractors import (EQTSharedTrunk, OriginalTEAM, PhaseNetBottleneck,
                            StationFrontend, checked_asset, load_seisbench)
from . import ON, OFF


def apply_policy(stats, mode):
    if mode == ON:
        return stats
    if mode == OFF:
        return torch.cat((torch.zeros_like(stats[..., :10]), stats[..., 10:]), dim=-1)
    raise ValueError('Unknown amplitude policy: ' + str(mode))


class A01FullModel(FE01FullModel):
    def _extract_scale_features(self, waveform):
        return apply_policy(super()._extract_scale_features(waveform), self.absolute_amplitude_mode)


def build_model(cfg, device='cpu', injected_encoder=None):
    from .provenance import validate
    validate(cfg, production=False)
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
                    station_waveform_model=frontend, full_model_class=A01FullModel)
    model._fe01_coords_only = False
    model.absolute_amplitude_mode = cfg['absolute_amplitude_mode']
    return model.to(device)
