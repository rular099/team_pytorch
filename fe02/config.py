"""Torch-free FE02 protocol expansion and fail-fast architecture validation."""
import copy
from pathlib import Path

from fe01.config import load, validate as validate_base

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    'R0': ('legacy_station_linear', 'none', 'target_cross_attention', False, False, None),
    'A': ('station_nonlinear', 'none', 'target_cross_attention', False, False, [64, 64]),
    'B': ('event_post_add', 'post_add', 'target_cross_attention', True, False, [64, 64]),
    'C': ('event_only', 'event_only', 'query_no_transformer', True, False, [64, 64]),
    'M': ('event_memory', 'memory', 'target_cross_attention', False, True, [64, 64]),
}
ARCH_KEYS = {'pga_readout_mode', 'pga_use_event_context', 'pga_event_memory',
             'pga_output_mlp_dims', 'pga_event_context_init_gate'}


def baseline():
    return load(ROOT / 'configs/fe01/formal/formal__diting_pretrained_frozen__seed42.json', expand=False)


def expand_architecture(cfg):
    cfg = copy.deepcopy(cfg)
    if cfg.get('variant_id') not in VARIANTS:
        raise ValueError('Unknown FE02 variant_id')
    name, fusion, readout, post, memory, dims = VARIANTS[cfg['variant_id']]
    if cfg.get('event_fusion', fusion) != fusion:
        raise ValueError('variant_id / event_fusion conflict')
    cfg.update(variant_name=name, event_fusion=fusion)
    expected = dict(pga_readout_mode=readout, pga_use_event_context=post,
                    pga_event_memory=memory, pga_output_mlp_dims=dims,
                    pga_event_context_init_gate=0.0)
    params = cfg['model_params']
    for key, value in expected.items():
        if key in params and params[key] != value:
            raise ValueError('FE02 fusion conflict: ' + key)
        params[key] = copy.deepcopy(value)
    return cfg


def validate(cfg):
    if cfg.get('fe02') != {'enabled': True, 'version': 1}:
        raise ValueError('FE02 requires explicit enabled/version')
    validate_base(cfg)
    if cfg['model_family'] != 'diting_pretrained_frozen':
        raise ValueError('FE02 uses only the real frozen pretrained DiTing encoder')
    expanded = expand_architecture(cfg)
    if expanded != cfg:
        raise ValueError('FE02 architecture must be explicitly resolved before running')
    params = cfg['model_params']
    fixed = baseline()['model_params']
    unknown = set(params) - set(fixed) - ARCH_KEYS
    if unknown:
        raise ValueError('Unknown FE02 model parameters: ' + ', '.join(sorted(unknown)))
    for key, value in fixed.items():
        if key not in ARCH_KEYS and params.get(key) != value:
            raise ValueError('FE02 common downstream changed: ' + key)
    reference = baseline()
    for section in ('window', 'realtime', 'geometry', 'selection'):
        if cfg[section] != reference[section]:
            raise ValueError('FE02 fixed sampling/selection changed: ' + section)
    if cfg['spatial']['enabled']:
        raise ValueError('Spatial holdout is not in the FE02 default matrix')
    if cfg.get('stage') not in ('formal', 'pilot') or cfg['seed'] not in (42, 43, 44):
        raise ValueError('Unknown FE02 stage/seed')
    expected_training = copy.deepcopy(reference['training'])
    if cfg['stage'] == 'pilot':
        expected_training.update(epochs=1, max_updates=4, global_batch=16, microbatch=1)
        if cfg.get('audit_limits') != {'train_events': 16, 'val_events': 8}:
            raise ValueError('FE02 pilot support limits changed')
    elif cfg.get('audit_limits'):
        raise ValueError('Formal FE02 must not limit train/validation events')
    if cfg['training'] != expected_training or cfg['sampling_seed'] != 42:
        raise ValueError('FE02 fixed optimization/exposure budget changed')
    return cfg
