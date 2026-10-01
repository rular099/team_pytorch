"""Torch-free configuration and identity contracts."""
import copy
import hashlib
import json
import os
import re
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else copy.deepcopy(value)
    return out


def load(path, expand=True, stack=()):
    path = Path(os.path.expandvars(str(path))).resolve()
    if path in stack:
        raise ValueError('Configuration inheritance cycle')
    with path.open() as stream:
        if path.suffix in ('.yml', '.yaml'):
            import yaml
            cfg = yaml.safe_load(stream)
        else:
            cfg = json.load(stream)
    parent = cfg.pop('extends', None)
    if parent:
        cfg = merge(load(path.parent / parent, False, (*stack, path)), cfg)
    return expand_environment(cfg) if expand else cfg


def expand_environment(value):
    if isinstance(value, dict):
        return {k: expand_environment(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_environment(v) for v in value]
    if isinstance(value, str):
        value = os.path.expanduser(os.path.expandvars(value))
        if re.search(r'\$(?:\{|[A-Za-z_])', value):
            raise ValueError('Unresolved environment variable: ' + value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def safe_output(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if root == path or root not in path.parents:
        raise ValueError('Output must be a child of FE01_OUTPUT_ROOT: ' + str(path))
    return path


def validate(cfg):
    if cfg.get('fe01', {}).get('enabled') is not True or cfg['fe01'].get('version') != 2:
        raise ValueError('FE01 requires an explicit enabled=true, version=2')
    if cfg['window']['protocol'] not in ('native_prefix_v2', 'native_rolling_v2'):
        raise ValueError('Unknown window protocol')
    if cfg.get('transfer_model_path') or cfg.get('load_model_path'):
        raise ValueError('FE01 common downstream must start from scratch')
    if cfg['data']['station_filter'] != 'knet' or cfg['data']['units'] != 'm/s^2':
        raise ValueError('FE01 requires KNET acceleration in m/s^2')
    if cfg['data']['sampling_rate'] != 100 or cfg['data']['component_order'] != 'NEZ':
        raise ValueError('Declared Japan data contract is 100 Hz / NS,EW,UD (NEZ)')
    from .windows import CAPABILITIES
    if cfg['native_n_samples']!=CAPABILITIES[cfg['model_family']].native_n_samples:
        raise ValueError('Native capability length cannot be overridden without a new protocol')
    if cfg['window']['pre_p_seconds']!=5 or cfg['window']['strict_causal'] is not True:
        raise ValueError('FE01 V2 preregisters 5s pre-P and strict causal preprocessing')
    params=cfg['model_params']
    for name in ['use_vs30','use_rope','use_target_temporal_pooling','pga_layerwise_refinement','use_pga_temporal_residual']:
        if params.get(name,False): raise ValueError('FE01 base disables '+name)
    if cfg['geometry']['train_random_probability']!=.5 or cfg['realtime']['draws_per_event']!=3:
        raise ValueError('FE01 V2 preregisters 50% random geometry and 3 draws/event/epoch')
    if cfg['training']['scheduler']!='cosine_fixed_updates' or cfg['training']['early_stopping']:
        raise ValueError('Unexpected optimizer budget/stopping protocol')
    if cfg['window']['protocol'] == 'native_rolling_v2' and not cfg['window'].get('rolling_trained', False):
        raise ValueError('Rolling training requires an explicit rolling_trained protocol')
    return cfg
