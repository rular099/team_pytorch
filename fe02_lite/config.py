"""Torch-free configuration. Dataset statistics are fitted by the compute audit."""
import copy
import os
from pathlib import Path

from fe01.config import load, fingerprint, sha256, write_json

VIEWS = {'knet': ('knt', 'single_surface', ''),
         'kiknet_borehole': ('kik', 'borehole', '1'),
         'kiknet_surface': ('kik', 'surface', '2')}
VARIANTS = {'main_d128': {}, 'd64': {'model_dim': 64},
            'amp16': {'amplitude_dim': 16}, 'amp32': {'amplitude_dim': 32},
            'pool4': {'station_pool_queries': 4}, 'no_source_aux': {}}


def resolve(path, view=None, variant=None, seed=None, stage=None):
    cfg = load(path)
    cfg['data']['shard_identity_manifest'] = os.environ.get('FE02_LITE_SHARD_IDENTITY_MANIFEST',cfg['data'].get('shard_identity_manifest',''))
    for key, value in [('dataset_view', view), ('variant', variant), ('seed', seed), ('stage', stage)]:
        if value is not None:
            cfg[key] = value
    arch = dict(model_dim=128, amplitude_dim=8, patch_size=50,
                station_pool_queries=2, attention_heads=4,
                event_readout_layers=1, target_readout_layers=1)
    arch.update(VARIANTS[cfg['variant']])
    cfg['architecture'] = arch
    if cfg['variant'] == 'no_source_aux':
        cfg['loss'].update(location=0.0,magnitude=0.0)
    else:
        cfg['loss'].setdefault('location',cfg['loss']['source_weight'])
        cfg['loss'].setdefault('magnitude',cfg['loss']['source_weight'])
    if cfg['stage'] == 'pilot':
        cfg['training'].update(epochs=1, max_updates=2, global_batch=2, microbatch=1)
        cfg['audit_limits'] = {'train': 8, 'val': 4}
    cfg['run_id'] = f"fe02_lite__{cfg['dataset_view']}__{cfg['stage']}__{cfg['variant']}__seed{cfg['seed']}"
    validate(cfg)
    return cfg


def validate(cfg):
    if cfg.get('experiment') != 'FE02-Lite' or cfg['dataset_view'] not in VIEWS:
        raise ValueError('Explicit FE02-Lite dataset_view required')
    if cfg['variant'] not in VARIANTS or cfg['stage'] not in ('formal', 'pilot'):
        raise ValueError('Unknown variant/stage')
    if cfg['seed'] not in (42, 43, 44):
        raise ValueError('Only preregistered seeds 42/43/44')
    a = cfg['architecture']
    if a['patch_size'] != 50 or a['event_readout_layers'] != 1 or a['target_readout_layers'] != 1:
        raise ValueError('Native token alignment / one-layer readouts are fixed')
    if a['model_dim'] % a['attention_heads'] or not 0 < a['amplitude_dim'] < a['model_dim']:
        raise ValueError('Invalid feature dimensions')
    if cfg['data']['sampling_rate'] != 100 or cfg['data']['units'] != 'm/s^2':
        raise ValueError('No implicit unit conversion / resampling')
    if cfg['training']['precision'] != 'fp32':
        raise ValueError('This implementation locks fp32; AMP state is explicitly null')
    if any(cfg['loss'][k]<0 for k in ('pga_nll','location','magnitude','mean_huber')) or cfg['loss']['pga_nll']<=0:
        raise ValueError('Loss weights must be nonnegative and PGA supervision positive')
    if cfg['holdouts']['station_manifest'] or cfg['holdouts']['source_region_manifest']:
        raise ValueError('Holdout extensions need a separate protocol, outside this matrix')
    return cfg


def sampling_config(cfg):
    """Small FE01 sampler interface, without its KNET-only validation/factory."""
    out = copy.deepcopy(cfg)
    out.update(model_family='diting_pretrained_frozen', native_n_samples=10000)
    out['model_params'] = dict(max_stations=cfg['sampling']['max_stations'],
                               n_pga_targets=cfg['sampling']['targets_per_instance'])
    return out


def code_identity():
    root = Path(__file__).resolve().parents[1]
    paths = [*root.joinpath('fe02_lite').glob('*.py'),
             *root.joinpath('scripts/fe02_lite').glob('*'),
             *root.joinpath('configs/fe02_lite').rglob('*.json')]
    paths += [root / p for p in ('fe01/config.py', 'fe01/data.py', 'fe01/windows.py',
              'fe01/sampling.py', 'fe01/metrics.py', 'gemini_models.py', 'train_light.py',
              'fe01/extractors.py', 'fe01/model.py')]
    # Include the actual external dependency used for encoder construction.
    try:
        import dtbench
        roots = ([Path(dtbench.__file__).parent] if getattr(dtbench,'__file__',None)
                 else [Path(p) for p in getattr(dtbench,'__path__',[])])
        deps = {str(p.relative_to(dependency)): sha256(p) for dependency in roots for p in dependency.rglob('*.py')}
        if not deps:
            deps = {'status': 'no real dependency source available'}
    except ImportError:
        deps = {'status': 'not available'}
    return fingerprint(dict(files={str(p.relative_to(root)): sha256(p) for p in paths if p.is_file()}, dtbench=deps))
