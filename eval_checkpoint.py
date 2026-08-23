"""
Evaluate a trained TEAM checkpoint on train/validation/test splits.

Usage:
    python eval_checkpoint.py --config pga_configs/transformer_japan_overfit.json \
        --diting_config ./diting/config/conf_reg.yml \
        --checkpoint weights_japan_overfit/full_model_200.pth \
        --output eval_results.npz \
        [--single_station_checkpoint weights_japan_overfit/single_station_best.pth] \
        [--overfit_n 16] [--device cuda:0] [--input_station_selection epidist] \
        [--case_station_sweep --case_station_counts 3,5,8,12,16,25] \
        [--splits val] [--waveform_station_permutation roll] \
        [--num_shards 4 --shard_id 0]
"""

import argparse
import copy
import json
import math
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'diting'))
sys.path.insert(0, os.path.join(_dir, '..', 'ditingbench'))
import gemini_models as models
import gemini_util_light as util
import loader_light as loader

# Reuse the same build_diting_args / overfit split logic from train_light.py
from train_light import (
    CHECKPOINT_ENCODER_PREFIXES,
    SingleStationTaskDataset,
    build_diting_args as load_diting_args,
    build_overfit_event_metadata_splits,
    clean_state_dict_keys,
    dpk_prior_cache_for_split,
    expand_partitioned_generator_params,
    indexed_config_override,
    load_config_file,
    load_model_state_dict_compatible,
    metadata_cache_stub_path,
    read_overfit_event_ids,
)


def shifted_p_picks_array(p_picks):
    if isinstance(p_picks, dict):
        shifted = p_picks.get('shifted')
        return shifted.numpy() if isinstance(shifted, torch.Tensor) else np.array(shifted)
    return p_picks.numpy() if isinstance(p_picks, torch.Tensor) else np.array(p_picks)


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.numpy()
    return np.array(value)


def _is_point_output(out_np):
    return out_np.ndim <= 1 or (out_np.ndim >= 2 and out_np.shape[-1] == 1)


def _pga_norm_config(config):
    cfg = config.get('training_params', {}).get('pga_target_normalization') or {}
    if not cfg.get('enabled', False):
        return None
    mean = cfg.get('mean')
    std = cfg.get('std')
    if mean in (None, 'auto') or std in (None, 'auto'):
        print('[WARN] PGA target normalization is enabled but mean/std are unresolved; using raw model outputs.')
        return None
    return float(mean), max(float(std), 1e-8)


def _maybe_unnormalize_pga(name, arr, config):
    if name != 'pga':
        return arr
    norm = _pga_norm_config(config)
    if norm is None:
        return arr
    mean, std = norm
    return arr * std + mean


def _maybe_unnormalize_pga_delta(arr, config):
    norm = _pga_norm_config(config)
    if norm is None:
        return arr
    _mean, std = norm
    return arr * std


def _pga_model_space_threshold(threshold, config):
    norm = _pga_norm_config(config)
    if norm is None:
        return float(threshold)
    mean, std = norm
    return (float(threshold) - mean) / std


def _maybe_unnormalize_pga_sigma(name, arr, config):
    if name != 'pga':
        return arr
    norm = _pga_norm_config(config)
    if norm is None:
        return arr
    _mean, std = norm
    return arr * std


def _softmax_np(logits, axis=-1):
    logits = np.asarray(logits)
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _mixture_stats_from_output(out_np):
    out_np = np.asarray(out_np)
    d = (out_np.shape[-1] - 1) // 2
    weights = _softmax_np(out_np[..., 0], axis=-1)
    mu = out_np[..., 1:1 + d]
    sigma = np.maximum(out_np[..., 1 + d:1 + 2 * d], 1e-8)
    mean = np.sum(weights[..., None] * mu, axis=-2)
    second = np.sum(weights[..., None] * (sigma ** 2 + mu ** 2), axis=-2)
    var = np.maximum(second - mean ** 2, 0.0)
    return weights, mu, sigma, mean, np.sqrt(var)


def _mixture_tail_prob_1d(weights, mu, sigma, threshold):
    z = (float(threshold) - mu[..., 0]) / np.maximum(sigma[..., 0], 1e-8)
    erfc = np.vectorize(math.erfc)
    tail = 0.5 * erfc(z / math.sqrt(2.0))
    return np.sum(weights * tail, axis=-1)


def _logsumexp_np(values, axis=-1):
    values = np.asarray(values, dtype=np.float64)
    maximum = np.max(values, axis=axis, keepdims=True)
    summed = np.sum(np.exp(values - maximum), axis=axis, keepdims=True)
    return np.squeeze(maximum + np.log(np.maximum(summed, np.finfo(np.float64).tiny)), axis=axis)


def _mixture_nll_1d(weights, mu, sigma, target, alpha_logits=None):
    """Return per-target MDN NLL in the same coordinate system as mu/sigma."""
    weights = np.asarray(weights, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)[..., 0]
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64)[..., 0], 1e-8)
    target = np.asarray(target, dtype=np.float64).reshape(weights.shape[:-1])
    target = target[..., None]
    if alpha_logits is None:
        log_weights = np.log(np.maximum(weights, np.finfo(np.float64).tiny))
    else:
        alpha_logits = np.asarray(alpha_logits, dtype=np.float64)
        log_weights = alpha_logits - _logsumexp_np(alpha_logits, axis=-1)[..., None]
    log_component = (
        log_weights
        - 0.5 * math.log(2.0 * math.pi)
        - np.log(sigma)
        - 0.5 * ((target - mu) / sigma) ** 2
    )
    return -_logsumexp_np(log_component, axis=-1)


def _point_mu_from_output(name, out_np):
    if name == 'pga':
        return np.asarray(out_np)[..., 0]
    return np.asarray(out_np).reshape(-1)


def append_pga_temporal_residual_outputs(results, raw_model, config):
    tensors = {
        'pga_temporal_base': getattr(raw_model, '_last_pga_temporal_base', None),
        'pga_temporal_delta': getattr(raw_model, '_last_pga_temporal_delta', None),
        'pga_temporal_pred': getattr(raw_model, '_last_pga_temporal_pred', None),
        'pga_temporal_final': getattr(raw_model, '_last_pga_temporal_final', None),
    }
    for key, value in tensors.items():
        if value is None:
            continue
        arr = value.detach().cpu().numpy().squeeze(0)
        if key == 'pga_temporal_delta':
            arr = _maybe_unnormalize_pga_delta(arr, config)
        elif key == 'pga_temporal_pred':
            mode = getattr(raw_model, 'pga_temporal_residual_mode', 'residual')
            if mode == 'absolute':
                arr = _maybe_unnormalize_pga('pga', arr, config)
            else:
                arr = _maybe_unnormalize_pga_delta(arr, config)
        else:
            arr = _maybe_unnormalize_pga('pga', arr, config)
        results[key].append(arr)


def build_model_and_load(config, diting_args, checkpoint_path, device):
    """Build model and load checkpoint."""
    full_model = models.build_transformer_model(
        **config['model_params'], trace_length=10000, diting_args=diting_args)
    full_model = full_model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    load_model_state_dict_compatible(
        full_model,
        state_dict,
        strict=True,
        context=checkpoint_path,
        allowed_missing_prefixes=tuple(checkpoint.get('excluded_prefixes', CHECKPOINT_ENCODER_PREFIXES)),
    )
    full_model._eval_checkpoint_metadata = {
        key: checkpoint.get(key)
        for key in ('epoch', 'loss', 'checkpoint_format', 'encoder_source')
        if checkpoint.get(key) is not None
    }
    full_model.eval()

    epoch = checkpoint.get('epoch', '?')
    print(f'Loaded checkpoint: {checkpoint_path} (epoch {epoch})')
    return full_model


def _load_checkpoint_state(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    return checkpoint, clean_state_dict_keys(state_dict)


def build_single_station_model_and_load(config, diting_args, checkpoint_path, device):
    checkpoint, state_dict = _load_checkpoint_state(checkpoint_path, device)
    pretrain_params = config['training_params'].get('single_station_pretrain', {})
    model_params = copy.deepcopy(config['model_params'])
    model_params['single_station_tasks'] = checkpoint.get(
        'tasks',
        pretrain_params.get('tasks', ['mag', 'epidist', 'pga']),
    )
    model_params['single_station_hidden_dim'] = pretrain_params.get(
        'hidden_dim', model_params.get('single_station_hidden_dim', None)
    )
    model_params['single_station_task_output_init'] = pretrain_params.get('task_output_init', None)
    model_params['single_station_task_sigma_init'] = pretrain_params.get('task_sigma_init', None)

    single_model = models.build_single_station_model(
        **model_params,
        trace_length=10000,
        diting_args=diting_args,
    )
    missing, unexpected = load_model_state_dict_compatible(
        single_model,
        state_dict,
        strict=False,
        context=checkpoint_path,
    )
    single_model = single_model.to(device)
    single_model.eval()

    epoch = checkpoint.get('epoch', '?') if isinstance(checkpoint, dict) else '?'
    print(f'Loaded single-station checkpoint: {checkpoint_path} (epoch {epoch})')
    print(f'  single tasks: {single_model.tasks}')
    if missing:
        print(f'  missing tensors: {len(missing)}')
    if unexpected:
        print(f'  unexpected tensors: {len(unexpected)}')
    return single_model


def _canonical_eval_splits(splits=None):
    if splits is None:
        return ['train', 'val']
    if isinstance(splits, str):
        values = splits.split(',')
    else:
        values = splits
    aliases = {
        'train': 'train',
        'training': 'train',
        'val': 'val',
        'validation': 'val',
        'dev': 'val',
        'test': 'test',
    }
    parsed = []
    for value in values:
        name = str(value).strip().lower()
        if not name:
            continue
        if name not in aliases:
            raise ValueError(
                f'Unknown eval split {value!r}; expected train, val/validation/dev, or test.'
            )
        canonical = aliases[name]
        if canonical not in parsed:
            parsed.append(canonical)
    if not parsed:
        raise ValueError('At least one eval split is required.')
    return parsed


def build_datasets(config, overfit_n=0, input_station_selection='config', splits=None):
    """Build deterministic eval datasets, matching train_light.py split logic."""
    training_params = config['training_params']
    generator_params = expand_partitioned_generator_params(training_params)
    requested_splits = _canonical_eval_splits(splits)

    overwrite_sampling_rate = training_params.get('overwrite_sampling_rate', None)
    min_stalta_ratio_at_pick = training_params.get('min_stalta_ratio_at_pick', 0.1)
    metadata_cache_columns = training_params.get('metadata_cache_columns', None)
    metadata_cache_stub = metadata_cache_stub_path(training_params)
    os.makedirs(os.path.dirname(metadata_cache_stub), exist_ok=True)

    split_parts = {
        'train': (True, False, False),
        'val': (False, True, False),
        'test': (False, False, True),
    }
    full_data_by_split = {}
    for split_name in requested_splits:
        full_data_by_split[split_name] = [loader.load_events(
            data_path,
            event_metadata_path=metadata_cache_stub,
            parts=split_parts[split_name],
            shuffle_train_dev=g.get('shuffle_train_dev', False),
            custom_split=g.get('custom_split', None),
            min_mag=g.get('min_mag', None),
            mag_key=g.get('key', 'MA'),
            overwrite_sampling_rate=overwrite_sampling_rate,
            decimate_events=g.get('decimate_events', None),
            min_stalta_ratio_at_pick=min_stalta_ratio_at_pick,
            metadata_cache_columns=metadata_cache_columns,
            station_filter=g.get('station_filter', training_params.get('station_filter', None)),
        ) for data_path, g in zip(training_params['data_path'], generator_params)]

    event_metadata_by_split = {
        split_name: [data[0] for data in split_data]
        for split_name, split_data in full_data_by_split.items()
    }
    metadata_by_split = {
        split_name: [data[2] for data in split_data]
        for split_name, split_data in full_data_by_split.items()
    }

    if overfit_n > 0:
        full_data_all = [loader.load_events(
            data_path, event_metadata_path=metadata_cache_stub,
            parts=None,
            shuffle_train_dev=g.get('shuffle_train_dev', False),
            custom_split=g.get('custom_split', None),
            min_mag=g.get('min_mag', None),
            mag_key=g.get('key', 'MA'),
            overwrite_sampling_rate=overwrite_sampling_rate,
            decimate_events=g.get('decimate_events', None),
            min_stalta_ratio_at_pick=min_stalta_ratio_at_pick,
            metadata_cache_columns=metadata_cache_columns,
            station_filter=g.get('station_filter', training_params.get('station_filter', None)))
            for data_path, g in zip(training_params['data_path'], generator_params)]
        fixed_overfit_ids = None
        if training_params.get('overfit_event_ids_path'):
            if len(full_data_all) != 1:
                raise ValueError('overfit_event_ids_path currently supports exactly one data_path')
            fixed_overfit_ids = [read_overfit_event_ids(training_params['overfit_event_ids_path'])]
        overfit_train, overfit_dev, overfit_test, _ = build_overfit_event_metadata_splits(
            full_data_all, generator_params, overfit_n, selected_event_ids=fixed_overfit_ids
        )
        overfit_splits = {
            'train': overfit_train,
            'val': overfit_dev,
            'test': overfit_test,
        }
        for split_name in requested_splits:
            event_metadata_by_split[split_name] = overfit_splits[split_name]
        generator_params = [copy.deepcopy(g) for g in generator_params]
        for gp in generator_params:
            realtime_cfg = gp.get('realtime_training') or {}
            if realtime_cfg.get('enabled', False):
                gp['trigger_based'] = True
                gp['disable_station_foreshadowing'] = True
            else:
                fixed_cutout = gp.get('cutout_end', gp.get('cutout_start', 0))
                gp['trigger_based'] = False
                gp['disable_station_foreshadowing'] = False
                gp['cutout_start'] = fixed_cutout
                gp['cutout_end'] = fixed_cutout
            gp['shuffle_train_dev'] = False
            gp['selection_skew'] = None
            gp['oversample'] = 1
            gp['magnitude_resampling'] = 1.0

    sampling_rates = {
        metadata['sampling_rate']
        for split_name in requested_splits
        for metadata in metadata_by_split[split_name]
    }
    if len(sampling_rates) != 1:
        raise ValueError(f'Eval datasets must share one sampling rate, got {sorted(sampling_rates)}')
    sampling_rate = sampling_rates.pop()
    max_stations = config['model_params']['max_stations']
    n_pga_targets = config['model_params'].get('n_pga_targets', 0)
    no_event_token = config['model_params'].get('no_event_token', False)
    station_experiment_cfg = training_params.get('station_experiment', None)
    train_generator_overrides = training_params.get('train_generator_overrides', None)
    validation_generator_overrides = training_params.get('validation_generator_overrides', None)
    test_generator_overrides = training_params.get(
        'test_generator_overrides',
        validation_generator_overrides,
    )
    dpk_prior_cache_cfg = training_params.get('dpk_prior_cache') or {}

    datasets = {}
    for split_name in requested_splits:
        em_list = event_metadata_by_split[split_name]
        generators = []
        for i, gp in enumerate(generator_params):
            noise_seconds = gp.get('noise_seconds', 5)
            cutout = (sampling_rate * (noise_seconds + gp['cutout_start']),
                      sampling_rate * (noise_seconds + gp['cutout_end']))
            cutout = tuple(int(round(x)) for x in cutout)
            gp_copy = copy.deepcopy(gp)
            gp_copy['transform_target_only'] = gp_copy.get('transform_target_only', True)
            gp_copy['oversample'] = 1  # no oversampling for eval
            defaults = dict(
                coords_target=True, label_smoothing=False, station_blinding=False,
                cutout=cutout, pga_targets=n_pga_targets, max_stations=max_stations,
                sampling_rate=sampling_rate, no_event_token=no_event_token,
                use_coords_rel=config['model_params'].get('use_coords_rel', False),
                use_coords_abs=config['model_params'].get('use_coords_abs', True),
                use_coords_rel_abs_fusion=config['model_params'].get('use_coords_rel_abs_fusion', False),
                use_vs30=config['model_params'].get('use_vs30', False),
                station_experiment=station_experiment_cfg,
                shuffle=False,  # deterministic eval order
            )
            cache_split = {'train': 'train', 'val': 'dev', 'test': 'test'}[split_name]
            defaults.update({
                'dpk_prior_cache': dpk_prior_cache_for_split(
                    training_params,
                    cache_split,
                    dataset_id=i,
                ),
                'dpk_prior_cache_split': cache_split,
                'dpk_prior_cache_dataset_id': i,
                'dpk_prior_cache_align_realtime': dpk_prior_cache_cfg.get(
                    'align_realtime_to_cache',
                    True,
                ),
                'dpk_prior_cache_filter_missing_stations': dpk_prior_cache_cfg.get(
                    'filter_missing_stations',
                    True,
                ),
            })
            if training_params.get('deterministic_sampling', False):
                defaults['deterministic_sampling_seed'] = int(config.get('seed', 42)) + i * 1000003
            split_overrides = {
                'train': train_generator_overrides,
                'val': validation_generator_overrides,
                'test': test_generator_overrides,
            }[split_name]
            split_override = indexed_config_override(split_overrides, i)
            merged = {**defaults, **gp_copy, **split_override}
            # Evaluation must never duplicate or reshuffle events. Test inherits
            # the validation realtime override by default, yielding the same
            # fixed 1/3/5/10/20/40/90-second protocol for rt55.
            merged['oversample'] = 1
            merged['shuffle'] = False
            if input_station_selection and input_station_selection not in ('config', 'default'):
                merged['input_station_selection'] = input_station_selection
                if input_station_selection == 'random':
                    merged['select_first_inputs'] = False
                    merged['selection_skew'] = None
            experiment = merged.get('station_experiment') or {}
            print(
                f'[generator/{split_name}/{i}] '
                f'select_first_inputs={merged.get("select_first_inputs", merged.get("select_first"))}, '
                f'select_first_pga_targets={merged.get("select_first_pga_targets", merged.get("select_first"))}, '
                f'input_station_selection={merged.get("input_station_selection", "config")}, '
                f'integrate={merged.get("integrate", False)}, '
                f'selection_skew={merged.get("selection_skew")}, '
                f'pga_selection_skew={merged.get("pga_selection_skew")}, '
                f'causal_random_input_mask={merged.get("causal_random_input_mask")}, '
                f'max_stations={merged.get("max_stations")}, '
                f'station_experiment={experiment.get("mode") if experiment.get("enabled") else None}, '
                f'cutout=({merged["cutout"][0]}, {merged["cutout"][1]})'
            )
            generators.append(util.PreloadedEventGenerator(
                event_metadata=em_list[i], metadata=metadata_by_split[split_name][i],
                data_path=training_params['data_path'][i],
                generator_params=generator_params[i],
                **merged))
        if len(generators) == 1:
            datasets[split_name] = generators[0]
        else:
            dataset_bias = config['model_params'].get('dataset_bias', False)
            datasets[split_name] = util.JointGenerator(generators, shuffle=False, dataset_id=dataset_bias)
    return datasets


def _pairwise_cosine_summary(vectors):
    if vectors.ndim != 2 or vectors.shape[0] <= 1:
        return None
    vectors = vectors.float()
    normed = vectors / (vectors.norm(dim=-1, keepdim=True) + 1e-8)
    cos_sim = normed @ normed.T
    mask = ~torch.eye(cos_sim.shape[0], dtype=bool, device=cos_sim.device)
    off_diag = cos_sim[mask]
    if off_diag.numel() == 0:
        return None
    return off_diag.min().item(), off_diag.max().item(), off_diag.mean().item()


def _print_pairwise_cosine(label, vectors, indent='  '):
    summary = _pairwise_cosine_summary(vectors)
    if summary is None:
        return
    mn, mx, mean = summary
    print(f'{indent}{label}: min={mn:.4f}, max={mx:.4f}, mean={mean:.4f}')


def _print_tensor_station_similarity(stacked):
    """Print inter-station similarity for tensor features of shape S x ... ."""
    n_station = stacked.shape[0]
    if n_station <= 1:
        return
    flat = stacked.reshape(n_station, -1)
    flat_norm = flat.norm(dim=-1)
    print(
        f'  flat L2 norm: min={flat_norm.min():.4f}, '
        f'max={flat_norm.max():.4f}, mean={flat_norm.mean():.4f}'
    )
    _print_pairwise_cosine('Cosine similarity flat (off-diag)', flat)

    flat_var = flat.var(dim=0)
    print(
        f'  Flat per-element variance: min={flat_var.min():.6f}, '
        f'max={flat_var.max():.6f}, mean={flat_var.mean():.6f}'
    )
    centered = flat - flat.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered.float())
    spectral_energy = singular_values.square()
    if spectral_energy.sum() > 0:
        spectrum = spectral_energy / spectral_energy.sum()
        entropy_rank = torch.exp(
            -(spectrum * spectrum.clamp_min(1e-12).log()).sum()
        )
        participation_rank = 1.0 / spectrum.square().sum().clamp_min(1e-12)
        centered_ratio = (
            centered.norm(dim=-1).mean()
            / flat.norm(dim=-1).mean().clamp_min(1e-12)
        )
        print(
            f'  Station effective rank: entropy={entropy_rank:.3f}, '
            f'participation={participation_rank:.3f}, '
            f'centered/raw norm={centered_ratio:.6f}'
        )

    if stacked.ndim == 2:
        return

    # For DiTing encoder output this is usually S x C x T. Report several
    # projections so a high flat cosine is not mistaken for a complete diagnosis.
    gap_last = stacked.mean(dim=-1).reshape(n_station, -1)
    _print_pairwise_cosine('Cosine similarity GAP-last (off-diag)', gap_last)
    if stacked.ndim >= 3:
        gap_penultimate = stacked.mean(dim=-2).reshape(n_station, -1)
        _print_pairwise_cosine('Cosine similarity GAP-penultimate (off-diag)', gap_penultimate)

    token_view = stacked.flatten(start_dim=1, end_dim=-2).transpose(1, 2)
    token_means = []
    token_mins = []
    token_maxs = []
    for token_idx in range(token_view.shape[1]):
        summary = _pairwise_cosine_summary(token_view[:, token_idx, :])
        if summary is None:
            continue
        token_mins.append(summary[0])
        token_maxs.append(summary[1])
        token_means.append(summary[2])
    if token_means:
        token_means_t = torch.tensor(token_means)
        token_mins_t = torch.tensor(token_mins)
        token_maxs_t = torch.tensor(token_maxs)
        print(
            '  Token-wise cosine mean over last axis: '
            f'min_token_mean={token_means_t.min():.4f}, '
            f'max_token_mean={token_means_t.max():.4f}, '
            f'mean_token_mean={token_means_t.mean():.4f}, '
            f'global_min={token_mins_t.min():.4f}, '
            f'global_max={token_maxs_t.max():.4f}'
        )


def _feature_to_token_view(feature, encoder_dim=None):
    """Return a feature as (T,C) for per-token station diagnostics."""
    feat = feature.squeeze(0).detach().float().cpu()
    if feat.dim() != 2:
        return None
    if encoder_dim is not None:
        if feat.shape[-1] == encoder_dim:
            return feat
        if feat.shape[0] == encoder_dim:
            return feat.transpose(0, 1).contiguous()
    # DiTing f2/f3/f4 are usually C x T and x is usually T x C.
    if feat.shape[0] > feat.shape[1]:
        return feat.transpose(0, 1).contiguous()
    return feat


def _resample_eventness(eventness, length):
    eventness = eventness.detach().float().cpu().reshape(1, 1, -1)
    if eventness.shape[-1] == length:
        return eventness.reshape(-1)
    return F.adaptive_max_pool1d(eventness, int(length)).reshape(-1)


def _weighted_token_summary(tokens, weights):
    weights = weights.to(tokens.dtype).clamp_min(1e-8)
    return (tokens * weights[:, None]).sum(dim=0) / weights.sum().clamp_min(1e-8)


def _effective_count_from_weights(weights):
    probs = weights.float().clamp_min(1e-8)
    probs = probs / probs.sum().clamp_min(1e-8)
    entropy = -(probs * probs.log()).sum()
    return torch.exp(entropy)


def _batched_waveform_padding_mask(inputs):
    if not isinstance(inputs, list) or not inputs or not isinstance(inputs[0], torch.Tensor):
        return None
    waveform = inputs[0]
    if waveform.dim() != 4:
        return None
    expected = (waveform.shape[0], waveform.shape[1], waveform.shape[-1])
    for value in inputs[5:]:
        if (
            isinstance(value, torch.Tensor)
            and value.dtype == torch.bool
            and value.dim() == 3
            and tuple(value.shape) == expected
        ):
            return value
    return None


@torch.no_grad()
def _print_dpk_event_partition_diagnostics(raw_model, inputs_dev, valid_idx):
    has_dpk = (
        getattr(raw_model, '_dpk_head_ref', None) is not None
        or getattr(raw_model, '_dpk_model_ref', None) is not None
    )
    if not has_dpk:
        return
    if not isinstance(raw_model.waveform_model, torch.nn.Sequential) or len(raw_model.waveform_model) < 2:
        return

    waveform_inp = inputs_dev[0]
    station_valid = inputs_dev[2].bool()
    waveform_padding_mask = _batched_waveform_padding_mask(inputs_dev)
    waveform_norm = raw_model._normalize(
        waveform_inp.clone(),
        mode='std',
        axis=3,
        sample_mask=waveform_padding_mask,
    )
    waveforms_masked = waveform_norm * station_valid[:, :, None, None].float()
    if waveform_padding_mask is not None:
        waveforms_masked *= waveform_padding_mask[:, :, None, :].to(waveforms_masked.dtype)
    adapter = raw_model.waveform_model[1]
    encoder_dim = getattr(adapter, 'encoder_dim', None)

    names = ['f2', 'f3', 'f4', 'x']
    event_summaries = {}
    non_event_summaries = {}
    residual_summaries = {}
    same_station_cos = {}
    effective_counts = {}
    shapes = {}

    for station_idx in valid_idx:
        features = raw_model.waveform_model[0](waveforms_masked[:, station_idx, :, :])
        dpk_outputs = raw_model._dpk_outputs(waveforms_masked[:, station_idx, :, :], features)
        if not isinstance(dpk_outputs, dict) or 'det' not in dpk_outputs:
            continue
        det = dpk_outputs['det'].detach().float().cpu()
        if det.dim() == 3 and det.shape[1] == 1:
            det = det.squeeze(1)
        det = det.squeeze(0)
        feature_list = list(features) if isinstance(features, (list, tuple)) else [features]
        for idx, feature in enumerate(feature_list):
            name = names[idx] if idx < len(names) else f'feature{idx}'
            tokens = _feature_to_token_view(feature, encoder_dim=encoder_dim)
            if tokens is None:
                continue
            event_w = _resample_eventness(det, tokens.shape[0]).clamp(0.0, 1.0)
            non_event_w = (1.0 - event_w).clamp_min(1e-8)
            event_vec = _weighted_token_summary(tokens, event_w)
            non_event_vec = _weighted_token_summary(tokens, non_event_w)
            event_summaries.setdefault(name, []).append(event_vec)
            non_event_summaries.setdefault(name, []).append(non_event_vec)
            residual_summaries.setdefault(name, []).append(event_vec - non_event_vec)
            same_station_cos.setdefault(name, []).append(
                torch.nn.functional.cosine_similarity(event_vec, non_event_vec, dim=0)
            )
            effective_counts.setdefault(name, []).append(_effective_count_from_weights(event_w))
            shapes[name] = tuple(tokens.shape)

    if not event_summaries:
        return

    print(f'\n{"="*60}')
    print('  DPK eventness partition diagnostics (1 sample)')
    print(f'{"="*60}')
    consistency = getattr(raw_model, 'dpk_encoder_consistency', None)
    if isinstance(consistency, dict):
        print(
            '  DPK encoder equal/shared: '
            f'all_equal={consistency.get("all_equal")}, '
            f'shared={getattr(raw_model, "dpk_encoder_shared", False)}, '
            f'head_on_current={getattr(raw_model, "dpk_head_on_current_encoder", False)}, '
            f'runtime_policy={getattr(raw_model, "dpk_encoder_runtime_policy", "none")}, '
            f'max_abs_diff={consistency.get("max_abs_diff")}, '
            f'first_mismatch={consistency.get("first_mismatch")}'
        )
    for name in sorted(event_summaries.keys()):
        event_stack = torch.stack(event_summaries[name])
        non_event_stack = torch.stack(non_event_summaries[name])
        residual_stack = torch.stack(residual_summaries[name])
        eff = torch.stack(effective_counts[name])
        same_cos = torch.stack(same_station_cos[name])
        print(f'\n--- {name} eventness partitions, token_view_shape={shapes.get(name)} ---')
        print(
            f'  event prior effective tokens: min={eff.min():.2f}, '
            f'max={eff.max():.2f}, mean={eff.mean():.2f}'
        )
        _print_pairwise_cosine('event-weighted station cosine', event_stack, indent='  ')
        _print_pairwise_cosine('non-event-weighted station cosine', non_event_stack, indent='  ')
        _print_pairwise_cosine('event-minus-non-event residual station cosine', residual_stack, indent='  ')
        print(
            '  same-station event vs non-event cosine: '
            f'min={same_cos.min():.4f}, max={same_cos.max():.4f}, mean={same_cos.mean():.4f}'
        )


def diagnose_diting_features(model, dataset, device):
    """Check whether diting produces distinct features per station."""
    raw_model = model.module if hasattr(model, 'module') else model
    waveform_model = raw_model.waveform_model

    # Identify submodules: waveform_model is nn.Sequential [encoder, EncoderFeatures, ..., dt2team]
    module_names = list(waveform_model._modules.keys())
    print(f'\nwaveform_model submodules: {module_names}')

    # Register hooks on key layers
    captured = {}
    def make_hook(name):
        def hook(module, inp, out):
            if name not in captured:
                captured[name] = []
            # out may be tensor or list of tensors
            if isinstance(out, torch.Tensor):
                captured[name].append(out.detach().cpu())
            elif isinstance(out, (list, tuple)):
                captured[name].append([x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in out])
        return hook

    hooks = []
    for name, module in waveform_model.named_modules():
        if name in module_names:  # top-level submodules only
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Run one sample
    inputs, labels, p_picks = dataset[0]
    station_valid = inputs[2].bool().cpu()
    inputs_dev = [x.unsqueeze(0).to(device) if isinstance(x, torch.Tensor) else x for x in inputs]

    with torch.no_grad():
        _ = raw_model(*inputs_dev)

    for h in hooks:
        h.remove()

    # Analyze: waveform_model is called per-station in FullModel.forward
    # so each hook fires N_stations times per forward pass
    print(f'\n{"="*60}')
    print('  Diting feature diagnostics (1 sample)')
    print(f'{"="*60}')

    total_idx = list(range(len(station_valid)))
    valid_idx = station_valid.nonzero(as_tuple=False).flatten().tolist()

    for name in module_names:
        if name not in captured:
            continue
        feats = captured[name]
        n_calls = len(feats)
        if len(total_idx) != n_calls:
            print(f'\n--- {name} ({n_calls} calls, total stations={len(total_idx)}, valid stations={len(valid_idx)}) ---')
            print('  skipped: hook count does not match station_valid length')
            continue
        feats = [feats[i] for i in valid_idx]
        n_valid = len(feats)
        print(f'\n--- {name} ({n_valid} valid stations out of {n_calls}) ---')

        # For tensor outputs, compute inter-station similarity
        if isinstance(feats[0], torch.Tensor):
            stacked = torch.stack(feats).squeeze(1)  # (n_stations, dim)
            print(f'  output shape per station: {feats[0].shape}')
            print(f'  stacked shape: {stacked.shape}')

            _print_tensor_station_similarity(stacked)
        elif isinstance(feats[0], (list, tuple)):
            print(f'  output is list of {len(feats[0])} elements')
            for j in range(len(feats[0])):
                elem0 = feats[0][j]
                if not isinstance(elem0, torch.Tensor):
                    continue
                print(f'    [{j}] shape={elem0.shape}')
                # Stack same element across stations and compute similarity
                if len(feats) > 1:
                    elems = [feats[s][j] for s in range(len(feats))]
                    # Flatten each to 1D: (C, T) -> (C*T)
                    flat = torch.stack([e.squeeze(0).flatten() for e in elems])  # (n_stations, C*T)
                    _print_pairwise_cosine('cosine sim (flat)', flat, indent='        ')
                    # Also try GAP: average over temporal dim, then cosine sim
                    gap = torch.stack([e.squeeze(0).mean(dim=-1) for e in elems])  # (n_stations, C)
                    _print_pairwise_cosine('cosine sim (GAP)', gap, indent='        ')

    _print_dpk_event_partition_diagnostics(raw_model, inputs_dev, valid_idx)
    print()


@torch.no_grad()
def diagnose_amplitude_sensitivity(model, dataset, device, config, scales=(0.5, 1.0, 2.0)):
    raw_model = model.module if hasattr(model, 'module') else model
    if not getattr(raw_model, 'use_amplitude_info', True):
        print(f'{"="*60}')
        print('  Amplitude sensitivity diagnostics skipped: amplitude path disabled')
        print(f'{"="*60}')
        print()
        return
    inputs, labels, _ = dataset[0]

    waveform_inp = inputs[0].unsqueeze(0).to(device)
    metadata_inp = inputs[1].unsqueeze(0).to(device)
    station_valid = inputs[2].unsqueeze(0).to(device)
    valid_idx = inputs[2].bool().nonzero(as_tuple=False).flatten()

    pga_targets_inp = inputs[3].unsqueeze(0).to(device) if len(inputs) > 3 else None
    pga_target_valid = inputs[4].unsqueeze(0).to(device) if len(inputs) > 4 else None
    extra_inputs = [
        item.unsqueeze(0).to(device) if torch.is_tensor(item) else item
        for item in inputs[5:]
    ]
    inputs_dev = [
        waveform_inp,
        metadata_inp,
        station_valid,
        pga_targets_inp,
        pga_target_valid,
        *extra_inputs,
    ]
    waveform_padding_mask = _batched_waveform_padding_mask(inputs_dev)

    print(f'{"="*60}')
    print('  Amplitude sensitivity diagnostics (1 sample)')
    print(f'{"="*60}')

    base_emb = None
    for scale in scales:
        scaled_waveform = waveform_inp * scale
        waveform_norm = raw_model._normalize(
            scaled_waveform,
            mode='std',
            axis=3,
            sample_mask=waveform_padding_mask,
        )
        waveforms_masked = waveform_norm * station_valid[:, :, None, None].float()
        if waveform_padding_mask is not None:
            waveforms_masked *= waveform_padding_mask[:, :, None, :].to(waveforms_masked.dtype)
        waveforms_emb = torch.stack(
            [
                raw_model._encode_station_waveform(
                    waveforms_masked[:, i, :, :],
                    raw_waveform=scaled_waveform[:, i, :, :],
                    sample_valid_mask=(
                        waveform_padding_mask[:, i, :]
                        if waveform_padding_mask is not None
                        else None
                    ),
                )[0]
                for i in range(waveforms_masked.shape[1])
            ],
            dim=1
        )

        trunk_valid = waveforms_emb[0, valid_idx]
        trunk_norm = trunk_valid.norm(dim=-1).mean().item()

        scale_norm = 0.0
        ratio = 0.0
        if raw_model.waveform_scale_proj is not None and getattr(raw_model, 'use_amplitude_info', True):
            scale_features = raw_model._extract_scale_features(scaled_waveform)
            scale_emb = raw_model.waveform_scale_proj(scale_features)
            scale_valid = scale_emb[0, valid_idx]
            gain_scale_emb = raw_model.waveform_scale_gain * raw_model.waveform_scale_gate * scale_emb
            gain_scale_valid = gain_scale_emb[0, valid_idx]
            scale_norm = gain_scale_valid.norm(dim=-1).mean().item()
            ratio = scale_norm / max(trunk_norm, 1e-8)
        else:
            scale_emb = None

        print(f'\n--- waveform x{scale:.1f} ---')
        print(f'  valid stations: {len(valid_idx)}')
        print(f'  mean ||waveforms_emb||: {trunk_norm:.4f}')
        print(f'  mean ||gain*gate*waveform_scale_proj(scale)||: {scale_norm:.4f}')
        print(f'  injected scale/trunk norm ratio: {ratio:.4f}')

        if base_emb is None:
            base_emb = trunk_valid
        else:
            cos = torch.nn.functional.cosine_similarity(base_emb, trunk_valid, dim=-1)
            print(f'  cosine vs x1.0 per-station: min={cos.min():.4f}, max={cos.max():.4f}, mean={cos.mean():.4f}')

        outputs = raw_model(
            scaled_waveform,
            metadata_inp,
            station_valid,
            pga_targets_inp,
            pga_target_valid,
            *extra_inputs,
        )
        if raw_model.n_pga_targets > 0:
            pga_out = outputs[raw_model.output_layout.index('pga')][0].detach().cpu().numpy()
            if _is_point_output(pga_out):
                mu_best = pga_out[:, 0]
            else:
                _weights, _mu, _sigma, mu_best, _sigma_mix = _mixture_stats_from_output(pga_out)
                if np.asarray(mu_best).shape[-1:] == (1,):
                    mu_best = np.asarray(mu_best)[..., 0]
            mu_best = _maybe_unnormalize_pga('pga', mu_best, config)
            if pga_target_valid is not None:
                valid_pga = pga_target_valid[0].bool().cpu().numpy()
                mu_best = mu_best[valid_pga]
            n_show = min(5, len(mu_best))
            print(f'  PGA mu_best[:{n_show}]: {mu_best[:n_show]}')

    print()


@torch.no_grad()
def diagnose_embedding_scales(model, dataset, device):
    raw_model = model.module if hasattr(model, 'module') else model
    inputs, _, _ = dataset[0]

    waveform_inp = inputs[0].unsqueeze(0).to(device)
    metadata_inp = inputs[1].unsqueeze(0).to(device)
    station_valid = inputs[2].unsqueeze(0).to(device)
    valid_idx = inputs[2].bool().nonzero(as_tuple=False).flatten()
    inputs_dev = [
        item.unsqueeze(0).to(device) if torch.is_tensor(item) else item
        for item in inputs
    ]
    waveform_padding_mask = _batched_waveform_padding_mask(inputs_dev)

    waveform_norm = raw_model._normalize(
        waveform_inp,
        mode='std',
        axis=3,
        sample_mask=waveform_padding_mask,
    )
    waveforms_masked = waveform_norm * station_valid[:, :, None, None].float()
    if waveform_padding_mask is not None:
        waveforms_masked *= waveform_padding_mask[:, :, None, :].to(waveforms_masked.dtype)
    coords_abs = metadata_inp * station_valid[:, :, None].float()
    coords_rel, _ = raw_model._make_relative_coords(coords_abs, station_valid.bool())

    waveforms_emb = torch.stack(
        [
            raw_model._encode_station_waveform(
                waveforms_masked[:, i, :, :],
                raw_waveform=waveform_inp[:, i, :, :],
                sample_valid_mask=(
                    waveform_padding_mask[:, i, :]
                    if waveform_padding_mask is not None
                    else None
                ),
            )[0]
            for i in range(waveforms_masked.shape[1])
        ],
        dim=1
    )
    waveforms_emb_valid = waveforms_emb[0, valid_idx]

    scale_emb = None
    scale_emb_valid = None
    if raw_model.waveform_scale_proj is not None and getattr(raw_model, 'use_amplitude_info', True):
        scale_emb = raw_model.waveform_scale_proj(raw_model._extract_scale_features(waveform_inp))
        scale_emb_valid = scale_emb[0, valid_idx]

    base_waveforms_emb = raw_model.layernorm(waveforms_emb)
    wave_plus_scale = base_waveforms_emb
    if scale_emb is not None and getattr(raw_model, 'use_amplitude_info', True):
        wave_plus_scale = wave_plus_scale + raw_model.waveform_scale_gain * raw_model.waveform_scale_gate * scale_emb
    wave_plus_scale_valid = wave_plus_scale[0, valid_idx]

    coords_feat, coords_emb = raw_model._station_coord_features(coords_abs, coords_rel, station_valid.bool())
    coords_feat_valid = coords_feat[0, valid_idx]
    coords_emb_valid = None if coords_emb is None else coords_emb[0, valid_idx]

    if raw_model.alternative_coords_embedding:
        station_emb = torch.cat([wave_plus_scale, coords_feat], dim=-1)
    else:
        station_emb = wave_plus_scale + coords_feat
    station_emb_valid = station_emb[0, valid_idx]

    def mean_norm(x):
        return x.norm(dim=-1).mean().item()

    print(f'{"="*60}')
    print('  Embedding scale diagnostics (1 sample)')
    print(f'{"="*60}')
    print(f'  valid stations: {len(valid_idx)}')
    print(f'  mean ||raw waveforms_emb||: {mean_norm(waveforms_emb_valid):.4f}')
    print(f'  mean ||layernorm(waveforms_emb)||: {mean_norm(base_waveforms_emb[0, valid_idx]):.4f}')
    if scale_emb_valid is not None:
        print(f'  mean ||scale_emb||: {mean_norm(scale_emb_valid):.4f}')
        print(f'  waveform_scale_gain: {raw_model.waveform_scale_gain:.4f}')
        print(f'  waveform_scale_gate: {float(raw_model.waveform_scale_gate.detach().cpu()):.4f}')
        print(f'  mean ||gain*gate*scale_emb||: {(raw_model.waveform_scale_gain * raw_model.waveform_scale_gate * scale_emb_valid.norm(dim=-1)).mean().item():.4f}')
    print(f'  mean ||layernorm(wave)+scale||: {mean_norm(wave_plus_scale_valid):.4f}')
    if coords_emb_valid is not None:
        print(f'  mean ||coords_emb||: {mean_norm(coords_emb_valid):.4f}')
    else:
        print(f'  mean ||coords_feat||: {mean_norm(coords_feat_valid):.4f}')
    print(f'  mean ||station_emb before transformer||: {mean_norm(station_emb_valid):.4f}')
    print()


def shard_indices(n_items, num_shards=1, shard_id=0):
    num_shards = int(num_shards)
    shard_id = int(shard_id)
    if num_shards <= 1:
        return list(range(n_items))
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f'shard_id must be in [0, {num_shards}), got {shard_id}')
    return [idx for idx in range(n_items) if idx % num_shards == shard_id]


def _station_source_slots(station_valid, mode='none', seed=12345, sample_index=0):
    """Return source station slots for waveform-station mismatch evaluation."""
    mode = str(mode or 'none').strip().lower()
    station_valid = torch.as_tensor(station_valid).bool().flatten()
    n_station = int(station_valid.numel())
    source_slots = torch.arange(n_station, dtype=torch.long)
    valid_slots = torch.nonzero(station_valid, as_tuple=False).flatten().cpu()
    if mode == 'none' or valid_slots.numel() <= 1:
        return source_slots
    if mode == 'roll':
        source_slots[valid_slots] = torch.roll(valid_slots, shifts=1, dims=0)
        return source_slots
    if mode == 'random':
        rng = np.random.default_rng(int(seed) + int(sample_index) * 1009)
        n_valid = int(valid_slots.numel())
        perm = None
        for _ in range(64):
            candidate = rng.permutation(n_valid)
            if not np.any(candidate == np.arange(n_valid)):
                perm = candidate
                break
        if perm is None:
            perm = np.roll(np.arange(n_valid), 1)
        source_slots[valid_slots] = valid_slots[torch.from_numpy(perm).long()]
        return source_slots
    raise ValueError(f'Unknown waveform_station_permutation mode: {mode}')


def _permute_first_dim_if_station_aligned(value, source_slots):
    if not isinstance(value, torch.Tensor):
        return value
    if value.dim() < 1 or int(value.shape[0]) != int(source_slots.numel()):
        return value
    return value.index_select(0, source_slots.to(value.device))


def _cached_token_weights_input_index(inputs):
    """Infer cached DPK token-weight input appended by GeminiDataset when present."""
    if not isinstance(inputs, list) or len(inputs) < 6:
        return None
    if len(inputs) < 3 or not isinstance(inputs[0], torch.Tensor):
        return None
    station_count = int(inputs[0].shape[0])
    for idx in range(len(inputs) - 1, 4, -1):
        candidate = inputs[idx]
        if not isinstance(candidate, torch.Tensor):
            continue
        if (
            torch.is_floating_point(candidate)
            and candidate.dim() >= 3
            and int(candidate.shape[0]) == station_count
        ):
            return idx
    return None


def _waveform_padding_mask_input_index(inputs):
    """Find a per-station, per-sample boolean mask among optional inputs."""
    if not isinstance(inputs, list) or len(inputs) < 6:
        return None
    waveform = inputs[0]
    if not isinstance(waveform, torch.Tensor) or waveform.dim() != 3:
        return None
    for idx in range(5, len(inputs)):
        candidate = inputs[idx]
        if (
            isinstance(candidate, torch.Tensor)
            and candidate.dtype == torch.bool
            and candidate.dim() == 2
            and tuple(candidate.shape) == (waveform.shape[0], waveform.shape[-1])
        ):
            return idx
    return None


def apply_waveform_station_permutation(
        inputs,
        mode='none',
        seed=12345,
        sample_index=0,
        permute_cached_token_weights=True):
    """Permute waveform slots while keeping station metadata and PGA targets fixed."""
    mode = str(mode or 'none').strip().lower()
    if mode == 'none':
        return inputs, None
    if not isinstance(inputs, list) or len(inputs) < 3:
        return inputs, None
    if not isinstance(inputs[0], torch.Tensor) or not isinstance(inputs[2], torch.Tensor):
        return inputs, None

    source_slots = _station_source_slots(
        inputs[2],
        mode=mode,
        seed=seed,
        sample_index=sample_index,
    )
    identity = torch.arange(source_slots.numel(), dtype=torch.long)
    if torch.equal(source_slots.cpu(), identity):
        return inputs, source_slots.numpy()

    permuted = list(inputs)
    permuted[0] = _permute_first_dim_if_station_aligned(permuted[0], source_slots)
    padding_mask_idx = _waveform_padding_mask_input_index(permuted)
    if padding_mask_idx is not None:
        permuted[padding_mask_idx] = _permute_first_dim_if_station_aligned(
            permuted[padding_mask_idx],
            source_slots,
        )
    if permute_cached_token_weights:
        cached_idx = _cached_token_weights_input_index(permuted)
        if cached_idx is not None:
            permuted[cached_idx] = _permute_first_dim_if_station_aligned(
                permuted[cached_idx],
                source_slots,
            )
    return permuted, source_slots.numpy()


@torch.no_grad()
def run_inference(
        model,
        dataset,
        device,
        config,
        indices=None,
        waveform_station_permutation='none',
        waveform_station_permutation_seed=12345,
        permute_cached_token_weights=True):
    """Run inference on all samples, collect predictions and labels."""
    raw_model = model.module if hasattr(model, 'module') else model
    head_names = raw_model.output_layout  # e.g. ['mag', 'loc', 'pga']
    results = defaultdict(list)
    if indices is None:
        indices = range(len(dataset))

    permutation_mode = str(waveform_station_permutation or 'none').strip().lower()
    for idx in indices:
        inputs, labels, p_picks = dataset[idx]
        inputs, source_slots = apply_waveform_station_permutation(
            inputs,
            mode=permutation_mode,
            seed=waveform_station_permutation_seed,
            sample_index=idx,
            permute_cached_token_weights=permute_cached_token_weights,
        )
        results['event_index'].append(int(idx))
        if permutation_mode != 'none':
            if source_slots is None and isinstance(inputs, list) and inputs and isinstance(inputs[0], torch.Tensor):
                source_slots = np.arange(int(inputs[0].shape[0]), dtype=np.int64)
            elif source_slots is None:
                source_slots = np.array([], dtype=np.int64)
            results['waveform_station_permutation'].append(np.asarray(source_slots, dtype=np.int64))
            results['waveform_station_permutation_mode'].append(permutation_mode)
            results['waveform_station_permutation_cached_weights'].append(bool(permute_cached_token_weights))

        # Move to device
        inputs_dev = [x.unsqueeze(0).to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
        outputs = model(*inputs_dev)
        append_pga_temporal_residual_outputs(results, raw_model, config)

        # Save pga_target_valid if present
        if isinstance(inputs, list) and len(inputs) >= 5:
            ptv = inputs[4]
            ptv_np = ptv.numpy() if isinstance(ptv, torch.Tensor) else np.array(ptv)
            results['pga_target_valid'].append(ptv_np)
        if isinstance(inputs, list) and len(inputs) >= 4:
            pga_targets = inputs[3]
            pga_targets_np = pga_targets.numpy() if isinstance(pga_targets, torch.Tensor) else np.array(pga_targets)
            results['pga_target_abs'].append(pga_targets_np)
        if isinstance(inputs, list) and len(inputs) >= 3:
            station_coords = inputs[1]
            station_coords_np = station_coords.numpy() if isinstance(station_coords, torch.Tensor) else np.array(station_coords)
            results['station_coords_abs'].append(station_coords_np)
            station_valid = inputs[2]
            sv_np = station_valid.numpy() if isinstance(station_valid, torch.Tensor) else np.array(station_valid)
            results['station_valid'].append(sv_np)
            results['station_valid_count'].append(int(np.asarray(sv_np).astype(bool).sum()))
        if isinstance(p_picks, dict):
            if 'loc_target_abs' in p_picks:
                results['loc_label_abs'].append(_to_numpy(p_picks['loc_target_abs']))
            if 'loc_center' in p_picks:
                results['loc_center'].append(_to_numpy(p_picks['loc_center']))
            for info_key in (
                'event_id',
                'pga_target_indices',
                'realtime_elapsed_time',
                'realtime_requested_elapsed_time',
                'realtime_current_sample',
                'realtime_first_p_pick_sample',
                'realtime_time_bin',
                'realtime_target_type',
                'realtime_target_lead_time',
                'waveform_valid_sample_count',
                'waveform_valid_seconds',
                'waveform_post_p_valid_sample_count',
                'waveform_post_p_valid_seconds',
                'selected_input_indices',
                'selected_original_input_indices',
                'causal_random_mask_applied',
                'causal_random_available_station_count',
                'causal_random_requested_station_count',
                'causal_random_selected_station_count',
            ):
                if info_key in p_picks:
                    results[info_key].append(_to_numpy(p_picks[info_key]))

        # Parse outputs using model.output_layout
        for i, name in enumerate(head_names):
            out_np = outputs[i].cpu().numpy().squeeze(0)
            results[f'{name}_pred'].append(out_np)

        # Parse labels (same layout as outputs)
        for i, name in enumerate(head_names):
            lab_np = labels[i].numpy() if isinstance(labels[i], torch.Tensor) else np.array(labels[i])
            results[f'{name}_label'].append(lab_np)

        # Extract point prediction. For Gaussian/MDN heads, use the predictive
        # mixture mean as the default point estimate.
        for name in head_names:
            out_np = results[f'{name}_pred'][-1]
            if _is_point_output(out_np):
                mu_best = _point_mu_from_output(name, out_np)
                mu_best = _maybe_unnormalize_pga(name, mu_best, config)
                results[f'{name}_mu_best'].append(mu_best)
                if name == 'loc':
                    if raw_model.loc_target_mode == 'rel' and 'loc_center' in results:
                        center = results['loc_center'][-1]
                        results['loc_mu_best_abs'].append(mu_best + center)
                    else:
                        results['loc_mu_best_abs'].append(mu_best)
            else:
                weights, mu, sigma, mu_mean, sigma_mix = _mixture_stats_from_output(out_np)
                mu_mean = _maybe_unnormalize_pga(name, mu_mean, config)
                sigma_mix = _maybe_unnormalize_pga_sigma(name, sigma_mix, config)
                if name in ('mag', 'pga') and np.asarray(mu_mean).shape[-1:] == (1,):
                    mu_mean = np.asarray(mu_mean)[..., 0]
                    sigma_mix = np.asarray(sigma_mix)[..., 0]
                results[f'{name}_mu_best'].append(mu_mean)
                results[f'{name}_sigma'].append(sigma_mix)
                if name == 'pga':
                    raw_target = np.asarray(
                        results['pga_label'][-1],
                        dtype=np.float64,
                    ).reshape(weights.shape[:-1])
                    norm = _pga_norm_config(config)
                    if norm is None:
                        target_model = raw_target
                        target_std = 1.0
                    else:
                        target_mean, target_std = norm
                        target_model = (raw_target - target_mean) / target_std
                    nll_model = _mixture_nll_1d(
                        weights,
                        mu,
                        sigma,
                        target_model,
                        alpha_logits=out_np[..., 0],
                    )
                    results['pga_nll_model_space'].append(nll_model)
                    # If z=(y-mean)/std, then p_y(y)=p_z(z)/std.  Report the
                    # formal NLL in the raw log10(m/s^2) target coordinate.
                    results['pga_nll_log10_mps2'].append(
                        nll_model + math.log(target_std)
                    )
                    threshold = (
                        (config or {})
                        .get('training_params', {})
                        .get('pga_loss_weighting', {})
                        .get('threshold')
                    )
                    if threshold is not None:
                        threshold_model = _pga_model_space_threshold(threshold, config)
                        results['pga_prob_ge_threshold'].append(
                            _mixture_tail_prob_1d(weights, mu, sigma, threshold_model)
                        )
                if name == 'loc':
                    if raw_model.loc_target_mode == 'rel' and 'loc_center' in results:
                        center = results['loc_center'][-1]
                        results['loc_mu_best_abs'].append(mu_mean + center)
                    else:
                        results['loc_mu_best_abs'].append(mu_mean)

        results['p_picks'].append(shifted_p_picks_array(p_picks))

    return dict(results)


def _parse_int_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = str(value).split(',')
    parsed = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        parsed.append(int(text))
    return parsed


def _parse_name_list(value):
    if value is None:
        return []
    return [item.strip() for item in str(value).split(',') if item.strip()]


def _dataset_random_count_records(dataset):
    records = []
    visited = set()

    def visit(obj):
        oid = id(obj)
        if oid in visited:
            return
        visited.add(oid)
        if hasattr(obj, 'random_input_station_count'):
            records.append((obj, obj.random_input_station_count))
        for child_attr in ('generators',):
            children = getattr(obj, child_attr, None)
            if children is not None:
                for child in children:
                    visit(child)
        child = getattr(obj, 'generator', None)
        if child is not None:
            visit(child)

    visit(dataset)
    return records


def _restore_random_count_records(records):
    for obj, value in records:
        obj.random_input_station_count = value


def _set_random_count_records(records, count):
    choices = [int(count)] if count is not None else None
    for obj, _ in records:
        obj.random_input_station_count = choices


def select_case_event_indices(results, max_events):
    if max_events <= 0:
        return []
    if 'pga_label' not in results or 'pga_mu_best' not in results:
        n_samples = len(next(iter(results.values()))) if results else 0
        return list(range(min(max_events, n_samples)))

    labels = np.asarray(results['pga_label'])
    preds = np.asarray(results['pga_mu_best'])
    n_samples = labels.shape[0]
    labels = labels.reshape(n_samples, -1)
    preds = preds.reshape(n_samples, -1)
    if 'pga_target_valid' in results:
        valid = np.asarray(results['pga_target_valid']).astype(bool).reshape(n_samples, -1)
    else:
        valid = ~np.isnan(labels)

    scored = []
    for idx in range(n_samples):
        mask = valid[idx]
        if not mask.any():
            continue
        residual = preds[idx][mask] - labels[idx][mask]
        scored.append((float(np.mean(np.abs(residual))), idx))
    if not scored:
        return list(range(min(max_events, n_samples)))

    scored.sort()
    candidates = [scored[0][1], scored[-1][1], scored[len(scored) // 2][1]]
    if 'station_valid_count' in results:
        counts = np.asarray(results['station_valid_count'], dtype=np.int64).reshape(-1)
        sparse = sorted(scored, key=lambda x: (counts[x[1]], x[0]))
        dense = sorted(scored, key=lambda x: (-counts[x[1]], x[0]))
        candidates.extend([sparse[0][1], dense[0][1]])

    selected_positions = []
    for idx in candidates:
        if idx not in selected_positions:
            selected_positions.append(idx)
        if len(selected_positions) >= max_events:
            break

    event_indices = results.get('event_index')
    if event_indices is not None:
        event_indices = np.asarray(event_indices, dtype=np.int64).reshape(-1)
        return [int(event_indices[pos]) for pos in selected_positions]
    return selected_positions


@torch.no_grad()
def run_station_count_sweep(model, dataset, device, config, event_indices, station_counts, seed=1234):
    """Evaluate the same events with fixed requested input-station counts."""
    raw_model = model.module if hasattr(model, 'module') else model
    head_names = raw_model.output_layout
    if 'pga' not in head_names:
        print('[WARN] Station-count sweep skipped: model has no PGA output head.')
        return {}

    pga_head_idx = head_names.index('pga')
    results = defaultdict(list)
    records = _dataset_random_count_records(dataset)
    if not records:
        print('[WARN] Station-count sweep could not find random_input_station_count on dataset.')

    rng_state = np.random.get_state()
    try:
        for event_idx in event_indices:
            event_seed = int(seed) + int(event_idx) * 1009
            for count in station_counts:
                _set_random_count_records(records, count)
                # Reuse the same RNG seed for every count of one event so random
                # PGA-target selection stays comparable across the sweep.
                np.random.seed(event_seed)
                inputs, labels, p_picks = dataset[event_idx]

                inputs_dev = [x.unsqueeze(0).to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
                outputs = model(*inputs_dev)
                out_np = outputs[pga_head_idx].cpu().numpy().squeeze(0)
                if _is_point_output(out_np):
                    mu_best = _point_mu_from_output('pga', out_np)
                else:
                    _weights, _mu, _sigma, mu_best, _sigma_mix = _mixture_stats_from_output(out_np)
                    if np.asarray(mu_best).shape[-1:] == (1,):
                        mu_best = np.asarray(mu_best)[..., 0]
                mu_best = _maybe_unnormalize_pga('pga', mu_best, config)

                label_np = labels[pga_head_idx].numpy() if isinstance(labels[pga_head_idx], torch.Tensor) else np.array(labels[pga_head_idx])
                pga_target_valid = None
                if isinstance(inputs, list) and len(inputs) >= 5:
                    pga_target_valid = _to_numpy(inputs[4]).astype(bool)

                station_valid = _to_numpy(inputs[2]).astype(bool) if isinstance(inputs, list) and len(inputs) >= 3 else None
                actual_count = int(station_valid.sum()) if station_valid is not None else -1

                event_id = p_picks.get('event_id', str(event_idx)) if isinstance(p_picks, dict) else str(event_idx)
                results['event_index'].append(int(event_idx))
                results['event_id'].append(str(event_id))
                results['requested_station_count'].append(int(count))
                results['actual_station_count'].append(actual_count)
                results['pga_label'].append(label_np)
                results['pga_mu_best'].append(mu_best)
                if pga_target_valid is not None:
                    results['pga_target_valid'].append(pga_target_valid)
                if isinstance(inputs, list) and len(inputs) >= 4:
                    results['pga_target_abs'].append(_to_numpy(inputs[3]))
                if station_valid is not None:
                    results['station_valid'].append(station_valid)
                    results['station_coords_abs'].append(_to_numpy(inputs[1]))
                if isinstance(p_picks, dict):
                    if 'selected_input_indices' in p_picks:
                        results['selected_input_indices'].append(_to_numpy(p_picks['selected_input_indices']))
                    if 'loc_target_abs' in p_picks:
                        results['loc_label_abs'].append(_to_numpy(p_picks['loc_target_abs']))
                    if 'loc_center' in p_picks:
                        results['loc_center'].append(_to_numpy(p_picks['loc_center']))
                    results['p_picks'].append(shifted_p_picks_array(p_picks))
    finally:
        _restore_random_count_records(records)
        np.random.set_state(rng_state)

    return dict(results)


def print_station_count_sweep_summary(results, split_name):
    if not results:
        return
    requested = np.asarray(results.get('requested_station_count', []), dtype=np.int64)
    if requested.size == 0:
        return
    labels = np.asarray(results['pga_label']).reshape(requested.size, -1)
    preds = np.asarray(results['pga_mu_best']).reshape(requested.size, -1)
    if 'pga_target_valid' in results:
        valid = np.asarray(results['pga_target_valid']).astype(bool).reshape(requested.size, -1)
    else:
        valid = ~np.isnan(labels)

    print(f'\n{"="*60}')
    print(f'  CASE STATION-COUNT SWEEP {split_name.upper()}: {requested.size} runs')
    print(f'{"="*60}')
    for count in sorted(set(requested.tolist())):
        rows = requested == count
        mask = valid[rows]
        if not mask.any():
            continue
        y = labels[rows][mask]
        p = preds[rows][mask]
        residual = p - y
        actual = np.asarray(results.get('actual_station_count', []), dtype=np.int64)
        actual_text = ''
        if actual.size == requested.size:
            actual_text = f', actual_mean={actual[rows].mean():.1f}'
        print(
            f'  requested={count}: runs={int(rows.sum())}{actual_text}, '
            f'targets={int(mask.sum())}, '
            f'MAE={np.mean(np.abs(residual)):.4f}, '
            f'RMSE={np.sqrt(np.mean(residual**2)):.4f}, '
            f'bias={np.mean(residual):.4f}'
        )


def _single_station_valid_mask(tasks, station_valid, p_picks):
    valid = station_valid.bool().clone()
    if 'pga' in tasks:
        input_pga_valid = p_picks.get('input_pga_valid') if isinstance(p_picks, dict) else None
        if input_pga_valid is None:
            raise KeyError(
                'Single-station PGA eval requires input_pga_valid from PreloadedEventGenerator.'
            )
        valid &= input_pga_valid.bool()
    return valid


def _single_station_targets(tasks, metadata, labels, p_picks, station_indices, scale_metadata):
    targets = {}
    if 'mag' in tasks:
        targets['mag'] = np.full(
            len(station_indices),
            float(labels[0].float().reshape(-1)[0]),
            dtype=np.float32,
        )
    if 'epidist' in tasks:
        event_coords = p_picks.get('loc_target_abs') if isinstance(p_picks, dict) else None
        if event_coords is None:
            raise KeyError('Single-station epidist eval requires loc_target_abs in p_pick_info.')
        dist_vals = []
        for station_idx in station_indices:
            dist_km = SingleStationTaskDataset._epicentral_distance_km(
                metadata[station_idx],
                event_coords,
                scale_metadata=scale_metadata,
            )
            dist_vals.append(float(torch.log1p(dist_km).item()))
        targets['epidist'] = np.array(dist_vals, dtype=np.float32)
    if 'pga' in tasks:
        targets['pga'] = p_picks['input_pga_values'][station_indices].float().numpy()
    return targets


@torch.no_grad()
def run_single_station_inference(model, dataset, device, indices=None):
    """Run the single-station pretrain model on every valid input station."""
    raw_model = model.module if hasattr(model, 'module') else model
    tasks = list(raw_model.tasks)
    results = defaultdict(list)
    if indices is None:
        indices = range(len(dataset))

    for event_idx in indices:
        inputs, labels, p_picks = dataset[event_idx]
        waveforms, metadata, station_valid = inputs[:3]
        valid = _single_station_valid_mask(tasks, station_valid, p_picks)
        station_indices = torch.nonzero(valid, as_tuple=False).flatten()
        if station_indices.numel() == 0:
            continue

        waveforms_dev = waveforms[station_indices].to(device)
        outputs = raw_model(waveforms_dev)
        scale_metadata = getattr(dataset, 'scale_metadata', False)
        targets = _single_station_targets(
            tasks,
            metadata,
            labels,
            p_picks,
            station_indices,
            scale_metadata=scale_metadata,
        )

        event_id = p_picks.get('event_id', str(event_idx)) if isinstance(p_picks, dict) else str(event_idx)
        selected_input_indices = p_picks.get('selected_input_indices') if isinstance(p_picks, dict) else None
        for local_idx, station_idx in enumerate(station_indices):
            results['event_index'].append(event_idx)
            results['event_id'].append(event_id)
            results['station_slot'].append(int(station_idx))
            if selected_input_indices is not None:
                results['selected_input_index'].append(int(selected_input_indices[station_idx]))

        for task in tasks:
            pred = outputs[task].detach().cpu().numpy()
            results[f'{task}_pred'].append(pred)
            results[f'{task}_mu'].append(pred[:, 0])
            results[f'{task}_sigma'].append(pred[:, 1])
            results[f'{task}_label'].append(targets[task])

        embedding = outputs.get('embedding')
        if embedding is not None:
            emb = embedding.detach().cpu()
            results['embedding_norm'].append(emb.norm(dim=-1).numpy())

    packed = {}
    for key, values in results.items():
        if not values:
            packed[key] = np.array([])
        elif isinstance(values[0], np.ndarray):
            packed[key] = np.concatenate(values, axis=0)
        else:
            packed[key] = np.array(values)
    return packed


def print_single_station_summary(results, split_name):
    n_samples = len(results.get('event_index', []))
    print(f'\n{"="*60}')
    print(f'  SINGLE-STATION {split_name.upper()} set: {n_samples} station samples')
    print(f'{"="*60}')
    if n_samples == 0:
        return

    for task in ['mag', 'epidist', 'pga']:
        label_key = f'{task}_label'
        mu_key = f'{task}_mu'
        if label_key not in results or mu_key not in results:
            continue
        labels = np.asarray(results[label_key]).reshape(-1)
        preds = np.asarray(results[mu_key]).reshape(-1)
        residuals = preds - labels
        print(f'\n--- single/{task} ---')
        for i in range(min(len(labels), 12)):
            print(f'  [{i:2d}] label={labels[i]:.4f}, pred={preds[i]:.4f}')
        print(f'  MAE={np.mean(np.abs(residuals)):.4f}, RMSE={np.sqrt(np.mean(residuals**2)):.4f}')
        if len(labels) > 1:
            corr = np.corrcoef(labels, preds)[0, 1]
            print(f'  Correlation: {corr:.4f}')

    if 'embedding_norm' in results and len(results['embedding_norm']) > 0:
        norms = np.asarray(results['embedding_norm']).reshape(-1)
        print(f'\n--- single/embedding ---')
        print(f'  norm mean={norms.mean():.4f}, std={norms.std():.4f}, min={norms.min():.4f}, max={norms.max():.4f}')


def resolve_single_station_checkpoint(config, explicit_path=None):
    if explicit_path:
        return explicit_path if os.path.exists(explicit_path) else None

    training_params = config['training_params']
    configured = training_params.get('station_pretrain_path', None)
    if configured is None:
        configured = training_params.get('single_station_checkpoint', None)
    if configured:
        return configured if os.path.exists(configured) else None

    pretrain_params = training_params.get('single_station_pretrain', {})
    if not pretrain_params.get('enabled', False):
        return None

    weight_path = training_params.get('weight_path', '')
    candidates = [
        os.path.join(weight_path, 'single_station_best.pth'),
        os.path.join(weight_path, 'single_station_last.pth'),
        os.path.join(weight_path, 'single_station_final.pth'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _stack_result_array(results, key, dtype=None):
    values = results.get(key)
    if not values:
        return None
    arrs = [np.asarray(v) for v in values]
    try:
        stacked = np.stack(arrs)
    except ValueError:
        stacked = np.asarray(arrs, dtype=object)
    if dtype is not None and getattr(stacked, 'dtype', None) != object:
        stacked = stacked.astype(dtype)
    return stacked


def _finite_mean_or_none(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.mean(values))


def compute_formal_pga_metrics(results, config=None):
    """Compute publication-facing PGA metrics over valid target slots.

    MAE/RMSE/R2 and the primary NLL use the raw log10(m/s^2) target
    coordinate.  NLL and Brier are unweighted evaluation metrics; training
    loss weighting and auxiliary losses are intentionally not applied here.
    """
    labels = _stack_result_array(results, 'pga_label', dtype=float)
    predictions = _stack_result_array(results, 'pga_mu_best', dtype=float)
    if labels is None or predictions is None:
        return {}

    n_samples = int(labels.shape[0])
    labels = np.asarray(labels, dtype=np.float64).reshape(n_samples, -1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(n_samples, -1)
    target_valid = _stack_result_array(results, 'pga_target_valid', dtype=bool)
    if target_valid is None:
        target_valid = np.ones_like(labels, dtype=bool)
    else:
        target_valid = np.asarray(target_valid, dtype=bool).reshape(n_samples, -1)
    valid = target_valid & np.isfinite(labels) & np.isfinite(predictions)

    event_ids = _stack_result_array(results, 'event_id')
    if event_ids is None:
        unique_events = n_samples
    else:
        unique_events = len({str(value) for value in np.asarray(event_ids).reshape(-1)})

    metrics = {
        'coordinate': 'log10(m/s^2)',
        'point_estimate': 'predictive_mixture_mean',
        'events': int(unique_events),
        'realtime_samples': n_samples,
        'target_slots': int(labels.size),
        'targets': int(valid.sum()),
        'valid_target_fraction': float(valid.mean()) if valid.size else None,
        'mae': None,
        'rmse': None,
        'bias': None,
        'correlation': None,
        'r2': None,
        'slope': None,
        'intercept': None,
        'nll': None,
        'nll_log10_mps2': None,
        'nll_model_space': None,
        'predictive_sigma_mean': None,
        'predictive_sigma_median': None,
        'coverage_1sigma': None,
        'coverage_2sigma': None,
        'brier': None,
        'brier_threshold_log10_mps2': None,
        'brier_positive_rate': None,
        'probability_mean': None,
    }
    if not np.any(valid):
        return metrics

    random_mask_applied = _stack_result_array(
        results,
        'causal_random_mask_applied',
        dtype=bool,
    )
    if random_mask_applied is not None:
        applied = np.asarray(random_mask_applied, dtype=bool).reshape(-1)
        requested = _stack_result_array(
            results,
            'causal_random_requested_station_count',
            dtype=int,
        )
        available = _stack_result_array(
            results,
            'causal_random_available_station_count',
            dtype=int,
        )
        selected = _stack_result_array(
            results,
            'causal_random_selected_station_count',
            dtype=int,
        )
        geometry = {
            'enabled': True,
            'samples': int(applied.size),
            'applied_samples': int(applied.sum()),
            'applied_fraction': float(applied.mean()) if applied.size else None,
        }
        for name, values in (
            ('requested_station_count', requested),
            ('available_station_count', available),
            ('selected_station_count', selected),
        ):
            if values is None:
                continue
            values = np.asarray(values, dtype=np.int64).reshape(-1)
            active_values = values[applied] if values.size == applied.size else values
            if active_values.size:
                unique, counts = np.unique(active_values, return_counts=True)
                geometry[f'{name}_histogram'] = {
                    str(int(key)): int(count)
                    for key, count in zip(unique, counts)
                }
                geometry[f'{name}_mean'] = float(np.mean(active_values))
        metrics['causal_random_input_mask'] = geometry

    label_values = labels[valid]
    prediction_values = predictions[valid]
    residuals = prediction_values - label_values
    metrics['mae'] = float(np.mean(np.abs(residuals)))
    metrics['rmse'] = float(np.sqrt(np.mean(residuals ** 2)))
    metrics['bias'] = float(np.mean(residuals))

    if label_values.size > 1:
        label_std = float(np.std(label_values))
        prediction_std = float(np.std(prediction_values))
        if label_std > 0 and prediction_std > 0:
            metrics['correlation'] = float(
                np.corrcoef(label_values, prediction_values)[0, 1]
            )
        ss_tot = float(np.sum((label_values - np.mean(label_values)) ** 2))
        if ss_tot > 0:
            ss_res = float(np.sum(residuals ** 2))
            metrics['r2'] = float(1.0 - ss_res / ss_tot)
            slope, intercept = np.polyfit(label_values, prediction_values, 1)
            metrics['slope'] = float(slope)
            metrics['intercept'] = float(intercept)

    nll_raw = _stack_result_array(results, 'pga_nll_log10_mps2', dtype=float)
    if nll_raw is not None:
        nll_raw = np.asarray(nll_raw, dtype=np.float64).reshape(n_samples, -1)
        nll_mask = valid & np.isfinite(nll_raw)
        metrics['nll_log10_mps2'] = _finite_mean_or_none(nll_raw[nll_mask])
        metrics['nll'] = metrics['nll_log10_mps2']
    nll_model = _stack_result_array(results, 'pga_nll_model_space', dtype=float)
    if nll_model is not None:
        nll_model = np.asarray(nll_model, dtype=np.float64).reshape(n_samples, -1)
        nll_model_mask = valid & np.isfinite(nll_model)
        metrics['nll_model_space'] = _finite_mean_or_none(
            nll_model[nll_model_mask]
        )

    sigma = _stack_result_array(results, 'pga_sigma', dtype=float)
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=np.float64).reshape(n_samples, -1)
        sigma_mask = valid & np.isfinite(sigma) & (sigma >= 0)
        if np.any(sigma_mask):
            sigma_values = sigma[sigma_mask]
            absolute_error = np.abs(predictions[sigma_mask] - labels[sigma_mask])
            metrics['predictive_sigma_mean'] = float(np.mean(sigma_values))
            metrics['predictive_sigma_median'] = float(np.median(sigma_values))
            metrics['coverage_1sigma'] = float(
                np.mean(absolute_error <= sigma_values)
            )
            metrics['coverage_2sigma'] = float(
                np.mean(absolute_error <= 2.0 * sigma_values)
            )

    pga_probability = _stack_result_array(
        results,
        'pga_prob_ge_threshold',
        dtype=float,
    )
    weighting_cfg = (
        (config or {}).get('training_params', {}).get('pga_loss_weighting') or {}
    )
    threshold = weighting_cfg.get('threshold')
    if pga_probability is not None and threshold is not None:
        pga_probability = np.asarray(
            pga_probability,
            dtype=np.float64,
        ).reshape(n_samples, -1)
        probability_mask = valid & np.isfinite(pga_probability)
        if np.any(probability_mask):
            probabilities = np.clip(pga_probability[probability_mask], 0.0, 1.0)
            outcomes = (labels[probability_mask] >= float(threshold)).astype(np.float64)
            metrics['brier'] = float(np.mean((probabilities - outcomes) ** 2))
            metrics['brier_threshold_log10_mps2'] = float(threshold)
            metrics['brier_positive_rate'] = float(np.mean(outcomes))
            metrics['probability_mean'] = float(np.mean(probabilities))

    return metrics


def _print_pga_metric_line(prefix, label_values, pred_values):
    if label_values.size == 0:
        return
    residuals = pred_values - label_values
    msg = (
        f'{prefix}: targets={label_values.size}, '
        f'MAE={np.mean(np.abs(residuals)):.4f}, '
        f'RMSE={np.sqrt(np.mean(residuals**2)):.4f}, '
        f'bias={np.mean(residuals):.4f}'
    )
    if label_values.size > 1:
        msg += f', corr={np.corrcoef(label_values, pred_values)[0, 1]:.4f}'
    print(msg)


def _finite_regression_mask(labels, preds, sigma=None):
    labels = np.asarray(labels, dtype=float).reshape(-1)
    preds = np.asarray(preds, dtype=float).reshape(-1)
    mask = np.isfinite(labels) & np.isfinite(preds)
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float).reshape(-1)
        if sigma.shape != labels.shape:
            sigma = np.broadcast_to(sigma, labels.shape).reshape(-1)
        mask = mask & np.isfinite(sigma) & (sigma >= 0)
    return labels, preds, mask


def _print_regression_metric_line(prefix, labels, preds, sigma=None):
    labels, preds, mask = _finite_regression_mask(labels, preds, sigma=sigma)
    if not np.any(mask):
        print(f'{prefix}: no finite prediction/label pairs')
        return
    label_values = labels[mask]
    pred_values = preds[mask]
    residuals = pred_values - label_values
    msg = (
        f'{prefix}: N={label_values.size}, '
        f'MAE={np.mean(np.abs(residuals)):.4f}, '
        f'RMSE={np.sqrt(np.mean(residuals**2)):.4f}, '
        f'bias={np.mean(residuals):.4f}'
    )
    if label_values.size > 1 and np.std(label_values) > 0 and np.std(pred_values) > 0:
        msg += f', corr={np.corrcoef(label_values, pred_values)[0, 1]:.4f}'
    ss_tot = np.sum((label_values - np.mean(label_values)) ** 2)
    if ss_tot > 0:
        ss_res = np.sum(residuals ** 2)
        msg += f', R^2={1.0 - ss_res / ss_tot:.4f}'
    if label_values.size > 1 and np.std(label_values) > 0:
        slope, intercept = np.polyfit(label_values, pred_values, 1)
        msg += f', slope={slope:.4f}, intercept={intercept:.4f}'
    print(msg)
    if sigma is not None:
        sigma_values = np.asarray(sigma, dtype=float).reshape(-1)
        if sigma_values.shape != labels.shape:
            sigma_values = np.broadcast_to(sigma_values, labels.shape).reshape(-1)
        sigma_values = sigma_values[mask]
        abs_res = np.abs(residuals)
        print(
            f'{prefix} sigma: mean={np.mean(sigma_values):.4f}, '
            f'median={np.median(sigma_values):.4f}, '
            f'coverage(|err|<=1sigma)={np.mean(abs_res <= sigma_values):.4f}, '
            f'coverage(|err|<=2sigma)={np.mean(abs_res <= 2 * sigma_values):.4f}'
        )


def _print_loc_regression_metrics(prefix, labels, preds, sigma=None):
    labels = np.asarray(labels, dtype=float).reshape(-1, 3)
    preds = np.asarray(preds, dtype=float).reshape(-1, 3)
    sigma_arr = None
    if sigma is not None:
        sigma_arr = np.asarray(sigma, dtype=float).reshape(-1, 3)
    dim_names = ['x/lat', 'y/lon', 'z/depth']
    for dim, dim_name in enumerate(dim_names):
        dim_sigma = sigma_arr[:, dim] if sigma_arr is not None else None
        _print_regression_metric_line(
            f'  {prefix} {dim_name}',
            labels[:, dim],
            preds[:, dim],
            sigma=dim_sigma,
        )
    finite = np.all(np.isfinite(labels), axis=1) & np.all(np.isfinite(preds), axis=1)
    if np.any(finite):
        err_norm = np.linalg.norm(preds[finite] - labels[finite], axis=1)
        print(
            f'  {prefix} vector-error norm: mean={np.mean(err_norm):.4f}, '
            f'median={np.median(err_norm):.4f}, p90={np.percentile(err_norm, 90):.4f}'
        )


def print_realtime_pga_summary(results, labels_flat, preds_flat, valid_mask, config=None):
    elapsed = _stack_result_array(results, 'realtime_elapsed_time', dtype=float)
    if elapsed is None:
        return
    n_samples = labels_flat.shape[0]
    elapsed = np.asarray(elapsed, dtype=float).reshape(n_samples, -1)[:, 0]

    print('  Realtime PGA breakdown:')
    rounded_times = np.round(elapsed[np.isfinite(elapsed)], 3)
    unique_times = np.unique(rounded_times)
    if unique_times.size <= 12:
        print('    By current time:')
        for time_value in sorted(unique_times.tolist()):
            row_mask = np.isclose(np.round(elapsed, 3), time_value)
            mask = valid_mask & row_mask[:, None]
            _print_pga_metric_line(
                f'      t={time_value:g}s',
                labels_flat[mask],
                preds_flat[mask],
            )
    else:
        time_bins = _stack_result_array(results, 'realtime_time_bin', dtype=int)
        if time_bins is not None:
            time_bins = np.asarray(time_bins, dtype=int).reshape(n_samples, -1)[:, 0]
            print('    By train time bin:')
            for bin_id in sorted(x for x in np.unique(time_bins) if x >= 0):
                row_mask = time_bins == bin_id
                mask = valid_mask & row_mask[:, None]
                _print_pga_metric_line(
                    f'      bin={int(bin_id)}',
                    labels_flat[mask],
                    preds_flat[mask],
                )

    target_type = _stack_result_array(results, 'realtime_target_type', dtype=int)
    if target_type is not None:
        target_type = np.asarray(target_type, dtype=int).reshape(n_samples, -1)
        print('    By target type:')
        type_names = {
            0: 'input',
            1: 'triggered_noninput',
            2: 'untriggered',
        }
        for type_id, type_name in type_names.items():
            mask = valid_mask & (target_type == type_id)
            _print_pga_metric_line(
                f'      {type_name}',
                labels_flat[mask],
                preds_flat[mask],
            )

    lead_time = _stack_result_array(results, 'realtime_target_lead_time', dtype=float)
    if lead_time is not None:
        lead_time = np.asarray(lead_time, dtype=float).reshape(n_samples, -1)
        print('    By target lead time:')
        lead_bins = [
            ('<=0s', lead_time <= 0),
            ('0-5s', (lead_time > 0) & (lead_time <= 5)),
            ('5-20s', (lead_time > 5) & (lead_time <= 20)),
            ('20s+', lead_time > 20),
            ('unknown', ~np.isfinite(lead_time)),
        ]
        for name, lead_mask in lead_bins:
            mask = valid_mask & lead_mask
            _print_pga_metric_line(
                f'      lead={name}',
                labels_flat[mask],
                preds_flat[mask],
            )

    station_valid = _stack_result_array(results, 'station_valid', dtype=bool)
    if station_valid is not None:
        station_valid = np.asarray(station_valid, dtype=bool).reshape(n_samples, -1)
        second_bins = [
            ('0-1s', 0.0, 1.0),
            ('1-3s', 1.0, 3.0),
            ('3-10s', 3.0, 10.0),
            ('10-20s', 10.0, 20.0),
            ('20-40s', 20.0, 40.0),
            ('40s+', 40.0, np.inf),
        ]
        for key, label in (
            ('waveform_valid_seconds', 'valid waveform seconds'),
            ('waveform_post_p_valid_seconds', 'post-P valid seconds'),
        ):
            seconds = _stack_result_array(results, key, dtype=float)
            if seconds is None:
                continue
            seconds = np.asarray(seconds, dtype=float).reshape(n_samples, -1)
            seconds_valid = station_valid & np.isfinite(seconds)
            count = seconds_valid.sum(axis=1)
            event_mean = np.divide(
                np.where(seconds_valid, seconds, 0.0).sum(axis=1),
                np.maximum(count, 1),
            )
            event_mean[count == 0] = np.nan
            print(f'    By event-mean {label}:')
            for bin_name, lower, upper in second_bins:
                if np.isinf(upper):
                    row_mask = event_mean >= lower
                else:
                    row_mask = (event_mean >= lower) & (event_mean < upper)
                mask = valid_mask & row_mask[:, None]
                _print_pga_metric_line(
                    f'      {bin_name}',
                    labels_flat[mask],
                    preds_flat[mask],
                )

    cfg = (config or {}).get('training_params', {}).get('pga_loss_weighting') or {}
    threshold = cfg.get('threshold')
    if threshold is not None:
        threshold = float(threshold)
        print(f'    By PGA label threshold {threshold:g}:')
        weak_mask = valid_mask & (labels_flat < threshold)
        strong_mask = valid_mask & (labels_flat >= threshold)
        _print_pga_metric_line('      weak', labels_flat[weak_mask], preds_flat[weak_mask])
        _print_pga_metric_line('      strong', labels_flat[strong_mask], preds_flat[strong_mask])


def print_summary(results, split_name, config=None):
    """Print summary statistics of predictions vs labels."""
    summary_metrics = {}
    # Detect which heads are present from result keys
    head_names = []
    for candidate in ['mag', 'loc', 'pga']:
        if f'{candidate}_label' in results:
            head_names.append(candidate)
    n_samples = len(results[f'{head_names[0]}_label']) if head_names else 0

    print(f'\n{"="*60}')
    print(f'  {split_name.upper()} set: {n_samples} samples')
    print(f'{"="*60}')

    for name in head_names:
        labels = np.array(results[f'{name}_label'])
        mu_best = results.get(f'{name}_mu_best')
        if mu_best is None:
            continue
        mu_best = np.array(mu_best)

        print(f'\n--- {name} ---')
        if name == 'mag':
            for i in range(min(len(labels), 16)):
                pred_val = float(mu_best[i][0]) if mu_best[i].ndim > 0 else float(mu_best[i])
                print(f'  [{i:2d}] label={float(labels[i]):.3f}, pred={pred_val:.3f}')
            mag_sigma = results.get('mag_sigma')
            mag_sigma = np.asarray(mag_sigma).reshape(-1) if mag_sigma is not None else None
            _print_regression_metric_line(
                '  mag aggregate',
                labels.flatten(),
                mu_best.flatten(),
                sigma=mag_sigma,
            )

        elif name == 'loc':
            abs_labels = np.array(results.get('loc_label_abs', labels))
            abs_preds = np.array(results.get('loc_mu_best_abs', mu_best))
            for i in range(min(len(labels), 16)):
                l = labels[i]
                p = mu_best[i]
                print(f'  [{i:2d}] label=[{l[0]:.2f}, {l[1]:.2f}, {l[2]:.2f}], '
                      f'pred=[{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}]')
            residuals = mu_best - labels
            print(f'  MAE per dim: {np.mean(np.abs(residuals), axis=0)}')
            loc_sigma = results.get('loc_sigma')
            loc_sigma = np.asarray(loc_sigma) if loc_sigma is not None else None
            print('  relative/target-coordinate metrics:')
            _print_loc_regression_metrics('loc', labels, mu_best, sigma=loc_sigma)
            if 'loc_label_abs' in results:
                print('  absolute-coordinate view:')
                for i in range(min(len(abs_labels), 16)):
                    l = abs_labels[i]
                    p = abs_preds[i]
                    print(f'  [{i:2d}] label_abs=[{l[0]:.2f}, {l[1]:.2f}, {l[2]:.2f}], '
                          f'pred_abs=[{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}]')
                abs_residuals = abs_preds - abs_labels
                print(f'  ABS MAE per dim: {np.mean(np.abs(abs_residuals), axis=0)}')
                _print_loc_regression_metrics('loc_abs', abs_labels, abs_preds, sigma=loc_sigma)

        elif name == 'pga':
            # Mask invalid targets using pga_target_valid
            labels_flat = labels.reshape(len(labels), -1)
            ptv = results.get('pga_target_valid')
            if ptv is not None:
                mask = np.array(ptv).astype(bool).reshape(len(labels), -1)
            else:
                # Fallback: treat non-NaN as valid (pga padding is NaN)
                import warnings
                warnings.warn('pga_target_valid not found in results; falling back to NaN-based mask')
                mask = ~np.isnan(labels_flat)

            formal_metrics = compute_formal_pga_metrics(results, config=config)
            summary_metrics['pga'] = formal_metrics
            print(
                '  Formal PGA metrics [log10(m/s^2), predictive mixture mean]: '
                f'targets={formal_metrics.get("targets")}, '
                f'MAE={formal_metrics.get("mae")}, '
                f'RMSE={formal_metrics.get("rmse")}, '
                f'R^2={formal_metrics.get("r2")}, '
                f'NLL={formal_metrics.get("nll")}, '
                f'Brier={formal_metrics.get("brier")}, '
                f'coverage_1sigma={formal_metrics.get("coverage_1sigma")}, '
                f'coverage_2sigma={formal_metrics.get("coverage_2sigma")}'
            )
            random_geometry = formal_metrics.get('causal_random_input_mask')
            if random_geometry:
                print(
                    '  Causal random-input mask: '
                    f'applied={random_geometry.get("applied_samples")}/'
                    f'{random_geometry.get("samples")}, '
                    f'selected_mean={random_geometry.get("selected_station_count_mean")}, '
                    f'available_mean={random_geometry.get("available_station_count_mean")}'
                )

            for i in range(min(len(labels), 4)):
                l = labels_flat[i]
                p = mu_best[i]
                v = mask[i]
                n_show = min(5, len(l))
                valid_tag = ['v' if v[j] else '.' for j in range(n_show)]
                print(f'  [{i:2d}] label={l[:n_show]}, pred={p[:n_show]}, valid={"".join(valid_tag)}')

            # Compute metrics only on valid targets
            valid = mask.flatten()
            all_l = labels_flat.flatten()[valid]
            all_p = mu_best.flatten()[valid]
            if len(all_l) > 0:
                residuals = all_p - all_l
                print(f'  MAE={np.mean(np.abs(residuals)):.4f}, RMSE={np.sqrt(np.mean(residuals**2)):.4f} ({valid.sum()}/{len(valid)} valid targets)')
                if len(all_l) > 1:
                    corr = np.corrcoef(all_l, all_p)[0, 1]
                    print(f'  Correlation: {corr:.4f}')
                    ss_res = np.sum((all_p - all_l) ** 2)
                    ss_tot = np.sum((all_l - np.mean(all_l)) ** 2)
                    if ss_tot > 0:
                        r2 = 1.0 - ss_res / ss_tot
                        print(f'  R^2: {r2:.4f}')
                    slope, intercept = np.polyfit(all_l, all_p, 1)
                    print(f'  Linear fit: pred = {slope:.4f} * label + {intercept:.4f}')

                pga_sigma = results.get('pga_sigma')
                if pga_sigma is not None:
                    sigma_flat = np.asarray(pga_sigma).reshape(len(labels), -1)
                    sigma_v = sigma_flat.flatten()[valid]
                    if len(sigma_v) == len(all_l):
                        abs_res = np.abs(all_p - all_l)
                        print(
                            f'  Predictive sigma: mean={np.mean(sigma_v):.4f}, '
                            f'median={np.median(sigma_v):.4f}, '
                            f'coverage(|err|<=1sigma)={np.mean(abs_res <= sigma_v):.4f}, '
                            f'coverage(|err|<=2sigma)={np.mean(abs_res <= 2 * sigma_v):.4f}'
                        )

                pga_prob = results.get('pga_prob_ge_threshold')
                cfg = (config or {}).get('training_params', {}).get('pga_loss_weighting') or {}
                threshold = cfg.get('threshold')
                if pga_prob is not None and threshold is not None:
                    prob_flat = np.asarray(pga_prob).reshape(len(labels), -1)
                    prob_v = np.clip(prob_flat.flatten()[valid], 0.0, 1.0)
                    event_v = (all_l >= float(threshold)).astype(np.float64)
                    if len(prob_v) == len(event_v):
                        brier = np.mean((prob_v - event_v) ** 2)
                        strong_prob = np.mean(prob_v[event_v == 1]) if np.any(event_v == 1) else float('nan')
                        weak_prob = np.mean(prob_v[event_v == 0]) if np.any(event_v == 0) else float('nan')
                        print(
                            f'  P(PGA>={float(threshold):g}) calibration: '
                            f'Brier={brier:.4f}, prob_mean={np.mean(prob_v):.4f}, '
                            f'prob_strong_mean={strong_prob:.4f}, '
                            f'prob_weak_mean={weak_prob:.4f}'
                        )

                if 'pga_temporal_base' in results and 'pga_temporal_delta' in results:
                    base = np.asarray(results['pga_temporal_base']).reshape(len(labels), -1)
                    delta = np.asarray(results['pga_temporal_delta']).reshape(len(labels), -1)
                    base_v = base.flatten()[valid]
                    delta_v = delta.flatten()[valid]
                    target_delta = all_l - base_v
                    final_v = all_p
                    print('  Temporal residual branch:')
                    print(f'    base MAE={np.mean(np.abs(base_v - all_l)):.4f}')
                    print(f'    final MAE={np.mean(np.abs(final_v - all_l)):.4f}')
                    print(f'    final-base MAE gain={np.mean(np.abs(base_v - all_l)) - np.mean(np.abs(final_v - all_l)):.4f}')
                    print(f'    |delta| mean={np.mean(np.abs(delta_v)):.4f}, target |label-base| mean={np.mean(np.abs(target_delta)):.4f}')
                    if len(delta_v) > 1 and np.std(delta_v) > 0 and np.std(target_delta) > 0:
                        print(f'    corr(delta, label-base)={np.corrcoef(delta_v, target_delta)[0, 1]:.4f}')

                station_counts = results.get('station_valid_count')
                if station_counts is not None:
                    station_counts = np.asarray(station_counts, dtype=np.int64)
                    bucket_defs = [
                        ('1', station_counts == 1),
                        ('2-3', (station_counts >= 2) & (station_counts <= 3)),
                        ('4-5', (station_counts >= 4) & (station_counts <= 5)),
                        ('6-10', (station_counts >= 6) & (station_counts <= 10)),
                        ('11-15', (station_counts >= 11) & (station_counts <= 15)),
                        ('16+', station_counts >= 16),
                    ]
                    print('  By input station count:')
                    for bucket_name, event_mask in bucket_defs:
                        if not event_mask.any():
                            continue
                        target_mask = mask[event_mask]
                        if not target_mask.any():
                            continue
                        bucket_l = labels_flat[event_mask][target_mask]
                        bucket_p = mu_best[event_mask][target_mask]
                        bucket_res = bucket_p - bucket_l
                        msg = (
                            f'    n={bucket_name}: events={int(event_mask.sum())}, '
                            f'targets={int(target_mask.sum())}, '
                            f'MAE={np.mean(np.abs(bucket_res)):.4f}, '
                            f'RMSE={np.sqrt(np.mean(bucket_res**2)):.4f}'
                        )
                        if len(bucket_l) > 1:
                            msg += f', corr={np.corrcoef(bucket_l, bucket_p)[0, 1]:.4f}'
                        print(msg)

                print_realtime_pga_summary(results, labels_flat, mu_best, mask, config=config)
            else:
                print(f'  No valid PGA targets found')
    return summary_metrics


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description='Evaluate TEAM checkpoint')
    parser.add_argument('--config', required=True)
    parser.add_argument('--diting_config', default='./diting/config/conf_reg.yml')
    parser.add_argument('--diting_pretrained', default=None)
    parser.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint')
    parser.add_argument('--single_station_checkpoint', default=None,
                        help='Optional single-station checkpoint; overrides training_params.station_pretrain_path')
    parser.add_argument('--skip_single_station', action='store_true',
                        help='Disable single-station checkpoint evaluation')
    parser.add_argument('--output', default='eval_results.npz', help='Output file for results')
    parser.add_argument('--metrics_output', default=None,
                        help='Formal metrics JSON path. Defaults to <output stem>.metrics.json.')
    parser.add_argument('--splits', default='train,val',
                        help='Comma-separated eval splits: train, val/validation/dev, test.')
    parser.add_argument('--overfit_n', type=int, default=0)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--input_station_selection', default='config',
                        choices=['config', 'default', 'random', 'p_pick', 'epidist'],
                        help='Override eval input-station ordering/subsampling. '
                             'epidist selects nearest stations by epicentral distance.')
    parser.add_argument('--case_station_sweep', action='store_true',
                        help='Additionally evaluate selected events at each requested input-station count.')
    parser.add_argument('--case_station_counts', default='3,5,8,12,16,25',
                        help='Comma-separated requested input-station counts for --case_station_sweep.')
    parser.add_argument('--case_splits', default='val',
                        help='Comma-separated splits for --case_station_sweep, e.g. val or train,val.')
    parser.add_argument('--case_event_indices', default='',
                        help='Comma-separated event indices per split. Empty selects representative events automatically.')
    parser.add_argument('--case_max_events', type=int, default=3,
                        help='Number of automatically selected events per split for --case_station_sweep.')
    parser.add_argument('--case_seed', type=int, default=1234,
                        help='Base numpy seed used to keep PGA target selection comparable in the sweep.')
    parser.add_argument('--waveform_station_permutation', default='none',
                        choices=['none', 'roll', 'random'],
                        help='Mismatch eval: permute waveform slots among valid input stations while keeping '
                             'station metadata and PGA targets fixed.')
    parser.add_argument('--waveform_station_permutation_seed', type=int, default=12345,
                        help='Seed for --waveform_station_permutation=random.')
    parser.add_argument('--no_permute_cached_token_weights', action='store_true',
                        help='Do not move cached DPK token weights with permuted waveforms.')
    parser.add_argument('--num_shards', type=int, default=1,
                        help='Split each eval dataset into this many deterministic shards.')
    parser.add_argument('--shard_id', type=int, default=0,
                        help='Shard id to evaluate, in [0, num_shards).')
    parser.add_argument('--skip_diagnostics', action='store_true',
                        help='Skip the three first-sample feature/amplitude/embedding diagnostics.')
    args = parser.parse_args()

    config = load_config_file(args.config)
    args.overfit_n = args.overfit_n or int(config['training_params'].get('overfit_n', 0))
    requested_splits = _canonical_eval_splits(args.splits)

    # Reproducible evaluation
    import random
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(args.device)
    diting_args = load_diting_args(
        args.diting_config,
        device=str(device),
        pretrained_override=args.diting_pretrained,
    )

    print('Building model...')
    model = build_model_and_load(config, diting_args, args.checkpoint, device)

    permute_cached_token_weights = not args.no_permute_cached_token_weights
    if args.waveform_station_permutation != 'none':
        print(
            'Waveform-station mismatch eval: '
            f'mode={args.waveform_station_permutation}, '
            f'seed={args.waveform_station_permutation_seed}, '
            f'permute_cached_token_weights={permute_cached_token_weights}'
        )
        print('Note: feature/amplitude diagnostics use original station-waveform pairing; mismatch is applied during inference.')

    print('Building datasets...')
    datasets = build_datasets(
        config,
        overfit_n=args.overfit_n,
        input_station_selection=args.input_station_selection,
        splits=requested_splits,
    )

    single_station_model = None
    if not args.skip_single_station:
        single_station_checkpoint = resolve_single_station_checkpoint(
            config,
            explicit_path=args.single_station_checkpoint,
        )
        if single_station_checkpoint is None:
            if args.single_station_checkpoint:
                print(f'Single-station checkpoint not found: {args.single_station_checkpoint}')
            else:
                print('No single-station checkpoint found; skipping single-station eval')
        else:
            print('Building single-station model...')
            single_station_model = build_single_station_model_and_load(
                config,
                diting_args,
                single_station_checkpoint,
                device,
            )

    if not args.skip_diagnostics:
        # Diagnose DiTing features on the first available dataset.
        first_dataset = next(iter(datasets.values()))
        diagnose_diting_features(model, first_dataset, device)
        diagnose_amplitude_sensitivity(model, first_dataset, device, config)
        diagnose_embedding_scales(model, first_dataset, device)
    else:
        print('Skipping first-sample feature/amplitude/embedding diagnostics.')

    all_results = {}
    formal_metrics = {}
    case_station_counts = _parse_int_list(args.case_station_counts)
    case_splits = set(_parse_name_list(args.case_splits))
    explicit_case_indices = _parse_int_list(args.case_event_indices)
    for split_name, dataset in datasets.items():
        eval_indices = shard_indices(len(dataset), args.num_shards, args.shard_id)
        shard_text = ''
        if args.num_shards > 1:
            shard_text = f' shard {args.shard_id}/{args.num_shards} ({len(eval_indices)} samples)'
        print(f'\nRunning inference on {split_name} set ({len(dataset)} samples){shard_text}...')
        results = run_inference(
            model,
            dataset,
            device,
            config,
            indices=eval_indices,
            waveform_station_permutation=args.waveform_station_permutation,
            waveform_station_permutation_seed=args.waveform_station_permutation_seed,
            permute_cached_token_weights=permute_cached_token_weights,
        )
        formal_metrics[split_name] = print_summary(
            results,
            split_name,
            config=config,
        )
        # Prefix keys with split name for saving
        for k, v in results.items():
            all_results[f'{split_name}_{k}'] = np.array(v, dtype=object)

        if args.case_station_sweep and split_name in case_splits:
            if not case_station_counts:
                raise ValueError('--case_station_counts must contain at least one positive integer')
            if explicit_case_indices:
                event_indices = [idx for idx in explicit_case_indices if 0 <= idx < len(dataset)]
            else:
                event_indices = select_case_event_indices(results, args.case_max_events)
            if not event_indices:
                print(f'No valid case-study event indices selected for {split_name}; skipping station-count sweep.')
            else:
                print(
                    f'\nRunning station-count sweep on {split_name}: '
                    f'events={event_indices}, counts={case_station_counts}'
                )
                sweep_results = run_station_count_sweep(
                    model,
                    dataset,
                    device,
                    config,
                    event_indices=event_indices,
                    station_counts=case_station_counts,
                    seed=args.case_seed,
                )
                print_station_count_sweep_summary(sweep_results, split_name)
                for k, v in sweep_results.items():
                    all_results[f'case_sweep_{split_name}_{k}'] = np.array(v, dtype=object)

        if single_station_model is not None:
            print(f'\nRunning single-station inference on {split_name} set ({len(dataset)} events){shard_text}...')
            single_results = run_single_station_inference(single_station_model, dataset, device, indices=eval_indices)
            print_single_station_summary(single_results, split_name)
            for k, v in single_results.items():
                all_results[f'single_{split_name}_{k}'] = np.array(v, dtype=object)

    output_parent = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_parent, exist_ok=True)
    np.savez(args.output, **all_results)
    print(f'\nResults saved to {args.output}')

    metrics_output = args.metrics_output
    if metrics_output is None:
        metrics_output = os.path.splitext(args.output)[0] + '.metrics.json'
    metrics_parent = os.path.dirname(os.path.abspath(metrics_output))
    os.makedirs(metrics_parent, exist_ok=True)
    metrics_payload = {
        'schema_version': 1,
        'checkpoint': os.path.abspath(args.checkpoint),
        'checkpoint_metadata': getattr(model, '_eval_checkpoint_metadata', {}),
        'config': os.path.abspath(args.config),
        'splits': requested_splits,
        'num_shards': int(args.num_shards),
        'shard_id': int(args.shard_id),
        'waveform_station_permutation': args.waveform_station_permutation,
        'waveform_station_permutation_seed': int(
            args.waveform_station_permutation_seed
        ),
        'metric_protocol': {
            'pga_coordinate': 'log10(m/s^2)',
            'point_estimate': 'predictive_mixture_mean',
            'nll': 'unweighted mean MDN NLL over valid PGA targets in log10(m/s^2)',
            'brier': 'unweighted mean squared probability error at the configured PGA threshold',
            'coverage': '|predictive_mean-target| <= k * predictive_mixture_std',
        },
        'metrics': formal_metrics,
    }
    with open(metrics_output, 'w', encoding='utf-8') as f:
        json.dump(_json_safe(metrics_payload), f, indent=2, sort_keys=True)
        f.write('\n')
    print(f'Formal metrics saved to {metrics_output}')


if __name__ == '__main__':
    main()
