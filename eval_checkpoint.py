"""
Evaluate a trained TEAM checkpoint on train/val sets.

Usage:
    python eval_checkpoint.py --config pga_configs/transformer_japan_overfit.json \
        --diting_config ./diting/config/conf_reg.yml \
        --checkpoint weights_japan_overfit/full_model_200.pth \
        --output eval_results.npz \
        [--single_station_checkpoint weights_japan_overfit/single_station_best.pth] \
        [--overfit_n 16] [--device cuda:0] [--input_station_selection epidist] \
        [--case_station_sweep --case_station_counts 3,5,8,12,16,25] \
        [--num_shards 4 --shard_id 0]
"""

import argparse
import copy
import json
import os
import sys
import numpy as np
import torch
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
    load_model_state_dict_compatible,
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


def _point_mu_from_output(name, out_np):
    if name == 'pga':
        return np.asarray(out_np)[..., 0]
    return np.asarray(out_np).reshape(-1)


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


def build_datasets(config, overfit_n=0, input_station_selection='config'):
    """Build train and val datasets, matching train_light.py logic."""
    training_params = config['training_params']
    generator_params = training_params.get('generator_params', [training_params.copy()])

    overwrite_sampling_rate = training_params.get('overwrite_sampling_rate', None)
    min_stalta_ratio_at_pick = training_params.get('min_stalta_ratio_at_pick', 0.1)

    full_data_train = [loader.load_events(
        data_path, event_metadata_path='train_ev.csv',
        parts=(True, False, False),
        shuffle_train_dev=g.get('shuffle_train_dev', False),
        custom_split=g.get('custom_split', None),
        min_mag=g.get('min_mag', None),
        mag_key=g.get('key', 'MA'),
        overwrite_sampling_rate=overwrite_sampling_rate,
        decimate_events=g.get('decimate_events', None),
        min_stalta_ratio_at_pick=min_stalta_ratio_at_pick)
        for data_path, g in zip(training_params['data_path'], generator_params)]

    full_data_dev = [loader.load_events(
        data_path, event_metadata_path='test_ev.csv',
        parts=(False, True, False),
        shuffle_train_dev=g.get('shuffle_train_dev', False),
        custom_split=g.get('custom_split', None),
        min_mag=g.get('min_mag', None),
        mag_key=g.get('key', 'MA'),
        overwrite_sampling_rate=overwrite_sampling_rate,
        decimate_events=g.get('decimate_events', None),
        min_stalta_ratio_at_pick=min_stalta_ratio_at_pick)
        for data_path, g in zip(training_params['data_path'], generator_params)]

    event_metadata_train = [d[0] for d in full_data_train]
    metadata_train = [d[2] for d in full_data_train]
    event_metadata_dev = [d[0] for d in full_data_dev]
    metadata_dev = [d[2] for d in full_data_dev]

    if overfit_n > 0:
        full_data_all = [loader.load_events(
            data_path, event_metadata_path='overfit_ev.csv',
            parts=None,
            shuffle_train_dev=g.get('shuffle_train_dev', False),
            custom_split=g.get('custom_split', None),
            min_mag=g.get('min_mag', None),
            mag_key=g.get('key', 'MA'),
            overwrite_sampling_rate=overwrite_sampling_rate,
            decimate_events=g.get('decimate_events', None),
            min_stalta_ratio_at_pick=min_stalta_ratio_at_pick)
            for data_path, g in zip(training_params['data_path'], generator_params)]
        event_metadata_train, event_metadata_dev, _, _ = build_overfit_event_metadata_splits(
            full_data_all, generator_params, overfit_n
        )
        generator_params = [copy.deepcopy(g) for g in generator_params]
        for gp in generator_params:
            fixed_cutout = gp.get('cutout_end', gp.get('cutout_start', 0))
            gp['trigger_based'] = False
            gp['disable_station_foreshadowing'] = False
            gp['shuffle_train_dev'] = False
            gp['selection_skew'] = None
            gp['oversample'] = 1
            gp['magnitude_resampling'] = 1.0
            gp['cutout_start'] = fixed_cutout
            gp['cutout_end'] = fixed_cutout

    sampling_rate = metadata_train[0]['sampling_rate']
    max_stations = config['model_params']['max_stations']
    n_pga_targets = config['model_params'].get('n_pga_targets', 0)
    no_event_token = config['model_params'].get('no_event_token', False)
    station_experiment_cfg = training_params.get('station_experiment', None)

    datasets = {}
    for split_name, em_list, meta_list in [('train', event_metadata_train, metadata_train),
                                            ('val', event_metadata_dev, metadata_dev)]:
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
                station_experiment=station_experiment_cfg,
                shuffle=False,  # deterministic eval order
            )
            merged = {**defaults, **gp_copy}
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
                f'max_stations={merged.get("max_stations")}, '
                f'station_experiment={experiment.get("mode") if experiment.get("enabled") else None}, '
                f'cutout=({merged["cutout"][0]}, {merged["cutout"][1]})'
            )
            generators.append(util.PreloadedEventGenerator(
                event_metadata=em_list[i], metadata=meta_list[i],
                data_path=training_params['data_path'][i],
                generator_params=generator_params[i],
                **merged))
        if len(generators) == 1:
            datasets[split_name] = generators[0]
        else:
            dataset_bias = config['model_params'].get('dataset_bias', False)
            datasets[split_name] = util.JointGenerator(generators, shuffle=False, dataset_id=dataset_bias)
    return datasets


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

            # L2 norms
            norms = stacked.norm(dim=-1)
            print(f'  L2 norm: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}')

            # Pairwise cosine similarity
            if stacked.ndim == 2 and stacked.shape[0] > 1:
                normed = stacked / (stacked.norm(dim=-1, keepdim=True) + 1e-8)
                cos_sim = normed @ normed.T
                # Exclude diagonal
                n = cos_sim.shape[0]
                mask = ~torch.eye(n, dtype=bool)
                off_diag = cos_sim[mask]
                print(f'  Cosine similarity (off-diag): min={off_diag.min():.4f}, max={off_diag.max():.4f}, mean={off_diag.mean():.4f}')

                # Per-dim variance across stations
                var_per_dim = stacked.var(dim=0)
                print(f'  Per-dim variance: min={var_per_dim.min():.6f}, max={var_per_dim.max():.6f}, mean={var_per_dim.mean():.6f}')
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
                    normed = flat / (flat.norm(dim=-1, keepdim=True) + 1e-8)
                    cos_sim = normed @ normed.T
                    n = cos_sim.shape[0]
                    mask = ~torch.eye(n, dtype=bool)
                    off_diag = cos_sim[mask]
                    # Also try GAP: average over temporal dim, then cosine sim
                    gap = torch.stack([e.squeeze(0).mean(dim=-1) for e in elems])  # (n_stations, C)
                    gap_normed = gap / (gap.norm(dim=-1, keepdim=True) + 1e-8)
                    gap_cos = gap_normed @ gap_normed.T
                    gap_off = gap_cos[mask]
                    print(f'        cosine sim (flat): min={off_diag.min():.4f}, max={off_diag.max():.4f}, mean={off_diag.mean():.4f}')
                    print(f'        cosine sim (GAP):  min={gap_off.min():.4f}, max={gap_off.max():.4f}, mean={gap_off.mean():.4f}')

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

    print(f'{"="*60}')
    print('  Amplitude sensitivity diagnostics (1 sample)')
    print(f'{"="*60}')

    base_emb = None
    for scale in scales:
        scaled_waveform = waveform_inp * scale
        waveform_norm = raw_model._normalize(scaled_waveform, mode='std', axis=3)
        waveforms_masked = waveform_norm * station_valid[:, :, None, None].float()
        waveforms_emb = torch.stack(
            [raw_model.waveform_model(waveforms_masked[:, i, :, :]) for i in range(waveforms_masked.shape[1])],
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
            scaled_waveform, metadata_inp, station_valid,
            pga_targets_inp=pga_targets_inp, pga_target_valid=pga_target_valid
        )
        if raw_model.n_pga_targets > 0:
            pga_out = outputs[-1][0].detach().cpu().numpy()
            if _is_point_output(pga_out):
                mu_best = pga_out[:, 0]
            else:
                alpha = pga_out[:, :, 0]
                mu = pga_out[:, :, 1]
                best = np.argmax(alpha, axis=1)
                mu_best = mu[np.arange(len(best)), best]
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

    waveform_norm = raw_model._normalize(waveform_inp, mode='std', axis=3)
    waveforms_masked = waveform_norm * station_valid[:, :, None, None].float()
    coords_abs = metadata_inp * station_valid[:, :, None].float()
    coords_rel, _ = raw_model._make_relative_coords(coords_abs, station_valid.bool())

    waveforms_emb = torch.stack(
        [raw_model.waveform_model(waveforms_masked[:, i, :, :]) for i in range(waveforms_masked.shape[1])],
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


@torch.no_grad()
def run_inference(model, dataset, device, config, indices=None):
    """Run inference on all samples, collect predictions and labels."""
    raw_model = model.module if hasattr(model, 'module') else model
    head_names = raw_model.output_layout  # e.g. ['mag', 'loc', 'pga']
    results = defaultdict(list)
    if indices is None:
        indices = range(len(dataset))

    for idx in indices:
        inputs, labels, p_picks = dataset[idx]
        results['event_index'].append(int(idx))

        # Move to device
        inputs_dev = [x.unsqueeze(0).to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
        outputs = model(*inputs_dev)

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

        # Parse outputs using model.output_layout
        for i, name in enumerate(head_names):
            out_np = outputs[i].cpu().numpy().squeeze(0)
            results[f'{name}_pred'].append(out_np)

        # Parse labels (same layout as outputs)
        for i, name in enumerate(head_names):
            lab_np = labels[i].numpy() if isinstance(labels[i], torch.Tensor) else np.array(labels[i])
            results[f'{name}_label'].append(lab_np)

        # Extract mu from MDN (best mixture component by alpha)
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
            elif name == 'pga':
                # shape: (n_pga_targets, n_mixtures, 3)
                alpha = out_np[:, :, 0]
                mu = out_np[:, :, 1]
                best = np.argmax(alpha, axis=1)
                mu_best = mu[np.arange(len(best)), best]
                mu_best = _maybe_unnormalize_pga(name, mu_best, config)
                results[f'{name}_mu_best'].append(mu_best)
            else:
                # shape: (n_mixtures, D) where D=3 for [alpha, mu, sigma] or D=7 for loc
                if out_np.ndim == 2:
                    d = (out_np.shape[1] - 1) // 2
                    alpha = out_np[:, 0]
                    mu = out_np[:, 1:1+d]
                    best = np.argmax(alpha)
                    mu_best = mu[best]
                    results[f'{name}_mu_best'].append(mu_best)
                    if name == 'loc' and raw_model.loc_target_mode == 'rel' and 'loc_center' in results:
                        center = results['loc_center'][-1]
                        results['loc_mu_best_abs'].append(mu_best + center)
                    elif name == 'loc':
                        results['loc_mu_best_abs'].append(mu_best)

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
                    alpha = out_np[:, :, 0]
                    mu = out_np[:, :, 1]
                    best = np.argmax(alpha, axis=1)
                    mu_best = mu[np.arange(len(best)), best]
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


def print_summary(results, split_name):
    """Print summary statistics of predictions vs labels."""
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
            residuals = mu_best.flatten() - labels.flatten()
            print(f'  MAE={np.mean(np.abs(residuals)):.4f}, RMSE={np.sqrt(np.mean(residuals**2)):.4f}')

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
            if 'loc_label_abs' in results:
                print('  absolute-coordinate view:')
                for i in range(min(len(abs_labels), 16)):
                    l = abs_labels[i]
                    p = abs_preds[i]
                    print(f'  [{i:2d}] label_abs=[{l[0]:.2f}, {l[1]:.2f}, {l[2]:.2f}], '
                          f'pred_abs=[{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}]')
                abs_residuals = abs_preds - abs_labels
                print(f'  ABS MAE per dim: {np.mean(np.abs(abs_residuals), axis=0)}')

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
            else:
                print(f'  No valid PGA targets found')


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
    parser.add_argument('--num_shards', type=int, default=1,
                        help='Split each eval dataset into this many deterministic shards.')
    parser.add_argument('--shard_id', type=int, default=0,
                        help='Shard id to evaluate, in [0, num_shards).')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

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

    print('Building datasets...')
    datasets = build_datasets(
        config,
        overfit_n=args.overfit_n,
        input_station_selection=args.input_station_selection,
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

    # Diagnose diting features on first available dataset
    first_dataset = next(iter(datasets.values()))
    diagnose_diting_features(model, first_dataset, device)
    diagnose_amplitude_sensitivity(model, first_dataset, device, config)
    diagnose_embedding_scales(model, first_dataset, device)

    all_results = {}
    case_station_counts = _parse_int_list(args.case_station_counts)
    case_splits = set(_parse_name_list(args.case_splits))
    explicit_case_indices = _parse_int_list(args.case_event_indices)
    for split_name, dataset in datasets.items():
        eval_indices = shard_indices(len(dataset), args.num_shards, args.shard_id)
        shard_text = ''
        if args.num_shards > 1:
            shard_text = f' shard {args.shard_id}/{args.num_shards} ({len(eval_indices)} samples)'
        print(f'\nRunning inference on {split_name} set ({len(dataset)} samples){shard_text}...')
        results = run_inference(model, dataset, device, config, indices=eval_indices)
        print_summary(results, split_name)
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

    np.savez(args.output, **all_results)
    print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    main()
