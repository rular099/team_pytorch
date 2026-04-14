"""
Evaluate a trained TEAM checkpoint on train/val sets.

Usage:
    python eval_checkpoint.py --config pga_configs/transformer_japan_overfit.json \
        --diting_config ./diting/config/conf_reg.yml \
        --checkpoint weights_japan_overfit/full_model_200.pth \
        --output eval_results.npz \
        [--overfit_n 16] [--device cuda:0]
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

# Reuse the same build_diting_args from train_light.py
from train_light import build_diting_args as load_diting_args


def shifted_p_picks_array(p_picks):
    if isinstance(p_picks, dict):
        shifted = p_picks.get('shifted')
        return shifted.numpy() if isinstance(shifted, torch.Tensor) else np.array(shifted)
    return p_picks.numpy() if isinstance(p_picks, torch.Tensor) else np.array(p_picks)


def build_model_and_load(config, diting_args, checkpoint_path, device):
    """Build model and load checkpoint."""
    full_model = models.build_transformer_model(
        **config['model_params'], trace_length=10000, diting_args=diting_args)
    full_model = full_model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    # Handle DDP module. prefix
    if list(state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict
        state_dict = OrderedDict((k.replace('module.', '', 1), v) for k, v in state_dict.items())
    full_model.load_state_dict(state_dict)
    full_model.eval()

    epoch = checkpoint.get('epoch', '?')
    print(f'Loaded checkpoint: {checkpoint_path} (epoch {epoch})')
    return full_model


def build_datasets(config, overfit_n=0):
    """Build train and val datasets, matching train_light.py logic."""
    training_params = config['training_params']
    generator_params = training_params.get('generator_params', [training_params.copy()])

    overwrite_sampling_rate = training_params.get('overwrite_sampling_rate', None)

    full_data_train = [loader.load_events(
        data_path, event_metadata_path='train_ev.csv',
        parts=(True, False, False),
        shuffle_train_dev=g.get('shuffle_train_dev', False),
        custom_split=g.get('custom_split', None),
        min_mag=g.get('min_mag', None),
        mag_key=g.get('key', 'MA'),
        overwrite_sampling_rate=overwrite_sampling_rate,
        decimate_events=g.get('decimate_events', None))
        for data_path, g in zip(training_params['data_path'], generator_params)]

    full_data_dev = [loader.load_events(
        data_path, event_metadata_path='test_ev.csv',
        parts=(False, True, False),
        shuffle_train_dev=g.get('shuffle_train_dev', False),
        custom_split=g.get('custom_split', None),
        min_mag=g.get('min_mag', None),
        mag_key=g.get('key', 'MA'),
        overwrite_sampling_rate=overwrite_sampling_rate,
        decimate_events=g.get('decimate_events', None))
        for data_path, g in zip(training_params['data_path'], generator_params)]

    event_metadata_train = [d[0] for d in full_data_train]
    metadata_train = [d[2] for d in full_data_train]
    event_metadata_dev = [d[0] for d in full_data_dev]
    metadata_dev = [d[2] for d in full_data_dev]

    if overfit_n > 0:
        def subset_by_events(meta, n):
            for k in ['KiK_File', '#EventID', 'EVENT']:
                if k in meta.columns:
                    break
            unique_events = meta[k].unique()[:n]
            return meta[meta[k].isin(unique_events)].copy()
        event_metadata_train = [subset_by_events(meta, overfit_n) for meta in event_metadata_train]
        event_metadata_dev = [subset_by_events(meta, overfit_n) for meta in event_metadata_dev]
        generator_params = [copy.deepcopy(g) for g in generator_params]
        for gp in generator_params:
            fixed_cutout = gp.get('cutout_end', gp.get('cutout_start', 0))
            gp['trigger_based'] = False
            gp['disable_station_foreshadowing'] = False
            gp['shuffle_train_dev'] = False
            gp['selection_skew'] = None
            gp['oversample'] = 1
            gp['magnitude_resampling'] = 1.0
            gp['select_first'] = True
            gp['cutout_start'] = fixed_cutout
            gp['cutout_end'] = fixed_cutout

    sampling_rate = metadata_train[0]['sampling_rate']
    max_stations = config['model_params']['max_stations']
    n_pga_targets = config['model_params'].get('n_pga_targets', 0)
    no_event_token = config['model_params'].get('no_event_token', False)

    datasets = {}
    for split_name, em_list, meta_list in [('train', event_metadata_train, metadata_train),
                                            ('val', event_metadata_dev, metadata_dev)]:
        generators = []
        for i, gp in enumerate(generator_params):
            noise_seconds = gp.get('noise_seconds', 5)
            cutout = (sampling_rate * (noise_seconds + gp['cutout_start']),
                      sampling_rate * (noise_seconds + gp['cutout_end']))
            gp_copy = copy.deepcopy(gp)
            gp_copy['transform_target_only'] = gp_copy.get('transform_target_only', True)
            gp_copy['oversample'] = 1  # no oversampling for eval
            defaults = dict(
                coords_target=True, label_smoothing=False, station_blinding=False,
                cutout=cutout, pga_targets=n_pga_targets, max_stations=max_stations,
                sampling_rate=sampling_rate, no_event_token=no_event_token,
                shuffle=False,  # deterministic eval order
            )
            merged = {**defaults, **gp_copy}
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
def diagnose_amplitude_sensitivity(model, dataset, device, scales=(0.5, 1.0, 2.0)):
    raw_model = model.module if hasattr(model, 'module') else model
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
        if raw_model.waveform_scale_proj is not None:
            scale_emb = raw_model.waveform_scale_proj(raw_model._extract_scale(scaled_waveform))
            scale_valid = scale_emb[0, valid_idx]
            scale_norm = scale_valid.norm(dim=-1).mean().item()
            ratio = scale_norm / max(trunk_norm, 1e-8)
        else:
            scale_emb = None

        print(f'\n--- waveform x{scale:.1f} ---')
        print(f'  valid stations: {len(valid_idx)}')
        print(f'  mean ||waveforms_emb||: {trunk_norm:.4f}')
        print(f'  mean ||waveform_scale_proj(scale)||: {scale_norm:.4f}')
        print(f'  scale/trunk norm ratio: {ratio:.4f}')

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
            alpha = pga_out[:, :, 0]
            mu = pga_out[:, :, 1]
            best = np.argmax(alpha, axis=1)
            mu_best = mu[np.arange(len(best)), best]
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
    coords_masked = metadata_inp * station_valid[:, :, None].float()

    waveforms_emb = torch.stack(
        [raw_model.waveform_model(waveforms_masked[:, i, :, :]) for i in range(waveforms_masked.shape[1])],
        dim=1
    )
    waveforms_emb_valid = waveforms_emb[0, valid_idx]

    scale_emb = None
    scale_emb_valid = None
    if raw_model.waveform_scale_proj is not None:
        scale_emb = raw_model.waveform_scale_proj(raw_model._extract_scale(waveform_inp))
        scale_emb_valid = scale_emb[0, valid_idx]

    wave_plus_scale = waveforms_emb
    if scale_emb is not None:
        wave_plus_scale = wave_plus_scale + raw_model.waveform_scale_gain * scale_emb
    wave_plus_scale = raw_model.layernorm(wave_plus_scale)
    wave_plus_scale_valid = wave_plus_scale[0, valid_idx]

    coords_emb = raw_model.position_embedding(coords_masked)
    coords_emb_valid = coords_emb[0, valid_idx]

    station_emb = wave_plus_scale + coords_emb
    station_emb_valid = station_emb[0, valid_idx]

    def mean_norm(x):
        return x.norm(dim=-1).mean().item()

    print(f'{"="*60}')
    print('  Embedding scale diagnostics (1 sample)')
    print(f'{"="*60}')
    print(f'  valid stations: {len(valid_idx)}')
    print(f'  mean ||waveforms_emb||: {mean_norm(waveforms_emb_valid):.4f}')
    if scale_emb_valid is not None:
        print(f'  mean ||scale_emb||: {mean_norm(scale_emb_valid):.4f}')
        print(f'  waveform_scale_gain: {raw_model.waveform_scale_gain:.4f}')
        print(f'  mean ||gain*scale_emb||: {(raw_model.waveform_scale_gain * scale_emb_valid.norm(dim=-1)).mean().item():.4f}')
    print(f'  mean ||layernorm(wave+scale)||: {mean_norm(wave_plus_scale_valid):.4f}')
    print(f'  mean ||coords_emb||: {mean_norm(coords_emb_valid):.4f}')
    print(f'  mean ||station_emb before transformer||: {mean_norm(station_emb_valid):.4f}')
    print()


@torch.no_grad()
def run_inference(model, dataset, device):
    """Run inference on all samples, collect predictions and labels."""
    raw_model = model.module if hasattr(model, 'module') else model
    head_names = raw_model.output_layout  # e.g. ['mag', 'loc', 'pga']
    results = defaultdict(list)

    for idx in range(len(dataset)):
        inputs, labels, p_picks = dataset[idx]

        # Move to device
        inputs_dev = [x.unsqueeze(0).to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
        outputs = model(*inputs_dev)

        # Save pga_target_valid if present
        if isinstance(inputs, list) and len(inputs) >= 5:
            ptv = inputs[4]
            ptv_np = ptv.numpy() if isinstance(ptv, torch.Tensor) else np.array(ptv)
            results['pga_target_valid'].append(ptv_np)

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
            if name == 'pga':
                # shape: (n_pga_targets, n_mixtures, 3)
                alpha = out_np[:, :, 0]
                mu = out_np[:, :, 1]
                best = np.argmax(alpha, axis=1)
                mu_best = mu[np.arange(len(best)), best]
                results[f'{name}_mu_best'].append(mu_best)
            else:
                # shape: (n_mixtures, D) where D=3 for [alpha, mu, sigma] or D=7 for loc
                if out_np.ndim == 2:
                    d = (out_np.shape[1] - 1) // 2
                    alpha = out_np[:, 0]
                    mu = out_np[:, 1:1+d]
                    best = np.argmax(alpha)
                    results[f'{name}_mu_best'].append(mu[best])

        results['p_picks'].append(shifted_p_picks_array(p_picks))

    return dict(results)


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
            for i in range(min(len(labels), 16)):
                l = labels[i]
                p = mu_best[i]
                print(f'  [{i:2d}] label=[{l[0]:.2f}, {l[1]:.2f}, {l[2]:.2f}], '
                      f'pred=[{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}]')
            residuals = mu_best - labels
            print(f'  MAE per dim: {np.mean(np.abs(residuals), axis=0)}')

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
            else:
                print(f'  No valid PGA targets found')


def main():
    parser = argparse.ArgumentParser(description='Evaluate TEAM checkpoint')
    parser.add_argument('--config', required=True)
    parser.add_argument('--diting_config', default='./diting/config/conf_reg.yml')
    parser.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint')
    parser.add_argument('--output', default='eval_results.npz', help='Output file for results')
    parser.add_argument('--overfit_n', type=int, default=0)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
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
    diting_args = load_diting_args(args.diting_config, device=str(device))

    print('Building model...')
    model = build_model_and_load(config, diting_args, args.checkpoint, device)

    print('Building datasets...')
    datasets = build_datasets(config, overfit_n=args.overfit_n)

    # Diagnose diting features on first available dataset
    first_dataset = next(iter(datasets.values()))
    diagnose_diting_features(model, first_dataset, device)
    diagnose_amplitude_sensitivity(model, first_dataset, device)
    diagnose_embedding_scales(model, first_dataset, device)

    all_results = {}
    for split_name, dataset in datasets.items():
        print(f'\nRunning inference on {split_name} set ({len(dataset)} samples)...')
        results = run_inference(model, dataset, device)
        print_summary(results, split_name)
        # Prefix keys with split name for saving
        for k, v in results.items():
            all_results[f'{split_name}_{k}'] = np.array(v, dtype=object)

    np.savez(args.output, **all_results)
    print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    main()
