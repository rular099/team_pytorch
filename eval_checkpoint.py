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
import gemini_models as models
import gemini_util_light as util
import loader_light as loader

# diting imports are handled via gemini_models.py; not needed here directly


def load_diting_args(diting_config_path, device='cpu'):
    """Load diting config, matching train_light.py logic."""
    import yaml
    from diting.downstream.gemini_utils import get_args as get_args_diting
    diting_args, _ = get_args_diting()
    diting_args.conf_file = diting_config_path
    with open(diting_args.conf_file, 'r') as f:
        diting_conf_data = yaml.safe_load(f)
    vars(diting_args).update(diting_conf_data)
    depth = 24
    if depth % diting_args.num_interactions != 0:
        diting_args.num_interactions -= 1
    n = (depth - 1) // diting_args.num_interactions
    diting_args.interaction_indexes = [[i*n, (i+1)*n] for i in range(diting_args.num_interactions)]
    if diting_args.num_interactions * n != depth:
        diting_args.interaction_indexes.append([diting_args.num_interactions*n, depth])
    diting_args.distributed = False
    diting_args.device = device
    return diting_args


def build_model_and_load(config, diting_args, checkpoint_path, device):
    """Build model and load checkpoint."""
    single_station_model, full_model = models.build_transformer_model(
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
        overwrite_sampling_rate=overwrite_sampling_rate)
        for data_path, g in zip(training_params['data_path'], generator_params)]

    full_data_dev = [loader.load_events(
        data_path, event_metadata_path='test_ev.csv',
        parts=(False, True, False),
        shuffle_train_dev=g.get('shuffle_train_dev', False),
        custom_split=g.get('custom_split', None),
        overwrite_sampling_rate=overwrite_sampling_rate)
        for data_path, g in zip(training_params['data_path'], generator_params)]

    event_metadata_train = [d[0] for d in full_data_train]
    metadata_train = [d[2] for d in full_data_train]
    event_metadata_dev = [d[0] for d in full_data_dev]
    metadata_dev = [d[2] for d in full_data_dev]

    if overfit_n > 0:
        from train_light import subset_events
        event_metadata_train = [subset_events(meta, overfit_n) for meta in event_metadata_train]
        event_metadata_dev = [subset_events(meta, overfit_n) for meta in event_metadata_train]
        generator_params = [copy.deepcopy(g) for g in generator_params]
        for gp in generator_params:
            fixed_cutout = gp.get('cutout_end', gp.get('cutout_start', 0))
            gp['trigger_based'] = False
            gp['disable_station_foreshadowing'] = False
            gp['selection_skew'] = 0
            gp['oversample'] = 1
            gp['magnitude_resampling'] = 1.0
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
            generators.append(util.PreloadedEventGenerator(
                event_metadata=em_list[i], metadata=meta_list[i],
                data_path=training_params['data_path'][i],
                generator_params=generator_params[i],
                coords_target=True, label_smoothing=False, station_blinding=False,
                cutout=cutout, pga_targets=n_pga_targets, max_stations=max_stations,
                sampling_rate=sampling_rate, no_event_token=no_event_token,
                **gp_copy))
        datasets[split_name] = generators[0] if len(generators) == 1 else generators
    return datasets


@torch.no_grad()
def run_inference(model, dataset, device):
    """Run inference on all samples, collect predictions and labels."""
    results = defaultdict(list)

    for idx in range(len(dataset)):
        inputs, labels, p_picks = dataset[idx]

        # Move to device
        inputs_dev = [x.unsqueeze(0).to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
        outputs = model(*inputs_dev)

        # Parse outputs: [magnitude, location, pga]
        head_names = ['magnitude', 'location', 'pga']
        for i, (name, out) in enumerate(zip(head_names, outputs)):
            out_np = out.cpu().numpy().squeeze(0)  # remove batch dim
            # MDN output: (n_mixtures, 3) where last dim = [alpha_logit, mu, sigma]
            # or for PGA: (n_pga_targets, n_mixtures, 3)
            results[f'{name}_pred'].append(out_np)

        # Parse labels
        label_names = ['magnitude', 'location', 'pga']
        for i, (name, lab) in enumerate(zip(label_names, labels)):
            lab_np = lab.numpy() if isinstance(lab, torch.Tensor) else np.array(lab)
            results[f'{name}_label'].append(lab_np)

        # Extract mu from MDN (best mixture component by alpha)
        for i, name in enumerate(head_names):
            out_np = results[f'{name}_pred'][-1]
            if name == 'pga':
                # shape: (n_pga_targets, n_mixtures, 3)
                n_mix = out_np.shape[1]
                alpha = out_np[:, :, 0]
                mu = out_np[:, :, 1]
                best = np.argmax(alpha, axis=1)
                mu_best = mu[np.arange(len(best)), best]
                results[f'{name}_mu_best'].append(mu_best)
            else:
                # shape: (n_mixtures, D) where D=3 for [alpha, mu, sigma] or D=7 for loc
                if out_np.ndim == 2:
                    n_mix = out_np.shape[0]
                    d = (out_np.shape[1] - 1) // 2  # alpha(1) + mu(d) + sigma(d)
                    alpha = out_np[:, 0]
                    mu = out_np[:, 1:1+d]
                    best = np.argmax(alpha)
                    results[f'{name}_mu_best'].append(mu[best])

        results['p_picks'].append(p_picks.numpy() if isinstance(p_picks, torch.Tensor) else np.array(p_picks))

    return dict(results)


def print_summary(results, split_name):
    """Print summary statistics of predictions vs labels."""
    print(f'\n{"="*60}')
    print(f'  {split_name.upper()} set: {len(results["magnitude_label"])} samples')
    print(f'{"="*60}')

    for name in ['magnitude', 'location', 'pga']:
        labels = np.array(results[f'{name}_label'])
        mu_best = results.get(f'{name}_mu_best')
        if mu_best is None:
            continue
        mu_best = np.array(mu_best)

        print(f'\n--- {name} ---')
        if name == 'magnitude':
            # label shape: (N,) scalar
            for i in range(min(len(labels), 16)):
                print(f'  [{i:2d}] label={labels[i]:.3f}, pred={mu_best[i][0] if mu_best[i].ndim > 0 else mu_best[i]:.3f}')
            residuals = mu_best.flatten() - labels.flatten()
            print(f'  MAE={np.mean(np.abs(residuals)):.4f}, RMSE={np.sqrt(np.mean(residuals**2)):.4f}')

        elif name == 'location':
            # label shape: (N, 3) [lat, lon, depth]
            for i in range(min(len(labels), 16)):
                l = labels[i]
                p = mu_best[i]
                print(f'  [{i:2d}] label=[{l[0]:.2f}, {l[1]:.2f}, {l[2]:.2f}], '
                      f'pred=[{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}]')
            residuals = mu_best - labels
            print(f'  MAE per dim: {np.mean(np.abs(residuals), axis=0)}')

        elif name == 'pga':
            # label shape: (N, n_pga_targets, 1), pred shape: (N, n_pga_targets)
            labels_flat = labels.reshape(len(labels), -1)
            for i in range(min(len(labels), 4)):
                l = labels_flat[i]
                p = mu_best[i]
                n_show = min(5, len(l))
                print(f'  [{i:2d}] label={l[:n_show]}, pred={p[:n_show]}')
            residuals = mu_best - labels_flat
            print(f'  MAE={np.mean(np.abs(residuals)):.4f}, RMSE={np.sqrt(np.mean(residuals**2)):.4f}')
            # Per-station correlation
            all_l = labels_flat.flatten()
            all_p = mu_best.flatten()
            if len(all_l) > 1:
                corr = np.corrcoef(all_l, all_p)[0, 1]
                print(f'  Correlation: {corr:.4f}')


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

    device = torch.device(args.device)
    diting_args = load_diting_args(args.diting_config, device=str(device))

    print('Building model...')
    model = build_model_and_load(config, diting_args, args.checkpoint, device)

    print('Building datasets...')
    datasets = build_datasets(config, overfit_n=args.overfit_n)

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
