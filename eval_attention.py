#!/usr/bin/env python3
"""Export PGA target cross-attention for report case studies.

This script is intentionally narrower than eval_checkpoint.py. It evaluates a
small set of events and saves the attention weights and geometry needed to draw
attention maps:

    pga target -> input station attention

For target_cross_attention models, weights are read from
model.pga_cross_attention._last_attention after each forward pass.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'diting'))
sys.path.insert(0, os.path.join(_dir, '..', 'ditingbench'))

from eval_checkpoint import (  # noqa: E402
    _is_point_output,
    _maybe_unnormalize_pga,
    _parse_int_list,
    _parse_name_list,
    _point_mu_from_output,
    _set_random_count_records,
    _dataset_random_count_records,
    _restore_random_count_records,
    _to_numpy,
    build_datasets,
    build_model_and_load,
    select_case_event_indices,
)
from train_light import build_diting_args as load_diting_args  # noqa: E402


def shifted_p_picks_array(p_picks):
    if isinstance(p_picks, dict):
        shifted = p_picks.get('shifted')
        return shifted.numpy() if isinstance(shifted, torch.Tensor) else np.array(shifted)
    return p_picks.numpy() if isinstance(p_picks, torch.Tensor) else np.array(p_picks)


def flatten_results_for_case_selection(raw_results):
    """Convert per-event minimal outputs into select_case_event_indices input."""
    return {
        'event_index': raw_results.get('event_index', []),
        'pga_label': raw_results.get('pga_label', []),
        'pga_mu_best': raw_results.get('pga_mu_best', []),
        'pga_target_valid': raw_results.get('pga_target_valid', []),
        'station_valid_count': raw_results.get('station_valid_count', []),
    }


@torch.no_grad()
def probe_events_for_selection(model, dataset, device, config, max_probe_events):
    """Run light PGA inference over a prefix of the split to select case events."""
    raw_model = model.module if hasattr(model, 'module') else model
    if 'pga' not in raw_model.output_layout:
        return {}
    pga_head_idx = raw_model.output_layout.index('pga')
    n = len(dataset) if max_probe_events <= 0 else min(len(dataset), int(max_probe_events))
    results = defaultdict(list)
    for event_idx in range(n):
        inputs, labels, _ = dataset[event_idx]
        inputs_dev = [x.unsqueeze(0).to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
        outputs = model(*inputs_dev)
        out_np = outputs[pga_head_idx].detach().cpu().numpy().squeeze(0)
        if _is_point_output(out_np):
            mu_best = _point_mu_from_output('pga', out_np)
        else:
            alpha = out_np[:, :, 0]
            mu = out_np[:, :, 1]
            best = np.argmax(alpha, axis=1)
            mu_best = mu[np.arange(len(best)), best]
        mu_best = _maybe_unnormalize_pga('pga', mu_best, config)

        label_np = labels[pga_head_idx].numpy() if isinstance(labels[pga_head_idx], torch.Tensor) else np.array(labels[pga_head_idx])
        valid_np = _to_numpy(inputs[4]).astype(bool) if isinstance(inputs, list) and len(inputs) >= 5 else np.isfinite(label_np.reshape(-1))
        station_valid = _to_numpy(inputs[2]).astype(bool) if isinstance(inputs, list) and len(inputs) >= 3 else np.array([], dtype=bool)

        results['event_index'].append(int(event_idx))
        results['pga_label'].append(label_np)
        results['pga_mu_best'].append(mu_best)
        results['pga_target_valid'].append(valid_np)
        results['station_valid_count'].append(int(station_valid.sum()))
    return dict(results)


def _attention_entropy(attn, valid_station_mask):
    attn = np.asarray(attn, dtype=float)
    valid = np.asarray(valid_station_mask, dtype=bool)
    if attn.ndim != 2 or valid.ndim != 1:
        return np.full(attn.shape[0] if attn.ndim else 0, np.nan)
    masked = attn[:, valid]
    masked = masked / np.clip(masked.sum(axis=1, keepdims=True), 1e-12, None)
    return -(masked.clip(1e-12) * np.log(masked.clip(1e-12))).sum(axis=1)


def _topk_indices(attn, valid_station_mask, k):
    attn = np.asarray(attn, dtype=float)
    valid = np.asarray(valid_station_mask, dtype=bool)
    valid_idx = np.where(valid)[0]
    if attn.ndim != 2 or valid_idx.size == 0:
        return np.full((attn.shape[0] if attn.ndim else 0, int(k)), -1, dtype=np.int64)
    out = np.full((attn.shape[0], int(k)), -1, dtype=np.int64)
    valid_attn = attn[:, valid_idx]
    for i in range(attn.shape[0]):
        order = np.argsort(-valid_attn[i])[:k]
        out[i, :len(order)] = valid_idx[order]
    return out


@torch.no_grad()
def run_attention_export(model, dataset, device, config, event_indices, station_counts=None, seed=1234, topk=5):
    raw_model = model.module if hasattr(model, 'module') else model
    if 'pga' not in raw_model.output_layout:
        raise ValueError('Model has no PGA output head.')
    pga_head_idx = raw_model.output_layout.index('pga')

    records = _dataset_random_count_records(dataset)
    counts = [None] if not station_counts else [int(x) for x in station_counts]
    results = defaultdict(list)
    rng_state = np.random.get_state()

    try:
        for event_idx in event_indices:
            event_seed = int(seed) + int(event_idx) * 1009
            for requested_count in counts:
                if requested_count is not None:
                    _set_random_count_records(records, requested_count)
                np.random.seed(event_seed)

                inputs, labels, p_picks = dataset[event_idx]
                inputs_dev = [x.unsqueeze(0).to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
                outputs = model(*inputs_dev)

                attn = getattr(raw_model.pga_cross_attention, '_last_attention', None)
                if attn is None:
                    raise RuntimeError(
                        'No pga_cross_attention._last_attention found. '
                        'This export requires pga_readout_mode=target_cross_attention.'
                    )
                attn_np = attn.detach().cpu().numpy().squeeze(0)

                out_np = outputs[pga_head_idx].detach().cpu().numpy().squeeze(0)
                if _is_point_output(out_np):
                    mu_best = _point_mu_from_output('pga', out_np)
                else:
                    alpha = out_np[:, :, 0]
                    mu = out_np[:, :, 1]
                    best = np.argmax(alpha, axis=1)
                    mu_best = mu[np.arange(len(best)), best]
                mu_best = _maybe_unnormalize_pga('pga', mu_best, config)

                label_np = labels[pga_head_idx].numpy() if isinstance(labels[pga_head_idx], torch.Tensor) else np.array(labels[pga_head_idx])
                target_valid = _to_numpy(inputs[4]).astype(bool) if len(inputs) >= 5 else np.isfinite(label_np.reshape(-1))
                station_valid = _to_numpy(inputs[2]).astype(bool)
                station_coords = _to_numpy(inputs[1])
                target_coords = _to_numpy(inputs[3]) if len(inputs) >= 4 else np.zeros((0, 3))

                event_id = p_picks.get('event_id', str(event_idx)) if isinstance(p_picks, dict) else str(event_idx)
                selected_input_indices = (
                    _to_numpy(p_picks['selected_input_indices'])
                    if isinstance(p_picks, dict) and 'selected_input_indices' in p_picks
                    else np.arange(station_valid.shape[0], dtype=np.int64)
                )
                loc_label_abs = (
                    _to_numpy(p_picks['loc_target_abs'])
                    if isinstance(p_picks, dict) and 'loc_target_abs' in p_picks
                    else np.zeros((1, station_coords.shape[-1]), dtype=float)
                )

                residual = np.asarray(mu_best).reshape(-1) - np.asarray(label_np).reshape(-1)
                entropy = _attention_entropy(attn_np, station_valid)
                topk_idx = _topk_indices(attn_np, station_valid, topk)

                results['event_index'].append(int(event_idx))
                results['event_id'].append(str(event_id))
                results['requested_station_count'].append(-1 if requested_count is None else int(requested_count))
                results['actual_station_count'].append(int(station_valid.sum()))
                results['pga_attention'].append(attn_np)
                results['pga_label'].append(label_np)
                results['pga_mu_best'].append(mu_best)
                results['pga_residual'].append(residual)
                results['pga_target_valid'].append(target_valid)
                results['pga_target_abs'].append(target_coords)
                results['station_valid'].append(station_valid)
                results['station_coords_abs'].append(station_coords)
                results['selected_input_indices'].append(selected_input_indices)
                results['loc_label_abs'].append(loc_label_abs)
                results['p_picks'].append(shifted_p_picks_array(p_picks))
                results['attention_entropy'].append(entropy)
                results['attention_topk_station_slots'].append(topk_idx)
    finally:
        _restore_random_count_records(records)
        np.random.set_state(rng_state)

    return dict(results)


def parse_event_indices(spec):
    by_split = {}
    if not spec:
        return by_split
    for part in str(spec).split(';'):
        part = part.strip()
        if not part:
            continue
        if ':' in part:
            split, values = part.split(':', 1)
            by_split[split.strip()] = _parse_int_list(values)
        else:
            by_split['*'] = _parse_int_list(part)
    return by_split


def main():
    parser = argparse.ArgumentParser(description='Export PGA cross-attention weights for selected events.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--diting_config', default='./diting/config/conf_reg.yml')
    parser.add_argument('--diting_pretrained', default=None)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='eval_attention.npz')
    parser.add_argument('--overfit_n', type=int, default=0)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--input_station_selection', default='config',
                        choices=['config', 'default', 'random', 'p_pick', 'epidist'])
    parser.add_argument('--splits', default='val', help='Comma-separated splits, e.g. val or train,val.')
    parser.add_argument('--event_indices', default='',
                        help='Either "1,2,3" for all splits or "train:1,2;val:3,4".')
    parser.add_argument('--max_events', type=int, default=3,
                        help='Number of auto-selected events per split when --event_indices is empty.')
    parser.add_argument('--probe_events', type=int, default=0,
                        help='Auto-selection probe prefix size. 0 means probe the full split.')
    parser.add_argument('--station_counts', default='',
                        help='Optional comma-separated requested input-station counts, e.g. 3,5,8,12,16,25.')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--topk', type=int, default=5)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

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
    raw_model = model.module if hasattr(model, 'module') else model
    print(f'pga_readout_mode={getattr(raw_model, "pga_readout_mode", None)}')
    if getattr(raw_model, 'pga_readout_mode', None) != 'target_cross_attention':
        raise ValueError('eval_attention.py currently requires pga_readout_mode=target_cross_attention.')

    print('Building datasets...')
    datasets = build_datasets(
        config,
        overfit_n=args.overfit_n,
        input_station_selection=args.input_station_selection,
    )

    split_names = set(_parse_name_list(args.splits))
    explicit_indices = parse_event_indices(args.event_indices)
    station_counts = _parse_int_list(args.station_counts)

    all_results = {}
    for split_name, dataset in datasets.items():
        if split_name not in split_names:
            continue
        if split_name in explicit_indices:
            event_indices = explicit_indices[split_name]
        elif '*' in explicit_indices:
            event_indices = explicit_indices['*']
        else:
            print(f'Auto-selecting {args.max_events} events for {split_name}...')
            probe = probe_events_for_selection(model, dataset, device, config, args.probe_events)
            event_indices = select_case_event_indices(
                flatten_results_for_case_selection(probe),
                args.max_events,
            )
        event_indices = [idx for idx in event_indices if 0 <= idx < len(dataset)]
        print(f'Running attention export on {split_name}: events={event_indices}, station_counts={station_counts or ["config"]}')
        results = run_attention_export(
            model,
            dataset,
            device,
            config,
            event_indices=event_indices,
            station_counts=station_counts,
            seed=args.seed,
            topk=args.topk,
        )
        for key, value in results.items():
            all_results[f'{split_name}_{key}'] = np.array(value, dtype=object)

    if not all_results:
        raise RuntimeError('No attention results were produced. Check --splits and --event_indices.')

    np.savez(args.output, **all_results)
    print(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()
