import numpy as np
import os
import sys
sys.path.insert(0, './diting')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ditingbench'))
import yaml
import random
import h5py
import copy
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import argparse
import json
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import gemini_util_light as util
import loader_light as loader
import gemini_models as models

from dtbench.training.modeling import build_interaction_indexes, parse_hps

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _normalize_optional_path(path):
    if not path:
        return path
    expanded = os.path.expanduser(os.path.expandvars(path))
    if "$" in expanded:
        raise ValueError(
            f"Unresolved environment variable in path: {path}. "
            "Set the required environment variable before running."
        )
    return os.path.abspath(expanded)


def build_diting_args(diting_config_path, device='cpu', distributed=False, pretrained_override=None, resume_override=None):
    """Build diting args from YAML config, replacing the old get_args_diting() approach.

    Uses ditingbench's parse_hps and build_interaction_indexes.
    """
    with open(diting_config_path, 'r') as f:
        conf = yaml.safe_load(f)

    # Defaults matching ditingbench cli.py parser
    defaults = dict(
        base_width=256, target_width=256, model_depth=24,
        in_samples=10000, patch_size=50,
        num_interactions=4, out_channels=256,
        diting_frontend='vit_adapter',
        attn_pool_hidden_dim=None, attn_pool_temperature=0.5, attn_pool_topk=0,
        pale_size=5, stem_convKs=3, cpe_kernel_size=3,
        ffn_convKS=3, fpn_convKS=3, aggregate_convKS=3, head_convKS=3,
        inter_mode='fpn_deep_5',
        add_vit_feature=True, use_extra_extractor=False,
        norm_layer='rmsnorm', xattn=False,
        drop_path=0.0, head_drop_rate=0.0,
        init_std=0.02, input_mult=1.0, attn_mult=256.0, output_mult=1.0,
        eval_type='linear_probe',
        pretrained='', resume='',
        pretrain_method='lp', pretrained_load_mode='backbone',
        hps='',
        downstream_task='emg',
        reuse_ppm=True,
        loss_type='bce',
    )
    defaults.update(conf)
    if pretrained_override is not None:
        defaults['pretrained'] = pretrained_override
    if resume_override is not None:
        defaults['resume'] = resume_override

    diting_args = argparse.Namespace(**defaults)
    diting_args.conf_file = diting_config_path
    diting_args.pretrained = _normalize_optional_path(getattr(diting_args, 'pretrained', ''))
    diting_args.resume = _normalize_optional_path(getattr(diting_args, 'resume', ''))

    # Parse HPS string to extract target_width, input_mult, attn_mult, output_mult
    parse_hps(diting_args)

    # Build interaction indexes
    diting_args.interaction_indexes = build_interaction_indexes(
        diting_args.model_depth, diting_args.num_interactions
    )

    diting_args.distributed = distributed
    diting_args.device = device

    return diting_args


def select_diverse_event_ids(event_metadata, n, mag_key=None):
    if n is None:
        return None
    if not hasattr(event_metadata, "columns"):
        return None
    event_key = loader.detect_event_key(event_metadata.columns)
    unique_events = event_metadata.drop_duplicates(subset=event_key, keep='first').copy()
    event_ids = unique_events[event_key].to_numpy()
    if len(event_ids) <= n:
        return event_ids

    feature_cols = []
    resolved_mag_key = loader.resolve_target_key(unique_events.columns, mag_key)
    if resolved_mag_key in unique_events.columns:
        feature_cols.append(resolved_mag_key)
    for coord_key in util.detect_location_keys(unique_events.columns):
        if coord_key in unique_events.columns and coord_key not in feature_cols:
            feature_cols.append(coord_key)

    if not feature_cols:
        return event_ids[:n]

    feature_df = unique_events[feature_cols].apply(pd.to_numeric, errors='coerce')
    valid_mask = ~feature_df.isna().all(axis=1)
    feature_df = feature_df.loc[valid_mask].copy()
    event_ids = event_ids[valid_mask.to_numpy()]
    if len(event_ids) <= n:
        return event_ids

    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))
    features = feature_df.to_numpy(dtype=float)
    scale = np.nanstd(features, axis=0)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    features = (features - np.nanmean(features, axis=0)) / scale

    selected = []
    for col_idx in range(features.shape[1]):
        selected.append(int(np.argmin(features[:, col_idx])))
        selected.append(int(np.argmax(features[:, col_idx])))
    selected = list(dict.fromkeys(selected))
    if not selected:
        selected = [0]
    if len(selected) > n:
        selected = selected[:n]

    while len(selected) < n:
        remaining = np.array([idx for idx in range(len(event_ids)) if idx not in selected], dtype=int)
        if remaining.size == 0:
            break
        dist = np.linalg.norm(features[remaining, None, :] - features[np.array(selected)][None, :, :], axis=2)
        min_dist = dist.min(axis=1)
        selected.append(int(remaining[np.argmax(min_dist)]))

    return event_ids[np.array(selected[:n], dtype=int)]


def subset_events(event_metadata, n, mag_key=None, selected_event_ids=None):
    """Subset to n events, preferring a wide spread in magnitude and location."""
    if n is None:
        return event_metadata
    if hasattr(event_metadata, "columns"):
        event_key = loader.detect_event_key(event_metadata.columns)
        if selected_event_ids is None:
            selected_event_ids = select_diverse_event_ids(event_metadata, n, mag_key=mag_key)
        if selected_event_ids is None:
            return event_metadata.iloc[:n].copy()
        return event_metadata[event_metadata[event_key].isin(set(selected_event_ids))].copy()
    if hasattr(event_metadata, "iloc"):
        return event_metadata.iloc[:n].copy()
    return event_metadata[:n]


def split_event_metadata_by_selected_ids(event_metadata, selected_event_ids, custom_split=None, shuffle_train_dev=False):
    """Re-split a selected event subset with TrainDevTestSplitter."""
    if not hasattr(event_metadata, "columns"):
        raise ValueError("event_metadata must be a DataFrame when splitting selected event ids")

    event_key = loader.detect_event_key(event_metadata.columns)
    selected_set = set(selected_event_ids)
    selected_event_metadata = event_metadata[event_metadata[event_key].isin(selected_set)].copy()
    unique_selected_events = selected_event_metadata.drop_duplicates(subset=event_key, keep='first').reset_index(drop=True)

    split_parts = {
        'train': (True, False, False),
        'dev': (False, True, False),
        'test': (False, False, True),
    }
    split_event_metadata = {}
    split_event_ids = {}

    for split_name, parts in split_parts.items():
        event_mask = loader.TrainDevTestSplitter.run_method(
            unique_selected_events, custom_split, shuffle_train_dev, parts=parts
        )
        event_ids = unique_selected_events.loc[event_mask, event_key].to_numpy()
        split_event_ids[split_name] = event_ids
        split_event_metadata[split_name] = selected_event_metadata[
            selected_event_metadata[event_key].isin(set(event_ids))
        ].copy()

    return split_event_metadata, split_event_ids


def build_overfit_event_metadata_splits(full_data_all, generator_params, overfit_n):
    """Select a diverse subset, then split that subset into train/dev/test."""
    event_metadata_train = []
    event_metadata_dev = []
    event_metadata_test = []
    selected_event_ids = []

    for all_meta, generator in zip(full_data_all, generator_params):
        all_event_metadata = all_meta[0]
        cur_selected_event_ids = select_diverse_event_ids(
            all_event_metadata, overfit_n, mag_key=generator.get('key', 'MA')
        )
        split_event_metadata, _ = split_event_metadata_by_selected_ids(
            all_event_metadata,
            cur_selected_event_ids,
            custom_split=generator.get('custom_split', None),
            shuffle_train_dev=generator.get('shuffle_train_dev', False),
        )
        selected_event_ids.append(cur_selected_event_ids)
        event_metadata_train.append(split_event_metadata['train'])
        event_metadata_dev.append(split_event_metadata['dev'])
        event_metadata_test.append(split_event_metadata['test'])

    return event_metadata_train, event_metadata_dev, event_metadata_test, selected_event_ids


def count_unique_events(event_metadata):
    if event_metadata is None or len(event_metadata) == 0:
        return 0
    event_key = loader.detect_event_key(event_metadata.columns)
    return len(event_metadata.drop_duplicates(subset=event_key, keep='first'))


def _event_columns_for_export(df):
    preferred = [
        'EVENT', 'Latitude', 'Longitude', 'DEPTH', 'Magnitude',
        'Origin_Time(JST)', 'Source_Mix', 'Sampling_Rate_Hz',
        'Event_Length_Samples', 'Event_Length_Seconds',
    ]
    cols = [c for c in preferred if c in df.columns]
    if not cols:
        event_key = loader.detect_event_key(df.columns)
        cols = [event_key]
    return cols


def export_split_metadata(weight_path, data_paths, train_dfs, dev_dfs, test_dfs, selected_event_ids=None):
    station_frames = []
    selected_event_ids = selected_event_ids or [None] * len(train_dfs)

    for dataset_index, (data_path, train_df, dev_df, test_df, selected_ids) in enumerate(
            zip(data_paths, train_dfs, dev_dfs, test_dfs, selected_event_ids)):
        event_key = loader.detect_event_key(train_df.columns if len(train_df) else (dev_df.columns if len(dev_df) else test_df.columns))
        selected_set = set(map(str, selected_ids)) if selected_ids is not None else set()

        for split_name, split_df in [('train', train_df), ('dev', dev_df), ('test', test_df)]:
            if split_df is None or len(split_df) == 0:
                continue
            df = split_df.copy()
            df[event_key] = df[event_key].astype(str)
            if 'EVENT' in df.columns:
                df['EVENT'] = df['EVENT'].astype(str)
            else:
                df['EVENT'] = df[event_key]
            df['dataset_index'] = dataset_index
            df['source_data_path'] = str(data_path)
            df['split'] = split_name
            df['is_overfit_selected'] = df['EVENT'].isin(selected_set)
            df['is_overfit_train'] = df['is_overfit_selected'] & (df['split'] == 'train')
            df['is_overfit_dev'] = df['is_overfit_selected'] & (df['split'] == 'dev')
            df['is_overfit_test'] = df['is_overfit_selected'] & (df['split'] == 'test')
            station_frames.append(df)

    if not station_frames:
        return

    split_stations = pd.concat(station_frames, ignore_index=True)
    event_cols = _event_columns_for_export(split_stations)
    event_cols = [c for c in ['dataset_index', 'source_data_path', 'split',
                              'is_overfit_selected', 'is_overfit_train',
                              'is_overfit_dev', 'is_overfit_test'] + event_cols
                  if c in split_stations.columns]
    agg_kwargs = {col: (col, 'first') for col in event_cols if col not in ['dataset_index', 'EVENT', 'split']}
    if 'wave_idx' in split_stations.columns:
        agg_kwargs['n_station_rows'] = ('wave_idx', 'nunique')
    else:
        agg_kwargs['n_station_rows'] = ('EVENT', 'size')
    split_events = split_stations.groupby(['dataset_index', 'EVENT', 'split'], as_index=False).agg(**agg_kwargs)

    split_stations.to_csv(os.path.join(weight_path, 'split_stations.csv'), index=False)
    split_events.to_csv(os.path.join(weight_path, 'split_events.csv'), index=False)


def module_grad_norm(module):
    sq_sum = None
    for param in module.parameters():
        if param.grad is None:
            continue
        grad_sq = torch.sum(param.grad.detach() ** 2)
        sq_sum = grad_sq if sq_sum is None else sq_sum + grad_sq
    if sq_sum is None:
        return None
    return torch.sqrt(sq_sum)


def module_grad_rms(module):
    sq_sum = None
    count = 0
    for param in module.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        grad_sq = torch.sum(grad ** 2)
        sq_sum = grad_sq if sq_sum is None else sq_sum + grad_sq
        count += grad.numel()
    if sq_sum is None or count == 0:
        return None
    return torch.sqrt(sq_sum / count)


def global_grad_norm(parameters):
    sq_sum = None
    for param in parameters:
        if param.grad is None:
            continue
        grad_sq = torch.sum(param.grad.detach() ** 2)
        sq_sum = grad_sq if sq_sum is None else sq_sum + grad_sq
    if sq_sum is None:
        return None
    return torch.sqrt(sq_sum)


def collect_pga_output_stats(outputs, labels, output_layout, pga_target_valid):
    if 'pga' not in output_layout:
        return {}
    pga_idx = output_layout.index('pga')
    pga_pred = outputs[pga_idx]
    pga_true = labels[pga_idx]
    alpha_logits = pga_pred[..., 0]
    mu = pga_pred[..., 1]
    sigma = pga_pred[..., 2]
    best_idx = alpha_logits.argmax(dim=-1, keepdim=True)
    mu_best = mu.gather(-1, best_idx).squeeze(-1)

    stats = {
        'diag/pga_alpha_logits_mean': alpha_logits.mean().detach(),
        'diag/pga_alpha_logits_std': alpha_logits.std(unbiased=False).detach(),
        'diag/pga_mu_mean': mu.mean().detach(),
        'diag/pga_mu_std': mu.std(unbiased=False).detach(),
        'diag/pga_sigma_mean': sigma.mean().detach(),
        'diag/pga_sigma_std': sigma.std(unbiased=False).detach(),
        'diag/pga_mu_best_mean': mu_best.mean().detach(),
        'diag/pga_mu_best_std': mu_best.std(unbiased=False).detach(),
    }
    if pga_target_valid is not None:
        mask = pga_target_valid.bool()
        if mask.any():
            valid_pred = mu_best[mask]
            valid_true = pga_true[mask]
            stats.update({
                'diag/pga_target_mean': valid_true.mean().detach(),
                'diag/pga_target_std': valid_true.std(unbiased=False).detach(),
                'diag/pga_pred_target_mean_gap': (valid_pred.mean() - valid_true.mean()).detach(),
                'diag/pga_pred_target_std_gap': (valid_pred.std(unbiased=False) - valid_true.std(unbiased=False)).detach(),
                'diag/pga_valid_target_count': mask.sum().detach().float(),
            })
    return stats


def collect_input_stats(inputs, labels, p_picks):
    if not isinstance(inputs, list):
        return {}

    stats = {}
    waveforms = inputs[0]
    metadata = inputs[1] if len(inputs) > 1 else None
    station_valid = inputs[2].bool() if len(inputs) > 2 else None
    pga_targets = inputs[3] if len(inputs) > 3 else None
    pga_target_valid = inputs[4].bool() if len(inputs) > 4 else None
    event_mag = labels[0] if isinstance(labels, list) and len(labels) > 0 else None
    pga_labels = labels[-1] if isinstance(labels, list) and len(labels) > 0 else None

    abs_wave = waveforms.abs()
    stats.update({
        'data/raw_wave_abs_mean': abs_wave.mean().detach(),
        'data/raw_wave_abs_max': abs_wave.max().detach(),
        'data/raw_wave_std': waveforms.std(unbiased=False).detach(),
        'data/raw_wave_nonzero_ratio': (waveforms != 0).float().mean().detach(),
        'data/raw_wave_nan_count': torch.isnan(waveforms).sum().detach().float(),
        'data/raw_wave_inf_count': torch.isinf(waveforms).sum().detach().float(),
    })

    if station_valid is not None:
        stats.update({
            'data/station_valid_count_mean': station_valid.float().sum(dim=1).mean().detach(),
            'data/station_valid_ratio': station_valid.float().mean().detach(),
        })

    if metadata is not None:
        stats.update({
            'data/coords_abs_mean': metadata.abs().mean().detach(),
            'data/coords_abs_max': metadata.abs().max().detach(),
            'data/coords_nan_count': torch.isnan(metadata).sum().detach().float(),
        })

    if pga_targets is not None:
        stats.update({
            'data/pga_target_abs_mean': pga_targets.abs().mean().detach(),
            'data/pga_target_abs_max': pga_targets.abs().max().detach(),
        })

    if pga_target_valid is not None:
        stats.update({
            'data/pga_target_valid_count_mean': pga_target_valid.float().sum(dim=1).mean().detach(),
            'data/pga_target_valid_ratio': pga_target_valid.float().mean().detach(),
        })

    if event_mag is not None and event_mag.shape[-1] == 1:
        event_mag_vals = event_mag.float().reshape(event_mag.shape[0], -1).mean(dim=1)
        stats.update({
            'data/event_mag_mean': event_mag_vals.mean().detach(),
            'data/event_mag_std': event_mag_vals.std(unbiased=False).detach(),
            'data/event_mag_min': event_mag_vals.min().detach(),
            'data/event_mag_max': event_mag_vals.max().detach(),
        })

    if pga_labels is not None:
        stats.update({
            'data/pga_label_mean': pga_labels.mean().detach(),
            'data/pga_label_std': pga_labels.std(unbiased=False).detach(),
        })

    if p_picks is not None:
        if isinstance(p_picks, dict):
            shifted_p_picks = p_picks['shifted'].to(waveforms.device)
            raw_p_picks = p_picks['raw'].to(waveforms.device)
            shift_values = p_picks['shift'].to(waveforms.device)
        else:
            shifted_p_picks = p_picks.to(waveforms.device)
            raw_p_picks = shifted_p_picks
            shift_values = None

        raw_p_pick_valid = raw_p_picks > 0
        stats['data/raw_p_pick_valid_count_mean'] = raw_p_pick_valid.float().sum(dim=1).mean().detach()
        stats['data/raw_p_pick_missing_ratio'] = (~raw_p_pick_valid).float().mean().detach()
        if raw_p_pick_valid.any():
            raw_valid_p = raw_p_picks[raw_p_pick_valid]
            stats.update({
                'data/raw_p_pick_mean': raw_valid_p.mean().detach(),
                'data/raw_p_pick_min': raw_valid_p.min().detach(),
                'data/raw_p_pick_max': raw_valid_p.max().detach(),
            })

        shifted_p_pick_valid = raw_p_pick_valid
        stats['data/p_pick_valid_count_mean'] = shifted_p_pick_valid.float().sum(dim=1).mean().detach()
        if shifted_p_pick_valid.any():
            shifted_valid_p = shifted_p_picks[shifted_p_pick_valid]
            stats.update({
                'data/p_pick_mean': shifted_valid_p.mean().detach(),
                'data/p_pick_min': shifted_valid_p.min().detach(),
                'data/p_pick_max': shifted_valid_p.max().detach(),
            })

        if shift_values is not None:
            stats.update({
                'data/p_pick_shift_mean': shift_values.mean().detach(),
                'data/p_pick_shift_min': shift_values.min().detach(),
                'data/p_pick_shift_max': shift_values.max().detach(),
            })

    return stats

def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, clipnorm=None, is_dist=False, rank=0, save_name=None,
                res_comps=None, res_weight=None, post_train_sanity=False, epoch_sanity=False, train_sampler=None, lr_monitor='val'):
    tb_path = f'runs/{save_name}'
    eval_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    if (not is_dist) or (is_dist and (rank == 0)):
        os.makedirs(tb_path, exist_ok=True)
        writer = SummaryWriter(log_dir = tb_path)
    try:
        device = next(model.parameters()).device
    except:
        device = 'cpu'
    global_step = 0
    steps_per_epoch = 0
    for epoch in range(num_epochs):
        if is_dist and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        num_train_batches = 0
        first_batch_logged = False
        for inputs, labels, p_picks in train_loader:
            if isinstance(inputs, list):
                inputs, labels = [i.to(device) for i in inputs], [l.to(device) for l in labels]
            else:
                inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            if isinstance(inputs, list):
                outputs = model(*inputs)
            else:
                outputs = model(inputs)
            # Layout when n_pga_targets > 0:
            #   inputs = [waveforms, metadata, station_valid, pga_targets, pga_target_valid, (dataset_id?)]
            pga_target_valid = inputs[4] if (isinstance(inputs, list) and len(inputs) >= 5) else None
            sel_pred, sel_true = models.select_loss_components(
                outputs, labels, eval_model.output_layout, res_comps)
            loss = models.mixture_density_loss_full(sel_pred, sel_true, res_comps=res_comps, res_weight=res_weight,
                                                    pga_target_valid=pga_target_valid)
            loss.backward()
            pre_clip_global_grad = global_grad_norm(model.parameters())
            if (((not is_dist) or (is_dist and (rank == 0))) and not first_batch_logged):
                diag_scalars = {}
                forward_diag = getattr(eval_model, '_last_diag', {})
                for key, value in forward_diag.items():
                    if torch.is_tensor(value):
                        if torch.isnan(value).any():
                            continue
                        diag_scalars[f'diag/{key}'] = value.detach()
                diag_scalars.update(collect_input_stats(inputs, labels, p_picks))
                diag_scalars.update(collect_pga_output_stats(outputs, labels, eval_model.output_layout, pga_target_valid))

                grad_targets = {
                    'grad/station_adapter': module_grad_norm(eval_model.waveform_model[1]),
                    'grad_rms/station_adapter': module_grad_rms(eval_model.waveform_model[1]),
                    'grad/mlp_pga': module_grad_norm(eval_model.mlp_pga),
                    'grad_rms/mlp_pga': module_grad_rms(eval_model.mlp_pga),
                    'grad/output_model_pga': module_grad_norm(eval_model.output_model_pga),
                    'grad_rms/output_model_pga': module_grad_rms(eval_model.output_model_pga),
                }
                if pre_clip_global_grad is not None and not torch.isnan(pre_clip_global_grad).any():
                    diag_scalars['grad/global_pre_clip_norm'] = pre_clip_global_grad.detach()
                for key, value in grad_targets.items():
                    if value is not None and not torch.isnan(value).any():
                        diag_scalars[key] = value.detach()
            if (not is_dist) or (is_dist and (rank == 0)):
                writer.add_scalar('train/loss', loss.item(), global_step)
                step_in_ep = global_step - steps_per_epoch * epoch
                print(f'Step/Epoch {step_in_ep}/{epoch}, Loss: {loss.item():.4f}')
            if clipnorm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clipnorm)
            post_clip_global_grad = global_grad_norm(model.parameters())
            if (((not is_dist) or (is_dist and (rank == 0))) and not first_batch_logged):
                if post_clip_global_grad is not None and not torch.isnan(post_clip_global_grad).any():
                    diag_scalars['grad/global_post_clip_norm'] = post_clip_global_grad.detach()
                for key, value in diag_scalars.items():
                    writer.add_scalar(key, value.item(), epoch)
                first_batch_logged = True
            optimizer.step()
            running_loss += loss.item()
            num_train_batches += 1
            if global_step % 100 == 0:
                if (not is_dist) or (is_dist and (rank == 0)):
                    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
            global_step += 1
        if is_dist:
            train_stats = torch.tensor([running_loss, float(num_train_batches)], device=device)
            dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)
            epoch_loss = (train_stats[0] / train_stats[1]).item()
        else:
            epoch_loss = running_loss / max(num_train_batches, 1)
        if steps_per_epoch == 0:
            steps_per_epoch = global_step
        if (not is_dist) or (is_dist and (rank == 0)):
            writer.add_scalar('train/epoch_loss',epoch_loss, epoch)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # Validation step
        eval_model.eval()
        val_running_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                if isinstance(inputs, list):
                    inputs, labels = [i.to(device) for i in inputs], [l.to(device) for l in labels]
                else:
                    inputs, labels = inputs.to(device), labels.to(device)
                if isinstance(inputs, list):
                    outputs = eval_model(*inputs)
                else:
                    outputs = eval_model(inputs)
                pga_target_valid = inputs[4] if (isinstance(inputs, list) and len(inputs) >= 5) else None
                sel_pred, sel_true = models.select_loss_components(
                    outputs, labels, eval_model.output_layout, res_comps)
                loss = models.mixture_density_loss_full(sel_pred, sel_true, res_comps=res_comps, res_weight=res_weight,
                                                        pga_target_valid=pga_target_valid)
                val_running_loss += loss.item()
                num_val_batches += 1

        if is_dist:
            val_stats = torch.tensor([val_running_loss, float(num_val_batches)], device=device)
            dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)
            val_loss = (val_stats[0] / val_stats[1]).item()
        else:
            val_loss = val_running_loss / max(num_val_batches, 1)

        if (not is_dist) or (is_dist and (rank == 0)):
            writer.add_scalar('val/epoch_loss',val_loss, epoch)
            print(f'Validation Loss: {val_loss:.4f}')

            # Save checkpoint
            filepath = os.path.join(training_params['weight_path'], f'{save_name}_{epoch+1}.pth')
            state_dict = eval_model.state_dict()

            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': val_loss,
                }, filepath)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            monitor_loss = epoch_loss if lr_monitor == 'train' else val_loss
            scheduler.step(monitor_loss)
        else:
            scheduler.step()
        if is_dist:
            dist.barrier()
        if epoch_sanity and ((not is_dist) or (is_dist and (rank == 0))):
            run_sanity_check(eval_model, train_loader, device, name=f'{save_name}_train_epoch_{epoch+1}')
            run_sanity_check(eval_model, val_loader, device, name=f'{save_name}_val_epoch_{epoch+1}')
    if post_train_sanity and ((not is_dist) or (is_dist and (rank == 0))):
        run_sanity_check(eval_model, train_loader, device, name=f'{save_name}_train_post')
        run_sanity_check(eval_model, val_loader, device, name=f'{save_name}_val_post')
    if (not is_dist) or (is_dist and (rank == 0)):
        writer.close()

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device, is_dist=False, rank=0):
    checkpoint = None

    if rank == 0:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if is_dist:
        # 让 rank 0 把 state_dict 广播给所有进程
        dist.barrier()
        buf = [checkpoint]
        dist.broadcast_object_list(buf, src=0)
        checkpoint = buf[0]

    # 兼容 module. 前缀
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith("module.") and not isinstance(model, torch.nn.parallel.DistributedDataParallel):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "", 1)] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint.get('epoch', 0)
    val_loss = checkpoint.get('val_loss', None)

    return model, optimizer, scheduler, start_epoch, val_loss

def transfer_weights(model, weights_path, ensemble_load=False, wait_for_load=False,
                     ens_id=None, sleeptime=600, device="cpu"):
    """
    PyTorch 版本的权重迁移函数
    """
    if ensemble_load:
        weights_path = os.path.join(weights_path, f"{ens_id}")

    # 如果文件还没生成，循环等待
    if wait_for_load:
        while not os.path.exists(weights_path) and not os.path.isdir(weights_path):
            print(f"Path {weights_path} for weight transfer missing. Sleeping {sleeptime}s")
            time.sleep(sleeptime)

    # 如果是目录，取最新 .pth checkpoint
    if os.path.isdir(weights_path):
        pth_files = sorted([x for x in os.listdir(weights_path) if x.endswith('.pth')])
        if not pth_files:
            raise FileNotFoundError(f'No .pth checkpoints found in {weights_path}')
        weights_path = os.path.join(weights_path, pth_files[-1])

    # 加载 checkpoint
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt

    # Unwrap DDP if needed
    raw_model = model.module if hasattr(model, 'module') else model

    # 判断目标模型是不是 borehole (Conv1d 输入通道是否 64)
    conv1d_layer = None
    conv1d_name = None
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Conv1d):
            conv1d_layer = module
            conv1d_name = name + ".weight"
            break

    model_borehole = conv1d_layer.in_channels == 64
    weights_borehole = state_dict[conv1d_name].shape[1] == 64

    # 删除 embedding 层权重（如果存在）
    for key in list(state_dict.keys()):
        if key.startswith("embedding"):
            del state_dict[key]

    # 处理输入通道不一致的情况
    if model_borehole and not weights_borehole:
        # surface -> borehole: 复制 + 平均
        kernel = state_dict[conv1d_name]
        combine_weights = torch.cat([kernel, kernel], dim=1) / 2.0
        state_dict[conv1d_name] = combine_weights
    elif not model_borehole and weights_borehole:
        # borehole -> surface: 截取前 32 通道 + 缩放
        kernel = state_dict[conv1d_name][:, :32, :]
        state_dict[conv1d_name] = kernel * 2.0

    # 加载参数
    missing, unexpected = raw_model.load_state_dict(state_dict, strict=False)
    print(f"Transferred {len(state_dict) - len(missing)} weights, "
          f"Missing: {missing}, Unexpected: {unexpected}")
    return model

def generate_weights_dict(weights, name=None):
    weights_dict = {}
    for key in weights.keys():
        if isinstance(weights[key], h5py.Dataset):
            weights_dict[f'{name}/{key}'] = weights[key].value
        else:
            weights_dict.update(generate_weights_dict(weights[key], key))
    return weights_dict


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = f'{seed}'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_sanity_check(model, data_loader, device, name='sanity', max_batches=1):
    model.eval()
    print(f'===== {name} check =====')
    with torch.no_grad():
        for batch_id, (inputs, labels, _) in enumerate(data_loader):
            if batch_id >= max_batches:
                break

            if isinstance(inputs, list):
                inputs = [i.to(device) for i in inputs]
                labels = [l.to(device) for l in labels]
            else:
                inputs = inputs.to(device)
                labels = labels.to(device)

            if isinstance(inputs, list):
                wave = inputs[0]
                outputs = model(*inputs)
            else:
                wave = inputs
                outputs = model(inputs)

            if wave.ndim == 3:
                amp = torch.log(torch.amax(torch.abs(wave), dim=(1, 2)) + 1e-6)
            elif wave.ndim == 4:
                amp = torch.log(torch.amax(torch.abs(wave), dim=(2, 3)) + 1e-6)
                amp = amp.mean(dim=1)
            else:
                amp = None

            print(f'[{name}] batch={batch_id}')
            if amp is not None:
                print(f'  waveform log-scale: mean={amp.mean().item():.4f}, std={amp.std().item():.4f}, min={amp.min().item():.4f}, max={amp.max().item():.4f}')
                if wave.ndim == 4:
                    station_has_signal = (torch.abs(wave) > 1e-7).any(dim=(2, 3))
                    signal_frac = (torch.abs(wave) > 1e-7).float().mean(dim=(1, 2, 3))
                    print(f'  active stations/sample: {station_has_signal.sum(dim=1).tolist()}')
                    print(f'  nonzero waveform fraction/sample: {[round(float(x), 4) for x in signal_frac]}')

            if isinstance(labels, list):
                for i, lab in enumerate(labels):
                    labf = lab.float()
                    print(f'  label[{i}]: shape={tuple(lab.shape)}, mean={labf.mean().item():.4f}, std={labf.std().item():.4f}, min={labf.min().item():.4f}, max={labf.max().item():.4f}')
            else:
                labf = labels.float()
                print(f'  label: shape={tuple(labels.shape)}, mean={labf.mean().item():.4f}, std={labf.std().item():.4f}, min={labf.min().item():.4f}, max={labf.max().item():.4f}')

            if not isinstance(outputs, list):
                outputs = [outputs]
            for i, out in enumerate(outputs):
                outf = out.float()
                print(f'  output[{i}]: shape={tuple(out.shape)}, mean={outf.mean().item():.4f}, std={outf.std().item():.4f}, min={outf.min().item():.4f}, max={outf.max().item():.4f}')
                if out.ndim >= 3:
                    # Determine d (number of mu/sigma dimensions) from output shape
                    # Layout: [alpha, mu_1..mu_d, sigma_1..sigma_d] → total = 1 + 2d
                    last_dim = out.shape[-1]
                    d = (last_dim - 1) // 2
                    alpha_logits = out[..., 0]
                    print(f'    alpha_logits: mean={alpha_logits.mean().item():.4f}, std={alpha_logits.std().item():.4f}')
                    dim_names = ['lat', 'lon', 'depth'] if d == 3 else [str(j) for j in range(d)]
                    for j in range(d):
                        mu_j = out[..., 1 + j]
                        sigma_j = out[..., 1 + d + j]
                        name_j = dim_names[j] if j < len(dim_names) else str(j)
                        print(f'    mu_{name_j}: mean={mu_j.mean().item():.4f}, std={mu_j.std().item():.4f}')
                        print(f'    sigma_{name_j}: mean={sigma_j.mean().item():.4f}, std={sigma_j.std().item():.4f}')
    model.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--diting_config', type=str, required=True)
    parser.add_argument('--diting_pretrained', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--test_run', action='store_true')  # Test run with less data
    parser.add_argument('--overfit_n', type=int, default=0)  # Select a tiny subset, then re-split it for train/dev/test
    parser.add_argument('--no_multiprocessing', action='store_true')  # Prevents certain deadlocks
    parser.add_argument('--continue_ensemble', action='store_true')  # Continues a stopped ensemble training
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))
    set_seed(config.get('seed', 42))

    is_dist, rank, world_size, local_rank = util.setup_distributed()
    if is_dist:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
# args for diting model
    diting_args = build_diting_args(
        args.diting_config,
        device=device,
        distributed=is_dist,
        pretrained_override=args.diting_pretrained,
    )
 # end diting args

    training_params = config['training_params']
    generator_params = training_params.get('generator_params', [training_params.copy()])

    if (not is_dist) or (is_dist and (rank == 0)):
        os.makedirs(training_params['weight_path'], exist_ok=True)
    if is_dist:
        dist.barrier()
    listdir = os.listdir(training_params['weight_path'])
    if not args.continue_ensemble and listdir:
        if len(listdir) != 1 or listdir[0] != 'config.json':
            raise ValueError(f'Weight path needs to be empty. ({training_params["weight_path"]})')

    with open(os.path.join(training_params['weight_path'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    print('Loading data')
    if args.test_run:
        limit = 300
    else:
        limit = None

    if not isinstance(training_params['data_path'], list):
        training_params['data_path'] = [training_params['data_path']]

    assert len(generator_params) == len(training_params['data_path'])

    overwrite_sampling_rate = training_params.get('overwrite_sampling_rate', None)
    min_stalta_ratio_at_pick = training_params.get('min_stalta_ratio_at_pick', 0.1)

    full_data_train = [loader.load_events(data_path, event_metadata_path='train_ev.csv',limit=limit,
                                          parts=(True, False, False),
                                          shuffle_train_dev=generator.get('shuffle_train_dev', False),
                                          custom_split=generator.get('custom_split', None),
                                          min_mag=generator.get('min_mag', None),
                                          mag_key=generator.get('key', 'MA'),
                                          overwrite_sampling_rate=overwrite_sampling_rate,
                                          decimate_events=generator.get('decimate_events', None),
                                          min_stalta_ratio_at_pick=min_stalta_ratio_at_pick)
                       for data_path, generator in zip(training_params['data_path'], generator_params)]
    full_data_dev = [loader.load_events(data_path, event_metadata_path='test_ev.csv',limit=limit,
                                        parts=(False, True, False),
                                        shuffle_train_dev=generator.get('shuffle_train_dev', False),
                                        custom_split=generator.get('custom_split', None),
                                        min_mag=generator.get('min_mag', None),
                                        mag_key=generator.get('key', 'MA'),
                                        overwrite_sampling_rate=overwrite_sampling_rate,
                                        decimate_events=generator.get('decimate_events', None),
                                        min_stalta_ratio_at_pick=min_stalta_ratio_at_pick)
                     for data_path, generator in zip(training_params['data_path'], generator_params)]
    full_data_test = [loader.load_events(data_path, event_metadata_path='test_ev.csv', limit=limit,
                                         parts=(False, False, True),
                                         shuffle_train_dev=generator.get('shuffle_train_dev', False),
                                         custom_split=generator.get('custom_split', None),
                                         min_mag=generator.get('min_mag', None),
                                         mag_key=generator.get('key', 'MA'),
                                         overwrite_sampling_rate=overwrite_sampling_rate,
                                         decimate_events=generator.get('decimate_events', None),
                                         min_stalta_ratio_at_pick=min_stalta_ratio_at_pick)
                      for data_path, generator in zip(training_params['data_path'], generator_params)]

    event_metadata_train = [d[0] for d in full_data_train]
    metadata_train = [d[2] for d in full_data_train]
    event_metadata_dev = [d[0] for d in full_data_dev]
    metadata_dev = [d[2] for d in full_data_dev]
    event_metadata_test = [d[0] for d in full_data_test]
    metadata_test = [d[2] for d in full_data_test]
    selected_event_ids = [None for _ in generator_params]

    if args.overfit_n > 0:
        full_data_all = [loader.load_events(data_path, event_metadata_path='overfit_ev.csv', limit=limit,
                                            parts=None,
                                            shuffle_train_dev=generator.get('shuffle_train_dev', False),
                                            custom_split=generator.get('custom_split', None),
                                            min_mag=generator.get('min_mag', None),
                                            mag_key=generator.get('key', 'MA'),
                                            overwrite_sampling_rate=overwrite_sampling_rate,
                                            decimate_events=generator.get('decimate_events', None),
                                            min_stalta_ratio_at_pick=min_stalta_ratio_at_pick)
                         for data_path, generator in zip(training_params['data_path'], generator_params)]
        event_metadata_train, event_metadata_dev, event_metadata_test, selected_event_ids = \
            build_overfit_event_metadata_splits(full_data_all, generator_params, args.overfit_n)
        generator_params = [copy.deepcopy(g) for g in generator_params]
        for generator_param in generator_params:
            fixed_cutout = generator_param.get('cutout_end', generator_param.get('cutout_start', 0))
            generator_param['trigger_based'] = False
            generator_param['disable_station_foreshadowing'] = False
            generator_param['shuffle_train_dev'] = False
            generator_param['oversample'] = 1
            generator_param['select_first'] = True
            generator_param['cutout_start'] = fixed_cutout
            generator_param['cutout_end'] = fixed_cutout
        if (not is_dist) or (is_dist and (rank == 0)):
            split_counts = [
                (
                    count_unique_events(df_train),
                    count_unique_events(df_dev),
                    count_unique_events(df_test),
                )
                for df_train, df_dev, df_test in zip(event_metadata_train, event_metadata_dev, event_metadata_test)
            ]
            print(f'Overfit mode enabled: selected {args.overfit_n} diverse events and re-split them for train/dev/test')
            print(f'Overfit split event counts (train/dev/test): {split_counts}')
            print('Overfit mode adjustments: trigger_based disabled, station foreshadowing enabled, oversample=1, fixed cutout, earliest-trigger station/target selection, no train/dev split shuffling')

    if (not is_dist) or (is_dist and (rank == 0)):
        export_split_metadata(training_params['weight_path'],
                              training_params['data_path'],
                              event_metadata_train,
                              event_metadata_dev,
                              event_metadata_test,
                              selected_event_ids=selected_event_ids)
        print(f'Exported split metadata to {training_params["weight_path"]}/split_events.csv and split_stations.csv')

    sampling_rate = metadata_train[0]['sampling_rate']
    assert all(m['sampling_rate'] == sampling_rate for m in metadata_train + metadata_dev + metadata_test)
    overfit_mode = args.overfit_n > 0

    max_stations = config['model_params']['max_stations']

    config['model_params']['n_datasets'] = len(metadata_train)

    ensemble = config.get('ensemble', 1)

    super_config = config.copy()
    super_training_params = training_params.copy()
    super_model_params = config['model_params'].copy()

    for ens_id in range(ensemble):
        if ensemble > 1:
            print(f'Starting ensemble member {ens_id + 1}/{ensemble}')
            set_seed(ens_id)

            config = super_config.copy()
            config['ens_id'] = ens_id
            training_params = super_training_params.copy()
            training_params['weight_path'] = os.path.join(training_params['weight_path'], f'{ens_id}')
            config['training_params'] = training_params
            config['model_params'] = super_model_params.copy()

            if training_params.get('ensemble_rotation', False):
                # Rotated by angles between 0 and pi/4
                config['model_params']['rotation'] = np.pi / 4 * ens_id / (ensemble - 1)

            if args.continue_ensemble and os.path.isdir(training_params['weight_path']):
                # Check if any checkpoint exists in this ensemble member's directory
                pth_files = [x for x in os.listdir(training_params['weight_path']) if x.endswith('.pth')]
                if pth_files:
                    continue
                else:
                    raise ValueError(f'Can not continue unclean ensemble. No .pth checkpoints in {training_params["weight_path"]}')

            if (not is_dist) or (is_dist and (rank == 0)):
                os.makedirs(training_params['weight_path'], exist_ok=True)
            if is_dist:
                dist.barrier()

            with open(os.path.join(training_params['weight_path'], 'config.json'), 'w') as f:
                json.dump(config, f, indent=4)

        print('Building model')
        full_model = models.build_transformer_model(**config['model_params'],
                                                    trace_length=10000,
                                                    diting_args=diting_args)
        full_model.to(device)
        if is_dist:
            full_model = DDP(
                full_model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )

        if 'load_model_path' in training_params:
            print('Loading full model')
            ckpt = torch.load(training_params['load_model_path'], map_location=device)
            state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
            load_target = full_model.module if is_dist else full_model
            load_target.load_state_dict(state_dict)

        if 'transfer_model_path' in training_params:
            print('Transfering model weights')
            ensemble_load = training_params.get('ensemble_load', False)
            wait_for_load = training_params.get('wait_for_load', False)
            transfer_weights(full_model, training_params['transfer_model_path'],
                             ensemble_load=ensemble_load, wait_for_load=wait_for_load, ens_id=ens_id)

        # Freeze the DiTing encoder; keep the TEAM-side station adapter trainable.
        raw_full = full_model.module if is_dist else full_model
        for param in raw_full.waveform_model[0].parameters():
            param.requires_grad = False

        reinit_fpn = training_params.get('reinit_fpn', True)
        if reinit_fpn:
            station_adapter = raw_full.waveform_model[1]
            if hasattr(station_adapter, 'reset_parameters'):
                station_adapter.reset_parameters()
            else:
                for _, module in station_adapter.named_modules():
                    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
                    elif isinstance(module, nn.Linear):
                        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
                    elif hasattr(module, 'weight') and hasattr(module, 'bias') and isinstance(module.weight, nn.Parameter):
                        nn.init.constant_(module.weight, 1)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)

        if rank == 0:
            if reinit_fpn:
                print('Re-initialized station adapter')
            else:
                print('Kept current station adapter initialization')

        no_event_token = config['model_params'].get('no_event_token', False)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, full_model.parameters()), lr=training_params['lr'])

        n_pga_targets = config['model_params'].get('n_pga_targets', 0)

        train_generators = []
        validation_generators = []

        for i, generator_param_set in enumerate(generator_params):
            noise_seconds = generator_param_set.get('noise_seconds', 5)
            cutout = (
                int(round(sampling_rate * (noise_seconds + generator_param_set['cutout_start']))),
                int(round(sampling_rate * (noise_seconds + generator_param_set['cutout_end']))))

            generator_param_set['transform_target_only'] = generator_param_set.get('transform_target_only', True)

            # Defaults for keys that are computed here or have non-trivial
            # logic; config values override these if present.
            defaults = dict(
                coords_target=True,
                label_smoothing=False if overfit_mode else True,
                station_blinding=False,
                cutout=cutout,
                pga_targets=n_pga_targets,
                max_stations=max_stations,
                sampling_rate=sampling_rate,
                no_event_token=no_event_token,
                use_coords_rel=config['model_params'].get('use_coords_rel', False),
                use_coords_abs=config['model_params'].get('use_coords_abs', True),
                use_coords_rel_abs_fusion=config['model_params'].get('use_coords_rel_abs_fusion', False),
            )
            # Config wins: overlay generator_param_set on top of defaults.
            merged = {**defaults, **generator_param_set}

            train_generators += [util.PreloadedEventGenerator(event_metadata=event_metadata_train[i],
                                                              metadata=metadata_train[i],
                                                              data_path=training_params['data_path'][i],
                                                              generator_params=generator_params[i],
                                                              **merged)]

            old_oversample = generator_param_set.get('oversample', 1)
            generator_param_set['oversample'] = 1 if overfit_mode else 4
            merged_val = {**defaults, **generator_param_set}
            validation_generators += [util.PreloadedEventGenerator(event_metadata=event_metadata_dev[i],
                                                                   metadata=metadata_dev[i],
                                                                   data_path=training_params['data_path'][i],
                                                                   generator_params=generator_params[i],
                                                                   **merged_val)]
            generator_param_set['oversample'] = old_oversample

        if len(train_generators) == 1:
            train_dataset = train_generators[0]
            val_dataset = validation_generators[0]
        else:
            dataset_bias = config['model_params'].get('dataset_bias', False)
            train_dataset = util.JointGenerator(train_generators, shuffle=True, dataset_id=dataset_bias)
            val_dataset = util.JointGenerator(validation_generators, shuffle=True, dataset_id=dataset_bias)

        patience = training_params.get('lr_decay_patience', 6)
#        lr_decay = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=patience, factor=0.3, verbose=1) # need modify
        lr_decay = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=patience, verbose=1)
        logdir = os.path.join('logs/scalars/', training_params['weight_path'])

        if is_dist:
            train_sampler = DistributedSampler(train_dataset)
            train_loader = DataLoader(train_dataset, batch_size=generator_params[0]['batch_size'], sampler=train_sampler, shuffle=(train_sampler is None))
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None
            train_loader = DataLoader(train_dataset, batch_size=generator_params[0]['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=generator_params[0]['batch_size'], sampler=val_sampler, shuffle=(val_sampler is None))
        if ((not is_dist) or local_rank == 0):
            # Save initial checkpoint before training
            init_ckpt_path = os.path.join(training_params['weight_path'], 'full_model_init.pth')
            eval_model = full_model.module if is_dist else full_model
            torch.save({
                'epoch': 0,
                'model_state_dict': eval_model.state_dict(),
            }, init_ckpt_path)
            print(f'Saved initial checkpoint to {init_ckpt_path}')
            run_sanity_check(full_model, train_loader, device, name='full_model_train_pre')
        # Task enable switches: set training_params['res_comps'] in the JSON
        # config to e.g. ["pga"] to train PGA only, ["mag","pga"] for mag+PGA,
        # etc. Default trains all three tasks. res_weight is parallel to
        # res_comps and is normalized inside the loss.
        res_comps_cfg = training_params.get('res_comps', ['mag', 'loc', 'pga'])
        res_weight_cfg = np.asarray(
            training_params.get('res_weight', [1.0] * len(res_comps_cfg)),
            dtype=float,
        )
        assert len(res_comps_cfg) == len(res_weight_cfg), \
            f'res_comps ({len(res_comps_cfg)}) and res_weight ({len(res_weight_cfg)}) length mismatch'
        lr_monitor = training_params.get('lr_monitor', 'val')
        if (not is_dist) or local_rank == 0:
            print(f'[tasks] training on {res_comps_cfg} with weights {res_weight_cfg.tolist()}')
            print(f'[lr] ReduceLROnPlateau monitors {lr_monitor} loss')
        train_model(
            full_model,
            train_loader,
            val_loader,
            optimizer,
            lr_decay,
            num_epochs=training_params['epochs_full_model'],
            clipnorm=training_params.get('clipnorm', None),
            is_dist=is_dist,
            rank=local_rank,
            save_name='full_model',
            res_comps=res_comps_cfg,
            res_weight=res_weight_cfg,
            post_train_sanity=True,
            epoch_sanity=training_params.get('epoch_sanity', False),
            train_sampler=train_sampler,
            lr_monitor=lr_monitor,
        )

#        hist = full_model.fit_generator(generator=train_generator,
#                                        validation_data=validation_generator,
#                                        epochs=training_params['epochs_full_model'],
#                                        use_multiprocessing=use_multiprocessing,
#                                        workers=workers,
#                                        callbacks=callbacks)

#        pickle.dump(hist.history, open(os.path.join(training_params['weight_path'], 'hist.pkl'), 'wb')) change to tensorboard
    if is_dist:
        dist.destroy_process_group()
