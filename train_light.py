import numpy as np
import os
import sys
from collections import defaultdict
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
import math
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler, ConcatDataset
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


def _iter_trainable_params(module):
    for param in module.parameters():
        if param.requires_grad:
            yield param


def build_optimizer_with_groups(model, training_params, is_dist=False):
    raw_model = model.module if is_dist else model

    base_lr = training_params['lr']
    adapter_lr = training_params.get('lr_adapter', base_lr)
    team_lr = training_params.get('lr_team', base_lr)
    encoder_lr = training_params.get('lr_encoder', None)

    encoder_params = list(_iter_trainable_params(raw_model.waveform_model[0]))
    adapter_params = list(_iter_trainable_params(raw_model.waveform_model[1]))
    adapter_param_ids = {id(p) for p in adapter_params}
    encoder_param_ids = {id(p) for p in encoder_params}

    team_params = []
    for param in _iter_trainable_params(raw_model):
        pid = id(param)
        if pid in adapter_param_ids or pid in encoder_param_ids:
            continue
        team_params.append(param)

    param_groups = []
    if adapter_params:
        param_groups.append({
            'params': adapter_params,
            'lr': adapter_lr,
            'name': 'adapter',
        })
    if team_params:
        param_groups.append({
            'params': team_params,
            'lr': team_lr,
            'name': 'team',
        })
    if encoder_params:
        if encoder_lr is None:
            raise ValueError(
                'Found trainable encoder parameters, but training_params.lr_encoder is not set.'
            )
        param_groups.append({
            'params': encoder_params,
            'lr': encoder_lr,
            'name': 'encoder',
        })

    if not param_groups:
        raise ValueError('No trainable parameters found for optimizer.')

    optimizer = optim.Adam(param_groups)
    group_summary = {
        group['name']: {
            'lr': group['lr'],
            'n_params': sum(p.numel() for p in group['params']),
        }
        for group in param_groups
    }
    return optimizer, group_summary


class SingleStationTaskDataset(torch.utils.data.Dataset):
    """Flatten event samples to one randomly selected valid station per item."""

    def __init__(self, event_dataset, tasks=('mag', 'epidist', 'pga'), samples_per_event=1,
                 station_sampling='random'):
        self.event_dataset = event_dataset
        self.tasks = tuple(tasks)
        self.samples_per_event = max(1, int(samples_per_event))
        self.station_sampling = station_sampling

    def __len__(self):
        return len(self.event_dataset) * self.samples_per_event

    def __getitem__(self, index):
        n_events = len(self.event_dataset)
        start = (index // self.samples_per_event) % n_events
        for offset in range(n_events):
            event_index = (start + offset) % n_events
            inputs, labels, info = self.event_dataset[event_index]
            sample = self._make_station_sample(inputs, labels, info)
            if sample is not None:
                return sample
        raise RuntimeError('No valid single-station sample found in dataset.')

    def _select_station(self, valid):
        active = torch.nonzero(valid, as_tuple=False).flatten()
        if active.numel() == 0:
            return None
        if self.station_sampling == 'first':
            return active[0]
        pick = np.random.randint(0, active.numel())
        return active[pick]

    @staticmethod
    def _epicentral_distance_km(station_coords, event_coords, scale_metadata):
        delta_xy = station_coords[:2].float() - event_coords[:2].float()
        if scale_metadata:
            # location_transformation stores horizontal coordinates as km / 100.
            delta_xy = delta_xy * 100.0
        else:
            # Matches the existing flat-earth conversion used by TEAM.
            delta_xy = delta_xy * util.D2KM
        return torch.linalg.norm(delta_xy).clamp_min(0.0)

    def _make_station_sample(self, inputs, labels, info):
        waveforms, metadata, station_valid = inputs[:3]
        valid = station_valid.bool().clone()
        if 'pga' in self.tasks:
            input_pga_valid = info.get('input_pga_valid')
            if input_pga_valid is None:
                raise KeyError(
                    'Single-station pga task requires PreloadedEventGenerator '
                    'to return input_pga_valid.'
                )
            valid &= input_pga_valid.bool()
        station_idx = self._select_station(valid)
        if station_idx is None:
            return None

        targets = {}
        if 'mag' in self.tasks:
            targets['mag'] = labels[0].float().reshape(-1)[0]
        if 'epidist' in self.tasks:
            event_coords = info.get('loc_target_abs')
            if event_coords is None:
                raise KeyError('Single-station epidist task requires loc_target_abs in p_pick_info.')
            dist_km = self._epicentral_distance_km(
                metadata[station_idx],
                event_coords,
                scale_metadata=getattr(self.event_dataset, 'scale_metadata', False),
            )
            targets['epidist'] = torch.log1p(dist_km)
            targets['epidist_km'] = dist_km
        if 'pga' in self.tasks:
            targets['pga'] = info['input_pga_values'][station_idx].float()

        sample_info = {
            'event_id': info.get('event_id', ''),
            'station_slot': station_idx.long(),
            'selected_input_index': info['selected_input_indices'][station_idx].long(),
        }
        return waveforms[station_idx].float(), targets, sample_info


CHECKPOINT_ENCODER_PREFIXES = ('waveform_model.0.',)


def clean_state_dict_keys(state_dict):
    clean = {}
    for key, value in state_dict.items():
        clean[key.replace('module.', '', 1) if key.startswith('module.') else key] = value
    return clean


def _state_dict_from_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    return clean_state_dict_keys(state_dict)


def _checkpoint_config(training_params=None):
    cfg = (training_params or {}).get('checkpoint', {}) or {}
    return {
        'exclude_frozen_encoder': bool(cfg.get('exclude_frozen_encoder', True)),
        'save_optimizer_state': bool(cfg.get('save_optimizer_state', True)),
        'excluded_prefixes': tuple(cfg.get('excluded_prefixes', CHECKPOINT_ENCODER_PREFIXES)),
        'encoder_source': cfg.get('encoder_source', None),
    }


def _encoder_is_frozen(raw_model, excluded_prefixes):
    if not excluded_prefixes:
        return False
    state_keys = raw_model.state_dict().keys()
    if not any(any(key.startswith(prefix) for prefix in excluded_prefixes) for key in state_keys):
        return False
    encoder = getattr(raw_model, 'waveform_model', None)
    if encoder is None:
        return False
    try:
        encoder_module = encoder[0]
    except (TypeError, IndexError, KeyError):
        return False
    return all(not param.requires_grad for param in encoder_module.parameters())


def _checkpoint_state_dict(raw_model, training_params=None):
    cfg = _checkpoint_config(training_params)
    state_dict = raw_model.state_dict()
    excluded_prefixes = cfg['excluded_prefixes']
    exclude_encoder = (
        cfg['exclude_frozen_encoder']
        and _encoder_is_frozen(raw_model, excluded_prefixes)
    )
    if not exclude_encoder:
        return state_dict, (), 0, len(state_dict)

    filtered = {
        key: value
        for key, value in state_dict.items()
        if not any(key.startswith(prefix) for prefix in excluded_prefixes)
    }
    return filtered, excluded_prefixes, len(state_dict) - len(filtered), len(state_dict)


def save_model_checkpoint(path, model, epoch, training_params=None, optimizer=None,
                          scheduler=None, loss=None, extra=None):
    raw_model = model.module if hasattr(model, 'module') else model
    state_dict, excluded_prefixes, excluded_count, total_count = _checkpoint_state_dict(
        raw_model, training_params=training_params
    )
    cfg = _checkpoint_config(training_params)
    payload = {
        'epoch': epoch,
        'model_state_dict': state_dict,
        'checkpoint_format': 'non_encoder_v1' if excluded_prefixes else 'full_v1',
        'excluded_prefixes': list(excluded_prefixes),
        'excluded_tensor_count': excluded_count,
        'saved_tensor_count': len(state_dict),
        'total_tensor_count': total_count,
    }
    if cfg.get('encoder_source'):
        payload['encoder_source'] = cfg['encoder_source']
    if loss is not None:
        payload['loss'] = loss
    if cfg['save_optimizer_state'] and optimizer is not None:
        payload['optimizer_state_dict'] = optimizer.state_dict()
    if cfg['save_optimizer_state'] and scheduler is not None:
        payload['scheduler_state_dict'] = scheduler.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    if excluded_count:
        print(
            f'Saved non-encoder checkpoint to {path} '
            f'({len(state_dict)}/{total_count} tensors; excluded {excluded_count} frozen encoder tensors)'
        )
    else:
        print(f'Saved full checkpoint to {path} ({len(state_dict)} tensors)')
    return payload


def load_model_state_dict_compatible(model, state_dict, strict=True, context='checkpoint',
                                     allowed_missing_prefixes=CHECKPOINT_ENCODER_PREFIXES):
    raw_model = model.module if hasattr(model, 'module') else model
    state_dict = clean_state_dict_keys(state_dict)
    missing, unexpected = raw_model.load_state_dict(state_dict, strict=False)
    disallowed_missing = [
        key for key in missing
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if strict and (disallowed_missing or unexpected):
        raise RuntimeError(
            f'Failed to load {context}: '
            f'disallowed missing tensors={disallowed_missing}, unexpected tensors={unexpected}'
        )
    if missing or unexpected:
        print(
            f'Loaded {context} with partial state: '
            f'missing={len(missing)}, unexpected={len(unexpected)}'
        )
    return missing, unexpected


def load_station_pretrain_weights(full_model, weights_path, device='cpu',
                                  load_encoder=False, load_scale=True, load_layernorm=True):
    """Load representation weights from a single-station pretrain checkpoint."""
    raw_model = full_model.module if hasattr(full_model, 'module') else full_model
    state_dict = _state_dict_from_checkpoint(weights_path, device)
    prefixes = ['waveform_model.1.']
    if load_encoder:
        prefixes.append('waveform_model.0.')
    if load_scale:
        prefixes.append('waveform_scale_proj.')
    exact = {'waveform_scale_gate'} if load_scale else set()
    if load_layernorm:
        prefixes.append('layernorm.')

    filtered = {
        key: value
        for key, value in state_dict.items()
        if key in exact or any(key.startswith(prefix) for prefix in prefixes)
    }
    if not filtered:
        raise ValueError(f'No station pretrain weights matched expected prefixes in {weights_path}')
    missing, unexpected = raw_model.load_state_dict(filtered, strict=False)
    loaded_keys = len(filtered)
    print(
        f'Loaded {loaded_keys} station-pretrain tensors from {weights_path}. '
        f'Missing after partial load: {len(missing)}, unexpected: {len(unexpected)}'
    )
    return filtered


def configure_single_station_trainability(model, pretrain_params, rank=0):
    raw_model = model.module if hasattr(model, 'module') else model
    freeze_encoder = pretrain_params.get('freeze_encoder', True)
    for param in raw_model.waveform_model[0].parameters():
        param.requires_grad = not freeze_encoder

    if pretrain_params.get('reinit_adapter', True):
        station_adapter = raw_model.waveform_model[1]
        if hasattr(station_adapter, 'reset_parameters'):
            station_adapter.reset_parameters()
        if rank == 0:
            print('Re-initialized station adapter for single-station pretraining')


def build_single_station_optimizer(model, training_params, pretrain_params, is_dist=False):
    raw_model = model.module if is_dist else model
    base_lr = pretrain_params.get('lr', training_params['lr'])
    adapter_lr = pretrain_params.get('lr_adapter', training_params.get('lr_adapter', base_lr))
    head_lr = pretrain_params.get('lr_head', pretrain_params.get('lr_team', training_params.get('lr_team', base_lr)))
    encoder_lr = pretrain_params.get('lr_encoder', training_params.get('lr_encoder', None))

    encoder_params = list(_iter_trainable_params(raw_model.waveform_model[0]))
    adapter_params = list(_iter_trainable_params(raw_model.waveform_model[1]))
    adapter_ids = {id(p) for p in adapter_params}
    encoder_ids = {id(p) for p in encoder_params}
    head_params = []
    for param in _iter_trainable_params(raw_model):
        pid = id(param)
        if pid in adapter_ids or pid in encoder_ids:
            continue
        head_params.append(param)

    param_groups = []
    if adapter_params:
        param_groups.append({'params': adapter_params, 'lr': adapter_lr, 'name': 'adapter'})
    if head_params:
        param_groups.append({'params': head_params, 'lr': head_lr, 'name': 'heads'})
    if encoder_params:
        if encoder_lr is None:
            raise ValueError(
                'Single-station encoder is trainable, but single_station_pretrain.lr_encoder is not set.'
            )
        param_groups.append({'params': encoder_params, 'lr': encoder_lr, 'name': 'encoder'})
    if not param_groups:
        raise ValueError('No trainable parameters found for single-station optimizer.')

    optimizer_name = pretrain_params.get('optimizer', training_params.get('optimizer', 'adam')).lower()
    weight_decay = pretrain_params.get('weight_decay', training_params.get('weight_decay', 0.0))
    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(param_groups, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer {optimizer_name!r}; use 'adam' or 'adamw'.")

    group_summary = {
        group['name']: {
            'lr': group['lr'],
            'n_params': sum(p.numel() for p in group['params']),
        }
        for group in param_groups
    }
    return optimizer, group_summary


def single_station_multitask_loss(outputs, targets, tasks, task_weights,
                                  loss_type='huber', huber_delta=1.0):
    weights = torch.tensor([float(task_weights[t]) for t in tasks], device=next(iter(outputs.values())).device)
    weights = weights / weights.sum().clamp_min(1e-8)
    total = outputs[tasks[0]][:, 0].new_tensor(0.0)
    metrics = {}
    for weight, task in zip(weights, tasks):
        pred = outputs[task]
        mu = pred[:, 0]
        sigma = pred[:, 1].clamp_min(1e-4)
        target = targets[task].float().to(mu.device).reshape_as(mu)
        if loss_type == 'gaussian_nll':
            comp = 0.5 * ((target - mu) / sigma) ** 2 + torch.log(sigma)
            comp_loss = comp.mean()
        elif loss_type == 'huber':
            comp_loss = F.smooth_l1_loss(mu, target, beta=huber_delta, reduction='mean')
        else:
            raise ValueError(f"Unsupported single-station loss {loss_type!r}")
        total = total + weight * comp_loss
        metrics[f'{task}/loss'] = comp_loss.detach()
        metrics[f'{task}/mae'] = torch.mean(torch.abs(mu - target)).detach()
        metrics[f'{task}/pred_mean'] = mu.mean().detach()
        metrics[f'{task}/target_mean'] = target.mean().detach()
    return total, metrics


def train_single_station_model(model, train_loader, val_loader, optimizer, scheduler, pretrain_params,
                               weight_path, device, is_dist=False, rank=0, train_sampler=None,
                               checkpoint_params=None):
    raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    tasks = list(pretrain_params.get('tasks', ['mag', 'epidist', 'pga']))
    task_weights_cfg = pretrain_params.get('task_weights', {task: 1.0 for task in tasks})
    if isinstance(task_weights_cfg, list):
        if len(task_weights_cfg) != len(tasks):
            raise ValueError('single_station_pretrain.task_weights list must match tasks length.')
        task_weights = dict(zip(tasks, task_weights_cfg))
    else:
        task_weights = {task: float(task_weights_cfg.get(task, 1.0)) for task in tasks}
    loss_type = pretrain_params.get('loss', 'huber')
    huber_delta = float(pretrain_params.get('huber_delta', 1.0))
    num_epochs = int(pretrain_params.get('epochs', 1))
    clipnorm = pretrain_params.get('clipnorm', pretrain_params.get('clip_norm', None))
    lr_monitor = pretrain_params.get('lr_monitor', 'val')

    scalar_history = defaultdict(list)
    writer = None
    if (not is_dist) or rank == 0:
        writer = SummaryWriter(log_dir=f'runs/{weight_path}/single_station')
    global_step = 0
    best_val = float('inf')
    best_path = os.path.join(weight_path, 'single_station_best.pth')
    last_path = os.path.join(weight_path, 'single_station_last.pth')
    checkpoint_training_params = {
        'checkpoint': checkpoint_params if checkpoint_params is not None else pretrain_params.get('checkpoint', {})
    }
    try:
        for epoch in range(num_epochs):
            if is_dist and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            running_loss = 0.0
            num_train_batches = 0
            first_batch_logged = False
            for batch_idx, (waveforms, targets, _) in enumerate(train_loader):
                waveforms = waveforms.to(device)
                targets = {key: value.to(device) for key, value in targets.items() if torch.is_tensor(value)}
                optimizer.zero_grad()
                outputs = model(waveforms)
                loss, metrics = single_station_multitask_loss(
                    outputs, targets, tasks, task_weights, loss_type=loss_type, huber_delta=huber_delta
                )
                loss.backward()
                pre_clip_global_grad = global_grad_norm(model.parameters())
                if clipnorm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clipnorm)
                post_clip_global_grad = global_grad_norm(model.parameters())
                optimizer.step()

                if (not is_dist) or rank == 0:
                    _record_scalar(writer, scalar_history, 'single_station/train_loss', loss.item(), global_step)
                    if global_step % 50 == 0:
                        print(f'[single] Step/Epoch {batch_idx}/{epoch}, Loss: {loss.item():.4f}')
                    if not first_batch_logged:
                        for key, value in getattr(raw_model, '_last_diag', {}).items():
                            if torch.is_tensor(value) and not torch.isnan(value).any():
                                _record_scalar(writer, scalar_history, f'single_station/diag/{key}', value, epoch)
                        for key, value in metrics.items():
                            _record_scalar(writer, scalar_history, f'single_station/train_{key}', value, epoch)
                        if pre_clip_global_grad is not None and not torch.isnan(pre_clip_global_grad).any():
                            _record_scalar(writer, scalar_history, 'single_station/grad/global_pre_clip_norm', pre_clip_global_grad, epoch)
                        if post_clip_global_grad is not None and not torch.isnan(post_clip_global_grad).any():
                            _record_scalar(writer, scalar_history, 'single_station/grad/global_post_clip_norm', post_clip_global_grad, epoch)
                        first_batch_logged = True
                    if global_step % 100 == 0:
                        for group in optimizer.param_groups:
                            group_name = group.get('name', 'group')
                            _record_scalar(writer, scalar_history, f'single_station/lr_{group_name}', group['lr'], global_step)

                running_loss += loss.item()
                num_train_batches += 1
                global_step += 1

            if is_dist:
                train_stats = torch.tensor([running_loss, float(num_train_batches)], device=device)
                dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)
                epoch_loss = (train_stats[0] / train_stats[1].clamp_min(1.0)).item()
            else:
                epoch_loss = running_loss / max(num_train_batches, 1)

            raw_model.eval()
            val_running_loss = 0.0
            num_val_batches = 0
            val_metrics_accum = defaultdict(float)
            with torch.no_grad():
                for waveforms, targets, _ in val_loader:
                    waveforms = waveforms.to(device)
                    targets = {key: value.to(device) for key, value in targets.items() if torch.is_tensor(value)}
                    outputs = raw_model(waveforms)
                    loss, metrics = single_station_multitask_loss(
                        outputs, targets, tasks, task_weights, loss_type=loss_type, huber_delta=huber_delta
                    )
                    val_running_loss += loss.item()
                    num_val_batches += 1
                    for key, value in metrics.items():
                        val_metrics_accum[key] += float(value.item())

            if is_dist:
                val_stats = torch.tensor([val_running_loss, float(num_val_batches)], device=device)
                dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)
                val_loss = (val_stats[0] / val_stats[1].clamp_min(1.0)).item()
            else:
                val_loss = val_running_loss / max(num_val_batches, 1)

            if (not is_dist) or rank == 0:
                _record_scalar(writer, scalar_history, 'single_station/train_epoch_loss', epoch_loss, epoch)
                _record_scalar(writer, scalar_history, 'single_station/val_epoch_loss', val_loss, epoch)
                for key, value in val_metrics_accum.items():
                    _record_scalar(
                        writer,
                        scalar_history,
                        f'single_station/val_{key}',
                        value / max(num_val_batches, 1),
                        epoch,
                    )
                print(f'[single] Epoch [{epoch+1}/{num_epochs}], train={epoch_loss:.4f}, val={val_loss:.4f}')

                if val_loss < best_val:
                    best_val = val_loss
                    save_model_checkpoint(
                        best_path,
                        raw_model,
                        epoch=epoch + 1,
                        training_params=checkpoint_training_params,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss=val_loss,
                        extra={'tasks': tasks},
                    )

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_loss if lr_monitor == 'train' else val_loss)
            else:
                scheduler.step()
            if is_dist:
                dist.barrier()

        if (not is_dist) or rank == 0:
            save_model_checkpoint(
                last_path,
                raw_model,
                epoch=num_epochs,
                training_params=checkpoint_training_params,
                optimizer=optimizer,
                scheduler=scheduler,
                loss=best_val,
                extra={'tasks': tasks},
            )
    finally:
        if (not is_dist) or rank == 0:
            export_path, manifest_path = export_scalar_history(
                scalar_history,
                os.path.join('logs', weight_path, 'single_station'),
            )
            print(f'[single] Exported scalar history to {export_path} (manifest: {manifest_path})')
            if writer is not None:
                writer.close()
    return best_path if os.path.exists(best_path) else last_path


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


def _pga_norm_enabled(norm_cfg):
    return bool(norm_cfg and norm_cfg.get('enabled', False))


def _pga_norm_values(norm_cfg):
    return float(norm_cfg.get('mean', 0.0)), max(float(norm_cfg.get('std', 1.0)), 1e-8)


def _maybe_unnormalize_pga(pred, norm_cfg):
    if not _pga_norm_enabled(norm_cfg):
        return pred
    mean, std = _pga_norm_values(norm_cfg)
    return pred * std + mean


def collect_event_output_stats(outputs, labels, output_layout):
    stats = {}
    label_layout = [n for n in output_layout if n in ('mag', 'loc', 'pga')]
    for name in ('mag', 'loc'):
        if name not in output_layout or name not in label_layout:
            continue
        pred = outputs[output_layout.index(name)]
        target = labels[label_layout.index(name)].to(pred.device, dtype=pred.dtype)
        d = 3 if name == 'loc' else 1
        if pred.shape[-1] == d:
            mu = pred.reshape(pred.shape[0], -1, d)[:, 0, :]
        else:
            alpha_logits = pred[..., 0]
            mu_all = pred[..., 1:1 + d]
            best_idx = alpha_logits.argmax(dim=-1)
            if mu_all.ndim == 3:
                batch_idx = torch.arange(mu_all.shape[0], device=mu_all.device)
                mu = mu_all[batch_idx, best_idx]
            else:
                mu = mu_all
        target = target.reshape(target.shape[0], -1, d)[:, 0, :]
        stats.update({
            f'diag/{name}_mu_best_mean': mu.mean().detach(),
            f'diag/{name}_mu_best_std': mu.std(unbiased=False).detach(),
            f'diag/{name}_target_mean': target.mean().detach(),
            f'diag/{name}_target_std': target.std(unbiased=False).detach(),
            f'diag/{name}_pred_target_mean_gap': (mu.mean() - target.mean()).detach(),
            f'diag/{name}_pred_target_std_gap': (
                mu.std(unbiased=False) - target.std(unbiased=False)
            ).detach(),
        })
    return stats


def collect_pga_output_stats(outputs, labels, output_layout, pga_target_valid, pga_target_normalization=None):
    if 'pga' not in output_layout:
        return {}
    pga_idx = output_layout.index('pga')
    pga_pred = outputs[pga_idx]
    if pga_pred.shape[-1] == 1:
        mu = _maybe_unnormalize_pga(pga_pred[..., 0], pga_target_normalization)
        stats = {
            'diag/pga_point_mu_mean': mu.mean().detach(),
            'diag/pga_point_mu_std': mu.std(unbiased=False).detach(),
        }
        if _pga_norm_enabled(pga_target_normalization):
            stats.update({
                'diag/pga_norm_mean': torch.as_tensor(
                    float(pga_target_normalization['mean']), device=mu.device
                ),
                'diag/pga_norm_std': torch.as_tensor(
                    float(pga_target_normalization['std']), device=mu.device
                ),
            })
        if pga_target_valid is not None:
            mask = pga_target_valid.bool()
            if mask.any():
                pga_true = labels[pga_idx]
                valid_pred = mu[mask]
                valid_true = pga_true[..., 0][mask]
                stats.update({
                    'diag/pga_target_mean': valid_true.mean().detach(),
                    'diag/pga_target_std': valid_true.std(unbiased=False).detach(),
                    'diag/pga_mu_best_valid_mean': valid_pred.mean().detach(),
                    'diag/pga_mu_best_valid_std': valid_pred.std(unbiased=False).detach(),
                    'diag/pga_pred_target_mean_gap': (valid_pred.mean() - valid_true.mean()).detach(),
                    'diag/pga_pred_target_std_gap': (valid_pred.std(unbiased=False) - valid_true.std(unbiased=False)).detach(),
                    'diag/pga_valid_target_count': mask.sum().detach().float(),
                })
        return stats

    pga_true = labels[pga_idx]
    alpha_logits = pga_pred[..., 0]
    mu = pga_pred[..., 1]
    sigma = pga_pred[..., 2]
    alpha_probs = torch.softmax(alpha_logits, dim=-1)
    best_idx = alpha_logits.argmax(dim=-1, keepdim=True)
    mu_best = _maybe_unnormalize_pga(mu.gather(-1, best_idx).squeeze(-1), pga_target_normalization)
    sigma_best = sigma.gather(-1, best_idx).squeeze(-1)

    stats = {
        'diag/pga_alpha_logits_mean': alpha_logits.mean().detach(),
        'diag/pga_alpha_logits_std': alpha_logits.std(unbiased=False).detach(),
        'diag/pga_alpha_prob_mean': alpha_probs.mean().detach(),
        'diag/pga_alpha_prob_std': alpha_probs.std(unbiased=False).detach(),
        'diag/pga_alpha_entropy': (-(alpha_probs * torch.log(alpha_probs.clamp_min(1e-8))).sum(dim=-1)).mean().detach(),
        'diag/pga_mu_mean': mu.mean().detach(),
        'diag/pga_mu_std': mu.std(unbiased=False).detach(),
        'diag/pga_sigma_mean': sigma.mean().detach(),
        'diag/pga_sigma_std': sigma.std(unbiased=False).detach(),
        'diag/pga_mu_best_mean': mu_best.mean().detach(),
        'diag/pga_mu_best_std': mu_best.std(unbiased=False).detach(),
        'diag/pga_sigma_best_mean': sigma_best.mean().detach(),
        'diag/pga_sigma_best_std': sigma_best.std(unbiased=False).detach(),
    }
    num_components = alpha_logits.shape[-1]
    best_component = best_idx.squeeze(-1)
    for comp_idx in range(num_components):
        comp_mask = best_component == comp_idx
        stats.update({
            f'diag/pga_alpha_logit_{comp_idx}_mean': alpha_logits[..., comp_idx].mean().detach(),
            f'diag/pga_alpha_logit_{comp_idx}_std': alpha_logits[..., comp_idx].std(unbiased=False).detach(),
            f'diag/pga_alpha_prob_{comp_idx}_mean': alpha_probs[..., comp_idx].mean().detach(),
            f'diag/pga_alpha_prob_{comp_idx}_std': alpha_probs[..., comp_idx].std(unbiased=False).detach(),
            f'diag/pga_mu_{comp_idx}_mean': mu[..., comp_idx].mean().detach(),
            f'diag/pga_mu_{comp_idx}_std': mu[..., comp_idx].std(unbiased=False).detach(),
            f'diag/pga_sigma_{comp_idx}_mean': sigma[..., comp_idx].mean().detach(),
            f'diag/pga_sigma_{comp_idx}_std': sigma[..., comp_idx].std(unbiased=False).detach(),
            f'diag/pga_best_component_frac_{comp_idx}': comp_mask.float().mean().detach(),
        })
    if pga_target_valid is not None:
        mask = pga_target_valid.bool()
        if mask.any():
            valid_pred = mu_best[mask]
            valid_true = pga_true[mask]
            stats.update({
                'diag/pga_target_mean': valid_true.mean().detach(),
                'diag/pga_target_std': valid_true.std(unbiased=False).detach(),
                'diag/pga_mu_best_valid_mean': valid_pred.mean().detach(),
                'diag/pga_mu_best_valid_std': valid_pred.std(unbiased=False).detach(),
                'diag/pga_pred_target_mean_gap': (valid_pred.mean() - valid_true.mean()).detach(),
                'diag/pga_pred_target_std_gap': (valid_pred.std(unbiased=False) - valid_true.std(unbiased=False)).detach(),
                'diag/pga_valid_target_count': mask.sum().detach().float(),
            })
            for comp_idx in range(num_components):
                stats.update({
                    f'diag/pga_mu_{comp_idx}_valid_mean': mu[..., comp_idx][mask].mean().detach(),
                    f'diag/pga_mu_{comp_idx}_valid_std': mu[..., comp_idx][mask].std(unbiased=False).detach(),
                    f'diag/pga_sigma_{comp_idx}_valid_mean': sigma[..., comp_idx][mask].mean().detach(),
                    f'diag/pga_sigma_{comp_idx}_valid_std': sigma[..., comp_idx][mask].std(unbiased=False).detach(),
                })
    return stats


def resolve_pga_target_normalization(training_params, train_dataset, batch_size, is_dist=False, rank=0, device='cpu'):
    norm_cfg = training_params.get('pga_target_normalization') or {}
    if not norm_cfg.get('enabled', False):
        return None
    mean = norm_cfg.get('mean', None)
    std = norm_cfg.get('std', None)
    needs_auto = mean in (None, 'auto') or std in (None, 'auto')
    if not needs_auto:
        norm_cfg['mean'] = float(mean)
        norm_cfg['std'] = max(float(std), 1e-8)
        return norm_cfg

    stats = torch.zeros(3, dtype=torch.float64, device=device)
    if (not is_dist) or rank == 0:
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        total = 0.0
        total_sq = 0.0
        count = 0
        for inputs, labels, _ in loader:
            if not isinstance(inputs, list) or len(inputs) < 5:
                continue
            pga_target_valid = inputs[4].bool()
            pga_labels = labels[-1]
            values = pga_labels[..., 0] if pga_labels.ndim >= 3 else pga_labels
            valid_values = values[pga_target_valid]
            if valid_values.numel() == 0:
                continue
            valid_values = valid_values.double()
            total += float(valid_values.sum().item())
            total_sq += float((valid_values ** 2).sum().item())
            count += int(valid_values.numel())
        if count <= 1:
            raise ValueError('Unable to estimate PGA normalization statistics: no valid train targets.')
        mean_val = total / count
        var_val = max(total_sq / count - mean_val ** 2, 1e-12)
        stats[:] = torch.tensor([mean_val, math.sqrt(var_val), float(count)], dtype=torch.float64, device=device)

    if is_dist:
        dist.broadcast(stats, src=0)
    norm_cfg['mean'] = float(stats[0].item())
    norm_cfg['std'] = max(float(stats[1].item()), 1e-8)
    norm_cfg['count'] = int(stats[2].item())
    if (not is_dist) or rank == 0:
        print(
            '[pga_target_normalization] '
            f'mean={norm_cfg["mean"]:.6f}, std={norm_cfg["std"]:.6f}, count={norm_cfg["count"]}'
        )
    training_params['pga_target_normalization'] = norm_cfg
    return norm_cfg


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
        if isinstance(p_picks, dict) and 'station_snr' in p_picks and station_valid is not None:
            station_snr = p_picks['station_snr'].to(waveforms.device)
            if station_valid.any():
                valid_snr = station_snr[station_valid]
                stats.update({
                    'data/station_snr_mean': valid_snr.mean().detach(),
                    'data/station_snr_min': valid_snr.min().detach(),
                    'data/station_snr_max': valid_snr.max().detach(),
                })

    return stats


def _scalar_to_float(value):
    if torch.is_tensor(value):
        value = value.detach()
        if value.numel() != 1:
            raise ValueError(f'Expected scalar tensor, got shape {tuple(value.shape)}')
        value = value.item()
    return float(value)


def _record_scalar(writer, scalar_history, tag, value, step):
    scalar_value = _scalar_to_float(value)
    if writer is not None:
        writer.add_scalar(tag, scalar_value, step)
    scalar_history[tag].append({
        'step': int(step),
        'value': scalar_value,
    })


def _sanitize_scalar_tag(tag):
    return tag.replace('/', '_')


def export_scalar_history(scalar_history, export_dir):
    os.makedirs(export_dir, exist_ok=True)
    for name in os.listdir(export_dir):
        if name.endswith('.csv') or name == 'manifest.json':
            os.remove(os.path.join(export_dir, name))
    manifest = []
    for tag in sorted(scalar_history.keys()):
        entries = scalar_history[tag]
        if not entries:
            continue
        safe_name = _sanitize_scalar_tag(tag)
        path = os.path.join(export_dir, f'{safe_name}.csv')
        pd.DataFrame(entries, columns=['step', 'value']).to_csv(path, index=False)
        manifest.append({
            'tag': tag,
            'file': os.path.basename(path),
            'count': len(entries),
        })
    manifest_path = os.path.join(export_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    return export_dir, manifest_path

def _to_cpu_dumpable(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu_dumpable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_cpu_dumpable(v) for v in obj]
    return obj


def prepare_input_dump_config(training_params):
    cfg = training_params.get('model_input_dump', None)
    if cfg is None:
        cfg = {}
    enabled = bool(cfg.get('enabled', False))
    epochs = sorted({int(e) for e in cfg.get('epochs', [])})
    include_val = bool(cfg.get('include_val', False))
    max_batches = cfg.get('max_batches_per_epoch', None)
    if max_batches is not None:
        max_batches = int(max_batches)
    dump_root = os.path.join(training_params['weight_path'], 'model_input_dumps')
    return {
        'enabled': enabled,
        'epochs': epochs,
        'include_val': include_val,
        'max_batches_per_epoch': max_batches,
        'dump_root': dump_root,
    }


def initialize_input_dump(input_dump_config):
    if not input_dump_config['enabled']:
        return
    dump_root = input_dump_config['dump_root']
    os.makedirs(dump_root, exist_ok=True)
    manifest_path = os.path.join(dump_root, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump({
            'epochs': input_dump_config['epochs'],
            'include_val': input_dump_config['include_val'],
            'max_batches_per_epoch': input_dump_config['max_batches_per_epoch'],
        }, f, indent=2)


def maybe_dump_model_batch(input_dump_config, split_name, epoch_idx, batch_idx, global_step, inputs, labels, p_picks):
    if not input_dump_config['enabled']:
        return
    epoch_num = epoch_idx + 1
    if epoch_num not in input_dump_config['epochs']:
        return
    max_batches = input_dump_config['max_batches_per_epoch']
    if max_batches is not None and batch_idx >= max_batches:
        return
    epoch_dir = os.path.join(
        input_dump_config['dump_root'],
        split_name,
        f'epoch_{epoch_num:03d}',
    )
    os.makedirs(epoch_dir, exist_ok=True)
    path = os.path.join(epoch_dir, f'batch_{batch_idx:04d}.pt')
    payload = {
        'epoch': epoch_num,
        'split': split_name,
        'batch_idx': int(batch_idx),
        'global_step': int(global_step),
        'inputs': _to_cpu_dumpable(inputs),
        'labels': _to_cpu_dumpable(labels),
        'p_pick_info': _to_cpu_dumpable(p_picks),
    }
    torch.save(payload, path)


def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, clipnorm=None, is_dist=False, rank=0, save_name=None,
                res_comps=None, res_weight=None, post_train_sanity=False, epoch_sanity=False, train_sampler=None, lr_monitor='val',
                input_dump_config=None, loss_type='mdn', huber_delta=1.0, checkpoint_params=None,
                pga_target_normalization=None, station_decorrelation_weight=0.0):
    tb_path = f'runs/{save_name}'
    eval_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    scalar_history = defaultdict(list)
    writer = None
    if (not is_dist) or (is_dist and (rank == 0)):
        os.makedirs(tb_path, exist_ok=True)
        writer = SummaryWriter(log_dir = tb_path)
        if input_dump_config is not None:
            initialize_input_dump(input_dump_config)
    try:
        device = next(model.parameters()).device
    except:
        device = 'cpu'
    global_step = 0
    steps_per_epoch = 0
    export_dir = os.path.join('logs', training_params['weight_path'])
    best_val = float('inf')
    best_path = os.path.join(training_params['weight_path'], f'{save_name}_best.pth')
    last_path = os.path.join(training_params['weight_path'], f'{save_name}_last.pth')
    try:
        for epoch in range(num_epochs):
            if is_dist and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            running_loss = 0.0
            num_train_batches = 0
            first_batch_logged = False
            for batch_idx, (inputs, labels, p_picks) in enumerate(train_loader):
                if ((not is_dist) or (is_dist and (rank == 0))) and input_dump_config is not None:
                    maybe_dump_model_batch(
                        input_dump_config, 'train', epoch, batch_idx, global_step, inputs, labels, p_picks
                    )
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
                if loss_type in ('mdn', 'gaussian', 'gaussian_nll'):
                    loss = models.mixture_density_loss_full(sel_pred, sel_true, res_comps=res_comps, res_weight=res_weight,
                                                            pga_target_valid=pga_target_valid)
                else:
                    loss = models.point_regression_loss_full(
                        sel_pred, sel_true, res_comps=res_comps, res_weight=res_weight,
                        pga_target_valid=pga_target_valid, loss_type=loss_type,
                        huber_delta=huber_delta,
                        pga_target_normalization=pga_target_normalization,
                    )
                if station_decorrelation_weight:
                    decor_loss = models.station_embedding_decorrelation_loss(
                        getattr(eval_model, '_last_station_feature_emb', None),
                        getattr(eval_model, '_last_station_valid', None),
                    )
                    if decor_loss is not None:
                        loss = loss + float(station_decorrelation_weight) * decor_loss
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
                    diag_scalars.update(collect_event_output_stats(outputs, labels, eval_model.output_layout))
                    diag_scalars.update(collect_pga_output_stats(
                        outputs,
                        labels,
                        eval_model.output_layout,
                        pga_target_valid,
                        pga_target_normalization=pga_target_normalization,
                    ))
                    if station_decorrelation_weight:
                        decor_loss = models.station_embedding_decorrelation_loss(
                            getattr(eval_model, '_last_station_feature_emb', None),
                            getattr(eval_model, '_last_station_valid', None),
                        )
                        if decor_loss is not None and not torch.isnan(decor_loss).any():
                            diag_scalars['diag/station_decorrelation_loss'] = decor_loss.detach()

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
                    _record_scalar(writer, scalar_history, 'train/loss', loss.item(), global_step)
                    step_in_ep = global_step - steps_per_epoch * epoch
                    print(f'Step/Epoch {step_in_ep}/{epoch}, Loss: {loss.item():.4f}')
                if clipnorm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clipnorm)
                post_clip_global_grad = global_grad_norm(model.parameters())
                if (((not is_dist) or (is_dist and (rank == 0))) and not first_batch_logged):
                    if post_clip_global_grad is not None and not torch.isnan(post_clip_global_grad).any():
                        diag_scalars['grad/global_post_clip_norm'] = post_clip_global_grad.detach()
                    for key, value in diag_scalars.items():
                        _record_scalar(writer, scalar_history, key, value, epoch)
                    first_batch_logged = True
                optimizer.step()
                running_loss += loss.item()
                num_train_batches += 1
                if global_step % 100 == 0:
                    if (not is_dist) or (is_dist and (rank == 0)):
                        _record_scalar(writer, scalar_history, 'train/lr', optimizer.param_groups[0]['lr'], global_step)
                        for group in optimizer.param_groups:
                            group_name = group.get('name')
                            if group_name is not None:
                                _record_scalar(writer, scalar_history, f'train/lr_{group_name}', group['lr'], global_step)
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
                _record_scalar(writer, scalar_history, 'train/epoch_loss', epoch_loss, epoch)
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

            # Validation step
            eval_model.eval()
            val_running_loss = 0.0
            num_val_batches = 0
            with torch.no_grad():
                for batch_idx, (inputs, labels, p_picks) in enumerate(val_loader):
                    if ((not is_dist) or (is_dist and (rank == 0))) and input_dump_config is not None and input_dump_config['include_val']:
                        maybe_dump_model_batch(
                            input_dump_config, 'val', epoch, batch_idx, global_step, inputs, labels, p_picks
                        )
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
                    if loss_type in ('mdn', 'gaussian', 'gaussian_nll'):
                        loss = models.mixture_density_loss_full(sel_pred, sel_true, res_comps=res_comps, res_weight=res_weight,
                                                                pga_target_valid=pga_target_valid)
                    else:
                        loss = models.point_regression_loss_full(
                            sel_pred, sel_true, res_comps=res_comps, res_weight=res_weight,
                            pga_target_valid=pga_target_valid, loss_type=loss_type,
                            huber_delta=huber_delta,
                            pga_target_normalization=pga_target_normalization,
                        )
                    val_running_loss += loss.item()
                    num_val_batches += 1

            if is_dist:
                val_stats = torch.tensor([val_running_loss, float(num_val_batches)], device=device)
                dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)
                val_loss = (val_stats[0] / val_stats[1]).item()
            else:
                val_loss = val_running_loss / max(num_val_batches, 1)

            if (not is_dist) or (is_dist and (rank == 0)):
                _record_scalar(writer, scalar_history, 'val/epoch_loss', val_loss, epoch)
                print(f'Validation Loss: {val_loss:.4f}')

                if val_loss < best_val:
                    best_val = val_loss
                    save_model_checkpoint(
                        best_path,
                        eval_model,
                        epoch=epoch + 1,
                        training_params={'checkpoint': checkpoint_params or {}},
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss=val_loss,
                    )
                save_model_checkpoint(
                    last_path,
                    eval_model,
                    epoch=epoch + 1,
                    training_params={'checkpoint': checkpoint_params or {}},
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss=val_loss,
                )
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
    finally:
        if (not is_dist) or (is_dist and (rank == 0)):
            export_path, manifest_path = export_scalar_history(scalar_history, export_dir)
            print(f'Exported scalar history to {export_path} (manifest: {manifest_path})')
            if writer is not None:
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

    state_dict = checkpoint['model_state_dict']

    load_model_state_dict_compatible(
        model,
        state_dict,
        strict=True,
        context=checkpoint_path,
        allowed_missing_prefixes=tuple(checkpoint.get('excluded_prefixes', CHECKPOINT_ENCODER_PREFIXES)),
    )

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint.get('epoch', 0)
    val_loss = checkpoint.get('val_loss', checkpoint.get('loss', None))

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
    state_dict = clean_state_dict_keys(state_dict)

    # Unwrap DDP if needed
    raw_model = model.module if hasattr(model, 'module') else model

    # 判断目标模型是不是 borehole (Conv1d 输入通道是否 64)
    conv1d_layer = None
    conv1d_name = None
    for name, module in raw_model.named_modules():
        weight_name = name + ".weight"
        if isinstance(module, nn.Conv1d) and weight_name in state_dict:
            conv1d_layer = module
            conv1d_name = weight_name
            break

    if conv1d_layer is not None:
        model_borehole = conv1d_layer.in_channels == 64
        weights_borehole = state_dict[conv1d_name].shape[1] == 64

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

    # 删除 embedding 层权重（如果存在）
    for key in list(state_dict.keys()):
        if key.startswith("embedding"):
            del state_dict[key]

    # 加载参数
    missing, unexpected = raw_model.load_state_dict(state_dict, strict=False)
    print(f"Transferred {len(state_dict) - len(unexpected)} weights, "
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
                    station_has_signal = (torch.abs(wave) > 1e-7).flatten(2).any(dim=2)
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
                if out.ndim >= 3 and out.shape[-1] == 1:
                    mu_point = out[..., 0]
                    print(f'    point_mu: mean={mu_point.mean().item():.4f}, std={mu_point.std(unbiased=False).item():.4f}')
                elif out.ndim >= 3:
                    # Determine d (number of mu/sigma dimensions) from output shape
                    # Layout: [alpha, mu_1..mu_d, sigma_1..sigma_d] → total = 1 + 2d
                    last_dim = out.shape[-1]
                    d = (last_dim - 1) // 2
                    alpha_logits = out[..., 0]
                    alpha_probs = torch.softmax(alpha_logits, dim=-1)
                    best_idx = alpha_logits.argmax(dim=-1)
                    print(f'    alpha_logits: mean={alpha_logits.mean().item():.4f}, std={alpha_logits.std().item():.4f}')
                    print(f'    alpha_probs: mean={alpha_probs.mean().item():.4f}, std={alpha_probs.std().item():.4f}, entropy={(-(alpha_probs * torch.log(alpha_probs.clamp_min(1e-8))).sum(dim=-1)).mean().item():.4f}')
                    dim_names = ['lat', 'lon', 'depth'] if d == 3 else [str(j) for j in range(d)]
                    num_components = out.shape[-2]
                    component_names = range(min(num_components, 5))
                    for comp_idx in component_names:
                        comp_frac = (best_idx == comp_idx).float().mean().item()
                        print(
                            f'    comp[{comp_idx}]: '
                            f'logit_mean={alpha_logits[..., comp_idx].mean().item():.4f}, '
                            f'prob_mean={alpha_probs[..., comp_idx].mean().item():.4f}, '
                            f'best_frac={comp_frac:.4f}'
                        )
                    for j in range(d):
                        mu_j = out[..., :, 1 + j]
                        sigma_j = out[..., :, 1 + d + j]
                        name_j = dim_names[j] if j < len(dim_names) else str(j)
                        print(f'    mu_{name_j}: mean={mu_j.mean().item():.4f}, std={mu_j.std().item():.4f}')
                        print(f'    sigma_{name_j}: mean={sigma_j.mean().item():.4f}, std={sigma_j.std().item():.4f}')
                        for comp_idx in component_names:
                            print(
                                f'      comp[{comp_idx}] mu_{name_j}: mean={mu_j[..., comp_idx].mean().item():.4f}, std={mu_j[..., comp_idx].std().item():.4f}'
                            )
                            print(
                                f'      comp[{comp_idx}] sigma_{name_j}: mean={sigma_j[..., comp_idx].mean().item():.4f}, std={sigma_j[..., comp_idx].std().item():.4f}'
                            )
                        if d == 1:
                            mu_best = mu_j.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
                            sigma_best = sigma_j.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
                            print(f'    mu_best: mean={mu_best.mean().item():.4f}, std={mu_best.std().item():.4f}')
                            print(f'    sigma_best: mean={sigma_best.mean().item():.4f}, std={sigma_best.std().item():.4f}')
                    if num_components > 5:
                        print(f'    ... skipped remaining {num_components - 5} mixture components')
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
    parser.add_argument('--skip_single_station_pretrain', action='store_true')
    parser.add_argument('--single_station_only', action='store_true')
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
    checkpoint_cfg = training_params.setdefault('checkpoint', {})
    checkpoint_cfg.setdefault('encoder_source', getattr(diting_args, 'pretrained', None))
    generator_params = training_params.get('generator_params', [training_params.copy()])

    if (not is_dist) or (is_dist and (rank == 0)):
        os.makedirs(training_params['weight_path'], exist_ok=True)
    if is_dist:
        dist.barrier()
    listdir = os.listdir(training_params['weight_path'])
    if not args.continue_ensemble and listdir:
        if len(listdir) != 1 or listdir[0] != 'config.json':
            raise ValueError(f'Weight path needs to be empty. ({training_params["weight_path"]})')

    if (not is_dist) or (is_dist and (rank == 0)):
        with open(os.path.join(training_params['weight_path'], 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
    if is_dist:
        dist.barrier()

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
    input_dump_config = prepare_input_dump_config(training_params)
    if ((not is_dist) or (is_dist and (rank == 0))) and input_dump_config['enabled']:
        print(
            f'[model_input_dump] enabled epochs={input_dump_config["epochs"]}, '
            f'include_val={input_dump_config["include_val"]}, '
            f'max_batches_per_epoch={input_dump_config["max_batches_per_epoch"]}, '
            f'root={input_dump_config["dump_root"]}'
        )

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
            print('Overfit mode adjustments: trigger_based disabled, station foreshadowing enabled, oversample=1, fixed cutout, no train/dev split shuffling; input/target station selection follows config')

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
    auto_station_pretrain_path = None
    single_pretrain_params = training_params.get('single_station_pretrain', {})
    run_single_pretrain = bool(single_pretrain_params.get('enabled', False)) and not args.skip_single_station_pretrain

    if run_single_pretrain:
        if rank == 0:
            print('Building single-station pretraining model')
        single_model_params = copy.deepcopy(config['model_params'])
        single_model_params['single_station_tasks'] = single_pretrain_params.get(
            'tasks', ['mag', 'epidist', 'pga']
        )
        single_model_params['single_station_hidden_dim'] = single_pretrain_params.get(
            'hidden_dim', single_model_params.get('single_station_hidden_dim', None)
        )
        single_model_params['single_station_task_output_init'] = single_pretrain_params.get(
            'task_output_init', None
        )
        single_model_params['single_station_task_sigma_init'] = single_pretrain_params.get(
            'task_sigma_init', None
        )
        single_model = models.build_single_station_model(
            **single_model_params,
            trace_length=10000,
            diting_args=diting_args,
        )
        single_model.to(device)

        single_load_path = single_pretrain_params.get('load_model_path', None)
        if single_load_path:
            if rank == 0:
                print(f'Loading single-station pretrain checkpoint from {single_load_path}')
            state_dict = _state_dict_from_checkpoint(single_load_path, device)
            load_model_state_dict_compatible(
                single_model,
                state_dict,
                strict=False,
                context=single_load_path,
            )

        configure_single_station_trainability(single_model, single_pretrain_params, rank=rank)
        if is_dist:
            single_model = DDP(
                single_model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )

        single_optimizer, single_group_summary = build_single_station_optimizer(
            single_model, training_params, single_pretrain_params, is_dist=is_dist
        )
        if rank == 0:
            for group_name, info in single_group_summary.items():
                print(
                    f'Single optimizer group {group_name}: '
                    f'lr={info["lr"]:.6g}, n_params={info["n_params"]}'
                )

        single_train_generators = []
        single_validation_generators = []
        single_tasks = single_pretrain_params.get('tasks', ['mag', 'epidist', 'pga'])
        single_samples_per_event = single_pretrain_params.get('samples_per_event', 1)
        single_station_sampling = single_pretrain_params.get('station_sampling', 'random')

        for i, generator_param_set_src in enumerate(generator_params):
            generator_param_set = copy.deepcopy(generator_param_set_src)
            noise_seconds = generator_param_set.get('noise_seconds', 5)
            cutout = (
                int(round(sampling_rate * (noise_seconds + generator_param_set['cutout_start']))),
                int(round(sampling_rate * (noise_seconds + generator_param_set['cutout_end']))))
            generator_param_set['transform_target_only'] = False
            defaults = dict(
                coords_target=True,
                label_smoothing=False,
                station_blinding=False,
                cutout=cutout,
                pga_targets=0,
                max_stations=max_stations,
                sampling_rate=sampling_rate,
                no_event_token=False,
                dump_debug_snapshot=False,
                use_coords_rel=config['model_params'].get('use_coords_rel', False),
                use_coords_abs=config['model_params'].get('use_coords_abs', True),
                use_coords_rel_abs_fusion=config['model_params'].get('use_coords_rel_abs_fusion', False),
            )
            merged = {**defaults, **generator_param_set}
            merged['transform_target_only'] = False
            merged['pga_targets'] = 0
            if rank == 0:
                print(
                    f'[single/generator/train/{i}] '
                    f'tasks={single_tasks}, samples_per_event={single_samples_per_event}, '
                    f'cutout=({merged["cutout"][0]}, {merged["cutout"][1]})'
                )
            event_train_generator = util.PreloadedEventGenerator(
                event_metadata=event_metadata_train[i],
                metadata=metadata_train[i],
                data_path=training_params['data_path'][i],
                generator_params=generator_param_set,
                **merged,
            )
            single_train_generators.append(SingleStationTaskDataset(
                event_train_generator,
                tasks=single_tasks,
                samples_per_event=single_samples_per_event,
                station_sampling=single_station_sampling,
            ))

            val_generator_param_set = copy.deepcopy(generator_param_set_src)
            val_generator_param_set['oversample'] = 1
            val_generator_param_set['transform_target_only'] = False
            merged_val = {**defaults, **val_generator_param_set}
            merged_val['transform_target_only'] = False
            merged_val['pga_targets'] = 0
            if rank == 0:
                print(
                    f'[single/generator/val/{i}] '
                    f'tasks={single_tasks}, samples_per_event=1, '
                    f'cutout=({merged_val["cutout"][0]}, {merged_val["cutout"][1]})'
                )
            event_val_generator = util.PreloadedEventGenerator(
                event_metadata=event_metadata_dev[i],
                metadata=metadata_dev[i],
                data_path=training_params['data_path'][i],
                generator_params=val_generator_param_set,
                **merged_val,
            )
            single_validation_generators.append(SingleStationTaskDataset(
                event_val_generator,
                tasks=single_tasks,
                samples_per_event=1,
                station_sampling=single_station_sampling,
            ))

        if len(single_train_generators) == 1:
            single_train_dataset = single_train_generators[0]
            single_val_dataset = single_validation_generators[0]
        else:
            single_train_dataset = ConcatDataset(single_train_generators)
            single_val_dataset = ConcatDataset(single_validation_generators)

        if is_dist:
            single_train_sampler = DistributedSampler(single_train_dataset)
            single_val_sampler = DistributedSampler(single_val_dataset, shuffle=False)
        else:
            single_train_sampler = None
            single_val_sampler = None
        single_batch_size = int(single_pretrain_params.get('batch_size', generator_params[0]['batch_size']))
        single_train_loader = DataLoader(
            single_train_dataset,
            batch_size=single_batch_size,
            sampler=single_train_sampler,
            shuffle=(single_train_sampler is None),
        )
        single_val_loader = DataLoader(
            single_val_dataset,
            batch_size=single_batch_size,
            sampler=single_val_sampler,
            shuffle=False,
        )

        single_patience = single_pretrain_params.get(
            'lr_decay_patience',
            training_params.get('lr_decay_patience', 6),
        )
        single_lr_decay = ReduceLROnPlateau(
            single_optimizer,
            mode='min',
            factor=single_pretrain_params.get('lr_decay_factor', 0.3),
            patience=single_patience,
            verbose=1,
        )
        if rank == 0:
            init_path = os.path.join(training_params['weight_path'], 'single_station_init.pth')
            raw_single = single_model.module if is_dist else single_model
            save_model_checkpoint(
                init_path,
                raw_single,
                epoch=0,
                training_params=training_params,
            )

        auto_station_pretrain_path = train_single_station_model(
            single_model,
            single_train_loader,
            single_val_loader,
            single_optimizer,
            single_lr_decay,
            single_pretrain_params,
            training_params['weight_path'],
            device,
            is_dist=is_dist,
            rank=rank,
            train_sampler=single_train_sampler,
            checkpoint_params=training_params.get('checkpoint', None),
        )
        if is_dist:
            dist.barrier()
        if args.single_station_only:
            if is_dist:
                dist.destroy_process_group()
            sys.exit(0)
    elif args.single_station_only:
        raise ValueError('--single_station_only requires training_params.single_station_pretrain.enabled=true')

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

            if (not is_dist) or (is_dist and (rank == 0)):
                with open(os.path.join(training_params['weight_path'], 'config.json'), 'w') as f:
                    json.dump(config, f, indent=4)
            if is_dist:
                dist.barrier()

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
            load_model_state_dict_compatible(
                load_target,
                state_dict,
                strict=True,
                context=training_params['load_model_path'],
                allowed_missing_prefixes=tuple(
                    ckpt.get('excluded_prefixes', CHECKPOINT_ENCODER_PREFIXES)
                    if isinstance(ckpt, dict) else CHECKPOINT_ENCODER_PREFIXES
                ),
            )

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

        station_pretrain_path = training_params.get('station_pretrain_path', auto_station_pretrain_path)
        if station_pretrain_path and 'load_model_path' in training_params and 'station_pretrain_path' not in training_params:
            station_pretrain_path = None

        reinit_fpn = training_params.get('reinit_fpn', True)
        if station_pretrain_path:
            reinit_fpn = False
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
            elif station_pretrain_path:
                print('Skipped station adapter reinit because station pretrain weights will be loaded')
            else:
                print('Kept current station adapter initialization')

        if station_pretrain_path:
            load_station_pretrain_weights(
                full_model,
                station_pretrain_path,
                device=device,
                load_encoder=training_params.get('station_pretrain_load_encoder', False),
                load_scale=training_params.get('station_pretrain_load_scale', True),
                load_layernorm=training_params.get('station_pretrain_load_layernorm', True),
            )

        no_event_token = config['model_params'].get('no_event_token', False)
        optimizer, optimizer_group_summary = build_optimizer_with_groups(
            full_model, training_params, is_dist=is_dist
        )
        if rank == 0:
            for group_name, info in optimizer_group_summary.items():
                print(
                    f'Optimizer group {group_name}: '
                    f'lr={info["lr"]:.6g}, n_params={info["n_params"]}'
                )

        n_pga_targets = config['model_params'].get('n_pga_targets', 0)
        station_experiment_cfg = training_params.get('station_experiment', None)

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
                dump_debug_snapshot=input_dump_config['enabled'],
                use_coords_rel=config['model_params'].get('use_coords_rel', False),
                use_coords_abs=config['model_params'].get('use_coords_abs', True),
                use_coords_rel_abs_fusion=config['model_params'].get('use_coords_rel_abs_fusion', False),
                station_experiment=station_experiment_cfg,
            )
            # Config wins: overlay generator_param_set on top of defaults.
            merged = {**defaults, **generator_param_set}
            if rank == 0:
                experiment = merged.get('station_experiment') or {}
                print(
                    f'[generator/train/{i}] '
                    f'select_first_inputs={merged.get("select_first_inputs", merged.get("select_first"))}, '
                    f'select_first_pga_targets={merged.get("select_first_pga_targets", merged.get("select_first"))}, '
                    f'integrate={merged.get("integrate", False)}, '
                    f'selection_skew={merged.get("selection_skew")}, '
                    f'pga_selection_skew={merged.get("pga_selection_skew")}, '
                    f'max_stations={merged.get("max_stations")}, '
                    f'station_experiment={experiment.get("mode") if experiment.get("enabled") else None}, '
                    f'cutout=({merged["cutout"][0]}, {merged["cutout"][1]})'
                )

            train_generators += [util.PreloadedEventGenerator(event_metadata=event_metadata_train[i],
                                                              metadata=metadata_train[i],
                                                              data_path=training_params['data_path'][i],
                                                              generator_params=generator_params[i],
                                                              **merged)]

            old_oversample = generator_param_set.get('oversample', 1)
            generator_param_set['oversample'] = 1 if overfit_mode else 4
            merged_val = {**defaults, **generator_param_set}
            if rank == 0:
                experiment_val = merged_val.get('station_experiment') or {}
                print(
                    f'[generator/val/{i}] '
                    f'select_first_inputs={merged_val.get("select_first_inputs", merged_val.get("select_first"))}, '
                    f'select_first_pga_targets={merged_val.get("select_first_pga_targets", merged_val.get("select_first"))}, '
                    f'integrate={merged_val.get("integrate", False)}, '
                    f'selection_skew={merged_val.get("selection_skew")}, '
                    f'pga_selection_skew={merged_val.get("pga_selection_skew")}, '
                    f'max_stations={merged_val.get("max_stations")}, '
                    f'station_experiment={experiment_val.get("mode") if experiment_val.get("enabled") else None}, '
                    f'cutout=({merged_val["cutout"][0]}, {merged_val["cutout"][1]})'
                )
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

        pga_target_norm_cfg = resolve_pga_target_normalization(
            training_params,
            train_dataset,
            batch_size=generator_params[0]['batch_size'],
            is_dist=is_dist,
            rank=rank,
            device=device,
        )
        if (not is_dist) or (is_dist and rank == 0):
            with open(os.path.join(training_params['weight_path'], 'config.json'), 'w') as f:
                json.dump(config, f, indent=4)
        if is_dist:
            dist.barrier()

        patience = training_params.get('lr_decay_patience', 6)
        lr_decay_factor = float(training_params.get('lr_decay_factor', 0.3))
        min_lr = float(training_params.get('min_lr', 0.0))
#        lr_decay = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=patience, factor=0.3, verbose=1) # need modify
        lr_decay = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=lr_decay_factor,
            patience=patience,
            min_lr=min_lr,
            verbose=1,
        )
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
        if ((not is_dist) or rank == 0):
            # Save initial checkpoint before training
            init_ckpt_path = os.path.join(training_params['weight_path'], 'full_model_init.pth')
            eval_model = full_model.module if is_dist else full_model
            save_model_checkpoint(
                init_ckpt_path,
                eval_model,
                epoch=0,
                training_params=training_params,
            )
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
        default_full_loss = 'huber' if config['model_params'].get('output_distribution', 'mdn') == 'point' else 'mdn'
        full_loss_type = training_params.get('full_model_loss', training_params.get('loss', default_full_loss))
        full_huber_delta = float(training_params.get('full_model_huber_delta', 1.0))
        station_decorrelation_weight = float(training_params.get('station_embedding_decorrelation_weight', 0.0))
        if (not is_dist) or rank == 0:
            print(f'[tasks] training on {res_comps_cfg} with weights {res_weight_cfg.tolist()}')
            station_experiment_active = bool((training_params.get('station_experiment') or {}).get('enabled', False))
            if station_experiment_active and res_comps_cfg != ['pga']:
                print('[tasks] warning: station_experiment is active; set res_comps=["pga"] for PGA-only comparison experiments')
            print(f'[loss] full_model_loss={full_loss_type}, huber_delta={full_huber_delta}')
            if pga_target_norm_cfg is not None:
                print(
                    '[loss] pga_target_normalization enabled '
                    f'mean={pga_target_norm_cfg["mean"]:.6f}, std={pga_target_norm_cfg["std"]:.6f}'
                )
            if station_decorrelation_weight:
                print(f'[loss] station_embedding_decorrelation_weight={station_decorrelation_weight:g}')
            print(
                f'[lr] ReduceLROnPlateau monitors {lr_monitor} loss '
                f'(patience={patience}, factor={lr_decay_factor:g}, min_lr={min_lr:g})'
            )
        train_model(
            full_model,
            train_loader,
            val_loader,
            optimizer,
            lr_decay,
            num_epochs=training_params['epochs_full_model'],
            clipnorm=training_params.get('clipnorm', None),
            is_dist=is_dist,
            rank=rank,
            save_name='full_model',
            res_comps=res_comps_cfg,
            res_weight=res_weight_cfg,
            post_train_sanity=True,
            epoch_sanity=training_params.get('epoch_sanity', False),
            train_sampler=train_sampler,
            lr_monitor=lr_monitor,
            input_dump_config=input_dump_config,
            loss_type=full_loss_type,
            huber_delta=full_huber_delta,
            checkpoint_params=training_params.get('checkpoint', None),
            pga_target_normalization=pga_target_norm_cfg,
            station_decorrelation_weight=station_decorrelation_weight,
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
