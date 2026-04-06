import numpy as np
import sys
sys.path.append('./diting')
import yaml
import random
import h5py
import os
import copy
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

from diting.downstream.gemini_utils import get_args as get_args_diting

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def subset_events(event_metadata, n):
    """Subset to the first n *events* (not rows). event_metadata is station-level."""
    if n is None:
        return event_metadata
    if hasattr(event_metadata, "columns"):
        for event_key in ['KiK_File', '#EventID', 'EVENT']:
            if event_key in event_metadata.columns:
                unique_events = event_metadata[event_key].unique()[:n]
                return event_metadata[event_metadata[event_key].isin(unique_events)].copy()
    # Fallback for non-DataFrame
    if hasattr(event_metadata, "iloc"):
        return event_metadata.iloc[:n].copy()
    return event_metadata[:n]

def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, clipnorm=None, is_dist=False, rank=0, save_name=None,
                res_comps=None, res_weight=None, post_train_sanity=False, epoch_sanity=False, train_sampler=None):
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
        for inputs, labels, _ in train_loader:
            if isinstance(inputs, list): 
                inputs, labels = [i.to(device) for i in inputs], [l.to(device) for l in labels]
            else:
                inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            if isinstance(inputs, list): 
                outputs = model(*inputs)
            else:
                outputs = model(inputs)
            loss = models.mixture_density_loss_full(outputs, labels, res_comps=res_comps, res_weight=res_weight)
            loss.backward()
            if (not is_dist) or (is_dist and (rank == 0)):
                writer.add_scalar('train/loss', loss.item(), global_step)
                step_in_ep = global_step - steps_per_epoch * epoch
                print(f'Step/Epoch {step_in_ep}/{epoch}, Loss: {loss.item():.4f}')
            if clipnorm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clipnorm)
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
                loss = models.mixture_density_loss_full(outputs, labels, res_comps=res_comps, res_weight=res_weight)
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
            scheduler.step(val_loss)
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
        checkpoint = dist.broadcast_object_list([checkpoint], src=0)[0]

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
                    station_has_signal = (wave != 0).any(dim=(2, 3))
                    signal_frac = (wave != 0).float().mean(dim=(1, 2, 3))
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
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--test_run', action='store_true')  # Test run with less data
    parser.add_argument('--overfit_n', type=int, default=0)  # Use the same tiny subset for train/val
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
    diting_args, ds_init = get_args_diting()
    diting_args.conf_file = args.diting_config
    with open(diting_args.conf_file, 'r') as f:
        diting_conf_data = yaml.safe_load(f)
    vars(diting_args).update(diting_conf_data)
    depth = 24
    if depth % diting_args.num_interactions != 0:
        diting_args.num_interactions -= 1
    n = (depth - 1)//diting_args.num_interactions
    diting_args.interaction_indexes = [[i*n, (i+1)*n] for i in range(diting_args.num_interactions)]
    if diting_args.num_interactions * n != depth:
        diting_args.interaction_indexes.append([diting_args.num_interactions*n, depth])

    diting_args.distributed = is_dist
    if diting_args.distributed:
        diting_args.device = device
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

    full_data_train = [loader.load_events(data_path, event_metadata_path='train_ev.csv',limit=limit,
                                          parts=(True, False, False),
                                          shuffle_train_dev=generator.get('shuffle_train_dev', False),
                                          custom_split=generator.get('custom_split', None),
                                          min_mag=generator.get('min_mag', None),
                                          mag_key=generator.get('key', 'MA'),
                                          overwrite_sampling_rate=overwrite_sampling_rate,
                                          decimate_events=generator.get('decimate_events', None))
                       for data_path, generator in zip(training_params['data_path'], generator_params)]
    full_data_dev = [loader.load_events(data_path, event_metadata_path='test_ev.csv',limit=limit,
                                        parts=(False, True, False),
                                        shuffle_train_dev=generator.get('shuffle_train_dev', False),
                                        custom_split=generator.get('custom_split', None),
                                        min_mag=generator.get('min_mag', None),
                                        mag_key=generator.get('key', 'MA'),
                                        overwrite_sampling_rate=overwrite_sampling_rate,
                                        decimate_events=generator.get('decimate_events', None))
                     for data_path, generator in zip(training_params['data_path'], generator_params)]

    event_metadata_train = [d[0] for d in full_data_train]
    metadata_train = [d[2] for d in full_data_train]
    event_metadata_dev = [d[0] for d in full_data_dev]
    metadata_dev = [d[2] for d in full_data_dev]

    if args.overfit_n > 0:
        event_metadata_train = [subset_events(meta, args.overfit_n) for meta in event_metadata_train]
        event_metadata_dev = [subset_events(meta, args.overfit_n) for meta in event_metadata_train]
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
        training_params['epochs_single_station'] = 0
        if (not is_dist) or (is_dist and (rank == 0)):
            print(f'Overfit mode enabled: using the first {args.overfit_n} samples for both train and val')
            print('Overfit mode adjustments: single-station pretraining disabled, trigger_based disabled, station foreshadowing enabled, oversample=1, fixed cutout, deterministic station selection, no train/dev shuffling')

    sampling_rate = metadata_train[0]['sampling_rate']
    assert all(m['sampling_rate'] == sampling_rate for m in metadata_train + metadata_dev)
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
        single_station_model, full_model = models.build_transformer_model(**config['model_params'],
#                                                                          trace_length=data_train[0]['waveforms'][0].shape[1],
                                                                          trace_length=10000,
                                                                          diting_args=diting_args
)
        if is_dist:
            single_station_model.to(device)
            full_model.to(device)
            single_station_model = DDP(
                single_station_model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )
            full_model = DDP(
                full_model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )
        else:
            single_station_model.to(device)
            full_model.to(device)

        if 'single_station_model_path' in training_params:
            print('Loading single station model')
            checkpoint = torch.load(training_params['single_station_model_path'], map_location=device)
            load_target = single_station_model.module if is_dist else single_station_model
            load_target.load_state_dict(checkpoint["model_state_dict"])
        elif 'transfer_model_path' not in training_params and not overfit_mode:
            optimizer = optim.Adam(single_station_model.parameters(),lr=training_params['lr'])
            key = generator_params[0]['key']
            filter_single_station_by_pick = training_params.get('filter_single_station_by_pick', False)
            # TODO: filter_single_station_by_pick is not yet implemented for DataGenerator-based loading.
            # In train.py it filters stations with p_pick >= 3000 from the concatenated arrays.

            noise_seconds = generator_params[0].get('noise_seconds', 5)
            cutout = (
                sampling_rate * (noise_seconds + generator_params[0]['cutout_start']), sampling_rate * (noise_seconds + generator_params[0]['cutout_end']))

            sliding_window = generator_params[0].get('sliding_window', False)

            train_dataset = util.DataGenerator(event_metadata_train[0], metadata_train[0], training_params['data_path'][0], generator_params[0],
                                                 cutout=cutout, label_smoothing=False if overfit_mode else True, sliding_window=sliding_window)
            val_dataset = util.DataGenerator(event_metadata_dev[0], metadata_dev[0], training_params['data_path'][0], generator_params[0],
                                                 cutout=cutout, label_smoothing=False if overfit_mode else True, sliding_window=sliding_window)
            if is_dist:
                train_sampler = DistributedSampler(train_dataset)
                train_loader = DataLoader(train_dataset, batch_size=generator_params[0]['batch_size'], sampler=train_sampler, shuffle=(train_sampler is None))
                val_sampler = DistributedSampler(val_dataset, shuffle=False)
            else:
                train_sampler = None
                val_sampler = None
                train_loader = DataLoader(train_dataset, batch_size=generator_params[0]['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=generator_params[0]['batch_size'], sampler=val_sampler, shuffle=(val_sampler is None))
            # Only save weights due to open issue:
            # https://github.com/matterport/Mask_RCNN/issues/308

            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=4, verbose=1)

            if ((not is_dist) or local_rank == 0):
                run_sanity_check(single_station_model, train_loader, device, name='single_station_train_pre')

            train_model(
                single_station_model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                num_epochs=training_params['epochs_single_station'],
                clipnorm=training_params.get('clipnorm', None),
                is_dist=is_dist,
                rank=local_rank,
                save_name='simple_model',
                res_comps=['mag'],
                res_weight=np.array([1.]),
                post_train_sanity=True,
                epoch_sanity=training_params.get('epoch_sanity', False),
                train_sampler=train_sampler
            )
            # Free memory

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

        def location_loss(y_true, y_pred):
            return models.mixture_density_loss(y_true, y_pred, eps=1e-5, d=3)

        # Freeze only the diting encoder (ViTAdapter), unfreeze EncoderFeatures head + dt2team
        raw_full = full_model.module if is_dist else full_model
        for param in raw_full.waveform_model[0].parameters():
            param.requires_grad = False
        # waveform_model[1] (EncoderFeatures) and dt2team remain trainable

        # Re-initialize all of EncoderFeatures (FPN + bottleneck + task head).
        # Only the diting encoder (ViTAdapter) keeps pretrained weights.
        encoder_features = raw_full.waveform_model[1]
        for name, module in encoder_features.named_modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif hasattr(module, 'weight') and hasattr(module, 'bias') and isinstance(module.weight, nn.Parameter):
                # LayerNorm and similar
                nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        # Also re-initialize dt2team MLP for consistency
        dt2team = raw_full.waveform_model.dt2team
        for name, module in dt2team.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        if rank == 0:
            print('Re-initialized EncoderFeatures (FPN, bottleneck, task head) and dt2team')

        no_event_token = config['model_params'].get('no_event_token', False)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, full_model.parameters()), lr=training_params['lr'])
        if not no_event_token:
            losses = {'magnitude': models.mixture_density_loss, 'location': location_loss}
        else:
            losses = {}

        n_pga_targets = config['model_params'].get('n_pga_targets', 0)

        if n_pga_targets:

            def pga_loss(y_true, y_pred):
                return models.time_distributed_loss(y_true, y_pred, models.mixture_density_loss, mean=True,
                                                    kwloss={'mean': False})

            losses['pga'] = pga_loss


        train_generators = []
        validation_generators = []

        for i, generator_param_set in enumerate(generator_params):
            noise_seconds = generator_param_set.get('noise_seconds', 5)
            cutout = (
                sampling_rate * (noise_seconds + generator_param_set['cutout_start']), sampling_rate * (noise_seconds + generator_param_set['cutout_end']))

            generator_param_set['transform_target_only'] = generator_param_set.get('transform_target_only', True)

            train_generators += [util.PreloadedEventGenerator(event_metadata=event_metadata_train[i],
                                                              metadata=metadata_train[i],
                                                              data_path=training_params['data_path'][i],
                                                              generator_params=generator_params[i],
                                                              coords_target=True,
                                                              label_smoothing=False if overfit_mode else True,
                                                              station_blinding=False,
                                                              cutout=cutout,
                                                              pga_targets=n_pga_targets,
                                                              max_stations=max_stations,
                                                              sampling_rate=sampling_rate,
                                                              no_event_token=no_event_token,
                                                              **generator_param_set)]

            old_oversample = generator_param_set.get('oversample', 1)
            generator_param_set['oversample'] = 1 if overfit_mode else 4
            validation_generators += [util.PreloadedEventGenerator(event_metadata=event_metadata_dev[i],
                                                                   metadata=metadata_dev[i],
                                                                   data_path=training_params['data_path'][i],
                                                                   generator_params=generator_params[i],
                                                                   coords_target=True,
                                                                   station_blinding=False,
                                                                   cutout=cutout,
                                                                   pga_targets=n_pga_targets,
                                                                   max_stations=max_stations,
                                                                   sampling_rate=sampling_rate,
                                                                   no_event_token=no_event_token,
                                                                   **generator_param_set)]
            generator_param_set['oversample'] = old_oversample

        if len(train_generators) == 1:
            train_dataset = train_generators[0]
            val_dataset = validation_generators[0]
        else:
            dataset_bias = config['model_params'].get('dataset_bias', False)
            train_dataset = util.JointGenerator(train_generators, shuffle=True, dataset_id=dataset_bias)
            val_dataset = util.JointGenerator(validation_generators, shuffle=True, dataset_id=dataset_bias)

        # filepath variable kept for reference; actual saving uses save_name in train_model
        filepath = os.path.join(training_params['weight_path'], 'full_model_{epoch}.pth')
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
	    res_comps=['mag','loc','pga'],
	    res_weight=np.array([1.,1.,1.]),
            post_train_sanity=True,
            epoch_sanity=training_params.get('epoch_sanity', False),
            train_sampler=train_sampler
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
