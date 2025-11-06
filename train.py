import numpy as np
import sys
sys.path.append('./diting')
import yaml
import random
import h5py
import os
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
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import gemini_util as util
import loader
import gemini_models as models

from diting.downstream.gemini_utils import get_args as get_args_diting

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, clipnorm=None, is_dist=False, rank=0, save_name=None):
    tb_path = f'runs/{save_name}'
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
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            if isinstance(inputs, list): 
                inputs, labels = [i.to(device) for i in inputs], [l.to(device) for l in labels]
            else:
                inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            if isinstance(inputs, list): 
                outputs = model(*inputs)
            else:
                outputs = model(inputs)
            loss = models.mixture_density_loss(outputs, labels)
            loss.backward()
            if (not is_dist) or (is_dist and (rank == 0)):
                writer.add_scalar('train/loss', loss.item(), global_step)
                step_in_ep = global_step - steps_per_epoch * epoch
                print(f'Step/Epoch {step_in_ep}/{epoch}, Loss: {loss.item():.4f}')
            if clipnorm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clipnorm)
            optimizer.step()
            running_loss += loss.item()
            if global_step % 100 == 0:
                if (not is_dist) or (is_dist and (rank == 0)):
                    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
            global_step += 1
        epoch_loss = running_loss/len(train_loader)
        if steps_per_epoch == 0:
            steps_per_epoch = global_step
        if (not is_dist) or (is_dist and (rank == 0)):
            writer.add_scalar('train/epoch_loss',epoch_loss, epoch)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # Validation step
        if (not is_dist) or (is_dist and (rank == 0)):
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    if isinstance(inputs, list): 
                        inputs, labels = [i.to(device) for i in inputs], [l.to(device) for l in labels]
                    else:
                        inputs, labels = inputs.to(device), labels.to(device)
                    if isinstance(inputs, list): 
                        outputs = model(*inputs)
                    else:
                        outputs = model(inputs)
                    loss = models.mixture_density_loss(outputs, labels)
                    val_running_loss += loss.item()

            val_loss = val_running_loss / len(val_loader)
            writer.add_scalar('val/epoch_loss',val_loss, epoch)
            print(f'Validation Loss: {val_loss:.4f}')
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            # Save checkpoint
            filepath = os.path.join(training_params['weight_path'], f'{save_name}_{epoch+1}.pth')
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save({
                'epoch': epoch+1,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
            }, filepath)
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
        target_object = weights_path if os.path.isfile(weights_path) else os.path.join(weights_path, "hist.pkl")
        while not os.path.exists(target_object):
            print(f"File {target_object} for weight transfer missing. Sleeping {sleeptime}s")
            time.sleep(sleeptime)

    # 如果是目录，取最新 event- 文件
    if os.path.isdir(weights_path):
        last_weight = sorted([x for x in os.listdir(weights_path) if x.startswith("event-")])[-1]
        weights_path = os.path.join(weights_path, last_weight)

    # 加载 checkpoint
    state_dict = torch.load(weights_path, map_location=device)

    # 判断目标模型是不是 borehole (Conv1d 输入通道是否 64)
    conv1d_layer = None
    conv1d_name = None
    for name, module in model.named_modules():
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
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--diting_config', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--test_run', action='store_true')  # Test run with less data
    parser.add_argument('--no_multiprocessing', action='store_true')  # Prevents certain deadlocks
    parser.add_argument('--continue_ensemble', action='store_true')  # Continues a stopped ensemble training
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))
    set_seed(config.get('seed', 42))

    is_dist, rank, world_size, local_rank = util.setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
# args for diting model
    diting_args, ds_init = get_args_diting()
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

    if not os.path.isdir(training_params['weight_path']):
        if (not is_dist) or (is_dist and (rank == 0)):
            os.makedirs(training_params['weight_path'], exist_ok=True)
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

    full_data_train = [loader.load_events(data_path, limit=limit,
                                          parts=(True, False, False),
                                          shuffle_train_dev=generator.get('shuffle_train_dev', False),
                                          custom_split=generator.get('custom_split', None),
                                          min_mag=generator.get('min_mag', None),
                                          mag_key=generator.get('key', 'MA'),
                                          overwrite_sampling_rate=overwrite_sampling_rate,
                                          decimate_events=generator.get('decimate_events', None))
                       for data_path, generator in zip(training_params['data_path'], generator_params)]
    full_data_dev = [loader.load_events(data_path, limit=limit,
                                        parts=(False, True, False),
                                        shuffle_train_dev=generator.get('shuffle_train_dev', False),
                                        custom_split=generator.get('custom_split', None),
                                        min_mag=generator.get('min_mag', None),
                                        mag_key=generator.get('key', 'MA'),
                                        overwrite_sampling_rate=overwrite_sampling_rate,
                                        decimate_events=generator.get('decimate_events', None))
                     for data_path, generator in zip(training_params['data_path'], generator_params)]

    event_metadata_train = [d[0] for d in full_data_train]
    data_train = [d[1] for d in full_data_train]
    metadata_train = [d[2] for d in full_data_train]
    event_metadata_dev = [d[0] for d in full_data_dev]
    data_dev = [d[1] for d in full_data_dev]
    metadata_dev = [d[2] for d in full_data_dev]

    sampling_rate = metadata_train[0]['sampling_rate']
    assert all(m['sampling_rate'] == sampling_rate for m in metadata_train + metadata_dev)
    waveforms = data_train[0]['waveforms']

    max_stations = config['model_params']['max_stations']

    config['model_params']['n_datasets'] = len(data_train)

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
                hist_path = os.path.join(training_params['weight_path'], 'hist.pkl')
                if os.path.isfile(hist_path):
                    continue
                else:
                    raise ValueError(f'Can not continue unclean ensemble. Checking for {hist_path} failed.')

            if not os.path.isdir(training_params['weight_path']):
                if (not is_dist) or (is_dist and (rank == 0)):
                    os.makedirs(training_params['weight_path'], exist_ok=True)

            with open(os.path.join(training_params['weight_path'], 'config.json'), 'w') as f:
                json.dump(config, f, indent=4)

        print('Building model')
        single_station_model, full_model = models.build_transformer_model(**config['model_params'],
                                                                          trace_length=data_train[0]['waveforms'][0].shape[1],
                                                                          diting_args=diting_args
)
        if is_dist:
            single_station_model = DDP(single_station_model, device_ids=[local_rank],output_device=local_rank)
            full_model = DDP(full_model, device_ids=[local_rank],output_device=local_rank)
        else:
            single_station_model.to(device)
            full_model.to(device)

        if 'single_station_model_path' in training_params:
            print('Loading single station model')
            checkpoint = torch.load(training_params['single_station_model_path'])
            single_station_model.load_state_dict(checkpoint["model_state_dict"]) # need modify with save state dict.
        elif 'transfer_model_path' not in training_params:
            optimizer = optim.Adam(single_station_model.parameters(),lr=training_params['lr'])
            key = generator_params[0]['key']
            filter_single_station_by_pick = training_params.get('filter_single_station_by_pick', False)

            x_train = np.concatenate(data_train[0]['waveforms'], axis=0)
            x_dev = np.concatenate(data_dev[0]['waveforms'], axis=0)
            y_train = np.concatenate([np.full(x.shape[0], mag) for x, mag in
                                      zip(data_train[0]['waveforms'], event_metadata_train[0][key])])
            y_dev = np.concatenate([np.full(x.shape[0], mag) for x, mag in
                                    zip(data_dev[0]['waveforms'], event_metadata_dev[0][key])])

            train_mask = (x_train != 0).any(axis=(1, 2))
            dev_mask = (x_dev != 0).any(axis=(1, 2))
            if filter_single_station_by_pick:
                picks_train = np.concatenate(data_train[0]['p_picks'], axis=0)
                train_mask = np.logical_and(train_mask, picks_train < 3000)
                picks_dev = np.concatenate(data_dev[0]['p_picks'], axis=0)
                dev_mask = np.logical_and(dev_mask, picks_dev < 3000)
            x_train = x_train[train_mask]
            y_train = y_train[train_mask]
            x_dev = x_dev[dev_mask]
            y_dev = y_dev[dev_mask]

            noise_seconds = generator_params[0].get('noise_seconds', 5)
            cutout = (
                sampling_rate * (noise_seconds + generator_params[0]['cutout_start']), sampling_rate * (noise_seconds + generator_params[0]['cutout_end']))

            sliding_window = generator_params[0].get('sliding_window', False)

            train_dataset = util.DataGenerator(x_train, np.expand_dims(np.expand_dims(y_train, axis=1), axis=2),
                                                 cutout=cutout, label_smoothing=True, sliding_window=sliding_window)
            val_dataset = util.DataGenerator(x_dev, np.expand_dims(np.expand_dims(y_dev, axis=1), axis=2),
                                                 cutout=cutout, label_smoothing=True, sliding_window=sliding_window)
            if is_dist:
                train_sampler = DistributedSampler(train_dataset)
                train_loader = DataLoader(train_dataset, batch_size=generator_params[0]['batch_size'], sampler=train_sampler, shuffle=(train_sampler is None))
            else:
                train_loader = DataLoader(train_dataset, batch_size=generator_params[0]['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=generator_params[0]['batch_size'], shuffle=True)
            # Only save weights due to open issue:
            # https://github.com/matterport/Mask_RCNN/issues/308

            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=4, verbose=1)

            train_model(
                single_station_model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                num_epochs=training_params['epochs_single_station'],
                is_dist=is_dist,
                rank=local_rank,
                save_name='simple_model'
            )
            # Free memory
            del x_train
            del x_dev

        if 'load_model_path' in training_params:
            print('Loading full model')
            full_model.load_weights(training_params['load_model_path'])

        if 'transfer_model_path' in training_params:
            print('Transfering model weights')
            ensemble_load = training_params.get('ensemble_load', False)
            wait_for_load = training_params.get('wait_for_load', False)
            transfer_weights(full_model, training_params['transfer_model_path'],
                             ensemble_load=ensemble_load, wait_for_load=wait_for_load, ens_id=ens_id)

        def location_loss(y_true, y_pred):
            return models.mixture_density_loss(y_true, y_pred, eps=1e-5, d=3)

        no_event_token = config['model_params'].get('no_event_token', False)
        optimizer = optim.Adam(full_model.parameters(), lr=training_params['lr'])
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

            train_generators += [util.PreloadedEventGenerator(data=data_train[i],
                                                              event_metadata=event_metadata_train[i],
                                                              coords_target=True,
                                                              label_smoothing=True,
                                                              station_blinding=True,
                                                              cutout=cutout,
                                                              pga_targets=n_pga_targets,
                                                              max_stations=max_stations,
                                                              sampling_rate=sampling_rate,
                                                              no_event_token=no_event_token,
                                                              **generator_param_set)]

            old_oversample = generator_param_set.get('oversample', 1)
            generator_param_set['oversample'] = 4
            validation_generators += [util.PreloadedEventGenerator(data=data_dev[i],
                                                                   event_metadata=event_metadata_dev[i],
                                                                   coords_target=True,
                                                                   station_blinding=True,
                                                                   cutout=cutout,
                                                                   pga_targets=n_pga_targets,
                                                                   max_stations=max_stations,
                                                                   sampling_rate=sampling_rate,
                                                                   no_event_token=no_event_token,
                                                                   **generator_param_set)]
            generator_param_set['oversample'] = old_oversample

        if len(train_generators) == 0:
            train_dataset = train_generators[0]
            val_dataset = validation_generators[0]
        else:
            dataset_bias = config['model_params'].get('dataset_bias', False)
            train_dataset = util.JointGenerator(train_generators, shuffle=True, dataset_id=dataset_bias)
            val_dataset = util.JointGenerator(validation_generators, shuffle=True, dataset_id=dataset_bias)

        filepath = os.path.join(training_params['weight_path'], 'event-{epoch:02d}.hdf5')
        patience = training_params.get('lr_decay_patience', 6)
#        lr_decay = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=patience, factor=0.3, verbose=1) # need modify
        lr_decay = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=patience, verbose=1)
        logdir = os.path.join('logs/scalars/', training_params['weight_path'])

        if is_dist:
            train_sampler = DistributedSampler(train_dataset)
            train_loader = DataLoader(train_dataset, batch_size=generator_params[0]['batch_size'], sampler=train_sampler, shuffle=(train_sampler is None))
        else:
            train_loader = DataLoader(train_dataset, batch_size=generator_params[0]['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=generator_params[0]['batch_size'], shuffle=True)
        train_model(
            full_model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            num_epochs=training_params['epochs_full_model'],
            is_dist=is_dist,
            rank=local_rank,
            save_name='full_model'
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
