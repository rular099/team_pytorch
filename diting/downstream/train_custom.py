import datetime
import math
import os
from typing import Union
from contextlib import suppress

import numpy as np
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from finetuneing.config import Config
from finetuneing.models import load_checkpoint, save_checkpoint
from finetuneing.utils import *
import finetuneing.utils.help_builder as help_builder
from optim import get_all_parameters

from finetuneing.training.lars import LARS
import LSD.models.backbone_ablation as backbone_ablation

from mup import set_base_shapes

from finetuneing.datasets.SFTData import get_loss
from finetuneing.models.loss import multi_task_loss_fn
from torch.utils.data.dataloader import default_collate
import torch.distributed as dist

def custom_collate_fn(batch):
    def to_tensor(item):
        if isinstance(item, np.ndarray):
            return torch.as_tensor(item, dtype=torch.float32)
        elif isinstance(item, (list, tuple)):
            return type(item)(to_tensor(i) for i in item)
        elif isinstance(item, dict):
            return {k: to_tensor(v) for k, v in item.items()}
        else:
            return item

    batch = [to_tensor(item) for item in batch]
    return default_collate(batch)

def train(
    args,
    tasks,
    label_names,
    model,
    optimizer,
    scheduler,
    train_loader,
    epoch,
    device,
    tensor_writer,
    checkpoint_save_dir,
    best_loss,
    process_outputs,
) -> Union[list, dict]:
    '''
        # tasks: [["det", "ppk", "spk"], ["dis"], ["emg"]] or ["det", "ppk", "spk"]
        # label_names: [[["det", "ppk", "spk"]], ["dis"], ["emg"]] or [["det", "ppk", "spk"]]
    '''
    autocast = get_autocast(args.precision)
    model.train()

    # Save and display metrics
    train_loss_per_step = []
    average_meters = {}
    metrics_merged = {}
    sampling_rate = train_loader.dataset.sampling_rate()

    if len(label_names) > 1:
        for subtasks in tasks:
            for task in subtasks:
                metrics = Metrics(
                    task=task,
                    metric_names=Config.get_metrics(task),
                    sampling_rate=sampling_rate,
                    time_threshold=args.time_threshold,
                    num_samples=args.in_samples,
                    device=device,
                )
                metrics_merged[f"{task}"] = metrics
                for metric in metrics.metric_names():
                    average_meters[f"{task}_{metric}"] = AverageMeter(
                        f"[{task.upper()}]{metric}", ":6.4f"
                    )
            
    else: # det_ppk_spk_dis_emg
        for task in tasks:
            metrics = Metrics(
                task=task,
                metric_names=Config.get_metrics(task),
                sampling_rate=sampling_rate,
                time_threshold=args.time_threshold,
                num_samples=args.in_samples,
                device=device,
            )
            metrics_merged[f"{task}"] = metrics
            for metric in metrics.metric_names():
                average_meters[f"{task}_{metric}"] = AverageMeter(
                    f"[{task.upper()}]{metric}", ":6.4f"
                )
                
                            
            average_meters[f"{task}_loss"] = AverageMeter(f"{task}_loss", ":6.4f")

    average_meters["loss"] = AverageMeter("Loss", ":6.4f")
    average_meters["Time"] = AverageMeter("Time", ":6.4f")
    average_meters["Data Time"] = AverageMeter("Data Time", ":6.4f")
    if args.splitPS:
        average_meters["Pn_index_loss"] = AverageMeter("Pn_index_loss", ":6.4f")
        average_meters["Pg_index_loss"] = AverageMeter("Pg_index_loss", ":6.4f")
        average_meters["Sn_index_loss"] = AverageMeter("Sn_index_loss", ":6.4f")
        average_meters["Sg_index_loss"] = AverageMeter("Sg_index_loss", ":6.4f")
        average_meters["det_loss"] = AverageMeter("det_loss", ":6.4f")

    progress = ProgressMeter(
        len(train_loader),
        [m for m in average_meters.values()],
        prefix=f"Train: [{epoch}/{args.epochs}]",
    )

    tgts_trans_for_loss = None
    outs_trans_for_loss = None
    outs_trans_for_res = None
    end = time.time()
    iters_per_epoch = len(train_loader)
    for step, (x, loss_targets, metrics_targets, meta_data_jsons) in enumerate(train_loader):
        # [Debug]
        # if step > 200:
        #     break
        average_meters["Data Time"].update(time.time() - end)
        if isinstance(x, (list, tuple)):
            x = [xi.to(device) for xi in x]
        else:
            x = x.to(device)

        if isinstance(loss_targets, (list, tuple)):
            loss_targets = [yi.to(device) for yi in loss_targets]
        elif isinstance(loss_targets, dict):
            loss_targets = {k: v.to(device) for k, v in loss_targets.items()}
        else:
            loss_targets = loss_targets.to(device)

        if args.enable_deepspeed:
            model.zero_grad()
            # model.micro_steps = 0
        else:
            optimizer.zero_grad()

        # Forward
        with autocast():
            outputs = model(x)

            # Loss
            outputs_for_loss = (
                outs_trans_for_loss(outputs) if outs_trans_for_loss is not None else outputs
            )
            loss_targets = (
                tgts_trans_for_loss(loss_targets)
                if tgts_trans_for_loss is not None
                else loss_targets
            )

            # add by lhl            
            if isinstance(loss_targets, dict): # det_ppk_spk_dis_emg
                loss,loss_log_dict = multi_task_loss_fn(outputs_for_loss,loss_targets,metrics_targets,args.model_tasks,args.task_loss_weight)
                
                for key in loss_log_dict.keys():
                    average_meters[f"{task}_loss"].update(loss_log_dict[key])
            
            else:
                raise NotImplementedError

        if is_main_process() and args.visualize and step % 100000 == 0 and 'ppk' in label_names[0]:
            # Only applicable to phase-picking task.
            vis_waves_preds_targets(x[0].detach().cpu().numpy(),
                                    outputs[0].detach().cpu().numpy(),
                                    loss_targets[0].detach().cpu().numpy(),
                                    sampling_rate,
                                    args.visualize_save_dir,
                                    step_epoch=(step,epoch))

        # Backward
        if args.enable_deepspeed:
            model.backward(loss)
            model.step()
        else: # TODO support scalar
            loss.backward()
            optimizer.step()
        
        # Adjust learning rate
        if scheduler is not None:
            current_step = iters_per_epoch * epoch + step
            scheduler(current_step)
            lr = optimizer.param_groups[0]['lr']
        else:
            lr = optimizer.param_groups[0]["lr"]

        # Batch size of the step
        if isinstance(x, (list, tuple)):
            step_batch_size = x[0].size(0)
        else:
            step_batch_size = x.size(0)

        # Save loss
        average_meters["loss"].update(loss.item(), step_batch_size)
        train_loss_per_step.append(loss.item())

        # Process outputs
        if not isinstance(outputs, (dict, tuple, list)):
            outputs = outputs.to(torch.float32)
        elif isinstance(outputs, dict):
            for k in outputs.keys():
                outputs[k] = outputs[k].to(torch.float32)
        else:
            for i in range(len(outputs)):
                outputs[i] = outputs[i].to(torch.float32)
        outputs_for_metrics = (
            outs_trans_for_res(outputs) if outs_trans_for_res is not None else outputs
        )
        results = process_outputs(args, outputs_for_metrics, label_names, sampling_rate)

        # Calculate metrics
        tasks_metrics = {}
        if len(label_names) > 1:
            for i, subtasks in enumerate(tasks):
                for task in subtasks:
                    metrics = Metrics(
                        task=task,
                        metric_names=Config.get_metrics(task),
                        sampling_rate=sampling_rate,
                        time_threshold=args.time_threshold,
                        num_samples=args.in_samples,
                        device=device,
                    )
                    tasks_metrics[task] = metrics
                    metrics.compute(
                        targets=metrics_targets[i][task],
                        preds=results[task],
                        reduce=is_dist_avail_and_initialized(),
                    )
                    for metric in metrics.metric_names():
                        average_meters[f"{task}_{metric}"].update(
                            metrics.get_metric(name=metric), step_batch_size
                        )
                    metrics_merged[f"{task}"].add(metrics)
        else:
            for task in tasks:
                metrics = Metrics(
                    task=task,
                    metric_names=Config.get_metrics(task),
                    sampling_rate=sampling_rate,
                    time_threshold=args.time_threshold,
                    num_samples=args.in_samples,
                    device=device,
                )
                tasks_metrics[task] = metrics
                metrics.compute(
                    targets=metrics_targets[task],
                    preds=results[task],
                    reduce=is_dist_avail_and_initialized(),
                )
                for metric in metrics.metric_names():
                    average_meters[f"{task}_{metric}"].update(
                        metrics.get_metric(name=metric), step_batch_size
                    )
                metrics_merged[f"{task}"].add(metrics)

        # Tensorboard
        if tensor_writer is not None and is_main_process():
            gstep = epoch * len(train_loader) + step
            tensor_writer.add_scalar("learning-rate/step", lr, gstep)
            tensor_writer.add_scalar("train-loss/step", loss.item(), gstep)
            tensor_writer.add_scalar("train-time/step", time.time()-end, gstep)
            if len(label_names) > 1:
                for subtasks in tasks:
                    for task in subtasks:
                        values = tasks_metrics[task].get_all_metrics()
                        tensor_writer.add_scalars(f"train.{task}.metrics/step", values, gstep)
            else:
                for task in tasks:
                    values = tasks_metrics[task].get_all_metrics()
                    tensor_writer.add_scalars(f"train.{task}.metrics/step", values, gstep)
        average_meters["Time"].update(time.time() - end)
        end = time.time()

        if step % args.log_step == 0 and is_main_process():
            prg_str = progress.get_str(batch_idx=step, name=f"{args.model_name}_train")
            print(prg_str)

        # save ckpt per 5000 steps
        # if (step+1) % 5000 == 0:
        #     if args.enable_deepspeed:
        #         client_state = {'epoch': epoch, 'step': step}
        #         model.save_checkpoint(
        #             save_dir=checkpoint_save_dir, 
        #             tag=f"checkpoint_pt_epoch_{str(epoch)}_iter_{str(step)}", 
        #             client_state=client_state
        #         )
        #     else: # TODO support scalar saved
        #         if is_main_process():
        #             ckpt_path = os.path.join(checkpoint_save_dir, f"model-{epoch}-{step}-latest.pth")
        #             save_checkpoint(ckpt_path, epoch, model, optimizer, best_loss)
        #             print(f"Model saved: {ckpt_path}")

    return train_loss_per_step, metrics_merged


def train_worker(args, device, ds_init, train_dataset, CustomHead, process_outputs) -> str:

    checkpoint_save_dir = os.path.join(args.log_dir, "checkpoints")
    tb_dir = os.path.join(args.log_dir, "tensorboard")
    args.visualize_save_dir = os.path.join(args.log_dir, "train_visualize")
    
    tensor_writer = SummaryWriter(tb_dir) if args.use_tensorboard else None

    if is_main_process():
        with open(os.path.join(args.log_dir, f"run_tb_{get_time_str()}.sh"), "w") as f:
            f.write(f"tensorboard --logdir '{tb_dir}' --port 8080")
        if not os.path.exists(checkpoint_save_dir):
            os.makedirs(checkpoint_save_dir)
        if not os.path.exists(args.visualize_save_dir) and args.visualize:
            os.makedirs(args.visualize_save_dir)

    # Data loader
    model_inputs = [['z', 'n', 'e']]
    model_labels, model_tasks = args.model_labels, args.model_tasks

    print(f"train size: {len(train_dataset)}")

    train_sampler = (
        torch.utils.data.DistributedSampler(train_dataset)
        if is_dist_avail_and_initialized()
        else None
    )

    # Train DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=((not is_dist_avail_and_initialized()) and args.shuffle),
        pin_memory=args.pin_memory,
        num_workers=args.workers,
        sampler=train_sampler,
        collate_fn=custom_collate_fn
    )

    # Model (todo) enable mup init
    base_encoder_size_dict = backbone_ablation.get_encoder_size_dict(width=args.base_width, depth=24) # args.encoder_size
    base_model = help_builder.create_model_custom(
        model_name=args.model_name,
        downstream_tasks=args.downstream_task,
        in_samples=args.in_samples,
        encoder_size=base_encoder_size_dict,
        eval_type=args.eval_type,
        pool_type=args.pool_type,
        CustomHead=CustomHead,
        args=args,
    )
    target_encoder_size_dict = backbone_ablation.get_encoder_size_dict(width=args.target_width, depth=24)
    model = help_builder.create_model_custom(
        model_name=args.model_name,
        downstream_tasks=args.downstream_task,
        in_samples=args.in_samples,
        encoder_size=target_encoder_size_dict,
        eval_type=args.eval_type,
        pool_type=args.pool_type,
        CustomHead=CustomHead,
        args=args,
    )
    print(model)
    ### muP: set base_shapes
    set_base_shapes(model, base_model) # do_assert=False

    if not os.path.exists(args.resume) and args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=False)

            # rename moco pre-trained keys
            print('############# ckpt keys', checkpoint.keys())
            if args.pretrain_method == "mae":
                if args.pretrained.endswith('.pt'):
                    key = 'module'
                else:
                    key = 'state_dict'
                state_dict = checkpoint[key]
                # print(f'############# {key} has ', state_dict.keys())
                for k in list(state_dict.keys()):
                    # from deepspeed
                    if k.startswith("base_encoder."):
                        # remove prefix
                        state_dict['0.'+k[len("base_encoder.") :]] = state_dict[k] # for MAE:0. is mapped to encoder(in func:help_builder.create_model)
                    elif k.startswith("module.base_encoder."):
                        # remove prefix
                        state_dict['0.'+k[len("module.base_encoder.") :]] = state_dict[k] # for MAE:0. is mapped to encoder(in func:help_builder.create_model)
                    if args.dpk_head == 'vit_adapter_decoder_new':
                        if k.startswith("base_decoder."):
                            state_dict['1.decoder.'+k[len("base_decoder.") :]] = state_dict[k]
                        elif k.startswith("module.base_decoder."): 
                            state_dict['1.decoder.'+k[len("module.base_decoder.") :]] = state_dict[k]
                    elif 'decoder' in args.dpk_head or args.dpk_head == 'vit_adapter_TaskSeparatedUPerHead':
                        if k.startswith("base_decoder."):
                            state_dict['0.decoder.'+k[len("base_decoder.") :]] = state_dict[k]
                        elif k.startswith("module.base_decoder."): 
                            state_dict['0.decoder.'+k[len("module.base_decoder.") :]] = state_dict[k]
                    
                    del state_dict[k]
            elif args.pretrain_method == "lp":
                if args.pretrained.endswith('.pt'):
                    key = 'module'
                else:
                    key = 'model_dict'
                state_dict = checkpoint[key]
                del checkpoint["optimizer_dict"]
            else:
                raise NotImplementedError(f"Unsupported pretrain method:'{args.pretrain_method}'")
            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)
            
            if args.pool_type == 'cls':
                assert msg.missing_keys == ['2.fc.weight', '2.fc.bias'],"load pretrain model fail!"
            elif args.pool_type == 'avg' or args.pool_type == 'attentive':
                missing_keys_except_attentive = [k for k in msg.missing_keys if not k.startswith('1.')]
                assert missing_keys_except_attentive == ['3.fc.weight', '3.fc.bias'],"load pretrain model fail!"
            
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))
            assert os.path.isfile(args.pretrained),"no checkpoint found at '{}'".format(args.pretrained)

    model = model.to(device)

    # https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_finetune.py#L267
    eff_batch_size = args.batch_size * misc.get_world_size()
    args.lr = args.base_lr # * eff_batch_size / 256
    # print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)
    # Optimizer
    print('===> Constructing criterion and optimizer....')
    optimizer_params = get_all_parameters(args, model)
    optim_lower = args.optim.lower()
    if not args.enable_deepspeed:
        if optim_lower == "adam":
            optimizer = torch.optim.Adam(
                optimizer_params,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        elif optim_lower == "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_params,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        elif optim_lower == "sgd":
            optimizer = torch.optim.SGD(
                optimizer_params,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
        # add lars
        elif optim_lower == "lars":
            optimizer = LARS(
                params=optimizer_params,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer:'{args.optim}'")
    else:
        if optim_lower != "adamw":
            raise NotImplementedError("Only adamw is supported.")
        else:
            model, optimizer, _, _ = ds_init(
                args=args,
                model=model,
                model_parameters=optimizer_params,
                dist_init_required=not args.distributed,
            )

    # Load checkpoint (resume training)
    if args.resume:
        if args.enable_deepspeed: # TODO correct resume for deepspeed
            if os.path.exists(args.resume):
                latest_ckpt = -1
                import glob
                all_checkpoints = glob.glob(os.path.join(args.resume, 'checkpoint_pt_epoch_*'))
                for ckpt in all_checkpoints:
                    t = ckpt.split('/')[-1].split('_')[-1]
                    if t.isdigit():
                        latest_ckpt = max(int(t), latest_ckpt)
                
                if latest_ckpt >= 0:
                    args.start_epoch = latest_ckpt + 1
                    model.load_checkpoint(
                        args.resume, tag='checkpoint_pt_epoch_%d' % latest_ckpt
                    ) #tag=f"epoch_{completed_epoch}"
                    print(f"=> resuming checkpoint '{args.resume}' (epoch {latest_ckpt})")
                else:
                    print("=> no checkpoint found at '{}'".format(args.resume))
            else:
                print("=> '{}' is not existing!".format(args.resume))
            checkpoint = None
        else:
            checkpoint = load_checkpoint(
                args.resume,
                device=device,
                dist_mode=args.distributed,
                compile_mode=args.use_torch_compile,
                resume=True,
            )
            if checkpoint is not None: 
                if "epoch" in checkpoint:
                    args.start_epoch = checkpoint["epoch"] + 1
                if "model_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_dict"])
                if "optimizer_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_dict"])
                    print(f"optimizer.load_state_dict")
        print(f"model.load_state_dict")
        print(f"Model loaded: {args.resume}")
    else:
        checkpoint = None

    print('===> DDP preparing....')
    print('is dist:',dist.is_available())
    print('is init:',dist.is_initialized())
    if is_dist_avail_and_initialized():
        if not args.enable_deepspeed:
            local_rank = get_local_rank()
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                find_unused_parameters=args.find_unused_parameters,
            )
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # Epochs & Steps
    if args.steps > 0:
        args.epochs = math.ceil(args.steps / len(train_loader))
    args.steps = args.epochs * len(train_loader)
    print(f"`args.epochs` -> {args.epochs}, `args.steps` -> {args.steps}")
    
    # Loss
    best_loss = (
        float("inf")
        if (checkpoint is None or "loss" not in checkpoint)
        else checkpoint["loss"]
    ) # todo: add best_loss to checkpoint
    
    # Initialize GradScaler
    # scaler = GradScaler()
    # create scheduler if train
    if args.use_lr_scheduler:
        args.warmup = int(args.steps * args.warmup_steps) # default: 2%
        args.drop_step = int(args.steps * args.down_steps) # default: 10%
        if args.lr_scheduler == 'cosine':
            scheduler = warmup_cosine_lr(optimizer, args, args.steps)
        elif args.lr_scheduler == 'wsd':
            scheduler = warmup_stable_lr(optimizer, args, args.steps)
        else:
            raise NotImplementedError(f'{args.lr_schedule} is not supported, choose from [cosine, wsd]')
    else:
        scheduler = None

    # Save loss
    losses_dict = {
        n: []
        for n in ["train_loss_per_step", "train_loss_per_epoch"]
        # for n in ["train_loss_per_step", "train_loss_per_epoch", "val_loss_per_epoch"]
    }
    # num_saved = 0
    # epochs_since_improvement = 0
    if args.start_epoch == args.epochs:
        last_epoch = args.epochs - 1
        if args.enable_deepspeed:
            ckpt_path = os.path.join(checkpoint_save_dir, f"checkpoint_pt_epoch_{last_epoch}/mp_rank_00_model_states.pt")
        else:
            ckpt_path = os.path.join(checkpoint_save_dir, f"model-{last_epoch}-latest.pth")
    else:
        ckpt_path = None
    
    cost_time = datetime.timedelta()
    for i, epoch in enumerate(range(args.start_epoch, args.epochs)):
        # [Debug]
        # if i > 0:
        #     break
            
        epoch_start_time = datetime.datetime.now()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch=epoch)

        # Train
        train_losses, train_metrics_dict = train(
            args,
            model_tasks,
            model_labels,
            model,
            optimizer,
            scheduler,
            train_loader,
            epoch,
            device,
            tensor_writer,
            checkpoint_save_dir,
            best_loss,
            process_outputs
        )
        train_loss = np.mean(train_losses)
        losses_dict["train_loss_per_step"].extend(train_losses)
        losses_dict["train_loss_per_epoch"].append(train_loss)

        # Validate
        # val_loss, val_metrics_dict = validate(
        #     args, model_tasks, model, loss_fn, val_loader, epoch, device
        # )
        # losses_dict["val_loss_per_epoch"].append(val_loss)

        if args.enable_deepspeed:
            client_state = {'epoch': epoch}
            # client_state = {'epoch': epoch, 'loss': val_loss}
            model.save_checkpoint(
                save_dir=checkpoint_save_dir, 
                tag="checkpoint_pt_epoch_%s" % str(epoch), 
                client_state=client_state
            )
            ckpt_path = os.path.join(checkpoint_save_dir, f"checkpoint_pt_epoch_{epoch}/mp_rank_00_model_states.pt")
        else: # TODO support scalar saved
            if is_main_process():
                ckpt_path = os.path.join(checkpoint_save_dir, f"model-{epoch}-latest.pth")
                save_checkpoint(ckpt_path, epoch, model, optimizer, best_loss)
                print(f"Model saved: {ckpt_path}")
        
        # Save best model
        '''
        if val_loss < best_loss:
            best_loss = val_loss
            if args.enable_deepspeed:
                ckpt_path = os.path.join(checkpoint_save_dir, f"checkpoint_pt_epoch_{epoch}/mp_rank_00_model_states.pt")
            else:
                if is_main_process():
                    ckpt_path = os.path.join(checkpoint_save_dir, f"model-{epoch}.pth")
                    save_checkpoint(ckpt_path, epoch, model, optimizer, best_loss)
                    print(f"Model saved: {ckpt_path}")
            num_saved += 1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            print(f"Epochs since last improvement:{epochs_since_improvement}")
        '''

        if is_main_process():
            # Tensorboard
            if tensor_writer is not None:
                tensor_writer.add_scalars(
                    "train-val.loss/epoch",
                    {"train": train_loss},
                    # {"train": train_loss, "val": val_loss},
                    epoch,
                )
                if len(model_labels) > 1:
                    for subtasks in model_tasks:
                        for task in subtasks:
                            tensor_writer.add_scalars(
                                f"train.{task}.metrics/epoch",
                                train_metrics_dict[task].get_all_metrics(),
                                epoch,
                            )
                            # tensor_writer.add_scalars(
                            #     f"val.{task}.metrics/epoch",
                            #     val_metrics_dict[task].get_all_metrics(),
                            #     epoch,
                            # )
                            # tensor_writer.add_scalars(
                            #     f"val.{task}.allvalues/epoch",
                            #     val_metrics_dict[task].to_dict(),
                            #     epoch,
                            # )
                else:
                    for task in model_tasks:
                        tensor_writer.add_scalars(
                            f"train.{task}.metrics/epoch",
                            train_metrics_dict[task].get_all_metrics(),
                            epoch,
                        )
                        # tensor_writer.add_scalars(
                        #     f"val.{task}.metrics/epoch",
                        #     val_metrics_dict[task].get_all_metrics(),
                        #     epoch,
                        # )
                        # tensor_writer.add_scalars(
                        #     f"val.{task}.allvalues/epoch",
                        #     val_metrics_dict[task].to_dict(),
                        #     epoch,
                        # )

            # Save log
            train_metrics_str = "* [Train Metrics]"
            # val_metrics_str = "* [Val Metrics]"
            if len(model_labels) > 1:
                for subtasks in model_tasks:
                    for task in subtasks:
                        train_metrics_str += f"[{task.upper()}]{train_metrics_dict[task]} "
                        # val_metrics_str += f"[{task.upper()}]{val_metrics_dict[task]} "
            else:
                for task in model_tasks:
                    train_metrics_str += f"[{task.upper()}]{train_metrics_dict[task]} "
                    # val_metrics_str += f"[{task.upper()}]{val_metrics_dict[task]} "
            print(train_metrics_str)
            # print(val_metrics_str)

            # Early stopping
            # if epochs_since_improvement > args.patience:
            #     print(f"\n* Stop training.")
            #     break

            # Time
            epoch_end_time = datetime.datetime.now()
            epoch_cost_time = epoch_end_time - epoch_start_time
            cost_time += epoch_cost_time
            estimated_end_time = (
                (cost_time / (i + 1)) * 0.1 + epoch_cost_time * 0.9
            ) * (args.epochs - (i + 1)) + epoch_end_time
            print(f"* Epoch cost time: {strftimedelta(epoch_cost_time)}")
            print(
                f"* Estimated end time: {estimated_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )

    return model


def find_latest_file(directory):
    if not os.path.isdir(directory):
        raise ValueError("Provided path is not a directory")

    latest_file = None
    latest_time = 0

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        if os.path.isfile(filepath):
            file_mod_time = os.path.getmtime(filepath)
            if file_mod_time > latest_time:
                latest_time = file_mod_time
                latest_file = filepath

    return latest_file


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def warmup_cosine_lr(optimizer, args, steps):
    def _lr_adjuster(step):
        for param_group in optimizer.param_groups:
            base_lr = param_group.get("base_lr", args.lr) # (todo) 分组设立base lr

            if step < args.warmup:
                lr = _warmup_lr(base_lr, args.warmup, step)
            else:
                e = step - args.warmup
                es = steps - args.warmup
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            scale = param_group.get("lr_scale", 1.0)
            param_group["lr"] = scale * lr
        return lr
    return _lr_adjuster


# following https://github.com/OpenBMB/MiniCPM/issues/73
def warmup_stable_lr(optimizer, args, steps):
    def _lr_adjuster(step):
        for param_group in optimizer.param_groups:
            base_lr = param_group.get("base_lr", args.lr) # (todo) 分组设立base lr

            if step < args.warmup: # warmup stage
                lr = _warmup_lr(base_lr, args.warmup, step)
            elif step > steps - args.drop_step: # decay stage
                start_decay_step = steps - args.drop_step
                e = step - start_decay_step
                es = args.drop_step
                process = e / es
                lr = 0.5 * (1 + np.cos(np.pi * process)) * base_lr
            else: # stable stage
                lr = base_lr
            
            scale = param_group.get("lr_scale", 1.0)
            param_group["lr"] = scale * lr
        return lr
    return _lr_adjuster


def get_autocast(precision):
    if precision == 'fp16':
        return torch.cuda.amp.autocast
    elif precision == 'bf16':
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress
