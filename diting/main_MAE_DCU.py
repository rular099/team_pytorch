#!/usr/bin/env python
import builtins
import os
from functools import partial
# import sys
import random
import shutil
import time
import warnings
import numpy as np

import torch
# import torch.backends.cudnn as cudnn
import torch.distributed as dist
try:
    from torch._six import inf
except:
    from torch import inf
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import mixdataloaders.augmentations as augmentations
import LSD.datasets as datasets
import LSD.models as models
import LSD.models.backbone_ablation as backbone_ablation
import LSD.builder
import LSD.optimizer

from params import parse_args
from optim import get_all_parameters_mup_wd
from distributed import create_deepspeed_config, init_distributed_device, is_master
from contextlib import suppress

from mup import set_base_shapes
from datetime import datetime

# mix dataloader
from mixdataloaders.mixLoader import get_data

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ['DEEPSPEED_LOG_LEVEL'] = 'OFF'

encoder_size_dict = {
    'tiny':backbone_ablation.encoder_tiny,
    'base':backbone_ablation.encoder_base,
    'large':backbone_ablation.encoder_large,
    'giant':backbone_ablation.encoder_giant,
    'proxy':backbone_ablation.encoder_tiny,
}
decoder_size_dict = {
    'base':backbone_ablation.decoder_base,
}
model_dict = {
    'ConvTasNet': models.ConvTasNet.ConvTasNet_enc,
    'SeisT': models.SeisT_LSD.SeismogramTransformer_L,
    'PatchTST': models.PatchTST_LSD_MAE.PatchTST_samll,
    'PatchTST_speed': models.PatchTST_LSD_MAE.PatchTST_samll,
    #   llama
    'Encoder_baseline_llama':backbone_ablation.Encoder_baseline_llama,
    'Decoder_baseline_llama':backbone_ablation.Decoder_baseline_llama,
    
    # llama as baseline
    'Encoder_llama_bias':backbone_ablation.Encoder_llama_bias,
    'Decoder_llama_bias':backbone_ablation.Decoder_llama_bias,
}
model_names = list(model_dict.keys())
encoder_size_names = list(encoder_size_dict.keys())
decoder_size_names = list(decoder_size_dict.keys())


def setup_without_gpu(args):
    assert dist.is_available()

    # DDP Job is being run via `srun` on a slurm cluster.
    world_size = 0
    if world_size > 1:
        args.distributed = True
    else:
        args.distributed = False
    # SLURM var -> torch.distributed vars in case needed
    # NOTE: Setting these values isn't exactly necessary, but some code might assume it's
    # being run via torchrun or torch.distributed.launch, so setting these can be a good idea.
    os.environ["RANK"] = str(0)
    os.environ["LOCAL_RANK"] = str(0)
    os.environ["WORLD_SIZE"] = str(world_size)
    args.local_rank = 0
    
    # torch.distributed.init_process_group(
    #     backend="nccl",
    #     init_method="env://",
    #     world_size=world_size,
    #     rank=args.local_rank,
    # )
    args.rank = 0
    args.world_size = world_size
    print(f"    world_size:{world_size}")
    print(f"    local_rank:{args.local_rank}")
    print("finish init_process_group")
    return world_size, args.local_rank


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available(): # GPU特性
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True # cudnn error 


def main():
    args, ds_init = parse_args(model_names, encoder_size_names, decoder_size_names) # 参数初始化

    if ds_init is not None:
        create_deepspeed_config(args) # deepspeed配置文件

    if args.seed is not None:
        set_seed(args.seed)
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )
    
    # fully initialize distributed device environment
    if args.debug:
        setup_without_gpu(args)
        args.gpu = None
    else:
        args.gpu = init_distributed_device(args) # setup(args)
        torch.distributed.barrier()
        # args.gpu = args.local_rank
    ngpus_per_node = torch.cuda.device_count()

    # show args
    print("=> args info")
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    if args.copy_codebase: # 为每个实验创建一个新的代码库（可复现性）
        copy_codebase(args)

    if args.precision == 'fp16':
        print(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')
    elif args.distributed:
        print(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        print(f'Running with a single process. Device {args.device}.')

    # infer learning rate before changing batch size
    # TODO
    # args.lr = args.lr * args.batch_size * args.world_size / 256

    # create model - 当前节点
    print('===> Constructing model....')
    base_encoder_size_dict = backbone_ablation.get_encoder_size_dict(width=args.base_width, depth=args.base_depth)
    base_decoder_size_dict = backbone_ablation.get_decoder_size_dict(width=args.base_width_dec, depth=args.base_depth_dec)
    base_model = LSD.builder.MAE_LSD(
            # backbone exp
            base_encoder=model_dict[args.arch],
            base_decoder=model_dict[args.arch_decoder],
            encoder_size=base_encoder_size_dict,
            decoder_size=base_decoder_size_dict,
            # MAE task related
            mask_ratio=args.mask_ratio,
            mask_way=args.mask_way,
            loss_type=args.loss_type,
            norm_pix_loss=args.norm_pix_loss,
            # channel_way
            channel_way=args.channel_way,
            patch_len=args.patch_size,
            args=args,
        )
    target_encoder_size_dict = backbone_ablation.get_encoder_size_dict(width=args.target_width, depth=args.base_depth)
    model = LSD.builder.MAE_LSD(
            # backbone exp
            base_encoder=model_dict[args.arch],
            base_decoder=model_dict[args.arch_decoder],
            encoder_size=target_encoder_size_dict,
            decoder_size=base_decoder_size_dict,
            # MAE task related
            mask_ratio=args.mask_ratio,
            mask_way=args.mask_way,
            loss_type=args.loss_type,
            norm_pix_loss=args.norm_pix_loss,
            # channel_way
            channel_way=args.channel_way,
            patch_len=args.patch_size,
            args=args,
        ) # support fusedLN, fast and memory efficient MHSA
    
    ### muP: set base_shapes
    set_base_shapes(model, base_model) # do_assert=False
    ### muP: Replace your custom init, if any
    model.apply(
        partial(
            model._init_weights,
            readout_zero_init=False,
            query_zero_init=False,
        )
    )
    
    total_n_parameters = sum(p.numel() for p in model.parameters())
    print(f'number of total params: {total_n_parameters/1e6}M parameters')

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'number of params with requires_grad: {n_parameters/1e6}M parameters')

    for name, p in model.named_parameters():
        width_mult = p.infshape.width_mult()
        ninf = p.infshape.ninf()
        print(f"param {name} has n_inf {ninf}, width_mult {width_mult}")
    
    print(model)
    model.to(args.gpu)
    model_without_ddp = model

    if hasattr(args, 'grad_checkpointing') and args.grad_checkpointing: # 优化内存（todo）
        model.set_grad_checkpointing()

    print('===> DDP preparing....')
    if args.distributed:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if not args.enable_deepspeed:
            model = torch.nn.parallel.DistributedDataParallel(model)
            model_without_ddp = model.module
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        if args.debug:
            pass
        else:
            raise NotImplementedError("Only DistributedDataParallel is supported.")

    print('===> Constructing criterion and optimizer....')
    optimizer_params = get_all_parameters_mup_wd(
        model, 
        decoupled_wd=True,
        lr=args.lr, 
        weight_decay=args.weight_decay, 
        args=args,
    )
    if not args.enable_deepspeed:
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                optimizer_params, 
                args.lr,
                momentum=args.momentum, 
                weight_decay=args.weight_decay
            )
        elif args.optimizer == 'lars':
            optimizer = LSD.optimizer.LARS(
                optimizer_params, 
                args.lr,
                weight_decay=args.weight_decay,
                momentum=args.momentum
            )
        elif args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                optimizer_params, 
                args.lr,
                weight_decay=args.weight_decay, 
                betas=(args.beta1, args.beta2)
            )
    else:
        if args.optimizer != "adamw":
            raise NotImplementedError("Only adamw is supported.")
        else:
            model, optimizer, _, _ = ds_init(
                args=args,
                model=model,
                model_parameters=optimizer_params,
                dist_init_required=not args.distributed,
            )
    
    # optionally resume from a checkpoint
    if args.resume:
        if args.enable_deepspeed:
            if os.path.exists(args.resume):
                import glob
                all_checkpoints = glob.glob(os.path.join(args.resume, 'checkpoint_pt_epoch_*'))
                latest_ckpt = -1
                for ckpt in all_checkpoints:
                    t = ckpt.split('/')[-1].split('_')[-1]
                    if t.isdigit():
                        latest_ckpt = max(int(t), latest_ckpt)
                if latest_ckpt >= 0:
                    args.start_epoch = latest_ckpt
                    _, client_states = model.load_checkpoint(
                        args.resume, tag='checkpoint_pt_epoch_%d' % latest_ckpt
                    ) #tag=f"epoch_{completed_epoch}"
                    print(f"=> resuming checkpoint '{args.resume}' (epoch {latest_ckpt})")
                else:
                    print("=> no checkpoint found at '{}'".format(args.resume))
            else:
                print("=> '{}' is not existing!".format(args.resume))
        else:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                # load to gpu will oom
                checkpoint = torch.load(args.resume,map_location='cpu')
                # # for no ddp
                # adjusted_checkpoint = {}
                # for key, value in checkpoint['state_dict'].items():
                #     if key.startswith("module."):
                #         adjusted_key = key.replace('module.', '')  # 替换掉 'module.' 前缀
                #     adjusted_checkpoint[adjusted_key] = value
                # checkpoint['state_dict'] = adjusted_checkpoint
                
                args.start_epoch = checkpoint['epoch']
                msg = model.load_state_dict(checkpoint['state_dict'])
                print(msg.missing_keys)
                optimizer.load_state_dict(checkpoint['optimizer'])
                # scaler.load_state_dict(checkpoint['scaler'])
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
    
    # Data loading code
    # print('===> Constructing dataset....') # 混和数据集的构建（todo）
    # # todo: consider the order of augmentations
    # if args.manual_feature == '':
    #     augs = [
    #         augmentations.RandomCropByPSIndex(p=1.0, window_size=args.input_length),
    #         augmentations.NormalizeStandardization(),
    #     ]
    # else:
    #     augs = [
    #         augmentations.RandomCropByPSIndex_leftPadding(p=1.0, window_size=args.input_length),
    #         augmentations.NormalizeStandardization(),
    #     ]
    # if args.debug:
    #     train_dataset = datasets.LSD_pretrain(
    #         LSD_type='toy-lhl725', 
    #         sample_num=args.sample_num, 
    #         augmentations=augmentations.Compose(augs),
    #         manual_feature=args.manual_feature
    #     )
    #     test_dataset = datasets.LSD_pretrain(
    #         LSD_type='toy-lhl725-test',
    #         sample_num=args.sample_num, 
    #         augmentations=augmentations.Compose(augs),
    #         manual_feature=args.manual_feature
    #     )
    # else:
    #     train_dataset = datasets.LSD_pretrain(
    #         LSD_type=args.train_dataset_type,
    #         sample_num=args.sample_num, 
    #         augmentations=augmentations.Compose(augs),
    #         manual_feature=args.manual_feature,
    #         stride=args.sample_stride,
    #     )
    #     test_dataset = None
    #     if args.eval_freq < args.epochs:
    #         test_dataset = datasets.LSD_pretrain(
    #             LSD_type=args.test_dataset_type,
    #             sample_num=args.sample_num, 
    #             augmentations=augmentations.Compose(augs),
    #             manual_feature=args.manual_feature,
    #             stride=args.sample_stride,
    #         )

    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #     test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) if test_dataset is not None else None
    # else:
    #     train_sampler = None
    #     test_sampler = None
        
    # print('===> Constructing dataloader....')
    # time_construct_dataloader = time.time()
    # train_loader = datasets.FastDataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=(train_sampler is None),
    #     num_workers=args.workers,
    #     pin_memory=True,
    #     sampler=train_sampler,
    #     drop_last=True,
    #     persistent_workers=True
    # )
    # if test_dataset is not None:
    #     test_loader = datasets.FastDataLoader(
    #         test_dataset,
    #         batch_size=args.batch_size,
    #         shuffle=(test_sampler is None),
    #         num_workers=args.workers,
    #         pin_memory=True,
    #         sampler=test_sampler,
    #         drop_last=True,
    #         persistent_workers=True
    #     )
    # print(f"contruct dataloader cost {time.time()-time_construct_dataloader}s")
    
    # Data loading code
    print('===> Constructing dataloader....')
    augs = [
        augmentations.RandomCropByPSIndex(p=1.0, window_size=args.input_length),
        augmentations.NormalizeStandardization(),
    ]
    data = get_data(args, augmentations.Compose(augs), epoch=args.start_epoch)
    train_loader = data['train'].dataloader
    # create scheduler if train
    scheduler = None
    if optimizer is not None:
        iters_per_epoch = len(train_loader)
        total_steps = int(iters_per_epoch * args.epochs)
        args.warmup = int(total_steps * args.warmup_iters) # default: 2%
        args.drop_step = int(total_steps * args.drop_iters) # default: 10%
        if is_master(args):
            print(f"total_steps: {total_steps}")
        if args.lr_schedule == 'cosine':
            scheduler = warmup_cosine_lr(optimizer, args, total_steps)
        elif args.lr_schedule == 'wsd':
            scheduler = warmup_stable_lr(optimizer, args, total_steps)
        else:
            raise NotImplementedError(f'{args.lr_schedule} is not supported, choose from [cosine, wsd]')

    # 训练监控（todo）
    summary_writer = SummaryWriter(os.path.join(args.ckpt_folder, 'tensorboard')) if is_master(args) else None

    if args.predict_feature:
        ipe=len(train_loader)
        # print(args.ema[0],type(args.ema[0]))
        args.momentum_scheduler = (args.ema[0] + i*(args.ema[1]-args.ema[0])/(ipe*args.epochs*args.ipe_scale)
                        for i in range(int(ipe*args.epochs*args.ipe_scale)+1))
    
    print('===> Start training...')
    for epoch in range(args.start_epoch, args.epochs):
        data['train'].set_epoch(epoch)
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)

        # train for one epoch
        print('===> Entering training epoch...')
        train(train_loader, model, optimizer, scheduler, epoch, summary_writer, args)

        if (epoch+1) % args.save_freq == 0 or epoch == args.epochs-1:
            if args.enable_deepspeed:
                client_state = {'epoch': epoch + 1}
                model.save_checkpoint(
                    save_dir=args.ckpt_folder, 
                    tag="checkpoint_pt_epoch_%s" % str(epoch + 1), 
                    client_state=client_state
                )
            else:
                if not args.distributed or (
                    args.distributed and args.rank % ngpus_per_node == 0
                ):
                    save_checkpoint(
                        {
                            "epoch": epoch + 1,
                            "arch": args.arch,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        is_best=False,
                        filename=args.ckpt_folder +"/checkpoint_pt_{:04d}.pth.tar".format(epoch+1),
                    )
            
        # if (epoch+1) % args.eval_freq == 0:
        #     print("evaluating...")
        #     evaluete_time = time.time()
        #     waves, pred, mask, feature_wave = evaluate(model, test_loader, summary_writer, epoch, args)
        #     print(f"evaluate cost {time.time()-evaluete_time}")
        #     # you can not visualize before getting waves
        #     if not args.predict_feature and epoch % args.visualize_freq == 0:
        #         visualize(model, waves, pred, mask, epoch, args, feature_wave)

    print('===> Training DONE...')    
    if is_master(args):
        summary_writer.close()


def train(train_loader, model, optimizer, scheduler, epoch, summary_writer, args):
    average_meters = {}
    average_meters['batch_time'] = AverageMeter("Time", ":6.3f")
    average_meters['data_time'] = AverageMeter("Data", ":6.3f")
    average_meters['losses'] = AverageMeter("Loss", ":.4e")
    average_meters['loss_scaler'] = AverageMeter("Loss_scale")
    average_meters['grad_norm'] = AverageMeter("Grad_norm")
    progress = ProgressMeter(
        len(train_loader),
        [m for m in average_meters.values()],
        prefix=f"Train: [{epoch}/{args.epochs}]",
    )
    
    autocast = get_autocast(args.precision)

    # switch to train mode
    model.train()

    iters_per_epoch = len(train_loader)
    end = time.time()
    epoch_start_time = time.time()
    for i, waves in enumerate(train_loader):
        step = iters_per_epoch * epoch + i
        scheduler(step)
            
        if args.gpu is not None:
            waves_feature = waves['manual_feature'].cuda(args.gpu, non_blocking=True) if waves.get('manual_feature') is not None else None
            waves = waves['data'].cuda(args.gpu, non_blocking=True)
        else:
            waves_feature = waves.get('manual_feature')
            waves = waves['data']
        
        average_meters['data_time'].update(time.time() - end) # measure data loading time
        if args.enable_deepspeed:
            model.zero_grad()
            # model.micro_steps = 0
        else:
            optimizer.zero_grad()

        # compute output
        with autocast():
            loss = model(waves, waves_feature)
        '''
        loss_list = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, loss)
        loss_list = torch.tensor(loss_list)

        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()
        '''

        loss_list_isnan = torch.isnan(loss).any()
        loss_list_isinf = torch.isinf(loss).any()
        if loss_list_isnan or loss_list_isinf:
            print(f" ==================== loss_isnan = {loss_list_isnan},  loss_isinf = {loss_list_isinf} ==================== ")

        if args.enable_deepspeed:
            model.backward(loss)
            model.step()
        else:
            loss.backward()
            optimizer.step()
        
        if args.predict_feature:
            m = next(args.momentum_scheduler)
            with torch.no_grad():
                for param_q, param_k in zip(model.module.base_encoder.parameters(), model.module.target_encoder.parameters()):
                    param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

        # NOTE loss is coarsely sampled, just master node and per log update
        average_meters['losses'].update(loss.item())
        if args.enable_deepspeed:
            loss_scale_value, grad_nrom = get_loss_scale_for_deepspeed(model)
        else:
            loss_scale_value = 0.0
            grad_nrom = get_grad_norm_(model.parameters())
        average_meters['grad_norm'].update(grad_nrom)
        average_meters['loss_scaler'].update(loss_scale_value)
        
        # measure elapsed time
        average_meters['batch_time'].update(time.time() - end)
        end = time.time()
        batch_count = i + 1

        if is_master(args) and (i % args.print_freq == 0):
            # check learning rate
            '''
            for group_id, param_group in enumerate(optimizer.param_groups):
                print(f"Parameter Group: {group_id}, \
                        Learning Rate: {param_group['lr']}, \
                        LR Scale: {param_group['lr_scale']}, \
                        Weight Decay: {param_group['weight_decay']}")
            '''
            # batch_size = waves.shape[0]
            # print("########## batch size is ", batch_size, "##########")
            percent_complete = 100.0 * batch_count / iters_per_epoch

            prg_str = progress.get_str(batch_idx=i, name=f"{args.encoder_size}_model_train")
            print(prg_str)
            
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": average_meters['losses'].val,
                "loss_scaler": average_meters['loss_scaler'].val,
                "grad_nrom": average_meters['grad_norm'].val,
                "lr": optimizer.param_groups[0]["lr"],
                "data_time": average_meters['data_time'].val,
                "batch_time": average_meters['batch_time'].val,
                "samples_per_scond": args.batch_size*args.world_size / average_meters['batch_time'].val,
            }

            for name, val in log_data.items():
                name = "train/" + name
                if summary_writer is not None:
                    summary_writer.add_scalar(name, val, step)

            # resetting batch / data time meters per log window
            # average_meters['batch_time'].reset()
            # average_meters['data_time'].reset()

    if is_master(args):
        summary_writer.add_scalar("train/epoch_loss", average_meters['losses'].avg, epoch)
        summary_writer.add_scalar("train/epoch_time", time.time() - epoch_start_time, epoch)


def evaluate(model, test_loader, summary_writer, epoch, args):
    model.eval()

    eval_losses = AverageMeter("evalLoss", ":.4e")
    autocast = get_autocast(args.precision)
    record_waves, record_pred, record_mask = None, None, None
    with torch.no_grad():
        for i, waves in enumerate(test_loader):
            if args.gpu is not None:
                waves_feature = waves['manual_feature'].cuda(args.gpu, non_blocking=True) if waves.get('manual_feature') is not None else None
                waves = waves['data'].cuda(args.gpu, non_blocking=True)
            else:
                waves_feature = waves.get('manual_feature')
                waves = waves['data']

            with autocast():
                loss, pred, mask = model(waves, waves_feature)
            eval_losses.update(loss.item(), waves.size(0))
            
            if record_waves is None:
                record_waves = waves
                record_pred = pred
                record_mask = mask
                record_waves_feature = waves_feature if waves_feature is not None else None
            
        if is_master(args):
            summary_writer.add_scalar("eval/epoch_loss", eval_losses.avg, epoch)
    return record_waves, record_pred, record_mask, record_waves_feature


def visualize(model, waves, pred, mask, epoch, args, feature_wave=None):
    model.eval()
    with torch.no_grad():
        if not args.distributed or (
            args.distributed and args.rank % torch.cuda.device_count() == 0
        ):
            print("visualizing...")
            
            if args.distributed:
                model = model.module
            pred = model.unpatchify(pred)
            mask = mask.detach()
            if args.channel_way.startswith('independent'):
                mask = mask.unsqueeze(-1).repeat(1, 1, 50)  # (N, H*W, p*p*3)
            else:
                mask = mask.unsqueeze(-1).repeat(1, 1, 50 *3)  # (N, H*W, p*p*3)
            mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
            
            pred = pred.detach().cpu()
            mask = mask.detach().cpu()
            waves = waves.detach().cpu()
            
            draw_row = 3
            if args.manual_feature:
                ori_waves = waves
                waves = feature_wave.detach().cpu()
                draw_row = 4
            # masked waves
            im_masked = waves * (1 - mask)

            # MAE reconstruction pasted with visible patches
            im_paste = waves * (1 - mask) + pred * mask
            
            folder_path = os.path.join(args.ckpt_folder, f'visualize/epoch{epoch}')
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)
                
            import matplotlib.pyplot as plt
            if isinstance(waves, torch.Tensor):
                waves = waves.cpu().numpy()
            if isinstance(im_masked, torch.Tensor):
                im_masked = im_masked.cpu().numpy()
            if isinstance(im_paste, torch.Tensor):
                im_paste = im_paste.cpu().numpy()
            
            x = np.arange(0, waves.shape[2])
            
            for sampleid in range(2):
                for channel in range(1):
                    plt.figure(figsize=(8, 6))
                    
                    plt.subplot(draw_row, 1, 1)
                    plt.plot(x,waves[sampleid][channel], c='black', linewidth = 0.5)
                    plt.title('origin') if args.manual_feature == '' else plt.title('feature')
                    
                    plt.subplot(draw_row, 1, 2)
                    plt.plot(x,im_masked[sampleid][channel], c='black', linewidth = 0.5)
                    plt.title('masked')
                    
                    plt.subplot(draw_row, 1, 3)
                    plt.plot(x,im_paste[sampleid][channel], c='black', linewidth = 0.5)
                    plt.title('reconstruction')
                    
                    if args.manual_feature:
                        plt.subplot(draw_row, 1, 4)
                        plt.plot(x,ori_waves[sampleid][channel], c='black', linewidth = 0.5)
                        plt.title('original waves')
                        
                    plt.tight_layout()
                    plt.savefig(os.path.join(folder_path, f"id{sampleid}-ch{channel}-gpu{args.rank}.png"))
            
                    plt.close()
            print(f"{folder_path} is saved!")
            
            
def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def get_str(self, batch_idx, name):
        entries = [self.prefix + self.batch_fmtstr.format(batch_idx) + name]
        entries += [str(meter) for meter in self.meters]
        string = "  ".join(entries)
        return string

    def set_meters(self, meters):
        self.meters = meters

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.ckpt_folder, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def warmup_cosine_lr(optimizer, args, steps):
    def _lr_adjuster(step):
        for param_group in optimizer.param_groups:
            base_lr = param_group['base_lr']

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
            base_lr = param_group['base_lr']

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


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    loss_scale = None
    if hasattr(optimizer, 'loss_scale'):
        loss_scale = optimizer.loss_scale
    elif hasattr(optimizer, 'cur_scale'):
        loss_scale = optimizer.cur_scale
    return loss_scale, optimizer._global_grad_norm


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), 
            norm_type
        )
    return total_norm.to(dtype=torch.float32)


def get_param_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    device = parameters[0].data.device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.data.detach(), norm_type).to(device) for p in parameters]), 
        norm_type
    )
    return total_norm.to(dtype=torch.float32)


def get_autocast(precision):
    if precision == 'fp16':
        return torch.cuda.amp.autocast
    elif precision == 'bf16':
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress


if __name__ == "__main__":
    # mp.set_start_method('spawn') # 注释
    main()