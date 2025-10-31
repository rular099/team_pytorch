import os
import argparse
from main_MAE_DCU import encoder_size_dict

def get_args():
    args = argparse.Namespace()
    
    # Mode
    args.conf_file = "/data/huawei_share/zhangbei/work_dir/zhangbei/TEAM_PYTORCH/diting/config/conf_reg.yml"
    args.mode = "train_test"
    args.hps = ""
    
    # Model
    args.model_name = "PatchTST"
    args.downstream_task = "dpk"
    args.downstream_task_type = "cls"
    args.train_target_column = ""
    args.test_target_column = ""
    args.resume = ""
    args.use_torch_compile = False
    args.norm_layer = 'rmsnorm'
    args.xattn = False
    args.patch_size = 50
    args.use_bn_sync = False
    args.num_classes = 2
    args.head_scale_factor = 9.0
    
    # Maximal Update Params (µP)
    args.base_width = 256
    args.target_width = 512
    args.init_std = 0.02
    args.input_mult = 1
    args.attn_mult = 256
    args.output_mult = 1
    args.seed = 0
    args.use_deterministic = False
    
    # Logs
    args.log_dir = "./logs"
    args.log_step = 4
    args.use_tensorboard = True
    args.eval_log_dir = "./logs"
    
    # Save results
    args.save_test_results = True
    
    # Distributed training
    args.find_unused_parameters = True
    args.local_rank = -1
    args.dist_backend = "nccl"
    args.no_set_device_rank = False
    
    # Single GPU
    args.device = "cuda:0"
    
    # Dataset
    args.data_split = False
    args.train_size = 0.95
    args.val_size = 0.05
    
    # mixdataset
    args.train_meta_data_path = None
    args.test_meta_data_path = None
    args.train_data_dir = None
    args.test_data_dir = None
    
    # Data loader
    args.shuffle = True
    args.workers = 8
    args.pin_memory = True
    
    # Data preprocess
    args.in_samples = 8192
    args.label_width = 0.5
    args.label_shape = "gaussian"
    args.coda_ratio = 2.0
    args.norm_mode = "std"
    args.min_snr = float("-inf")
    
    # Data augmentation
    args.augmentation = True
    args.operations = [""]
    args.add_event_rate = 0.0
    args.max_event_num = 1
    args.shift_event_rate = 0
    args.shift_event_distance_percent = 0
    args.add_noise_rate = 0.4
    args.noise_type = "gaussian"
    args.add_gap_rate = 0.4
    args.min_event_gap = 0.5
    args.drop_channel_rate = 0.5
    args.scale_amplitude_rate = 0.5
    args.pre_emphasis_rate = 0.5
    args.pre_emphasis_ratio = 0.97
    args.generate_noise_rate = 0.5
    args.add_mask_window_rate = 0.5
    args.add_noise_window_rate = 0.2
    args.mask_percent = 0.35
    args.noise_percent = 0.3
    args.whether_fixed_mask_percent = True
    args.lower_bound_mask_percent = 30.0
    args.upper_bound_mask_percent = 40.0
    args.whether_fixed_noise_percent = True
    args.lower_bound_noise_percent = 25.0
    args.upper_bound_noise_percent = 35.0
    args.num_aug = 2
    args.mag_aug = 0.5
    
    # Train
    args.epochs = 200
    args.patience = 30
    args.steps = 0
    args.start_epoch = 0
    args.batch_size = 500
    args.optim = "Adam"
    args.beta1 = 0.9
    args.beta2 = 0.999
    args.eps = 1.0e-8
    args.grad_clip_norm = None
    args.momentum = 0.9
    args.weight_decay = 0.0
    args.layer_decay = 1.0
    args.use_lr_scheduler = True
    args.lr_scheduler_mode = "exp_range"
    args.base_lr = 8e-5
    args.max_lr = 1e-3
    args.warmup_steps = 0.02
    args.down_steps = 0.1
    args.det_weight = 0.1
    args.p_weight = 10
    args.s_weight = 10
    
    # Loss related
    args.loss_type = 'bce'
    args.focal_loss_gamma = 1
    args.block_decay = [1, 1]
    args.reuse_ppm = True
    args.task_loss_weight = [1., 1., 1., 10., 1.]
    args.default_label_dis = 500
    
    # Val/Test
    args.time_threshold = 0.5
    args.min_peak_dist = 1.0
    args.ppk_threshold = 0.3
    args.spk_threshold = 0.3
    args.det_threshold = 0.5
    args.max_detect_event_num = 1
    args.pretrained = ""
    
    # subset names
    args.subset_names = "diting"
    args.pretrain_method = "speed"
    args.lr_scheduler = "cos"
    
    # model size and type
    encoder_size_names = list(encoder_size_dict.keys())
    args.encoder_size = 'tiny'
    # 将字符串转换为对应的字典值
    args.encoder_size = encoder_size_dict[args.encoder_size]
    
    args.decoder_projlast = True
    args.decoder_proj_init = True
    args.dpk_head = 'vit_adapter_TaskSeparatedUPerHead'
    args.convKS = 3
    args.freeze_layers = 24
    
    # fpn (head_v3)
    args.aggregate_tyee = 'concat'
    args.fpn_layer_idx = 4
    args.enable_fpn = True
    args.num_scales = 2
    
    # adapter (head_v4)
    args.num_interactions = 4
    args.pale_size = 5
    args.stem_convKs = 3
    args.cpe_kernel_size = 3
    args.ffn_convKS = 3
    args.out_channels = 256
    args.fpn_convKS = 3
    args.aggregate_convKS = 3
    args.head_convKS = 3
    args.fused_feature = False
    args.inter_mode = "fixed"
    args.freeze_type = "enc"
    args.hed_loss_weight = [0.5, 0.5]
    args.fused_weight = [0.5, 0.5]
    
    # eval type
    args.eval_type = 'finetune'
    # pool type
    args.pool_type = 'avg'
    
    # p_index shift
    args.p_position_ratio = 0.3
    args.p_position_ratio_type = 'gaussian'
    args.p_position_ratio_range_or_sigma = 0.01
    args.no_event_p = 0.2
    args.random_crop_p = 1
    args.no_event_label = 'ignore'
    
    # Reg.
    args.train_sample_num = None
    args.head_drop_rate = 0
    args.drop_path = 0
    
    # visualize
    args.visualize_save_dir = ""
    args.visualize = False
    args.splitPS = False
    
    # DeepSpeed
    args.enable_deepspeed = False
    args.zero_stage = 1
    args.grad_checkpointing = False
    args.precision = "fp32"
    
    # 最终的参数处理，如路径转换等
    if args.resume:
        args.resume = os.path.abspath(args.resume)
    if args.pretrained:
        args.pretrained = os.path.abspath(args.pretrained)
    
    # 其他参数的额外处理（如hps参数解析等）可在此添加
    if args.hps != '':
        parts = args.hps.split("input-mult")
        if len(parts) > 1:
            args.input_mult = float(parts[-1].split("_")[0])
        parts = args.hps.split("attn-mult")
        if len(parts) > 1:
            args.attn_mult = float(parts[-1].split("_")[0])
        parts = args.hps.split("output-mult")
        if len(parts) > 1:
            args.output_mult = float(parts[-1].split("_")[0])
        parts = args.hps.split("modelsize")
        if len(parts) > 1:
            args.target_width = int(parts[-1].split("_")[0])
    
    if args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            os.environ['ENV_TYPE'] = "deepspeed"
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.8.1'")
            exit(0)
    else:
        os.environ['ENV_TYPE'] = "pytorch"
        ds_init = None
    
    if args.train_target_column == "":
        args.train_target_column = args.downstream_task
    if args.test_target_column == "":
        args.test_target_column = args.downstream_task
    return args, ds_init
