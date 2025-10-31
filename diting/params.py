import argparse
import os
import random
import numpy as np

def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.95, "eps": 1.0e-8}


def parse_args(model_names, encoder_size_names, decoder_size_names):
    parser = argparse.ArgumentParser(description="PyTorch LSD Training")
    # Global configs
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: PatchTST)",
    )
    parser.add_argument(
        "-a_d",
        "--arch_decoder",
        metavar="ARCH",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: PatchTST)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        metavar="N",
        help="number of data loading workers (default: 32)",
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        metavar="N", 
        help="number of total epochs to run"
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--seed", 
        default=0, 
        type=int, 
        help="seed for initializing training. "
    )
    parser.add_argument(
        '--pin-mem', 
        action='store_true',
        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.'
    )
    parser.add_argument(
        '--no-pin-mem', 
        action='store_false', 
        dest='pin_mem'
    )
    parser.add_argument(
        "--eval_freq", 
        default=10000, 
        type=int, 
        help="frequency of evaluate"
    )
    parser.add_argument(
        "--save_freq", 
        default=10, 
        type=int, 
        help="frequency of save checkpoints"
    )
    parser.add_argument(
        "--debug", 
        default=0, 
        type=int
    )
    parser.add_argument(
        "--visualize_freq", 
        default=10, 
        type=int
    )
    
    # Dataset
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        metavar="N",
        help="batch size (default: 256), this is the batch size of single GPU",
    )
    parser.add_argument(
        "--input-length", 
        default=10000, 
        type=int, 
        help="length of input data, equal to cropped window size (default: 3000)"
    )
    parser.add_argument(
        "--sample_num", 
        default=0, 
        type=int, 
        help="0 is all,else is sample num"
    )
    parser.add_argument(
        "--sample_stride", 
        default=1, 
        type=int, 
        help="the stride of sample"
    )
    parser.add_argument(
        "--train_dataset_type", 
        default='mini_train', 
        type=str, 
        help="train dataset"
    )
    parser.add_argument(
        "--test_dataset_type", 
        default='mini_test', 
        type=str, 
        help="test dataset"
    )

    # Optim
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        default=1e-4,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        '--lr_schedule', 
        type=str,
        default='cosine', 
        choices=['cosine', 'wsd'],
        help='lr schedule used (default: cosine)'
    )
    parser.add_argument(
        '--min_lr', 
        type=float, 
        default=0., 
        metavar='LR',
        help='lower lr bound for cyclic schedulers that hit 0'
    )
    parser.add_argument(
        '--warmup_iters', 
        type=float, 
        default=0.02, 
        metavar='N',
        help='relative iterations to warmup LR, should be in (0~1)'
    )
    parser.add_argument(
        '--drop_iters', 
        type=float, 
        default=0.1, 
        metavar='N',
        help='relative iterations to drop LR, should be in (0~1)'
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        type=float,
        default=0.05, 
        metavar="W",
        help="weight decay (default: 0.05)",
        dest="weight_decay",
    )
    parser.add_argument(
        '--optimizer', 
        type=str,
        choices=['lamb', 'adamw'],
        help='optimizer used (default: adamw)'
    )
    parser.add_argument(
        "--beta1", 
        type=float, 
        default=None, 
        help="Adam beta 1."
    )
    parser.add_argument(
        "--beta2", 
        type=float, 
        default=None, 
        help="Adam beta 2."
    )
    parser.add_argument(
        "--eps", 
        type=float, 
        default=None, 
        help="Adam epsilon."
    )
    parser.add_argument(
        "--grad-clip-norm", 
        type=float, 
        default=None, 
        help="Gradient clip."
    )
    
    # Distributed training configs
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", 
        default=-1, 
        type=int, 
        help="node rank for distributed training"
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log diretory, and execute from there."
    )
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="local_rank"
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://224.66.41.62:23456",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", 
        default="nccl", 
        type=str, 
        help="distributed backend"
    )
    parser.add_argument(
        "--gpu", 
        default=None, 
        type=int, 
        help="GPU id to use."
    )
    parser.set_defaults(pin_mem=True)
    
    # MAE specific configs
    parser.add_argument(
        "--mask-ratio", 
        default=0.9, 
        type=float, 
        help="Masking ratio (percentage of removed patches)."
    )
    parser.add_argument(
        "--ckpt-folder", 
        default="../results/pretrain/mae-pt", 
        type=str
    )
    parser.add_argument(
        "--base_path", 
        default="./", 
        type=str
    )
    parser.add_argument(
        '--norm_pix_loss', 
        action='store_true',
        help='Use (per-patch) normalized pixels as targets for computing loss'
    )
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument(
        '--predict_feature', 
        action='store_true',
        help='Use ema feature as targets for computing loss'
    )
    parser.set_defaults(predict_feature=False)
    loss_type_list = ['l1','l2','smooth_l1']
    parser.add_argument(
        '--loss_type', 
        type=str, 
        default='l2',
        choices=loss_type_list,
        help="loss type choice: " + " | ".join(loss_type_list)
    )
    mask_way_list = ['random','grid','block']
    parser.add_argument(
        '--mask_way', 
        type=str, 
        default='random',
        choices=mask_way_list,help="mask_way choice: " + " | ".join(mask_way_list)
    )

    # Model configs
    norm_layer_list = ['layernorm', 'fusedln', 'rmsnorm', 'batchnorm']
    parser.add_argument(
        '--norm_layer', 
        type=str,
        default='rmsnorm',
        choices=norm_layer_list,
        help="normalization layer" + " | ".join(norm_layer_list)
    )
    parser.add_argument(
        "--xattn",
        default=False,
        action="store_true",
        help="Whether to use flash attention."
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync."
    )
    parser.add_argument(
        '--ema', 
        nargs='+',
        type=float, 
        help='emas for enc feature'
    )
    parser.add_argument(
        '--ipe_scale', 
        type=float, 
        default=1, 
        help='ipe scale for enc feature'
    )
    parser.add_argument(
        '--encoder_size', 
        type=str,
        choices=encoder_size_names,
        help="encoder size" + " | ".join(encoder_size_names)
    )
    parser.add_argument(
        '--decoder_size', 
        type=str,
        choices=decoder_size_names,
        help="decoder size: " + " | ".join(decoder_size_names)
    )
    parser.add_argument(
        "--patch_size", 
        default=50, 
        type=int, 
        help="Patch size."
    )
    # chanel
    channel_way_list = ['dependent','independent','independent_shuffle']
    parser.add_argument(
        '--channel_way', 
        type=str, 
        default='dependent',
        choices=channel_way_list,
        help="channel_way choice: " + " | ".join(channel_way_list)
    )
    # manual feature
    manual_feature_list = ['','classic_sta_lta','recursive_sta_lta','kurtosis','envelope']
    parser.add_argument(
        '--manual_feature', 
        type=str, 
        default='',
        choices=manual_feature_list,
        help="manual_feature choice: " + " | ".join(manual_feature_list)
    )

    # Maximal Update Params (µP)
    parser.add_argument(
        '--hps', 
        type=str, 
        default='',
        help="the sampled hps of exp"
    )
    parser.add_argument(
        "--base_width_dec", 
        default=256, 
        type=int, 
        help="Decoder model’s layer width."
    )
    parser.add_argument(
        "--base_depth_dec", 
        default=8, 
        type=int, 
        help="Decoder model’s layer depth."
    )
    parser.add_argument(
        "--base_depth", 
        default=28, 
        type=int, 
        help="Proxy (base) model’s layer depth."
    )
    parser.add_argument(
        "--base_width", 
        default=256, 
        type=int, 
        help="Proxy (base) model’s layer width."
    )
    parser.add_argument(
        "--target_width", 
        default=256, 
        type=int, 
        help="Target (base) model’s layer width."
    )
    parser.add_argument(
        "--init-std", 
        default=0.02, 
        type=float, 
        help="Initialization std."
    )
    parser.add_argument(
        "--input-mult", 
        default=1, 
        type=float, 
        help="input is multiplied by sqrt(input_mult*d_model)."
    )
    parser.add_argument(
        "--attn-mult", 
        default=256, 
        type=float, 
        help="attn is multiplied by sqrt(attn_mult)/head_dim."
    )
    parser.add_argument(
        "--output-mult", 
        type=float, 
        default=1,
        help="output is multiplied by sqrt(output_mult/d_model)"
    )

    # DeepSpeed
    parser.add_argument(
        '--enable-deepspeed', 
        action='store_true', 
        default=False
    )
    parser.add_argument(
        '--zero-stage', 
        type=int, 
        default=1,
        help='stage of ZERO'
    )
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--precision",
        choices=["bf16", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision."
    )
    args = parser.parse_args()

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.arch)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)
    
    # HPs grid search sampling
    hps = args.hps.split('_')
    hps = [float(hp) for hp in hps]
    args.init_std, args.lr, args.weight_decay, args.input_mult, args.attn_mult, args.output_mult = hps

    # mkdir exp dir based on above HPs 
    exp_infos=f"HPs_schedule{args.lr_schedule}_" + \
            f"ep{args.epochs}_data{args.sample_num}_modelsize{args.target_width}_" + \
            f"lr{args.lr}_wd{args.weight_decay}_mr{args.mask_ratio}_" + \
            f"init-std{args.init_std}_attn-mult{args.attn_mult}_" + \
            f"input-mult{args.input_mult}_output-mult{args.output_mult}"
    pt_folder=f"{args.base_path}/results/scaling_mae_mup_ablation/transfered"
    args.ckpt_folder=f"{pt_folder}/{exp_infos}"
    
    os.makedirs(args.ckpt_folder, exist_ok=True)
    
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
    
    return args, ds_init