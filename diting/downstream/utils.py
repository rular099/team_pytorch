import argparse
import os
from main_MAE_DCU import encoder_size_dict

def get_args():
    parser = argparse.ArgumentParser(description="Model training/testing arguments")

    def bool_(x):
        return False if str(x).strip().lower() in ("0", "false", "f", "no", "n") else bool(x)

    # Mode
    parser.add_argument("--conf_file", type=str, default="./train/conf.yml",
                        help="path to configuration file")
    parser.add_argument("--mode", type=str, default="train_test", metavar="MODE",
                        help="train/test/train_test (default:'train_test')")
    parser.add_argument("--hps", type=str, default="",
                        help="mup hyperparameters (default: '')")

    # Model
    parser.add_argument("--model-name", default="PatchTST", type=str, metavar="MODEL_NAME",
                        help="model name: 'patchTST' (default: patchTST)")
    parser.add_argument("--downstream-task", default="dpk", type=str, metavar="DOWNSTREAM_TASK",)
    parser.add_argument("--downstream-task-type", default="cls", type=str, metavar="DOWNSTREAM_TASK_TYPE",)
    parser.add_argument("--train_target_column", default="", type=str, metavar="TARGET_COLUMN",)
    parser.add_argument("--test_target_column", default="", type=str, metavar="TARGET_COLUMN",)
    parser.add_argument("--resume", default="", type=str, help="path to latest checkpoint (default: none)")
    parser.add_argument("--use-torch-compile", type=bool_, default=False, metavar="USE_TORCH_COMPILE",
                        help="if `True`, `torch.compile` will be called before training (default:True)")
    norm_layer_list = ['layernorm', 'fusedln', 'rmsnorm', 'batchnorm']
    parser.add_argument('--norm_layer', type=str, default='rmsnorm', choices=norm_layer_list,
                        help="normalization layer" + " | ".join(norm_layer_list))
    parser.add_argument("--xattn", default=False, action="store_true",
                        help="Whether to use flash attention.")
    parser.add_argument("--patch_size", default=50, type=int,
                        help="Patch size.")
    parser.add_argument("--use-bn-sync", default=False, action="store_true",
                        help="Whether to use batch norm sync.")
    parser.add_argument("--num_classes", default=2, type=int, 
                        help="number of classes for classification task.")
    parser.add_argument("--head_scale_factor", default=9.0, type=float, 
                        help="scaling factor for head output.")


    # Maximal Update Params (µP)
    parser.add_argument("--base_width", default=256, type=int,
                        help="Proxy (base) model’s layer width.")
    parser.add_argument("--target_width", default=512, type=int,
                        help="Target (base) model’s layer width.")
    parser.add_argument("--init-std", default=0.02, type=float,
                        help="Initialization std.")
    parser.add_argument("--input-mult", default=1, type=float,
                        help="input is multiplied by sqrt(input_mult*d_model).")
    parser.add_argument("--attn-mult", default=256, type=float,
                        help="attn is multiplied by sqrt(attn_mult)/head_dim.")
    parser.add_argument("--output-mult", type=float, default=1,
                        help="output is multiplied by sqrt(output_mult/d_model)")
    # Random seed
    parser.add_argument("--seed", default=0, type=int, metavar="SEED",
                        help="random seed for everything (default:0)")
    parser.add_argument("--use_deterministic", default=False, type=bool_)

    # Logs
    parser.add_argument("--log-dir", default="./logs", type=str, metavar="LOG_DIR",
                        help="path to save logs (default: './logs')")
    parser.add_argument("--log-step", default=4, type=int, metavar="log_step",
                        help="print metrics every log_step steps (default: 4)")
    parser.add_argument("--use-tensorboard", default=True, type=bool_, metavar="USE_TENSORBOARD",
                        help="whether to use tensorboard (default: True)")
    parser.add_argument("--eval-log-dir", default="./logs", type=str,
                        help="path to save evaluate logs (default: './logs')")

    # Save results
    parser.add_argument("--save-test-results", default=True, type=bool_, metavar="SAVE_TEST_RESULTS",
                        help="whether to save test restuls (default: True)")

    # Distributed training
    parser.add_argument("--find-unused-parameters", type=bool_, default=True, metavar="FUP",
                        help="argument of `torch.nn.parallel.DistributedDataParallel` (default:False)")
    parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--no-set-device-rank", default=False, action="store_true",
                        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).")


    # Single GPU
    parser.add_argument("--device", type=str, default="cuda:0", metavar="DEVICE",
                        help="device. If distributed mode is initialized, this argument will be ignored. (default:'cuda:0')")

    # Dataset
    # parser.add_argument("--data", default="/root/data/Datasets/Diting50hz", metavar="DATA", type=str,
    #                     help="path to dataset")
    # parser.add_argument("--dataset-name", default="diting_light", type=str, metavar="DATASET_NAME",
    #                     help="name of dataset ('diting', 'diting_light', 'pnw', 'pnw_light' or 'sos') (default: 'diting_light')")
    parser.add_argument("--data-split", type=bool_, default=False, metavar="DATA_SPLIT",
                        help="whether split dataset to train/val/test (default:True)")
    parser.add_argument("--train-size", type=float, default=0.95, metavar="TRAIN_SIZE",
                        help="size of train set (default:0.8)")
    parser.add_argument("--val-size", type=float, default=0.05, metavar="VAL_SIZE",
                        help="size of val set (default:0.1)")

    # mixdataset
    parser.add_argument("--train_meta_data_path", metavar="DATA", type=str,
                        help="path to dataset")
    parser.add_argument("--test_meta_data_path", metavar="DATA", type=str,
                        help="path to dataset")
    parser.add_argument("--train_data_dir", metavar="DATA", type=str,
                        help="path to dataset")
    parser.add_argument("--test_data_dir", metavar="DATA", type=str,
                        help="path to dataset")

    # Data loader
    parser.add_argument("--shuffle", type=bool_, default=True, metavar="SHUFFLE",
                        help="whether shuffle data. (default:True)")
    parser.add_argument("--workers", default=8, type=int, metavar="WORKERS",
                        help="number of data loading workers (default: 8)")
    parser.add_argument("--pin-memory", default=True, type=bool_, metavar="PM",
                        help="pin memory (default: True)")

    # Data preprocess
    parser.add_argument("--in-samples", default=8192, type=int, metavar="IN_SAMPLES",
                        help="the length of input data (default: 8192)")
    parser.add_argument("--label-width", type=float, default=0.5, metavar="LABEL_WIDTH",
                        help="width of soft-label (in seconds) (default:0.5)")
    parser.add_argument("--label-shape", type=str, default="gaussian", metavar="LABEL_SHAPE",
                        help="shape of soft-label ('gaussian' 'triangle' 'box' or 'sigmoid') (default: gaussian)")
    parser.add_argument("--coda-ratio", default=2.0, type=float, metavar="CODA_RATIO",
                        help="coda ratio (default:2)")
    parser.add_argument("--norm-mode", default="std", type=str, metavar="NORM_MODE",
                        help="mode of normalization ('max','std' or '') (default: 'std')")
    parser.add_argument("--min-snr", type=float, default=-float("inf"), metavar="MIN_SNR",
                        help="waveform will be regarded as noise if `all(snr)<min_snr` (default:-inf)")

    # Data augmentation
    parser.add_argument("--augmentation", type=bool_, default=True, metavar="AUGMENTATION",
                        help="whether use data augmentation. (default:True)")
    parser.add_argument("--operations", nargs='+', type=str, default=[""], metavar="OPERATIONS",
                         help="what data augmentations will be used. (default:shift_event)")
    parser.add_argument("--add-event-rate", default=0.0, type=float, metavar="ADD_EV_RATE",
                        help="Add event rate (default:0.0)")
    parser.add_argument("--max-event-num", default=1, type=int, metavar="MAX_EV_NUM",
                        help="max number of event (default:2)")
    parser.add_argument("--shift-event-rate", default=0, type=float, metavar="SHIFT_EV_RATE",
                        help="shift event rate (default:0.2)")
    parser.add_argument("--shift-event-distance-percent", default=0, type=float, metavar="SHIFT_EV_DISTANCE_PERCENT",
                        help="shift event distance percent (range:0-100)(default:0)")
    parser.add_argument("--add-noise-rate", default=0.4, type=float, metavar="ADD_NOISE_RATE",
                        help="add noise rate (default:0.4)")
    parser.add_argument("--noise-type", default="gaussian", type=str, metavar="NOISE_TYPE",
                        help="type of noise ('gaussian' or 'uniform') (default: 'gaussian')")
    parser.add_argument("--add-gap-rate", default=0.4, type=float, metavar="ADD_GAP_RATE",
                        help="add gap rate (default:0.4)")
    parser.add_argument("--min-event-gap", default=0.5, type=float, metavar="MIN_EV_GAP",
                        help="minimum event gap (in seconds) (default:0.5)")
    parser.add_argument("--drop-channel-rate", default=0.5, type=float, metavar="DROP_CH_RATE",
                        help="drop channel rate")
    parser.add_argument("--scale-amplitude-rate", default=0.5, type=float, metavar="SCALE_AMP_RATE",
                        help="scale amplitude rate")
    parser.add_argument("--pre-emphasis-rate", default=0.5, type=float, metavar="PRE_EMPH_RATE",
                        help="pre-emphaseis rate")
    parser.add_argument("--pre-emphasis-ratio", default=0.97, type=float, metavar="PRE_EMPH_RATIO",
                        help="pre-emphasis ratio (default:0.97)")
    parser.add_argument("--generate-noise-rate", default=0.5, type=float, metavar="GEN_NOISE_RATE",
                        help="generate noise rate")
    parser.add_argument("--add-mask-window-rate", default=0.5, type=float, metavar="ADD_MASK_WINDOW_RATE",
                        help="add mask window rate")
    parser.add_argument("--add-noise-window-rate", default=0.2, type=float, metavar="ADD_NOISE_WINDOW_RATE",
                        help="add noise window rate")
    parser.add_argument("--mask-percent", default=0.35, type=float, metavar="MASK_PERCENT",
                        help="the percentage of the total mask window size to the entire waveform length,"
                             " where the window size is 0.5s (range:0-100) (default: 0.35)")
    parser.add_argument("--noise-percent", default=0.3, type=float, metavar="NOISE_PERCENT",
                        help="the percentage of the total noise window size to the entire waveform length,"
                             " where the window size is 0.5s (range:0-100) (default: 0.3)")
    parser.add_argument("--whether-fixed-mask-percent", default=True, type=bool_,
                        help="whether to use fixed add mask window percent (default: True)")
    parser.add_argument("--lower-bound-mask-percent", default=30.0, type=float,
                        help="lower bound of window percent (default: 30.0)")
    parser.add_argument("--upper-bound-mask-percent", default=40.0, type=float,
                        help="upper bound of window percent (default: 40.0)")
    parser.add_argument("--whether-fixed-noise-percent", default=True, type=bool_,
                        help="whether to use fixed add noise window percent (default: True)")
    parser.add_argument("--lower-bound-noise-percent", default=25.0, type=float,
                        help="lower bound of window percent (default: 25.0)")
    parser.add_argument("--upper-bound-noise-percent", default=35.0, type=float,
                        help="upper bound of window percent (default: 35.0)")
    parser.add_argument("--num_aug", default=2, type=int,
                        help="number of applied data augmentation (default:2)")
    parser.add_argument("--mag_aug", default=0.5, type=float,
                        help="shift event rate (default:0.5)")

    # Train
    parser.add_argument("--epochs", default=200, type=int, metavar="EPOCHS",
                        help="number of total epochs (default: 200)")
    parser.add_argument("--patience", default=30, type=int, metavar="PATIENCE",
                        help="how many epochs to wait before stopping when loss is not improving (default: 30)")
    parser.add_argument("--steps", default=0, type=int, metavar="STEPS",
                        help="number of total steps. if `steps > 0`, `epochs` will be ignored. (default: 0)")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="START_EPOCH",
                        help="manual epoch number (useful on restarts) (default: 0)")
    parser.add_argument("--batch-size", default=500, type=int, metavar="BATCH_SIZE",
                        help="batch size (default: 500), this is the batch size of each worker (process)")
    parser.add_argument("--optim", default="Adam", type=str, metavar="OPTIM",
                        help="name of optimizer (default: 'Adam')")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=1.0e-8, help="Adam epsilon.")
    parser.add_argument("--grad-clip-norm", type=float, default=None, help="Gradient clip.")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="MOMENTUM",
                        help="momentum of optimizer SGD (default: 0.9)")
    parser.add_argument("--weight_decay", default=0.0, type=float, metavar="WEIGHT_DECAY",
                        help="weight_decay of optimizer (default: 0.)")
    parser.add_argument("--layer_decay", default=1.0, type=float, metavar="LAYER_DECAY",
                        help="layer_decay of lr (default: 1.)")
    parser.add_argument("--use-lr-scheduler", default=True, type=bool_, metavar="USE_LR_SCHEDULER",
                        help="whether use lr_scheduler (default: True)")
    parser.add_argument("--lr-scheduler-mode", default="exp_range", metavar="LR_SCHEDULER_MODE", type=str,
                        help="one of {'triangular', 'triangular2', 'exp_range'} (default: 'exp_range')")
    parser.add_argument("--base-lr", default=8e-5, type=float, metavar="BASE_LR",
                        help="minimum learning rate (default: 5e-5)")
    parser.add_argument("--max-lr", default=1e-3, type=float, metavar="MAX_LR",
                        help="maximum learning rate (default: 1e-3)")
    parser.add_argument("--warmup-steps", default=0.02, type=float, metavar="WARMUP_STEPS",
                        help="number of training iterations in the increasing half of a cycle."
                        " If `0 < warmup_steps < 1`, it will be treated as a ratio of total steps. (default: 1000)")
    parser.add_argument("--down-steps", default=0.1, type=float, metavar="DOWN_STEPS",
                        help="number of training iterations in the decreasing half of a cycle."
                        " If `0 < down_steps < 1`, it will be treated as a ratio of total steps."
                        " If `down_steps == 0`, it will be set to `steps - warmup_steps`(default: 1000)")
    parser.add_argument("--det-weight", default=0.1, type=float, metavar="DET_WEIGHT",
                        help="Det-weight (default: 0.5)")
    parser.add_argument("--p-weight", default=10, type=float, metavar="P_WEIGHT",
                        help="P-weight (default: 0.5)")
    parser.add_argument("--s-weight", default=10, type=float, metavar="S_WEIGHT",
                        help="S-weight (default: 0.5)")
    parser.add_argument('--loss_type', type=str, default='bce')
    parser.add_argument("--focal_loss_gamma", type=float, default=1)
    parser.add_argument('--block_decay', nargs='+', type=float, default=[1,1], help='index0 is backbone, index1 is task head')
    # multi task
    parser.add_argument("--reuse_ppm", type=bool_, default=True)
    parser.add_argument('--task_loss_weight', nargs='+', type=float,default=[1.,1.,1.,10.,1.],help='task_loss_weight follow in order of model_labels function')
    parser.add_argument("--default_label_dis", default=500, type=float)

    # Val/Test
    parser.add_argument("--time-threshold", default=0.5, type=float, metavar="TIME_THRESHOLD",
                        help="Residual threshold (in seconds) (default: 0.5)")
    parser.add_argument("--min-peak-dist", default=1.0, type=float, metavar="MIN_PEAK_DIST",
                        help="Detect peaks that are at least separated by minimum peak distance (in seconds) (defult: 1.0)")
    parser.add_argument("--ppk-threshold", default=0.3, type=float, metavar="PPK_THRESHOLD",
                        help="Probability threshold of phase-P PicKing (default: 0.3)")
    parser.add_argument("--spk-threshold", default=0.3, type=float, metavar="SPK_THRESHOLD",
                        help="Probability threshold of phase-S PicKing (default: 0.3)")
    parser.add_argument("--det-threshold", default=0.5, type=float, metavar="DET_THRESHOLD",
                        help="Probability threshold of DETection (default: 0.5)")
    parser.add_argument("--max-detect-event-num", default=1, type=int, metavar="MAX_DETECT_EV_NUM",
                        help="max number of detected events (default: 1)")
    parser.add_argument("--pretrained", default="", type=str, help="path to moco pretrained checkpoint")

    # subset names
    parser.add_argument("--subset_names", default="diting", type=str, metavar="SUBSET_NAMES",
                        help="subset names (default: DiTing)")
    parser.add_argument("--pretrain-method", default="speed", type=str)
    parser.add_argument("--lr-scheduler", default="cos", type=str)

    # model size and type
    encoder_size_names = list(encoder_size_dict.keys())
    parser.add_argument('--encoder_size', type=str,default='tiny',
                        choices=encoder_size_names,help="encoder size" + " | ".join(encoder_size_names))
    parser.add_argument("--decoder_projlast", type=bool_, default=True)
    parser.add_argument("--decoder_proj_init", type=bool_, default=True)
    parser.add_argument('--dpk_head', type=str, default='vit_adapter_TaskSeparatedUPerHead') # vit_adapter_TaskSeparatedUPerHead , vit_adapter
    parser.add_argument("--convKS", default=3, type=int)
    parser.add_argument("--freeze_layers", default=24, type=int, help="freeze the first n layers in encoder")
    # fpn (head_v3)
    # parser.add_argument("--fpn_convKS", default=5, type=int)
    parser.add_argument('--aggregate_tyee', type=str, default='concat')
    parser.add_argument("--fpn_layer_idx", default=4, type=int)
    parser.add_argument("--enable_fpn", type=bool_, default=True)
    parser.add_argument("--num_scales", default=2, type=int)
    # adapter (head_v4)
    parser.add_argument("--num_interactions", default=4, type=int, help="the number of interactions between adapter and backbone")
    parser.add_argument("--pale_size", default=5, type=int, help="the number of pales")
    parser.add_argument("--stem_convKs", default=3, type=int)
    parser.add_argument("--cpe_kernel_size", default=3, type=int)
    parser.add_argument("--ffn_convKS", default=3, type=int)
    parser.add_argument("--out_channels", default=256, type=int)
    parser.add_argument("--fpn_convKS", default=3, type=int)
    parser.add_argument("--aggregate_convKS", default=3, type=int)
    parser.add_argument("--head_convKS", default=3, type=int)
    parser.add_argument("--fused_feature", type=bool_, default=False, help="Fuse the features from decoder pred (default:False)")
    parser.add_argument("--inter_mode", default="fixed", type=str,
                        help="interaction type")
    parser.add_argument("--freeze_type", default="enc", type=str,
                        help="freeze type")
    parser.add_argument(
        "--hed_loss_weight",
        nargs="+",
        type=int,
        default=[0.5, 0.5],
        help="side output and fused output loss weight"
    )
    parser.add_argument(
        "--fused_weight",
        nargs="+",
        type=int,
        default=[0.5, 0.5],
        help="weight for side outputs aggrgated to fused output"
    )

    # eval type
    eval_type = ['finetune','linear_probe','partial_finetune']
    parser.add_argument('--eval_type', type=str,default='finetune',
                        choices=eval_type,help="eval_type" + " | ".join(eval_type))
    pool_type = ['avg','attentive','none','decoder'] # some downstream task need no pooling
    parser.add_argument('--pool_type', type=str,default='avg',
                        choices=pool_type,help="pool_type" + " | ".join(pool_type))

    # p_index shift
    parser.add_argument("--p-position-ratio", type=float, default=0.3, metavar="P_POSITION_RATIO",
                        help="The position of phase-p in the waveform. Only takes effect when `0 <= p_position_ratio <= 1` (default: -1)")
    p_position_ratio_type = ['uniform','gaussian']
    parser.add_argument('--p_position_ratio_type', type=str, default='gaussian',
                        choices=p_position_ratio_type,help="p_position_ratio_type" + " | ".join(p_position_ratio_type))
    parser.add_argument("--p_position_ratio_range_or_sigma",  type=float, default=0.01, help="p_position_ratio_range for `uniform` distribution and p_position_ratio_range for `gaussian` distribution")
    parser.add_argument("--no_event_p",  type=float, default=0.2, help="sample no event probability,only for emg task")
    parser.add_argument("--random_crop_p",  type=float, default=1, help="sample no event probability,only for emg task")
    parser.add_argument('--no_event_label', type=str, default='ignore')

    # Reg.
    parser.add_argument(
        "--train_sample_num",
        type=int,
        help="sample according to dataset ratio",
    )
    parser.add_argument(
        "--head_drop_rate",
        type=float,
        default=0,
        help="sample according to dataset ratio",
    )
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0,
        help="sample according to dataset ratio",
    )

    # visualize
    parser.add_argument("--visualize_save_dir", default="", type=str, help="path to visualize_save_dir for test")
    parser.add_argument("--visualize", type=bool_, default=False, metavar="VISUALIZE",
                    help="if `True`, visualize in args.log_dir/train_visualize and args.log_dir/test_visualize(default:True)")
    parser.add_argument("--splitPS", type=bool_, default=False, metavar="VISUALIZE",
                    help="if `True`, visualize in args.log_dir/train_visualize and args.log_dir/test_visualize(default:True)")

    # DeepSpeed
    parser.add_argument(
        '--enable-deepspeed',
        type=bool_,
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
        type=bool_,
        default=False,
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--precision",
        choices=["bf16", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision."
    )

    args = parser.parse_args()
    args.encoder_size = encoder_size_dict[args.encoder_size]

    if not 0 <= args.p_position_ratio <= 1:
        args.p_position_ratio = -1
    else:
        print(f"P position ratio: {args.p_position_ratio}")

    if args.resume:
        args.resume = os.path.abspath(args.resume)
    if args.pretrained:
        args.pretrained = os.path.abspath(args.pretrained)

    # "HPs_lr6e-05_wd0.05_mr0.9_init-std0.16_attn-mult16.0_input-mult10.0_output-mult4.0"
    if args.hps != '':
        args.input_mult = float(args.hps.split("input-mult")[-1].split("_")[0])
        args.attn_mult = float(args.hps.split("attn-mult")[-1].split("_")[0])
        args.output_mult = float(args.hps.split("output-mult")[-1].split("_")[0])
        args.target_width = int(args.hps.split("modelsize")[-1].split("_")[0])

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
