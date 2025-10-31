import torch
import torch.nn as nn
import os
import yaml
from finetuneing.utils import *
from finetuneing.training import *
from finetuneing.config import Config
from distributed import create_deepspeed_config
from finetuneing.training.preprocess import SFTDataset
#from finetuneing.models.loss import LossConfig,CELoss
from finetuneing.models import loss as LossMod
from finetuneing.training.postprocess import process_outputs
from downstream.utils import get_args
from downstream.train_custom import *
from downstream.downstream_head import heads

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def get_custom_data(path,key):
    raise NotImplementedError

def main_worker(args, device, ds_init):
    if is_main_process():
        print(f"device: {device}")
        print(f"pid: {os.getpid()}")
        print(f"\n{strfargs(args, Config)}")

    mode = args.mode.split("_")
    if "train" in mode:
        #
        model_inputs = [['z', 'n', 'e']]
        model_labels, model_tasks = [args.downstream_task],[args.downstream_task]
        args.model_tasks = model_tasks
        args.model_labels = model_labels
        args.model_tasks = model_tasks
        # should have columns 'Key','From','{target_column}'. 'From' use only substring before the 1st '_' and ignore case.
        # for test data, if contain p pick, should have column name 'P_index','Pn_index','Pg_index'
        # for test data, if contain s pick, should have column name 'S_index','Sn_index','Sg_index'
        # for test data, if donwstream task is 'dis'. should contain epicenter information, and have column 'Dis'
        # for test data, if donwstream task is 'emg'. should contain magnitude information, and have column 'Mag_value'
        # for train data, if contain p pick, should have column name 'p_target'
        # for train data, if contain s pick, should have column name 's_target'
        # for train data, if donwstream task is 'dis'. should contain epicenter information, and have column 'Dis'
        # for train data, if donwstream task is 'emg'. should contain magnitude information, and have column 'Mag_value'
        # all epicenter>=500km willbe discarded.

        train_dataset = SFTDataset(
            args=args,
            input_names=model_inputs,
            label_names=model_labels,
            task_names=model_tasks,
            mode="train",
        )
        #train_dataset._dataset.hdf5_file['diting1'] = [
        #    os.path.join(args.train_data_dir, 'DiTing330km_publish', f'DiTing330km_part_{part}.hdf5') 
        #    for part in range(28)
        #]
#        train_dataset._dataset.get_data_funcs['diting1'] = partial()

        setup_seed(args.seed, args.use_deterministic)
        model = train_worker(args,  device=device,  ds_init=ds_init,  train_dataset=train_dataset, CustomHead=heads[args.downstream_task_type],  process_outputs=process_outputs)
    print('----------------------------------------evaluate-------------------------------------')
    if "test" in mode:
        save_folder = args.eval_log_dir
        # for emg/dis
        p_postion = [3,5,10,20,50,90]
        p_postion = [(10000 - x*100)/10000 for x in p_postion]
        for p in p_postion:
            args.test_p_postion_ratio = p
            setup_seed(args.seed, args.use_deterministic)
            test_worker(args, device, model,save_folder)

    if not (set(("train", "test")) & set(mode)):
        raise ValueError(
            f"`mode` must be 'train','test' or 'train_test', got '{args.mode}'"
        )


if __name__ == "__main__":
    print("My rank is: ", os.environ["LOCAL_RANK"],"world size is:",os.environ["WORLD_SIZE"])
    args, ds_init = get_args()
    with open(args.conf_file, 'r') as f:
        conf_data = yaml.safe_load(f)
    vars(args).update(conf_data)
        
    os.makedirs(args.log_dir,exist_ok=True)
    os.makedirs(args.eval_log_dir,exist_ok=True)
    if args.downstream_task_type == 'cls':
        Config._avl_io_items[args.downstream_task] = {
                    "type": "onehot",
                    "metrics": ["precision", "recall", "f1"],
                    "num_classes": args.num_classes,
                }
    elif args.downstream_task_type == 'reg':
        Config._avl_io_items[args.downstream_task] = {
                    "type": "value",
                    "metrics": ["mean", "std", "mae", "r2"]
                }
    elif args.downstream_task_type == 'seg':
        Config._avl_io_items[args.downstream_task] = {
                    "type": "soft",
                    "metrics": ["precision", "recall", "f1"],
                }
    else:
        raise ValueError("downstream_task_type must be one of cls/reg/seg.")

    LossMod.LossConfig.loss_fns[args.downstream_task] = getattr(LossMod,args.loss_type)()

    depth = 24
    if depth % args.num_interactions != 0:
        args.num_interactions -= 1
    n = (depth - 1)//args.num_interactions
    args.interaction_indexes = [[i*n, (i+1)*n] for i in range(args.num_interactions)]
    if args.num_interactions * n != depth:
        args.interaction_indexes.append([args.num_interactions*n, depth])

    if ds_init is not None:
        args.optimizer = args.optim
        create_deepspeed_config(args) # deepspeed配置文件

    args.distributed = init_distributed_mode(args)
    
    if args.distributed:
        args.device = f"cuda:{get_local_rank()}"

    device = torch.device(args.device)

    main_worker(args, device, ds_init)
    clean_up_process()
