import os
import torch
from finetuneing.config import Config
from finetuneing.models import create_model,load_checkpoint
from finetuneing.utils import *
from .preprocess import SeismicDataset,SFTDataset
from .validate import validate
import finetuneing.utils.help_builder as help_builder
from finetuneing.models.loss import multi_task_loss_fn
import numpy as np
from torch.utils.data.dataloader import default_collate

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

def test_worker(args,device,model,save_folder)->float:
    '''
    only support dis/emg tasks
    '''
    # Log
    # logger.set_logger("test")

    # Data loader
    model_inputs = [['z', 'n', 'e']]
    model_labels, model_tasks = help_builder.get_labels_tasks(args.downstream_task)
    args.test_model_tasks = model_tasks
    args.test_task_loss_weight = [1.0] * len(model_tasks)
    args.test_model_labels = model_labels

    # in_channels = Config.get_num_inchannels(model_name=args.model_name)
    test_dataset = SFTDataset(
        args=args,
        input_names=model_inputs,
        label_names=model_labels,
        task_names=model_tasks,
        mode="test",
    )

    test_sampler = (
        torch.utils.data.DistributedSampler(test_dataset)
        if is_dist_avail_and_initialized()
        else None
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=((not is_dist_avail_and_initialized()) and args.shuffle),
        pin_memory=args.pin_memory,
        num_workers=args.workers,
        sampler=test_sampler,
        collate_fn=custom_collate_fn
    )

    # Load checkpoint
    # if args.checkpoint:
    #     checkpoint = load_checkpoint(args.checkpoint, device=device)
    #     # logger.info(f"Model loaded: {args.checkpoint}")
    # else:
    #     raise ValueError("checkpoint is None.")

    # Loss
    # loss_fn = Config.get_loss(model_name=args.model_name)
    # loss_fn = loss_fn.to(device)
    loss_fn = multi_task_loss_fn

    # # Model
    # model = create_model(
    #     model_name=args.model_name,
    #     in_channels=in_channels,
    #     in_samples=args.in_samples,
    # )
    # if checkpoint is not None and "model_dict" in checkpoint:
    #     model.load_state_dict(checkpoint["model_dict"])
    #     # logger.info(f"model.load_state_dict")
    
    # if is_main_process():
    #     # logger.info(f"Model parameters: {count_parameters(model)}")

    # model = model.to(device)

    # if is_dist_avail_and_initialized():
    #     local_rank = get_local_rank()
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model,
    #         device_ids=[local_rank],
    #         find_unused_parameters=args.find_unused_parameters,
    #     )
    #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)


    test_loss, test_metrics_dict = validate(
        args, model_tasks, model, loss_fn, test_loader, 0, device, testing=True,save_folder=save_folder
    )

    if is_main_process():
        # Metrics merged
        test_metrics_str = "* "
        for task in model_tasks:
            test_metrics_str += f"[{task.upper()}]{test_metrics_dict[task]} "
        # logger.info(test_metrics_str)
        print(test_metrics_str,file=open(os.path.join(save_folder,f"{args.downstream_task}_{args.test_p_postion_ratio}.txt"),"w"))

    return test_loss
