from typing import Union
import os
import torch
import torch.distributed as dist
from finetuneing.config import Config
from finetuneing.utils import *
from .postprocess import process_outputs,ResultSaver
import json

def validate(
    args, tasks,model, loss_fn, val_loader, epoch, device, testing=False,save_folder=None
) -> Union[float, dict]:
    
    model.eval()
    
    # model_labels,tgts_trans_for_loss,outs_trans_for_loss, outs_trans_for_res = Config.get_model_config_(
    #         'test_dis_emg',"labels","targets_transform_for_loss","outputs_transform_for_loss", "outputs_transform_for_results"
    #     )
    outs_trans_for_loss = None
    outs_trans_for_res = None
    tgts_trans_for_loss = None

    average_meters = {}
    metrics_merged = {}

    sampling_rate = val_loader.dataset.sampling_rate()
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

    average_meters["loss"] = AverageMeter("Loss", ":6.4f")

    progress = ProgressMeter(
        len(val_loader),
        [m for m in average_meters.values()],
        prefix=f"{'Test' if testing else 'Val'}: [{epoch}/{args.epochs}]",
    )
    
    
    if testing and args.save_test_results and is_main_process():
        results_saver = ResultSaver(item_names=tasks)
    else:
        results_saver = None

    with torch.no_grad():
        for step, (x, loss_targets, metrics_targets, meta_data_jsons) in enumerate(val_loader):
            
            # debug
            # metrics_targets['dis'] = torch.tensor(
            #     [
            #         [3],
            #         [-0.5],
            #         [2],
            #         [7]
            #     ]
            # ).to(device)
            # if device.index == 0:
            #     metrics_targets['dis'] = metrics_targets['dis'][:2]
            # else:
            #     metrics_targets['dis'] = metrics_targets['dis'][2:]
            
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

            # debug
            # loss_targets['dis'] = torch.tensor(
            #     [
            #         [1],
            #         [1],
            #         [1],
            #         [1]
            #     ]
            # ).to(device)
            # if device.index == 0:
            #     loss_targets['dis'] = loss_targets['dis'][:2]
            # else:
            #     loss_targets['dis'] = loss_targets['dis'][2:]

            outputs = model(x)
            
            # debug
            # outputs['dis'] = torch.tensor(
            #     [
            #         [2.5],
            #         [0],
            #         [2],
            #         [8]
            #     ]
            # ).to(device)
            # if device.index == 0:
            #     outputs['dis'] = outputs['dis'][:2]
            # else:
            #     outputs['dis'] = outputs['dis'][2:]

            # Loss
            outputs_for_loss = outs_trans_for_loss(outputs) if outs_trans_for_loss is not None else outputs
            loss_targets = tgts_trans_for_loss(loss_targets) if tgts_trans_for_loss is not None else loss_targets
            if loss_fn.__name__ == "multi_task_loss_fn":
                loss, loss_log_dict = loss_fn(outputs, loss_targets,metrics_targets,args.test_model_tasks,args.test_task_loss_weight,meta_data_jsons)
            else:
                loss = loss_fn(outputs_for_loss, loss_targets)

            # Batch size of this step
            step_batch_size = x.size(0)

            # Reduce
            if is_dist_avail_and_initialized():
                loss = reduce_tensor(loss, "AVG")
                step_batch_size = torch.tensor(
                    step_batch_size, device=device, dtype=torch.int32
                )
                step_batch_size = reduce_tensor(step_batch_size)
                dist.barrier()
                step_batch_size = step_batch_size.item()

            # Save loss
            average_meters["loss"].update(loss.item(), step_batch_size)

            # Process outputs
            outputs_for_metrics = outs_trans_for_res(outputs) if outs_trans_for_res is not None else outputs
            results = process_outputs(args, outputs_for_metrics,args.test_model_labels,sampling_rate)

            if results_saver is not None:
                if isinstance(meta_data_jsons,torch.Tensor):
                    meta_data_jsons = meta_data_jsons.detach().cpu().tolist()
                
                meta_data_dict={k:[] for k in json.loads(meta_data_jsons[0]).keys()}
                for j in meta_data_jsons:
                    for k,v in json.loads(j).items():
                        meta_data_dict[k].append(v)
                results_saver.append(meta_data_dict,metrics_targets,results)
                
            
            for task in tasks:
                metrics = Metrics(
                    task=task,
                    metric_names=Config.get_metrics(task),
                    sampling_rate=sampling_rate,
                    time_threshold=args.time_threshold,
                    num_samples=args.in_samples,
                    device=device,
                )
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


            if is_main_process() and step % args.log_step == 0:
                prg_str = progress.get_str(batch_idx=step,name = f"{'test' if testing else 'val'}")
                print(prg_str)

    if results_saver is not None:
        results_save_path = os.path.join(save_folder,f"test_detailed_results_p{args.test_p_postion_ratio}.csv")
        results_saver.save_as_csv(results_save_path)
        print("Saving test results to ",results_save_path)
    
    loss_avg = average_meters["loss"].avg
    return loss_avg, metrics_merged