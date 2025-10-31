import sys
sys.path.append('/public/home/test_bigmodel/seismogram/mx/code/LP_single_task/single_task/SFT')
import os
import h5py
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import torch
from finetuneing.utils import *
from finetuneing.training import *
from .postprocess import postprocesser_ev_center

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# plt.rcParams["font.family"] = "Arial"
warnings.filterwarnings("ignore")
Threshold = 100 #100输出一次结果，300输出一次结果
Threshold2 = 10*Threshold
Magnitude = 0


def vis_waves_preds_postres_targets(
    waveforms: np.ndarray, # (3,l3n)
    preds: np.ndarray,
    postres: dict,
    targets: dict,
    threshold: dict,
    sampling_rate=None,
    save_dir="./",
    tag=None,
    linewidth=0.3,
    fontsize=5
):
    fig = plt.figure()
    num_row = waveforms.shape[0] + preds.shape[0]
    for idx, wave in enumerate(waveforms):
        plt.subplot(num_row, 1, idx + 1)
        if sampling_rate is None:
            plt.plot(wave, "-", color="k", linewidth=linewidth)
        else:
            x = [i / sampling_rate for i in range(len(wave))]
            plt.plot(x, wave, "-", color="k", linewidth=linewidth)
        plt.text(
            0.001,
            0.95,
            f"Channel-{idx}",
            horizontalalignment="left",
            verticalalignment="top",
            transform=plt.gca().transAxes,
            fontsize="small",
            fontweight="normal",
        )
        if np.isnan(targets['p']) == False:
            plt.axvline(x=targets['p'], color='b', linestyle='-', linewidth=linewidth)
        if np.isnan(targets['s']) == False:
            plt.axvline(x=targets['s'], color='b', linestyle='-.', linewidth=linewidth)

        plt.xticks([targets['p'],targets['s']],[targets['p'],targets['s']],rotation=45,fontsize=fontsize)
        plt.xlim(0, len(wave))

    for idx, data in enumerate(preds):
        plt.subplot(num_row, 1, waveforms.shape[0] + idx + 1)
        if sampling_rate is None:
            plt.plot(data, "-", color="k", linewidth=linewidth)
        else:
            x = [i / sampling_rate for i in range(len(data))]
            plt.plot(x, data, "-", color="k", linewidth=linewidth)
        plt.text(
            0.001,
            0.95,
            f"Pred-{idx}",
            horizontalalignment="left",
            verticalalignment="top",
            transform=plt.gca().transAxes,
            fontsize="small",
            fontweight="normal",
        )
        
        # 绘制p_pre_list
        if np.isnan(targets['p']) == False:
            for p_pre in postres['p_pre_list']:
                if p_pre == np.nan:continue
                if np.abs(p_pre - targets['p']) <= threshold['threshold_1']:
                    plt.axvline(x=p_pre, color='g', linestyle='--', linewidth=linewidth)
                elif np.abs(p_pre - targets['p']) < threshold['threshold_2']:
                    plt.axvline(x=p_pre, color='r', linestyle='--', linewidth=linewidth)
        
        if np.isnan(targets['s']) == False:
            for s_pre in postres['s_pre_list']:
                if s_pre == np.nan:continue
                if np.abs(s_pre - targets['s']) <= threshold['threshold_1']:
                    plt.axvline(x=s_pre, color='g', linestyle='-.', linewidth=linewidth)
                elif np.abs(s_pre - targets['s']) < threshold['threshold_2']:
                    plt.axvline(x=s_pre, color='r', linestyle='-.', linewidth=linewidth)
                    
        plt.ylim(-0.1, 1)
        plt.yticks([0,0.5,0.9])
        plt.xticks(postres['p_pre_list'] + postres['s_pre_list'],postres['p_pre_list'] + postres['s_pre_list'],rotation=45,fontsize=fontsize)
        plt.xlim(0, len(wave))

    if sampling_rate is None:
        plt.xlabel("Sample points")
    else:
        plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=1)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(
        os.path.join(save_dir,tag),
        dpi=400,
    )
    plt.close()


def convert_model_to_fp32(model):
    for param in model.parameters():
        param.data = param.data.to(torch.float32)


def evaluate_worker(args, device, model, visualize=True, det_th=0.1, p_th=0.1, s_th=0.1):
    logdir = args.eval_log_dir
    convert_model_to_fp32(model)
    model.eval()
    
    # Dataset
    base_path="/public/home/test_bigmodel/LargeSeismicDatasets/DiTingV3_Test" 
    # debug
    # base_path = "/mnt/seismic_datasets/seismic_datasets_for_foundation_models/diting3"
    lsdcsv = pd.read_csv(f'{base_path}/LSD_ditingV3_for_test.csv')
    lsdh5 = h5py.File(f'{base_path}/LSD_ditingV3_for_test.hdf5', 'r')
    LEN = int(len(lsdcsv) // 5) # 选择数据集长度
    # 多个波预测
    window_length = 10000
    step_size = 3000

    # Loop through the data and append to the DataFrame
    world_size = get_world_size()
    rank_id = get_rank()
    part_total = LEN // world_size
    data_parts = [start for start in range(0, LEN, part_total)]
    
    p_diff_list = []
    s_diff_list = []
    TP_P = 0
    FP_P = 0
    FN_P = 0
    TP_S = 0
    FP_S = 0
    FN_S = 0
    STRICT_FP_S = 0
    STRICT_FP_P = 0
    STRICT_FN_P = 0
    STRICT_FN_S = 0
    counter = 0
    for idx in range(data_parts[rank_id], data_parts[rank_id] + part_total):
        line = lsdcsv.iloc[idx]
        tmp_key = line.Key
        tmp_waveform = lsdh5.get(tmp_key)
        tmp_waveform = np.array(tmp_waveform).astype(np.float32)

        # Skip conditions
        if tmp_waveform.shape[1] != 3: 
            continue
        line = lsdcsv[lsdcsv['Key'] == tmp_key].squeeze()
        if type(line.Mag_value) == pd.Series:
            print(line.Mag_value, line.Mag_value.values)
            if not line.Mag_value.values:
                print(line.Mag_value.values, ' continue')
                continue
        if line.Mag_value < Magnitude:
            continue
        
        # annotated ====================================================
        test = [line.P_index, line.Pn_index, line.Pg_index]
        P_value = np.nanmin(test) 
        test = [line.S_index, line.Sn_index, line.Sg_index]
        S_value = np.nanmin(test)
        
        # inference =============================================
        # model window prediction
        p_pre_list = []
        s_pre_list = []
        count = np.zeros((1,3,tmp_waveform.shape[0]))
        result = np.zeros((1,3,tmp_waveform.shape[0]))
        
        num_windows = (tmp_waveform.shape[0] - window_length) // step_size + 1
        for i in range(num_windows):
            start = i * step_size
            end = start + window_length
            count[:,:,start:end] += 1
            
            # Perform operations on the windowed data
            window = tmp_waveform.copy()[start:end, :]
            window -= np.mean(window)
            window /= np.std(window)
            # Fill empty window with zeros
            if window.shape[0] < window_length:
                padding = np.zeros((window_length - window.shape[0], window.shape[1]))
                window = np.vstack((window, padding))
                
            window_tensor = torch.from_numpy(window)[None, :]
            window_tensor = window_tensor.permute(0, 2, 1)
            window_tensor = window_tensor.to(device).float()

            with torch.no_grad():
                output = model(window_tensor)
                if isinstance(output, (list, tuple)):
                    output = output[0]
                elif isinstance(output, dict):
                    output =  torch.concat((output['det'].unsqueeze(1), output['ppk'].unsqueeze(1), output['spk'].unsqueeze(1)), dim=1)

            output_np = output.cpu().detach().numpy()
            result[:,:,start:end] = result[:,:,start:end] + output_np
                
        result = result / count
        # 使用所有窗口confidence均值预测的结果
        events = postprocesser_ev_center(
            yh1=result[0, 0, :], yh2=result[0, 1, :], yh3=result[0, 2, :],
            det_th=det_th, p_th=p_th, p_mpd=10, s_th=s_th, s_mpd=10
        )
    
        for event in events:
            p_pre_list.append(event[1][:][0][0])
            s_pre_list.append(event[2][:][0][0])
        # inference ====================================================================================================

        if is_main_process() and visualize and counter % 50 == 0:
            vis_waves_preds_postres_targets(
                waveforms=tmp_waveform.T, # (3,l3n)
                preds=result.squeeze(), # (3,l3n)
                postres={'p_pre_list':p_pre_list,'s_pre_list':s_pre_list},
                targets={'p':P_value,'s':S_value},
                threshold={'threshold_1':Threshold,'threshold_2':Threshold2},
                save_dir=logdir,
                tag=f"{str(tmp_key).replace(':','_')}.png",
            )

        # calculate the difference between predicted and annotated P
        if len(p_pre_list) == 0 and np.isnan(P_value) == False:
            FN_P = FN_P + 1
            STRICT_FN_P = STRICT_FN_P + 1
        elif len(p_pre_list) != 0 and np.isnan(P_value) == False:
            min_pdiff = []
            for p_pre in p_pre_list:
                p_diff = p_pre - P_value
                min_pdiff.append(np.abs(p_diff))
                if np.abs(p_diff) <= Threshold:
                    p_diff_list.append(p_diff)
                else:
                    STRICT_FP_P = STRICT_FP_P + 1
                    if np.abs(p_diff) < Threshold2:
                        FP_P = FP_P + 1

            min_pdiff = np.min(min_pdiff)
            if min_pdiff <= Threshold:
                TP_P = TP_P + 1
            elif min_pdiff > Threshold:
                STRICT_FN_P = STRICT_FN_P + 1
                if min_pdiff > Threshold2:
                    FN_P = FN_P + 1
                    
        # calculate the difference between predicted and annotated S
        if len(s_pre_list) == 0 and np.isnan(S_value) == False:
            FN_S = FN_S + 1
            STRICT_FN_S = STRICT_FN_S + 1
        elif len(s_pre_list) != 0 and np.isnan(S_value) == False:
            min_sdiff = []
            for s_pre in s_pre_list:
                s_diff = s_pre - S_value
                min_sdiff.append(np.abs(s_diff))
                if np.abs(s_diff) <= Threshold:
                    s_diff_list.append(s_diff)
                else:
                    STRICT_FP_S = STRICT_FP_S + 1
                    if np.abs(s_diff) < Threshold2:
                        FP_S = FP_S + 1
            
            min_sdiff = np.min(min_sdiff)
            if min_sdiff <= Threshold:
                TP_S = TP_S + 1
            elif min_sdiff > Threshold:
                STRICT_FN_S = STRICT_FN_S + 1
                if min_sdiff > Threshold2:
                    FN_S = FN_S + 1
        
        counter += 1
        if counter % 50 == 0:
            if is_main_process():
                print(counter,'/', part_total)
    
    result = [p_diff_list, s_diff_list, TP_P, FP_P, FN_P, TP_S, FP_S, FN_S, STRICT_FP_P, STRICT_FP_S, STRICT_FN_P, STRICT_FN_S]
    save_result(rank_id, result, logdir)
    return logdir


def save_result(rank_id, result, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(os.path.join(save_folder, f'result_{rank_id}.pkl'), 'wb') as f:
        pickle.dump(result, f)


def load_results(save_folder):
    results = []
    for filename in os.listdir(save_folder):
        if filename.startswith('result_') and filename.endswith('.pkl'):
            with open(os.path.join(save_folder, filename), 'rb') as f:
                results.append(pickle.load(f))
    return results


def merge_result(results):
    print('Merging results...')
    p_diff_list,s_diff_list,TP_P,FP_P,FN_P,TP_S,FP_S,FN_S,STRICT_FP_P,STRICT_FP_S,STRICT_FN_P,STRICT_FN_S = [],[],0,0,0,0,0,0,0,0,0,0
    for result in results:
        p_diff_list += result[0]
        s_diff_list += result[1]
        TP_P += result[2]
        FP_P += result[3]
        FN_P += result[4]
        TP_S += result[5]
        FP_S += result[6]
        FN_S += result[7]
        STRICT_FP_P += result[8]
        STRICT_FP_S += result[9]
        STRICT_FN_P += result[10]
        STRICT_FN_S += result[11]
    
    merge_result = [p_diff_list, s_diff_list, TP_P, FP_P, FN_P, TP_S, FP_S, FN_S, STRICT_FP_P, STRICT_FP_S, STRICT_FN_P, STRICT_FN_S]
    return merge_result


def get_result(p_diff_list, s_diff_list, TP_P, FP_P, FN_P, TP_S, FP_S, FN_S, STRICT_FP_P, STRICT_FP_S, STRICT_FN_P, STRICT_FN_S, save_folder):
   
    epsilon = 1e-10

    Acc_P = TP_P / (TP_P + FP_P + FN_P + epsilon)
    Precision_P = TP_P / (TP_P + FP_P + epsilon)
    Recall_P = TP_P / (TP_P + FN_P + epsilon)
    F1_P = 2 * Precision_P * Recall_P / (Precision_P + Recall_P + epsilon)

    Acc_S = TP_S / (TP_S + FP_S + FN_S + epsilon)
    Precision_S = TP_S / (TP_S + FP_S + epsilon)
    Recall_S = TP_S / (TP_S + FN_S + epsilon)
    F1_S = 2 * Precision_S * Recall_S / (Precision_S + Recall_S + epsilon)

    strict_Acc_P = TP_P / (TP_P + STRICT_FP_P + STRICT_FN_P + epsilon)
    strict_Precision_P = TP_P / (TP_P + STRICT_FP_P + epsilon)
    strict_Recall_P = TP_P / (TP_P + STRICT_FN_P + epsilon)
    strict_F1_P = 2 * strict_Precision_P * strict_Recall_P / (strict_Precision_P + strict_Recall_P + epsilon)

    strict_Acc_S = TP_S / (TP_S + STRICT_FP_S + STRICT_FN_S + epsilon)
    strict_Precision_S = TP_S / (TP_S + STRICT_FP_S + epsilon)
    strict_Recall_S = TP_S / (TP_S + STRICT_FN_S + epsilon)
    strict_F1_S = 2 * strict_Precision_S * strict_Recall_S / (strict_Precision_S + strict_Recall_S + epsilon)

    # 将结果保存到文件
    print('Save results to', save_folder)
    print('Acc_P, Precision_P, Recall_P, F1_P, Acc_S, Precision_S, Recall_S, F1_S \n{:.2%}, {:.2%}, {:.2%}, {:.2%}, {:.2%}, {:.2%}, {:.2%}, {:.2%}'.format(Acc_P, Precision_P, Recall_P, F1_P, Acc_S, Precision_S, Recall_S, F1_S),file=open(f'{save_folder}/result_acc_threshold_{Threshold}_M_more_{Magnitude}.txt','w'))
    print('TP_P, FP_P, FN_P, TP_S, FP_S, FN_S \n{}, {}, {}, {}, {}, {}'.format(TP_P, FP_P, FN_P, TP_S, FP_S, FN_S),file=open(f'{save_folder}/result_TP_threshold_{Threshold}_M_more_{Magnitude}.txt','w'))

    print('strict_Acc_P, strict_Precision_P, strict_Recall_P, strict_F1_P, strict_Acc_S, strict_Precision_S, strict_Recall_S, strict_F1_S \n{:.2%}, {:.2%}, {:.2%}, {:.2%}, {:.2%}, {:.2%}, {:.2%}, {:.2%}'.format(strict_Acc_P, strict_Precision_P, strict_Recall_P, strict_F1_P, strict_Acc_S, strict_Precision_S, strict_Recall_S, strict_F1_S),file=open(f'{save_folder}/result_strict_acc_threshold_{Threshold}_M_more_{Magnitude}.txt','w'))
    print('TP_P, STRICT_FP_P, STRICT_FN_P, TP_S, STRICT_FP_S, STRICT_FN_S \n{}, {}, {}, {}, {}, {}'.format(TP_P, STRICT_FP_P, STRICT_FN_P, TP_S, STRICT_FP_S, STRICT_FN_S),file=open(f'{save_folder}/result_strict_TP_threshold_{Threshold}_M_more_{Magnitude}.txt','w'))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(p_diff_list, bins=np.arange(-Threshold, Threshold, 6))
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.title('Histogram of p_diff')

    plt.subplot(1, 2, 2)
    plt.hist(s_diff_list, bins=np.arange(-Threshold, Threshold, 6))
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.title('Histogram of s_diff')

    plt.tight_layout()
    plt.savefig('{}/histogram_TP_threshold_{}_M_more_{}.png'.format(save_folder, Threshold, Magnitude))
    

if __name__ == '__main__':
    save_folder = './result'
    results = load_results(save_folder)
    p_diff_list, s_diff_list, TP_P, FP_P, FN_P, TP_S, FP_S, FN_S, STRICT_FP_P, STRICT_FP_S, STRICT_FN_P, STRICT_FN_S, logdir = merge_result(results)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    get_result(p_diff_list, s_diff_list, TP_P, FP_P, FN_P, TP_S, FP_S, FN_S, STRICT_FP_P, STRICT_FP_S, STRICT_FN_P, STRICT_FN_S, logdir)
