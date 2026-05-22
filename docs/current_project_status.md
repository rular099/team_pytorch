# TEAM PyTorch 当前项目状态

更新日期：2026-05-21

本文档记录当前 `team_pytorch` 项目的工程状态、已完成实验、主要结论和下一步建议。当前工程重点包括基于 `target_cross_attention` readout 的 PGA 全数据训练，以及 Japan 训练集 origin time 校正、理论 P 到时诊断和 Hi-net velocity 对照检查。

## 0. 最新数据管线状态（2026-05-21）

新增 JMA hypocenter origin time 校正流程、Japan 训练集重建脚本，以及 Hi-net velocity 数据检查与下载容错流程。当前判断是：加速度和速度波形在图上基本对齐，原先约 50 秒量级的理论 P 偏差主要来自部分训练集事件 origin time 只有分钟精度，而不是台站波形时间标记错误。最新训练集构建策略新增严格 DiTing pick 模式：优先使用 velocity DiTing P pick，缺失时使用 acceleration DiTing P pick，两者都缺失的 station 不写入训练集。

当前相关入口：

| 文件 | 状态 |
| --- | --- |
| `tools/fetch_jma_hypocenters.py` | 从 JMA daily hypocenter 页面抓取秒级震源时刻，并与训练集事件按时间、经纬度、深度、震级匹配，生成 origin correction CSV。 |
| `jma_origin_corrections/japan_2024_origin_corrections.csv` | 2024 年 Japan 训练集 origin time 校正表。当前 1004 个训练事件中 981 个 accepted、19 个 ambiguous、4 个 no candidate。 |
| `build_japan_training_data.py` | 新增 `--origin_corrections_csv` 和 `--final_pick diting_vel_then_acc`，在理论 P 到时和 pick 修复前先应用 JMA 秒级 origin time，并可按 DiTing pick 可用性过滤 station。 |
| `rebuild_japan_training_data.sh` | 本地重建训练集脚本，默认使用 JMA correction CSV，并可选择 DiTing pick 或 STA/LTA final pick。 |
| `rebuild_japan_training_data_server.sh` | 服务器重建训练集脚本，基于服务器路径和 checkpoint 默认值，可自动生成缺失的 JMA correction CSV。 |
| `rebuild_japan_training_data_server_diting_vel_acc.sh` | 服务器严格 DiTing pick 数据集重建 preset，默认输出到 `/opt/zb/data/japan_origin_corrected_diting_vel_acc`。 |
| `tools/download_hinet_velocity.py` | 根据训练 HDF5 中的事件和台站，下载匹配 Hi-net velocity raw WIN32 数据，并写 station-level MiniSEED。 |
| `tools/plot_hinet_accel_velocity_qc.py` | 画同一事件/台站的训练加速度与 Hi-net 速度，上下两行按理论 P 到时对齐，并标出训练数据中的各类 P pick。 |
| `docs/hinet_velocity_download.md` | 记录下载、fallback、MiniSEED 和 QC 绘图用法。 |

关键实现细节：

- `fetch_jma_hypocenters.py` 已生成
  `jma_origin_corrections/japan_2024_origin_corrections.csv` 和 summary。accepted
  校正的 median 为 `+28.2 s`，p05/p95 为 `+0.9/+56.7 s`。例如
  `20240101185300` 从训练集原始 `2024-01-01T18:53:00+09:00` 校正为
  JMA `2024-01-01T18:53:49.900+09:00`。
- `japan_dataset_builder.py` 在应用 origin correction 时保留原始 origin 字段，并写入
  `Origin_Time(JST)_Raw`、`Origin_Time_Correction_Source`、
  `Origin_Time_Correction_Status`、`Origin_Time_Correction_S` 和
  `Origin_Time_JMA_Event_ID`。
- `--final_pick diting_vel_then_acc` 会优先把 `p_pick_diting_vel_aligned`
  写入 `p_picks`；若 velocity pick 缺失但 `p_pick_diting_acc_aligned`
  有效，则使用 acceleration pick；两者都缺失的 station 会在写 HDF5
  前过滤掉。过滤后若 event 剩余 station 数小于 `--min_stations`，整个
  event 会被跳过。
- DiTing candidate pick 不再在 HDF5 中用 repaired/STALTA pick 回填；
  `diting_vel_pick_valid` 和 `diting_acc_pick_valid` 显式记录候选 pick
  是否存在且落在有效记录范围内。训练代码仍只读取原有 `p_picks` 字段。
- 训练集 HDF5 和事件 CSV 新增理论 P 覆盖诊断字段，包括理论 P 是否落在原始记录内、是否落在允许窗口内、相对记录起止的秒差，以及 pick clipping 原因。
- `theoretical_p_record_coverage_summary.csv` 和
  `theoretical_p_record_offset_hist.png` 用于批量检查理论 P 与记录窗口的关系。
- `download_hinet_velocity.py` 新增 `--origin-corrections`，Hi-net 下载窗口也使用同一套秒级 origin correction，避免训练集重建和速度下载使用不同的起点。
- HinetPy 的一分钟 `.cnt/.euc.ch` 临时文件现在落到
  `output-root/raw/<event_id>/segments/`，不再污染仓库当前目录。
- 若本机缺少 `catwin32` 导致合并 raw WIN32 失败，manifest 记录
  `raw_status=downloaded_unmerged`，并保留 segments 作为权威 raw 数据。
- MiniSEED 写出不再强依赖 `catwin32/win2sac_32`。当只有 raw segments
  可用时，脚本用纯 Python WIN32 parser 抽取匹配 Hi-net 台站的 U/N/E
  分量，写出 raw-count MiniSEED。
- `plot_hinet_accel_velocity_qc.py` 默认以理论 P 到时为 `t=0`，窗口为
  P 前后各 50 秒，输出 PNG 和 `qc_summary.csv`。默认图只保留 theoretical
  P 竖线，PGA、trigger、final pick 改为时间轴小三角标记；candidate
  picks 和 search windows 可通过 `--show-candidate-picks`、
  `--show-search-windows` 显式打开。summary 中仍包含各类 training pick
  相对理论 P 的秒差，便于批量筛查台站时间标记偏移。

已做离线验证：

- `python -m py_compile build_japan_training_data.py japan_dataset_builder.py tools/download_hinet_velocity.py tools/fetch_jma_hypocenters.py tools/plot_hinet_accel_velocity_qc.py`
- `bash -n rebuild_japan_training_data.sh rebuild_japan_training_data_server.sh rebuild_japan_training_data_server_diting_vel_acc.sh`
- 用已有 `202401011852/1853/1854...cnt` 和 `01_01_20240101.euc.ch`
  离线写出 `ISKH01/ISKH02/ISKH03` 的三分量 MiniSEED，ObsPy 可读回。
- JMA correction 抓取覆盖 2024 年训练事件：`matched=981`、`ambiguous=19`、
  `no_candidate=4`、`accepted_count=981`。
- QC 样例显示加速度和速度波形基本对齐；用 JMA 秒级 origin time 后，理论 P
  与人工可见 P 波的系统性大偏差应显著减小。

## 1. 当前目标

项目目标是用 DiTing backbone 提取单台波形表征，再接 TEAM 风格的多台站 Transformer，预测目标台站 PGA。

当前约束和实验原则：

- 不使用显式振幅信息：`use_amplitude_info=false`。
- PGA 输出使用 point regression，不使用 MDN。
- PGA readout 使用 target cross-attention。
- single-station pretrain 继续作为 station encoder/adaptor 的预训练阶段。
- 重点解决 PGA 动态范围压缩和泛化回缩问题。

当前主要训练入口：

```bash
bash train_light_slurm.sh <config.json>
```

只跑评估：

```bash
bash eval_checkpoint_slurm.sh <config.json>
```

## 2. 当前模型方案

当前主线模型链路：

```text
waveform
  -> frozen DiTing encoder
  -> trainable station adapter
  -> station feature embedding
  -> coordinate feature fusion
  -> TEAM transformer
  -> event cross-attention readout
  -> PGA target cross-attention readout
  -> point PGA output
```

PGA readout 当前采用：

```json
"pga_readout_mode": "target_cross_attention",
"event_readout_mode": "event_cross_attention"
```

含义：

- PGA target query 不再直接混入 TEAM transformer token 序列。
- 输入台站先通过 transformer 得到 station tokens。
- 每个 PGA target 用自己的 query cross-attend station tokens。
- PGA query 不是纯位置编码，代码中已经加入 learned query token。
- cross-attention 使用 station valid mask，避免无效台站参与。

已修复过的问题：

- 早期 `pga attn = 0.11111` 的诊断 bug 已修复。
- PGA target query 已加入可学习 token。
- `eval_checkpoint.py` 中 PGA normalization 反归一化时 `config` 作用域错误已修复。

## 3. 关键代码状态

核心文件：

| 文件 | 当前作用 |
| --- | --- |
| `gemini_models.py` | 模型定义；包含 `CrossAttentionReadout`、PGA target cross-attention、relative geometry bias、PGA target normalization loss、station decorrelation loss。 |
| `train_light.py` | 训练入口；支持 single-station pretrain、PGA target normalization、station decorrelation regularization、best/last checkpoint、可配置 LR decay factor 和 min LR。 |
| `eval_checkpoint.py` | 评估入口；支持 full model 和 single-station model 评估，PGA normalization 反归一化后再算指标。 |
| `train_light_slurm.sh` | 超算训练脚本；训练后可自动 eval。 |
| `eval_checkpoint_slurm.sh` | 只跑 eval 的超算脚本；默认同时评估 `full_model_last.pth` 和 `full_model_best.pth`。 |

当前新增或重要配置：

| 配置 | 用途 |
| --- | --- |
| `pga_configs/transformer_japan_overfit_pga15_stage1_b0_baseline_noamp_chaosuan.json` | stage1 baseline。 |
| `pga_configs/transformer_japan_overfit_pga15_stage1_b1_single_pga08_noamp_chaosuan.json` | single-station PGA 权重提高到 0.8。 |
| `pga_configs/transformer_japan_overfit_pga15_stage1_b2_single_pga_only_noamp_chaosuan.json` | single-station 只训 PGA。 |
| `pga_configs/transformer_japan_overfit_pga15_stage1_b3_pga_norm_noamp_chaosuan.json` | PGA target normalization。 |
| `pga_configs/transformer_japan_overfit_pga15_stage1_b4_mse_noamp_chaosuan.json` | full model 使用 MSE。 |
| `pga_configs/transformer_japan_overfit_pga15_stage1_b5_huber3_noamp_chaosuan.json` | Huber delta=3。 |
| `pga_configs/transformer_japan_overfit_pga15_stage1_b6_station_decor_1em4_noamp_chaosuan.json` | station embedding decorrelation weight=1e-4。 |
| `pga_configs/transformer_japan_overfit_pga15_stage1_b7_relative_geometry_noamp_chaosuan.json` | PGA cross-attention 加 relative geometry bias。 |
| `pga_configs/transformer_japan_full_pga15_b1_single_pga08_noamp_lr_chaosuan.json` | 全数据 b1 主线配置。 |
| `pga_configs/transformer_japan_full_pga15_b1_b3_b5_b7_noamp_lr_chaosuan.json` | 全数据综合配置：b1+b3+b5+b7。 |

## 4. Stage1 实验设置

Stage1 实验使用 `overfit_n=128`，主要目的是筛选影响 PGA 性能的因素。

共同设置：

- `use_amplitude_info=false`
- `n_pga_targets=15`
- `output_distribution=point`
- `pga_readout_mode=target_cross_attention`
- `event_readout_mode=event_cross_attention`
- `random_input_station_count=[3,5,8,12,16,25]`
- `select_first_inputs=true`
- `select_first_pga_targets=false`
- `res_comps=["pga"]`

已完成 8 个实验：

| ID | 因素 |
| --- | --- |
| b0 | baseline，无振幅。 |
| b1 | single-station pretrain 中 PGA 权重提高到 0.8，mag/epidist 各 0.1。 |
| b2 | single-station pretrain 只训 PGA。 |
| b3 | PGA label 使用训练集 mean/std 标准化。 |
| b4 | full model 损失改为 MSE。 |
| b5 | full model 使用 Huber delta=3。 |
| b6 | station embedding decorrelation 正则，weight=1e-4。 |
| b7 | PGA target cross-attention 加 relative geometry bias。 |

目前 `chaosuan_res/weights_japan_overfit_pga15_stage1*` 中的 `eval_results.txt` 是 `full_model_last.pth` 的结果。后续仍建议对 `full_model_best.pth` 全部重跑一次 eval。

## 5. Stage1 结果总结

PGA 总指标如下：

| 实验 | Train MAE | Train R2 | Train slope | Val MAE | Val R2 | Val slope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| b0 baseline | 0.2817 | 0.5757 | 0.6042 | 0.4254 | 0.2402 | 0.3961 |
| b1 PGA 权重 0.8 | **0.2243** | **0.7118** | 0.7100 | 0.4042 | 0.2981 | 0.4661 |
| b2 只训 PGA | 0.2770 | 0.5940 | **0.7438** | 0.4535 | 0.1856 | 0.4828 |
| b3 target norm | 0.2248 | 0.6924 | 0.6578 | 0.4262 | 0.2475 | 0.4106 |
| b4 MSE | 0.4460 | 0.1282 | 0.7444 | 0.5124 | -0.0687 | **0.4948** |
| b5 Huber delta=3 | 0.2820 | 0.5813 | 0.6279 | 0.4068 | 0.2937 | 0.4051 |
| b6 decor 1e-4 | 0.2680 | 0.5870 | 0.5673 | 0.4207 | 0.2840 | 0.3807 |
| b7 relative geometry | 0.2489 | 0.6321 | 0.5339 | **0.3940** | **0.3607** | 0.3651 |

主要结论：

1. `b1` 在 train set 上最有效。
   - Train MAE/R2/slope 都明显改善。
   - Val 也优于 baseline，是稳定有效因素。
   - 保留 mag/epidist 辅助任务比只训 PGA 更好。

2. `b3` 在 train set 上有效，但 val 不明显。
   - Target normalization 能增强训练拟合。
   - 泛化收益不稳定，但作为组合因素仍值得尝试。

3. `b5` 有正信号。
   - Val MAE 接近 b1。
   - Huber delta=3 改变 loss 尺度，不能直接用训练 loss 和 delta=1 比。
   - 可作为组合候选。

4. `b7` 在 val set 上最有效。
   - Val MAE/R2 最好。
   - 提升更像来自泛化，而不是单纯增强 train 拟合。
   - relative geometry 应作为后续主线因素保留。

5. `b2` 和 `b4` 不建议继续。
   - `b2` slope 较高，但 MAE/R2 变差。
   - `b4` val R2 为负，MSE 不适合作为当前主线。

6. `b6` 当前权重下效果有限。
   - 只比 baseline 略好。
   - 没有明显解决 station feature 趋同问题。

## 6. 动态范围和 station feature 观察

PGA 动态范围仍然压缩。所有 stage1 实验的 val slope 都显著小于 1。

当前最值得注意的是：

- `b1` 提高 val slope 到 `0.4661`，对动态范围最有帮助。
- `b7` val MAE/R2 最好，但 val slope 只有 `0.3651`，说明它提升了误差和泛化，但没有直接解决动态范围压缩。
- `b2/b4` slope 较高，但 MAE/R2 较差，不能只看 slope。

station feature cosine 诊断显示 feature 仍然有趋同倾向：

| 实验 | eval sample station feature off-diag cosine mean |
| --- | ---: |
| b0 | 0.9134 |
| b1 | 0.9544 |
| b2 | 0.9294 |
| b3 | 0.8966 |
| b4 | 0.9076 |
| b5 | 0.9219 |
| b6 | 0.9359 |
| b7 | 0.9328 |

这个诊断不能单独决定性能：`b1` cosine 最高但性能很好。当前更应把它作为表征监控指标，而不是唯一优化目标。

## 7. 当前推荐的全数据训练

### 7.1 主线综合配置

当前建议先跑综合配置：

```text
pga_configs/transformer_japan_full_pga15_b1_b3_b5_b7_noamp_lr_chaosuan.json
```

组合因素：

- b1: single-station pretrain 权重 `mag=0.1, epidist=0.1, pga=0.8`
- b3: PGA target normalization，训练集自动统计 mean/std
- b5: `full_model_huber_delta=3.0`
- b7: `pga_distance_bias=true`
- 不使用振幅
- 不使用 AdamW
- 不使用 station decor

关键训练参数：

```json
"epochs_full_model": 120,
"lr": 0.001,
"lr_adapter": 0.001,
"lr_team": 0.001,
"lr_monitor": "val",
"lr_decay_patience": 12,
"lr_decay_factor": 0.5,
"min_lr": 1e-5
```

注意：`pga_target_normalization` 会在训练开始时从训练集统计 mean/std，并写入实验目录下的 `config.json`。后续 eval 必须优先使用训练目录里的 `config.json`，不要直接用原始 `"auto"` 配置评估。

### 7.2 对照配置

可作为对照的配置：

```text
pga_configs/transformer_japan_full_pga15_b1_single_pga08_noamp_lr_chaosuan.json
```

该配置只使用 b1 加新的 LR 策略，不包含 target norm、Huber delta=3 和 relative geometry。它适合作为全数据 baseline。

## 8. 建议运行方式

全数据训练：

```bash
bash train_light_slurm.sh pga_configs/transformer_japan_full_pga15_b1_b3_b5_b7_noamp_lr_chaosuan.json
```

只跑 eval：

```bash
bash eval_checkpoint_slurm.sh weights_japan_full_pga15_b1_b3_b5_b7_noamp_lr/config.json
```

`eval_checkpoint_slurm.sh` 默认会评估：

```text
full_model_last.pth
full_model_best.pth
```

并分别输出：

```text
eval_results_last.txt
eval_results_last.npz
eval_results_best.txt
eval_results_best.npz
```

如果只想评估指定 checkpoint：

```bash
EVAL_CHECKPOINT=weights_japan_full_pga15_b1_b3_b5_b7_noamp_lr/full_model_best.pth \
bash eval_checkpoint_slurm.sh weights_japan_full_pga15_b1_b3_b5_b7_noamp_lr/config.json
```

## 9. 后续待办

优先级较高：

1. 对 8 个 stage1 实验全部补跑 `full_model_best.pth` eval，确认 last/best 排序是否一致。
2. 跑全数据综合配置 `b1+b3+b5+b7`。
3. 同时保留 b1-only 全数据配置作为对照。
4. 评估时同时看 train/val、MAE/RMSE/corr/R2/slope，不要只看 val MAE。
5. 重点观察 full-data 上 val slope 是否仍严重小于 1。

可选后续：

1. 如果综合配置过拟合，考虑去掉 b3 target normalization 或降低学习率。
2. 如果动态范围仍压缩，考虑专门设计 slope/range calibration loss，但应作为第二阶段实验。
3. station decor 当前 `1e-4` 效果弱，不建议直接加入主线；如继续研究，需要单独调权重。
4. AdamW 暂不作为当前主线，避免和 b1/b3/b5/b7 组合因素混杂。
