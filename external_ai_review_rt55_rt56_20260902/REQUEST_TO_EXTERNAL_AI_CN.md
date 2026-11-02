# RT55 / RT56 PGA 模型独立代码审查请求

## 1. 请求目标

请基于本压缩包中的实际代码、配置和结果，对 RT55 / RT56 模型进行独立审查：

1. 梳理从 Slurm 启动、配置继承、数据生成、实时截断、输入台站选择、目标台站选择、模型前向、损失计算、checkpoint 加载到正式验证的完整调用链。
2. 分析下述现象可能产生的原因，并区分代码实现问题、模型结构问题、训练目标问题、数据采样问题、概率输出问题以及信息可辨识性限制。
3. 对可能原因按可信度和影响程度排序，每项结论尽量引用具体文件、函数和代码行作为证据。
4. 给出可以区分不同原因的最小诊断实验和消融实验。
5. 在完成原因分析后，再提出解决方案，并说明修改范围、训练代价、预期收益、风险以及验证标准。

本说明只陈述实验设置、已观察结果和研究约束，不预设问题原因，也不提供候选解决方案。请不要假定任何单一原因一定成立。

## 2. 研究任务

模型接收：

- 截至某个实时时刻的多台站三分量波形；
- 每个输入台站的地理位置；
- 一个或多个 PGA 查询目标台站的位置；
- 当前代码还包含震级和位置输出任务，但本次重点是 PGA。

模型输出目标台站的 PGA 概率分布。报告中的 PGA 使用：

```text
log10(PGA / (m/s²))
```

原始 LaTeX：

```latex
\log_{10}\!\left(\frac{\mathrm{PGA}}{\mathrm{m/s^2}}\right)
```

研究目标是：模型主要根据波形与输入/查询台站坐标，端到端学习事件、传播和空间变化规律，而不是要求推理阶段必须提供人工确定的震中距、传播路径参数或场地参数。请同时评估这一目标在当前数据条件下的可辨识边界。

## 3. 数据与划分

- 数据范围：日本 2000–2024 年，K-NET 台站。
- 事件级划分，划分后统计如下。

| split | events | station rows | unique stations | magnitude median | magnitude max | stations/event median |
|---|---:|---:|---:|---:|---:|---:|
| train | 9,627 | 146,842 | 960 | 3.9 | 9.0 | 10 |
| dev | 1,383 | 22,336 | 917 | 4.0 | 8.0 | 11 |
| test | 2,769 | 42,368 | 952 | 3.9 | 7.4 | 10 |

- RT55 正常协议在固定实时时刻 `1, 3, 5, 10, 20, 40, 90 s` 验证。
- RT56 随机几何验证中共有 1,310 个满足随机输入/非输入目标条件的验证事件、75,654 个目标。
- RT56 的坐标排除审计显示：目标坐标与已选择输入台站坐标在 `1e-6 degree` 容差内的重合数为 0。
- 原始年度 HDF5、权重文件和逐目标 NPZ 因体积原因未放入压缩包；正式 metrics JSON、精简结果表、关键图和运行配置快照已包含。

## 4. RT55 训练与验证协议

实际入口：

```text
tools/run_rt55_japan_full_2000_2024_slurm.sh
  -> train_light_slurm.sh
  -> train_light.py
```

配置继承：

```text
transformer_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_chaosuan.json
  -> transformer_japan_overfit_pga15_stage2_512_rt52_knet_legacy_paddingmask_cached_dpk_event_temporal_residual_scale4_chaosuan.json
```

主要协议事实：

- 实时训练开启，以第一个有效 P pick 为参考。
- 训练时间分为 7 个时间区间，每个事件每轮抽取其中 3 个区间。
- 正常协议优先按 P pick 顺序选择输入台站。
- RT55 的实时目标采样比例配置为：输入目标 0.3、已触发非输入目标 0.2、未触发目标 0.5，并在缺失时补齐。
- 当前重点 checkpoint 为 epoch 20 和 epoch 32。
- RT55 正式测试结果来自 epoch-20 checkpoint；epoch-32 当前有正常 validation 和 waveform-station-roll control。

## 5. RT56 训练与验证协议

实际入口：

```text
tools/run_rt56_random_geometry_slurm.sh
  -> train_light_slurm.sh
  -> train_light.py
```

配置继承：

```text
transformer_japan_full_2000_2024_rt56_ep32_mixed_random_geometry_seed42_chaosuan.json
  -> transformer_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_chaosuan.json
  -> transformer_japan_overfit_pga15_stage2_512_rt52_knet_legacy_paddingmask_cached_dpk_event_temporal_residual_scale4_chaosuan.json
```

主要协议事实：

- 从固定的 RT55 epoch-32 checkpoint 加载，重新初始化 optimizer 和 scheduler。
- 微调 6 个 epoch。
- 每个训练样本以 0.5 概率采用 RT55 原协议，以 0.5 概率采用 causal random input mask。
- 随机分支从 `[1, 3, 5, 8, 12, 16]` 中选择输入台站数上限。
- 随机分支只从当前实时时刻已经因果可用的触发台站中选择输入。
- 随机分支的目标排除已选输入台站。
- 随机分支目标采样配置为：输入目标 0.0、已触发非输入目标 0.2、未触发目标 0.5，并在缺失时补齐。
- validation 对所有样本启用 deterministic causal random geometry。
- zero-shot 使用 RT55 epoch-32 权重；fine-tuned 使用 RT56 epoch-6 best checkpoint。
- 启动参数中的 `--skip_single_station` 表示跳过独立 single-station checkpoint 分支，并不从随机几何验证数据中删除 `K=1` 样本。

## 6. 当前模型配置事实

请以代码和合并后的配置快照为最终依据。当前配置的关键项包括：

- DiTing 1200M backbone + attention-pooling adapter；
- `use_amplitude_info=true`；
- station waveform embedding 与 station coordinate feature 在当前模式下相加；
- station 和 PGA target 坐标都经过正弦位置编码；
- `use_coords_abs=true`；
- `use_coords_rel=false`；
- `use_coords_rel_abs_fusion=false`；
- `coord_fusion_mode=add`；
- `pga_readout_mode=target_cross_attention`；
- `pga_readout_layers=4`；
- `readout_first_residual=true`；
- `readout_residual_gates=true`；
- `readout_residual_gate_init=0.0`；
- `readout_ffn_gate_init=0.0`；
- `pga_distance_bias=false`；
- `use_rope=false`；
- `pga_use_event_context=false`；
- `use_vs30=false`；
- `station_context_mode=off`；
- PGA 输出为 3-component MDN；
- PGA target normalization 开启；
- RT56 没有改变模型结构，只改变 checkpoint 来源和数据采样协议。

## 7. RT55 主要结果

### 7.1 Epoch-20 正式 test

| target group | targets | MAE | RMSE | R² | corr | slope | bias | NLL | 1σ coverage | 2σ coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 170,767 | 0.1151 | 0.1777 | 0.7578 | 0.8714 | 0.7783 | 0.0116 | -0.8720 | 0.7004 | 0.9544 |
| input | 122,135 | 0.0834 | 0.1368 | 0.8636 | 0.9293 | 0.8617 | 0.0018 | -1.2164 | 0.7168 | 0.9621 |
| non-input | 48,632 | 0.1947 | 0.2527 | 0.4029 | 0.6574 | 0.5181 | 0.0363 | -0.0073 | 0.6590 | 0.9350 |
| triggered non-input | 7,109 | 0.1615 | 0.2119 | 0.6208 | 0.7914 | 0.6756 | 0.0135 | -0.0370 | 0.6054 | 0.8793 |
| untriggered | 41,523 | 0.2003 | 0.2591 | 0.3563 | 0.6275 | 0.4867 | 0.0402 | -0.0022 | 0.6682 | 0.9445 |

### 7.2 Epoch-32 normal validation

| target group | targets | MAE | RMSE | R² | corr | slope | bias | NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 89,770 | 0.1166 | 0.1806 | 0.7541 | 0.8721 | 0.7720 | 0.0288 | -0.8867 |
| non-input | 26,119 | 0.1986 | 0.2581 | 0.4117 | 0.6759 | 0.5188 | 0.0644 | 0.0023 |

### 7.3 单输入台站、同一样本内多目标近似常数现象

审计定义：

- 一个 realtime sample 至少有 5 个有效目标；
- prediction range 不超过 `0.03 dex`；
- truth range 至少为 `0.30 dex`。

审计结果：

| quantity | value |
|---|---:|
| eligible realtime samples | 13,467 |
| eligible targets | 156,171 |
| flat-output samples | 519 (3.854%) |
| flat + diverse-truth samples | 503 (3.735%) |
| one-station eligible samples | 518 |
| one-station flat samples | 518 (100%) |
| flat + diverse truth + sample MAE ≥ 0.3 | 154 (1.144%) |
| flat + diverse truth + sample MAE ≥ 0.5 | 15 (0.111%) |

两个已导出的具体样本：

| case | time | input stations | targets | truth range | prediction range | sample MAE |
|---|---:|---:|---:|---:|---:|---:|
| flat, low aggregate MAE | 1 s | 1 | 15 | 0.6399 | 0.0031 | 0.0996 |
| flat failure | 10 s | 1 | 15 | 1.9124 | 0.0057 | 1.0909 |

这里的 flat audit 衡量的是同一事件、同一截止时刻、多个目标之间的预测范围。下面的 station-count 分桶则把不同事件和不同目标汇总，因此二者统计对象不同。

### 7.4 RT55 epoch-20 test 的非输入目标按输入台站数分桶

| observed input count | targets | MAE | RMSE | R² | corr | slope | bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5,656 | 0.2619 | 0.3280 | 0.0523 | 0.4268 | 0.2763 | 0.0961 |
| 2–3 | 12,224 | 0.2119 | 0.2713 | 0.3019 | 0.5807 | 0.4174 | 0.0414 |
| 4–5 | 6,556 | 0.1904 | 0.2441 | 0.4043 | 0.6566 | 0.5152 | 0.0323 |
| 6–10 | 8,639 | 0.1802 | 0.2329 | 0.4645 | 0.6987 | 0.5781 | 0.0269 |
| 11–15 | 3,713 | 0.1648 | 0.2122 | 0.5605 | 0.7600 | 0.6604 | 0.0232 |
| ≥16 | 11,844 | 0.1670 | 0.2197 | 0.5683 | 0.7614 | 0.6531 | 0.0155 |

## 8. RT56 主要结果

### 8.1 Random-geometry validation

| run | targets | MAE | RMSE | R² | corr | slope | bias | NLL | 1σ coverage | 2σ coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RT55 ep32 zero-shot | 75,654 | 0.2831 | 0.3820 | 0.0885 | 0.4405 | 0.2959 | -0.0913 | 1.4739 | 0.4423 | 0.6918 |
| RT56 ep6 fine-tuned | 75,654 | 0.2572 | 0.3313 | 0.3144 | 0.5710 | 0.3478 | 0.0404 | 0.2553 | 0.6549 | 0.9411 |

强、弱目标的结果：

| run/group | targets | MAE | RMSE | R² | corr | bias |
|---|---:|---:|---:|---:|---:|---:|
| zero-shot weak | 40,596 | 0.2060 | 0.2670 | -1.6283 | 0.2345 | 0.1122 |
| zero-shot strong | 35,058 | 0.3724 | 0.4821 | -1.3342 | 0.3011 | -0.3269 |
| fine-tuned weak | 40,596 | 0.2543 | 0.3112 | -2.5699 | 0.2876 | 0.2297 |
| fine-tuned strong | 35,058 | 0.2607 | 0.3532 | -0.2531 | 0.4356 | -0.1788 |

### 8.2 RT56 fine-tuned random validation 按输入台站数分桶

| selected input count | targets | MAE | RMSE | R² | corr | slope | bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19,861 | 0.3035 | 0.3757 | 0.0523 | 0.3280 | 0.1430 | 0.0806 |
| 2–3 | 20,732 | 0.2524 | 0.3265 | 0.2963 | 0.5569 | 0.3411 | 0.0404 |
| 4–5 | 13,583 | 0.2410 | 0.3141 | 0.3697 | 0.6118 | 0.3912 | 0.0245 |
| 6–10 | 11,885 | 0.2311 | 0.3037 | 0.4539 | 0.6752 | 0.4645 | 0.0178 |
| 11–15 | 5,925 | 0.2288 | 0.2997 | 0.4747 | 0.6895 | 0.4767 | 0.0109 |
| ≥16 | 3,668 | 0.2249 | 0.2970 | 0.5668 | 0.7535 | 0.5452 | 0.0031 |

对应的 zero-shot `K=1` 为：MAE `0.3194`、RMSE `0.4142`、R² `-0.1518`、corr `0.2195`。微调后这些指标发生改善，但 `K=1` 仍是最困难分桶。

### 8.3 Normal-protocol retention

| run | target group | MAE | RMSE | R² | corr | bias | NLL |
|---|---|---:|---:|---:|---:|---:|---:|
| RT55 ep32 | all | 0.1166 | 0.1806 | 0.7541 | 0.8721 | 0.0288 | -0.8867 |
| RT56 ep6 | all | 0.1338 | 0.2000 | 0.6984 | 0.8512 | 0.0587 | -0.8163 |
| RT55 ep32 | non-input | 0.1986 | 0.2581 | 0.4117 | 0.6759 | 0.0644 | 0.0023 |
| RT56 ep6 | non-input | 0.2220 | 0.2838 | 0.2886 | 0.6570 | 0.1200 | 0.0468 |

## 9. 需要重点解释的现象

请不要把下列描述当作原因判断，它们只是待解释的观测：

1. RT55 全目标总体 MAE、RMSE 和 R² 较好，但 prediction-versus-truth 图仍表现出明显离散，并且 non-input 目标显著弱于 input 目标。
2. RT55 的 sample-level flat audit 中，满足审计条件的单输入台站样本全部呈现同一样本内多目标近似常数输出；部分样本的 truth spatial range 很大。
3. RT55/RT56 的输入台站数越多，非输入目标或随机目标的 R²、相关性和 slope 总体越高。
4. RT56 mixed-random 微调显著改善 random-geometry 总体指标和概率覆盖，但 random `K=1` 仍较弱。
5. RT56 微调改善 strong-target MAE，但 weak-target bias、R² 和部分 calibration 指标仍不理想。
6. RT56 random-geometry 改善的同时，normal-protocol retention 相比 RT55 epoch-32 有一定下降，尤其是 non-input 目标。
7. 当前模型同时拥有输入台站和查询目标的位置编码，但单输入场景的同一样本多目标空间变化仍可能非常小。
8. 当前正式结果只来自一个 seed；尚无多 seed 方差。

## 10. 希望外部审查回答的问题

1. 当前代码中，输入台站坐标和目标坐标分别在哪些模块进入，经过哪些归一化、融合、attention、residual 和输出头？
2. 单输入台站、多目标近似常数输出可能在哪一层产生？如何通过 hook、坐标置换、梯度敏感度或中间表示方差来定位？
3. 当前现象是否可能由模型结构、训练采样、损失聚合、标签分布、目标归一化、MDN 均值计算、checkpoint 加载或验证导出逻辑造成？每种可能性的代码证据是什么？
4. 为什么总体 MAE 可以较好，而散点图、slope、强弱目标分桶和 non-input 指标仍显示明显问题？这些指标是否存在统计口径或样本权重差异？
5. RT56 微调后的概率分布、NLL、coverage、Brier 和 ECE 是否一致？是否存在通过扩大预测方差改善 NLL/coverage、但点预测空间结构仍不足的情况？
6. 从波形和坐标自动学习事件、传播和场地效应，在当前数据划分下哪些部分可辨识，哪些部分可能只能学习条件均值或空间先验？
7. 请提出最小修改方案和长期模型方案，并说明如何保持 RT55 checkpoint 加载及原 RT55 推理功能兼容。
8. 请给出严格的消融矩阵、成功标准和回归保护指标，至少覆盖 normal/random、input/non-input、`K=1/2–3/...`、震级、截止时刻、强弱 PGA 和概率校准。

## 11. 代码阅读建议

建议从以下顺序开始，但请自行验证实际调用关系：

1. `code/team_pytorch/tools/run_rt56_random_geometry_slurm.sh`
2. `code/team_pytorch/train_light_slurm.sh`
3. `code/team_pytorch/train_light.py`
4. `code/team_pytorch/gemini_util_light.py`
5. `code/team_pytorch/gemini_models.py`
6. `code/team_pytorch/eval_checkpoint.py`
7. 三层 `pga_configs/*.json`
8. `results/run_snapshots/*_merged_config.json`
9. `results/rt55/tables/rt55_flat_output_sample_details.csv`
10. `results/rt55/tables/rt55_stratified_*.csv`
11. `results/rt56/tables/rt56_stratified_*.csv`
12. 正式 metrics JSON 和关键 PNG。

请在最终答复中明确区分：代码中可以直接确认的事实、根据结果作出的推断、仍需实验验证的假设。
