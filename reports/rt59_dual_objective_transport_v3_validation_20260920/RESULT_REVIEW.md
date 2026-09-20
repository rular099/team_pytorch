# RT59-v3 dual-objective transport：epoch-8 validation 结果审阅

更新日期：2026-09-20（Asia/Shanghai）

本目录汇总 RT59-v3 seed42 固定 8 epoch 训练及同一 epoch-8 checkpoint 的
deterministic random-geometry / normal validation。所有主比较都使用各 NPZ 内同次导出的
frozen RT57 gamma=1.66 base，未使用 held-out test，也没有从多个 epoch 中挑选 checkpoint。

原始 object-array NPZ 约 360 MB（解压后），保留在用户的超算结果包中，不提交 Git。
本目录提交轻量的完整门槛表、机器可读摘要、分层计数、训练摘要、前向控制和图。

## 1. 结论先行

严格按预声明的 conjunctive 条件，本轮结论是 **NO_GO：29 项必需门槛中 22 项通过、
7 项失败**。但这不是“整体无效”：RT59 是目前证据中第一次在同一 checkpoint 上让
random 与 normal 点误差都相对同次 frozen base 明确改善的方案。

- random MAE 从 `0.245578` 降到 `0.237828`，配对变化 `-0.007749 dex`，
  event-cluster 95% CI `[-0.008785, -0.006701]`；R2 从 `0.367504` 升到
  `0.394995`，NLL 和 Brier 同时改善。
- normal all/input/non-input 的 MAE 和 RMSE 全部改善；all 与 non-input 的四个配对
  CI 上界均小于 0。non-input MAE 从 `0.210371` 降到 `0.199301`，是最明显的收益。
- requested 1 s random MAE 为 `0.243391`，**通过** `<=0.253` 门槛；相对 base 改善
  `-0.011700 dex`，95% CI `[-0.014884, -0.008461]`。
- 主要未解决问题仍是空间动态范围：all-field 和 actual-one-station range ratio 均比
  base 更低；单站 range absolute error 和 pairwise-delta MAE 也没有改善。
- slope 仅从 `0.400316` 升到 `0.402453`，仍低于 `0.45`。random/normal 的 1-sigma
  coverage 相对 base 变化分别为 `+0.020713` 和 `+0.014860`，超过预设的 `0.01`
  retention budget；但这不等于 NLL/Brier 变差，二者实际上都改善。

因此最准确的表述是：**局地/远程分路和分组目标显著改善了 pointwise accuracy 与
probability scores，但当前 transport correction 仍在压缩 field contrast，尚未满足
空间场预测的开发锁定条件。**

## 2. 身份、训练和协议

- repository：`rular099/team_pytorch`
- branch：`rt59-dual-objective-transport-v3`
- RT59 implementation commit：`54fd63de2d17b771049943013d45265f32520137`
- epoch-statistics fix commit：`7e00824b3aab7a15286dfd9bf9a264b03080c1cd`
- submitted-source manifest SHA-256：
  `3e1164bc4fb0441fb33e5d8cd03aa1708416708d930b01e71c4a82fd3779715e`
- run：`weights_japan_full_2000_2024_rt59_dual_objective_transport_v3_seed42_retry1`
- seed：42；固定 8 个 epoch；checkpoint metadata：`epoch=8`、
  `loss=0.7177332043647766`、`checkpoint_format=non_encoder_v1`
- trainable scope：`dual_anchor_residual_only`，旧 RT57/RT58 表示冻结
- training data：resolved config 明确列出 2000--2024 共 25 个 HDF5 年度 shard，
  因而本次使用的是计划中的全量训练数据，而非 smoke 子集
- validation：Japan 2018 fixed validation；random 与 normal 各 9,681 realtime rows
- PGA 坐标：`log10(m/s^2)`；point estimate：MDN predictive mixture mean
- uncertainty：按 `event_id` 聚类的 paired percentile bootstrap，5,000 draws，
  seed `20260915`；同一事件的重复实时时刻一起重采样

结果包没有包含 `full_model_last.pth`，因此本地不能给 checkpoint 本体计算 SHA-256；
两份 formal metrics JSON 的 checkpoint path 与 metadata 完全一致。归档和输入文件的
SHA-256 见 `artifact_manifest.sha256`。

## 3. Random validation

共 1,310 events、9,681 realtime rows、75,654 valid non-input targets。

| Metric | frozen base | RT59 final | Change |
|---|---:|---:|---:|
| MAE | 0.245578 | 0.237828 | -0.007749 |
| RMSE | 0.318243 | 0.311250 | -0.006993 |
| bias | +0.034466 | +0.006955 | -0.027511 |
| R2 | 0.367504 | 0.394995 | +0.027492 |
| slope | 0.400316 | 0.402453 | +0.002138 |
| NLL | 0.217530 | 0.199018 | -0.018513 |
| Brier at -1.2 | 0.189875 | 0.183952 | -0.005923 |
| coverage 1 sigma | 0.681762 | 0.702474 | +0.020713 |
| coverage 2 sigma | 0.948198 | 0.949758 | +0.001560 |

Random paired event-cluster intervals：

| Difference (final - base) | Point | 95% CI |
|---|---:|---:|
| MAE | -0.007749 | [-0.008785, -0.006701] |
| RMSE | -0.006993 | [-0.008683, -0.005258] |
| Brier | -0.005923 | [-0.006920, -0.004930] |

Point error、R2、NLL、Brier 都有方向一致的收益，且 MAE/RMSE/Brier 的 CI 均排除 0。
失败项不是总体误差，而是 slope 和空间场动态范围。

### 3.1 Requested 1 s

| Metric | frozen base | RT59 final | Change |
|---|---:|---:|---:|
| MAE | 0.255091 | 0.243391 | -0.011700 |
| RMSE | 0.327924 | 0.318425 | -0.009499 |
| bias | +0.065257 | +0.015624 | -0.049633 |
| R2 | 0.247569 | 0.290528 | +0.042959 |
| NLL | 0.205939 | 0.183957 | -0.021982 |
| Brier | 0.199188 | 0.190546 | -0.008642 |

这里有 16,348 targets。MAE CI 为 `[-0.014884, -0.008461]`，RMSE CI 为
`[-0.015073, -0.003502]`。因此本轮早时刻结果不是临界通过，而是明确改善。

### 3.2 Spatial-field 指标

至少 5 个 targets 的 5,762 个 realtime fields，actual-one-station 子集 1,494 fields。

| Metric | frozen base | RT59 final | Change |
|---|---:|---:|---:|
| all-field mean range ratio | 0.504499 | 0.467950 | -0.036549 |
| all-field range absolute error | 0.538608 | 0.566397 | +0.027790 |
| one-station mean range ratio | 0.203564 | 0.174563 | -0.029001 |
| one-station range absolute error | 0.793189 | 0.820196 | +0.027007 |
| one-station pairwise-delta MAE | 0.336126 | 0.336202 | +0.000076 |

这些结果表明 RT59 的总体点误差收益没有转化为 field contrast 收益。尤其单输入站时
没有多站聚合歧义，动态范围仍明显收缩，瓶颈至少包含 station-to-query transport 或
该分支的优化目标，不应只归因于 aggregation。

## 4. Normal validation

共 1,383 events、89,770 targets；formal target type 分为 63,651 input 和 26,119
non-input。observable route 恰好得到 63,651 observed、0 ambiguous，与 formal input
计数完全对齐，但报告仍保留两种语义，不用 route 覆写 formal population。

| Population / metric | frozen base | RT59 final | Change | 95% paired CI |
|---|---:|---:|---:|---:|
| all MAE | 0.131016 | 0.125006 | -0.006009 | [-0.006467, -0.005522] |
| all RMSE | 0.195544 | 0.186050 | -0.009493 | [-0.010406, -0.008499] |
| input MAE | 0.098452 | 0.094520 | -0.003932 | [-0.004236, -0.003603] |
| input RMSE | 0.155096 | 0.148322 | -0.006775 | [-0.007447, -0.005972] |
| non-input MAE | 0.210371 | 0.199301 | -0.011071 | [-0.012313, -0.009912] |
| non-input RMSE | 0.269814 | 0.255652 | -0.014161 | [-0.015917, -0.012490] |

Normal all 的 bias 从 `+0.053126` 降到 `+0.041032`，slope 从 `0.736897` 升到
`0.743491`；non-input bias 从 `+0.093456` 降到 `+0.072466`，slope 从 `0.501805`
升到 `0.511275`。四个 absolute bias/slope-error 保护项均通过。

| Probability metric, all targets | frozen base | RT59 final | Change |
|---|---:|---:|---:|
| NLL | -0.822244 | -0.848157 | -0.025913 |
| Brier | 0.100965 | 0.095412 | -0.005553 |
| coverage 1 sigma | 0.764509 | 0.779369 | +0.014860 |
| coverage 2 sigma | 0.973577 | 0.976986 | +0.003409 |

NLL、Brier 与点误差同时改善；1-sigma coverage 变化超过预设 retention budget，故该
gate 失败，但不应将它表述成所有概率校准指标都退化。

## 5. 前向与 feature-pairing 控制

`forward_control.json` 从同一两份 NPZ 复算，结果如下：

- final mean 与 `base + applied_delta` 的最大误差为 random `3.17e-7 dex`、normal
  `3.43e-7 dex`，与 mean-only shift 实现一致。
- mean absolute applied correction 为 random `0.047279 dex`、normal `0.020149 dex`；
  新 head 不再像 RT58 那样只产生极小 correction。
- 固定正确 base/event/query/geometry/masks，仅 roll 有效 selected u/d：多站 target 的
  applied delta 全部发生非零变化，平均绝对变化 random `0.011350 dex`、normal
  `0.010294 dex`。
- actual-one-station roll 是恒等置换，最大差异严格为 0；该控制实现正确。

这些证据支持“RT59 head 使用了正确 station-feature pairing”，但不能把 roll sensitivity
本身当作空间精度改善；正式 spatial metrics 仍然失败。

## 6. 训练轨迹

`training_summary.csv` 保存全部 8 epoch 的关键标量。训练完整执行：每个 RT59 epoch
scalar 均有 8 行，random / normal-observed / normal-remote 三组 target 计数均非零。

- validation epoch loss 单调从 `0.726080` 降至 `0.717733`；train epoch loss 在
  `0.896740--0.916550` 间波动，没有出现发散。
- relative loss 从 `0.329543` 降至 `0.311470`；candidate loss 从 `0.255250` 降至
  `0.250577`；difference loss 只从 `0.358180` 降至 `0.354861`，改善较小。
- regret loss 从 `0.003480` 升至 `0.005539`，说明随 correction 增强仍存在少量 normal
  target 比 frozen base 更差；这与 guard 是软约束而非硬选择一致。
- local/transport 分支的 epoch-average pre-clip norms 均低于 1；逐 batch 仍发生少量
  clipping，post-clip 均小于等于 pre-clip。两分支均获得了非零梯度。
- 固定学习率日程按预设完成：epoch 1--4 为 `5e-4`，5--6 为 `2.5e-4`，7--8 为
  `1.25e-4`。

训练曲线和单次 validation 不能识别 local/transport、分组归约、MSE、guard 各自的独立
贡献；本轮只能评价整个联合方案。

## 7. 预声明 gate 审计

完整 29 项见 `gates.csv` 和 analyzer 生成的 `README.md`。7 个失败项是：

| Failed gate | Value | Required |
|---|---:|---:|
| random slope | 0.402453 | >= 0.45 |
| random all-field range ratio | 0.467950 | >= 0.55 |
| random one-station range ratio | 0.174563 | >= 0.25 |
| random one-station pairwise-delta MAE | 0.336202 | <= 0.330 |
| random coverage-1 absolute change | 0.020713 | <= 0.01 |
| one-station range absolute error change | +0.027007 | <= 0 |
| normal coverage-1 absolute change | 0.014860 | <= 0.01 |

其余 22 项通过，包括 random MAE/R2/requested-1-s/NLL/paired MAE and Brier、normal
all/input/non-input point error、历史红线、NLL/Brier、2-sigma coverage、bias/slope-error
保护项。因为判据明确为 conjunctive，不可用 `22/29` 多数票改判 GO。

## 8. 可支持与不可支持的解释

### 已验证事实

- 固定 epoch-8 checkpoint 同时改善 random 与 normal 的 pointwise MAE/RMSE，并改善
  NLL/Brier；收益相对同次 frozen base 严格配对。
- local/transport observable route 工作正常，normal 无 ambiguous target。
- feature-pairing control 在多站时敏感、单站时严格恒等。
- 空间动态范围和单站 field-difference 指标没有改善，且多数比 base 更差。

### 合理推断

- RT59 的独立 local path 与 grouped objective 成功缓解了 RT58 的 random/normal 冲突。
- 剩余首要瓶颈更接近 transport 分支的空间对比表达/监督，而不是总体 pointwise
  calibration 或“模型完全不使用 waveform feature”。
- 由于 fixed MDN sigma，mean correction 会改变 coverage；NLL/Brier 改善而 1-sigma
  retention 失败提示下轮应明确区分 point uncertainty calibration 与空间 range 目标。

### 尚不支持

- 不能称本轮达到预声明 GO，也不能称空间 PGA field 已解决。
- 单 seed validation 不覆盖训练 seed 不确定性，更不是 held-out test/generalization 结论。
- 未包含 checkpoint 本体，不能在本地审计 checkpoint SHA-256 或重新推理。
- 不能从这一个联合实验给各个新增机制分配独立因果贡献。

## 9. 给 ChatGPT Project 的审阅请求

请优先读取本文件、`README.md`、`summary.json`、`gates.csv`、
`training_summary.csv` 和 `forward_control.json`，然后：

1. 核对 protocol、base/final 定义、formal input 与 observable route、bootstrap direction
   及 29 项门槛；确认 **NO_GO（22 PASS / 7 FAIL）** 是否判定正确。
2. 区分“点预测/概率分数双提升”与“空间动态范围失败”，不要把 1-sigma coverage gate
   失败误写成 NLL/Brier 失败，也不要误写 requested 1 s 为失败。
3. 判断 RT59 epoch8 是否应保留为当前最强的 pointwise development checkpoint，同时因
   spatial gates 未过而不做最终锁定。
4. 基于现有 NPZ、训练曲线和 pairing control，给出一个首要根因判断；明确哪些是事实、
   哪些只是推断。
5. 只提出一个下一步最高价值实验，目标应直接解决 spatial range / one-station field
   difference；不要要求重复 smoke、旧 query diagnostics、formal test，或可由现有 NPZ
   回答的检查。
6. 若需改代码，输出完整 `AI-HANDOFF`，明确 architecture、loss、trainable scope、
   RT55--RT59 compatibility、单元测试、Slurm 和 validation-only go/no；除非有新的实质性
   理由，不扩大为 sweep、多 seed 或重复全量评估。

## 10. 目录说明

- `README.md`：analyzer 自动生成的简明 gate 表
- `summary.json`：全部 group/stratum/CI/gate 的机器可读结果
- `gates.csv`：29 项预声明门槛
- `group_metrics.csv`：base/final 分组指标
- `paired_ci.csv`：event-cluster paired intervals
- `strata_counts.csv`：time/station-count/target-type/route 的真实计数
- `truth_prediction_density.png`：统一坐标和色阶的四组 truth-prediction 图
- `training_summary.csv`：8 epoch 训练、验证、loss、梯度和 target 计数
- `forward_control.json`：route、mean-shift identity 和 feature-roll 控制
- `artifact_manifest.sha256`：用户结果归档及原始核心文件身份

核心分析命令：

```bash
python tools/analyze_rt59_dual_objective_npz.py \
  --random_npz '<archive>/epoch8_random_validation/eval_validation_epoch8_random.npz' \
  --normal_npz '<archive>/epoch8_normal_validation/eval_validation_epoch8_normal.npz' \
  --output_dir reports/rt59_dual_objective_transport_v3_validation_20260920
```
