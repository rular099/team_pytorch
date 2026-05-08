# SESSION_SUMMARY.md

## 1. 项目当前目标

当前项目在 `team_pytorch` 中做 Japan PGA 预测实验，主线是验证 graph 结构的 PGA readout 是否能比原始 TEAM/Transformer readout 更好地利用多台站信息。

当前分支：

```text
zhangb/graph-pga-experiment
```

远端仓库：

```text
origin: github.com:rular099/team_pytorch.git
```

最终目标：

```text
在 single-station PGA prior 的基础上，通过 graph message passing / target-station 空间关系提升目标台站 PGA 预测性能。
```

重要背景：

- `target-to-station cross attention` 方向已经交给其他人推进。
- 本分支主要负责 `graph_message_passing` 结构实验。
- 旧 graph readout 曾经退化为近似常数预测。
- 当前采用 prior-residual 设计：

```text
final_pga = distance_weighted_single_station_prior_baseline + graph_residual
```

## 2. 当前完成状态

已经完成：

1. 新增 graph PGA readout 主体结构。
2. 新增 single-station PGA prior head 到 full graph model。
3. 支持从 single-station checkpoint 加载 `heads.pga.*` 到 `station_pga_prior_head.*`。
4. 支持 distance-weighted baseline。
5. 支持 graph residual 加到 baseline 上。
6. 修复超算数据路径：

```text
/public/home/test_bigmodel/seismogram/zb/team_pytorch/japan_2024.hdf5
```

7. 确认超算工作目录应为：

```text
/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_graph
```

8. 确认 `select_first_inputs=true` 在 graph model 下仍然有效，输入台站会优先选择 P pick 较早的台站。
9. 已生成 graph model 可用的 first-inputs prior-residual 配置。
10. 已新增 5 个消融实验配置。
11. 已提交并推送最近修改。

最近推送 commit：

```text
1199a46 Add graph PGA ablation configs
```

最近提交历史：

```text
1199a46 Add graph PGA ablation configs
b147ef2 Limit checkpoints to init best and last
6b2b0f0 Add graph PGA prior residual readout
11b2b97 Add graph PGA-only first-inputs config
3a82943 Fix first-input pick ordering dtype
```

当前工作区仍有未跟踪文件，不在最近提交内：

```text
GRAPH_PGA_PROJECT_SUMMARY.md
jieshu_prompt.txt
jixu_prompt.txt
```

不要默认把这些文件加入提交，除非用户明确要求。

## 3. 最近的重要修改

最近一次提交：

```text
1199a46 Add graph PGA ablation configs
```

修改文件：

```text
gemini_models.py
pga_configs/graph_japan_overfit_pga_first_inputs_baseline_only_chaosuan.json
pga_configs/graph_japan_overfit_pga_first_inputs_no_prior_chaosuan.json
pga_configs/graph_japan_overfit_pga_first_inputs_power05_chaosuan.json
pga_configs/graph_japan_overfit_pga_first_inputs_power20_chaosuan.json
pga_configs/graph_japan_overfit_pga_first_inputs_residual_only_chaosuan.json
```

### `gemini_models.py`

新增参数：

```python
graph_pga_residual_scale=1.0
```

作用：

- 控制 graph residual 输出的缩放。
- `1.0` 表示正常训练。
- `0.0` 表示 residual 被关闭，只保留 distance baseline。

新增诊断指标：

```text
pga_residual_mean
pga_residual_std
pga_scaled_residual_mean
pga_scaled_residual_std
```

目的：

- 判断 residual 是否真的在输出有效变化。
- 判断 baseline-only 消融时 residual 是否被正确关掉。
- 判断当前模型预测方差偏小的问题是否来自 residual 太弱。

### 新增消融配置

#### baseline only

```text
pga_configs/graph_japan_overfit_pga_first_inputs_baseline_only_chaosuan.json
```

关键设置：

```json
"graph_pga_use_station_prior": true,
"graph_pga_use_distance_baseline": true,
"graph_pga_distance_baseline_power": 1.0,
"graph_pga_residual_scale": 0.0
```

含义：

```text
final_pga = distance_weighted_single_station_prior_baseline
```

注意：这是“baseline-only 输出”，但 `station_pga_prior_head` 默认仍可能参与训练。若要做严格 frozen single-station baseline，需要额外冻结 prior head。

#### residual only

```text
pga_configs/graph_japan_overfit_pga_first_inputs_residual_only_chaosuan.json
```

关键设置：

```json
"graph_pga_use_station_prior": true,
"graph_pga_use_distance_baseline": false
```

含义：

```text
使用 single-station prior 作为 graph 输入特征，但不把 distance baseline 直接加到最终输出。
```

#### no prior

```text
pga_configs/graph_japan_overfit_pga_first_inputs_no_prior_chaosuan.json
```

关键设置：

```json
"graph_pga_use_station_prior": false,
"graph_pga_use_distance_baseline": false,
"station_pretrain_load_pga_prior_head": false
```

含义：

```text
测试 graph readout 在没有 single-station PGA prior 帮助时是否仍能学习。
```

#### distance power 0.5

```text
pga_configs/graph_japan_overfit_pga_first_inputs_power05_chaosuan.json
```

关键设置：

```json
"graph_pga_distance_baseline_power": 0.5
```

含义：

```text
距离衰减更弱，远台站权重更高。
```

#### distance power 2.0

```text
pga_configs/graph_japan_overfit_pga_first_inputs_power20_chaosuan.json
```

关键设置：

```json
"graph_pga_distance_baseline_power": 2.0
```

含义：

```text
距离衰减更强，更依赖近台站。
```

## 4. 当前架构 / 设计决策

### Graph PGA Readout

核心类：

```text
gemini_models.py::GraphPGAReadout
```

关键输入：

```text
station embeddings
target station coordinates
station coordinates
station valid mask
target valid mask
station_pga_prior_values
```

edge feature 当前包括：

```text
target - station relative coordinates
distance
log1p(distance)
1 / distance
```

### Single-station PGA prior

`station_pga_prior_head` 不是完整 single-station model，而是 single-station model 里的 PGA head。

它的作用：

```text
每个输入台站 waveform -> embedding -> station_pga_prior_head -> 单台站 PGA prior
```

如果从 single-station checkpoint 加载权重，那么初始输出相当于 single-station model 对每个输入台站的 PGA 预测。

注意：

```text
如果没有冻结 station_pga_prior_head，它会在 full graph training 中继续被更新。
```

### Distance weighted baseline

当前公式概念上是：

```text
baseline(target) =
    sum_i [station_prior_i * (1 / distance_i^p)]
    / sum_i [(1 / distance_i^p)]
```

其中：

```text
p = graph_pga_distance_baseline_power
```

当前默认：

```json
"graph_pga_distance_baseline_power": 1.0
```

最终输出：

```text
final_pga = distance_weighted_baseline + graph_residual
```

### 设计取舍

当前 prior-residual 方案不是纯 graph from scratch，而是有意把 single-station PGA 信息作为强先验注入。

原因：

- 旧 graph-only readout 几乎常数预测。
- single-station PGA 模型已经明显能学到有效信号。
- graph 当前主要应该学习空间传播和 target 修正，而不是重新从零学习 PGA 幅值。

风险：

- 如果 baseline 过强，graph residual 可能学不到东西。
- 如果 prior head 不冻结，baseline-only 消融不等价于固定 single-station baseline。
- 如果验证集太小，指标波动会很大。

## 5. 未完成事项 TODO

优先级从高到低。

### P0: 跑第一批消融实验

建议先跑：

```text
baseline_only
prior_residual
residual_only
no_prior
power05
power20
```

重点比较：

```text
val MAE
val RMSE
val Corr
val R2
val slope
pred std
target std
baseline std
residual std
```

最关键的问题：

```text
当前提升到底来自 distance baseline，还是 graph residual 真有贡献？
```

### P0: 收集 residual 诊断

重点看训练输出里是否出现：

```text
diag_pga_residual_mean.csv
diag_pga_residual_std.csv
diag_pga_scaled_residual_mean.csv
diag_pga_scaled_residual_std.csv
```

如果没有生成，检查训练日志诊断收集逻辑是否自动保存 `_last_diag` 里的新 key。

### P1: 做 frozen prior baseline

当前 baseline-only 配置只是把 residual scale 设为 0。

但如果 `station_pga_prior_head` 仍然训练，那么它不是严格的 frozen single-station baseline。

建议后续新增配置和代码参数：

```text
station_pga_prior_trainable: false
```

在 `train_light.py` 加载 single-station 权重后、optimizer 创建前：

```python
for p in raw_full.station_pga_prior_head.parameters():
    p.requires_grad = False
```

用途：

```text
判断固定 single-station prior + 距离加权本身有多强。
```

### P1: 增强 residual head

如果 residual std 很小，可以测试：

```json
"output_mlp_dims": [128, 64]
```

目的：

```text
增强 residual 对 baseline 压缩问题的修正能力。
```

### P2: 学习率实验

如果 residual 明显太弱，可小幅测试：

```json
"lr_team": 0.002
```

不要在 residual 诊断前盲目调大学习率。

### P2: 扩大验证规模

当前 overfit / 小样本验证事件数很少，历史结果里验证集大约只有 4 个 event，指标不稳定。

后续需要更大的 event split 才能判断泛化。

## 6. 已知问题 / 风险

### full graph 仍弱于 single-station

已有结果：

```text
first-inputs prior-residual full graph:
val MAE 0.2784
val Corr 0.6118
val R2 0.3586

single-station:
val MAE 0.1977
val Corr 0.8721
```

说明：

```text
graph 结构已经比旧 graph-only 明显好，但还没有超过 single-station。
```

### 预测方差仍偏小

历史结果：

```text
first-inputs prior-residual pred std: 0.1853
target std: 0.6463
```

说明模型仍偏保守，可能低估高 PGA / 高动态范围样本。

### baseline-only 消融解释要谨慎

如果 prior head 未冻结：

```text
baseline-only = 可训练 single-station prior head + distance weighted output
```

不是严格的 frozen single-station baseline。

### PGA-only 配置下 mag/loc 指标无意义

当前很多 graph PGA 配置：

```json
"res_comps": ["pga"]
```

因此 full model 的 mag/loc 输出没有训练意义。eval 里 mag/loc 近似常数不是当前 PGA 任务 bug。

### 超算路径敏感

正确数据路径：

```text
/public/home/test_bigmodel/seismogram/zb/team_pytorch/japan_2024.hdf5
```

正确 graph 工作目录：

```text
/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_graph
```

不要再误用：

```text
/public/home/test_bigmodel/seismogram/zb/team_pytorch_graph
```

### 当前本地工具状态

本次会话中，普通沙箱命令一度报错：

```text
error: unexpected argument '--sandbox-policy' found
```

读取文件和 git 检查使用了 escalated command。下一会话如果遇到同样问题，不一定是项目代码问题。

## 7. 下一步推荐行动

下一次会话第一步应该做：

```bash
cd /public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_graph
git fetch origin
git checkout zhangb/graph-pga-experiment
git pull origin zhangb/graph-pga-experiment
```

然后优先跑 baseline-only 和当前 prior-residual 对照：

```bash
JOB_NAME=graph-pga-baseline-only \
RUN_EVAL=1 \
bash train_light_slurm.sh pga_configs/graph_japan_overfit_pga_first_inputs_baseline_only_chaosuan.json
```

再跑：

```bash
JOB_NAME=graph-pga-residual-only \
RUN_EVAL=1 \
bash train_light_slurm.sh pga_configs/graph_japan_overfit_pga_first_inputs_residual_only_chaosuan.json
```

以及当前主线：

```bash
JOB_NAME=graph-pga-first-prior-residual \
RUN_EVAL=1 \
bash train_light_slurm.sh pga_configs/graph_japan_overfit_pga_first_inputs_prior_residual_chaosuan.json
```

第一批结果出来后，不要只看 MAE，要同时看：

```text
eval_results.txt
diag_pga_target_std.csv
diag_pga_mu_best_valid_std.csv
diag_pga_distance_baseline_std.csv
diag_pga_residual_std.csv
diag_pga_scaled_residual_std.csv
```

判断标准：

```text
如果 baseline-only 已经接近 prior-residual，说明主要收益来自 single-station prior + distance baseline。
如果 residual-only 明显差，说明直接 baseline addition 是关键。
如果 prior-residual 明显优于 baseline-only，说明 graph residual 有真实贡献。
```

## 8. 不要做什么

不要重复以下方向：

1. 不要继续只跑旧 graph PGA-only 配置并期待它自然变好。历史结果显示它接近常数预测。
2. 不要把 `target-to-station cross attention` 当作本分支主线；该方向已由其他人处理。
3. 不要默认 `station_pga_prior_head` 等同完整 single-station model；它只是 single-station PGA head。
4. 不要把 baseline-only 的结果解释为 frozen single-station baseline，除非明确冻结 prior head。
5. 不要把未跟踪的 `GRAPH_PGA_PROJECT_SUMMARY.md`、`jieshu_prompt.txt`、`jixu_prompt.txt` 默认加入提交。
6. 不要误用数据路径。
7. 不要只看 validation MAE；当前验证集小，必须结合 Corr、R2、slope、pred std、residual std 看。
8. 不要把 PGA-only 配置下的 mag/loc 常数输出当作当前问题。
9. 不要在没有 residual 诊断的情况下盲目扩大模型或学习率。

## 9. 关键上下文速记

### Repo

本地路径：

```text
/home/zhangb/work/people/zhangbei/team_claude/team_pytorch2/team_pytorch
```

超算路径：

```text
/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_graph
```

远端分支：

```text
origin/zhangb/graph-pga-experiment
```

### 关键文件

```text
gemini_models.py
train_light.py
eval_checkpoint.py
train_light_slurm.sh
pga_configs/
GRAPH_PGA_PROJECT_SUMMARY.md
```

### 关键类 / 函数

```text
gemini_models.py::GraphPGAReadout
gemini_models.py::FullModel
gemini_models.py::build_transformer_model
train_light.py::load_station_pretrain_weights
```

### 推荐主线配置

```text
pga_configs/graph_japan_overfit_pga_first_inputs_prior_residual_chaosuan.json
```

### 新增消融配置

```text
pga_configs/graph_japan_overfit_pga_first_inputs_baseline_only_chaosuan.json
pga_configs/graph_japan_overfit_pga_first_inputs_residual_only_chaosuan.json
pga_configs/graph_japan_overfit_pga_first_inputs_no_prior_chaosuan.json
pga_configs/graph_japan_overfit_pga_first_inputs_power05_chaosuan.json
pga_configs/graph_japan_overfit_pga_first_inputs_power20_chaosuan.json
```

### 训练命令模板

```bash
JOB_NAME=<job-name> \
RUN_EVAL=1 \
bash train_light_slurm.sh <config.json>
```

### 本地轻量检查

```bash
python -m py_compile gemini_models.py
python -m json.tool pga_configs/<config>.json
```

最近一次提交前已通过：

```text
python -m py_compile gemini_models.py
python -m json.tool 5 个新增消融配置
```

### Slurm 脚本默认设置

`train_light_slurm.sh` 默认：

```text
WORKDIR=/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_graph
SLURM_PARTITION=diting
SLURM_GPUS_PER_NODE=4
CONDA_ENV=lsm_env
RESET_WEIGHT_PATH=1
RUN_EVAL=1
```

默认 DiTing config：

```text
/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_graph/diting/config/diting_1200m_backbone_attnpool.yml
```

默认 DiTing pretrained：

```text
/public/home/test_bigmodel/seismogram/mx/results/scaling_diting_1b/scaling_diting_1200M/checkpoint_pt_epoch_70/mp_rank_00_model_states.pt
```

### Eval checkpoint 逻辑

`RUN_EVAL=1` 时 full model 默认按顺序找：

```text
full_model_best.pth
full_model_last.pth
```

single-station 默认按顺序找：

```text
single_station_best.pth
single_station_last.pth
```

### 已知历史结果

旧 graph PGA-only first-inputs：

```text
val MAE 0.3452
val Corr -0.0588
val R2 -0.0413
slope -0.0005
```

first-inputs prior-residual：

```text
val MAE 0.2784
val RMSE 0.3431
val Corr 0.6118
val R2 0.3586
slope 0.2976
```

random-inputs prior-residual：

```text
val MAE 0.3137
val RMSE 0.4138
val Corr 0.5602
val R2 0.2089
slope 0.4090
```

single-station first-inputs：

```text
val MAE 0.1977
val Corr 0.8721
```

single-station random-inputs：

```text
val MAE 0.2515
val Corr 0.7873
```

### 当前一句话结论

当前 graph PGA 已经从“旧 graph readout 近似常数预测”推进到“single-station PGA prior + distance baseline + graph residual 的可用原型”。下一阶段核心不是继续证明能跑通，而是用消融实验拆清楚 baseline 和 residual 各自贡献，并决定是否需要冻结 prior head、增强 residual head 或调整 distance power。
