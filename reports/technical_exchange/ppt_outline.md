# DiTing + TEAM/Graph 多台站 PGA 技术交流 PPT 大纲 v4

目标：面向外部/组内专家汇报当前 DiTing 表征接入多台站 event / PGA 预测的实验进展，说明当前任务、实验结果、主要问题和下一步计划。交流时间约 30 分钟。

对应文件：

- HTML 幻灯片：`diting_team_graph_pga_technical_exchange_v4.html`
- PPTX：`diting_team_graph_pga_technical_exchange_v4.pptx`
- 全部实验汇总：`tables/all_experiments_summary.csv`
- 新增支撑表：`tables/model_scale_params.csv`、`tables/loss_design.csv`、`tables/single_station_metrics.csv`

## 主线判断

当前核心问题不是 DiTing station feature 完全无效，而是多台站 full model 如何稳定地把 station feature、event 信息、target station 位置和空间传播先验组合起来。

当前实验支持几个判断：

1. 当前任务已经从 DiTing 常见的“单台站输入 -> 单台站/局部任务输出”变成“多台站输入 -> event-level 输出 + target-level PGA 输出”。
2. single-station pretrain 在 mag / epidist / PGA 上都有可观相关性，说明 waveform encoder + station adapter 的表征本身有信号。
3. 原始 full self-attention / query_transformer readout 容易让 PGA、mag、loc 输出坍塌成常数；loss 曲线下降并不能单独排除均值解。
4. cross-attention 明显改善 PGA readout，尤其在 overfit_n=32/128 和 fixed/input target 设置中能看到 train/val 信号。
5. graph message passing 本身不够；graph prior-residual 通过 single-station PGA prior + distance baseline + residual 才明显解除常数化。
6. P-pick 是重要数据质量风险：目前由走时曲线粗定位 + STA/LTA refine 得到，需要专家抽样检查准确性。
7. 当前仍是 overfit 诊断实验，train 表现、val 表现、loss 曲线、输出方差都要一起看。

## 正文页

### 1. 标题页

- 标题：DiTing 表征接入多台站 PGA / Event 预测
- 副标题：Transformer readout 与 graph prior-residual 的实验诊断
- 目的：说明做了什么、结果如何、问题在哪里，并请专家对数据、结构和先验设计提意见。

### 2. 这次希望专家帮忙看的点

- P-pick 是否足够准确，是否会污染 station feature 和 PGA label 对齐。
- event 信息如何进入 PGA：event query、mag/loc context、GMPE prior，还是显式震源参数。
- 任意 target PGA 更适合 cross-attention readout，还是 graph message passing。
- 小样本 overfit 诊断后，下一步验证集和消融实验如何设计。

### 3. 当前任务：从单台站 DiTing 到多台站 event/PGA 预测

- 对比 DiTing 常见任务：单个地震台三分量波形输入，输出震相拾取、单台站震级 proxy 或 station-level 任务。
- 当前任务：多个台站波形 + 台站坐标 + station_valid，经过共享 DiTing encoder 得到 per-station embedding。
- 输出分两类：event-level 震级/位置/震中距，target-station-level PGA。
- 核心区别：多个台站共同约束一个事件和一个空间 PGA 场。

### 4. 任务设置

- 输入：`N` 个 station 的三分量波形、station 坐标、`station_valid` mask。
- 查询：event query 用于震级/位置；PGA target query 用于给定 target station 的 PGA。
- 输出：event-level `magnitude / location / distance`，target-level `PGA`。
- 说明：当前 Japan overfit 小样本用于结构诊断，不作为最终泛化结论。

### 5. 数据覆盖与抽样样本

- 地图：Japan overfit event 与 station 分布。
- 图表：震级分布、overfit split。
- 样例事件：输入台站、target 台站、PGA 与震中距关系。
- 说明：数据图只用于帮助理解 overfit 任务边界；验证集很小，需要多 seed / 多 split。

### 6. P-pick 生成流程与元数据诊断

- 当前 P-pick 流程：先用走时曲线给粗定位，再用 STA/LTA 在搜索窗口内 refine。
- 左图：最终 pick 来源。
- 中图：refined pick 与走时粗定位 pick 的时间差。
- 右图：P pick 处 STA/LTA ratio 随震中距和 PGA 的分布。
- 专家问题：STA/LTA refine 是否合理，哪些样本应剔除或降权？

### 7. P-pick 波形抽样

- 从 `team_pytorch/japan_overfit.hdf5` 中抽取 P-pick 落在当前 100 s 窗口内的样本。
- 每条记录按三分量分别归一化，方便看小振幅波形的起跳。
- 红虚线为当前存储 P-pick。
- caveat：很多 HDF5 中的 `p_picks` 是 aligned/global sample，超出当前 100 s 训练窗口；这些样本需要回到原始 aligned 记录继续抽查。

### 8. 实现级模型总览：和后面表格的对应关系

- 所有路线共用 station feature 生成：waveform normalize / station_valid mask / DiTing encoder / station adapter / amplitude scale gate / coordinate fusion。
- 下半部分对应后面实验表里的 readout route：original `query_transformer`、PGA/event cross-attention、graph prior-residual。
- 后续结果页按 readout route 分组，再比较 input/target 选择、overfit_n、是否有 prior/baseline。

### 9. 模型规模与可训练参数

- DiTing encoder：约 1.234B 参数，当前冻结。
- TEAM cross-attention full model：约 1.262B 总参数，约 28.0M 可训练参数。
- station adapter：约 3.59M 可训练参数。
- graph prior-residual 额外参数：约 10.0M，full model 约 1.272B 总参数、约 38M 可训练参数。
- caveat：TEAM cross-attention 为本地精确计数；graph prior-residual 为按实现结构的近似估计。

### 10. Loss 函数设计

- single-station pretrain：Huber(delta=1)，`L = 0.3 L_mag + 0.3 L_epidist + 0.4 L_pga`。
- PGA-only full model：Huber(delta=1)，只在有效 target PGA 上计算。
- event+PGA full model：Huber(delta=1)，`L = 0.2 L_mag + 0.2 L_loc + 1.0 L_pga`。
- graph prior-residual：PGA Huber loss，预测值为 distance baseline + learned residual。
- 说明：这些实验使用 point prediction；loss 下降不等于 readout 不坍塌。

### 11. Single-station model 效果：证明 waveform 表征有效

- 展示 single-station val mag / epidist / PGA 的 MAE 和 Corr。
- 代表结果：
  - Cross overfit128 single val PGA：MAE 0.1654，Corr 0.9313。
  - Graph prior first-inputs single val PGA：MAE 0.1977，Corr 0.8721。
  - Graph prior random-inputs single val PGA：MAE 0.2515，Corr 0.7873。
- 解释：station waveform 表征有信号；full model 坍塌更像 readout/空间传播问题。

### 12. 全部实验总览

- TEAM 路线当前有 14 个实验，来自 `chaosuan_res`。
- Graph 路线当前有 7 个实验，来自 `team_pytorch2/chaosuan_res`。
- 图中圆点为 train PGA correlation，方块为 val PGA correlation。
- 灰色/空缺表示 eval 失败或没有 PGA metrics。

### 13. 原始 TEAM / query_transformer readout

- 结构：`[event token, station tokens, PGA target tokens]` 共同进入 full Transformer。
- PGA query = target coordinate embedding + learned PGA query token。
- `padding_mask=[event valid, station_valid, pga_target_valid]`。
- `att_mask` 让 PGA token 作为 query-only：PGA key 被 mask。
- 风险：event/PGA readout 路由需要训练自己发现，容易走向均值解/常数解。

### 14. 原始 readout ablation：各模型设置

- `query_transformer`：原始 full self-attention readout。
- `mask_batch1`：batch=1 mask sanity，检查跨 batch 泄露或 mask 维度问题。
- `query_no_transformer`：target coordinate + learned query token，不读取 station，是 target-query-only 负对照。
- `direct_station`：masked mean station_feature_emb -> PGA head，检查 station feature/head 是否有信号。
- `target_cross_attention`：PGA target query cross-attends station_feature_emb，早期 eval 失败/无 PGA metrics。

### 15. 原始 full self-attention readout 的坍塌

- 展示 TEAM 全部相关实验的 train/val PGA corr。
- 展示 ablation 表：Train Corr、Train slope、Val Corr、Val slope。
- 重点观察：`query_transformer` slope 约 0，`mask_batch1` 仍常数，说明跨 batch 泄露不是主因。

### 16. Attention / mask 诊断结论

- PGA query 后期可以 attend 到 station，但输出方差仍可能被压扁。
- 已修 `0.11111` 诊断 bug：区分 key-type mask 与 padding 后 effective key mask。
- mag/loc 也会常数化，说明问题不只在 PGA label，而在 event/PGA readout 路由。

### 17. Cross-attention readout 设计用意

- 设计目标：让 target/event query 单向读取 station tokens，避免 full self-attention 中 event/PGA token 互相干扰。
- PGA query：target coordinate embedding + learned PGA token。
- K/V：`station_feature_emb`。
- mask：`key_padding_mask = ~station_valid`。
- 输出不加 query residual，使输出值更直接来自 station tokens。

### 18. Cross-attention 各实验设置

- `pga15_cross_overfit32`：overfit_n=32，15 个 PGA targets，random inputs/targets。
- `pga15_cross_overfit128`：overfit_n=128，15 个 PGA targets，random inputs/targets。
- `fixed_inputs_targets`：固定输入台站和固定 target，最容易记忆，用作 route sanity。
- `input_targets`：input stations 同时作为 targets，same-station PGA 读出 sanity。
- `fixed_inputs_random_targets`：固定 inputs、随机 targets，测试固定输入下的空间插值。
- `event+pga_cross_first_inputs`：event 和 PGA 都用 cross-attention，first input stations。
- `mag/loc_cross_overfit32`：强调 event task，检查 event readout 坍塌。

### 19. Cross-attention 实验结果：train 和 val 都要看

- 图：代表 cross-attention 实验的 train/val PGA corr 与 MAE。
- 表：Experiment、overfit_n、Setting、Train Corr、Val Corr、Train MAE、Val MAE。
- 重点：overfit_n=32/128 都能看到 PGA readout 信号；fixed/input target 设置 train 表现强；fixed inputs random targets 的 val 明显弱。

### 20. Graph prior-residual：实现细节与为什么有效

- `station_pga_prior_head` 从 single-station pretrain 加载，给每个输入台站一个 PGA prior。
- distance baseline = 输入台站 prior 的距离加权平均。
- GraphPGAReadout 用 target-station edge features 学 residual。
- final PGA = distance baseline + learned residual。
- 关键判断：旧 graph 仍坍塌，说明“有 graph”本身不够，关键是 prior + baseline + residual。

### 21. Graph 路线各实验设置

- `graph first-inputs`：旧 graph readout，无 station prior / distance baseline。
- `graph event+pga first-inputs`：graph PGA 加 event task。
- `prior-residual first-inputs`：single-station PGA prior + distance baseline + residual。
- `prior-residual random-inputs`：同 prior-residual，但随机输入台站。
- `exp1 same-station`：single input + same-station target。
- `exp2 multi-target`：single input + multiple targets。
- `exp3 holdout`：SNR-filtered holdout targets。

### 22. Graph 路线实验结果：train 和 val 都要看

- 图：graph 相关实验 train/val PGA corr 与 MAE。
- 表：Experiment、Train Corr、Val Corr、Train MAE、Val MAE。
- 重点：old graph first-inputs 仍接近常数；prior-residual first-inputs 和 random-inputs 明显改善。

### 23. Loss 曲线与收敛情况

- 展示代表 full-model 实验 train/val Huber loss 曲线。
- 结论：多数实验 loss 基本随 epoch 下降，优化过程没有明显发散。
- caveat：`query_transformer` 的 loss 也会下降，但输出仍可坍塌到均值解。
- 因此 loss 曲线必须和 corr、slope、pred std 一起解释。

### 24. Single-station loss 曲线

- 展示 single-station pretrain train/val weighted Huber loss。
- 结论：single-station 预训练稳定收敛。
- 解释：支持先训练单台站 waveform 表征和 PGA prior，再作为 graph prior-residual 的输入。
- 后续建议：保存并汇报分任务 loss，避免 mag/epidist/PGA 被加权总 loss 掩盖。

### 25. 两条路线对比

- `query_transformer`：理论统一，但 PGA/event 都容易常数化。
- `direct_station`：证明 station feature/head 可用，但不能处理任意 target 空间传播。
- `cross-attention`：target 显式读 station，适合诊断 readout。
- `graph prior-residual`：引入 single-station prior 和距离传播，当前 graph 路线最好。

### 26. 当前主要问题

- P-pick 准确性还没有人工抽查闭环，可能影响波形窗口、station SNR 和 label 对齐。
- event mag/loc 联合训练仍会常数化，需要独立定位 event readout。
- PGA pred std 仍小于 target std，模型偏保守。
- 验证集太小，多 seed / 多 split 后结论才稳。
- input/target 抽样策略影响明显，需要把 overfit_n、first/random/fixed target 都纳入结果解释。

### 27. 下一步实验建议

- 数据：做 P-pick 人工质检样本集，统计 pick 偏差和 STA/LTA 失败类型，对低 SNR 或 pick 不可靠样本剔除/降权。
- Cross-attention：event cross-attention、PGA event context gate、distance bias、relative coordinate MLP。
- Graph：baseline-only、residual-only、prior-only、prior + baseline + residual、kNN / distance power `p` 消融。
- 评估：扩大 split、多 seed、记录 baseline-target corr、residual-target corr、attention top-k distance。

### 28. 希望专家重点建议

- P-pick 的 STA/LTA refine 是否合理，哪些样本应该剔除或降权？
- PGA target 是否必须显式使用 event 信息？event 信息应如何进入模型？
- 距离衰减先验更适合作为 baseline、attention bias，还是 residual feature？
- 是否需要加入台站场地项、方位角、路径效应、区域衰减参数？
- 当前 overfit 诊断做到哪一步可以转向更大训练集？

## 备份材料

- `tables/all_experiments_summary.csv`：TEAM 14 个实验 + Graph 7 个实验完整 train/val 汇总。
- `tables/model_scale_params.csv`：模型参数量和可训练参数量。
- `tables/loss_design.csv`：single-station / full-model / graph prior-residual 的 loss 设计。
- `tables/single_station_metrics.csv`：single-station mag / epidist / PGA 指标。
- `tables/transformer_ablation.csv`：原始 readout ablation 结果。
- `tables/cross_attention.csv`：cross-attention 代表实验结果。
- `tables/graph_results.csv`：graph 路线代表实验结果。
- `tables/variance_summary.csv`：PGA prediction std 与 target std 对比。
- `assets/ppick_audit_waveforms.png`：P-pick 抽样波形图。
- `assets/ppick_metadata_stats.png`：P-pick 元数据统计图。
- `assets/task_setup_multistation.png`：当前任务示意图。
- `assets/architecture_implementation_detail.png`：实现级模型总览。
- `assets/architecture_graph_detail.png`：graph prior-residual 细节图。
- `assets/single_station_val_metrics.png`：single-station val 指标图。
- `assets/representative_full_loss_curves.png`：full-model loss 曲线。
- `assets/single_station_loss_curves.png`：single-station pretrain loss 曲线。
