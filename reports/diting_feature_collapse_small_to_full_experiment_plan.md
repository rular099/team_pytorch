# DiTing 台站特征坍塌：从 512 事件到全量数据的实验方案

版本：2026-07-29

执行分支：`zhangb/native-scale-adapter-scaling`

第一阶段基线：KNET-only rt46

## 1. 目标与待判定假设

本方案不把“更多数据一定能解决坍塌”或“结构一定有问题”作为先验结论，而是用嵌套数据规模和结构对照区分三种情况：

1. **结构性坍塌**：即使训练事件增加，legacy summary/GAP/adapter 后的台站差异仍被持续压平。
2. **数据驱动坍塌**：小数据下模型优先学习 event-common 捷径；数据量和事件多样性增加后，台站特异信息自然增强。
3. **结构与数据共同作用**：新 adapter 在小数据上改善可辨识性，但需要多年份数据才能转化为 validation 泛化。

核心判据不是单一 validation PGA MAE，而是同时观察：

- train 是否能过拟合；
- train/validation 的 waveform-station mismatch 退化；
- 有效波形区间内的 token 与 station embedding 差异；
- 这些量随事件数增加的学习曲线。

## 2. 已冻结的设计原则

### 2.1 不在四个尺度 pooling 前再做全局交互

ViTAdapter 已经通过共享的 `x` 与 `c2/c3/c4` 多次交互。额外将四个尺度强制重采样到同一长度再交互会：

- 丢失 `f2` 的高时间分辨率；
- 引入插值或混叠；
- 增加显存和不可控变量；
- 重复 ViTAdapter 已完成的跨尺度交换。

因此新 adapter 保留 `f2/f3/f4/x` 的原生时间长度，各自提取摘要，最后融合。

### 2.2 padding mask 与“有效地震信息”分开

硬 mask 只解决确定无效的人工补零，不人为规定 P 波以后哪一段最重要。

mask 来源优先级：

1. HDF5 中的 `waveform_valid_mask` 或 `valid_sample_mask`；
2. `valid_start_sample + valid_n_samples`；
3. 原始存储波形首尾连续全零段。

第 3 种只裁掉首尾连续 padding。有效区间内部即使某个采样点恰好为零，也仍保留为有效。mask 在数据生成时产生，并同时用于：

- 波形均值/方差归一化；
- legacy attention pooling；
- NLTA 的卷积、attention 和统计摘要。

不使用模型内部的 `waveform == 0` 临时判断。

### 2.3 adapter 内不接 DPK 概率，也不使用可关闭 gate

DPK 概率来自同一个 DiTing encoder，adapter 内再次把它作为人工先验，可能限制模型学习尾波、震级和震中距所需的信息。

第一阶段中：

- adapter 只看 DiTing 多尺度特征和确定的 padding mask；
- 不使用 P/S/detection 概率；
- 不使用 `z_keep + gate × z_new`；
- 不重复已失败的 event + residual 方案。

为保证与 rt46 的因果可比性，rt46 已有的 **adapter 外部** PGA temporal residual 和 cached DPK weighting 暂时保持不变。只有选出 adapter 后，才单独做外部 DPK 去除实验。

### 2.4 pooling 必须兼顾多任务

新结构不能只针对 PGA：

- x 主分支的 TCN 保留长时间上下文；
- attention 学习选择信息；
- masked mean/std 保留全局幅值与分布信息；
- 多尺度侧分支保留局部和不同分辨率信息；
- 输出仍为通用 1000 维 station embedding，供 PGA、震级和位置任务共同使用。

## 3. 候选结构

### 3.1 Legacy adapter

当前结构：

- x 的 4-query attention pooling → 7168 → 1000 基础分支；
- f2/f3/f4/x 各自 1792 → 256、卷积、4-query pooling；
- 四支拼接后映射为 1000 维 delta；
- `LayerNorm(base + delta_scale × delta)`；
- 参数量：13,913,017。

风险是大维度直接 summary、基础分支和可学习 delta scale 共同鼓励 event-common 捷径。

### 3.2 Native-Scale Late-Fusion TCN Adapter（NLTA）

x 主分支：

1. 1792 → `D_x` 的 1×1 投影；
2. dilation 为 1/2/4/8/16/32/64 的 7 层 causal depthwise-separable TCN；
3. 4 个 latent query 对 x token 做 cross-attention；
4. 并行计算 masked mean 和 masked std。

f2/f3/f4 侧分支：

1. 各自 1792 → `D_s`；
2. 一层局部 depthwise-separable block；
3. 2-query attention pooling；
4. 并行 masked mean/std。

最后：

- 拼接 `6D_x + 12D_s` 维摘要；
- 直接 Linear → 1000；
- LayerNorm；
- 无 base/delta 双路、无 gate、无 pooling 前跨尺度重采样。

复杂度方面，x attention 为 latent-to-token：

```text
attention map: B × heads × Q × T
Q = 4
```

而不是：

```text
self-attention map: B × heads × T × T
```

因此即使有效区间覆盖整个输入，也不会恢复此前 full temporal attention 的显存风险。

候选宽度：

| 结构 | D_x | D_s | x heads | adapter 参数量 | legacy 占比 |
|---|---:|---:|---:|---:|---:|
| NLTA-S | 256 | 96 | 4 | 4,698,744 | 33.8% |
| NLTA-M | 384 | 128 | 6 | 7,512,888 | 54.0% |
| NLTA-L（后续按需） | 512 | 192 | 8 | 约 11.4M | 约 82% |

NLTA-L 不进入第一阶段。只有 NLTA-M 在更大数据上显示明确容量不足时才启用。

## 4. 数据层级与冻结规则

### 4.1 两条 512-event 轨道

当前 512 事件来自 2024 年约一半数据，继续保留为：

```text
D0-2024：历史结果连续性、代码验证、过拟合能力测试
```

但它不能代表约 20 年数据分布。进入缩放实验前，另建：

```text
D0-MY：从最终训练年份池中分层抽取的 512 个多年份事件
```

D0-MY 按以下变量分层或覆盖采样：

- 年份；
- 震级；
- 强/弱 PGA；
- 台站数；
- K-NET / KiK-net 与传感器类型；
- 震中距；
- 可用波形秒数；
- P 后有效波形秒数；
- 实时观测时刻。

### 4.2 嵌套规模

所有缩放子集必须是事件级嵌套：

```text
D0-MY (512) ⊂ D1 (约 2,000) ⊂ D2 (约 8,000) ⊂ D3 (全部训练事件)
```

这样学习曲线变化来自新增事件，而不是每个规模抽到完全不同的数据。

### 4.3 最终 split

2024 数据已经用于大量结构选择，不再声称它是完全未见的最终 test。

在生成 D0-MY 前必须先完成全量年份清点，并冻结：

1. `train_events.txt`
2. `validation_events.txt`
3. `test_events.txt`
4. 每个 D0/D1/D2 子集的事件清单
5. split 生成脚本、随机种子和数据版本摘要

优先采用事件不重叠的 chronological split。最终 test 年份必须是尚未参与当前调参的年份；若现有数据中不存在真正未使用年份，则：

- 使用 event-disjoint IID test 作为当前内部评估；
- 明确标注其不是严格时间外推；
- 后续保留新增年份作为 prospective test。

所有同一地震事件的台站记录只能属于一个 split。

### 4.4 目标归一化

确定最终训练池后，用该训练池一次性计算 PGA/震级/位置归一化统计，并在 D0-MY/D1/D2/D3 中固定。不能让不同数据规模各自计算均值/方差，否则部分学习曲线差异来自目标坐标变化。

D0-2024 第一阶段为复现现有 rt46，仍沿用当前配置的 auto 统计。

## 5. 第一阶段实验矩阵：D0-2024

除表中变量外，四组保持完全一致：

- KNET-only；
- 同一个 `stage2_512_event_ids.txt`；
- seed=42；
- 相同 train/validation split；
- 相同实时采样时刻；
- 相同 target sampling、损失和优化器；
- 相同冻结 DiTing encoder；
- 相同外部 cached DPK temporal residual；
- 相同 80 epoch；
- 相同 normal 与 station-roll eval。

| ID | 配置 | adapter | 显式 padding mask | adapter 参数 | 目的 |
|---|---|---|---|---:|---|
| E0 | rt46（已有） | legacy | 否 | 13.913M | 历史对照 |
| E1 | rt52 | legacy | 是 | 13.913M | 单独测 padding 处理 |
| E2 | rt53 | NLTA-S | 是 | 4.699M | 测轻量新结构 |
| E3 | rt54 | NLTA-M | 是 | 7.513M | 测容量敏感性 |

配置文件：

```text
pga_configs/transformer_japan_overfit_pga15_stage2_512_rt46_knet_cached_dpk_event_temporal_residual_scale4_chaosuan.json
pga_configs/transformer_japan_overfit_pga15_stage2_512_rt52_knet_legacy_paddingmask_cached_dpk_event_temporal_residual_scale4_chaosuan.json
pga_configs/transformer_japan_overfit_pga15_stage2_512_rt53_knet_nlta_s_paddingmask_cached_dpk_event_temporal_residual_scale4_chaosuan.json
pga_configs/transformer_japan_overfit_pga15_stage2_512_rt54_knet_nlta_m_paddingmask_cached_dpk_event_temporal_residual_scale4_chaosuan.json
```

### 5.1 第一阶段运行顺序

1. 在超算同步实验分支。
2. 运行配置/输入自检和提交 dry-run。
3. 只提交 rt52–rt54；rt46 使用已有结果。
4. 每组完成后跑 normal eval；一次任务同时计算 train 和 validation。
5. 再跑 station-roll；同样同时计算 train 和 validation。
6. 汇总 mask 秒数、时间分桶、台站数分桶和表征诊断。

自检与 dry-run：

```bash
PYTHONPATH=../ditingbench python tools/validate_native_scale_adapter_stage1.py
DRY_RUN=1 bash tools/run_rt52_rt54_native_scale_adapter_stage1_slurm.sh
DRY_RUN=1 bash tools/eval_rt52_rt54_native_scale_adapter_stage1_slurm.sh
```

训练：

```bash
bash tools/run_rt52_rt54_native_scale_adapter_stage1_slurm.sh
```

只跑部分实验：

```bash
RT_LIST="52 53" bash tools/run_rt52_rt54_native_scale_adapter_stage1_slurm.sh
```

若作业因 Slurm walltime 超时，从每组的 `full_model_last.pth` 严格续训；默认重新申请 48 小时：

```bash
bash tools/resume_rt52_rt54_native_scale_adapter_stage1_slurm.sh
```

也可只续训指定实验或覆盖 walltime：

```bash
RT_LIST="53 54" SLURM_TIME=72:00:00 \
bash tools/resume_rt52_rt54_native_scale_adapter_stage1_slurm.sh
```

该脚本不会回退到 best/init，也不会删除权重目录；若存在同名活动任务则默认跳过。超时发生在一个 epoch 中间时，只能从上一个完整 epoch 末保存的 checkpoint 继续。

eval-only：

```bash
bash tools/eval_rt52_rt54_native_scale_adapter_stage1_slurm.sh
```

### 5.2 第一阶段硬性健康检查

任何一项失败都先修代码，不进入性能比较：

- mask shape 必须为 `(batch, station, 10000)`；
- invalid station 的 mask 必须全 False；
- padding 区归一化后仍为 0；
- attention 在全 False mask 行不产生 NaN；
- `valid_token_count` 与 `waveform_valid_seconds` 同步变化；
- p-pick 后秒数满足 `0 ≤ post_p_seconds ≤ valid_seconds`；
- rt52–rt54 的 DPK cache key/coverage 与 rt46 一致；
- station-roll 时 waveform mask 与 waveform 一起置换；
- 最大显存和 step time 有记录。

### 5.3 第一阶段主判据

按以下顺序分析：

1. **train PGA/总 loss**：能否拟合 512 事件；
2. **train station-roll delta**：matched waveform 是否被使用；
3. **表征差异**：adapter 后 cosine、有效秩、台站间方差；
4. **validation station-roll delta**：station use 是否泛化；
5. **validation PGA/震级/位置指标**。

不因单次 validation PGA MAE 最低就直接胜出。候选必须在下列三轴上形成 Pareto 优势：

- 任务性能；
- matched-station waveform 依赖；
- station 表征非坍塌程度。

相对警戒线：

- 新模型 train PGA MAE 若比 E1 高超过 25% 且继续训练不下降，优先判为容量或优化不足；
- validation PGA MAE 若恶化超过 0.01，需要用 station-roll 和分桶结果证明存在明确补偿收益；
- adapter cosine 下降但 train/roll delta 不增加，不能视为有效改善；
- train 改善而 validation 不改善，进入多年份小数据验证，不立即否定结构。

第一阶段输出两个候选：

```text
C_legacy：E1
C_nlta：E2/E3 中较优者
```

## 6. 第二阶段：多年份 512 与约 2,000 事件

目的：排除“2024 单年份分布”造成的结论，并观察最早的数据量趋势。

矩阵：

| 数据 | 模型 | seed |
|---|---|---|
| D0-MY | C_legacy, C_nlta | 42, 2027 |
| D1≈2k | C_legacy, C_nlta | 42, 2027 |

这一阶段不增加模型宽度，不改变外部 DPK，不改任务头。只允许在两种候选上共同调整：

- adapter learning rate：`3e-4` 与 `1e-3` 的小范围检查；
- dropout：`0` 与 `0.05`，仅在明显过拟合时启用；
- warmup 和总 optimizer step。

选择超参数时先在 D0-MY 确定，再原样迁移到 D1，避免为每个规模单独过度调参。

判定：

- 如果 legacy 随数据量增加时 collapse 指标持续改善并追平 NLTA，说明数据驱动成分较强；
- 如果 legacy 曲线近似水平而 NLTA 保持明显优势，说明结构性限制较强；
- 如果二者都只改善 train、不改善 validation，优先检查 split、多年份分布、采样和目标泄漏；
- 如果 NLTA-S 在 D1 持续 train underfit，保留 NLTA-M，不直接上 L。

## 7. 第三阶段：约 8,000 事件与容量/DPK 解耦

### 7.1 宽度决策

在 D2 上运行：

```text
C_legacy
NLTA-S 或 NLTA-M 的阶段二胜者
```

仅当 NLTA-M 同时满足以下条件才增加 NLTA-L：

- train 指标明显劣于 legacy；
- train loss 仍随训练下降，非优化停滞；
- station-roll delta 表明模型确实使用 waveform；
- 显存预算允许。

### 7.2 adapter 外部 DPK 消融

固定最佳 adapter 后，再单独比较：

1. `temporal_token_weight_mode=cached_dpk_event`；
2. `temporal_token_weight_mode=none`，保留同一 temporal residual head。

该实验回答“DiTing 特征本身是否足够”，不能与第一阶段 adapter 结构变化混在一起。

如果无 DPK 版本达到相当或更好性能，则全量实验优先采用无 DPK 路线，减少缓存、置信度校准和跨数据集迁移依赖。

### 7.3 冻结/解冻消融

第一、二阶段保持当前冻结 DiTing encoder。到 D2 才测试：

- frozen encoder；
- 只解冻最后 interaction / 最后若干 backbone blocks；
- 必要时全量微调。

解冻实验使用更小 backbone LR，例如 adapter LR 的 0.05–0.1 倍，并记录显存与训练稳定性。

## 8. 第四阶段：全量训练

全量 D3 只保留：

1. 最强可复现 legacy baseline；
2. 前三阶段选出的最终 NLTA；
3. 如第三阶段证明必要，再保留 DPK/no-DPK 两个版本。

每个最终候选至少 2 个 seed；用于论文级结论时使用 3 个 seed。

最终结果必须同时报告：

- seed 均值和标准差；
- validation 与冻结 test；
- 年份分桶；
- 震级、PGA、震中距、台站数和传感器类型分桶；
- valid seconds 与 post-P seconds 分桶；
- normal 与 station-roll；
- 资源指标。

## 9. 训练预算与超参数随数据量的处理

### 9.1 不用 epoch 直接跨规模比较计算量

事件数扩大约 40 倍时，同样 epoch 会带来约 40 倍 optimizer update。比较时同时记录：

```text
optimizer_updates
events_seen
event_exposures = events_seen / unique_train_events
wallclock
peak_memory
```

同一数据规模内的结构对照使用相同 update 和采样序列。跨规模学习曲线至少提供：

- 相同 event exposure 的比较；
- 各自训练到 validation 收敛的比较。

D0-2024 的 80 epoch 是过拟合能力测试，不直接照搬为 D3 的 epoch。

### 9.2 learning rate

learning rate 主要随 global batch、优化器和宽度变化，不应因数据集变大自动线性增加。

- global batch 不变：先保持 adapter LR；
- global batch 改变：只做线性或平方根缩放的窄范围验证；
- 解冻 backbone：单独设置较小 LR；
- NLTA-S/M 第一阶段 LR 都与 E1 相同，避免把优化器也变成变量。

### 9.3 weight decay

AdamW 的累计收缩近似：

```text
theta_T ≈ exp(-learning_rate × weight_decay × optimizer_updates) × theta_0
```

因此数据增大导致 update 增多时，不能机械保持 epoch 和 weight decay 的组合。每阶段记录累计 `lr × wd`，在 D1/D2 上校准一次后冻结到 D3。

### 9.4 dropout 与模型宽度

- D0 第一阶段 dropout=0，目的是测试表达能力；
- D0-MY/D1 若 train/validation gap 明显，再测试 0.05；
- 不因“全量数据更多”自动增大宽度；
- 只有学习曲线显示容量不足时才从 S→M→L。

### 9.5 sampling

实时切片与同一事件内台站样本高度相关，有效样本量是事件数而不是切片数。

训练采用 event-first 采样，并控制：

- 实时时间 bin；
- 强/弱 PGA；
- 震级；
- 年份；
- 台站数；
- 传感器/网络；
- 震中距；
- valid/post-P seconds。

不能用大量同事件切片替代新增独立事件。

## 10. 统一评估指标

### 10.1 任务指标

PGA：

- MAE、RMSE、NLL；
- 强/弱 PGA；
- Brier score 与 calibration；
- 1/3/5/10/20/40/90 s；
- 台站数从少到多；
- 震中距；
- normal 与 waveform-station roll。

震级：

- train/validation MAE、bias；
- 震级分桶。

位置：

- epicenter error；
- depth error；
- 距离分桶。

### 10.2 有效波形分桶

token 数必须和物理时间同时报告。至少使用：

```text
valid waveform seconds:
0–1, 1–3, 3–10, 10–20, 20–40, 40+

post-P valid seconds:
0–1, 1–3, 3–10, 10–20, 20–40, 40+
```

每个分桶报告事件数、台站数和上述任务指标，避免只说“有效 token 数”。

### 10.3 feature-collapse 指标

在相同 valid mask 上计算：

- raw x token station cosine；
- TCN 后 x token station cosine；
- attention/summary 后 station embedding cosine；
- station embedding 有效秩；
- 台站间方差与 event-center residual norm；
- waveform-station roll 前后表征与预测变化；
- query 间 attention cosine；
- attention entropy/effective token count；
- valid token count 及其对应物理秒数。

判断数据驱动或结构驱动时，绘制这些指标相对于：

```text
log(unique_train_events)
```

的曲线。

## 11. 结果晋级和停止规则

每阶段结束建立一张决策表，不临时改变标准。

晋级条件：

- 无数据/数值/缓存错误；
- train 能学习；
- station-roll 证明 matched waveform 有贡献；
- collapse 指标改善能转化为任务或泛化收益；
- 资源可接受。

停止某候选：

- 两个连续数据规模都被另一候选 Pareto 支配；
- train 长期 underfit 且增加训练预算无改善；
- attention/query 实际退化为重复摘要且任务无收益；
- 峰值显存或 step time 不适合全量扩展。

不因一次 seed 的微小 MAE 差异停止候选。

## 12. 实验产物和可复现性

每个 run 保存：

- 完整 config；
- git commit 和 branch；
- 事件清单及 SHA256；
- 数据文件版本摘要；
- seed；
- 参数量；
- optimizer group、global batch、update 数；
- best/last checkpoint；
- train/validation normal eval；
- train/validation station-roll eval；
- 分桶结果；
- collapse diagnostics；
- peak memory、step time、wallclock；
- 一页 decision log。

命名中必须包含：

```text
dataset tier + adapter + mask mode + DPK mode + seed
```

## 13. 当前立即执行内容

本分支只实现第一阶段必要改动：

- 显式 waveform padding mask；
- mask-aware normalization；
- legacy pooling mask 对照；
- NLTA-S/M；
- rt52–rt54 configs；
- 第一阶段训练与 eval-only Slurm launcher；
- 参数量、mask 和配置自检。

D0-MY/D1/D2/D3 的事件清单、NLTA-L、外部 DPK 消融和 encoder 解冻配置，在上一阶段达到晋级条件后再生成，避免一次性铺开大量未经验证的配置。
