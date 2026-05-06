# DiTing + TEAM/Graph 多台站 PGA 技术交流 PPT 优化大纲

目标：面向组内/外部专家汇报 DiTing 表征接入多台站 event / PGA 预测的阶段性实验诊断。重点不是宣称最终模型已解决泛化问题，而是清楚说明：**单台站表征有效，但多台站 PGA/Event readout 容易坍塌；显式 cross-attention 与 graph prior-residual 能部分解除常数化，下一步关键在数据质检、空间传播先验和可复现实验设计。**

建议时长：30 分钟，其中正文 22 页左右，备份页若干。

对应当前材料：

- 最新 PPTX / HTML：`diting_team_graph_pga_technical_exchange_v4.pptx`、`diting_team_graph_pga_technical_exchange_v4.html`
- 实验总表：`tables/all_experiments_summary.csv`
- 关键结果表：`tables/transformer_ablation.csv`、`tables/cross_attention.csv`、`tables/graph_results.csv`、`tables/variance_summary.csv`
- 支撑图：`assets/*.png`

> 注意：`tables/all_experiments_summary.csv` 与 `tables/transformer_ablation.csv` 中 `direct_station` 指标不完全一致。正文建议采用 `transformer_ablation.csv` 作为 ablation 精修表；总览图/总表中需标注数据来源或更新一致，避免答辩时被问到数值冲突。

---

## 一、优化后的汇报主线

### 原大纲的问题

当前 v4 大纲信息很全，但有几个风险：

1. 页数偏多，28 页正文对 30 分钟交流较紧，容易变成“实验流水账”。
2. 模型路线、实验设置、结果页分得较细，听众需要持续记忆多个实验名，认知负担较高。
3. 核心结论出现较晚。建议在前 5 页就给出“目前确认了什么 / 还没确认什么”。
4. P-pick 数据风险很重要，但当前放在第 6–7 页，容易打断模型主线。建议前面简述，详细抽查放到风险与专家问题部分。
5. `query_transformer`、cross-attention、graph prior-residual 的逻辑关系可以更锋利：不是三条并列路线，而是一个诊断链条：
   - 表征是否有信号？有。
   - full self-attention 为什么失败？readout/路由坍塌。
   - 显式 query-to-station 能否改善？能。
   - 仅 graph 是否足够？不够。
   - 加入 single-station prior + distance baseline + residual 后是否改善？明显改善。

### 优化后的核心叙事

建议用下面这条线贯穿全 PPT：

> **我们把 DiTing 的单台站表征扩展到多台站 event/PGA 预测。实验首先证明 station waveform 表征本身有 PGA/震级/距离信号；但原始 TEAM 式 full self-attention readout 在小样本 overfit 中也会收敛到均值/常数解。进一步诊断表明，问题主要在多台站 readout 与空间传播建模，而不是 DiTing 表征完全失效。显式 cross-attention 能改善 target PGA 读取，graph prior-residual 通过 single-station PGA prior + 距离 baseline + residual 明显缓解常数化。下一步需要围绕 P-pick 质检、event 信息注入、距离/路径先验、多 seed/split 做系统验证。**

一句话结论：

> **有效信号在 station encoder 里；瓶颈在 event/target readout 与空间传播先验。**

---

## 二、推荐正文结构：22 页版本

### 1. 标题页

**标题**：DiTing 表征接入多台站 PGA / Event 预测  
**副标题**：从 readout 坍塌诊断到 graph prior-residual 传播先验  
**讲述目标**：汇报阶段性诊断结果，并请专家对 P-pick、空间先验和下一步实验设计给意见。

建议视觉：简洁标题 + `assets/task_setup_multistation.png` 的缩略图或模型示意。

---

### 2. 先给结论：当前确认了什么？

建议把核心判断前置，避免听众等到最后才知道主线。

- **DiTing station 表征有信号**：single-station PGA / epidist / mag 均有相关性。
- **原始 full self-attention readout 不稳**：loss 可下降，但 PGA / mag / loc 可能坍塌成近常数。
- **cross-attention 有效改善 target PGA readout**：overfit_n=32/128 和 fixed/input target sanity 中看到明显 train/val 信号。
- **graph message passing 本身不够**：旧 graph 仍接近常数。
- **prior-residual 是当前最有解释性的改进**：single-station PGA prior + distance baseline + residual 能显著提高 val corr，并提升预测方差。
- **还不能下最终泛化结论**：目前仍是 overfit 诊断，小验证集、多 seed/split 和 P-pick 质检仍需补齐。

建议图：一张“evidence ladder”示意图：Station signal → Readout collapse → Cross-attn repair → Graph prior-residual。

---

### 3. 希望专家重点帮忙判断的问题

把专家问题放前面，让后面所有实验都围绕这些问题展开。

1. P-pick / 波形窗口是否可靠？哪些样本应剔除或降权？
2. PGA target 是否必须显式使用 event 信息？event 信息应作为 query、context gate，还是 GMPE/震源先验？
3. 距离衰减和路径效应应作为 baseline、attention bias，还是 residual feature？
4. 当前 overfit 诊断做到什么程度，可以转向更大训练集？

---

### 4. 任务转换：从单台站 DiTing 到多台站 event/PGA

重点突出任务难度变化，而不是只列输入输出。

- DiTing 常见模式：单台站三分量波形 → pick / station-level 表征或任务。
- 当前模式：多台站波形 + 坐标 + mask → 共享事件表征 + 任意 target station PGA。
- 难点：
  - 多台站信息融合；
  - target station 与输入 station 不一定重合；
  - PGA 是空间场，需要传播/衰减先验；
  - event-level mag/loc 与 target-level PGA 需要共享但不互相污染。

建议图：`assets/task_setup_multistation.png`。

---

### 5. 数据与 overfit 诊断边界

这页不要展开太多 P-pick 细节，只说明实验边界。

- 当前为 Japan overfit 小样本结构诊断。
- 关注：train 是否可记忆、val 是否有初步信号、输出方差是否摆脱常数解。
- 指标同时看：MAE、Corr、R²、slope、pred std / target std。
- 解释原则：loss 下降不能单独证明模型学到了有效 PGA 空间变化。

建议图：`assets/data_overview.png` 或 `assets/sample_event_geometry.png`。

---

### 6. 实现级模型总览：共用 encoder + 不同 readout

这页用于统一后续术语，减少听众迷失。

共用部分：

- waveform normalize
- station_valid mask
- frozen DiTing encoder
- station adapter
- amplitude scale gate
- coordinate fusion

分支 readout：

1. 原始 TEAM / full self-attention query readout
2. target/event cross-attention readout
3. graph prior-residual readout

建议图：`assets/architecture_implementation_detail.png`。

---

### 7. 模型规模与训练参数

这页服务于“为什么能训练”和“冻结 DiTing 的意义”。

- DiTing encoder：约 1.234B 参数，当前冻结。
- TEAM cross-attention full model：约 1.262B 总参数，约 28.0M 可训练。
- station adapter：约 3.59M 可训练。
- graph prior-residual：额外约 10M，可训练参数约 38M。
- 解释：当前阶段主要诊断 readout / adapter / prior，而不是端到端微调整个 DiTing。

建议表：`tables/model_scale_params.csv`。

---

### 8. Loss 与评估指标：为什么只看 loss 不够

建议合并原大纲第 10、23、24 页的核心信息，避免重复。

- point prediction + Huber loss。
- single-station pretrain：`0.3 L_mag + 0.3 L_epidist + 0.4 L_pga`。
- full PGA：有效 target 上计算 PGA Huber。
- event+PGA：`0.2 L_mag + 0.2 L_loc + 1.0 L_pga`。
- 关键提醒：均值解也能降低 Huber loss，因此必须看 corr / slope / pred std。

建议视觉：左侧 loss 设计表，右侧 `assets/variance_compression.png` 或一个小示意。

---

### 9. 证据 1：single-station 表征确实有效

这是整场汇报的第一个关键证据。

代表结果：

- Cross overfit128 single val PGA：MAE 0.1654，Corr 0.9313。
- Graph prior first-inputs single val PGA：MAE 0.1977，Corr 0.8721。
- Graph prior random-inputs single val PGA：MAE 0.2515，Corr 0.7873。

结论：

- waveform encoder + station adapter 不是完全无效；
- full model 中 PGA 坍塌更可能来自 readout / target query / 空间传播建模，而非底层表征完全失败。

建议图：`assets/single_station_val_metrics.png`。

---

### 10. 证据 2：原始 full self-attention readout 容易坍塌

建议把机制图和结果放一页或两页，突出“为什么失败”。

结构：

- `[event token, station tokens, PGA target tokens]` 共同进入 full Transformer。
- PGA query = target coordinate embedding + learned PGA query token。
- PGA key 被 mask，PGA token 主要作为 query-only。

观测：

- `query_transformer` pred std ≈ `3.7e-6`，target std ≈ `0.609`。
- val slope ≈ 0，说明输出几乎不随真实 PGA 变化。
- `mask_batch1` 仍坍塌，说明不是简单的跨 batch 泄露。
- mag/loc 也出现常数化，问题不只在 PGA label。

建议图：`assets/architecture_team.png` + `assets/result_transformer_ablation.png` 或 `assets/team_all_train_val_corr.png`。

---

### 11. 原始 readout ablation：排除哪些解释？

建议用“诊断问题 → 结论”的方式，而不是只列实验名。

| 诊断问题 | 实验 | 结论 |
|---|---|---|
| 是否是 batch/mask 泄露？ | `mask_batch1` | 仍近常数，不是主因 |
| 只靠 target 坐标是否能解释？ | `query_no_transformer` | 有一定相关但不是有效多台站读取 |
| station embedding/head 是否完全无信号？ | `direct_station` | 可读出信号，说明 station feature 可用 |
| full transformer 是否能自动学会 query route？ | `query_transformer` | 不稳定，容易均值解 |

建议采用 `tables/transformer_ablation.csv` 的精修数值，并在总表中同步修正 `direct_station` 数值冲突。

---

### 12. Cross-attention 设计：把信息路由显式化

这页讲方法动机。

- 目标：让 target/event query **单向读取 station tokens**。
- PGA query：target coordinate embedding + learned PGA token。
- K/V：station feature embedding。
- mask：`key_padding_mask = ~station_valid`。
- 输出不加 query residual，使输出更直接依赖 station tokens。

核心观点：

> full self-attention 让模型自己发现 readout 路径；cross-attention 把 readout 路径直接指定出来。

建议图：`assets/architecture_cross_attention.png`。

---

### 13. Cross-attention 结果：PGA readout 明显改善

建议聚焦代表实验，不要把所有实验名堆满。

代表结果：

| Setting | Train Corr | Val Corr | Val MAE | Val slope | 解释 |
|---|---:|---:|---:|---:|---|
| overfit32 random | 0.7983 | 0.4615 | 0.4060 | 0.3879 | 初步摆脱常数 |
| overfit128 random | 0.8433 | 0.6183 | 0.3377 | 0.5078 | 样本增大后更稳 |
| fixed inputs + fixed targets | 0.9680 | 0.6214 | 0.2998 | 0.5741 | route sanity 通过 |
| input stations as targets | 0.9910 | 0.6409 | 0.3670 | 0.6287 | same-station 读出强 |
| event+pga first inputs | 0.5201 | 0.2408 | 0.3591 | 0.0281 | event+PGA 联合仍不稳 |

结论：

- cross-attention 对 PGA target readout 有明显帮助；
- fixed/input target sanity 证明模型能学到非平凡映射；
- event+PGA 联合任务仍可能压扁 PGA slope，说明 event 信息注入和多任务权重需要继续设计。

建议图：`assets/cross_train_val_detail.png` 或 `assets/result_cross_attention.png`。

---

### 14. Cross-attention 的剩余问题

把 cross-attention 的局限讲清楚，避免被理解为已经解决。

- fixed inputs + random targets 的 val 较弱，说明空间插值/外推仍难。
- event+pga 联合时 PGA slope 很低，event context 未必以正确方式进入 PGA。
- 目前没有显式距离衰减、路径效应、site term，target 只靠坐标 query 学空间场，样本效率低。
- 需要加入 relative coordinate / distance bias / event context gate / GMPE-style prior。

建议图：可以用 `assets/variance_compression.png` 支撑“pred std 仍不足”的观点。

---

### 15. Graph route：为什么“有 graph”还不够？

先讲旧 graph 失败，再引出 prior-residual。

旧 graph message passing：

- 希望通过 station-target edge 传播 station 信息。
- 但 old graph first-inputs：Train Corr 0.1419，Val Corr -0.0588，Val slope 约 0。
- graph exp1/2/3 也接近常数或不稳定。

关键结论：

> 图结构本身不是先验；如果没有可用的 station prior 和距离传播约束，graph readout 仍可能学成均值解。

建议图：`assets/graph_all_train_val_corr.png` 或旧 graph 与 prior graph 对比柱状图。

---

### 16. Graph prior-residual：把物理直觉变成可学习结构

方法细节：

1. `station_pga_prior_head` 从 single-station pretrain 加载，为每个输入台站预测 PGA prior。
2. distance baseline = 输入台站 prior 的距离加权平均。
3. GraphPGAReadout 使用 target-station edge features 学 residual。
4. final PGA = distance baseline + learned residual。

解释：

- single-station prior 提供局部观测强度；
- distance baseline 提供传播/衰减的低频趋势；
- residual 负责修正路径、方位、站点和非线性偏差。

建议图：`assets/architecture_graph_prior_residual.png` 或 `assets/architecture_graph_detail.png`。

---

### 17. Graph prior-residual 结果：当前最有解释性的改善

代表结果：

| Method | Train Corr | Val Corr | Val MAE | Val slope | Single val PGA corr |
|---|---:|---:|---:|---:|---:|
| old graph first-inputs | 0.1419 | -0.0588 | 0.3452 | -0.0005 | 0.8721 |
| prior-residual first-inputs | 0.3674 | 0.6118 | 0.2784 | 0.2976 | 0.8721 |
| prior-residual random-inputs | 0.5003 | 0.5602 | 0.3137 | 0.4090 | 0.7873 |

方差诊断：

- old graph pred std ≈ 0.0059，target std ≈ 0.6463。
- prior-residual pred std ≈ 0.1853，虽仍偏小，但明显摆脱近常数。

结论：

- 改善主要来自 prior + baseline + residual 的组合，而不是 graph message passing 本身。
- 当前 graph prior-residual 是最符合地震传播直觉、也最容易设计消融的路线。

建议图：`assets/graph_train_val_detail.png` + `assets/result_graph_prior_residual.png`。

---

### 18. 两条可行路线的阶段性对比

这页是技术判断页，帮助专家给建议。

| 路线 | 优点 | 问题 | 下一步 |
|---|---|---|---|
| Cross-attention | query-to-station 路由清晰；PGA sanity 强 | 空间传播先验弱；event+PGA 联合不稳 | distance bias、relative coord MLP、event context gate |
| Graph prior-residual | 可解释；利用 single-station prior 和距离 baseline；当前改善最明显 | 仍低估方差；需证明泛化和消融 | baseline-only / residual-only / prior-only / kNN / distance power |

一句话判断：

> cross-attention 更适合诊断 readout，graph prior-residual 更适合作为下一阶段主模型候选。

---

### 19. P-pick 和数据质量风险

建议把 P-pick 放在模型结果之后，因为此时听众已经知道为什么数据质量会影响 station feature 和 PGA label 对齐。

当前流程：

- 走时曲线粗定位；
- STA/LTA 在搜索窗口内 refine；
- P-pick 决定波形窗口、station SNR 和 feature 对齐。

风险：

- 粗 pick 偏差会影响 DiTing feature；
- 低 SNR 样本可能让 single-station prior 不可靠；
- HDF5 中部分 `p_picks` 是 aligned/global sample，可能超出 100 s 当前训练窗口，需要回到原始 aligned 记录抽查。

建议图：`assets/ppick_metadata_stats.png` + `assets/ppick_audit_waveforms.png`。

---

### 20. 当前主要问题与解释边界

集中讲 caveat，防止结论被过度解读。

- 当前仍是 overfit 诊断，不是最终泛化评估。
- 验证集很小，需要多 seed / 多 split。
- pred std 仍小于 target std，模型偏保守。
- event mag/loc 联合训练仍会常数化，需要单独定位 event readout。
- input/target 抽样策略影响明显：first/random/fixed/input target 不能混为一谈。
- 表格中个别实验 eval failed/no PGA metrics，应从主结论中剔除，只作为工程状态说明。

---

### 21. 下一步实验路线图

建议按优先级排序，而不是平铺所有想法。

**P0：结果可信度与数据闭环**

- P-pick 人工质检样本集；
- 统计 pick 偏差、STA/LTA 失败类型、低 SNR 样本；
- 多 seed / 多 split / 扩大 overfit_n；
- 统一指标表：MAE、Corr、R²、slope、pred std、target std。

**P1：Graph prior-residual 消融**

- baseline-only；
- residual-only；
- prior-only；
- prior + baseline；
- prior + baseline + residual；
- distance power `p`、kNN 邻居数、edge feature 消融。

**P2：Cross-attention 增强**

- relative coordinate MLP；
- distance bias；
- event context gate；
- mag/loc context 是否进入 PGA query；
- target-input station 重合/不重合分组评估。

**P3：event 任务重建**

- event-only cross-attention；
- mag/loc 与 PGA 分阶段训练；
- event token 是否作为 PGA context，而不是与 PGA 同时在 full self-attention 中竞争。

---

### 22. 结束页：请专家重点给建议

建议回到第 3 页的问题，并更具体：

1. P-pick refine：STA/LTA 搜索窗口、阈值、失败样本剔除策略是否合理？
2. PGA 建模：是否需要显式 GMPE / 距离衰减 / site term / 方位角 / 路径效应？
3. 模型结构：任意 target PGA 更适合 cross-attention、graph prior-residual，还是二者融合？
4. event 信息：mag/loc 应作为监督任务、latent context，还是显式条件？
5. 实验节奏：当前 overfit 诊断达到什么标准后，可以转向大训练集？

最后一句总结：

> 当前最值得推进的是：**先把数据质检和 prior-residual 消融做扎实，再扩大训练集验证 cross-attention/graph 融合模型的泛化。**

---

## 三、建议删减或移入备份的内容

为了控制 30 分钟节奏，建议正文不再展开以下内容，放备份即可：

1. 所有 21 个实验的完整逐项解释。正文只放代表实验和总览图。
2. `mag_cross_attention_overfit32`、`loc_cross_attention_overfit32` 这类 event 任务细节，除非专门讨论 event readout。
3. P-pick 抽样的所有 caveat，正文只讲主要风险，详细波形样例放备份。
4. loss 曲线单独两页可合并为一页；single-station loss 曲线放备份。
5. 模型参数量可保留一页，但不宜讲太久。

---

## 四、推荐备份页

### B1. 全部实验总表

来源：`tables/all_experiments_summary.csv`  
用途：专家追问某个实验名时使用。

### B2. Transformer ablation 精修表

来源：`tables/transformer_ablation.csv`  
用途：解释 query_transformer / mask_batch1 / query_no_transformer / direct_station。

### B3. Cross-attention 全部设置

来源：`tables/cross_attention.csv`、`assets/cross_train_val_detail.png`。

### B4. Graph 全部设置

来源：`tables/graph_results.csv`、`assets/graph_train_val_detail.png`。

### B5. Variance compression 诊断

来源：`tables/variance_summary.csv`、`assets/variance_compression.png`。

### B6. P-pick 波形抽样

来源：`assets/ppick_audit_waveforms.png`。

### B7. P-pick 元数据统计

来源：`assets/ppick_metadata_stats.png`。

### B8. Single-station pretrain loss

来源：`assets/single_station_loss_curves.png`。

### B9. Full model representative loss curves

来源：`assets/representative_full_loss_curves.png`。

---

## 五、优化后的 PPT 讲述节奏

30 分钟建议分配：

| 部分 | 页码 | 时间 | 目标 |
|---|---:|---:|---|
| 开场与结论 | 1–3 | 4 min | 告诉听众今天要判断什么 |
| 任务与实现 | 4–8 | 7 min | 建立共同语境 |
| 表征有效性与坍塌诊断 | 9–11 | 5 min | 证明问题在哪里 |
| Cross-attention | 12–14 | 5 min | 说明显式 readout 改善 |
| Graph prior-residual | 15–18 | 6 min | 给出当前最有希望路线 |
| 数据风险与计划 | 19–22 | 3 min | 收束到专家建议 |

---

## 六、建议对现有 v4 PPT 的具体修改

1. 将“主线判断”压缩成第 2 页 conclusion-first。
2. 将“希望专家帮忙看的点”保留为第 3 页，并与结尾页呼应。
3. 合并原第 3–4 页任务设置，避免重复。
4. 将 P-pick 详细页从第 6–7 页移动到模型结果之后。
5. 合并 loss 设计与 loss 曲线解释，突出“loss 下降不等于非坍塌”。
6. 原第 13–16 页可压缩为 2 页：结构 + ablation 诊断。
7. Cross-attention 保留 3 页：设计、结果、剩余问题。
8. Graph 保留 3 页：旧 graph 为什么不够、prior-residual 设计、结果。
9. 新增一页“两条路线对比”，帮助专家讨论。
10. 检查并统一 `direct_station` 数值来源，避免 `all_experiments_summary.csv` 与 `transformer_ablation.csv` 冲突。

---

## 七、最终推荐标题与摘要

### 推荐标题

**DiTing 表征接入多台站 PGA 预测：从 Readout 坍塌诊断到 Graph Prior-Residual**

### 推荐摘要

本次交流汇报 DiTing station representation 接入多台站 event/PGA 预测任务的阶段性实验。当前实验表明，single-station waveform 表征在 PGA、震级和震中距上均包含有效信号，但原始 TEAM 式 full self-attention readout 在小样本 overfit 中也容易出现 PGA、mag、loc 输出常数化。为定位问题，我们比较了 query-only、direct station、cross-attention 和 graph message passing 等路线。结果显示，显式 PGA target cross-attention 能明显改善 readout；旧 graph message passing 本身不足以避免坍塌，而引入 single-station PGA prior、distance baseline 和 learned residual 后，PGA train/val 相关性和输出方差均明显改善。下一步将重点围绕 P-pick 数据质检、prior-residual 消融、event context 注入和多 seed/split 泛化验证开展实验。
