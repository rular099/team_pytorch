# V01 速度输入 / P 前缺失：首轮结果与审计

整理日期：2026-10-03。仓库：`rular099/team_pytorch`；分支：
`exp/v01-velocity-prep-padding-control`。本轮开始 HEAD 为
`8f54477ed11bf2623f40121648f7fcc756ada1a2`；此前交付但未提交的 recovery 修复现归档为
`8829029f7a1a3ef49cedeccef9385298c683ac01`。结果提交的精确 SHA 见本文件所在 commit
及用户收到的 CODEX-RESULT。**超算上传源码哈希未回传，不把本地 Git SHA 当成超算源码证明。**

## 1. 一页结论

- 三个训练臂均有真实 init/last/best checkpoint；直接读取确认 last 和 best 都是 epoch 8，
  optimizer update 为 1,496，三个 init 文件及其保存的 191 个模型张量完全一致。
- 10 个 validation 中，**5 个 random 导出成功，5 个 normal 以同一错误失败**。
  这是部分完成的 validation，不是 held-out test，也不是全矩阵完成。
- random 四个速度条件在 84,259 个目标观测、1,194 个事件上完全配对。
  `vmissing` 模型在缺失视图的 MAE 为 0.236895，`vfull` 模型在完整视图为 0.244516，
  相对降低 3.12%；早期 1/3/5 s 合并降低约 3.36%。
- **同一模型仅切换输入视图的影响远小于两个训练臂的差别**。结果没有显示预期的
  “P 前缺失造成明显损失”，但不能据此宣称补零无害、删除前缀普遍有益或机制成立。
- A-pair 总体 MAE 为 0.235566，数值略好于速度臂，但有 **45 条额外重复 event/time
  sample**，且不是同仪器控制，不能作严格配对的速度/加速度优劣结论。
- 单输入台站的预测空间范围仍严重压缩：速度臂的 range ratio 中位数约 0.0063，
  约为真实范围的 0.63%。V01 没有解决原 RT55 的空间差异辨识问题。

## 2. 本轮工作与训练身份

读取用户回传的 `chaosuan_res/vel/{weights_*,eval_retry1}`，未重新训练、下载、物化数据或
提交 Slurm。归档此前 recovery 修复，并新增独立离线整理工具
`tools/summarize_v01_results.py`；不改模型、sampler、loss、标签或现有实验配置。

| 已核实项目 | 结果 | 证据 |
|---|---|---|
| 模型配置 | 三臂 `model_params` 相同 | `training_contract.json`、原配置副本 |
| 初始化 | 三臂 init 文件 hash 与保存的张量 hash 相同；epoch 0；无 optimizer | `checkpoint_audit.json` |
| 最终 checkpoint | 各 last/best 均 epoch 8；同臂 last/best 模型张量相同 | `checkpoint_audit.json` |
| 优化预算 | 三臂 optimizer tensor step 均 1,496；三类 LR 均 1e-4 | checkpoint 与 scalar CSV |
| 验证选择 | 固定 last、epoch 8；没有按本次指标挑选 epoch | metrics 与 resolved config |
| 数据年份 | 配置指向 2004–2024 共 21 个 derived shard | 原配置副本 |
| 原 split / parent | 三臂记录同一 frozen split 路径与 RT55 ep32 初始化路径 | 原配置副本；本轮缺实际 split/parent body |
| 模型计算 | `gemini_models.py` 未改；本轮整理也未改 `train_light.py` / `eval_checkpoint.py` / `gemini_util_light.py` | `analysis_provenance.json` |

共同 init 文件 SHA-256：
`6e91102989e1d6440dd878c6339001c1000e9a737e1071a7c6255bbe54d26577`。
保存模型张量的 canonical SHA-256：
`46e688d093019fa4e877ddaa86ac81e1fd7b07af574c504cf5ed380584158f43`。
checkpoint 是 `non_encoder_v1`：704 个冻结 encoder 张量没有存入，故本轮只能证明
保存的 191 个张量一致；冻结 encoder 的实际超算文件未回传，不能声称已做全模型逐张量比较。

训练 validation objective 首轮到末轮：vfull 1.162263 → 1.051565，
vmissing 1.168601 → 1.044732，apair 1.156330 → 1.050443。
这是多任务/训练 loss，不等于下表的无权重 PGA NLL。

配置继承的 `full_data_manifest` 仍写历史 2000–2024 / 25 shard / 原 KNET split 数，
**不是实际 V01 cohort**。本轮没有 preflight summary、cohort CSV 或 split CSV，
不能确认实际 train 事件数，也不能称全量日本速度数据训练。

## 3. 完成矩阵与 normal 失败

| checkpoint → input view | random | normal |
|---|---|---|
| vfull → vfull（FF） | 完整 NPZ + metrics + config + txt | 失败，无 NPZ/metrics |
| vfull → vmissing（FM） | 同上 | 同上 |
| vmissing → vfull（MF） | 同上 | 同上 |
| vmissing → vmissing（MM） | 同上 | 同上 |
| apair → apair（AA） | 同上，含重复 sample | 同上 |

五份 normal 日志均已加载 epoch 8，并在 `Running inference on val set (8358 samples)`
之后退出；共同终止错误：

```text
gemini_util_light.py -> PreloadedEventGenerator._get_one
ValueError: Found event without PGA idx=7
```

代码中的直接触发条件是 PGA target 候选集合为空。它发生在 generator，而不是模型
checkpoint 加载或 xFormers 警告处。**为何在该事件上变空仍未重现**：缺少对应 derived
HDF5/元数据，不能仅凭日志确定缓存、坐标、行选择或 mask 中的具体根因。
`idx=7` 是该 generator 的事件索引，不是已确认的物理 event ID。

五份终止 traceback 已保存到 `source_evidence/*.failure.txt`，状态见
`evaluation_status.json`。Slurm sacct 和 analysis job 日志缺失；原 analysis 使用全部
validation 的 `afterok` 依赖，因此“被失败依赖阻止”是推断，不能写为已核实的作业状态。

## 4. random：点误差和概率指标

下表为所有 7 个截止时刻（1/3/5/10/20/40/90 s）合并，固定 epoch 8、seed 42、validation。
误差坐标为 `log10(m/s^2)`，point estimate 为 MDN predictive mixture mean。
NPZ 的历史字段名 `pga_mu_best` 在当前 evaluator 中实际保存该 mixture mean。
NLL 是该坐标的无权重负对数密度；Brier 使用原配置阈值 -1.2；coverage 是误差落在
predictive mixture mean ± 1/2 倍 mixture standard deviation 内的比例，不是严格分位数区间。

| 条件 | 事件 | 目标观测 | MAE | RMSE | R2 | bias | slope | NLL | Brier | 1σ / 2σ coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FF | 1,194 | 84,259 | 0.244516 | 0.310286 | 0.307216 | +0.091564 | 0.399227 | 0.171931 | 0.202340 | 0.6199 / 0.9409 |
| FM | 1,194 | 84,259 | 0.244091 | 0.309952 | 0.308705 | +0.088778 | 0.395758 | 0.170561 | 0.201869 | 0.6209 / 0.9411 |
| MF | 1,194 | 84,259 | 0.236893 | 0.303735 | 0.336161 | +0.066403 | 0.399740 | 0.142036 | 0.194258 | 0.6397 / 0.9434 |
| MM | 1,194 | 84,259 | 0.236895 | 0.303782 | 0.335954 | +0.065769 | 0.397787 | 0.141841 | 0.194188 | 0.6401 / 0.9435 |
| AA（描述性） | 1,193 | 84,329 | 0.235566 | 0.303129 | 0.340229 | +0.048393 | 0.380772 | 0.137386 | 0.192385 | 0.6490 / 0.9463 |

所有已导出的有效目标均为 sampler 的 non-input target：速度四格含 triggered-noninput
60,080 与 untriggered 24,179；input-target 为 0，因此这里 all 与 non-input 数值相同。
**这不等于已做跨 A/V 视图、按真实传感器身份定义的 common-remote 审计**。
query-only 行进入有效波形输入槽的计数为 0。

| 条件 | 1 s MAE | 3 s MAE | 5 s MAE | 1/3/5 s 合并 MAE |
|---|---:|---:|---:|---:|
| FF | 0.284221 | 0.252923 | 0.240222 | 0.259122 |
| FM | 0.283235 | 0.252023 | 0.238883 | 0.258047 |
| MF | 0.276295 | 0.243933 | 0.231987 | 0.250739 |
| MM | 0.275782 | 0.243595 | 0.231843 | 0.250406 |
| AA（描述性） | 0.278042 | 0.241150 | 0.229824 | 0.249693 |

速度每个时刻均 12,037 个目标观测 / 1,194 个事件；早期合并为 36,111 / 1,194。
A-pair 每时刻目标/事件不同，见 `metrics_by_time_and_group.csv`，不可套用速度样本数。
同一目标重复出现在不同时间；“目标观测”不是独立台站数或独立统计样本数。

## 5. 严格速度配对与不确定性

使用 event + requested time + 精确导出 query 坐标键匹配，检查一对一、标签、target type、
current sample 和 first-pick sample 相同；四格无重复键且全部 84,259 个目标匹配。
`pga_target_indices` 是重排后的 sampler 槽位，不单独作为物理台站 ID。
本轮在四格中它也相同，但 NPZ 未导出传感器 ID，身份审计仍有边界。

以下为事件聚类 bootstrap，5,000 draws，seed 20260915；同一事件全部台站/时刻一起抽样。
差值定义为右条件 MAE 减左条件 MAE，正值表示右条件更差。CI 不覆盖训练 seed/DDP 随机性。

| 比较 | 1/3/5 s 差值 [95% CI] | 全 7 时刻差值 [95% CI] | 含义 |
|---|---|---|---|
| MM − FF | -0.008716 [-0.009655, -0.007787] | -0.007620 [-0.008285, -0.006954] | 两个训练/输入条件总体差异 |
| FM − FF | -0.001075 [-0.001580, -0.000568] | -0.000425 [-0.000698, -0.000142] | 固定 full 模型，仅删前缀 |
| MF − FF | -0.008383 [-0.009195, -0.007562] | -0.007622 [-0.008278, -0.006972] | 固定 full 视图，换训练臂 |
| MM − MF | -0.000332 [-0.000705, +0.000031] | +0.000002 [-0.000187, +0.000199] | 固定 missing 模型，仅换视图 |

event-macro 重加权后 MM−FF 方向仍一致；FM−FF 全时刻的 event-macro CI 则跨零。
完整单时刻、event-macro 和第五项 MM−FM 比较均在 `paired_event_cluster_ci.csv`。
MM−FF 与 FM−FF 是原计划的核心比较；另外三项作解释性补充，未作多重比较校正。
`paired_event_sufficient_statistics.csv` 保存每个事件的误差差值和/计数/均值，可离线复算 CI，
无需把原始 51 MB NPZ 推到 Git。

**可支持的窄结论**：当前速度 random validation 中，这个实际剂量的 P 前缺失没有导致
明显 pointwise 退化；更大差异出现在两个训练臂，而不是同一模型的视图切换。
**待审阅推断**：可能涉及训练适配/正则化/噪声上下文影响；仅一 seed、8 轮、正常几何缺失，
不能区分这些解释，也不能外推成“历史加速度 padding 是/不是主要瓶颈”。

## 6. 实际干预剂量与空间性能

FF/FM 的选中 source 观测共 38,302（跨事件、时间；不是独立传感器数）。P 前有效时长
平均由 9.091 s 降为 6.089 s；实际删除均值 3.002 s、中位数 0.970 s、P90 8.700 s；
21,380 个 source 观测发生删除，未发现时长增加。vmissing 不是“全部 P 前归零”视图。
两视图事件、selected source indices、有效输入 mask、source 坐标、绝对 cutoff 的导出
sample index、template/retained 值及 post-P 支持样本数逐项相同。
**只有支持计数的核对，不是原始 post-P 波形值逐样本证明**。

以下 field 描述指标仅统计每个 event/time 至少 5 个有效目标的速度样本；每个合格
sample 等权，range ratio 为预测 PGA 空间跨度与真实跨度之比，pairwise difference MAE
衡量台站两两 PGA 差值的误差。该坐标仍为 `log10(m/s^2)`。

| 条件 | 单输入合格 sample / 事件（全 7 时刻） | range ratio 中位数 | 两两差值 MAE |
|---|---|---:|---:|
| FF | 1,514 / 781 | 0.006364 | 0.354286 |
| FM | 1,514 / 781 | 0.006381 | 0.354285 |
| MF | 1,514 / 781 | 0.006339 | 0.354289 |
| MM | 1,514 / 781 | 0.006350 | 0.354288 |

单输入总体 pointwise MAE 从 FF 0.308553 降到 MM 0.300529，却没有恢复站间幅值差异。
全部输入数的 range ratio 中位数约 0.52，不掩盖单输入约 0.0063 的退化。
A-pair field 表仍是原导出上的描述统计，受重复 event/time 污染，不纳入上述机制证据。

## 7. A-pair 与完整审计的缺口

A-pair 请求/导出 8,358 行，但只有 8,313 个唯一 event/time，存在 45 条额外重复；
有效事件共 1,193，速度为 1,194。采样计数和选中 source 数也不同。
代码的 `_EmptySample` 处理会用后续样本替代当前样本，这与观察到的重复/缺失一致，
但仍是解释性推断，尚未核实每条原始缺失为何发生。本轮**未静默去重或伪造公平 cohort**。

尚缺：preflight summary / cohort counts / protocol lock / split CSV、source identity manifest、
Slurm sacct / job logs、source/query sensor IDs、原始 mask/波形值、原始单位/响应 provenance。
三臂的冻结起点和 8 轮预算已确认到保存张量/优化状态层面；实际 train cohort、训练可训练
参数名单、world size 和完整冻结 encoder 身份不能仅凭该结果包重新认证。

## 8. 复现与下一步审阅范围

在新目录复现（程序拒绝覆盖已有报告）：

```bash
python tools/summarize_v01_results.py \
  --input-root ../chaosuan_res/vel \
  --output-dir /tmp/v01_review_fresh \
  --audit-checkpoints --bootstrap-draws 5000 --bootstrap-seed 20260915
```

依赖 numpy/pandas；`--audit-checkpoints` 额外需要 torch。只读原始结果，不读取 train/test
波形。输入清单覆盖 627 个文件、共 4,232,149,207 bytes，每个文件 SHA-256 已记录。
只同步约 3 MB 文本/CSV/JSON，不同步原始权重、NPZ、HDF5 或完整 Slurm 日志。

建议 ChatGPT **先审已有核心结果**，判断结论边界与最小必要补救。若必须补 normal 或修复
A-pair 导出，应优先复用现有 epoch-8 权重，保留原结果，明确排除/保留事件策略后只补必要
验证；这是待批准方案，不是本轮已实现或已提交任务。不要自动恢复下载、重新 preflight、
重复三臂训练、继续 RT62、增 epoch/扫参/多 seed 或增加一串 smoke。

本地验证：完整 `unittest discover` 135 tests PASS（包括 5 项新增离线统计测试），
四个 shell 的 `bash -n` PASS；真实 5 个 NPZ 的主要指标与 metrics JSON 全部复算一致，
9 个 checkpoint 元数据/张量与文件 hash 已读验。更完整的 source/protocol 审计仍待缺失材料。
