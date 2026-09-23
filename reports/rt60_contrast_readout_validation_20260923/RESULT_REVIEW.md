# RT60 final-contrast readout 正式 validation 结果审阅

日期：2026-09-23。结论先行：**RT60 在预期的空间差分指标上产生了统计明确但幅度很小的改善，
同时使 normal 非输入目标的点误差和概率指标退化，因此 `rt60_mechanism_pass=false`；
legacy 结果仍为 `18/25`，`legacy_full_go=false`。按预注册规则，应保留同次 RT59 reference
作为下一轮 development parent，不选择 RT60，不追加 epoch，也不把本结果描述为 test 泛化结果。**

## 1. 证据与身份边界

- 分支：`rt60-final-contrast-readout`。
- RT60 implementation commit：`033f01a481fdb7499aff836abe6aaae35b72e5ca`。
- 本次可重复分析器 commit：`14e4ed83e3981245893b69b566f06f358961f6a0`；生成报告时
  worktree clean。
- 超算上传源码 manifest SHA-256：
  `405d533f7613adb4a6e52ac23096ae41a1c50d173417265b500c943bb7a27a38`。
- RT59 parent checkpoint SHA-256：
  `5dbc15c6af5b8d341dfd4a377a68217b1e1aa3589550690997c02e81332aa1cd`，
  metadata 明确为 RT59-v3 epoch 8。
- RT60 immutable reference readout SHA-256：
  `db11b3c6152ea8ea76d78441769da04e6f03bb313050d70c64e4d12e1259f021`。
- ZIP SHA-256：`26c75ca9dc730c20919855d9bff334495efe51b0acf9d474774d47662ca8c84a`；
  TAR.GZ SHA-256：`948b5f875166c5575b7107bd231ea7b61c1c1fda09547169eec4da1fbaba87ee`。
- random NPZ SHA-256：`300aa63d6538210ed41d153ab8385376f73b8fcc076d6351963cfded63158ad1`；
  normal NPZ SHA-256：`8b508fff8b3a4ceb26f2ad93075c2d685fc896a4ca3bddf4d82afaf2fdfc25a7`。
- 两份 formal metrics 均声明 `split=val`、`point_estimate=predictive_mixture_mean`、
  `PGA=log10(m/s²)`、checkpoint epoch 8、task id 正确。没有运行或读取 held-out test。
- ZIP 含两套 formal NPZ、metrics 和配置；TAR.GZ 含训练标量日志与配置。两个归档都不含
  `.pth` checkpoint，因此本报告能认证导出结果和 checkpoint metadata，不能用归档本身重载权重。

三组预测来自同一 NPZ 且逐目标对齐：historical 是原语义不变的 RT57 gamma=1.66，reference
是同次冻结 RT59 readout，candidate 是 RT60。候选 MDN 的 logits 与 reference 完全相同，
component sigma 最大差 `4.0e-8`，normal input 的 RT60 increment 最大绝对值为 `0`；
`reference + increment = candidate` 的最大浮点重建误差不超过 `3.58e-7`。

本报告统一采用以下差值方向：

```text
Δ(metric) = metric(RT60 candidate) - metric(same-forward RT59 reference)
```

原始 LaTeX：

```latex
\Delta(m)=m_{\mathrm{RT60}}-m_{\mathrm{RT59\ reference}}.
```

误差类指标中，`Δ < 0` 表示改善。

## 2. 数据、训练和评价是否完整

训练 resolved config 确认为 Japan 2000–2024 的 25 个年度 HDF5 shards、seed 42、8 个新 epoch、
`freeze_mode=rt60_contrast_readout_only`；仅 6 个 readout tensors / 67,077 scalars 可训练。
它使用全量年度训练输入和既有 event-level train split，不是只用一个小 smoke 子集。这里的“全量”
不表示把 validation/test 事件混入训练：manifest 仍声明过滤后 train/dev/test 为
9,627/1,383/2,769 events，formal 评价只使用固定 dev/validation。

训练标量完整覆盖 8 个 epoch，LR 精确为 `1e-4 × 4, 5e-5 × 2, 2.5e-5 × 2`；四个
contrast field buckets 每轮都有样本，readout gradient norm 有限。epoch 6 的 standard val loss
最低（`0.6480026841`），epoch 8 为 `0.6480399966`，但协议事前固定只评价最终 epoch 8，
所以没有基于 validation 挑 checkpoint。

formal validation 计数全部通过预期值核验：

| Protocol | Realtime rows | Events | Targets |
|---|---:|---:|---:|
| random non-input | 9,681 | 1,310 | 75,654 |
| normal all | 9,681 | 1,383 | 89,770 |
| normal input | — | 1,383 | 63,651 |
| normal non-input | — | 1,290 contributing | 26,119 |

random 中有 5,762 个至少含 5 个合法 query 的 fields，其中 actual-one-station 为 1,494 个。
所有置信区间均使用 event id 配对 cluster bootstrap，5,000 draws，seed 20260915；同一事件的多个
realtime fields 一起重采样。

## 3. RT60 相对同次 RT59 reference

### 3.1 预期机制确实被激活

| Random field metric | RT59 reference | RT60 | Δ(RT60−RT59) | 95% CI |
|---|---:|---:|---:|---:|
| all P95−P05 range abs. error | 0.465251 | 0.459486 | −0.005765 | [−0.006309, −0.005213] |
| one-station P95−P05 range abs. error | 0.680775 | 0.676849 | −0.003926 | [−0.005019, −0.003049] |
| one-station pairwise-delta MAE | 0.336202 | 0.335647 | −0.000555 | [−0.001122, −0.000235] |

三个方向都改善，两个事前要求的 one-station CI 上界严格小于零。random 全场 range ratio
由 `0.475336` 升至 `0.482669`，one-station 由 `0.178619` 升至 `0.183348`，说明 readout
没有只做 field-common 平移，而是真正恢复了一小部分 query 差异。

但效应幅度有限：one-station pairwise MAE 仅相对下降约 `0.17%`，one-station range error
约下降 `0.58%`，all-field range error 约下降 `1.24%`。candidate 的 range ratio 仍远低于
legacy 门槛 `0.55`，也低于 historical RT57 的 `0.510231`。

### 3.2 点预测和概率保真没有同时成立

| Population | Metric | RT59 reference | RT60 | Δ |
|---|---|---:|---:|---:|
| random | MAE | 0.237828 | 0.237824 | −0.000004 |
| random | RMSE | 0.311250 | 0.310753 | −0.000497 |
| random | NLL | 0.199018 | 0.197350 | −0.001668 |
| random | Brier | 0.183952 | 0.183801 | −0.000152 |
| normal all | MAE | 0.125006 | 0.125175 | +0.000169 |
| normal all | RMSE | 0.186050 | 0.186400 | +0.000350 |
| normal non-input | MAE | 0.199301 | 0.199880 | +0.000580 |
| normal non-input | RMSE | 0.255652 | 0.256526 | +0.000874 |
| normal non-input | NLL | −0.028293 | −0.027011 | +0.001281 |
| normal non-input | Brier | 0.157458 | 0.157634 | +0.000175 |

random MAE 的改善只有 `4.0e-6`，其 95% CI 跨零；random RMSE CI 刚好保持在改善方向。
normal non-input 的 MAE 与 RMSE 退化虽小（分别约 `0.29%` 和 `0.34%`），但 event-cluster
95% CI 分别为 `[+0.000246,+0.000933]` 与 `[+0.000365,+0.001424]`，不是浮点噪声。
normal all 的退化完全来自 non-input，因为 63,651 个 normal input predictions 按设计逐目标不变。

因此 14 个 conjunctive mechanism gates 中通过 8 个、失败 6 个。失败项正是：normal
non-input MAE/RMSE，normal all NLL/Brier，以及 normal non-input NLL/Brier。结论是“冻结表示下的
最终 readout 可以轻微改善空间反差，但无法同时保持 normal remote 精度”，不是“contrast loss
完全无效”，也不是“已经解决随机几何任务”。

## 4. Legacy 门控

RT60 相对 historical RT57 仍保留明显点指标收益：

- random MAE：`0.245578 → 0.237824`；
- normal all MAE：`0.131016 → 0.125175`；
- normal non-input MAE：`0.210371 → 0.199880`；
- normal input MAE：`0.098452 → 0.094520`。

但是 legacy 仍只有 `18/25` required gates 通过。7 个失败项为：

1. random slope `0.409613 < 0.45`；
2. random range ratio `0.482669 < 0.55`；
3. one-station range ratio `0.183348 < 0.25`；
4. one-station pairwise MAE `0.335647 > 0.330`；
5. random coverage1 相对 historical 的变化 `0.020581 > 0.01`；
6. one-station range absolute error 相对 historical 反而增加 `+0.018287`；
7. normal coverage1 相对 historical 的变化 `0.014693 > 0.01`。

四个 bias/slope diagnostic 均改善，但它们不是 required gate，不能用来翻转 NO-GO。

## 5. 决策与下一步审阅问题

按照事前约定，应：

- 将 RT60 保存为有价值的部分成功/机制负结果；
- 继续以 RT59 epoch-8 作为 development parent；
- 不因 epoch 6 val loss 略低而事后选择中间 checkpoint；
- 不自动追加 epoch、调权重、做第二 seed、跑 held-out test 或重复旧 smoke；
- 下一步若继续，应由 ChatGPT 先判断：结果是否已足以反证“只重拟合最终 readout”路线的充分性，
  以及是否应转向改善 frozen transport representation/conditioning，同时保留 RT59 的 normal 保护。

建议 ChatGPT 重点审查：

1. 三模型身份和 MDN 去归一化是否正确；
2. field 等权、P95−P05、`n>=5`、actual-one-station 与 event-cluster bootstrap 口径；
3. 14 个 mechanism gates 与 25+4 个 legacy gates 是否逐项忠于预注册文本；
4. 对“小幅空间改善 + 小幅但显著 normal remote 退化”的机制解释是否充分；
5. 下一轮是否需要表示层改动，以及最小、单一、可证伪的新实验应是什么。

机器可读入口依次为 `summary.json`、`mechanism_gates.csv`、`legacy_gates.csv`、
`group_metrics.csv`、`paired_ci.csv`、`field_metrics.csv` 与 `training_summary.csv`。
