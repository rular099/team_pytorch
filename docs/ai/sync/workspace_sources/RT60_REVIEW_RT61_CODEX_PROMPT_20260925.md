# RT60 审阅结论与 RT61 Codex 执行任务书

日期：2026-09-25。

## 0. 交接身份与结论

- repo：`rular099/team_pytorch`
- 审阅分支：`rt60-final-contrast-readout`
- 实际读取 HEAD：`00d624d2fb01c6dd98cac150043dd15b7d8366ff`
- RT60 结果证据 commit：`81be754bee32e41a28c79dc5611ff26f55efebd4`
- RT60 implementation commit：`033f01a481fdb7499aff836abe6aaae35b72e5ca`
- 下一轮任务：`20260925-rt61-wave-geometry-residual-conditioning`
- 建议新分支：`rt61-wave-geometry-residual-conditioning`

**结论：保留 RT59-v3 epoch 8 作为 development parent。RT60 记录为“统计明确但幅度很小的空间机制改善，同时 normal remote 退化”，不升级、不续训、不改选中间 checkpoint。下一轮只做一次受控的表示补充实验；不同时改变采样、损失权重、冻结 backbone、概率输出形状和训练预算。**

本任务书由 ChatGPT 基于仓库报告、CSV、配置、分析器与模型相关代码的独立审阅形成。ChatGPT 没有重载超算权重或重跑训练，也没有独立取得原始正式 NPZ 后重算全部结果。报告归档没有 .pth 权重；本轮认证范围不能扩大成“已独立复现训练”。

## 1. 经核对的 RT60 事实

评价坐标为 `log10(m/s^2)`，点预测是该坐标中的 predictive mixture mean。下表是同目标、同次前向的 RT59 reference 对照，不是 RT57 historical 对照。

| 指标 | RT59 reference | RT60 candidate | 解释 |
|---|---:|---:|---|
| random MAE | 0.2378282732 | 0.2378242321 | 改善约 0.0017%，CI 跨零 |
| random RMSE | 0.3112498031 | 0.3107532629 | 改善约 0.1595% |
| actual-one-station pairwise-delta MAE | 0.336202（四舍五入） | 0.335647（四舍五入） | 改善约 0.165%；配对差值 CI [-0.00112218,-0.00023475] |
| actual-one-station P95−P05 range abs. error | 0.680775 | 0.676849 | 改善约 0.58% |
| normal non-input MAE | 0.1993005417 | 0.1998803984 | 退化约 0.2909%，配对 CI 严格为正 |
| normal non-input RMSE | 0.2556522452 | 0.2565264952 | 退化约 0.3420%，配对 CI 严格为正 |
| normal non-input NLL | -0.0282925735 | -0.0270111099 | 退化 |
| normal non-input Brier | 0.1574583455 | 0.1576336002 | 退化 |

random：9,681 realtime rows、1,310 events、75,654 targets。normal：9,681 rows、1,383 events、89,770 targets，其中 input 63,651、non-input 26,119。random 合格空间场为 5,762 个，actual-one-station 为 1,494 个，后者来自 767 events，而不是 1,494 个独立事件。

14 个 conjunctive mechanism gates 通过 8 个；legacy required gates 通过 18/25。四个额外 diagnostic gates 不能加入 required gate 分母，也不能翻转 NO-GO。normal input 的 RT60 increment 按设计为零，不能用大量容易的 input targets 稀释 remote 退化。

训练是 seed 42、全 25 个 Japan 2000–2024 年度 shards 中固定 train split、8 个新 epoch，只有 residual readout 的 6 个 tensors / 67,077 scalars 可训练。不是小样本 smoke。validation 曾反复用于开发，不能当成新独立检验。

对长期 normal 目标，还必须保留 RT55 epoch-32 normal 历史基线，而不只追逐 RT57/RT59 相邻版本。仓库快照中 RT55 normal all MAE 约 0.1166，而当前 RT59 约 0.1250；non-input 则 MAE/RMSE 呈混合得失。该历史比较应在能核对逐目标身份时再升级为严格配对结论，禁止把快照四舍五入数值用于精确门控。

## 2. 研究判断：能推出什么，不能推出什么

1. RT60 的空间损失是最终预测误差去中心后的平方和，而不是无监督地强行拉大预测方差。实现表达式为 `sum((e-mean(e))^2)/(n-1)`，与平均 `0.5*(e_j-e_k)^2` 等价，且已针对 mask 与 DDP reduction 有测试。
2. normal 保护是软约束：`0.05 * ReLU(SmoothL1(candidate,y)-SmoothL1(reference,y))`，不能保证 MAE、RMSE、NLL、Brier 同时不退化。
3. “冻结上游 + 当前目标 + 当前预算”未获得双目标改进，**不等于**证明所有 readout-only 方案均不可能成功，更不等于已证明损失、采样和可观测性完全没有问题。
4. 空间范围比仍小，提示值得研究，但不能把 `1-range_ratio` 当成可兑现的误差改善百分比。单站、早期波形可能无法唯一识别真实震源/路径/场地；预测均值的收缩可能部分来自合理的不确定性。
5. 训练日志中 random 与 normal-remote 的 target 数量差距很大。原损失按 group 归一化，不能仅据原始计数断言梯度权重错误；但稀少 normal-remote 场的覆盖与梯度噪声仍是解释边界。本轮保留采样不变以避免混淆变量。

## 3. RT61 唯一主假设

**在保持 RT59 全部已学会的 local、anchor、set refinement、pair encoder、station weighting、level、backbone 与 MDN 概率形状不变的前提下，给 remote residual readout 增加一条低秩“波形/事件状态 × 查询几何”交互旁路，能否比 RT60 的纯 readout 重拟合实现更强的空间差分改善，并同时降低 normal non-input 点误差？**

本实验不引入新观测数据，不声称获得了原本不存在的信息；它只让 readout 在冻结 pair encoder 可能压缩信息之前，直接读取已存在的波形状态和几何组合。

禁止把本轮扩大为新 Transformer、全 backbone 解冻、多个 loss 扫描、多个 rank/seed 扫描、ground-motion diffusion、完整重建数据集或一组互相不可比的模型。

## 4. 必须读的仓库文件

先核对 Git 状态、分支与完整 commit；新 HEAD 有变化时报告与本任务 base 的差异，保留用户已有变更，不 reset 用户工作。

- `AGENTS.md`
- `docs/ai/PROJECT_CONTEXT.md`（这是带日期的 RT55/RT56 快照，不是 RT60 最新状态）
- `docs/ai/README.md`
- `SESSION_SUMMARY.md`（同样注意日期）
- `docs/ai/CODEX_RESULT_20260923_RT60_CONTRAST_READOUT_VALIDATION.md`
- `reports/rt60_contrast_readout_validation_20260923/RESULT_REVIEW.md`
- 同目录 `summary.json`、`group_metrics.csv`、`mechanism_gates.csv`、`legacy_gates.csv`、`paired_ci.csv`、`field_metrics.csv`、`training_summary.csv`
- `gemini_models.py` 中 `TargetConditionedTemporalPool._geometry_features`、`RT59ResidualTransportHead`、`PGAAnchorResidualTransportHead` 与最终 PGA MDN 合成路径
- `tools/rt60_contrast_objective.py`
- `tools/rt59_dual_objective.py`
- `tools/analyze_rt60_contrast_readout_npz.py`
- `tools/analyze_rt59_dual_objective_npz.py`
- `tests/test_rt60_contrast_readout.py`
- `train_light.py`、`gemini_util_light.py`、`eval_checkpoint.py` 中相关 trainability、mask/坐标、reference/export 路径
- `pga_configs/transformer_japan_full_2000_2024_rt60_contrast_readout_seed42_chaosuan.json` 及继承的 RT59 配置、启动器

将本任务书作为新预注册文档提交到 `docs/ai/`。本地若有仓库外的 `RT59_REVIEW_RT60_CODEX_PROMPT_20260920.md`，原样归档并记录 SHA；找不到就标注缺失，不允许“重建一份原始版本”。

## 5. 模型实现：固定原表示，新增 residual-only 的交互旁路

### 5.1 符号与具体结构

索引 i 为实际有效输入台站，q 为查询点。

- `h0[q,i]`：原 RT59 `pair_encoder` 的输出，保持冻结。
- `z[i]`：拼接现有冻结的 `station_u[i]`、`station_d[i]`、`event_emb`、`refinement`；不包含标签、未来样本或 protocol id。
- `phi[q,i]`：原 `_geometry_features` 的 10 维输出，即 query、station、relative vector、distance 的既有组合。保持原坐标预处理；不要把当前归一化坐标的距离擅称为 km。
- `r=16` 固定，不搜索 rank。

终端可读公式：

```text
a[i]     = tanh(Ww · LN_no_affine(z[i]))       # r=16
b[q,i]   = tanh(Wg · phi[q,i])                # r=16
h1[q,i]  = h0[q,i] + Wo · (a[i] ⊙ b[q,i])   # hidden_dim=256
```

原始 LaTeX：

```latex
a_i=\tanh(W_w\operatorname{LN}(z_i)),\quad
b_{qi}=\tanh(W_g\phi_{qi}),\quad
h^1_{qi}=h^0_{qi}+W_o(a_i\odot b_{qi}).
```

`LN_no_affine` 不引入可训练仿射参数；三个投影均不使用 bias。`Ww/Wg` 采用标准 Xavier 初始化；`Wo=0`，保证初始化 `h1=h0`。不要同时将 Ww、Wg、Wo 全部置零。invalid station/query 在进入特征运算前清零，防止 NaN padding 污染，最后继续使用既有 pair/query mask。

student residual readout 使用 `h1`，并保留原 readout 的 `anchor_residual` 与 `base_difference` 两个 scalar。既有 distance gate、local/remote coordinate routing、MDN common-mean-shift 逻辑不改。

### 5.2 这是最关键的 reference 隔离约束

原 RT60 `_rt60_reference_residual(residual_input)` 虽然冻结了 readout tensors，却复用了当次 `residual_input`。**若把 student 的 h1 传入 reference，就不再是冻结 RT59 对照。**

必须明确构造两套输入：

```text
student_input   = cat(h1, frozen_anchor_residual, frozen_base_difference)
reference_input = cat(h0, frozen_anchor_residual, frozen_base_difference)
```

- `score_head` 只能接收 h0；softmax station weights 保持 RT59 值。
- `level_head`、anchor、local、set_input/set_blocks、pair_encoder 和所有 backbone 均冻结。
- immutable RT59 readout 只能接收 reference_input。
- 所有 student 修改后，reference mean/MDN 必须仍等于真正的父模型输出。
- 不使用协议标签选择 normal/random 两套模型；local/remote 路由仍只依据可观测的坐标匹配。
- 不通过当前 query 集合的均值、分位数或其它 query-label 统计量做推理修正。

终端可读输出公式：

```text
remote Δ[q] = sum_i frozen_weight[q,i] * frozen_distance_gate[q,i]
              * (student_readout(h1[q,i], scalars)
                 - RT59_readout(h0[q,i], scalars))

mu_RT61[q,m] = mu_RT59[q,m] + remote Δ[q]
input Δ[q]  = 0
alpha_RT61  = alpha_RT59
sigma_RT61  = sigma_RT59
```

原始 LaTeX：

```latex
\Delta_q=\sum_i w^{59}_{qi}d_{qi}
\left[r_\theta(h^1_{qi},s_{qi})-r_{59}(h^0_{qi},s_{qi})\right],\quad
\mu^{61}_{qm}=\mu^{59}_{qm}+\Delta_q.
```

该式只是说明差分构成；实现可以保留现有 applied_delta 合成代码，但导出必须满足恒等式。

### 5.3 可训练参数与兼容性

只训练：

1. 原 `pga_anchor_residual_transport_head.transport.residual_head.*` 的 6 tensors / 67,077 scalars；
2. 新 `...transport.rt61_wave_geometry_adapter.*` 的 Ww、Wg、Wo。

最终参数数目由真实维度计算并写入 manifest，不猜总数。新功能显式 opt-in，例如 `use_rt61_wave_geometry_adapter`；旧 RT55–RT60 flag 关闭时 parameter keys、加载、推理语义不变。父 checkpoint 只允许新 adapter 前缀缺失，不放宽其它 missing/unexpected key 校验。

原 RT60 的“non-readout fingerprint”不适合直接套用到新增可训练 adapter。新断言应排除明确列出的 6 个 readout tensors 和新 adapter tensors，再对所有其余共享旧参数与 buffers 做 exact fingerprint。不得借此忽略其它变化。

## 6. 训练协议：只改变上述可学习表示通路

### 6.1 父模型

从真实 RT59-v3 epoch-8 checkpoint 做 weight-only 初始化，不从 RT60 继续训练。

```text
RT59 parent SHA-256:
5dbc15c6af5b8d341dfd4a377a68217b1e1aa3589550690997c02e81332aa1cd

RT59 immutable readout SHA-256:
db11b3c6152ea8ea76d78441769da04e6f03bb313050d70c64e4d12e1259f021
```

检查真实 checkpoint epoch、task_id、normalization、状态键和文件哈希。缺少真实父权重则只完成实现/本地测试/运行准备，明确 `HPC NOT RUN`，不得把 metadata 当权重、不得改用“相似文件”。

### 6.2 完全保留的实验设置

- seed 42，固定既有 train/dev/test event manifest，不读取 held-out test。
- Japan 2000–2024 的 25 个年度 train shards 全量既有训练 split。
- 既有 normal/random 采样、实时 cutoff、mask、target selection、normalization 不变。
- 正好 8 个新 epoch；只正式评价最终 epoch 8。不因 standard validation loss 更低改选 epoch 6 或其它 epoch。
- Adam、betas=(0.9,0.999)、eps=1e-8、weight_decay=0。
- readout 与 adapter 采用同一固定 LR：前 4 epochs 1e-4，随后 2 epochs 5e-5，最后 2 epochs 2.5e-5。
- 保留已有 gradient clip 1.0 的口径，施加到本轮全部允许训练参数，并记录 pre/post clip norms。
- frozen modules 维持 eval 模式；不得因 `model.train()` 使冻结 backbone dropout/buffers 漂移。

### 6.3 损失不改

使用 RT60 的同一个 final-field objective：

```text
L = grouped_point + 0.05 * remote_parent_regret + 0.40 * field_contrast
point group weights = [0.5,0.25,0.25]
smooth_weight = 0.1
mse_weight = 0.1
smooth_beta_model = 1.0
field bucket weights = [0.25,0.25,0.125,0.125]
```

保持模型坐标内的原损失单位、group/field 等权归约和 DDP global counts。不额外引入 variance matching、range hinge、PCGrad、蒸馏权重调节、更多 station sampling 或概率 sigma 学习。否则无法把结果解释为这条表示旁路的作用。

这不是对原软约束充分性的认可，而是固定混杂因素的受控实验。normal 是否改善，仍必须由独立 reference 与硬性结果门控判断。

## 7. 只做与本次改动相关的本地验证

在用户批准新的有成本训练前，完成以下本地单元/集成验证；不是重跑旧大规模 smoke 或诊断套件。

1. **初始化恒等性**：固定输入、eval、同设备，同一次父模型与 RT61 初始输出逐目标一致（允许预声明浮点容差）；reference + increment = candidate；input increment=0。
2. **teacher 不漂移**：随机修改 adapter 和 student readout，再做两个 optimizer steps；reference、weights、level、logits、component sigma 与 frozen shared-state fingerprint 不变。不能只查 teacher 六个 tensors 的哈希。
3. **梯度可达**：第一步 Wo 有非零有限梯度；Ww/Wg 因 Wo 零初始化第一步可为零，后续步骤应验证可获得有限非零梯度。旧参数没有梯度或状态变化。
4. **结构语义**：query permutation、追加其它 queries、query 分块/批处理、重复 query 均不改变原 query 预测；输入 permutation 等变。保留既有 ambiguous observed-match 规则，不暗改 actual_station_count 定义。
5. **mask/边界**：actual K=1、多站、padding、空/无效 query、NaN padding、observed 与 non-input 分支；不读取 query 波形或未来波形。
6. **加载/保存**：新参数白名单、resume、teacher 恢复、新 checkpoint 的 delta 重建与原 checkpoint 推理一致；原 RT60 测试和直接受影响的 RT55–RT59 兼容测试通过。
7. **分析器回归**：P95−P05 与 max−min 区分，n>=5、actual K=1、field 等权、event-cluster bootstrap；缺字段应 fail closed。保持旧分析器输出含义，不覆盖 RT60 报告。

不要把 no-op teacher drift test 做成“只改变新 adapter、reference 仍错误地接收 h1 却没有观察输出”的空测试。必须用非零 adapter 并断言原 reference 输出不变。

## 8. 评价：一个完整新模型运行，复用已有对照

### 8.1 身份与数据

评价只用既有固定 validation normal 与 random 协议。预期计数沿用第 1 节；计数/目标身份变动属于 `INVALID COMPARISON`，不是科学成功或失败。

必须区分：

- historical RT57 gamma=1.66；
- immutable RT59 reference；
- 新 RT61 candidate；
- RT60 既有导出结果（只有在 event/time/query/target/mask 严格对齐后，才用于配对跨实验比较）。

不得重用 `val_rt59_base_*` 表示 RT59；它仍是历史 RT57。新加 `val_rt61_reference_*`、`val_rt61_increment`、candidate MDN 等明确字段，或采用等价但无歧义的 schema。

可在同次前向额外导出 **RT61_adapter_off_counterfactual**：采用训练后的 RT61 readout、关闭 adapter 的输出。它不是 RT59，也不是独立训练的 RT60，不得混名。这是旁路使用诊断，不是一次额外训练消融。

### 8.2 必报指标

保留现有 14 mechanism + 25 required legacy + 4 diagnostic legacy 的全部指标与定义。误差差值统一为 candidate − reference。

按 random、normal_all、normal_input、normal_noninput 报告 MAE/RMSE/R2/bias/slope/P90/P95、within 0.1/0.2 dex、NLL/Brier、sigma、coverage。空间指标按 n>=5 合法 query 的 field 等权；single 只能按实际有效 K=1，不能按“请求了1站”代替。继续 event-cluster paired bootstrap 5,000 draws、seed 20260915。

必须同时按 target_type（input/triggered non-input/untriggered）、实际台站数、既有 1/3/5/10/20/40/90 秒、震级/PGA 的既有分桶输出样本数。重点看 normal non-input 与 untriggered，而不是只报告 all-target 平均值。

补充 Brier/校准诊断可以从同一已导出的 MDN 计算，不需再训练：原 -1.2 dex 阈值必须保留；额外多个 PGA 阈值需在训练前固定，单位明确，不得根据本轮结果调整。它们仅做解释，不追认新的成功门槛。

MDN 的 mean±sigma coverage 不是任意高斯混合的固定名义置信区间。保留 legacy 口径，同时可从导出 MDN 计算真正的 CDF/PIT 与固定概率分位数覆盖。不得把 coverage 增大一概称为失准或把覆盖阈值通过一概称为概率校准完整。

### 8.3 新增一项低成本机制分析，不另外训练模型

使用已有和本轮 NPZ，分解每个 field 的绝对误差与相对结构误差：

```text
e[q] = pred[q] - truth[q]
field_MSE = mean(e)^2 + mean((e-mean(e))^2)

p_c = pred - mean(pred)
y_c = truth - mean(truth)
a_star = dot(p_c,y_c) / dot(p_c,p_c)
orthogonal_error = mean((y_c-a_star*p_c)^2)
```

原始 LaTeX：

```latex
\operatorname{MSE}(e)=\bar e^2+\frac1Q\sum_q(e_q-\bar e)^2,\quad
a^*=\frac{\langle p_c,y_c\rangle}{\langle p_c,p_c\rangle}.
```

prediction 近常数时标记不可辨识，不用 eps 把 a_star 变成巨大但无意义的结果。可按正尺度 a>=0 再报一个明确命名的受约束诊断；有符号 a_star 用来发现反向空间关系。

这些使用真值的每场最优尺度仅是离线 oracle 诊断，不是可部署校准器，不能应用到正式 predictions，不能把其可降低误差等同于网络一定能学到的幅度。

它回答：空间改善来自正确形状、单纯放大、还是场均值偏移？normal 退化是否主要是整场 bias，还是相对结构也变差？保留每场结果，让 ChatGPT 可以独立分析，而不是仅提交结论截图。

## 9. 预注册判据与有限停止规则

### 9.1 科学有效性

身份、目标对齐、冻结 reference、causality、normal input 不变、MDN logits/sigma 不变必须先通过。否则 `INVALID`，不能统计“通过了几项指标”。

### 9.2 机制与实际收益分开命名

- `mechanism_pass`：沿用 RT60 的 14 个 conjunctive gates，不放宽。即 single pair MAE 和 single range error 的改善 CI 上界<0，random 点/概率不退化，normal remote 点误差不退化，normal 概率不退化，input 恒等。
- `useful_joint_progress`：在 mechanism_pass 基础上，actual-one-station pairwise-delta MAE 相对 RT59 至少下降 **2%**，且 normal non-input MAE、RMSE 的配对 CI 上界均<0。
- 2% 是本任务在训练前设置的最低实际收益标准，不是承诺的效果或自然界阈值。用完整精度的 baseline 计算阈值，约为 0.32948；不能用本文四舍五入数值执行门控。
- normal non-input MAE 至少下降 1% 可作为更高目标，但不事后改写上述二元门控。normal 精度是真实目标，不止“没有显著变差”。
- `legacy_full_go`：仍须满足 25 个 required legacy gates；4 个 diagnostic 分开列出。
- 任何 development GO 都不等于正式推广或 SOTA。没有新的独立测试和公平外部基线，不能宣称泛化或前沿性能已确立。

### 9.3 停止与解释

- 有正常精度/概率退化：保留 RT59，不升级。
- 只出现 RT60 量级的微小收益、没达到 useful_joint_progress：记为部分机制结果，不追加 epoch、不换 seed、不扫 rank/权重。
- 空间改善但 normal 又退化：说明在固定目标下新增交互仍未突破双目标限制，不能把空间范围更大包装成成功。
- 两类指标均改善：提交可审阅证据，不自动打开 held-out test；下一阶段再预注册独立验证与外部基线。
- 8 epoch 完成即停止。故障恢复可以继续同一 run，但必须恢复 optimizer/scheduler/RNG 与采样状态，不把重启变成新 seed 实验。

## 10. 图表和证据交付

新报告目录建议：`reports/rt61_wave_geometry_validation_YYYYMMDD/`。不覆盖旧结果。

必须提交到 GitHub 的轻量产物：

- `RESULT_REVIEW.md`、`summary.json`、`decision.json`；
- 新预注册任务书、resolved config 脱敏版本、精确训练参数白名单、代码与数据 split/source manifests；
- `group_metrics.csv`、`strata_metrics.csv`、`field_metrics.csv`、`paired_ci.csv`、全部 gates；
- `field_error_decomposition.csv`：每个 event/time/field 的均值误差、centered error、range、pair MAE、oracle projection 诊断；
- `reference_invariance.json`：真正父模型与同次 reference、训练前后冻结状态、normal input、alpha/sigma 的数值误差；
- `training_summary.csv`：每组/field bucket counts、loss 组成、各允许模块梯度/参数变化、LR、训练/验证 loss；
- normal_noninput 与 random 的统一坐标、统一色阶真值-预测密度图；RT59/RT60/RT61 的空间指标、配对差值和时间/站数曲线；不得只画最好的事件；
- 固定挑选规则的成功、退化、中位例子，所有图有坐标/单位/样本数/图注及对应数据；
- 完整产物 SHA-256 manifest、环境、运行命令、checkpoint metadata 与实际 checkpoint 哈希；
- 小型 RT61 trainable-delta 文件或等价重建包（新 adapter、更新后的 readout、所需元数据），并在真实父权重上验证重建与完整 checkpoint 一致。

大型父/候选 checkpoint、NPZ、Slurm 原日志继续按既有方式保存到超算/本地，不硬塞入 Git。必须记录可定位的脱敏路径/环境变量与哈希。小 delta 不等于完整权重，报告要说明仍依赖固定父 checkpoint。

## 11. 分工和授权

**ChatGPT 已完成的工作**：本次审阅结论、研究假设、表示旁路结构、对照隔离设计、固定预算、成功/失败判据、证据需求。

**Codex 的任务**：核对真实文件和父权重、实现、相关本地测试、固定单实验配置/启动器、checkpoint 与分析导出、提交轻量证据。不能把方案判断重新变成无人约束的模型搜索。

**用户/现有超算流程**：决定并执行有成本计算授权。Codex 本地实现和作业准备不等于有权自动提交新的 HPC 训练。若本次转交同时授权正式运行，则只执行这一个预注册 run；否则交付 ready-to-run 命令并明确 `not submitted`。无需重复旧问题，缺少执行环境就如实标注。

## 12. 前沿定位与后续边界（文献依据，不是本轮额外实验清单）

截至 2026-09-25 本次核验的相关原始研究：

- TEAM：Münchmeyer et al., GJI 2021，DOI `10.1093/gji/ggaa609`。任意台站位置/数量、概率 PGA、多个强震阈值的误报漏报及预警提前量都是既有比较维度。
- QuakeFormer：Feng et al., arXiv `2412.00815`，2024-12 提出的预印本；此处依据公开 v1 正文，未把它擅称为已核实的某期刊论文。绝对/相对几何、masking 和预测/插值/早期预警之间的迁移值得参考；其不同任务的输入可用性不同。
- TREAD：Xu & Chen, SRL，2026-07-21 first online，DOI `10.1785/0220260045`。本次可核验摘要明确采用距离感知辅助学习，并在 CWA/Kyoshin 数据上比较；未获取完整论文与可复现代码，不编造其具体数值或实现。
- TT-SAM：Chen et al., JGR Machine Learning and Computation，2026-04-29，DOI `10.1029/2025JH001005`。多站波形、物理特征与场地信息的组合具有参考意义，但其主要目标是 PGV，不能把指标直接同当前 PGA 横比。

达到前沿性能必须另有相同数据划分、因果 cutoff、PGA 定义、站点可见性、评价成本的外部基线。此轮只产出 benchmark protocol 文档：保留 RT55/RT57/RT59 内部锚点，并规划 TEAM 与适配任务的现代方法、可观测条件下的简单物理/空间基线。使用 catalog 真震源或最终观测 PGA 的方法只能标记为 oracle，不能与实时模型同列主榜。

不得现在为了“前沿”并行复现多套模型，也不现在打开 held-out test。先取得真实、实用的双目标改进，再做锁定模型后的独立验证。所有历史 validation/test 暴露必须在文档中记录；未来未用结果与已查看结果不可混称。

---

[AI-HANDOFF]
task_id: 20260925-rt61-wave-geometry-residual-conditioning
repo: rular099/team_pytorch
branch: rt61-wave-geometry-residual-conditioning
base_commit: 00d624d2fb01c6dd98cac150043dd15b7d8366ff
goal: 一次受控表示旁路实验，争取实际单站空间差分改进与 normal non-input 误差同步下降。
verified_facts:
  - RT60 mechanism 8/14，legacy required 18/25，normal non-input 显著退化。
  - RT60 仅改 readout；同次 RT59 reference 依赖共享 residual_input。
inferences:
  - 冻结表示对波形与几何组合的可读性可能限制 RT60，但尚未被严格证明为唯一瓶颈。
files_to_inspect:
  - 见第 4 节。
constraints:
  - 保持 RT55–RT60 opt-in 兼容；不读 held-out test；不改 sampler/loss/预算；不自动提交有成本训练。
proposed_change:
  - 只新增 rank-16 residual-only wave-geometry adapter，训练它和既有 readout，其余冻结；隔离 h0 reference 与 h1 student。
acceptance_checks:
  - 见第 7–9 节；reference 不漂移；既有 14 gates；实际收益与 legacy GO 分开。
hpc_followup:
  - 仅按授权准备或执行 seed42、8新epoch的一个正式 run，固定最终epoch8；无续训/扫参/test。
risks:
  - 同一 development validation 被多次使用；单seed；软normal约束；少站可观测性；旁路收益不确定。
open_questions:
  - none；执行权限或父权重缺失时完成不依赖它的部分并准确标注，不虚构完成。
[/AI-HANDOFF]

Codex 结束时必须按仓库约定提供 [CODEX-RESULT]：task_id、base_commit、result_commit、branch、changed_files、verification（pass/fail/not run）、compatibility、hpc_status、remaining_risks、review_request。GitHub 推送成功、HPC 提交成功、训练完成、指标通过是四件不同的事，分别报告。
