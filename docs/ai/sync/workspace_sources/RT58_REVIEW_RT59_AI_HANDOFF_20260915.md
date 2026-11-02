# RT58 审阅与唯一下一实验：RT59 Anchor Residual Transport

日期：2026-09-15。状态：审阅与实施提案；本文件不表示已修改仓库、运行训练或提交 Slurm。

## 1. 审阅身份与证据边界

- repo: `rular099/team_pytorch`
- branch: `rt58-waveform-anchor-transfer-field`
- 实际读取 HEAD: `40830acd475bd40347fef03427b52c4a04432dc3`
- HEAD 的直接父提交 / evidence commit: `37d97ad9ad67dfec11f188e2a350a8ab8b8b2056`
- 分析前 base: `0e8537513e623432339d84ae2525b588340b18c9`
- RT58 implementation: `7ff060806609b9c3d0425b939e7c75dd447ba20c`

已通过 GitHub 连接器核对分支、父提交、四个新增文件的变更范围，并读取指定结果、模型、loss、配置、测试和启动器。分析前 base 到 HEAD 仅新增四个结果/交接文件，没有模型、配置、loader 或推理改动。

原始 NPZ、checkpoint 本体、训练曲线不在本轮可直接执行的环境中。因此 NPZ 对齐、1e-12 指标重现和 5,000 次 bootstrap 的完成状态来自已提交证据，不冒称本轮独立执行过。本文独立核对了报告之间的数值一致性、可见代码语义与汇总量算术。

## 2. 审查结论

RT58 **NO-GO**，不替换冻结的 RT57 epoch-6 / gamma=1.66 development baseline。不继续无改动训练当前 RT58，不优先解冻 DiTing。保留 RT58 epoch-8 checkpoint 作为诊断资产，并复用其中已学好的 input-station AnchorHead。

所有以下 PGA 误差均为 `log10(m/s^2)`，point estimate 为 MDN predictive mixture mean。Random validation 包含 9,681 个 realtime rows、1,310 个有有效目标的事件、75,654 个 non-input targets；空间统计对至少 5 个有效目标的 realtime field 等权，不能与 pooled-target 权重混淆。

### 2.1 Scale 与新 head 的贡献

| 指标 | RT57 gamma=1 | 同协议 gamma=1.66 base | RT58 |
|---|---:|---:|---:|
| random MAE | 0.247857 | 0.2455776603 | 0.2460358282 |
| RMSE | 0.320040 | 0.3182428887 | 0.3177236815 |
| bias | 0.036834 | 0.0344664169 | 0.0408062100 |
| R2 | 0.360340 | 0.3675035891 | 0.3695657171 |
| slope | 0.379428 | 0.4003156740 | 0.3972696103 |
| NLL | 0.220260 | 0.217530 | 0.2146628805 |
| Brier, threshold=-1.2 | 0.192239 | 0.189875 | 0.1903842844 |
| mean per-field range ratio | 0.464444 | 0.5102311267 | 0.4922540329 |

注意：gamma=1.66 的 NLL/Brier 在 summary 中注明来自前一轮 RT57 offline analysis；不能说这两个数也在本轮进行了 1e-12 独立重现。

以候选减基线为方向，random MAE 的端点分解为：

```text
scale:         approximately -0.002279 dex
anchor:                     +0.0004581679 dex
total:         approximately -0.001821 dex
```

anchor 增量的 event-cluster 95% CI 为 `[+0.0000527579, +0.0008602687]`。因此，相对 gamma=1 的总体 **MAE** 收益来自 scale，anchor 抵消了部分收益。但不能把这句话扩张成“所有收益都来自 scale”：新 head 仍改善 RMSE、R2、NLL、少量空间排序和 untriggered/1s 结果。

相对同一 gamma=1.66 base，triggered non-input MAE 增加 0.002545 dex；untriggered 减少 0.001209 dex；requested 1s 减少 0.002200 dex。Normal all-target/non-input MAE 分别增加约 0.000594/0.000455 dex，仅满足 retention，不是超越同基线。

聚类 bootstrap 的可见实现按 event_id 合并同事件的重复时刻，baseline/candidate 共用重采样索引；pooled-target MAE 使用事件误差和与计数的比值。这个估计单位合理，但区间不包含训练 seed 波动，也未重做 gamma 的选择过程。不得据此宣称跨 seed、独立 test 或整体研究流程的显著性。

### 2.2 One-station 与瓶颈

1,494 个 actual-selected-station-count=1、且至少 5 个有效 targets 的 fields：

- mean range ratio: 0.2072846126 → 0.1833399933；相对下降约 11.55%。
- pairwise-delta MAE: 0.3361260651 → 0.3367127354。
- Pearson: 0.3224029097 → 0.3239689555。

单站时 station weights 只能为 1，排除了“问题只发生在多站聚合”的解释；没有把 transfer 与 fusion 两者分开，也不消除单台站条件下的固有不确定性。紧凑证据未给上述单站差值的 CI，不应新增“显著退化”的统计断言。

43,118 个 input-station/realtime observations 的 anchor MAE 为 0.116664、相关系数 0.896430；这不是 43,118 个不同物理台站，也不是“1s 时刻足以预测全场”的证明。多站 roll 使 anchor MAE 从 0.114697 升到 0.293783，支持配对波形信息确实被利用。

ungated field MAE 为 0.283641，差于 base 0.245578。Pair-candidate MAE 0.255543 与 ungated field MAE 的统计权重不同，不能仅凭二者差值给 aggregation 定罪。当前优先级应是 **anchor 之后的 transfer/fusion 联合重设计**，而非 encoder 解冻。

代码使用 `gate = tanh(output_gate(...))`，是可正可负的系数，不是 [0,1] 凸融合权重。反推的绝对 gate 均值约 0.106，只能说明最终改动受限，不能证明具体的正向混合机制或优化器已找到合理门控。

### 2.3 Fixed-base waveform-roll 的严格归因

记 `B` 为冻结 gamma=1.66 base，`A` 为 anchor field，`g` 为 signed gate，c/r 分别表示正确/rolled：

```text
D_c = g_c (A_c - B_c)
D_r = g_r (A_r - B_r)
P_c = B_c + D_c
P_r = B_r + D_r
C   = B_c + D_r
```

```latex
D_c=g_c(A_c-B_c),\quad D_r=g_r(A_r-B_r),\quad
P_c=B_c+D_c,\quad P_r=B_r+D_r,\quad C=B_c+D_r.
```

完整 roll 的 MAE penalty 为 0.0142594651 dex；报告的 fixed-output-base correction swap 为 0.0002039003 dex，95% CI `[0.0000486339, 0.0003626900]`。

但 `D_r` 内部仍含 `B_r`，其 event/query representations 也由 rolled 的冻结路径生成。所以 C 是输出层面的 correction-swap 敏感性，不是“冻结全部旧路径内部状态，仅干预新 head”的因果实验。

MAE 可以沿这个指定顺序做精确望远镜分解：

```text
L(P_r)-L(P_c) = [L(C)-L(P_c)] + [L(P_r)-L(C)]
              = 0.0002039003 + 0.0140555648
```

```latex
L(P_r)-L(P_c)=[L(C)-L(P_c)]+[L(P_r)-L(C)].
```

后一项在这条顺序路径中较大，但不是与顺序无关的模块因果贡献；不要把两个 penalty 的比值写成“约 98.6% 来自旧网络”。正确配对对 correction 有小幅条件收益，也不意味着 correction 胜过 delta=0 的 base。

## 3. 十项 go/no 原样审查

| # | 原条件 | observed | 本轮判定 |
|---:|---|---:|---|
| 1 | random MAE <= 0.242 | 0.2460358282 | FAIL |
| 2 | random slope >= 0.45 | 0.3972696103 | FAIL |
| 3 | random R2 >= 0.39 | 0.3695657171 | FAIL |
| 4 | >=5-target mean per-field range ratio >= 0.55 | 0.4922540329 | FAIL |
| 5 | actual-one-station, >=5-target mean range ratio >= 0.25 | 0.1833399933 | FAIL |
| 6 | same one-station population pairwise-delta MAE <= 0.330 | 0.3367127354 | FAIL |
| 7 | requested 1s MAE <= 0.253 | 0.2528910360 | PASS |
| 8 | random NLL <= 0.22 AND calibration not materially degraded | NLL 0.2146628805 | NOT_ESTABLISHED；仅 NLL 子条件通过 |
| 9 | normal all-target MAE <= 0.136 | 0.1316098477 | PASS |
| 10 | normal non-input MAE <= 0.218 | 0.2108261245 | PASS |

严格统计为 **3 PASS + 6 FAIL + 1 NOT_ESTABLISHED**。报告的 4/10 包含了第 8 条 provisional pass，不能记成四项无条件通过。十条是合取门槛，不是多数票。

第 7 条仅有 0.000108964 dex 余量，但原规则是点估计，不事后增加 CI 条件撤销该 PASS。第 8 条没有预设“materially”的数值范围；不能看见 Brier +0.000509 后才补一个容忍量，给 RT58 正式过关。Coverage 升降本身也不是校准好坏的单向判据。

按本轮 gates.csv 所列十条原样判定；本轮已读取的事前实现说明没有列出完整十个数值门槛，故“事前锁定”的原始交接时间戳不属于本轮独立核验到的证据。该溯源限制不改变 NO-GO。

## 4. 唯一下一实验的动机

**RT59 Base-relative Anchor Residual Transport**：固定 RT57 gamma=1.66 与已训练 RT58 AnchorHead，将“整张绝对 anchor field 与 base 之间的带门融合”改成“台站幅度创新项 + 直接受监督的空间差分残差”。

核心问题只有一个：在 anchor 不变的情况下，去掉对整个空间场差值的输出门控，直接学习相对 base 的 station-to-query PGA 差分，能否恢复 one-station 空间幅度并真实改善 non-input 预测？

它是一个联合 transfer/fusion 参数化实验，不承诺分别识别每个组件的独立因果作用。只有一条训练路线、seed=42、固定 8 个新 epoch；不附加超参数网格、额外 seed、旧 RT58 续训或 held-out test。

## 5. 完整 AI-HANDOFF

```text
[AI-HANDOFF]
task_id: 20260915-rt59-anchor-residual-transport
repo: rular099/team_pytorch
branch: rt58-waveform-anchor-transfer-field
base_commit: 40830acd475bd40347fef03427b52c4a04432dc3
evidence_commit: 37d97ad9ad67dfec11f188e2a350a8ab8b8b2056
status: PROPOSAL_NOT_IMPLEMENTED

goal:
  实施并准备唯一 RT59 validation-only 实验：冻结已训练 anchor 和 RT57 gamma=1.66，
  以直接监督的 station-to-query 相对残差替代 RT58 整场门控，检验 non-input 和
  actual-one-station 空间预测是否获得真实增益。代码实施、提交、HPC 执行分开报告。

verified_facts:
  - 指定 HEAD 已核对，直接父提交为 evidence_commit；本轮提交仅新增四个结果文件。
  - evidence README/summary/gates：RT58 相对 gamma=1.66 random MAE +0.0004581679 dex，
    event-cluster 95% CI [0.0000527579,0.0008602687]。
  - actual-one-station mean range ratio 0.2072846126 -> 0.1833399933。
  - input anchor MAE 0.1166643448；ungated query field MAE 0.2836406222。
  - gemini_models.py:PGAAnchorTransferHead 使用 signed tanh gate，delta=g*(field-base)。
  - RT58 epoch8 random/normal/full-roll validation 已完成；不得重提旧实验。
  - 当前轻量证据没有 checkpoint 本体/训练曲线；不得编造 SHA、曲线或执行结果。

inferences:
  - 优先瓶颈位于 anchor 之后的 transfer/fusion；多站 aggregation 不是唯一原因。
  - 当前小 correction 不等价于好的 fusion，也不构成优先解冻 DiTing 的理由。
  - 本方案可能仍失败；anchor 的预测误差和单站不可辨识性仍然存在。

files_to_inspect:
  - AGENTS.md
  - docs/ai/PROJECT_CONTEXT.md
  - docs/ai/README.md
  - SESSION_SUMMARY.md
  - docs/ai/CODEX_RESULT_20260915_RT58_WATF_VALIDATION.md
  - reports/rt58_waveform_anchor_transfer_validation_20260915/{README.md,summary.json,gates.csv}
  - docs/rt58_waveform_anchor_transfer.md
  - gemini_models.py: PGAAnchorTransferHead / FullModel query-readout and MDN shift
  - train_light.py: compatible loading, freeze modes, train mode, loss / normalization
  - gemini_util_light.py: existing causal masks / input label metadata contract; no sampler redesign
  - eval_checkpoint.py: existing temporal/anchor exports and probabilistic metrics
  - RT55/RT56/RT57/RT58 config inheritance and normal-validation companion
  - tools/run_rt58_waveform_anchor_transfer_slurm.sh
  - tools/analyze_random_geometry_full_npz.py
  - tests/test_rt58_waveform_anchor_transfer.py

constraints:
  - 首先核对当前工作树/HEAD；HEAD 漂移则记录差异，不重置或覆盖用户工作。
  - 所有新功能 default-off；RT55/RT56/RT57 和旧 RT58 的 loading、推理、默认值、
    参数键、已有 config 字节及关闭开关时导出 schema 必须保持兼容。
  - 不改 split、采样、cutoff、台站排序、target exclusion、gamma 或单位定义。
  - 仅 causal waveform 和 observable station/query geometry 进入 forward。
    input/query 最终 PGA 标签及真实震源参数只用于允许的 loss/统计，不能作为 forward 输入。
  - 不重复 smoke、query diagnostics、旧正式 validation 或 formal held-out test。
  - 不覆盖旧模型/结果；不自动添加 sweep、extra seeds、无变化续训或 encoder 解冻。

architecture:
  - 新增 PGAAnchorResidualTransportHead，checkpoint prefix:
      pga_anchor_residual_transport_head.
    新开关 use_pga_anchor_residual_transport，默认 false。
  - RT59 config 保留 use_pga_anchor_transfer=true 以构造并加载旧 RT58 state。
    当新开关 true 时执行互斥 RT59 分支，直接复用旧 anchor_head；不调用旧 transfer/
    output_gate 生成最终 correction，不把 RT58 delta 与 RT59 delta 累加。
  - 以下所有 PGA quantities 在 model-normalized coordinate 中计算：
      b_j = frozen RT57 gamma=1.66 mean at query q_j
      b_i = SAME frozen RT57 gamma=1.66 mean at selected input coordinate x_i
      a_i = frozen trained RT58.anchor_head([u_i,d_i,z_e])
      e_i = a_i - b_i
      k_i = tanh(LevelMLP([u_i,d_i,z_e]))
      h_ji = RT58-style PairEncoder([u_i,d_i,z_e,z_new,q_emb_j,g_ji])
      r_ji = tanh(||q_j-x_i||/0.05) * ResidualMLP([h_ji,e_i,b_j-b_i])
      w_ji = masked_softmax(ScoreHead(h_ji), over selected input stations)
      c_ji = b_j + k_i*e_i + r_ji
      delta_j = sum_i w_ji*(k_i*e_i+r_ji)
      final_j = b_j + delta_j
  - k_i 不依赖 query；空间 r_ji 不再乘旧的 final gate。
    LevelMLP: LayerNorm(2*256+1000) -> Linear(256) -> GELU -> Linear(1)。
    ResidualMLP: Linear(256+2,256) -> GELU -> Linear(256,1)。
    两者末层 weight/bias 均置零；不对 r 加 ReLU。
  - hidden_dim=256，set_layers=2，heads=4，geometry 10维与原 RT58 一致。
    新 set_input/set_blocks/pair_encoder/score_head 可从 RT58 对应模块显式复制初始化；
    保持维度/语义一一对应，记录复制清单。旧 transfer_head/output_gate 不复制到新输出层。
  - distance_scale=0.05 沿用当前 model-coordinate units，仅作同坐标归零因子；
    不把它重新解释为公里衰减、不同时改变坐标编码。
  - 单站时 w=1 且：
      final_j-final_k = (b_j-b_k)+(r_ji-r_ki)
    k_i*e_i 在差分中消掉；这只解除结构耦合，不保证训练后范围不会收缩。
  - 不跨 query 集合做均值中心化或归一化；每个 query 的预测不得依赖同批请求了哪些别的点。
  - 给 b_i 实现 opt-in 的 frozen-base 内部 query 解码，复用本次 forward 已有的 station
    memory/event/temporal features；不得为每个输入站重新运行完整 DiTing/FullModel。
    原 public Q queries 按原路径计算；内部 input-coordinate queries 只读解码并隔离临时态。
    不改变已计算的 public base tensors、diagnostics、公开 target masks、loss target 数量或 NPZ 行数。
    不以旧 local_absolute_pred 或 anchor_pred 冒充 b_i。
  - 所有 MDN component means 加同一 delta；logits/component sigmas 不变。
  - 无 valid input 时 correction=0；invalid query correction=0；padding 不可进入 softmax/聚合。

initialization_and_trainability:
  - 从实际 RT58 epoch8 checkpoint 做 weight-only transfer；使用既有 non_encoder_v1
    loading 与固定外部 DiTing 权重机制。RT58_BASE_CHECKPOINT 和 SHA 由实际 HPC 文件解析。
  - strict loading 仅允许新增 pga_anchor_residual_transport_head. 缺失；其他 missing/
    unexpected/shape mismatch 一律失败。复制初始化仅 fresh-transfer 时运行一次；
    resume RT59 checkpoint 时不得覆盖已训练的新 head。
  - 冻结 RT57 全路径和旧 RT58 全路径，含 anchor；旧路径持续 eval mode。
    仅 pga_anchor_residual_transport_head.* 可训练，新 freeze_mode=anchor_residual_transport_only。
    打印每个 trainable tensor、参数总数和 optimizer 覆盖集合；冻结参数没有 optimizer 状态。
  - optimizer/scheduler/epoch/best-loss 状态重置。只继承 RT58 已解析 optimizer/scheduler 算法，
    新 head 初始 lr=5e-4；seed=42；固定 8 个新 epoch；只评估本轮 epoch8 last。
  - 不把新初始化 head 第一步上游梯度为零误当 bug：必须核验末层梯度与后续有效梯度，
    同时保证冻结路径参数与 buffers 不更新。

loss:
  - 保留现有最终 MDN NLL，weight=1。
  - 保留 predictive-mean SmoothL1，weight=0.10，beta=1.0 model units。
  - 新增 relative-transfer SmoothL1，weight=0.20，beta=0.20/PGA_std：
      target_r_ji = (y_query_raw_j-y_input_raw_i)/PGA_std - (b_j-b_i)
      compare r_ji with stopgrad(target_r_ji)
    input PGA 只出现在本 loss 的 target 构造中；差分不再减 normalization mean。
  - candidate SmoothL1，weight=0.05，beta=0.20/PGA_std；c_ji 对 normalized y_query。
  - final within-event query-difference SmoothL1，weight=0.40，beta=0.15/PGA_std，max_pairs=105。
  - normal-only replay distillation，weight=0.05；保持旧 RT58 现有 beta=0.15 model units
    与 apply_to_random=false 的语义，不顺带改为 0.15 dex。
  - 关闭旧 anchor_absolute loss（anchor 已冻结）、旧 temporal residual/local auxiliary losses。
  - PGA_std=0.4312468402225416，以 train split 的既有 normalization/resolved config 为准。
    0.15 dex -> 0.3478286355 model units；0.20 dex -> 0.4637715140 model units。
  - 使用既有 F.smooth_l1_loss 语义，权重只乘一次；不要按 Huber 名称直接替换为尺度不同的 loss。
  - pair mask 同时要求 selected input valid、finite input PGA、valid query、finite query label；
    沿用原有效 pair/target reduction，不额外重采样或重新加权单站/时间桶。
  - 新辅助张量保留正确 DDP graph；不能 detach r/c，也不能污染旧 output_layout。

config:
  - 新增 pga_configs/transformer_japan_full_2000_2024_rt59_anchor_residual_transport_seed42_chaosuan.json
    及对应 _normal_validation_chaosuan.json；继承而不修改旧 config。
  - train 80% causal random +20% normal replay；requested station counts [1,3,5,8,12,16]。
  - target sampling、label/mask protocol、训练 realtime bins、validation seed/cutoff 全部沿用 RT58。
  - random validation 100% deterministic random；normal companion 关闭 random mask。
  - weight_path=${RT59_WEIGHT_PATH}，load_model_path=${RT58_BASE_CHECKPOINT}，epochs_full_model=8。

proposed_change:
  - gemini_models.py: 新 head、内部 frozen query readout、互斥 dispatch、独立诊断缓存；默认路径不动。
  - train_light.py: 新 freeze/train mode、一次性初始化、relative-transfer loss 与 config validation。
  - eval_checkpoint.py: RT59 独立 conditional exports 与同一 random job 内的新路径负对照。
  - 新 RT59 两个 config、tests/test_rt59_anchor_residual_transport.py、
    tools/run_rt59_anchor_residual_transport_slurm.sh、可复用 RT59 offline analyzer。
  - docs/rt59_anchor_residual_transport.md 与本 task 的 CODEX_RESULT；
    轻量结果放新 reports/rt59_anchor_residual_transport_validation_<date>/。
  - 旧 RT58 判定的机器可读摘要可作纯文字/状态修正：3 PASS/6 FAIL/1 NOT_ESTABLISHED；
    不重算或修改旧原始数字，不把修正文案变成另一项实验。

acceptance_checks:
  - 新开关 false：RT55/RT56/RT57/RT58 原 config、state keys、数值推理与导出 schema 不变。
  - 新开关 true 且 zero init：final == 本次 frozen gamma=1.66 base；logits/sigmas 完全不变。
  - 单站差分恒等式；相同坐标 r=0；全站联合 permutation invariance；invalid/empty mask 安全。
  - 内部 input-coordinate query 不改变 public base；public query 顺序/分批/附加其他 query
    不改变同一 query 的预测（仅容许明确记录的浮点舍入），不能要求重跑 query diagnostics。
  - 改动 label metadata 不改变 forward；loss 的 finite/padding masks、normalization、beta、梯度正确。
  - 唯一 trainable prefix、冻结参数/buffers、train/eval modes、严格 loading 和 resume 不重初始化。
  - 新 exports 可按 base+level+relative 重建 final；绝对量和差分量的 raw-dex 反归一化正确。
  - python -m py_compile gemini_models.py train_light.py eval_checkpoint.py <new analyzer>
  - bash -n tools/run_rt59_anchor_residual_transport_slurm.sh
  - PYTHONPATH=. python -m pytest -q tests/test_rt59_anchor_residual_transport.py
      tests/test_rt58_waveform_anchor_transfer.py tests/test_rt57_gtnp_v2_station_distinctive.py
      tests/test_causal_random_geometry.py tests/test_eval_checkpoint_formal.py
  - git diff --check；新 launcher 用 mock submission 验证依赖图，不运行 HPC smoke。

validation_and_controls:
  - 只做本轮固定 epoch8 的 random 与 normal 两个 primary validation jobs。
  - 同一 candidate random job 内可利用缓存特征做一次新路径配对负对照：
    固定 correct b_j、b_i、event/query 表示、coordinates/masks；仅将 valid selected stations
    的 u_i/d_i 一起 roll，重新计算冻结 anchor 与 RT59 head；原 RT57 路径不重算。
    这是 head-boundary feature-pairing intervention，不命名为旧 full-system waveform-roll。
    actual one-station 时恒等；multi-station 单列；不添加独立诊断矩阵或旧 roll job。
  - 导出 val_rt59_*: input_base_mean、anchor_pred、anchor_innovation、level_coefficient、
    level_correction、relative_transfer、station_weights、candidate、applied_delta、
    fixed_context_rolled_delta；保留公共 alignment 与 gamma=1.66 base 字段。
  - main comparator 必须为同批 forward/same NPZ 的冻结 gamma=1.66 base；gamma=1 仅历史参考。
  - 报告 point MAE/RMSE/bias/slope/R2，NLL/Brier/sigma/coverage，实际站数、requested times、
    triggered/untriggered 分桶、>=5-target fields 和 actual-one-station fields。
  - 固定 event_id paired percentile bootstrap=5000，seed=20260915；delta=CANDIDATE-BASE。
    逐项记录 target/field/event 数。field mean ratio 不能用 mean(pred range)/mean(true range) 替代。
  - baseline NLL/Brier 从本轮完整 MDN 参数/概率精确计算，不沿用历史四舍五入值。
    固定上下文 roll 的损失变化是负对照，不等价于相对 delta=0 的实际净增益。

predeclared_go_no_for_RT59:
  - 在新训练开始前把规则写入版本控制；不反向套用为 RT58 第8条的通过证明。
  - 原十条的 #1-7/#9-10 数值与定义不变：
      random MAE<=0.242; slope>=0.45; R2>=0.39;
      >=5-target mean range ratio>=0.55;
      actual-one-station >=5-target mean range ratio>=0.25;
      same one-station pairwise-delta MAE<=0.330;
      requested 1s MAE<=0.253;
      normal all-target MAE<=0.136; normal non-input MAE<=0.218.
  - 仅对新 RT59 前瞻性量化 #8：NLL<=0.22；paired ΔBrier 的95%CI上界<=0；
    与同协议 base 的 coverage1/coverage2 绝对差各<=0.01。
    这里 0.01 是本提案预先声明的 retention 预算，不是通用校准定律；
    不把 mean±sigma coverage 当成任意 Gaussian mixture 的精确 nominal coverage。
  - 加入实际净增益护栏：random ΔMAE 的95%CI上界<0；
    actual-one-station range absolute error 点估计不高于同协议 base。
  - 所有上述必要条件同时通过才记 GO_FOR_DEVELOPMENT；否则 NO_GO。
    部分成功必须按原名称报告，不升级为整体通过，不据失败后追加训练或换 threshold。
  - 即使通过，也只是单 seed / 同一 development validation 上的进展，不是独立泛化证明。

hpc_followup:
  - 新 launcher 默认 DRY_RUN=1；ACTION=train|eval|all；真实提交需要 CONFIRM_RT59=1。
  - 源码身份支持精确 clean Git commit 或手动上传文件夹 uploaded_sha256；
    manifest 包含新代码、共享 helper、全部继承 config、训练/评估脚本。
  - 启动前核验实际 RT58 checkpoint SHA；compute-node guard 核验初始化 checkpoint epoch=8；
    评估 guard 核验本轮 task_id 与新训练完成 epoch=8，不能因都叫 epoch8 而混用源 checkpoint。
  - 一项 train（固定8个新epoch）-> random validation + normal validation，afterok 依赖；
    新路径负对照复用 random job 缓存。无 held-out test action。
  - 沿用已工作的资源/环境参数，所有站点路径由 WORKDIR、JAPAN_FULL_DATA_ROOT、
    RT58_BASE_CHECKPOINT、RT58_BASE_CHECKPOINT_SHA256、RT59_WEIGHT_PATH 等提供。
    不硬编码新的集群路径，不覆盖旧输出目录；无实际 checkpoint 时明确 dependency-blocked，
    不静默重训 anchor 或改用别的 epoch。
  - 保存 resolved config、完整命令、初始化/最终 checkpoint 与 encoder/source/split hashes、
    trainable tensor manifest、每个 epoch 的各项 loss/lr/validation scalar、Slurm job/status、
    epoch8 checkpoint provenance、random/normal metrics+NPZ 和本轮门槛判定。
  - raw NPZ/weights 留本地或HPC；Git 提交轻量 README、summary.json、gates.csv、CSV/分析脚本。

risks:
  - 训练过的 anchor 仍有误差；e_i 可能包含噪声，直接 contrast residual 也可能不可辨识。
  - 单站幅度-offset 与空间 residual 解耦不保证空间预测改善。
  - 联合参数化/loss 与 warm-start 后再训8epoch 是一个方案实验，不能隔离每个改动的独立贡献。
  - 内部 b_i readout 的缓存污染/query-set 依赖及额外解码开销，必须由新代码局部测试约束。
  - 固定 MDN sigmas/logits 对不确定性适配有限；不可用 mean 指标遮蔽 Brier/NLL。
  - 单 seed、重复使用 development validation；不能作 test/generalization 声明。

open_questions:
  - none requiring a new scientific choice；实际路径/hash按已有HPC环境解析并记录。

required_return:
  - 按 docs/ai/README.md 返回 CODEX-RESULT，含 task_id/base_commit/result_commit/branch/
    changed_files/verification/compatibility/hpc_status/remaining_risks/review_request。
  - implementation/local tests/submission/HPC completion 分项写 pass/fail/not run 与真实理由。
  - 不把本 handoff 当成已实施结果；未获用户执行授权时只完成代码和 DRY_RUN 交接。
[/AI-HANDOFF]
```

### 公式的原始 LaTeX（与交接中的 normalized-coordinate 定义一致）

```latex
\begin{aligned}
b_j&=f_{57,\gamma=1.66}(q_j),\qquad
b_i=f_{57,\gamma=1.66}(x_i),\qquad e_i=a_i-b_i,\\
k_i&=\tanh K(u_i,d_i,z_e),\\
r_{ji}&=\tanh\!\left(\frac{\|q_j-x_i\|}{0.05}\right)
R(h_{ji},e_i,b_j-b_i),\\
m_j^{59}&=b_j+\sum_iw_{ji}(k_i e_i+r_{ji}),\\
r_{ji}^{*}&=\frac{y_j^{raw}-y_i^{raw}}{\sigma_{PGA}}-(b_j-b_i),\\
N_{input}=1&\;\Longrightarrow\;
m_j^{59}-m_k^{59}=(b_j-b_k)+(r_{ji}-r_{ki}).
\end{aligned}
```

## 6. 固定提交下的来源索引

本文件所有 repository-relative 路径均对应 HEAD `40830acd475bd40347fef03427b52c4a04432dc3`。

- `docs/ai/CODEX_RESULT_20260915_RT58_WATF_VALIDATION.md`
- `reports/rt58_waveform_anchor_transfer_validation_20260915/README.md`，尤其 §4–9 与 §12。
- 同目录 `summary.json` 与 `gates.csv`。
- `gemini_models.py`，`PGAAnchorTransferHead`、`FullModel.forward` 的 temporal/anchor dispatch。
- `train_light.py`，`pga_anchor_pair_candidate_aux_loss`、`normal_replay_distillation_aux_loss`、`pga_within_event_difference_aux_loss`。
- `tools/analyze_random_geometry_full_npz.py`，`_event_aggregates`、`_bootstrap_ratio_difference`。
- RT58 主/normal config、`tests/test_rt58_waveform_anchor_transfer.py`、RT58 Slurm launcher。
- `docs/ai/CODEX_RESULT_20260912_RT57_GTNP_V2_VALIDATION.md`，gamma=1.66 的 validation-only post-hoc 来源。
- `AGENTS.md`、`docs/ai/README.md`，兼容与交接约定。

上述来源为审阅证据。RT59 架构、loss 配比与新增前瞻性门槛是本文件提出的方案，并非这些来源已经实施或证实的结果。
