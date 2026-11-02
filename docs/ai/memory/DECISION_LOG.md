# DECISION_LOG

<!-- ai-memory:begin -->
更新时间：2026-11-02T08:55:53+00:00

生成所依据的协调分支 commit：`328550fa7ea2164739009c0cd61c6a2fd70830e6`

协调分支：`chore/workspace-sync`

此 SHA 是生成输入，不是整个科研项目的单一 HEAD；各线见下表。

## RT55_RT56

### D-RT-001 · 2026-09-02

- 实验线：RT55_RT56
- 背景：RT56迁移必须保留RT55行为与严格非输入query定义。
- 讨论选项：原来源未完整保存候选讨论；不补造历史理由。
- 决定：weight-only初始化；50%原任务/50%因果随机几何，validation固定random seed，query排除input。
- 适用范围：RT56与后续兼容路径
- 后续推翻：未见记录
- 历史依据：needs_history_review
- 来源：[docs/ai/PROJECT_CONTEXT.md](https://github.com/rular099/team_pytorch/blob/8bde65cb64fd860adf552dfe56505b4b31a78b13/docs/ai/PROJECT_CONTEXT.md)

## RT57

### D-RT57-001 · 2026-09-12

- 实验线：RT57
- 背景：gamma的证据等级。
- 讨论选项：原来源未完整保存候选讨论；不补造历史理由。
- 决定：gamma=1.66按同一validation选择，不称为test或已验证泛化。
- 适用范围：RT57 residual scaling
- 后续推翻：未见记录
- 历史依据：needs_history_review
- 来源：[docs/ai/CODEX_RESULT_20260912_RT57_GTNP_V2_VALIDATION.md](https://github.com/rular099/team_pytorch/blob/143b63af5d0a477ac388d310dc470eb7349b5c90/docs/ai/CODEX_RESULT_20260912_RT57_GTNP_V2_VALIDATION.md)

## RT58

### D-RT58-001 · 2026-09-15

- 实验线：RT58
- 背景：如何隔离已知scale收益和新增anchor收益。
- 讨论选项：原来源未完整保存候选讨论；不补造历史理由。
- 决定：以相同gamma=1.66 frozen base比较新增anchor，并保留原十项门。
- 适用范围：RT58
- 后续推翻：未见记录
- 历史依据：needs_history_review
- 来源：[docs/ai/CODEX_RESULT_20260915_RT58_WATF_VALIDATION.md](https://github.com/rular099/team_pytorch/blob/40830acd475bd40347fef03427b52c4a04432dc3/docs/ai/CODEX_RESULT_20260915_RT58_WATF_VALIDATION.md)；[pga_configs/transformer_japan_full_2000_2024_rt58_waveform_anchor_transfer_seed42_chaosuan.json](https://github.com/rular099/team_pytorch/blob/40830acd475bd40347fef03427b52c4a04432dc3/pga_configs/transformer_japan_full_2000_2024_rt58_waveform_anchor_transfer_seed42_chaosuan.json)

## RT59

### D-RT59-001 · 2026-09-20

- 实验线：RT59
- 背景：点指标进展与空间场成功是否相同。
- 讨论选项：按原联合门判定；或因部分指标好转而改写成功标准。来源选择前者。
- 决定：保留RT59 epoch8作开发parent；维持原29项联合判定NO_GO，不改变阈值。
- 适用范围：RT59–RT61 reference
- 后续推翻：未见记录
- 历史依据：source_recorded
- 来源：[docs/ai/CODEX_RESULT_20260920_RT59_DUAL_OBJECTIVE_TRANSPORT_V3_VALIDATION.md](https://github.com/rular099/team_pytorch/blob/d43c7d63528b3ade71c3506eeb2cb71f90fb3ffd/docs/ai/CODEX_RESULT_20260920_RT59_DUAL_OBJECTIVE_TRANSPORT_V3_VALIDATION.md)；[reports/rt59_dual_objective_transport_v3_validation_20260920/gates.csv](https://github.com/rular099/team_pytorch/blob/d43c7d63528b3ade71c3506eeb2cb71f90fb3ffd/reports/rt59_dual_objective_transport_v3_validation_20260920/gates.csv)

## RT60

### D-RT60-001 · 2026-09-23

- 实验线：RT60
- 背景：空间弱改善但normal远程误差变差。
- 讨论选项：保留RT59并归档部分机制/负结果；或自动扩大当前训练。预注册行动为前者。
- 决定：保留RT59，不自动续训RT60。
- 适用范围：RT60收尾
- 后续推翻：未见记录
- 历史依据：source_recorded
- 来源：[docs/ai/CODEX_RESULT_20260923_RT60_CONTRAST_READOUT_VALIDATION.md](https://github.com/rular099/team_pytorch/blob/00d624d2fb01c6dd98cac150043dd15b7d8366ff/docs/ai/CODEX_RESULT_20260923_RT60_CONTRAST_READOUT_VALIDATION.md)；[reports/rt60_contrast_readout_validation_20260923/summary.json](https://github.com/rular099/team_pytorch/blob/00d624d2fb01c6dd98cac150043dd15b7d8366ff/reports/rt60_contrast_readout_validation_20260923/summary.json)

## RT61

### D-RT61-001 · 2026-09-28

- 实验线：RT61
- 背景：微小空间收益是否达到事先要求。
- 讨论选项：原来源未完整保存候选讨论；不补造历史理由。
- 决定：保留RT59；不得把0.385% pairwise改善称为达到2%预注册阈值。
- 适用范围：RT61与后续解释
- 后续推翻：未见记录
- 历史依据：needs_history_review
- 来源：[docs/ai/CODEX_RESULT_20260928_RT61_WAVE_GEOMETRY_VALIDATION.md](https://github.com/rular099/team_pytorch/blob/9c95dbfaf92f36b2673d816026cd3f25b91eec66/docs/ai/CODEX_RESULT_20260928_RT61_WAVE_GEOMETRY_VALIDATION.md)；[reports/rt61_wave_geometry_validation_20260928/summary.json](https://github.com/rular099/team_pytorch/blob/9c95dbfaf92f36b2673d816026cd3f25b91eec66/reports/rt61_wave_geometry_validation_20260928/summary.json)

## FE01

### D-FE01-001 · 2026-10-06

- 实验线：FE01
- 背景：跨模型比较与实时/场地解释的边界。
- 讨论选项：原来源未完整保存候选讨论；不补造历史理由。
- 决定：使用共同train-only参照；回顾性catalog参照不作为实时竞争模型；不从站点曝光推出站点泛化。
- 适用范围：FE01及后续评价
- 后续推翻：未见记录
- 历史依据：needs_history_review
- 来源：[reports/fe01_eval1_hpc_gates_20261006/summary.json](https://github.com/rular099/team_pytorch/blob/ba4fa7740d9fde5fde87a7b0ae0397037209baf4/reports/fe01_eval1_hpc_gates_20261006/summary.json)；[docs/ai/FE01_PROTOCOL.md](https://github.com/rular099/team_pytorch/blob/ba4fa7740d9fde5fde87a7b0ae0397037209baf4/docs/ai/FE01_PROTOCOL.md)

## FE01_A01

### D-FE01-A01-001 · 2026-10-21

- 实验线：FE01_A01
- 背景：如何解释幅值ON/OFF和诊断。
- 讨论选项：原来源未完整保存候选讨论；不补造历史理由。
- 决定：报告保持中立事实/来源；独立判断与真实训练结果、推理干预、未训练初态严格分开。
- 适用范围：FE01-A01审阅与后续ON评价
- 后续推翻：未见记录
- 历史依据：needs_history_review
- 来源：[reports/fe01_a01_hpc_results_20261021/README.md](https://github.com/rular099/team_pytorch/blob/29ffb20ba0effc0c345a0329ba3f8c658dc8cef0/reports/fe01_a01_hpc_results_20261021/README.md)；[reports/fe01_a01_hpc_results_20261021/CHATGPT_ANALYSIS_PROMPT.md](https://github.com/rular099/team_pytorch/blob/29ffb20ba0effc0c345a0329ba3f8c658dc8cef0/reports/fe01_a01_hpc_results_20261021/CHATGPT_ANALYSIS_PROMPT.md)

## FE01_ON

### D-FE01-ON-001 · 2026-10-25

- 实验线：FE01_ON
- 背景：inventory路径搬迁破坏清单身份。
- 讨论选项：原来源未完整保存候选讨论；不补造历史理由。
- 决定：保留认证manifest文件名/路径与SHA；新run恢复inventory并导入已seal的CPU结果，原run不改。
- 适用范围：FE01 ON V4 inventory恢复
- 后续推翻：未见记录
- 历史依据：needs_history_review
- 来源：[docs/ai/FE01_ON_EVAL_INVENTORY_FIX.md](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/docs/ai/FE01_ON_EVAL_INVENTORY_FIX.md)

### D-FE01-ON-002 · 2026-10-22

- 实验线：FE01_ON
- 背景：固定query回放与旧snapshot人口不同。
- 讨论选项：原来源未完整保存候选讨论；不补造历史理由。
- 决定：fixed_query_replay_v4独立人口/hash；TTUM/shadow共用交付时钟，不从MAE推导警报收益。
- 适用范围：ON V4评价
- 后续推翻：未见记录
- 历史依据：needs_history_review
- 来源：[docs/ai/FE01_ON_EVAL_PROTOCOL.md](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/docs/ai/FE01_ON_EVAL_PROTOCOL.md)

## FE02

### D-FE02-001 · 2026-10-08

- 实验线：FE02
- 背景：事件摘要以什么路径进入PGA？
- 讨论选项：原来源未完整保存候选讨论；不补造历史理由。
- 决定：固定R0/A/B/C/M五组；C非严格等容量，M只增加key不增加独立观测信息。
- 适用范围：FE02矩阵
- 后续推翻：未见记录
- 历史依据：needs_history_review
- 来源：[docs/ai/FE02_PROTOCOL.md](https://github.com/rular099/team_pytorch/blob/42b3d5405d9fe9672f0a0925275ca4f7e9e6a9e7/docs/ai/FE02_PROTOCOL.md)

### D-FE02-002 · 2026-10-14

- 实验线：FE02
- 背景：直接改受锁engine会破坏已有resume身份。
- 讨论选项：原来源未完整保存候选讨论；不补造历史理由。
- 决定：通过opt-in独立hash-pinned兼容层保持RNG字节/顺序，并保留原lock与last-only恢复。
- 适用范围：已知9f70371来源任务
- 后续推翻：未见记录
- 历史依据：needs_history_review
- 来源：[docs/ai/CODEX_RESULT_20261014_FE02_RNG_RESUME.md](https://github.com/rular099/team_pytorch/blob/42b3d5405d9fe9672f0a0925275ca4f7e9e6a9e7/docs/ai/CODEX_RESULT_20261014_FE02_RNG_RESUME.md)

## V01

### D-V01-001 · 2026-10-06

- 实验线：V01
- 背景：旧AA重复替代请求与缺失人口。
- 讨论选项：原来源未完整保存候选讨论；不补造历史理由。
- 决定：严格一对一sensor/UTC匹配，缺失请求计入弃权，不以邻居替代。
- 适用范围：V01 closure；历史旧AA结果被该范围替代
- 后续推翻：未见记录
- 历史依据：needs_history_review
- 来源：[reports/v01_validation_closure_20261006/RESULT_REVIEW.md](https://github.com/rular099/team_pytorch/blob/d8f2436273fbe990e77f7151d86a6b629e99c5c1/reports/v01_validation_closure_20261006/RESULT_REVIEW.md)；[reports/v01_validation_closure_20261006/verification.json](https://github.com/rular099/team_pytorch/blob/d8f2436273fbe990e77f7151d86a6b629e99c5c1/reports/v01_validation_closure_20261006/verification.json)

### D-V01-002 · 2026-10-06

- 实验线：V01
- 背景：工程闭环是否足以证明padding机制。
- 讨论选项：原来源未完整保存候选讨论；不补造历史理由。
- 决定：仅报告当前干预/预算下的条件性结果，保留高PGA trade-off，不推断速度上限/正则化因果机制。
- 适用范围：V01科学审阅
- 后续推翻：未见记录
- 历史依据：needs_history_review
- 来源：[docs/ai/CHATGPT_REVIEW_REQUEST_20261006_V01_CLOSURE.md](https://github.com/rular099/team_pytorch/blob/d8f2436273fbe990e77f7151d86a6b629e99c5c1/docs/ai/CHATGPT_REVIEW_REQUEST_20261006_V01_CLOSURE.md)；[reports/v01_validation_closure_20261006/headline_metrics.csv](https://github.com/rular099/team_pytorch/blob/d8f2436273fbe990e77f7151d86a6b629e99c5c1/reports/v01_validation_closure_20261006/headline_metrics.csv)

## LEGACY

未登记；不代表没有历史决策。
<!-- ai-memory:end -->

<!-- 人工补充写在生成区外；record.json 是结构化事实源。 -->
