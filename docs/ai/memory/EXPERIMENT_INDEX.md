# EXPERIMENT_INDEX

<!-- ai-memory:begin -->
更新时间：2026-11-01T07:35:39+00:00

生成所依据的协调分支 commit：`ba4fa7740d9fde5fde87a7b0ae0397037209baf4`

协调分支：`chore/project-memory-bootstrap`

此 SHA 是生成输入，不是整个科研项目的单一 HEAD；各线见下表。

## RT55_RT56

### RT55

- 科学问题：匹配波形是否影响最终系统？
- branch：`query-geometry-diagnostics`
- commit：`e74f977271b0bcde42c7ac7a18cb77ac60107e34`
- 状态：`verified_complete`
- 数据/协议：固定split；epoch20 test与epoch32 validation分别报告，MDN mixture mean。
- 结果性质：系统依赖波形证据；空间外推能力未认证。
- 审阅：needs_independent_review
- 关系：RT56–RT61兼容基线
- 原始入口：[docs/ai/PROJECT_CONTEXT.md](https://github.com/rular099/team_pytorch/blob/8bde65cb64fd860adf552dfe56505b4b31a78b13/docs/ai/PROJECT_CONTEXT.md)

### RT56

- 科学问题：随机非输入查询训练是否改善空间任务？
- branch：`query-geometry-diagnostics`
- commit：`e74f977271b0bcde42c7ac7a18cb77ac60107e34`
- 状态：`verified_complete`
- 数据/协议：epoch6 random/normal validation，严格逐目标配对。
- 结果性质：随机任务收益与normal遗忘同时存在。
- 审阅：needs_independent_review
- 关系：RT55迁移
- 原始入口：[docs/ai/PROJECT_CONTEXT.md](https://github.com/rular099/team_pytorch/blob/8bde65cb64fd860adf552dfe56505b4b31a78b13/docs/ai/PROJECT_CONTEXT.md)；[docs/ai/CODEX_RESULT_20260907_random_geometry_full_analysis.md](https://github.com/rular099/team_pytorch/blob/e74f977271b0bcde42c7ac7a18cb77ac60107e34/docs/ai/CODEX_RESULT_20260907_random_geometry_full_analysis.md)

## RT57

### RT57-GTNP-V2

- 科学问题：空间输出退化发生在哪条信息路径？
- branch：`rt57-geo-temporal-neural-process`
- commit：`143b63af5d0a477ac388d310dc470eb7349b5c90`
- 状态：`verified_complete`
- 数据/协议：Japan KNET，固定event-disjoint split；random 1310events/75654targets，normal 1383events/89770targets；seed42；log10(m/s²)。固定epoch6。
- 结果性质：开发验证与离线scale分析。
- 审阅：needs_independent_review
- 关系：RT56后续；RT58的parent
- 原始入口：[docs/ai/CODEX_RESULT_20260912_RT57_GTNP_V2_VALIDATION.md](https://github.com/rular099/team_pytorch/blob/143b63af5d0a477ac388d310dc470eb7349b5c90/docs/ai/CODEX_RESULT_20260912_RT57_GTNP_V2_VALIDATION.md)

## RT58

### RT58-WATF

- 科学问题：局部幅值anchor能否改善远程空间场？
- branch：`rt58-waveform-anchor-transfer-field`
- commit：`40830acd475bd40347fef03427b52c4a04432dc3`
- 状态：`verified_complete`
- 数据/协议：Japan KNET，固定event-disjoint split；random 1310events/75654targets，normal 1383events/89770targets；seed42；log10(m/s²)。固定epoch8；5000 event-cluster draws。
- 结果性质：新anchor相对同gamma对照未取得所需空间收益。
- 审阅：needs_independent_review
- 关系：RT57(gamma1.66)→RT58；RT59重新设计
- 原始入口：[docs/ai/CODEX_RESULT_20260915_RT58_WATF_VALIDATION.md](https://github.com/rular099/team_pytorch/blob/40830acd475bd40347fef03427b52c4a04432dc3/docs/ai/CODEX_RESULT_20260915_RT58_WATF_VALIDATION.md)；[reports/rt58_waveform_anchor_transfer_validation_20260915/summary.json](https://github.com/rular099/team_pytorch/blob/40830acd475bd40347fef03427b52c4a04432dc3/reports/rt58_waveform_anchor_transfer_validation_20260915/summary.json)；[pga_configs/transformer_japan_full_2000_2024_rt58_waveform_anchor_transfer_seed42_chaosuan.json](https://github.com/rular099/team_pytorch/blob/40830acd475bd40347fef03427b52c4a04432dc3/pga_configs/transformer_japan_full_2000_2024_rt58_waveform_anchor_transfer_seed42_chaosuan.json)

## RT59

### RT59-V3

- 科学问题：dual-objective transport能否联合改善随机/正常任务和空间场？
- branch：`rt59-dual-objective-transport-v3`
- commit：`d43c7d63528b3ade71c3506eeb2cb71f90fb3ffd`
- 状态：`verified_complete`
- 数据/协议：Japan KNET，固定event-disjoint split；random 1310events/75654targets，normal 1383events/89770targets；seed42；log10(m/s²)。固定epoch8；5000配对事件簇bootstrap。
- 结果性质：点误差进展与空间场NO_GO并存。
- 审阅：needs_independent_review
- 关系：RT58后重新设计；RT60/61 reference
- 原始入口：[docs/ai/CODEX_RESULT_20260920_RT59_DUAL_OBJECTIVE_TRANSPORT_V3_VALIDATION.md](https://github.com/rular099/team_pytorch/blob/d43c7d63528b3ade71c3506eeb2cb71f90fb3ffd/docs/ai/CODEX_RESULT_20260920_RT59_DUAL_OBJECTIVE_TRANSPORT_V3_VALIDATION.md)；[reports/rt59_dual_objective_transport_v3_validation_20260920/gates.csv](https://github.com/rular099/team_pytorch/blob/d43c7d63528b3ade71c3506eeb2cb71f90fb3ffd/reports/rt59_dual_objective_transport_v3_validation_20260920/gates.csv)；[pga_configs/transformer_japan_full_2000_2024_rt59_dual_objective_transport_v3_seed42_chaosuan.json](https://github.com/rular099/team_pytorch/blob/d43c7d63528b3ade71c3506eeb2cb71f90fb3ffd/pga_configs/transformer_japan_full_2000_2024_rt59_dual_objective_transport_v3_seed42_chaosuan.json)

## RT60

### RT60-CONTRAST

- 科学问题：仅final readout能否恢复空间contrast并保护normal？
- branch：`rt60-final-contrast-readout`
- commit：`00d624d2fb01c6dd98cac150043dd15b7d8366ff`
- 状态：`verified_complete`
- 数据/协议：Japan KNET，固定event-disjoint split；random 1310events/75654targets，normal 1383events/89770targets；seed42；log10(m/s²)。固定epoch8。
- 结果性质：mechanism NO_GO；normal退化。
- 审阅：needs_independent_review
- 关系：RT59→RT60；RT61继续检验另一干预
- 原始入口：[docs/ai/CODEX_RESULT_20260923_RT60_CONTRAST_READOUT_VALIDATION.md](https://github.com/rular099/team_pytorch/blob/00d624d2fb01c6dd98cac150043dd15b7d8366ff/docs/ai/CODEX_RESULT_20260923_RT60_CONTRAST_READOUT_VALIDATION.md)；[reports/rt60_contrast_readout_validation_20260923/summary.json](https://github.com/rular099/team_pytorch/blob/00d624d2fb01c6dd98cac150043dd15b7d8366ff/reports/rt60_contrast_readout_validation_20260923/summary.json)

## RT61

### RT61-WAVE-GEOMETRY

- 科学问题：wave/geometry小残差能否保护normal并恢复单输入空间场？
- branch：`rt61-wave-geometry-residual-conditioning`
- commit：`9c95dbfaf92f36b2673d816026cd3f25b91eec66`
- 状态：`verified_complete`
- 数据/协议：Japan KNET，固定event-disjoint split；random 1310events/75654targets，normal 1383events/89770targets；seed42；log10(m/s²)。固定epoch8，99621个可训练scalars。
- 结果性质：机制部分通过；有用联合进展失败。
- 审阅：needs_independent_review
- 关系：RT59 parent；adapter-off仅反事实
- 原始入口：[docs/ai/CODEX_RESULT_20260928_RT61_WAVE_GEOMETRY_VALIDATION.md](https://github.com/rular099/team_pytorch/blob/9c95dbfaf92f36b2673d816026cd3f25b91eec66/docs/ai/CODEX_RESULT_20260928_RT61_WAVE_GEOMETRY_VALIDATION.md)；[reports/rt61_wave_geometry_validation_20260928/summary.json](https://github.com/rular099/team_pytorch/blob/9c95dbfaf92f36b2673d816026cd3f25b91eec66/reports/rt61_wave_geometry_validation_20260928/summary.json)；[pga_configs/transformer_japan_full_2000_2024_rt61_wave_geometry_residual_seed42_chaosuan.json](https://github.com/rular099/team_pytorch/blob/9c95dbfaf92f36b2673d816026cd3f25b91eec66/pga_configs/transformer_japan_full_2000_2024_rt61_wave_geometry_residual_seed42_chaosuan.json)

## FE01

### FE01-ENCODER

- 科学问题：不同冻结前端在共同下游下表现如何？
- branch：`exp/fe01-feature-extractor-site-effects`
- commit：`ba4fa7740d9fde5fde87a7b0ae0397037209baf4`
- 状态：`verified_complete`
- 数据/协议：val 1310events/139440 noninput请求/十cell等权；3家族×seed42/43/44；选轮来自validation。
- 结果性质：九组12轮验证比较，非test。
- 审阅：needs_independent_review
- 关系：A01显式幅值研究的来源
- 原始入口：[reports/fe01_hpc_results_20261004/README.md](https://github.com/rular099/team_pytorch/blob/ba4fa7740d9fde5fde87a7b0ae0397037209baf4/reports/fe01_hpc_results_20261004/README.md)；[docs/ai/FE01_PROTOCOL.md](https://github.com/rular099/team_pytorch/blob/ba4fa7740d9fde5fde87a7b0ae0397037209baf4/docs/ai/FE01_PROTOCOL.md)；[configs/fe01/formal/formal__phasenet_pretrained_frozen__seed42.json](https://github.com/rular099/team_pytorch/blob/ba4fa7740d9fde5fde87a7b0ae0397037209baf4/configs/fe01/formal/formal__phasenet_pretrained_frozen__seed42.json)

### FE01-EVAL1

- 科学问题：既有系统能否通过真实身份与包装器前向门？
- branch：`exp/fe01-feature-extractor-site-effects`
- commit：`ba4fa7740d9fde5fde87a7b0ae0397037209baf4`
- 状态：`in_progress`
- 数据/协议：11组identity，8组各32 decisions/544 targets forward通过；3 TEAM未前向。
- 结果性质：身份/等价核验与train-only参照，非新完整性能表。
- 审阅：needs_independent_review
- 关系：连接既有RT55/61，未合并比较排名
- 原始入口：[reports/fe01_eval1_hpc_gates_20261006/summary.json](https://github.com/rular099/team_pytorch/blob/ba4fa7740d9fde5fde87a7b0ae0397037209baf4/reports/fe01_eval1_hpc_gates_20261006/summary.json)

## FE01_A01

### FE01-A01

- 科学问题：显式绝对幅值旁路对PGA预测贡献多少？
- branch：`exp/fe01-a01-absolute-amplitude-ablation`
- commit：`29ffb20ba0effc0c345a0329ba3f8c658dc8cef0`
- 状态：`verified_complete`
- 数据/协议：val seed42；5新训+3复用ON；16导出；10cell等权noninput MAE；2000 dataset/event draws。
- 结果性质：真实封包已发布；报告审阅≠本轮复算权重。
- 审阅：needs_independent_review
- 关系：继承FE01；FE01_ON复用四ON导出
- 原始入口：[reports/fe01_a01_hpc_results_20261021/README.md](https://github.com/rular099/team_pytorch/blob/29ffb20ba0effc0c345a0329ba3f8c658dc8cef0/reports/fe01_a01_hpc_results_20261021/README.md)；[reports/fe01_a01_hpc_results_20261021/CHATGPT_ANALYSIS_PROMPT.md](https://github.com/rular099/team_pytorch/blob/29ffb20ba0effc0c345a0329ba3f8c658dc8cef0/reports/fe01_a01_hpc_results_20261021/CHATGPT_ANALYSIS_PROMPT.md)

## FE01_ON

### FE01-ON-V4

- 科学问题：四ON在精度、条件分层与真实回放地图方面差异如何？
- branch：`exp/fe01-on-warning-map-evaluation`
- commit：`50672e131d876a604432b283162d5b0189d40cf0`
- 状态：`in_progress`
- 数据/协议：旧val seed42/1310events/194265请求CPU比较；新fixed_query_replay_v4单独身份。
- 结果性质：CPU真实结果；新模型回放NOT_RUN；合成演示明确隔离。
- 审阅：needs_independent_review
- 关系：复用A01 ON结果；不加入RT55/61历史标量排名
- 原始入口：[reports/fe01_on_eval_20261022/README.md](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/reports/fe01_on_eval_20261022/README.md)；[docs/ai/FE01_ON_EVAL_INVENTORY_FIX.md](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/docs/ai/FE01_ON_EVAL_INVENTORY_FIX.md)；[docs/ai/FE01_ON_EVAL_PROTOCOL.md](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/docs/ai/FE01_ON_EVAL_PROTOCOL.md)

## FE02

### FE02-FUSION

- 科学问题：事件memory能否改善单输入query空间场且保护normal？
- branch：`exp/fe02-event-fusion-readout`
- commit：`42b3d5405d9fe9672f0a0925275ca4f7e9e6a9e7`
- 状态：`blocked`
- 数据/协议：FE01 V2 prefix/split/归一化；seed42五组各计划12轮；十cell等权noninput选轮；固定预算。
- 结果性质：实现与部分M训练日志；未形成完成矩阵或性能结论。
- 审阅：needs_independent_review
- 关系：与FE01并行；共享基础代码但独立输出
- 原始入口：[docs/ai/CODEX_RESULT_20261014_FE02_RNG_RESUME.md](https://github.com/rular099/team_pytorch/blob/42b3d5405d9fe9672f0a0925275ca4f7e9e6a9e7/docs/ai/CODEX_RESULT_20261014_FE02_RNG_RESUME.md)；[docs/ai/FE02_PROTOCOL.md](https://github.com/rular099/team_pytorch/blob/42b3d5405d9fe9672f0a0925275ca4f7e9e6a9e7/docs/ai/FE02_PROTOCOL.md)；[scripts/fe02/job.sh](https://github.com/rular099/team_pytorch/blob/42b3d5405d9fe9672f0a0925275ca4f7e9e6a9e7/scripts/fe02/job.sh)

## V01

### V01-CLOSURE

- 科学问题：速度输入/P前缺失是否解释空间场问题？
- branch：`exp/v01-velocity-prep-padding-control`
- commit：`d8f2436273fbe990e77f7151d86a6b629e99c5c1`
- 状态：`verified_complete`
- 数据/协议：val epoch8 seed42；每geometry1194events/84259 query-only targets；5000事件簇draws。
- 结果性质：有限负机制结果待独立判定；工程闭环完整。
- 审阅：needs_independent_review
- 关系：FF/FM/MF/MM与strict AA；非RT55含输入normal人口
- 原始入口：[reports/v01_validation_closure_20261006/RESULT_REVIEW.md](https://github.com/rular099/team_pytorch/blob/d8f2436273fbe990e77f7151d86a6b629e99c5c1/reports/v01_validation_closure_20261006/RESULT_REVIEW.md)；[reports/v01_validation_closure_20261006/verification.json](https://github.com/rular099/team_pytorch/blob/d8f2436273fbe990e77f7151d86a6b629e99c5c1/reports/v01_validation_closure_20261006/verification.json)；[reports/v01_validation_closure_20261006/evidence/configs/vfull__vfull__normal.json](https://github.com/rular099/team_pytorch/blob/d8f2436273fbe990e77f7151d86a6b629e99c5c1/reports/v01_validation_closure_20261006/evidence/configs/vfull__vfull__normal.json)；[reports/v01_validation_closure_20261006/headline_metrics.csv](https://github.com/rular099/team_pytorch/blob/d8f2436273fbe990e77f7151d86a6b629e99c5c1/reports/v01_validation_closure_20261006/headline_metrics.csv)

## LEGACY

### LEGACY-STAGE2-512

- 科学问题：早期容量与动态范围崩塌如何诊断？
- branch：`zhangb/layerwise-station-temporal`
- commit：`8d0131db09f9d89e215b27dda77399b481b3a92a`
- 状态：`unknown`
- 数据/协议：固定512-event overfit研究，不能与全量RT/FE验证直接排名。
- 结果性质：历史导航；最终进度未全恢复。
- 审阅：needs_history_review
- 关系：早于RT55；其他历史branch见audit inventory
- 原始入口：[reports/stage2_512_layerwise_station_temporal_plan.md](https://github.com/rular099/team_pytorch/blob/8d0131db09f9d89e215b27dda77399b481b3a92a/reports/stage2_512_layerwise_station_temporal_plan.md)
<!-- ai-memory:end -->

<!-- 人工补充写在生成区外；record.json 是结构化事实源。 -->
