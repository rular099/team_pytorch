# RT59 EXPERIMENT_INDEX

<!-- ai-memory:begin -->
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

### RT59-SMALL-EVENT-20260928

- 科学问题：既有random validation中的M=3.0小震远端PGA表现如何？
- branch：`rt59-dual-objective-transport-v3`
- commit：`d43c7d63528b3ade71c3506eeb2cb71f90fb3ffd`
- 状态：`verified_complete`
- 数据/协议：历史离线报告，validation单seed，23个M=3.0事件、211个有效非输入目标；事件簇bootstrap5000次。
- 结果性质：9月28日已有分析；本轮仅同步报告/源表/代码。报告位于sync/v01的固定证据commit，科研parent仍RT59。
- 审阅：needs_independent_review
- 关系：RT59 same-forward reference的条件子群；非新训练、非M<3泛化。
- 原始入口：[reports/rt59_small_event_warning_20260928/REPORT.md](https://github.com/rular099/team_pytorch/blob/52ea935249e490033f49209e106eecdedaa9275a/reports/rt59_small_event_warning_20260928/REPORT.md)
<!-- ai-memory:end -->

<!-- 人工补充写在生成区外；record.json 是结构化事实源。 -->
