# FE01 EXPERIMENT_INDEX

<!-- ai-memory:begin -->
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
<!-- ai-memory:end -->

<!-- 人工补充写在生成区外；record.json 是结构化事实源。 -->
