# RT59 STATUS

<!-- ai-memory:begin -->
track_id: `RT59`

branch: `rt59-dual-objective-transport-v3`

commit_sha: `d43c7d63528b3ade71c3506eeb2cb71f90fb3ffd`

last_verified_at: `2026-11-01T07:24:10+00:00`

current_state: `verified_complete`

原证据日期：2026-09-20；本次只核验可见来源，不代表重新跑过实验。

完成范围：epoch8报告及29项门已归档；假设未全面通过。

RT59同时改善random/normal点误差，但严格门为NO_GO 22/29，仍作为后续开发reference。

### verified_results

- 报告门表22 PASS/7 FAIL；normal与random点误差及NLL/Brier改善不能掩盖空间范围/校准保持门失败。 来源：[docs/ai/CODEX_RESULT_20260920_RT59_DUAL_OBJECTIVE_TRANSPORT_V3_VALIDATION.md](https://github.com/rular099/team_pytorch/blob/d43c7d63528b3ade71c3506eeb2cb71f90fb3ffd/docs/ai/CODEX_RESULT_20260920_RT59_DUAL_OBJECTIVE_TRANSPORT_V3_VALIDATION.md)；[reports/rt59_dual_objective_transport_v3_validation_20260920/gates.csv](https://github.com/rular099/team_pytorch/blob/d43c7d63528b3ade71c3506eeb2cb71f90fb3ffd/reports/rt59_dual_objective_transport_v3_validation_20260920/gates.csv)

### pending_tasks

- 后续比较保留同一forward的RT59 reference，避免混用历史RT57。

### limitations

- 仅开发validation/单seed；不等于held-out test、跨seed显著性或未见台站泛化。
- 原包缺checkpoint body，权重SHA不能独立重算。

### evidence_paths

- [docs/ai/CODEX_RESULT_20260920_RT59_DUAL_OBJECTIVE_TRANSPORT_V3_VALIDATION.md](https://github.com/rular099/team_pytorch/blob/d43c7d63528b3ade71c3506eeb2cb71f90fb3ffd/docs/ai/CODEX_RESULT_20260920_RT59_DUAL_OBJECTIVE_TRANSPORT_V3_VALIDATION.md)
- [reports/rt59_dual_objective_transport_v3_validation_20260920/gates.csv](https://github.com/rular099/team_pytorch/blob/d43c7d63528b3ade71c3506eeb2cb71f90fb3ffd/reports/rt59_dual_objective_transport_v3_validation_20260920/gates.csv)
- [pga_configs/transformer_japan_full_2000_2024_rt59_dual_objective_transport_v3_seed42_chaosuan.json](https://github.com/rular099/team_pytorch/blob/d43c7d63528b3ade71c3506eeb2cb71f90fb3ffd/pga_configs/transformer_japan_full_2000_2024_rt59_dual_objective_transport_v3_seed42_chaosuan.json)
<!-- ai-memory:end -->

<!-- 人工补充写在生成区外；record.json 是结构化事实源。 -->
