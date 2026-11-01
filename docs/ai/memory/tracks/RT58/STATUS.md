# RT58 STATUS

<!-- ai-memory:begin -->
track_id: `RT58`

branch: `rt58-waveform-anchor-transfer-field`

commit_sha: `40830acd475bd40347fef03427b52c4a04432dc3`

last_verified_at: `2026-11-01T07:24:10+00:00`

current_state: `verified_complete`

原证据日期：2026-09-15；本次只核验可见来源，不代表重新跑过实验。

完成范围：固定epoch8验证与十项门检查已归档。

WATF相对相同gamma=1.66基线的random MAE变差，准确input anchor未转成空间幅度收益。

### verified_results

- 报告给出anchor correction相对相同gamma基线random MAE +0.000458 dex；不能把gamma收益算作新anchor收益。 来源：[docs/ai/CODEX_RESULT_20260915_RT58_WATF_VALIDATION.md](https://github.com/rular099/team_pytorch/blob/40830acd475bd40347fef03427b52c4a04432dc3/docs/ai/CODEX_RESULT_20260915_RT58_WATF_VALIDATION.md)；[reports/rt58_waveform_anchor_transfer_validation_20260915/summary.json](https://github.com/rular099/team_pytorch/blob/40830acd475bd40347fef03427b52c4a04432dc3/reports/rt58_waveform_anchor_transfer_validation_20260915/summary.json)

### pending_tasks

- 保留负结果，追溯RT59重新设计；不因总指标好看而改对照。

### limitations

- 仅开发validation/单seed；不等于held-out test、跨seed显著性或未见台站泛化。
- 归档缺checkpoint body与训练标量，不能本地复查权重或epoch轨迹。

### evidence_paths

- [docs/ai/CODEX_RESULT_20260915_RT58_WATF_VALIDATION.md](https://github.com/rular099/team_pytorch/blob/40830acd475bd40347fef03427b52c4a04432dc3/docs/ai/CODEX_RESULT_20260915_RT58_WATF_VALIDATION.md)
- [reports/rt58_waveform_anchor_transfer_validation_20260915/summary.json](https://github.com/rular099/team_pytorch/blob/40830acd475bd40347fef03427b52c4a04432dc3/reports/rt58_waveform_anchor_transfer_validation_20260915/summary.json)
- [pga_configs/transformer_japan_full_2000_2024_rt58_waveform_anchor_transfer_seed42_chaosuan.json](https://github.com/rular099/team_pytorch/blob/40830acd475bd40347fef03427b52c4a04432dc3/pga_configs/transformer_japan_full_2000_2024_rt58_waveform_anchor_transfer_seed42_chaosuan.json)
<!-- ai-memory:end -->

<!-- 人工补充写在生成区外；record.json 是结构化事实源。 -->
