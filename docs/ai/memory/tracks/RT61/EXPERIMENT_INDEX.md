# RT61 EXPERIMENT_INDEX

<!-- ai-memory:begin -->
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
<!-- ai-memory:end -->

<!-- 人工补充写在生成区外；record.json 是结构化事实源。 -->
