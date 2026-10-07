# 诊断图注 / Diagnostic captions

证据类型以图标题为准。SYNTHETIC 表示合成输入与未训练的小型下游；
PhaseNet/EQT 的 encoder 可使用已认证 STEAD 权重，仍不是生产 checkpoint 结论。
查询流图展示各段的跨查询 RMS；不同表示量纲不同，不直接跨段比较绝对数值。
gate/范数图来自实际加载状态与本次前向；未来扰动图仅认证 HDF 后的前缀。
尺度图的 ON 幅值允许变化；OFF 保留时长。低能量 clamp 限制另见 eps_limit JSON。

Evidence: SYNTHETIC untrained downstream — team_original_scratch. Source data: ../query_stages.csv, ../gates.csv,
../scale_audit.csv and ../future_audit.csv. Arrays: ../traces/*/query_trace_arrays.npz.

1. query_flow.png: centered RMS across real query coordinates at each representation.
   Representation units differ; compare changes within a stage and K controls,
   not absolute sizes across unrelated representations. A single key makes pure
   attention constant; residual outputs can remain query dependent.
2. gates.png: values read from the actual loaded state, not config defaults;
   branch token norms use ../branch_norms.csv when present.
   Branch norms are separately recorded in traces/*/query_gates_branches.json.
3. interventions.png: measured maximum errors under positive gain and future
   pulse/NaN/Inf interventions. ON delivered amplitude may change; OFF should
   retain duration only. Last-legal-sample controls are excluded from the future
   invariance bars. HDF-prefix checks do not certify upstream preprocessing.
