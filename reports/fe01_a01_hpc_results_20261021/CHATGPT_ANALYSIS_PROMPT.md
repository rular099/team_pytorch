请独立分析本批 FE01-A01 超算结果。Codex 的判断另行封存，用户将在你完成后对照；请先只依据本包形成自己的结论。

仓库：`rular099/team_pytorch`
分支：`exp/fe01-a01-absolute-amplitude-ablation`
结果目录：`reports/fe01_a01_hpc_results_20261021/`
结果整理前基线：`59f3eb30d6c955ffe2904aee40693af71dc0a440`。
请以用户提供的本次 result commit 为准，先回报你实际读取的 repo/branch/commit。
如果使用上传ZIP，先说明依据的是上传的文件；不要假装已从GitHub读到某个commit。

按顺序读：

1. `README.md`、`evidence_checks.json`、`export_condition_checks.json`、`DATA_DICTIONARY.md`。
2. `paired_mae_summary.csv`、`model_summary.csv`、`training_audit_summary.csv`、`training_curves.csv`。
3. `validation_by_time_geometry_role.csv`、`time_role_population.csv`。
4. `paired_event_bootstrap.csv`、`bootstrap_support_audit.csv`。
5. `paired_field_levels.csv`、`paired_field_by_cell.csv`。
6. `diagnostic_query_stages.csv`、`diagnostic_query_sensitivity.csv`、`diagnostic_nested_matched.csv`、`diagnostic_invariance_checks.csv`。
7. 按需读 `source_metadata/`。若复算主区间，可用 `event_cells_noninput.csv.gz`，不必全文放进对话。

统计或诊断定义不清楚时，优先读 `method_sources/` 中与原HPC pin一致的六份源码快照。

请用中文给出独立的、证据可追溯的分析：

1. 四组ON/OFF的主端点变化、事件配对区间和固定epoch12检查分别说明什么？把效应大小、统计不确定性与单seed限制分开。
2. 全目标与非输入目标的模型排序是否相同？normal/random、早期/后期、K1/Kmulti和triggered/untriggered之间有哪些差异？解释时报告实际人口和空cell，注意时间变化时目标角色和目标集合也在变化。
3. NLL、CRPS、Brier、覆盖率、预测sigma及区间宽度是否给出一致判断？覆盖率更接近95%是否足以支持更好的概率预测？
4. 空间场level/shape分解、range与等距离端点有什么变化？按实际eligible field人口解释，不能把目标等权与field等权混合。
5. K1相关区间有哪些支持与重采样限制？检查effective_replicates和empty_cell_fraction，尤其空间场端点；报告这些限制是否改变结论的强弱。
6. 信息流诊断支持“坐标完全丢失”还是其他判断？定位证据来自哪些stage、局部梯度和嵌套K对照？明确区分旧ON权重的推理时干预、未训练DiTing初态与实际新训练ON/OFF结果。
7. 这批结果对显式绝对幅值旁路、归一化波形和时长旁路能支持怎样的结论？哪些关于因果性、预训练收益或空间泛化的结论超出了证据？
8. 最后按“已核验事实 / 有条件推断 / 尚缺证据 / 建议优先级”组织结论。若建议后续实验，先解释它要区分什么假设；不要默认要求重训、扩大搜索或打开test。

分析要求：

- 所有当前性能均为validation、seed42。5组新训练加3组复用ON，selected与epoch12不是独立训练重复。
- 主端点为1/3/5/10/20秒×normal/random的10个cell等权noninput MAE，单位log10(m/s²)。不要以pooled MAE、平均RMSE或平均R²替代。
- 每份导出1310事件/194265请求，noninput139440请求；不要将16份导出合计的行数作为独立样本量。
- selected按预定验证规则选轮；事件bootstrap条件于固定模型与所选轮次，不能代替训练seed区间或消除复用validation的选轮偏差。
- 主配对差值是OFF减ON。误差、proper score、区间宽度与覆盖率的正负方向含义不同。
- 不把多个次要端点的区间当作经过多重比较校正的验证性检验。
- triggered_noninput的normal/1秒为空；只支持9个cell。其他子组也要查具体支持表，不补零。
- K1空间field区间部分重采样因缺cell被排除；不要只看PASS标签和区间是否跨零。
- 普通nested表不同K的可用decision数不同。要比较相同cohort，优先使用`diagnostic_nested_matched.csv`。
- 诊断标识`matched_initial_checkpoint_NOT_TRAINED`必须保留其初态范围；`real_ON_checkpoint_inference_intervention`必须保留其推理时干预范围。
- 局部敏感性、跨query离散度和不物理的coords-only置换不是空间预测准确度；空间性能判断用真实标签上的场指标。
- 训练曝光、每秒回放、未见台站泛化、40/90秒预测和held-out test的证据范围分别核查，不能用manifest或能力表替代实际成绩。
- RT55/RT61完整系统没有进入本批结果；不要与历史标量强行排名。跨encoder比较仍有原生窗口和能力条件化采样差异。
- 归档manifest验证封装后的字节；原eval seal指向未脱敏字节。原plan/权重不在本包，不把脱敏JSON的SHA变化误报为损坏，也不要宣称本地重读了权重张量。
- 引用重要数字时给出文件名、checkpoint policy、目标组、时间/几何权重和样本量。

请在最后列出最有把握的3条结论、最容易被误读的3点，以及仍需要用户决定的事项。
