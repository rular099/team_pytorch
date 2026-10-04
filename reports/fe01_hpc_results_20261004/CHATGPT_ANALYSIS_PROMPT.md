请分析 `rular099/team_pytorch` 分支 `exp/fe01-feature-extractor-site-effects` 此次同步commit中的FE01超算结果。
先确认实际读到的repo/branch/commit。报告入口是 `reports/fe01_hpc_results_20261004/README.md`。

先读 `evidence_checks.json`、`run_summary.csv`、`seed_summary.csv`、
`validation_by_time_geometry_role.csv`、`validation_paired_event_bootstrap.csv`、
`training_time_exposure.csv` 与 `audit_summary.csv`；需要配置、参数或审计细节时再读metadata。
来源SHA、原始相对路径见source_inventory。22MB逐事件表用于下载后的程序复核，无须全文塞进对话。

请用中文回答：

1. 九组正式训练的验证结果能支持什么结论？哪个模型点预测更好，哪个模型校准更好？
2. normal/random和早期/后期时间差异在哪里？全目标与非输入目标是否给出一致判断？
3. 三seed稳定性与按事件bootstrap的证据各能说明什么？验证集选checkpoint造成哪些解释限制？
4. 各模型原生窗口及能力条件化采样是否影响对encoder/预训练收益的归因？
5. 在快速推进、复用已训练RT55/RT61的前提下，下一批超算评估应按什么优先级执行？给出需要的checkpoint/config/输出证据，不默认新增重训或打开test。

要求：

- 明确所有当前成绩都是validation，按曲线预定规则选出epoch，未取得best.pth内部epoch或SHA。
- 主指标为1/3/5/10/20s × normal/random的10个cell等权noninput MAE，坐标log10(m/s²)，不是pooled MAE。
- seed样本标准差与固定模型的事件bootstrap CI分开，不把不同seed的相同事件当作新的独立事件。
- 不把capability支持表、sampler-only审计、随机时刻manifest或日志曝光直方图当作对应预测成绩。
- 没有RT55同协议复评，不能与历史标量成绩直接排名；RT61只作增强完整系统参考。
- 场地效应/spatial/replay/test产物未提供；不能为它们编造结论。
- 区分已核验事实、推断和下一步建议；建议先等待用户授权再执行新增超算计算。
