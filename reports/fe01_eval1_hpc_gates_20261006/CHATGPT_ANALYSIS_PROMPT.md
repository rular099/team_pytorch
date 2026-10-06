# 交给ChatGPT：FE01-EVAL1实际超算核验结果

请通过GitHub读取仓库 `rular099/team_pytorch` 的分支 `exp/fe01-feature-extractor-site-effects`，回报实际读到的精确commit，优先读取 `reports/fe01_eval1_hpc_gates_20261006/README.md`、`summary.json` 和 `integrity_checks.json`。本次分析/整理base为 `221f8b24483148d630e0eb85a179d763f8c580ac`；超算运行source为 `4c9c1757f726e571022a398cb5a217ca6c325cbc`，评价模块SHA `e65a7dd017828da10adcf5c45c4aa32d1d702833910972556d4eda5b6a640564`。请勿把整理文档的commit当成超算执行commit。

本轮目标是审阅已提供的真实identity/verify/reference结果，决定下一步最小修复与评价次序。不要要求重训RT55/RT61、新训DiTing公共下游或开启test，不根据本批小样本门重选epoch、阈值或sigma。

请先确认已验证事实：

1. 11组checkpoint均IDENTITY_PASS；八组forward PASS，TEAM scratch三seed BLOCKED。内部epoch与权重SHA在 `model_gates.csv` / `evidence/identity/checkpoint_inventory.json`。
2. TEAM错误发生在模型构建/前向之前：`fe01_review/runners.py:124`无条件比较SHA。三份原TEAM lock的预训练清单SHA为null，lock原字节SHA与超算inventory一致，见 `evidence/training_locks/`；原合同由 `fe01/engine.py:359` 和 `fe01/model.py:130` 支持。这不是清单实际改动或TEAM预测数值失败的证据。
3. 八组核验每组32个validation事件/decision、544目标；扩展query/cache检查仅前4个decision，future扰动仅前2个。六组PhaseNet/EQT核对选定epoch的旧CSV；RT55/RT61的L0为同输入tensor的原模型/包装器等价，并未重新建立历史loader。
4. train-only参照有9084事件、960台站、146299最终标签；拟合使用train且为回顾性oracle metadata。九组FE01曝光表覆盖全部915个validation台站，input/query均见过；RT55/RT61曝光仍UNKNOWN。未见台站泛化不成立。
5. identity Job29351785、verify数组29354874、reference29354884有提交回执和产物；没有sacct，不声称scheduler COMPLETED或ExitCode。后续六阶段没有提供提交/预测结果。

请回答以下问题，并引用仓库内的具体文件/字段：

- 评价入口应该如何最小修复TEAM scratch的“不适用预训练清单”条件，同时保留所有pretrained模型的严格SHA校验、原lock和训练源码？请明确source pin/new run ID与旧PASS不能直接沿用的边界。
- 是否先让通过的8组继续 `legacy_fixed/random/long/replay/analyze`，还是先修复并重新核验以纳入TEAM？说明各选择会留下什么比较缺项；不能暗中覆盖失败输出或改成PASS。
- RT55 epoch32、RT61 epoch8的当前身份和L0/C1证据分别支持哪些主张？哪些仍需实际同人口成绩或历史loader对照？不要复制旧系统为三个DiTing seed。
- 训练参照和真实曝光怎样支持跨事件台站残差分析？区分“有参照系数”与“预测系统恢复可重复场地效应”；哪些阶段尚未执行？
- 从前轮 `reports/fe01_eval1_review_20261005/` 的validation性能证据到本批新核验，哪些结论现在更可信，哪些仍缺随机时刻、长时刻、空间图/回放和调度器证据？保持validation/test区分。

请按“已验证事实、原因分析、最小下一步、尚缺证据”组织回复。若输出实施建议，请用项目约定的 `[AI-HANDOFF]`，以实际读取的当前commit为base，明确不修改原115训练源码或重签旧lock。日志/JSON已替换私有路径；原字节SHA见 `source_inventory.csv`，实际导出SHA见 `artifact_manifest.sha256`。
