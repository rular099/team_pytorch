# FE01_ON CHAT_HANDOFF

<!-- ai-memory:begin -->
track_id: `FE01_ON`

branch: `exp/fe01-on-warning-map-evaluation`

commit_sha: `50672e131d876a604432b283162d5b0189d40cf0`

last_verified_at: `2026-11-01T07:24:10+00:00`

current_state: `in_progress`

原证据日期：2026-10-25；本次只核验可见来源，不代表重新跑过实验。

完成范围：真实CPU旧导出再分析已完成；生产inventory恢复/模型回放尚无完成回执。

四ON旧导出CPU比较已发布；10月25日修复inventory清单路径，真实修复inventory与新回放结果待核验。

已核验：

- 真实CPU比较与合成动态图均存在，但合成图醒目标SYNTHETIC，不能当Japan新前向结果。 来源：[reports/fe01_on_eval_20261022/README.md](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/reports/fe01_on_eval_20261022/README.md)
- 旧inventory报Pretrained manifest SHA differs；修复保留认证manifest原路径与SHA，新run只手动恢复inventory，生产模型级验证尚未运行。 来源：[docs/ai/FE01_ON_EVAL_INVENTORY_FIX.md](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/docs/ai/FE01_ON_EVAL_INVENTORY_FIX.md)；[docs/ai/FE01_ON_EVAL_PROTOCOL.md](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/docs/ai/FE01_ON_EVAL_PROTOCOL.md)

下一步：

- 核对新inventory run的seal与真实回执；沿用新run/source，不自动重提CPU或启动后续阶段。
- 只有取得真实模型级门后才能解释TTUM/shadow、生产网格与实测时延。

证据范围：

- 本次仅核验固定commit文档，不验证生产checkpoint/HDF5/DCU。
- 低noninput MAE不自动意味着更早TTUM或更小shadow；通信与上游因果性仍UNKNOWN。
<!-- ai-memory:end -->

<!-- 人工补充写在生成区外；record.json 是结构化事实源。 -->
