# 给 ChatGPT 的 FE02 独立复核任务

请读取 `rular099/team_pytorch` 的 `exp/fe02-event-fusion-readout` 分支。
以用户给定的本次结果 commit 为准，先报告实际读到的完整 SHA。
分析基点为 `42b3d5405d9fe9672f0a0925275ca4f7e9e6a9e7`，实际训练 source 为
`9f70371838ec31044e559096a939f023d2924f37`，两者不能混称。

2026-11-03 接手报告之后新增的本地证据，已放入本目录。五组 seed42 都有完整训练曲线和
真实固定 validation 导出，不能继续用“仅 M 五轮”描述当前证据；也不能宣称已核验 sacct 或权重字节。

请依次读根 AGENTS.md、docs/ai/PROJECT_CONTEXT.md、docs/ai/README.md、SESSION_SUMMARY.md、
本目录 README、data/verification.json、data/source_verification.json 和 data/comparison_contract.json。
再读主表、endpoints、配对 bootstrap、分层表、空间表及 data/evidence/ 各组配置/审计。
报告若过长，先读这些轻量表；primary_event_cells_noninput.csv 能重建主分数和 CI，
primary_spatial_fields.csv 能独立重建空间汇总。全部 raw 文件的 bytes/SHA 在 source_artifacts.csv。

请先独立得出事实和解释，再与 Codex README 比较，重点回答：

1. 五组是否确实共享 encoder、population、split、初始化、训练预算、输入和查询？
   哪些事实只由 provenance 记录支持，哪些已用真实 CSV 重算？
2. M 对 R0 主 MAE 仅约1.09%的差异，以及对 B 的 CI 包含零，足以支持什么级别的结论？
   应怎样处理单 seed、多项未校正比较、在同一 validation 选择 checkpoint 的限制？
3. normal/random 与 single/multi 的取舍是什么？不要只转述总体排名。
   normal 晚时刻单台极少、部分 bootstrap 空 cell 的影响有多大？
4. M/C 的空间范围恢复是否伴随 centered/pairwise 误差改善？
   B 的单台空间差异为何仍接近 R0/A？哪些机制解释只是推断？C 不等容量如何限制归因？
5. NLL、Brier、sigma、精确 MDN coverage/width、bias/slope 是否支持同一个判断？
   区分等 cell 与 target-pooled 指标，不把 all-target 代替 noninput。
6. 160 份 CPU RNG 回执支持了什么？最后 R0/A/B/C 的 launcher SHA 与仓库不同、
   原 launcher 字节/权重/journals/sacct 缺失，如何影响可复现性和结论强度？
7. 与 Codex 的判断有哪些一致与分歧？若建议补实验，请列出最小对照、预期可观察效果、
   兼容性风险和验收标准；不要把建议写成已实施，也不要使用 held-out test 调参。

所有数字需附 split、所选 epoch、协议、目标人口、样本数、单位和具体表名。
PGA 坐标为 log10(m/s²)。不得将这些结果称作 held-out test 或未见台站泛化。
输出使用仓库规定的 [AI-HANDOFF] 字段；分析任务的 proposed_change 写 none。
