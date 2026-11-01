# FE01 CHAT_HANDOFF

<!-- ai-memory:begin -->
track_id: `FE01`

branch: `exp/fe01-feature-extractor-site-effects`

commit_sha: `ba4fa7740d9fde5fde87a7b0ae0397037209baf4`

last_verified_at: `2026-11-01T07:24:10+00:00`

current_state: `in_progress`

原证据日期：2026-10-06；本次只核验可见来源，不代表重新跑过实验。

完成范围：九组训练结果已归档；EVAL1身份/前向门部分完成，完整补评未全部取得。

三家族×三seed原比较已完成；EVAL1八组forward PASS、TEAM三组被评价清单条件阻挡，后续A01/ON另线推进。

已核验：

- 原九组各12轮的报告与EVAL1 summary均已归档；本轮只核验保存文件，未重算大型原始产物。 来源：[reports/fe01_hpc_results_20261004/README.md](https://github.com/rular099/team_pytorch/blob/ba4fa7740d9fde5fde87a7b0ae0397037209baf4/reports/fe01_hpc_results_20261004/README.md)；[reports/fe01_eval1_hpc_gates_20261006/summary.json](https://github.com/rular099/team_pytorch/blob/ba4fa7740d9fde5fde87a7b0ae0397037209baf4/reports/fe01_eval1_hpc_gates_20261006/summary.json)
- EVAL1 identity 11组通过、forward 8 PASS/3 BLOCKED；缺预训练清单SHA的TEAM scratch在前向前被阻挡，不证明权重损坏。 来源：[reports/fe01_eval1_hpc_gates_20261006/summary.json](https://github.com/rular099/team_pytorch/blob/ba4fa7740d9fde5fde87a7b0ae0397037209baf4/reports/fe01_eval1_hpc_gates_20261006/summary.json)

下一步：

- 按FE01_A01与FE01_ON各自新来源审阅后续，不能把旧EVAL1阻挡当新线仍未完成。
- 站点泛化、场地效应、回放和held-out test要逐项查实际证据。

证据范围：

- 原比较跨encoder仍有原生窗口/能力条件化采样混杂。
- validation站点全部在训练曝光表中，不支持未见台站泛化。
- 缺sacct不能声称调度器COMPLETED。
<!-- ai-memory:end -->

<!-- 人工补充写在生成区外；record.json 是结构化事实源。 -->
