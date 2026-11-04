# FE01 STATUS

<!-- ai-memory:begin -->
track_id: `FE01`

branch: `exp/fe01-feature-extractor-site-effects`

commit_sha: `577a0cec43643597a61209209a770aeecf5fefa0`

last_verified_at: `2026-11-04T07:09:47+00:00`

current_state: `in_progress`

原证据日期：2026-10-06；本次只核验可见来源，不代表重新跑过实验。

完成范围：九组训练结果已归档；EVAL1身份/前向门部分完成，完整补评未全部取得。

三家族×三seed原比较已完成；EVAL1八组forward PASS、TEAM三组被评价清单条件阻挡，后续A01/ON另线推进。

实验目标：比较原生因果窗口下不同波形前端与共同PGA下游；原FE01/EVAL1和后续FE01_A01、FE01_ON各自保留实验身份。

默认动作：只读、核验和报告；待办不授予执行权限。

最近工作 result_commit：`5bcf4f558f9571a2ad44ff2639f62bb64e228c2b`；branch：`chore/persistent-ai-handoff`；scope：`handoff_infrastructure`。

### verified_results

- 原九组各12轮的报告与EVAL1 summary均已归档；本轮只核验保存文件，未重算大型原始产物。 来源：[reports/fe01_hpc_results_20261004/README.md](https://github.com/rular099/team_pytorch/blob/ba4fa7740d9fde5fde87a7b0ae0397037209baf4/reports/fe01_hpc_results_20261004/README.md)；[reports/fe01_eval1_hpc_gates_20261006/summary.json](https://github.com/rular099/team_pytorch/blob/ba4fa7740d9fde5fde87a7b0ae0397037209baf4/reports/fe01_eval1_hpc_gates_20261006/summary.json)
- EVAL1 identity 11组通过、forward 8 PASS/3 BLOCKED；缺预训练清单SHA的TEAM scratch在前向前被阻挡，不证明权重损坏。 来源：[reports/fe01_eval1_hpc_gates_20261006/summary.json](https://github.com/rular099/team_pytorch/blob/ba4fa7740d9fde5fde87a7b0ae0397037209baf4/reports/fe01_eval1_hpc_gates_20261006/summary.json)

### pending_tasks

- 按FE01_A01与FE01_ON各自新来源审阅后续，不能把旧EVAL1阻挡当新线仍未完成。
- 站点泛化、场地效应、回放和held-out test要逐项查实际证据。

### limitations

- 原比较跨encoder仍有原生窗口/能力条件化采样混杂。
- validation站点全部在训练曝光表中，不支持未见台站泛化。
- 缺sacct不能声称调度器COMPLETED。

### evidence_paths

- [reports/fe01_hpc_results_20261004/README.md](https://github.com/rular099/team_pytorch/blob/ba4fa7740d9fde5fde87a7b0ae0397037209baf4/reports/fe01_hpc_results_20261004/README.md)
- [reports/fe01_eval1_hpc_gates_20261006/summary.json](https://github.com/rular099/team_pytorch/blob/ba4fa7740d9fde5fde87a7b0ae0397037209baf4/reports/fe01_eval1_hpc_gates_20261006/summary.json)
- [docs/ai/FE01_PROTOCOL.md](https://github.com/rular099/team_pytorch/blob/ba4fa7740d9fde5fde87a7b0ae0397037209baf4/docs/ai/FE01_PROTOCOL.md)
- [configs/fe01/formal/formal__phasenet_pretrained_frozen__seed42.json](https://github.com/rular099/team_pytorch/blob/ba4fa7740d9fde5fde87a7b0ae0397037209baf4/configs/fe01/formal/formal__phasenet_pretrained_frozen__seed42.json)
- [docs/ai/README.md](https://github.com/rular099/team_pytorch/blob/577a0cec43643597a61209209a770aeecf5fefa0/docs/ai/README.md)
- [docs/ai/START_HERE.md](https://github.com/rular099/team_pytorch/blob/5bcf4f558f9571a2ad44ff2639f62bb64e228c2b/docs/ai/START_HERE.md)


### 未完成事项

- 维护机制已在独立分支完成；科研分支的一次性部署仍待审阅后引入。
- 按FE01_A01与FE01_ON各自新来源审阅后续，不能把旧EVAL1阻挡当新线仍未完成。
- 站点泛化、场地效应、回放和held-out test要逐项查实际证据。

### 恢复步骤（先读后报告）

- 先读START_HERE及本线STATUS/NEXT_TASK/DECISIONS，核对实际branch/HEAD和固定证据。
- 按HANDOFF_WORKFLOW在独立adoption worktree审阅部署，禁止复制覆盖原HPC源码或另一实验线。
- 接手只读报告；后续研究任务须由当前用户明确委托，不因NEXT_TASK存在自动执行。
<!-- ai-memory:end -->

<!-- 人工补充写在生成区外；record.json 是结构化事实源。 -->
