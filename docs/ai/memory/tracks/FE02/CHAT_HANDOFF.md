# FE02 CHAT_HANDOFF

<!-- ai-memory:begin -->
track_id: `FE02`

branch: `exp/fe02-event-fusion-readout`

commit_sha: `618661f99afb67f5c23c9f1a3fd713a0a156e7b5`

last_verified_at: `2026-11-04T07:09:47+00:00`

current_state: `in_progress`

原证据日期：2026-11-04；本次只核验可见来源，不代表重新跑过实验。

完成范围：五组seed42的12轮训练和固定validation证据已归档并复算；独立科研审阅和来源缺口仍待完成。

R0/A/B/C/M五组seed42各12轮/2544更新；M主指标最低但与B的总体差异CI含零，单台/random收益伴随normal多台代价。

实验目标：固定真实冻结DiTing前端，对比R0/A/B/C/M事件融合读出，研究单输入非输入查询的空间辨识及normal/random精度与校准取舍。

默认动作：只读、核验和报告；待办不授予执行权限。

最近工作 result_commit：`5bcf4f558f9571a2ad44ff2639f62bb64e228c2b`；branch：`chore/persistent-ai-handoff`；scope：`handoff_infrastructure`。

已核验：

- 已归档并复算五组seed42各12轮；共同validation十cell等权noninput域为normal/random×1/3/5/10/20秒，每组1310事件/139440非输入目标，PGA坐标log10(m/s²)；所选R0/A/B/C/M epoch为10/11/10/12/10。 来源：[reports/fe02_hpc_results_20261104/README.md](https://github.com/rular099/team_pytorch/blob/618661f99afb67f5c23c9f1a3fd713a0a156e7b5/reports/fe02_hpc_results_20261104/README.md)；[reports/fe02_hpc_results_20261104/data/headline_primary.csv](https://github.com/rular099/team_pytorch/blob/618661f99afb67f5c23c9f1a3fd713a0a156e7b5/reports/fe02_hpc_results_20261104/data/headline_primary.csv)；[reports/fe02_hpc_results_20261104/data/verification.json](https://github.com/rular099/team_pytorch/blob/618661f99afb67f5c23c9f1a3fd713a0a156e7b5/reports/fe02_hpc_results_20261104/data/verification.json)
- 轻量事件表独立重建五组主分数及63行5000次event-cluster bootstrap，seed20261008；本次为已保存复算记录的接手核验，不再次运行模型或原始数据计算。 来源：[reports/fe02_hpc_results_20261104/data/compact_verification.json](https://github.com/rular099/team_pytorch/blob/618661f99afb67f5c23c9f1a3fd713a0a156e7b5/reports/fe02_hpc_results_20261104/data/compact_verification.json)；[reports/fe02_hpc_results_20261104/README.md](https://github.com/rular099/team_pytorch/blob/618661f99afb67f5c23c9f1a3fd713a0a156e7b5/reports/fe02_hpc_results_20261104/README.md)
- 160份恢复回执覆盖10作业×16rank，shim SHA一致；最后R0/A/B/C launcher SHA不同于仓库入口，原入口字节未取得。 来源：[reports/fe02_hpc_results_20261104/data/verification.json](https://github.com/rular099/team_pytorch/blob/618661f99afb67f5c23c9f1a3fd713a0a156e7b5/reports/fe02_hpc_results_20261104/data/verification.json)；[reports/fe02_hpc_results_20261104/README.md](https://github.com/rular099/team_pytorch/blob/618661f99afb67f5c23c9f1a3fd713a0a156e7b5/reports/fe02_hpc_results_20261104/README.md)

下一步：

- 按已发布ChatGPT任务独立审阅点/概率/空间/几何/输入数分层及CI，比较Codex结论。
- 补齐后四组launcher来源、checkpoint字节或可核验元数据、committed journals及sacct；缺失保持UNKNOWN。
- 是否追加multi-seed、独立评价或新架构由用户决定；不自动训练、续跑、提交作业或动用held-out test。

证据范围：

- 单seed且validation同时用于checkpoint选择；bootstrap不含训练seed/选择不确定性，未校正多项比较。
- 原权重字节、committed journals、sacct未提供；不能宣称完整执行链复现或调度器COMPLETED。
- 单台normal晚时刻人口极稀疏，概率仍欠覆盖，C不是等容量消融。
- 预训练重叠、上游在线因果性、未见台站泛化均未认证。


### 未完成事项

- 维护机制已在独立分支完成；科研分支的一次性部署仍待审阅后引入。
- 按已发布ChatGPT任务独立审阅点/概率/空间/几何/输入数分层及CI，比较Codex结论。
- 补齐后四组launcher来源、checkpoint字节或可核验元数据、committed journals及sacct；缺失保持UNKNOWN。
- 是否追加multi-seed、独立评价或新架构由用户决定；不自动训练、续跑、提交作业或动用held-out test。

### 恢复步骤（先读后报告）

- 先读START_HERE及本线STATUS/NEXT_TASK/DECISIONS，核对实际branch/HEAD和固定证据。
- 按HANDOFF_WORKFLOW在独立adoption worktree审阅部署，禁止复制覆盖原HPC源码或另一实验线。
- 接手只读报告；后续研究任务须由当前用户明确委托，不因NEXT_TASK存在自动执行。
<!-- ai-memory:end -->

<!-- 人工补充写在生成区外；record.json 是结构化事实源。 -->
