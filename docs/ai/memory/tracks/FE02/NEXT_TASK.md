# FE02 NEXT_TASK

<!-- ai-memory:begin -->
track_id: `FE02`

branch: `exp/fe02-event-fusion-readout`

commit_sha: `618661f99afb67f5c23c9f1a3fd713a0a156e7b5`

last_verified_at: `2026-11-04T03:33:49+00:00`

current_state: `in_progress`

原证据日期：2026-11-04；本次只核验可见来源，不代表重新跑过实验。

完成范围：五组seed42的12轮训练和固定validation证据已归档并复算；独立科研审阅和来源缺口仍待完成。

R0/A/B/C/M五组seed42各12轮/2544更新；M主指标最低但与B的总体差异CI含零，单台/random收益伴随normal多台代价。

实验目标：固定真实冻结DiTing前端，对比R0/A/B/C/M事件融合读出，研究单输入非输入查询的空间辨识及normal/random精度与校准取舍。

默认动作：只读、核验和报告；待办不授予执行权限。

最近工作 result_commit：`618661f99afb67f5c23c9f1a3fd713a0a156e7b5`；branch：`exp/fe02-event-fusion-readout`；scope：`evidence_review`。

### 下一步提案

任务：`FE02-READONLY-REVIEW`；类型：`read_only_review`。

按现有完整五组validation证据独立审阅并补充来源差异说明；先核验，不启动下一轮实验。

**execution_authorization: not_granted_by_this_file**

默认只核验并报告当前状态；是否实施取决于当前用户任务。不得根据此文件自动训练、提交HPC作业、续跑或修改重要实验协议。

### 可先读取的材料

- 读取results、verification、headline及review固定commit证据。
- 核对single/multi、normal/random、all/noninput、空间误差及概率校准；注意稀疏cell和未校正CI。

### 若获授权后的验收条件

- 能报告五组完成记录、所选epoch、来源缺口及受限科学结论。
- 明确M与B差异CI包含零及normal多台代价，不以single空间范围变大单独认定成功。

### 未完成事项

- 按已发布ChatGPT任务独立审阅点/概率/空间/几何/输入数分层及CI，比较Codex结论。
- 补齐后四组launcher来源、checkpoint字节或可核验元数据、committed journals及sacct；缺失保持UNKNOWN。
- 是否追加multi-seed、独立评价或新架构由用户决定；不自动训练、续跑、提交作业或动用held-out test。

### 恢复步骤（先读后报告）

- 从固定结果commit读取轻量报告和ChatGPT复核任务，报告实际可见branch/commit。
- 核对launcher来源缺口；没有当前授权时只报告提案，不运行submit.sh、resume脚本或测试集。
<!-- ai-memory:end -->

<!-- 人工补充写在生成区外；record.json 是结构化事实源。 -->
