# FE01 NEXT_TASK

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

### 下一步提案

任务：`FE01-READONLY-REVIEW`；类型：`read_only_review`。

独立复核原FE01和EVAL1的已归档证据，区分原TEAM阻挡与A01/ON后续状态，报告仍缺的来源和评价。

**execution_authorization: not_granted_by_this_file**

默认只核验并报告当前状态；是否实施取决于当前用户任务。不得根据此文件自动训练、提交HPC作业、续跑或修改重要实验协议。

### 可先读取的材料

- 按本线固定commit读取原训练报告和EVAL1 summary；区分validation、身份门和完整性能评价。
- 需要A01或ON时显式选择该独立track及其来源，不把其任务写进FE01主线或FE02。

### 若获授权后的验收条件

- 能从固定commit恢复前端比较目标、已验证门与尚缺证据。
- 保留train/validation/test、未见台站与sacct的解释边界；没有新计算或协议变更。

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
