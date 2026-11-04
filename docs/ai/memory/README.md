# 项目记忆使用说明

**2026-11-04 入口扩展：** 新会话从 [START_HERE](../START_HERE.md) 自动路由到FE01或FE02。
这两线继续以同一record为事实源；新增 `handoff` 字段生成 NEXT_TASK.md，
与原四份视图一起更新。`handoff.last_session` 记录结果commit/branch、证据、未完成和恢复步骤；
`next_task.execution_authorization` 必须是 `not_granted_by_this_file`。
其他10条历史track不强制升级schema；全局快照仍有独立核验日期。
原下文三份全局入口用于跨线历史导航，不取代新会话的本线状态。

新对话只先读取 [PROJECT_MEMORY](PROJECT_MEMORY.md)、[CURRENT_STATUS](CURRENT_STATUS.md)、
[CHAT_HANDOFF](CHAT_HANDOFF.md)。需要历史时按 [EXPERIMENT_INDEX](EXPERIMENT_INDEX.md) 与
[DECISION_LOG](DECISION_LOG.md) 的ID找固定commit证据，不全文加载历史。

## 事实源与生成文件

`registry.json`列出12条独立工作线。每条线唯一人工维护的结构化事实源为
`tracks/<ID>/record.json`，四份Markdown为其可再生成视图：STATUS、DECISIONS、EXPERIMENT_INDEX、CHAT_HANDOFF。
全局 `observations/<ID>.json` 是协调者从独立来源读取的最后快照；`snapshot.json` 固定每条来源和摘要。
全局CURRENT_STATUS/CHAT_HANDOFF/DECISION_LOG/EXPERIMENT_INDEX均由此生成，不另写一份进度。
生成区标有 `ai-memory:begin/end`；人工补充写在区外，更新会保留。
结论来自人工编写的record与证据，不扫描指标文件名、不自动解释科学结果。

`last_verified_at`指本轮读取可见来源的时间，`evidence_as_of`是原实验/报告日期。
`verified_complete`只覆盖`completion_scope`；不等于假设通过或整个科研完成。
`unknown`表示缺证据，其他状态为in_progress/planned/blocked/superseded。

## 日常操作

```bash
# 在已安装公共工具的对应实验worktree中，人工编辑record.json后刷新本线四份视图
python tools/ai_memory_update.py --track FE01
python tools/ai_memory_check.py --scope FE01 --base <本轮任务base完整SHA>
# 协调worktree中，先按正常Git权限fetch，再指定两个精确的本地/远端ref读取
python tools/ai_memory_update.py --coordinator --source FE01=origin/exp/fe01-feature-extractor-site-effects --source FE02=origin/exp/fe02-event-fusion-readout
python tools/ai_memory_check.py
```

实验结果commit后，人工将record的commit_sha设为该科研/证据结果commit，写入真正可核验的结论、
原始证据路径、未解决问题；重要决定追加ID，新实验/收尾更新实验条目，再提交本线记忆。
FE01/FE02还须更新handoff.last_session/next_task，并使用只读ai_handoff核验。
不要试图让文档保存“包含自身的commit SHA”；生成所依据的commit与最后承载文档的commit区分。
未部署记忆的旧分支被指定为source时会报警并保留已审计状态，不会变成unknown/未完成。

```bash
# 只读预览、冲突风险与资源路径检查，不执行merge或创建产物目录
python tools/ai_memory_update.py --coordinator --source FE02=origin/exp/fe02-event-fusion-readout --dry-run
python tools/ai_memory_check.py --merge-risk <FE01-ref> <FE02-ref>
python tools/ai_memory_check.py --resources /path/to/actual-resources.json
python -m unittest discover -s tests -p test_ai_memory.py -v
```

工具只依赖Python标准库和Git。资源JSON格式及初始化/恢复步骤见
[PARALLEL_WORKFLOW](PARALLEL_WORKFLOW.md)。检查能发现结构化字段、commit/路径、
重复ID、生成区和快照冲突、过期核验；不能自动裁决任意自然语言科学矛盾。
跨commit证据用`git cat-file`核验。网络不可用且对象不在本地的两个较新FE01分支，
使用 `audit/remote_refs.json` 的GitHub连接器读取回执与原文Git blob SHA交叉核验；
这证明当次读取，不代替未来实时fetch或原始大产物复算。只读原文快照保留原相对链接，
应到其固定commit打开完整目录。

完整验收与新对话问答见 [审阅请求](../CHATGPT_REVIEW_REQUEST_PROJECT_MEMORY.md)。
