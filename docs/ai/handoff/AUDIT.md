# 持久接手入口审计与验收

任务：`20261104-persistent-ai-handoff`；仓库：`rular099/team_pytorch`。
本轮独立分支：`chore/persistent-ai-handoff`。
实现 base：`826acaadc89979d6ad324d014123def8353a429c`。

## 已有机制与本轮取舍

已审计当前 FE02 worktree 的 AGENTS、docs/ai/README、PROJECT_CONTEXT、SESSION_SUMMARY，
同时读取旧协调分支的 memory/README、PARALLEL_WORKFLOW、registry、逐线 record/视图、
三个维护工具及其14项测试。原 FE01 分支入口和 EVAL1 summary 也按完整 SHA 读取。

旧机制已有12条线、record事实源、四种生成视图、证据/路径/历史ID检查、分线锁和协调汇总，
因此本轮复用它们，保留原路径与schema兼容，不建立第二套STATUS事实源。
原工具对coordinator角色的分支没有强制约束，原接手也缺少只读路由和NEXT_TASK。
FE01/FE02实际科研分支尚未引入旧memory机制，不能声称新会话会自动读取未部署的文件。

新增 START_HERE 与只读 reader，FE01/FE02 record 新增 handoff 扩展并生成 NEXT_TASK，
强化 coordinator 分支门、scope-staged 检查及结果commit/来源分支检查。
其他10条线继续兼容旧schema。三份历史协作入口只添加导航；精简 Project Instructions
改为固定链接，避免再维护一份长科研进度 Prompt。

FE01 最新核验 source 为 `577a0cec43643597a61209209a770aeecf5fefa0`；
FE02 为 `618661f99afb67f5c23c9f1a3fd713a0a156e7b5`。
仅核验已有报告/JSON/commit，不重新加载权重或运行模型。
FE02 已完整归档结果取代旧快照的部分进度；保留缺sacct、launcher/journals/权重等限制。
FE01 原EVAL1的限制与A01/ON分开，不让FE02状态更新覆盖它们。

## 四种接手验收

| 模拟 | 实际测试入口 | 恢复内容 | 结果 |
| --- | --- | --- | --- |
| 新FE01 Codex | AGENTS→START→按当前分支自动识别 | 前端目标、归档训练与EVAL1门、派生线边界、证据缺口和只读下一步 | PASS |
| 新FE02 Codex | AGENTS→START→按当前分支自动识别 | R0/A/B/C/M目标、完整五组结果、来源缺口与独立复核提案 | PASS |
| 新FE01 ChatGPT | 固定入口→显式FE01→指定ref读取 | 同一事实与恢复步骤，角色限定科研审阅/AI-HANDOFF | PASS |
| 新FE02 ChatGPT | 固定入口→显式FE02→指定ref读取 | 同一完整结果与限制，待办不授权执行 | PASS |

这些是程序化“冷读取”模拟：临时仓库引用真实已固定的证据对象，测试分支HEAD是合成的
安装后命名空间，不是新科研结果；没有启动四个独立模型，也没有依赖旧聊天作答。
reader前后比较工作文件字节，确认只读。具体恢复答案见 [acceptance.json](acceptance.json)。

## 并行与失败路径

临时仓库中创建两个真实Git worktree，FE01/FE02分别提交合成结果、并发更新本线状态，
随后协调者显式读取两个来源。另一条线与全局字节在分线更新期间保持不变，两个结果都被保留。
错线更新、从科研分支使用coordinator、无效路径/fullSHA/目录归属、生成视图过期、
新科研变更未更新状态、伪造NEXT_TASK授权及越界staged文件均被测试覆盖。
待办中放入sbatch/touch文本不会被解释执行；用户未跟踪文件保持不变。

原14项回归加本轮11项，共25项通过。编译和diff空白检查通过。
memory check为0 errors；FE01_A01、FE01_ON、V01的三个提示是现有历史来源/本地ref差异，
不属于本轮路由故障，也不据此改写其科研状态。

## 保护范围与部署边界

[preservation.json](preservation.json) 记录原FE01、FE02和协调worktree的branch、HEAD、
status、tracked diff与既有untracked字节一致。模型、实验配置、launcher、历史reports与结果
相对实现base没有修改；不自动迁移任何已锁HPC源码。本轮HPC提交/模型运行均为0。

按用户要求只在独立分支实施和发布。审阅后需各实验分支一次性引入维护变更，旧分支AGENTS
才会自动指向START；具体模板见 [HANDOFF_WORKFLOW](../HANDOFF_WORKFLOW.md)。
维护工具可能改变旧训练代码扫描tools所得的身份，不能复制进正在恢复的原HPC目录。
ChatGPT Project Instructions/连接授权也需用户一次设置；普通聊天仍需入口链接与实验线。
这些部署边界明确保留，不将模拟PASS写成原worktree或账户设置已部署。
