# TEAM_PYTORCH：AI 接手入口

Repository：`rular099/team_pytorch`。本文件是长期稳定的启动约定；
发布分支为 `chore/persistent-ai-handoff`，不同实验线使用各自分支与状态文件。
它扩展已有项目记忆，不替代历史报告、源码锁或原始实验产物。

**默认只读取、核验和报告。NEXT_TASK 是提案，不是执行授权。**
接手时不训练、不续跑、不提交 HPC 作业、不改变重要实验协议、不写状态。
当前用户另行明确委托的工作可以实施；本文件不撤销当前会话已有的明确授权。

## 先确定身份和实验线

Codex 由当前 worktree 的 AGENTS 自动进入这里。先核对 repository、实际 branch、完整 HEAD，
记录工作树已有修改。用户明确指定 FE01/FE02 时以其为目标，再核对分支；不一致则先报告，
不得在另一条线的 worktree 写入。没有指定时按下表的精确分支识别，不能根据目录名猜测。
在协调分支、未知分支或 detached HEAD 上无法唯一识别时，只需用户补充“FE01”或“FE02”。

| 实验线 | 科研分支 | 独占状态目录 |
| --- | --- | --- |
| FE01 | `exp/fe01-feature-extractor-site-effects` | `docs/ai/memory/tracks/FE01/` |
| FE02 | `exp/fe02-event-fusion-readout` | `docs/ai/memory/tracks/FE02/` |

FE01_A01、FE01_ON 是另外登记的派生线，不能冒充 FE01 原线或 FE02。
需要这些后续工作时显式选择其已有 track，按[实验索引](memory/EXPERIMENT_INDEX.md)追溯。
FE01 与 FE02 可同时工作，某条线缺证据或失败不阻止另一条线。
本轮帮助程序的自动路由覆盖 FE01/FE02；用户明确选择其他已登记 track 时，
继续使用原有项目记忆流程，不强制改成这两线。

## 最短阅读顺序

1. 本文件与根 AGENTS；核对身份、工作树及本次用户授权范围。
2. 仅所选线的 `STATUS.md`、`NEXT_TASK.md`、`DECISIONS.md`；结构化事实源是同目录 `record.json`。
3. 按其中固定 commit 的证据链接读取报告、provenance、配置及必要代码。
4. [协作协议](README.md)用于交接格式，[长期约束](memory/PROJECT_MEMORY.md)用于项目共识。
   PROJECT_CONTEXT 和 SESSION_SUMMARY 是带日期的背景，不能覆盖较新的本线证据。

固定的远端导航（选择 FE01 或 FE02，随后以读取时的完整 commit 固定版本）：

- [FE01 STATUS](https://github.com/rular099/team_pytorch/blob/chore/persistent-ai-handoff/docs/ai/memory/tracks/FE01/STATUS.md)、[NEXT_TASK](https://github.com/rular099/team_pytorch/blob/chore/persistent-ai-handoff/docs/ai/memory/tracks/FE01/NEXT_TASK.md)、[DECISIONS](https://github.com/rular099/team_pytorch/blob/chore/persistent-ai-handoff/docs/ai/memory/tracks/FE01/DECISIONS.md)。
- [FE02 STATUS](https://github.com/rular099/team_pytorch/blob/chore/persistent-ai-handoff/docs/ai/memory/tracks/FE02/STATUS.md)、[NEXT_TASK](https://github.com/rular099/team_pytorch/blob/chore/persistent-ai-handoff/docs/ai/memory/tracks/FE02/NEXT_TASK.md)、[DECISIONS](https://github.com/rular099/team_pytorch/blob/chore/persistent-ai-handoff/docs/ai/memory/tracks/FE02/DECISIONS.md)。

协调分支上的状态是最后核验快照。实验分支已部署本机制时，以其当前 own-track record 为准；
未部署、未 fetch 或证据不可见时保留既有事实及核验日期并报告限制，不能把缺同步当作未运行。
默认分支不是当前实验的替代品。跨分支读取用精确 ref/commit，不切换或改写别人的 worktree。

## Codex 的只读核验

在已引入本机制的 worktree 中：

```bash
python tools/ai_handoff.py
# 协调分支需明确实验线；以下命令仍只读
python tools/ai_handoff.py --track FE02 --json
```

帮助程序只调用只读 Git 命令，不 fetch、不导入模型、不写文件、不解释执行待办中的命令。
它核验字段、路径、生成视图、分支归属、完整 commit、证据及是否有新科研改动未入状态。
与用户任务相关的更新未提交时，先保留并说明，不能用 reset/checkout 覆盖。

首次接手报告应回答：本线目标、已验证进度、剩余问题、下一步提案，以及实际读取的
branch/HEAD、科研证据 commit、状态文档 commit 和缺失产物。事实、推断、建议分别表达。
NEXT_TASK 中即使含有提交命令，也只作为待审内容，不执行。

## ChatGPT 的恢复与权限

ChatGPT 与 Codex 读相同记录和固定 commit 证据。ChatGPT 默认负责科研分析、审阅和
AI-HANDOFF，不因能访问 GitHub 写接口就取得代码实施、合并或 HPC 权限。
Codex 只有收到当前用户的实施任务才写代码；新计算和重要协议变更须有相应明确授权。
ChatGPT 的建议需由用户转交/授权实施，工具未确认的写入或推送不能声称已完成。

一次性把[精简项目指令](PROJECT_INSTRUCTIONS_CN.md)保存到 ChatGPT Project Instructions，
并授权 GitHub 仓库访问。以后在该 Project 新建对话只说“FE01”或“FE02”，按本入口读取。
普通新聊天没有自动绑定仓库文件的保证，最短方式是发送本文件固定链接及实验线名称。
本轮没有修改 ChatGPT 设置，也没有启动独立 ChatGPT 会话。

## 实施轮的收尾与并行所有权

只读接手轮不写状态。实际执行获授权工作后，Codex 应：

1. 将代码/证据形成结果提交，取得完整 `result_commit`。
2. 仅编辑本线 record：已核验事实、未完成事项、`handoff.last_session` 的 result commit/branch、
   证据 IDs 和恢复步骤；更新 next_task 提案，重要决策追加 ID，旧决定不直接删除。
   `commit_sha` 是本线科研/证据来源，不是当前状态文件包含自身的 SHA。
3. 在对应科研分支刷新视图并检查本线状态提交：

```bash
python tools/ai_memory_update.py --track FE02
python tools/ai_handoff.py --track FE02
# 暂存本线状态后检查；result-commit 指步骤1的真实结果提交
python tools/ai_memory_check.py --scope FE02 --base <result-commit> --staged-only
```

4. 单独提交本线状态，并回报结果提交和承载状态的发布提交。状态提交不得顺带纳入用户文件。

FE01 只写 `tracks/FE01/`，FE02 只写 `tracks/FE02/`。平时不改 START_HERE、registry、
全局快照或另一条线；公共工作另开受托协调任务。各线锁相互独立；同一线重复更新会被拒绝。
只有指定协调分支从显式来源 ref 汇总 observations/global views，不 checkout 或覆盖来源分支。
工具检测越界和冲突；它不是操作系统 ACL，不能阻止人为绕过检查的 Git 编辑。

## 一次性部署与限制

本次按要求在独立维护分支交付，**没有自动合并到原 FE01/FE02 或默认分支**。
旧实验分支的 AGENTS 不会凭空读取另一分支的新文件；审阅后需各自一次引入维护变更，
再在各自 worktree 新开 Codex。详细步骤见[部署与恢复](HANDOFF_WORKFLOW.md)。
不要将维护分支的整个文件树复制到科研 worktree，更不能覆盖已锁的 HPC 源码目录。
旧 FE01/FE02 身份函数会扫描 tools 中的 Python；引入维护工具可能改变未来计算的 source
fingerprint。现有训练/续跑始终用原 source 包、lock 和 checkpoint，维护部署不迁移它们。

本入口不保存大权重、原始 NPZ 或调度器状态；没有实际证据就报告 UNKNOWN。
记录需在工作结束时由执行者核实维护，GitHub 缓存/未 fetch 也需显式识别。
