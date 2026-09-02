# ChatGPT Project 与 Codex 协作约定

更新日期：2026-09-02

## 1. 协作边界

这个流程把 GitHub 作为共享的、可追溯的事实来源，但不假设两个 AI 之间存在自动通信。

- 用户：决定研究目标、实验优先级和是否执行有成本或有风险的操作。
- ChatGPT Project：通过 GitHub 插件/连接器阅读指定分支，负责独立分析、方案审阅、文献化解释和提交前复核。
- Codex：在本地工作区检查真实文件，实施修改，运行可执行的验证，提交并推送 Git 变更。
- 超算：保存大数据、权重、训练日志和正式评估产物；GitHub 通常只保存代码、配置、启动器、测试和轻量文档。

GitHub 连接器是否能写 issue、PR 或代码取决于实际安装项和权限。除非工具明确返回成功，ChatGPT 不应声称已经修改或推送仓库。最稳妥的默认方式是：ChatGPT 生成交接块，用户转交给 Codex；Codex 推送后，ChatGPT 按精确 commit/PR 复核。

## 2. 每轮工作的固定流程

1. ChatGPT 确认 `rular099/team_pytorch`、目标分支和当前 commit。
2. ChatGPT 阅读 `AGENTS.md`、`docs/ai/PROJECT_CONTEXT.md` 和与任务直接相关的代码、配置、测试及 provenance。
3. ChatGPT 区分事实、推断和建议，并输出 `AI-HANDOFF`。
4. 用户把交接块发送给 Codex；若交接会改变实验定义，用户先作决定。
5. Codex 在同一 base commit 上实施；若 HEAD 已变化，先报告差异，不机械套用旧方案。
6. Codex 运行本地验证、提交、推送，并输出 `CODEX-RESULT`。
7. ChatGPT 读取新 commit 或 PR diff，检查实现是否满足验收条件和 RT55 兼容性。

一个对话只处理一个明确产出。架构讨论、实现、超算运行和结果解释可以分别开聊，但都使用同一项目文档和精确 commit。

## 3. 事实来源优先级

发生冲突时按以下顺序处理，并显式报告冲突：

1. 当前分支上的代码、resolved config、测试和 checkpoint/result provenance；
2. `docs/ai/PROJECT_CONTEXT.md` 的带日期快照；
3. `SESSION_SUMMARY.md` 和专题文档；
4. 聊天中的用户补充；
5. 模型记忆或未核验推断。

文件名和输出目录名不是 checkpoint 身份证据。应检查 checkpoint 内 epoch/loss、评估 split、resolved config 和 metrics/NPZ 的来源。

## 4. ChatGPT 交给 Codex 的格式

```text
[AI-HANDOFF]
task_id: YYYYMMDD-short-name
repo: rular099/team_pytorch
branch: <target branch>
base_commit: <full SHA>
goal: <一个可验收的目标>
verified_facts:
  - <事实 + 文件/行号或结果来源>
inferences:
  - <明确标为推断>
files_to_inspect:
  - <path>
constraints:
  - preserve RT55 loading and inference compatibility
proposed_change:
  - <文件级修改；若只是诊断则写 none>
acceptance_checks:
  - <测试、指标或行为>
hpc_followup:
  - <需要超算执行的命令/产物；没有则写 none>
risks:
  - <兼容性、泄漏、算力、统计风险>
open_questions:
  - <必须由用户决定的问题；没有则写 none>
[/AI-HANDOFF]
```

如果只是原因分析，`proposed_change` 必须写 `none`，不要把建议伪装成已实现代码。

## 5. Codex 返回的格式

```text
[CODEX-RESULT]
task_id: <same task_id>
base_commit: <SHA before work>
result_commit: <full SHA, or none>
branch: <branch>
changed_files:
  - <path + change summary>
verification:
  - <command>: <pass/fail/not run + reason>
compatibility:
  - <RT55 regression evidence>
hpc_status:
  - <not submitted/submitted/job id/completed>
remaining_risks:
  - <known limitation>
review_request:
  - <what ChatGPT should review at result_commit>
[/CODEX-RESULT]
```

## 6. 避免冲突

- 不让 ChatGPT 和 Codex 同时改同一文件；先冻结交接所依据的 base commit。
- ChatGPT 提出的 patch 必须视为建议，Codex 应结合当前工作树重新核验。
- Codex 只暂存本任务文件，不顺带提交用户已有的未跟踪结果或脚本。
- 重要实验变更使用新 config、新输出目录和新实验标识，不回写 RT55 原始配置或结果。
- 推送后用 commit SHA 复核，不用“最新版”“刚才那版”等相对描述。

## 7. GitHub Project 初次连接检查

- 授权仓库：`rular099/team_pytorch`。
- 活跃开发分支：以用户指定为准；本快照使用 `zhangb/native-scale-adapter-scaling`，不要默认读取 `team_collab_baseline`。
- 在第一个 Project 对话里要求 ChatGPT 回报它实际读到的 branch、commit，以及 `AGENTS.md` 首个标题。
- 若连接器只能看到默认分支，应先通过正常 PR/合并流程把这些协作文档带到默认分支，不能假装已经读取当前实验分支。
