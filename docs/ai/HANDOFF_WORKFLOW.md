# 接手机制的部署、收尾与恢复

本轮基于已有 `chore/project-memory-bootstrap` 的
`826acaadc89979d6ad324d014123def8353a429c`，在独立
`chore/persistent-ai-handoff` 扩展入口与检查，不合并 FE02 科研实现。
本线事实分别核验于 FE01 `577a0cec43643597a61209209a770aeecf5fefa0` 与
FE02 `618661f99afb67f5c23c9f1a3fd713a0a156e7b5`。

## 引入各实验分支

用户审阅本轮维护提交后，由各线在单独、干净的 adoption worktree 引入；
不切换/清理正在工作的原 worktree，不批量覆盖文件。以下是人工部署模板，不是自动接手动作：

```bash
git fetch origin
git worktree add -b handoff/fe01-adoption <新FE01部署目录> origin/exp/fe01-feature-extractor-site-effects
git -C <新FE01部署目录> merge --no-commit --no-ff origin/chore/persistent-ai-handoff
# 审阅冲突和diff；科研源码/config/历史报告应无来自本任务的改动
# 逐项保留本线较新证据，不能用 ours/theirs 静默覆盖
```

FE02 使用另一个新目录和 `handoff/fe02-adoption`，base 换为其科研分支。
各线按正常审阅流程完成合并；不能将 FE01 与 FE02 科研分支互相合并来安装入口。
adoption 分支上明确 `--track` 只能只读恢复，写入器要求最终对应的科研分支名称。
引入后用 `python tools/ai_handoff.py` 核对自动识别；在科研分支上检查
`python tools/ai_memory_check.py`。全局缓存仍为协调者快照，本线 own record 才是本线写入来源。

## 每轮结果与状态分开提交

`record.commit_sha` 固定最近核验的科研/证据来源。
`handoff.last_session.result_commit` 固定本轮已完成代码/证据结果；
`scope=handoff_infrastructure` 允许它指向协调分支的维护提交，不冒充新训练结果。
承载这些状态文件的 commit 由 Git 自身记录，不能在同一文件中自引用。
最少两阶段：工作结果提交 R → record/视图更新 → 状态发布提交 S。
复核时同时提供 R 和 S。未完成的任务保持未完成，并明确恢复步骤。

状态提交只暂存 `docs/ai/memory/tracks/<ID>/`。`--staged-only` 会检查自指定 result
commit 之后的已提交路径及当前 staged 路径，忽略用户未暂存/未跟踪文件；
它不会因此删除或接管那些文件。普通 `--scope` 则保留原先审计整个工作树的严格模式。

## 协调者更新

```bash
# 只有协调分支可执行；来源refs须已按正常Git流程更新
python tools/ai_memory_update.py --coordinator \
  --source FE01=origin/exp/fe01-feature-extractor-site-effects \
  --source FE02=origin/exp/fe02-event-fusion-readout --dry-run
```

审查计划后才去掉 dry-run。来源尚未安装 record、缺 commit 或校验失败时，
工具报警并保留上一份已核验事实，不把状态抹成 UNKNOWN。
协调同步只更新 observations、snapshot 和全局视图，不覆写任何 owned track record。
因此协调分支上历史 own-track 副本也有日期：已部署的实验分支应通过源码 ref 读取其最新记录，
不能把复制的旧状态当最新。初次安装的两线记录由本任务在协调分支一次性核验初始化。

## 故障恢复

- 分支不匹配、字段缺失、full SHA 不可验证、路径越界或生成视图过期：保持只读，
  报告精确错误；检查 fetch/部署和本线 record，禁止跳过 provenance 或重签历史 lock。
- 在临时目录生成新 record 后，先验证候选再交给 update 的 `--record`；不要先覆盖原 record
  才期待工具恢复历史。生成中断后以保留的 record 重生成本线视图，区外人工说明会保留。
- 同线锁冲突：核验锁内 PID、时间与活跃进程；仅确认陈旧后人工移走，工具不按超时自动删锁。
- 科研 HEAD 新于已记录来源：帮助程序报告 needs_status_review；先检查新增差异，
  不用旧 NEXT_TASK 自动覆盖它。不要用当前 HEAD 时间推测 HPC 完成状态。
- 需要回滚：在维护/adoption 分支正常 revert 本任务的维护提交，保留原模型、结果和用户修改；
  不 reset 原科研 worktree，不清除产物。

GitHub 上的固定入口需要分支被持续保留；重命名入口分支时由协调者更新 registry、链接和项目设置。
ChatGPT Project 的 Instructions/仓库授权需用户一次性配置，普通聊天不会自动加载 GitHub 文件。
四种接手验收是程序化读取模拟，不等同于四个独立模型会话实际启动。
