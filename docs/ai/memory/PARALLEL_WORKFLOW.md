# 并行开发与记忆隔离

初始化先由协调者审计/实现/测试/提交全局记忆。**公共改动经过审阅后**，再引入FE01、FE02各工作分支；
本轮不自动传播、不合并科研分支。FE01与FE02互不等待，也不需要停止对方超算任务。

## Git与所有权

本轮协调分支 `chore/project-memory-bootstrap` 基于两条FE线共同祖先
`ba4fa7740d9fde5fde87a7b0ae0397037209baf4`，只新增维护工具、测试和文档。
FE01实际原分支HEAD仍为该SHA；FE02是 `42b3d5405d9fe9672f0a0925275ca4f7e9e6a9e7`。
FE01后续A01/ON另有较新远端HEAD，见CURRENT_STATUS，不能用原FE01头覆盖它们。

审阅通过后，在已有worktree里逐线引入本轮文档提交；若分支已有未提交工作则另建干净worktree。
以下是人工操作模板，不能在运行中的来源目录上直接执行：

```bash
git fetch origin
git worktree add -b memory/fe01-adoption ../fe01-memory-adoption origin/exp/fe01-feature-extractor-site-effects
git -C ../fe01-memory-adoption cherry-pick <已审阅的公共记忆提交完整SHA>
git worktree add -b memory/fe02-adoption ../fe02-memory-adoption origin/exp/fe02-event-fusion-readout
git -C ../fe02-memory-adoption cherry-pick <已审阅的公共记忆提交完整SHA>
```

逐个检查文档冲突并保留新旧来源；不使用ours/theirs一键覆盖。通过审阅后由各线正常合并adoption分支。
所有长期实例的实际科研分支应与其record.branch一致；更新工具会拒绝在其他分支执行本线写入。
每实例只暂存 `docs/ai/memory/tracks/<ID>/`；FE01/FE02实例无权改全局，A01/ON也独立选ID。
全局文档/observations/registry由单一协调者更新。公共代码/工具的修改需要单独授权任务，不能混入纯状态更新。

协调者从显式ref的固定commit用git show读取本线record，记录source_commit；不会checkout别人的分支。
旧分支没安装记录、fetch失败或来源损坏时保留上次已核验记录，标source_unavailable_retained并报警；
不会把未同步当未完成。各线可随时继续独立提交。核验时间不会因生成摘要自动更新。
生成区外人工内容保留，历史决定/实验ID不允许直接删除，应保留条目并记录superseded。

## 实验产物命名与实际审计

| 写入类型 | FE01 | FE02 | 本轮核验/约束 |
|---|---|---|---|
| checkpoint/评估/audit | FE01_OUTPUT_ROOT下run_id目录 | FE02_OUTPUT_ROOT下run_id目录 | 共享engine不等于共享输出；两根必须无包含关系 |
| 作业/日志 | FE01_OUTPUT_ROOT/slurm | FE02_OUTPUT_ROOT/slurm，launches | job_id/array/rank/run_id保留；不同根 |
| 权重注册清单 | FE01_WEIGHTS_ROOT/pretrained_manifest.json | FE02_WEIGHTS_ROOT/pretrained_manifest.json及register.lock | 权重张量可只读共享，清单和锁必须独立 |
| 缓存 | FE01_OUTPUT_ROOT/.cache | FE02_OUTPUT_ROOT/cache/<类型>/<job_id> | 两根隔离；FE01基线未保证所有同线rank cache独立 |
| 临时文件 | 本线新session/tmp/<job>/<rank> | 本线新session/tmp/<job>/<rank> | 原FE02 launch env已mktemp；TMPDIR需部署环境显式设置 |
| 实验ID | formal__<family>__seed等旧ID | fe02__<stage>__<variant>__seed | 原ID保持；全局索引用FE01-/FE02-前缀 |
| A01与ON | 独立A01_OUTPUT_ROOT、session/revision；ON另用新run ID | 不写入 | 不覆盖旧seal/plan/源码/评价产物 |

具体原代码入口已由FE01/FE02记录引用：FE01 cluster_common的cache是本线共享根；
FE02 job.sh在run_id上flock，并在weights-root下注册清单；该锁不能防止两条线共用同一个weights-root。
因此部署前必须对**真实解析后的路径**做只读检查，不能因为变量名称不同就认证无覆盖。
本轮没有读取私有env/HPC目录，实际路径隔离为 **deployment_not_verified**，不宣称已规范化在跑作业。
新任务的环境设置示例（不回写既有已锁任务）：

```bash
export TMPDIR="$FE02_OUTPUT_ROOT/tmp/${SLURM_JOB_ID:-local}/${RANK:-0}"
# 仅在对应实际任务准备阶段创建本线TMPDIR；记忆工具本身不创建这些目录
```

使用实际路径生成不入Git的资源表，再运行 `python tools/ai_memory_check.py --resources <表路径>`：

```json
[
  {"track_id":"FE01","write_paths":["/runs/fe01","/weights/manifests/fe01","/tmp/fe01"],"readonly_paths":["/datasets/japan","/weights/shared_encoder.pth"]},
  {"track_id":"FE02","write_paths":["/runs/fe02","/weights/manifests/fe02","/tmp/fe02"],"readonly_paths":["/datasets/japan","/weights/shared_encoder.pth"]}
]
```

检查比较同路径/祖先目录与symlink解析，禁止跨线write/write和write/read冲突；共享只读数据允许。
未展开环境变量或相对路径报错。列出全部可写cache/tmp/清单，不只列checkpoint目录。
该检查只读，不载入env脚本、不mkdir、不锁住或终止实验作业。

## 共享代码与合并

两个分支可以分别提交同一基础代码。协调者先运行：

```bash
python tools/ai_memory_check.py --merge-risk <FE01-ref> <FE02-ref>
```

输出相同merge-base、两方都修改的文件、架构审阅标记；这是保守风险检测，不宣称所有列出的修改必然文本冲突。
工具不执行merge。架构变更单独审阅，逐一核查checkpoint兼容、采样/前向/身份锁与各自真实测试；
在独立integration worktree合并并运行必要回归，绝不自动选一方。生产已锁source不随公共文档迁移。

## 故障恢复

- 移动目录导致旧worktree的.git指向旧绝对路径：先备份状态，从主仓库执行`git worktree repair <现有目录>`并只读核对HEAD/status；本轮未修复或prune这些旧注册。
- 协调者中断：逐文件原子写入可能留下生成区/快照不一致，先check报警，再从仍保留的observations与固定refs重新生成；不重写科研状态。
- 发现 `.git/ai-memory-coordinator.lock`：查看记录PID/时间并确认无活跃协调者，才由人工移走陈旧锁；不能按超时自动删除。不同track使用不同锁。
- 独立line未同步：继续该line的本地任务；协调者恢复fetch后显式指定source重读。不可用source不抹掉旧状态。
- 报告被新证据纠正：追加来源/决策，原条目标注superseded；无法确认的历史理由保留needs_history_review。
- 维护脚本故障：仅恢复本任务文档/工具提交，不reset工作分支、不删除产物、不重签旧科研锁。
