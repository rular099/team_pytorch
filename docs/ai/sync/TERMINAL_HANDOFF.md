# 多终端恢复与ChatGPT分析入口

同步日期：2026-11-02；repository：rular099/team_pytorch；总入口分支：`chore/workspace-sync`。
本轮不是新模型研发。已按内容筛选并发布三个独立实验线快照；原科研分支、原工作区与原产物保持。
总入口自身的精确SHA见交付回复或本checkout的git rev-parse HEAD；下面各线均固定完整SHA。

## 本次同步点

| 工作线 | 新机器目录 | 使用分支 | 精确提交 |
|---|---|---|---|
| RT55_RT56 | team_pytorch | `sync/rt55-rt56` | `bc648c75f4cb5f4d04bf10fe5a80d12745b20244` |
| FE01 | team_pytorch_fe01 | `sync/fe01` | `c6ee2ab195661b26adcb6f8b7cd068bf265985ea` |
| V01 | team_pytorch_query_geometry_diagnostics | `sync/v01` | `52ea935249e490033f49209e106eecdedaa9275a` |
| FE02 | team_pytorch_fe02 | `exp/fe02-event-fusion-readout` | `42b3d5405d9fe9672f0a0925275ca4f7e9e6a9e7` |
| FE01_A01 | team_pytorch_fe01_a01 | `exp/fe01-a01-absolute-amplitude-ablation` | `29ffb20ba0effc0c345a0329ba3f8c658dc8cef0` |
| FE01_ON | team_pytorch_fe01_on | `exp/fe01-on-warning-map-evaluation` | `50672e131d876a604432b283162d5b0189d40cf0` |

FE01_A01本地仍为236f72797ef4ccdec584cfbba952987e64a26437，远端已为29ffb20ba0effc0c345a0329ba3f8c658dc8cef0。
它不是未提交内容，不应把旧本地HEAD推回覆盖远端。恢复时优先采用上表较新结果版本。
FE01 ON原本没有本地worktree；现在可按上表单独建立。FE02没有本轮代码增量，直接复用既有远端提交。
机器可读对应关系见 [terminal_manifest.json](terminal_manifest.json)，原本地状态见
[local_worktrees_20261102.json](local_worktrees_20261102.json)。

## 换机器后从这里开始

在新的空工作父目录中克隆总入口，目录名保持以下示例可直接匹配本地产物清单：

```bash
git clone --branch chore/workspace-sync https://github.com/rular099/team_pytorch.git team_pytorch_memory
cd team_pytorch_memory
git rev-parse HEAD
# 核对总入口SHA和本页，再按需建立工作区（不是要求把全部线都启动）
git worktree add -b resume/rt55_rt56 ../team_pytorch bc648c75f4cb5f4d04bf10fe5a80d12745b20244
git worktree add -b resume/fe01 ../team_pytorch_fe01 c6ee2ab195661b26adcb6f8b7cd068bf265985ea
git worktree add -b resume/v01 ../team_pytorch_query_geometry_diagnostics 52ea935249e490033f49209e106eecdedaa9275a
git worktree add -b resume/fe02 ../team_pytorch_fe02 42b3d5405d9fe9672f0a0925275ca4f7e9e6a9e7
git worktree add -b resume/fe01_a01 ../team_pytorch_fe01_a01 29ffb20ba0effc0c345a0329ba3f8c658dc8cef0
git worktree add -b resume/fe01_on ../team_pytorch_fe01_on 50672e131d876a604432b283162d5b0189d40cf0
```

resume/*是新终端的独立本地工作分支，避免误覆盖原有科研分支；恢复完成后按各线正常审阅/提交流程继续。
这些命令仅准备工作区，不训练、不sbatch、不重签旧lock。若已经通过U盘复制整个仓库，先保留现有目录，
核对git worktree list；路径迁移问题使用git worktree repair修复关联，不要重建或覆盖已有工作区。

Git已提供代码与精选报告，但真正加载数据/权重前仍需恢复下一节的产物与私有env，按各线既有身份门核验。
FE02续跑、A01评价、ON inventory等沿用原协议和真实回执，不因同步完成就自动提交计算。

## 本地产物与U盘恢复

[local_artifacts/INDEX.json](local_artifacts/INDEX.json) 汇总2667项可见本地保留文件的清单。
合计约2.30GB是逐路径计数，含重复归档与派生输出，**不是唯一数据量，也不是已备份数据量**。
每条包含原工作父目录相对路径、字节数、SHA-256、来源branch/commit、恢复角色。
环境/bytecode/cache不作为科研备份；另一个项目TEAM、旧镜像team_pytorch2及私有配置没有混进本Git仓库。

当前chaosuan_res是指向旧机器移动硬盘挂载点的失效符号链接。其目录内容/权重/新结果本轮无法读取，
因此标记NO_SOURCE_HASH_AVAILABLE；不能推测这些大文件已经随U盘复制成功。新机器需挂载实际结果根，
用本线既有环境变量或相对链接定位它，并基于真实可读来源建立/核验清单。

可见文件清单可用于验证某个实际USB副本或恢复后的工作父目录：

```bash
# root应是含team_pytorch、team_pytorch_fe01等目录的父目录；只读校验，不复制/删除
python tools/workspace_sync.py verify-artifacts --root /实际USB工作父目录 --manifest docs/ai/sync/local_artifacts/team_pytorch_fe01.json
python tools/workspace_sync.py verify-artifacts --root /恢复后的工作父目录 --manifest docs/ai/sync/local_artifacts/team_pytorch_fe01.json
```

返回码0表示该清单下实际字节全部匹配；2表示缺失/不匹配/无来源hash，不能当作成功。
workspace_external.json会继续提示chaosuan_res来源hash缺失，直至独立补齐来源证据。
本轮只在当前源文件上计算hash，**没有访问或核验新的USB目标，也没有执行任何复制/clean/reset/delete**。

从USB复制时只恢复清单中的本地文件，不把整个旧worktree盖到新Git checkout上；否则会把已更新代码或
路径变量化的公开副本覆盖成旧版。可以先将各清单files的workspace_relative_path合并为一个去重路径表，
再使用rsync --files-from并先dry-run；保留现有文件，碰到不同hash先单独比较，不自动覆盖。
可再生成的图/PPT或旧归档可按当前任务需要选择恢复，不必为阅读ChatGPT报告恢复全部文件。

## 给ChatGPT看哪些文件

先读本分支的项目记忆三入口，再按研究问题选择固定commit下的材料：

| 主题 | 证据入口 | 边界 |
|---|---|---|
| RT55/RT56完整旧报告 | [RT55正文](https://github.com/rular099/team_pytorch/blob/bc648c75f4cb5f4d04bf10fe5a80d12745b20244/reports/rt55_rt56_current_results/rt55/REPORT_CN.md)、[RT56正文](https://github.com/rular099/team_pytorch/blob/bc648c75f4cb5f4d04bf10fe5a80d12745b20244/reports/rt55_rt56_current_results/rt56/REPORT_CN.md) | 2026年8–9月既有证据，21张正文PNG/聚合表/formal metrics已同步；非新运行 |
| FE01原始任务与幅值设计依据 | [FE01同步范围](https://github.com/rular099/team_pytorch/blob/c6ee2ab195661b26adcb6f8b7cd068bf265985ea/docs/ai/SYNC_SCOPE_20261102.md) | 原九组/EVAL1结果此前已入Git；本轮补4份历史任务/交接，不把旧计划当未完成 |
| 九月交流报告与技术说明 | [修订版技术说明](https://github.com/rular099/team_pytorch/blob/52ea935249e490033f49209e106eecdedaa9275a/reports/technical_exchange/multistation_pga_exchange_20260930_v2/TECHNICAL_DETAILS.md)及同目录speech、sources、assets/tables | v1和v2讲稿/复现代码/源表都保留，PPT/PDF/渲染成品走本地/U盘 |
| RT59小震预警诊断 | [既有离线报告](https://github.com/rular099/team_pytorch/blob/52ea935249e490033f49209e106eecdedaa9275a/reports/rt59_small_event_warning_20260928/REPORT.md) | validation、23个M=3.0事件，不能外推M<3或台站数因果效应 |
| A01、ON、FE02、V01当前科学进度 | 上表精确分支/commit及 [CURRENT_STATUS](../memory/CURRENT_STATUS.md) | 不因这次同步宣称新epoch、test、场地泛化或生产回放成功 |

大逐目标CSV/NPZ未上传；关键聚合/分层/epoch/split/样本量保留，不能根据缺少大文件补造细节。
原目录README中的PDF/PPT等本地文件链接可能不在Git子集中，具体以SYNC_SCOPE与总产物清单为准。
公开文档中的私有路径已变量化，原始文件与公开副本SHA分别记录在
[published_sources_20261102.json](published_sources_20261102.json)，不把脱敏副本与旧seal误比较。

## 下次离开机器前

先git status，按 [同步策略](README.md) 只暂存本轮代码/任务记录/聚合结果，再运行staged检查并提交/正常push。
协调者把每线精确SHA及新增本地产物清单更新到总索引；有重要科研决定时继续维护对应实验线记忆。
fetch后如两终端分叉，先审查差异，不用force解决。代码/配置的共享修改照常单独审阅；本轮不合并科研分支。

验证记录见 [VALIDATION_20261102.json](VALIDATION_20261102.json)。本机缺NumPy，原演示报告数值测试导入失败、断言未执行；
报告脚本已完成语法检查，传输工具与记忆工具用标准库临时仓库测试。没有安装大环境或重新执行历史科研计算。

原RT55/RT56的qa/artifact_manifest.csv存在7项原始字节数过时记录（包括其自身及散点图文件），详见VALIDATION中的historic_qa_manifest。
本轮保留原记录并用当前源文件/公开副本SHA区分，不把旧清单当新QA通过，也不据此推断指标错误。
