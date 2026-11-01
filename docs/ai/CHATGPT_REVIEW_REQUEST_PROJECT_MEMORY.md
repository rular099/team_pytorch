# 请审阅：项目记忆初始化与FE01/FE02并行隔离

Repository：`rular099/team_pytorch`。Branch：`chore/project-memory-bootstrap`。
实现base：`ba4fa7740d9fde5fde87a7b0ae0397037209baf4`。
已测试实现result commit：`PENDING_PUBLICATION`。
该精确SHA将在实现发布后由单独文档回执填入；回执提交只补发布身份，不改已测代码。
请回报实际读取的repo/branch/commit，比较此base与实现result的diff，不默认读取默认分支。

## 请求与阅读顺序

先只读 [PROJECT_MEMORY](memory/PROJECT_MEMORY.md)、[CURRENT_STATUS](memory/CURRENT_STATUS.md)、
[CHAT_HANDOFF](memory/CHAT_HANDOFF.md)，判断能否回答下面七问。然后按需检查：

1. [审计](memory/audit/AUDIT.md)、[远端回执](memory/audit/remote_refs.json)、[验收记录](memory/audit/acceptance.json)。
2. [并行工作流](memory/PARALLEL_WORKFLOW.md)、[使用说明](memory/README.md)、[补录模板](memory/HISTORY_IMPORT_TEMPLATE.md)。
3. [实验索引](memory/EXPERIMENT_INDEX.md)、[决策日志](memory/DECISION_LOG.md)、对应tracks下record与四份视图。
4. [CODEX结果](CODEX_RESULT_PROJECT_MEMORY_BOOTSTRAP.md)、[完整文件清单](memory/audit/changed_files.json)、三个维护脚本与14项测试。

重点审查：A01/ON的远端新结果是否正确取代本地过时进度；FE01原EVAL1与A01/ON是否分开；
FE02是否保留5/12轮和未获真实恢复证据的边界；V01工程完成是否与科学收尾分开；
RT59的点误差进展是否与空间NO_GO共存；不同协议人口是否避免强行比较。

## 恢复范围与仍缺信息

12条线：RT55_RT56、RT57、RT58、RT59、RT60、RT61、FE01、FE01_A01、FE01_ON、FE02、V01、LEGACY。
14个实验ID、14个决策ID。审计Git分支/近期提交/本地无tag以及主要结果交接/报告/配置；
远端24个branch在2026-11-01独立读取，A01/ON新HEAD本地缺对象，保留连接器回执和原文Git blob SHA。
新代码基于FE01/FE02共同祖先，只改记忆工具与文档，不把分支科研代码强行合并。

未恢复全部旧聊天原文、所有早期b系列/graph最终进度、全部历史决策候选与理由。
未重新取得大型权重/NPZ/完整journal、未核验HPC私有路径或调度退出码。
因此部分历史为unknown/needs_history_review；原报告的核验是已有报告记录，不能冒称本轮重算所有科学结果。

## 三份入口七问恢复自测

这是同一实例按三份入口作答的模拟恢复，未另请独立模型盲测，也没有读取全部旧聊天。

1. **最终目标是什么？** 用截至当前时刻的多台站波形与输入/查询位置，预测查询台站最终PGA概率分布，
   关注非输入/未触发目标、空间动态范围、校准与实时可用性；保持RT55兼容和无test泄漏。
2. **当前有哪些实验线？** 上述12条线独立记录；FE01_A01/ON是FE01后续，FE02为并行事件融合线，
   V01为速度/P前缺失控制，RT系列与LEGACY保留历史研究导航。
3. **各线做到哪一步？** RT55/56和RT57–61的对应验证报告已归档，完整科研成功不由verified_complete推断；
   FE01九组训练已归档但原EVAL1补评不完整；A01的10月21日真实封包已完成；ON真实CPU分析完成、
   10月25日inventory修复仍待生产回执；FE02已知M 5/12轮后RNG恢复失败、修复待验证；
   V01工程闭环完成而科学限定结论待独立审阅；LEGACY最终状态unknown。
4. **最新可信结论有哪些？** A01已有完整实际validation证据，不能沿用10月15日等待pack状态；
   ON旧导出CPU分析与新生产回放分开；FE02暂无完整五组结果。历史RT59点误差进展未通过空间联合门；
   V01总体收益伴随高PGA退化。这些均受原协议、validation与seed限制，非test/空间泛化认证。
5. **哪些问题未解决？** 单输入空间场压缩、任意非输入query辨识、normal/random收益保持、概率校准，
   以及各线明确的生产回执/权重归档/历史身份/在线因果性缺口。缺同步不代表失败或未完成。
6. **下一步优先做什么？** 审阅记忆体系后再引入公共文档；核对ON现有inventory恢复回执、FE02真实RNG receipt和epoch，
   按原中立请求独立分析A01完整结果并限定审阅V01收尾。不自动重训、扩大矩阵或提交超算。
7. **如何找历史证据？** EXPERIMENT_INDEX按实验ID、DECISION_LOG按决策ID，进入本线四份视图与固定commit的报告/配置/结果链接；
   Git可用时`git show <full-sha>:<path>`，本地缺新对象时先读原文blob快照并在联网后按精确SHA核验。

## 验收结果

| 验收 | 结果与证据 |
|---|---|
| A 新对话恢复 | 三入口七问已作答；CHAT_HANDOFF约1010中文字；仍需审阅者独立评价恢复准确性 |
| B 五项追溯 | seed20261101从有配置证据的线中抽取RT58、V01、RT59、FE01、RT61，逐个报告/结果/配置在完整SHA下可读并保存字节SHA256 |
| C 单线更新 | 临时仓库新结果commit后只刷新本线；其他线字节和全局文档保持，历史决定保留；手写区外内容保留 |
| 修订 并行汇总 | 两个真实临时worktree并发更新FE01/FE02，协调者显式读取两分支后两结果同时存在，CHAT_HANDOFF同时更新 |
| 修订 共享代码 | 两分支各改同一合成gemini_models.py、独立commit；互不覆盖，工具识别共同修改及架构审阅，不执行merge |
| 异常与只读 | 缺source保留旧状态；错SHA/路径/重复ID/错误状态/快照矛盾/原文篡改被检出；dry-run不写文件；资源根/symlink/write-read冲突可检出 |
| D 安全回归 | 5旧工作区branch/HEAD/status/tracked diff一致；模型/实验配置零修改、原报告零删除、HPC提交0；未追加入原未跟踪产物 |
| 工具检查 | 14项unittest PASS；py_compile/diff检查PASS；memory check 0 errors、2预期WARN：A01本地旧ref、ON本地缺ref |

精确命令、选择seed、各项原始Git文件hash、旧工作区状态hash见acceptance.json。
科学模型测试未重跑：代码、配置与加载/前向逻辑diff为零，本轮验证聚焦记忆工具与隔离合同。
实际FE01/FE02 refs当前无双方同时修改的文件（FE01头就是分叉基点），但FE02单方共享架构变更仍需审阅；
并行同文件风险已由临时仓库模拟验证。真实HPC路径未读取，不能将合成资源表PASS当部署认证。

## 日常维护与归档建议

每条线结果commit后只编辑本线record，补决定/实验/来源，运行`ai_memory_update.py --track ID`与check并提交；
协调者fetch后以`--coordinator --source ID=REF`逐线读取。未部署/不同步的分支保留既有快照并报警。
全局文件与observations只能由协调者维护。公共文档经审阅后cherry-pick到各线干净adoption分支，
有冲突逐项审查，不覆盖科研变更；具体命令与恢复流程见PARALLEL_WORKFLOW。

建议本轮审阅通过、用户保存本轮完整commit入口后，可以归档已被Git证据覆盖的聊天段落；
对于needs_history_review的关键取舍，先用补录模板保存原讨论。暂不建议一次性删除全部旧聊天。
本轮没有删除ChatGPT对话或Git历史。
