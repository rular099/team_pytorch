# A01 评价完成状态与 pack 入口检查（2026-10-15）

基线 `f29ecdcb4dcc4bdce9eaf560bcbc9453c2b4506e`；分支
`exp/fe01-a01-absolute-amplitude-ablation`。

用户报告 TEAM 修复提交入口执行成功，本地尚未取得本批实际评价结果。
新增只读登录节点检查脚本，逐项复用原 pack 的完整 result_gate，兼容 TEAM 新来源与其余旧来源。
只按完整成功 seal、两个 epoch 的必需产物、比较 PASS、来源和文件 SHA256 判断完成。
结合最近提交记录及 squeue/sacct 区分排队、运行、失败、未知，输出 4 和 pack 的手动下一步。
pack 已提交时不会建议重复提交；归档与 SHA256 匹配才提示下载。查询失败不等于完成。

37 项本地测试通过，覆盖旧 0 失败后重提、部分输出、作业完成但缺 seal、来源混合、
产物篡改、pack 排队/运行/失败、归档不完整、调度不可用、登录节点禁止导入 torch、
不写 session/bytecode 和不提交作业；Python 3.9 AST 与 bash -n 通过。
A01 认证核心 SHA 保持不变，三个受保护工作树的 HEAD/branch/status/diff/staged 快照一致。
测试使用明确的合成数据及假调度器，不是新的超算完成记录或 Japan 指标。

交付包和 SHA 见 verification.json；操作见 [状态说明](../../docs/ai/FE01_A01_EVAL_STATUS.md)。
未修改旧 session/plan/tools/env/seal/权重，不登记新 revision，不改模型/训练/评价算法。
真实五项完成情况需要用户在超算运行检查；本轮未提交真实 sbatch/srun/训练/eval/pack，未访问 test。
