# A01 TEAM OFF eval身份误判与不中断其它任务的修复

仓库rular099/team_pytorch；分支exp/fe01-a01-absolute-amplitude-ablation；
基线e1ad3bf547392738ee5e120b7d4d33f8e42d7d30。
证据为用户最后订正的traceback：eval_first的0在checkpoint_metadata组合检查退出，
StepId30424637.0，节点a02r4n08；用户报告1/2/3正常运行。
没有本批完整日志、sacct或checkpoint文件；不将“运行中”提升为“评价完成”。

代码核实TEAM scratch encoder可训练，原a01_checkpoint只对非TEAM要求冻结encoder
SHA等于audit初态。新增运营检查却无条件比较，因此合法训练后TEAM会被拒绝。
组合条件还含公共初始化SHA；本批trace不能独立证明其实际值正确，修复保留该项独立检查。
timm FutureWarning及srun ROUTE行不是此Python异常的触发位置。

修复[metadata检查](../../scripts/fe01_a01_ops/eval_pack.py)：
TEAM改为验证训练后实际encoder张量SHA等于checkpoint记录，公共初始化仍等于audit；
三种预训练冻结家族继续按初态及原严格loader认证。未知家族拒绝，epoch、config、
world、数据/权重/源码身份、选择checkpoint和validation协议不变。

[独立修复提交入口](../../scripts/fe01_a01_ops/submit_team_eval_fix.sh)仅重提0；
[revision登记](../../scripts/fe01_a01_ops/eval_revision.py)绑定当前已发布旧工具的精确SHA，
只新增session/operations中的revision，保持原plan、tools、private env及结果seal不变。
新0 gate记录真实新执行SHA和revision；后续原4仍可使用原脚本，新版pack逐项验证混合来源。
不会把新代码重新签成旧SHA，也不会替换正在运行任务的共享文件。

[具体操作](../../docs/ai/FE01_A01_TEAM_EVAL_FIX.md)；
[本地验证及保护证据](verification.json)。本地真实小TEAM一次前向/Adam更新改变encoder SHA，
准确复现旧检查失败，新检查及原严格loader通过；权重/初始化篡改仍失败。
84项聚焦检查通过、1项实际设备测试跳过，旧eval/helper历史字节来自基线Git对象核对。
另验证只提交0、拒绝扩展到1/2/3、旧plan/tools不变、原seal和新seal混合CPU打包及包内revision。
本轮只有本地测试和交付，真实超算重评NOT_RUN，作业NOT_SUBMITTED，held-out test NOT_ACCESSED。
