# A01 恢复入口：空日志目录与提交前诊断

仓库 `rular099/team_pytorch`，分支 `exp/fe01-a01-absolute-amplitude-ablation`，
本轮base `11f634a82de1ca925fdf1400130c30694dc535d5`。
用户报告日志目录为空，尚未给出空目录的绝对路径、实际JobID或终端输出。
不能据此推断作业已提交、已开始或已崩溃，也不把旧20261007日志当作当前证据。

原运维脚本在私有设置、旧日志扫描、preflight、mkdir、cp或sbatch失败时只向终端输出，
没有独立保存提交前错误；部分失败甚至尚未创建operations目录。这是已核验的可观测性缺口。
新版所有登录入口先在上传脚本的同级目录创建权限600的 `dcu_<模式>_<时间>_<PID>.log`，
捕获stdout/stderr、阶段与退出码，早于private/preflight/operations/sbatch。
新 `check [failed OR 0,2,4]` 仅检查原失败索引、原门控、实际目录和写权限，不调用
sbatch/srun、不创建operations目录、不导入torch或加载权重。遇到错误日志仍保留。
check默认使用最近一批原train日志中的DCU启动断言，不能猜测未提供的失败索引。

另外从源码发现并在假Slurm中复现了多节点路径错误：旧入口使用BASH_SOURCE推导自身，
在sbatch的batch暂存副本中变成节点本地路径，再由srun交给远端节点，假环境退出127。
新版提交时显式导出共享只读脚本路径 `A01_RECOVERY_LAUNCHER`，所有计算模式都使用它。
原共享副本及SHA继续记录。Slurm只负责传递batch脚本，不自动搬运其它用户文件，见
[官方sbatch说明](https://slurm.schedmd.com/sbatch.html)。
这个已复现的实现错误尚不能认定为用户“空目录”的实际原因；两类证据明确分开。

本地28项检查通过，11.82秒，只有timm弃用提示。新增检查覆盖：private缺失的早期日志、
preflight失败但operations尚不存在、check成功且零调度器调用、sbatch拒绝时保留错误、
spool路径不共享时仍能通过共享副本启动原训练。设备与调度器均为假环境。
Python3.9语法/编译（含三个嵌入程序）、bash语法、原115与A01核心SHA、原三工作区均通过。
原核心代码、配置、锁、正式训练循环、两项用户报告成功的输出都没有修改。

真实空目录原因、真实节点初始化仍为NOT_RUN/NOT_ESTABLISHED，本轮无实际超算提交。
用户应只上传新版 `dcu_recovery.sh`，先执行check并回传其提交日志；暂不再次重提训练。
无需重新environment/audit或上传权重。
使用说明见 [DCU恢复入口](../../docs/ai/FE01_A01_DCU_RECOVERY.md)。
