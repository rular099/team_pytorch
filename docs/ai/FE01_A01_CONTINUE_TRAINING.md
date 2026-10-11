# A01 DiTing 超时续训（2026-10-11）

**后续修订：数组30265280暴露了旧版Torch的RNG恢复错误及OFF启动超时。
当前提交请使用[新恢复包与操作说明](FE01_A01_RESUME_FIX.md)；以下是上一版交付的历史记录。**

基线 `be8cc1d9f240bd7da56b6ca174d78b03fa842a1e`，分支
`exp/fe01-a01-absolute-amplitude-ablation`。
本次实际日志分析见 [报告](../../reports/fe01_a01_retry_20261011/README.md)。

## 已知进度与处理

用户提供数组29953137的六份out/err，三项的16个设备探针全部PASS。
索引1 PhaseNet OFF记录12轮/2544 updates并正常关闭通信组，不再重训；未提供sacct，
这里认证的是日志进度。索引3 DiTing ON已记录5轮/1060 updates，随后24小时时限终止。
索引4 DiTing OFF没有epoch记录，出现RSMI mutex/rsmi_init失败并最终超时。
不能从混合stderr确定是哪一台设备有问题，也没有完整checkpoint/目录内容。

使用新 `continue` 入口，依据真实文件决定resume或fresh：有last.pth就续训；
仅空目录/不存在的目录才可从认证初始化开始。非空且无last.pth立即停止，绝不回退best。
完整12轮记录也会停止，不重新训练已完成项。

## 本地文件上传

将交付包 `fe01_a01_continue_20261011.zip` 上传到**超算原FE01根目录**，
与 `.a01_releases` 同级。解压后是独立 `a01_continue_20261011/` 目录，包含
`submit_diting_continue.sh`、`dcu_recovery.sh`、`continue_training.py`与SHA清单。
旧release、门控、数据、权重保持原位置，不上传新的核心源码。

## 超算登录节点提交

以下变量须由原FE01根目录的实际私有设置决定：

```bash
unzip -n fe01_a01_continue_20261011.zip
export A01_ENV_FILE="$PWD/.a01_releases/3ecf060a3334/a01.private.env"
bash a01_continue_20261011/submit_diting_continue.sh
```

入口默认仅提交3,4，一个sbatch数组、最多两项并行，共最多8节点32卡。
每项仍4节点16卡、12epochs、global batch128、microbatch8、原更新预算与采样。
使用 `A01_RECOVERY_EXCLUDE_NODES` 避开本次OFF分配的四节点；具体hostlist从本地原日志
复制到环境变量，不写入源码。以后某项已完成时，只传剩余索引，例如
`bash a01_continue_20261011/submit_diting_continue.sh 4`。

先使用stdlib读取原environment/audit/diagnostics/pilot的内容SHA与配对门控，再检查
每个训练目录。此过程不导入torch，不提交其它阶段。全部所选项通过后才调用sbatch。

## 计算节点行为

先在实际16个rank执行原环境和每rank缓存入口，检查一次真实NCCL/RCCL all-reduce，
正确sum=136并barrier。原生库若卡住，各rank由240秒timeout/15秒强制结束兜底，
Slurm step最长5分钟；失败即结束，不进入训练，不清理/dev/shm或修改驱动环境。
正式训练的第二次通信初始化也受外部父进程监控：每rank在20分钟内必须产生一个新的、
非空sample journal，旧中断journal不算进度；否则终止该进程并让Slurm结束本项。
首批采样记录出现后取消启动计时，完整epoch可以持续数小时。父进程不导入torch，
不消耗训练随机状态。`A01_STARTUP_TIMEOUT_SECONDS`可显式调整，最多一小时。

通信通过后仍调用原 `fe01_a01.engine.train`，有last时传 `resume=True`。
实际checkpoint由计算节点检查config/source/world/epoch/updates/曲线/每rank RNG等。
rank0先把旧last/best及曲线/config/lock保存在本次operation的checkpoint_before目录，
并记录SHA。大checkpoint优先用同文件系统hardlink；原训练循环的原子替换不会改旧inode。
原sample_journals和中断epoch证据保留，继续使用原引擎的恢复与提交记录逻辑。
shared flock阻止同一索引的多个新版续训同时运行。

## 时间与输出

默认只读 `scontrol show partition -o` 的分区MaxTime，申请最多72小时；分区上限较短
则申请该上限。不能识别分区时保留原24小时配置。`A01_RECOVERY_TIME`可显式指定。
账户/QOS约束仍由调度器检查，分区上限不保证账户允许该时长，拒绝时不会自动改值重提。
若仍到时限，保留last，下次用户手动仅提交未完成索引；不自动requeue或链式提交。
参考 [Slurm scontrol](https://slurm.schedmd.com/scontrol.html) 与
[sbatch时间限制](https://slurm.schedmd.com/sbatch.html#OPT_time)。

提交日志在新上传脚本目录 `dcu_continue_*.log`。计算日志在终端打印的 `Logs=...`，
即原输出根的 `operations/dcu_continue_<时间>_<随机串>/`，里面另有两份启动器SHA、
continuation_plan、资源/JobID、checkpoint_before备份。正式训练仍使用原各自run目录。

这是运营入口修复，原115文件和A01核心源码/config内容SHA不变；无需重新audit。
本地使用合成checkpoint和假调度器验证，不代表超算实际续训已成功。本轮不自动提交。
