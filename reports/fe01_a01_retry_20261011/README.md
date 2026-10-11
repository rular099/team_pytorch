# A01 数组29953137日志与续训交付

分析/实现基线 `be8cc1d9f240bd7da56b6ca174d78b03fa842a1e`，分支
`exp/fe01-a01-absolute-amplitude-ablation`，仓库`rular099/team_pytorch`。
输入仅用户下载的`chaosuan_res/dcu_retry_20261009T161342_soCC3y.zip`，共六份out/err，
SHA256 `f232b956108c1f5aafafbae31ace0ca07636fa6f8cb4b47fcd6152e42f53d3e3`。
[逐成员SHA与结构化事实](log_evidence.json)、[日志epoch记录](logged_epochs.csv)。

## 已核验事实

| 数组索引 | 任务 | 设备探针 | 已打印epoch/updates | 结束证据 |
|---|---|---|---|---|
| 1 | PhaseNet OFF | 16/16 PASS，4节点 | 12/2544 | 16个通信watchdog正常关闭；无traceback/时限取消 |
| 3 | DiTing ON | 16/16 PASS，4节点 | 5/1060 | Slurm记录24小时时限取消 |
| 4 | DiTing OFF | 16/16 PASS，4节点 | 无epoch记录 | RSMI mutex/rsmi_init失败；最终24小时时限取消 |

索引3的五轮日志平均耗时17199.265秒，约4.78小时/轮。若后续速度相近，
完整12轮约57.33小时，从epoch5之后尚需约33.44小时；这些是按日志线性估计，
不是新作业性能保证。24小时不足以完成本协议的DiTing训练，不能缩短到5轮冒充完整结果。
索引1日志记录12轮和预定更新数，不应再次提交。用户之前报告的其它两项成功不在本包。

本包没有checkpoint内部身份/完整训练目录、sacct State/ExitCode、launcher SHA或当前私有设置。
因此不认证checkpoint已封装完整，也不凭六份日志重建训练源SHA。本次new source SHA
在独立verification.json中核验；真实旧checkpoint与gate在用户下次计算节点运行时认证。
CSV里的validation_selection仅逐字导出日志值；尚无本次完整配对评价人口/请求与权重验证，
不把它当独立test或跨模型完整性能排名，不据此改变超参数或选择额外checkpoint。

## 原因边界

旧数组29933642提供了spool路径找不到/退出127的直接证据，对应此前已交付的共享路径修复。
本批48个设备探针均PASS，三项进入正式多卡训练入口，该路径故障已不再出现。
索引3是训练实际运行后触及时间预算。索引4同时有RSMI初始化锁失败和未打印epoch，
与通信启动异常相符，但混合stderr不足以定位某一台节点，也不能证明唯一驱动根因。
ROCm项目维护仓库有相同RSMI锁错误的[初始化问题记录](https://github.com/ROCm/rocm_smi_lib/issues/88)；
这是外部类似问题线索，不替代本集群诊断。脚本不删除/dev/shm共享锁或更换ROCm/torch。

## 实施与验证范围

交付[手动提交入口](../../scripts/fe01_a01_ops/submit_diting_continue.sh)、
[运营启动器](../../scripts/fe01_a01_ops/dcu_recovery.sh)、
[断点与限时检查](../../scripts/fe01_a01_ops/continue_training.py)，均位于核心源码指纹范围之外。
[操作说明](../../docs/ai/FE01_A01_CONTINUE_TRAINING.md)给出本地上传/登录节点提交/计算节点行为。

只提交用户选定的3,4；last存在则原引擎严格resume，空目录才fresh，非空无last及完整12轮均停止。
不删除旧锁或绕过environment/audit/diagnostics/pilot；旧last/best与曲线身份先备份。
实际16卡all-reduce最多5分钟，正式训练启动到新采样记录最多20分钟。
4节点16卡、seed42、batch128/micro8、12轮2544updates、原Adam/调度器/采样及每rank RNG均保留。
分区允许时申请最多72小时，无法读取分区时保留原24小时；账户限制仍由Slurm处理。
若时限仍不足，后续由用户只手动续训未完成项，不自动链式提交。

具体本地测试、Python3.9语法、原115文件与A01核心SHA、三个原工作树保护见
[verification.json](verification.json)。测试使用合成checkpoint、假调度器和本地受控子进程，
不能证明真实16卡通信已恢复。**本轮没有sbatch/srun、正式训练或生产评价。**
