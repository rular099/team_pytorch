# A01 train 启动阶段 DCU 不可用：保留成功项，检查并重提失败项

本轮分析基线为 `3ecf060a3334d1c7b48662cf4395a3bab07723ff`，分支
`exp/fe01-a01-absolute-amplitude-ablation`。用户报告2项成功、3项失败，错误为
`AssertionError: DCU unavailable in allocation`。尚未提供这批完整日志、数组作业号或失败索引，
所以不认证成功任务的epoch，也不猜测故障节点。

## 已能确定的事实

该断言是 `scripts/fe01_a01/env.sh` 最后一行。原 `job.sbatch` 在调用训练 `srun` 前
先 source 这个脚本。因此这些报错发生在 batch 进程的环境检查，尚未进入本次训练入口。
它不能单独证明整份 allocation 没分到DCU，也不能证明是模型、loss或数据错误。

Slurm的GRES按节点申请，step可以显式请求该作业已分配的同类型资源；设备可见性变量
具有节点/step作用域，见 [官方GRES说明](https://slurm.schedmd.com/gres.html)。
这些一般规则不等同于本集群实际配置；batch/step可见性差异、节点设备权限和HIP驱动
初始化异常都仍需实际记录来区分。不要凭这行断言清空可见性变量或更换torch。

## 上传一个独立脚本

将 `scripts/fe01_a01_ops/dcu_recovery.sh` 上传到**超算原FE01目录**，与 `.a01_releases/` 同级。
它会寻找已有 `.a01_releases/3ecf060a3334/a01.private.env`；也可以把脚本放在正在使用的
A01 release 内，与 `a01.private.env` 同级。若实际使用其它 release，用 `A01_ENV_FILE`
明确指定其私有设置文件。脚本不会新建release、编辑旧源码或重新签发gate。

## 登录节点：先检查原失败项

在上传的文件夹执行：

```bash
bash dcu_recovery.sh inspect
```

它只读 `jobs.tsv` 中最近一次 train 数组的五份 stderr，并输出发生此断言的索引，
不会调用sbatch/srun、不会导入torch。若要检查指定数组，执行 `inspect 数组作业号`。
没有断言的项不自动当作成功；完成训练仍需看原checkpoint和12轮曲线。

## 两种用户手动提交方式

要先收集设备诊断，执行：

```bash
bash dcu_recovery.sh probe
```

内部调用sbatch，申请原设置的4节点×4卡，最多10分钟。它对比batch与实际16个step rank：
记录host、可见性变量、`/dev/kfd`访问、torch/HIP版本、初始化错误和小张量设备运算。
不加载模型、权重或HDF5，不训练。只对本次分配的节点成立，不能据此认证旧故障节点。

要按新启动方式只重提旧日志中发生此断言的任务，执行：

```bash
bash dcu_recovery.sh retry failed
```

这是显式的训练提交命令；内部只提交一个数组，索引从原stderr识别。例如只识别出
0/2/4，就只提交这三项。也可显式用 `retry 0,2,4`，逗号分隔。
提交前对所选每项执行原 `preflight.py train`；缺门控或已有非空训练目录时整个提交停止。
因此已经完成的两项不会被覆盖或重训。不要对正在排队/运行的重提作业重复提交。

每项仍为原4节点16卡、global batch128、microbatch8、12 epochs、2544 updates，
最多3项并行。所有实际step rank先执行原module/conda环境脚本和设备运算检查，
全部通过后才启动原 `run.py train` 与原 `rank_exec.sh`。batch进程自身的DCU检查
不再作为训练入口；原检查保留在实际step内。没有CPU训练回退，没有调整GPU可见性变量。
每rank独立SeisBench缓存继续由原启动器提供。

如果实际step仍失败，本次训练不会开始，日志会留下具体host与初始化错误。
确认是故障节点后可设置 `A01_EXCLUDE_NODES` 再显式重提，或把日志交管理员定位；
本脚本不会猜测或自动屏蔽节点，也不会自动重试。

## 输出与来源

每次probe/retry的提交输出会打印 `Logs=...`，它位于原 `$A01_OUTPUT_ROOT/operations/`
下的新独立目录。里面保存out/err、所选索引/资源、JobID、只读启动脚本副本及SHA，
还有权限600的私有环境快照与原源码manifest副本。
私有环境含集群路径，应保留在原结果根内；分析通常只需要out/err、request.txt与job.tsv。
正式训练checkpoint仍写回原批次中对应的空run目录，不另建一套研究批次。

原115文件、A01核心源码/config/gate与成功输出保持不变；新增操作启动器的SHA单独记录。
无需重新environment/audit/diagnostics/pilot，也无需重传数据与权重。
兼容性和调度器测试均使用本地假环境/假调度器；生产节点实际可用性仍为NOT_RUN。
本轮未自动提交超算任务。
