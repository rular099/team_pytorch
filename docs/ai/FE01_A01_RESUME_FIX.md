# A01 本次恢复提交（2026-10-11，数组30265280之后）

请使用本页的新包，旧 `fe01_a01_continue_20261011.zip` 是上一版。
两项16卡通信探针均通过；ON恢复随机状态时报错，OFF启动20分钟没有新采样记录。
本次没有新增完成epoch，不能把探针PASS当成训练完成。详情见
[本批日志报告](../../reports/fe01_a01_continue_20261011/README.md)。

## 本地操作：只上传一个新ZIP

将 `artifacts/fe01_a01/resume_fix_20261011/fe01_a01_resume_fix_20261011.zip`
上传到超算目录：

```text
/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_fe01
```

包内是新的 `a01_resume_fix_20261011/`，包含提交脚本、运营启动器、恢复兼容循环、
源码差异检查、启动记录工具和SHA清单。不要把包内文件覆盖进旧release。
不用重新上传权重、HDF5或旧核心源码。

## 超算登录节点操作：解压并提交

```bash
cd /public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_fe01
unzip -n fe01_a01_resume_fix_20261011.zip
export A01_ENV_FILE="$PWD/.a01_releases/3ecf060a3334/a01.private.env"
bash a01_resume_fix_20261011/submit_diting_resume_fix.sh 3,4
```

最后一行会检查包和旧门控，然后调用真正的 `sbatch`，提交一个数组。
索引3=DiTing ON，索引4=DiTing OFF；每项4节点16卡，同时运行最多8节点32卡。
其它三项不提交。仍使用原zb环境、ROCm设置、12轮、global batch128、microbatch8、
seed42与2544 updates；时间继续按分区上限申请最多72小时，账户限制由Slurm检查。
将来只剩某一项时，最后参数改成 `3` 或 `4`，不要重提已完成项。

终端会打印 `SUBMITTED ... job=...`、`Logs=...` 和提交日志路径。
登录节点只做stdlib文件检查和提交；真正DCU通信、模型加载、训练都通过分配内的srun执行。

## 计算节点自动做什么

1. 每项16个rank做原环境检查与实际all-reduce，最多5分钟；失败即停。
2. ON若有last.pth，检查身份/epoch/updates/每rank RNG并备份旧last/best，然后续训。
   本批ON已打印恢复epoch5；下一次仍以真实last的内部状态为准，从下一轮继续。
3. OFF空目录才fresh。若只残留未提交的初始化文件，脚本在持有索引锁时把整个旧目录
   移到本次operation的 `uncommitted_before_index4/`，记录清单，再从原认证初始化开始。
   仅限没有last/best/临时checkpoint、epoch曲线记录、非空sample journal和未知文件的OFF；
   其余非空无last情况停止，保留目录等待检查。不会用best替代last，也不会让ON从头开始。
4. 原训练循环的独立静态副本只修正DCU RNG恢复：每个保存状态先 `.cpu()` 再传给
   `torch.cuda.set_rng_state_all`，不重设随机种子、不改变状态字节。
   原checkpoint、audit、diagnostics、pilot身份检查仍执行。
5. 每rank记录import、门控、通信组、barrier、数据集、模型构建、认证初态加载、DDP、
   checkpoint、RNG、首批数据、第一次optimizer update等阶段。第一次update前每120秒
   保存Python调用栈；原生库内部只可定位到Python调用位置。
6. 启动上限仍20分钟。出现新非空采样journal或确认第一次update后才解除启动限制；
   普通“模型开始加载”日志不算训练进展。超时会打印最后阶段，保留全部文件。

## 结果与需要下载的文件

训练checkpoint仍在原输出根的各自 `a01__diting_pretrained_frozen__on/off__seed42/`。
本次日志在提交时打印的 `Logs=` 目录，即：

```text
<A01_OUTPUT_ROOT>/operations/dcu_continue_<本次时间>_<随机串>/
```

作业结束或普通失败时，脚本自动在这个目录生成：

```text
return_bundle_index3.zip
return_bundle_index4.zip
```

下载这两个ZIP即可。包内有out/err、逐rank阶段/调用栈、训练目录文件名与大小、
曲线、执行源码身份和checkpoint备份SHA清单。不会放入权重或a01.private.env。
大于8MiB的日志仅携带有明确TAIL后缀的末尾8MiB。外部强杀/节点宕机可能来不及打包；
这种情况下下载该operation里的out/err和startup目录。不要下载整个checkpoint_before目录，
它是超算上保留的旧大权重备份。

## 身份与验证边界

原115文件和A01核心源码/config保持不变，不重签旧门控。新执行循环不是原文件原样执行：
它有独立SHA，记录在 `execution_identity_index3/4.json`，随返回包一起交接。
每次启动用AST检查“静态副本减去允许的观察语句和CPU转换”等于原train函数，
并检查所有实际调用的依赖绑定仍是原引擎对象；不全局monkeypatch torch或原模块。
旧checkpoint中的核心SHA描述不可变的认证核心，新运行必须同时保留这份执行身份，
不能仅凭旧checkpoint字段声称执行源码完全相同。

本地验证见报告中的verification.json。真实16卡续训和OFF卡点仍须本次超算结果确认；
本轮没有自动提交任何真实作业，也没有生产评价或held-out test访问。
