# FE02-Lite 空数组 / nounset 启动修复（2026-11-24）

用户报告 Job `35455202`：`slurm_script: line 45: train_args[@]: unbound variable`。
前面的timm FutureWarning不导致这次退出。v2 worker为普通训练创建了空train_args数组，
Bash <=4.3在set -u下展开该数组会报错；本地Bash4.3.0已复现。参数展开失败发生在
srun执行前，不据这段日志宣称有任何正式optimizer更新或完整audit/训练结果。

正式修复把完整train命令及公共参数放入始终非空的数组，resume时再加--resume。
不修改模型、数据、loss、配置、checkpoint和环境依赖。

## 已失败v2作业：使用小补丁

将 `fe02_lite_bash_fix_20261124.tar.gz` 和 `.sha256` 上传至**原上传代码目录**：

```bash
sha256sum -c fe02_lite_bash_fix_20261124.tar.gz.sha256
tar -xzf fe02_lite_bash_fix_20261124.tar.gz

# 使用原来的FE02_LITE_ENV_FILE / FE02_LITE_OUTPUT_ROOT设置。
# 只检查和打印，不提交：
bash repair_fe02_lite_bash.sh --job-id 35455202 --dry-run

# 用户手动重提该JobID对应的一个任务，返回新JobID：
bash repair_fe02_lite_bash.sh --job-id 35455202
```

如果新终端没有原结果根目录的环境设置，追加：

```bash
bash repair_fe02_lite_bash.sh --job-id 35455202 --output-root /actual/original/fe02_lite_runs
```

其他同类失败作业使用各自的原JobID。不会自动展开三个视图或消融。默认仅适用于该次
参数展开失败、尚未创建训练run的启动；如已有last checkpoint，明确加`--resume`，
没有last时拒绝续训，不回退best，不覆盖已完成run。

恢复工具从 `OUTPUT_ROOT/launches/fe02-lite-*/submission.json` 找到精确原提交，核验
env/task-map/source manifest及全部原源码hash，只支持v2基点
`d53c3e82184d9bb3d8f85b1907d73af2ccfcf08c`。资源、视图、seed、stage、数据/权重路径、
原code root、run ID与抽样预算沿用原提交。去掉旧dependency，只重提选定的一个任务。

新的fixed worker和只读env放在 `OUTPUT_ROOT/launcher_fixes/bash-array-*/`，原
`SOURCE_IDENTITY.json`、原worker、代码快照、audit锁及历史日志均不改；原科学代码的
code_identity保持相同。新Job日志名有bashfix及新JobID，launcher_fix.json单独保存
原JobID、新sbatch回执、worker/tool SHA、env和source身份、精确命令。原通过的audit
按现有入口核验/复用；原本没有通过锁的任务仍须先通过真实audit。

补丁只新增根目录repair入口和compat目录，不将文件写入原`fe02_lite/`、
`scripts/fe02_lite/`或`configs/fe02_lite/`，避免原code_identity失配。直接替换这些目录
中的旧worker后，旧成功审计会正确报来源变化；本恢复入口不依赖这种替换。

## 全新任务

新目录和新输出根使用完整包 `fe02_lite_source_20261124_v3.tar.gz` 及正常submit入口。
已有v2目录的失败作业使用上方兼容恢复入口，保留其原审计身份。

本地验证为fake Slurm/model命令，不接触实际数据或DiTing，也没有真正sbatch。
已覆盖Bash4.3原错误复现、4.3/5.3两版本完整worker路径、fresh/resume/eval参数、
audit失败不启动srun、恢复前后原源码与audit锁字节一致，以及新固定worker对原v2
快照运行。真实集群重提及后续训练状态仍待用户执行确认。
