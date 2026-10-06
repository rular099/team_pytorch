# FE01重复环境错误：独立计算节点诊断

**最新实际结果：用户已提供`envdiag_29351024.zip`，内含Job 29351034的日志。两份脚本SHA一致，purge和DTK加载成功，torch导入因缺`libmpi.so.40`失败。** 恢复MPI及直接重新提交identity的入口见[实际诊断报告](../../reports/fe01_environment_20261006/README.md)。

此前仅收到ROCm冲突和`run.py -> runners.py -> torch`堆栈，尚缺Job ID/stdout/脚本SHA。关于旧文件或旧日志的解释当时只是推断。现在以真实诊断证据为准：这次部署版本正确，当前配置漏掉了purge后的MPI重载。

下一步使用独立脚本`scripts/fe01_diagnostics/fe01_check_environment_20261006.sbatch`。它不调用评价入口、不读取HDF5/checkpoint、不执行模型前向、不写任何review阶段目录。它位于评价源码指纹目录之外；本次没有改动已有评价脚本、私有env或源码身份。

把这一份`.sbatch`文件拷到超算已有的**FE01项目根目录**，在该目录的登录节点执行：

```bash
sbatch --partition=diting fe01_check_environment_20261006.sbatch
```

脚本从该目录读取`scripts/fe01_review/cluster_zb.env`，遵循其中的实际代码根目录、模块和zb设置。日志写为项目根目录的`fe01_envdiag_<实际JobID>.out`和`.err`，使用1节点、2 CPU、4G内存，不申请DCU。正常CPU诊断的`accelerator_available=false`不是失败。

日志应包含以下证据：

| 标记 | 含义 |
|---|---|
| `FE01_DIAG_VERSION=20261006-1` | 确认这次执行的是独立诊断入口 |
| `FE01_DIAG_JOB` / `FE01_DIAG_CODE_ROOT` / `FE01_DIAG_ENV_FILE` | 实际Job ID、代码根目录和私有env位置 |
| 两组`FE01_DIAG_EXPECTED_SHA` / `ACTUAL_SHA` / `FILE_MATCH` | 实际计算节点文件与已交付修复版是否一致 |
| `FE01_DIAG_MODULES_INHERITED` / `AFTER_PURGE` | ROCm 2.9是否继承、是否在purge后仍保留 |
| `FE01_DIAG_EXPLICIT_UNLOAD` | 有ROCm模块残留时，按实际列表显式卸载及检查 |
| `FE01_DIAG_MODULES_READY` / `FE01_DIAG_TORCH` | 最终模块及实际Python/torch路径、版本 |
| `FE01_DIAG_RESULT`或`FE01_DIAG_ERROR` | 诊断结果或停止点，不能当作模型验证PASS |

哈希基准为代码commit `4c9c1757f726e571022a398cb5a217ca6c325cbc`，对应此前修复包SHA `7128805e4d6902cb2d2c1a55fad2d7a8ec3a27d9581428f58cbcefa035d9c915`：

```text
889baabc4dc8836312ea4f011ed83ccc0ad6a9d543b5e2ea8c5eb9c5445ff7f0  scripts/fe01_review/env.sh
ec0d4a48130749516255f75111bb57512e0a2f154dc94b3ae7ff23c09860ce2c  scripts/fe01_review/identity_job.sbatch
```

`SCRIPT_VERSION_MISMATCH`时退出码3：实际文件不同，但独立torch导入已通过。模块/Conda/torch导入错误时退出码2，并停止；Tcl错误返回0也不会继续。`PASS`仅说明两个文件匹配且本次独立环境能导入torch，不证明此前identity成功，也不认证任何checkpoint或预测。

提交后保存这两个实际日志供下一步定位。诊断不会自动重新提交identity或修改旧失败记录；依据实际文件/模块证据再修复或恢复评价。

本地验证：6项独立诊断测试通过，覆盖正常导入、旧脚本字节不匹配、Tcl冲突返回0、purge保留ROCm后的显式卸载、卸载返回0却仍保留、动态库导入失败。`bash -n`通过。测试使用stub模块/torch探针。用户实际执行结果另见上述报告；恢复MPI后的identity尚未重新核验。

原训练115文件SHA保持`3e537ff6693971d92b139e517f370b998237bd8b29989194e9ddcc06cc9dcfea`，新评价module SHA保持`e65a7dd017828da10adcf5c45c4aa32d1d702833910972556d4eda5b6a640564`。
