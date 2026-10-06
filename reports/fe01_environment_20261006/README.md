# FE01计算节点诊断：ROCm已加载，缺MPI动态库

真实来源是用户提供的`envdiag_29351024.zip`；其中实际日志和Job ID为**29351034**。归档/原日志SHA及机器可读结果见[evidence.json](evidence.json)。此目录的两份日志只脱敏了私有路径和节点名；原始字节SHA另行保存，不能把脱敏日志的SHA当作原日志SHA。

| 检查 | 实际结果 |
|---|---|
| `env.sh` / `identity_job.sbatch` | 两份SHA都与已交付修复版一致 |
| 继承环境 | 有devtoolset 7.3.1、ROCm 2.9、HPCX MPI 2.11.0、tmux、Miniconda |
| `module purge` | 模块列表变为空 |
| 新加载模块 | DTK 23.04、Miniconda 3，加载成功 |
| zb中的torch导入 | `OSError: libmpi.so.40: cannot open shared object file` |
| checkpoint/HDF5/模型评价 | 独立诊断未执行这些任务 |

这份日志已证实部署文件版本正确、ROCm冲突在此次诊断中已解除。此前对旧文件/旧日志的怀疑不能用于解释本次诊断失败。当前准备流程清理了原MPI模块，却只重载DTK和Miniconda；需要补回原编译器和MPI环境。恢复所需模块名直接取自真实继承列表，未猜测MPI安装路径或换装torch。

将[`fe01_recover_mpi_20261006.sh`](../../scripts/fe01_diagnostics/fe01_recover_mpi_20261006.sh)这一份文件拷到超算FE01项目根目录，在该目录登录节点执行：

```bash
bash fe01_recover_mpi_20261006.sh fe01_envdiag_29351034.out
```

这条命令验证日志与当前已部署脚本SHA一致，备份原`cluster_zb.env`，只追加模块列表和新run ID两个覆盖项，然后调用已有`submit.sh identity`直接提交一份CPU作业。数据路径、账户等用户配置保留，旧输出目录保留。新的run ID是`fe01_eval1_mpi_29351034`。

恢复后的模块列表：

```text
compiler/devtoolset/7.3.1
compiler/rocm/dtk-23.04
mpi/hpcx/2.11.0/gcc-7.3.1
apps/miniconda/3
```

原私有env备份为`cluster_zb.env.before_mpi_29351034`，配置变更回执为`cluster_zb.env.mpi_recovery_29351034.receipt`。变更私有env会改变该文件相对于原归档的校验值；原训练和评价源码指纹保持不变，未修改或重签lock/归档SHA。下一轮提交记录会保存新env的真实SHA。

identity通过后，仍在FE01项目根目录按原入口执行下一阶段：

```bash
bash scripts/fe01_review/submit.sh verify
bash scripts/fe01_review/submit.sh reference
```

无需再次执行恢复脚本。它拒绝重复恢复、源码不匹配、没有MPI证据和复用非空恢复目录。使用已有失败目录不会被授权覆盖。

本地6项恢复测试及shell语法检查通过。原训练115文件SHA保持`3e537ff6693971d92b139e517f370b998237bd8b29989194e9ddcc06cc9dcfea`，评价module SHA保持`e65a7dd017828da10adcf5c45c4aa32d1d702833910972556d4eda5b6a640564`。恢复配置后的真实超算torch/identity尚未运行；不能把配置修复或本地stub测试称作实际identity PASS。
