# FE01 ON V4 inventory 清单路径修复（2026-10-25）

用户下载 `outputs/fe01_v4/inventory_31089276_0.err/.out`。实际失败为
`ValueError: Pretrained manifest SHA differs`，发生在 `sources.inspect → check_family`。
这份记录没有 DCU unavailable、ROCm 冲突或缺动态库错误；timm FutureWarning 不是退出原因。
原运行源码为 `a5c677c06ae67ed5c47b0cab0905908e1275cf81`，原 run 为 `on_v4_20261022`。
本轮只实施修复和本地测试，**没有自动提交真实超算任务**。

## 原因与修复

通用 `fe01_review.runners.relocate` 将所有家族的 manifest 强制改为
`$FE01_WEIGHTS_ROOT/pretrained_manifest.json`。但是 A01 的配置生成器明确允许
DiTing 使用独立 A01 输出目录中的 `weights/diting_manifest.json`；它与 PhaseNet/EQT
的通用清单有不同 SHA。V4 inventory 第一个家族为 DiTing，进入这个不合适的通用重定位后报错。
原日志未打印实际选中文件内容；上述路径覆盖由实际源码和既有配置/lock 证实，不能据此声称权重损坏。

新 `sources.runtime_config` 在原配置已经通过认证后：

1. 原清单文件仍存在时保留其精确路径，允许它位于独立 A01 权重目录。
2. 原路径缺失时，DiTing 复用已有 `A01_DITING_MANIFEST`；否则仅重定位目录并保留原文件名。
3. 所选文件必须匹配原 lock 的完整 SHA；原文件存在但 SHA 错误时立即停止，不搜索另一份文件绕过错误。
4. 原 `check_family` 的实际 asset 校验继续执行，checkpoint 字节/epoch/config/tensor/encoder 门保持。

inventory 逐个打印家族、policy 和 epoch；成功记录同时保存原清单、运行时清单和 asset SHA。
TEAM 仍按 scratch 认证，不能要求其训练后 encoder 等于初态。
旧115、`fe01_a01/`、`fe01_review/`、旧锁和旧 seal 均不修改。

## 超算怎么恢复

复制本轮源码 `.tar.gz` 和同名 `.sha256` 到：

```text
/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_fe01
```

准确包名/解压目录以本轮交付消息为准；不要覆盖 `on_eval_v4_a5c677c06ae6`。
在项目根目录激活 `zb`，校验、解压修复包后，按交付消息设置 `FIX` 指向修复 release，执行：

```bash
bash "$FIX/scripts/fe01_a01_compare/recover_inventory.sh"
```

这是**实际提交入口**，内部用 `sbatch` 只提交一个修复后的 CPU inventory 作业。登录节点不会加载 Torch 或模型。
它自动创建新私有文件和新运行目录，不要求你手工改旧配置，也不需要重新拷权重/数据：

```text
新私有文件：项目根目录/on_eval_v4_inventory_fix.private.env
新运行：outputs/fe01_on_eval_v4/on_v4_20261025_inventory_fix/
日志：新运行/slurm/inventory_JOBID_0.out 与 .err
```

脚本严格支持上面的 a5c677c 原 release。它核旧 plan/contract/source 和源文件 SHA，再逐一证明
CPU 依赖未变；`sources.py` 除 `inspect/runtime_config` 之外的 AST 也必须相同。
只复用 `compare_existing`、`covariates`、`crossing_truth` 中**已完成并通过完整 seal 的项目**。
传输优先用同文件系统硬链接，否则复制派生结果；原权重和 HDF5 不复制。
新目录保存原 seal、原 attempt 和 `REUSE_PROVENANCE/receipt.json`，明确区分原计算身份与本次传输身份，
原 seal 字节不修改，不能把复用结果说成新源码重算。数据 SHA 缓存也保留原 seal，避免重复全量散列。

若显示 `WAIT_FOR_ORIGINAL_RESULT`，说明某项旧 CPU 结果尚未封存；先让原作业结束，再执行同一条恢复命令。
不要因为这个状态直接重跑三组 CPU。重复恢复会补入新完成项；已成功/正在运行的修复 inventory 不会重复提交。
若原 CPU 项实际失败，可以在确认旧作业已结束后，用新 release 的 `submit_stage.sh` 手动提交该阶段缺项。

后续统一使用新目录/新环境，避免还在检查原失败运行：

```bash
S="$FIX/scripts/fe01_a01_compare"
ONENV="$PWD/on_eval_v4_inventory_fix.private.env"
ONRUN=on_v4_20261025_inventory_fix
bash "$S/status.sh" --env "$ONENV" --run-id "$ONRUN"
```

inventory 与 covariates 全部 `COMPLETE_SEAL_VERIFIED` 后才继续：

```bash
bash "$S/submit_stage.sh" --env "$ONENV" --run-id "$ONRUN" --stage model_smoke
```

后续仍按[完整手册](FE01_ON_EVAL_RUNBOOK.md)的 pilot → replay/latency → analyze → render → pack，
但保持本节的 **新 S/ONENV/ONRUN**；不要复制旧手册里的旧 run ID。
所有阶段由你手动提交，本恢复入口不会连跑后续阶段。

## 本地证据边界

聚焦测试覆盖混合家族清单、独立原 A01 路径、严格负向 SHA/缺文件、原配置不变、
CPU 源码/人口/字节不变、损坏拒绝、幂等恢复、在跑去重、迟完成结果补入及数据缓存原 seal。
fake scheduler 实际验证准备新 plan、导入结果、只 sbatch inventory；没有执行真实 scheduler。
实际原 release 与修复执行树的 CPU 兼容门另行运行并记录。
本地没有生产 checkpoint/HDF5/DCU，实际修复 inventory 和模型前向仍须超算验证。
