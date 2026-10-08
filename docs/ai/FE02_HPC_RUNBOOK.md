# FE02 超算手动提交（2026-10-08）

本地代码完成不代表真实 DiTing/HPC 验证完成。本轮 **NOT_SUBMITTED**。
不需要先重复前轮的两组 smoke；五组正式 audit 包含必须的真实前向/梯度/冻结门，
通过即可正式训练。默认只 seed42，不提交 test、不自动扩展15组。

## 默认入口：只运行这一行（2026-10-08 简化版）

上传新版 `artifacts/fe02/fe02_source_20261008_autosubmit.tar.gz`，在解压后的代码目录运行：

```bash
bash scripts/fe02/submit.sh
```

脚本**直接调用sbatch提交**，屏幕返回 `Submitted batch job <JobID>`。
无需再复制命令，无需复制 env 文件、
手动登记权重、先提交 audit、填写依赖 Job ID，或另提 validation。
默认数组0–4分别为seed42的R0/A/B/C/M；每组4节点×4DCU，数组并发1，
单任务23:50h，不指定内存。每组在同一次allocation内依次：
初始化模块/zb环境 → 自动登记或核验同一个真实DiTing权重 → 必要audit →
12ep训练 → 固定时刻normal/random validation。audit失败立即停止，不启动训练。
这里只把原有必需阶段串联，未改变模型、预算、数据、选择规则或held-out test。

数据、split、权重采用已配置的超算路径；代码路径自动取当前脚本所在源码目录。
旧FE01 data audit不存在时自动完整计算hash，不要求用户另行准备。
入口不使用登录节点Python/torch，不读取大权重，登录节点只做环境快照和sbatch提交。
计算任务开始时校验快照SHA，确保排队期间配置未变。共享权重登记和同一run均加锁；
已有记录不覆盖。仅需预览时可选 `bash scripts/fe02/submit.sh --dry-run`，默认不是预览。

如果某组超时，只选择那组从last继续，例如B对应index2：

```bash
FE02_RESUME=1 FE02_INDICES=2 bash scripts/fe02/submit.sh
```

同样直接提交并返回JobID。必须已有last与原成功audit；不回退到best，不删除失败
记录。源代码、配置、数据、权重、world size不一致时原训练器拒绝恢复。

下文保留为**可选的分阶段高级操作**，不是启动本轮训练的必读步骤；
旧四条命令打印器仅用于需要人工分拆任务的情况。主入口已经代办登记/audit/train/eval。

## 1. 上传新源码包

本地交付文件位于 `team_pytorch_fe02/artifacts/fe02/fe02_source_20261008.tar.gz`
及其 `.sha256`。包包含源码、配置、脚本、dtbench snapshot/license，**不含大权重/数据**。
重新打包（要求此独立 worktree 已提交且干净）：

```bash
bash scripts/fe02/pack_source.sh --dtbench-root ../ditingbench \
  --output artifacts/fe02/fe02_source_new.tar.gz
```

上传两文件到超算后，选择不存在/空的新目录，不能覆盖 FE01/V01/RT55：

```bash
# 在两个上传文件所在目录执行
sha256sum -c fe02_source_20261008.tar.gz.sha256
mkdir /public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_fe02
tar -xzf fe02_source_20261008.tar.gz \
  -C /public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_fe02
cd /public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_fe02
```

HPC 无 GitHub/.git 不影响运行。`SOURCE_IDENTITY.json` 逐文件 SHA 在计算任务开始前
校验。不要上传旧单文件替换包内源码；修改源码后须新版本/新 audit，不能沿用旧 lock。

## 2. 登录节点：环境与权重登记（不 import torch）

```bash
export FE02_CODE_ROOT=/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_fe02
export FE02_ENV_FILE=/public/home/test_bigmodel/seismogram/zb/team_pytorch/fe02_cluster_20261008.env
cp -n fe02_cluster.env.example "$FE02_ENV_FILE"
source "$FE02_ENV_FILE"

# 沿用旧RT55/query-geometry launcher的MAE1200M路径，先确认存在。
# 如文件迁移过，export FE02_DITING_CHECKPOINT=实际路径；不是RT55/RT61 full_model_best。
ls -lh "$FE02_DITING_CHECKPOINT"
python scripts/fe02/register_diting.py \
  --checkpoint "$FE02_DITING_CHECKPOINT" \
  --output "$FE02_WEIGHTS_ROOT/pretrained_manifest.json" \
  --source '已有DiTing MAE1200M预训练文件；语料重叠unknown，补充已知来源'
python scripts/fe02/verify_source.py
```

登记只算文件 hash，不会加载模型。manifest 已存在时不要重复登记/覆盖；同一文件
让五组共同读取，不复制1.2B权重。**目前无法替你确认的是该真实权重文件是否仍在
默认路径**；默认来自已有launcher，不是本轮远端文件存在性核验。路径为
`/public/home/test_bigmodel/seismogram/mx/results/scaling_diting_1b/scaling_diting_1200M/checkpoint_pt_epoch_70/mp_rank_00_model_states.pt`。
如果存在可直接登记；如已移动，再覆盖FE02_DITING_CHECKPOINT。

env 默认复用已成功 FE01-EVAL1 的 `zb` 环境（Python3.9/torch1.13.1 ROCm），
purge 后加载 devtoolset7.3.1、DTK23.04、MPI hpcx2.11.0、miniconda3。
MPI 不可省略，否则复现 libmpi.so.40 缺失。无需登录节点 torch，无需重装 xFormers
或升级 torch。没有 conda shell 初始化时在私有 env 填 `FE02_CONDA_SH`。
若实际模块版本不同，使用与你已成功运行的 zb 环境一致的值，不猜版本。

data root 已设为 `/public/home/test_bigmodel/seismogram/zb/origin_corrected_diting_vel_acc_vs30`；
split 指向旧 **team_pytorch-zhangb-diting-backbone-attnpool-team** 的 RT55 目录；
新输出为 `/public/home/test_bigmodel/seismogram/zb/team_pytorch/fe02_runs_20261008`。
`FE02_REUSE_DATA_AUDIT` 默认旧 FE01 的 pilot TEAM data_split_audit，只复用年度
文件的 SHA/stat/split身份，不复用小 pilot cohort 或旧模型/audit lock；正式重新锁定
全部满足规则的 train/val。如果该文件不存在，执行 `export FE02_REUSE_DATA_AUDIT=''`
让新的 audit 完整算数据 hash。数据变动也必须重算，不能借旧 hash 隐藏变化。

默认4节点×4DCU、每task8CPU、23:50h。内存/account/qos 未默认硬写；确实需要时
在私有 env 按账户资源填写。这样避免此前固定内存造成 Memory specification 错误。
可选择1/2/4/8/16个总设备（microbatch8×world 必须整除128），但 audit前确定，
续训必须保持原 world size。五组 array 默认并发1；提高并发须显式设置
`FE02_ARRAY_CONCURRENCY`，每个训练run自身使用4节点，不是五组合用4节点。

## 3. 提交 audit，再提交训练

```bash
export FE02_STAGE=formal FE02_INDICES=0-4 FE02_RESUME=0
bash scripts/fe02/print_submit_commands.sh
```

打印器仅打印4条可复制的 `sbatch`，不会提交。先复制 **fe02-audit** 那一条，
得到 array Job ID。0=R0、1=A、2=B、3=C、4=M。每组单DCU audit，校验真实
encoder keys/shapes/native feature、实际 adapter、冻结和梯度、因果边界、query
batch/order 独立性，之后才发布 AUDIT_PASS。正式 audit 的数据人口与训练一致。

可以等待五组 audit 成功后复制 **fe02-train**；也可在用户手动提交 audit 后：

```bash
export FE02_AUDIT_JOB_ID=替换为刚提交的数字JobID
bash scripts/fe02/print_submit_commands.sh
# 此次手动复制 fe02-train 那一条，带 aftercorr，对应每个index的audit成功后才运行。
```

训练不读取旧训练好的 adapter/heads，而是加载同一个 MAE encoder 后从头训练
允许训练的模块。12 epochs、有效batch128、原采样与损失固定。可选 pilot 配置
已保留（stage=pilot），但不是默认任务，不必另跑一轮才能启动正式 audit/train。
43/44的index分别5–9/10–14；仅用户明确增加seed时设置FE02_INDICES，不自动提交。

## 4. 超时/续训

只重提失败/超时的index，禁止覆盖其他完成的输出：

```bash
export FE02_RESUME=1 FE02_INDICES=2
unset FE02_AUDIT_JOB_ID
bash scripts/fe02/print_submit_commands.sh
# 只复制 fe02-train 那条；不重复 audit。不改 config/source/world。
```

从 `last.pth` 完整恢复 optimizer/scheduler/各rank RNG/committed journals。
未发布 epoch 重算，旧尝试保留；不从best回退。初次任务尚未产生last时不能resume，
保留失败目录、选择全新 FE02_OUTPUT_ROOT 并重新对应 audit，不删除旧记录。
每组 init/last/best 包含冻结encoder，可能占数GB，预留数十至百GB输出空间。

用户检查 `squeue` 和 `sacct`，不以生成 config 或 init 文件视为训练成功。

## 5. 训练完成后，手动 validation 和汇总

```bash
export FE02_RESUME=0 FE02_INDICES=0-4 FE02_EVAL_TAG=fixed_val
unset FE02_AUDIT_JOB_ID
bash scripts/fe02/print_submit_commands.sh
# 只复制 fe02-eval 那条。包含1/3/5/10/20/40/90s、normal/random、所有合法目标。
```

检查五个 `evaluation_fixed_val/provenance.json` 成功落地，之后再次打印并仅复制
**fe02-collect**。它核验同encoder/公共初始化/预算/人口，生成独立
`$FE02_OUTPUT_ROOT/review_seed42/`：主成绩、全时刻/normal/random/单台多台/目标
角色的点与概率指标、空间误差、预定配对bootstrap、实际采样统计。
不需要另提交normal和random：同一个 eval 已包含两者。
随机非标准 validation 时刻仅是可选诊断（`FE02_EVAL_TAG=random_val`），
不进入 fixed-time 主比较，不默认要求运行。

记录实际 scheduler 结果（在登录节点只查状态即可）：

```bash
sacct -j 审计JobID,训练JobID,评价JobID,汇总JobID \
  --format=JobID,JobName,State,ExitCode,Elapsed,ElapsedRaw,MaxRSS,AllocTRES -P
```

保存该输出为新的 sacct.txt，回传时与slurm .out/.err一起提供。缺sacct不能宣称
State=COMPLETED。在已获计算节点allocation中封装（无需重新训练/推理）：

```bash
"$FE02_PYTHON" scripts/fe02/pack_results.py --roots \
  "$FE02_OUTPUT_ROOT/audits" "$FE02_OUTPUT_ROOT/review_seed42" "$FE02_OUTPUT_ROOT/slurm" \
  "$FE02_OUTPUT_ROOT/fe02__formal__R0__seed42" \
  "$FE02_OUTPUT_ROOT/fe02__formal__A__seed42" \
  "$FE02_OUTPUT_ROOT/fe02__formal__B__seed42" \
  "$FE02_OUTPUT_ROOT/fe02__formal__C__seed42" \
  "$FE02_OUTPUT_ROOT/fe02__formal__M__seed42" \
  --output "$FE02_OUTPUT_ROOT/fe02_review_seed42.tar.gz"
```

默认≤20MB的逐文件证据入包，较大预测导出保留外部路径/bytes/SHA；权重和波形
不进包/Git。下载tar.gz/.sha256和sacct；完整大预测需按外部索引补传时再操作。
ChatGPT审阅入口：[FE02_REVIEW_REQUEST.md](FE02_REVIEW_REQUEST.md)。
