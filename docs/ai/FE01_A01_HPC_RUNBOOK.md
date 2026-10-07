# FE01-A01：文件上传与超算提交说明

本轮新增代码在独立 A01 目录，旧 FE01/EVAL1 的源码、配置、权重与结果继续只读复用。
本说明中的 `$A01_CODE_ROOT`、`$A01_OUTPUT_ROOT` 等变量均由 `a01.private.env` 定义。
实际账户和路径只在随源码包交付的私有部署脚本中，未写入 Git。

## 1. 本地已经完成什么；你需要上传什么

本地已经完成代码、配置、聚焦测试、小型合成诊断与图源数据。正式训练/全量评价均未提交。

只需把以下三个文件上传到**超算原 FE01 代码目录**，保留在同一个文件夹：

- `fe01_a01_source.tar.gz`：独立 A01 源码包，包含冻结 dtbench 源码依赖。
- `fe01_a01_source.tar.gz.sha256`：源码包校验文件。
- `deploy_a01.sh`：已按已有资料填写实际路径的部署脚本。

无需重传 Japan HDF5、STEAD 权重、旧训练结果或 DiTing 大权重。
本地诊断结果在 `reports/fe01_a01_local_20261007/`，它们是合成例子，不能当作真实训练结果。

## 2. 超算登录节点：部署，然后检查

在上传三个文件的目录执行：

```bash
bash deploy_a01.sh
```

此脚本校验 SHA，解压到原 FE01 目录下的**新 A01 release 目录**，生成私有环境文件，
并打印后续需要进入的绝对目录。不提交作业、不加载大模型、不改旧环境。
若同名 release 已存在，会停止，避免覆盖。每个新源码版本使用独立 release。

进入脚本打印的 A01 目录后执行：

```bash
bash scripts/fe01_a01/prepare.sh
```

这一步检查旧 checkpoint/数据声明并打印提交命令，**不会调用 sbatch 或 srun**。
若需要手动安装部署，源码内的等价入口为：

```bash
bash scripts/fe01_a01/configure.sh "$ORIGINAL_FE01_CODE_ROOT" "$ORIGINAL_FE01_RUNS_ROOT" a01_batch_001
bash scripts/fe01_a01/prepare.sh
```

`configure.sh` 仅使用 Python 标准库读取原 resolved config，不导入 torch。
它生成 `a01.private.env`；现有环境文件不会被覆盖。

## 3. 索引到底指什么

索引是 Slurm job array 中的一项，在**所有 A01 阶段**保持同一含义；不是旧 FE01 的索引。

| A01 索引 | 对应任务 | ON 对照 |
|---|---|---|
| 0 | TEAM seed42，从原 epoch0 初始化训练 OFF | 原 TEAM ON，冻结选择 epoch11 |
| 1 | PhaseNet seed42，从原 epoch0 初始化训练 OFF | 原 PhaseNet ON，冻结选择 epoch12 |
| 2 | EQT seed42，从原 epoch0 初始化训练 OFF | 原 EQT ON，冻结选择 epoch10 |
| 3 | DiTing seed42，新训匹配的 ON | 此任务本身 |
| 4 | DiTing seed42，新训匹配的 OFF | 索引3 |

`0-4` 表示五项；`0-2` 表示三个 OFF；`3` 表示只提交 DiTing ON。
旧 ON 的复用必须先通过身份、初始权重、前向、单步更新和训练预算检查。
复用失败时停止相应家族，不自动增加新 ON 训练。
RT55/RT61 的下游网络不充当 DiTing ON；只复用已认证的 DiTing **预训练 encoder**。

## 4. 登录节点：逐阶段手动提交

下表每一行是一个独立的提交操作。**等待该阶段结束并确认 PASS 后，再执行下一行。**
脚本每次仅提交你指定的一个阶段，不自动串联或重提。

| 顺序 | 你在登录节点执行的命令 | 作业内部在计算节点做什么 | 输出目录 |
|---|---|---|---|
| 1 | `bash scripts/fe01_a01/submit.sh environment` | 清理冲突模块，加载已验证 DTK+MPI，激活 zb，检查 DCU 与源码 | `$A01_OUTPUT_ROOT/environment/` |
| 2 | `bash scripts/fe01_a01/submit.sh audit 0-4` | 各模型身份、只读数据/旧权重、ON 复用、matched init、采样与预算门控 | `$A01_OUTPUT_ROOT/audits/<run_id>/` |
| 3 | `bash scripts/fe01_a01/submit.sh diagnostics 0-4` | 查询逐层追踪/梯度/有限差分、置换、嵌套 K、前缀/尺度检查与图 | `$A01_OUTPUT_ROOT/diagnostics/<run_id>/` |
| 4 | `bash scripts/fe01_a01/submit.sh pilot 0-4` | 每项单 microbatch 更新，检查 loss 与冻结 encoder；不产生正式结果 | `$A01_OUTPUT_ROOT/pilot/<run_id>/` |
| 5 | `bash scripts/fe01_a01/submit.sh train 0-4` | 五个新正式训练，各12 epochs | `$A01_OUTPUT_ROOT/<run_id>/` |
| 6 | `bash scripts/fe01_a01/submit.sh eval 0-3` | 前四项的 selected/epoch12 评价；前三项另重算冻结旧 ON 的精确 MDN 指标 | `$A01_OUTPUT_ROOT/evaluation/`、`comparisons/` |
| 7 | `bash scripts/fe01_a01/submit.sh eval 4` | DiTing OFF 评价并与已完成的索引3比较 | 同上 |
| 8 | `bash scripts/fe01_a01/submit.sh pack` | 导出轻量审阅包，排除大 checkpoint、波形与逐层数组 | `$A01_OUTPUT_ROOT/review.tar.gz`、`.sha256` |

**如果 DiTing 门控未通过，可先把第2–6行中的 `0-4`/`0-3` 换为 `0-2` 推进前三个家族。**
不必让这些任务等待 DiTing；也不能在未注册真实 encoder 时启动随机替代。
若只重看一个阶段，用单索引，例如 `bash scripts/fe01_a01/submit.sh diagnostics 1`。
非空输出会拒绝覆盖，重复检查应使用新批次目录，见第6节。

## 5. 资源、前置条件和训练预算

- environment/audit/diagnostics/pilot/eval/pack：每项1节点1卡，默认最多4项同时运行。
- train：每项4节点，每节点4卡，world size16；默认最多3项同时运行，占用最多12节点。
  另外两项排队。已有16节点资源足以运行这一安排。
- global batch128、microbatch8、world16，累积步数1。
  程序从 PyTorch DistributedSampler/DataLoader 的 drop-last 规则计算实际预算。
  9084事件×3 draws=27252；每 epoch212 updates，12 epochs2544 updates，
  每 epoch消费27136样本、舍弃116样本。上述数量有执行核验，不是训练代码中的硬填常量。
- 旧三家族的真实 world size 是16；A01 要求保持一致，避免 per-rank loss 分母和样本分配改变。
- `a01.private.env` 可修改 partition、时间限制、CPU数和 Slurm gres 字段。
  默认每节点申请64GB主机内存，适合大 encoder 的只读载入；可按节点容量调整。
  更改节点/卡数会改变 world size，不能据 global batch 相同就绕过可比性门控。
- identity/audit 需要原 resolved config、protocol lock、best/last checkpoint、原 audit 数据声明。
  init 缺失时只有完整重建状态指纹与冻结 inventory 一致才允许继续。
- DiTing ON/OFF 都必须有匹配的 init 和采样门控。旧预训练清单不会被修改；
  如需注册既有 DiTing encoder，只写入 A01 输出中的专用 manifest。
- 后续阶段验证当前 A01 源码 SHA、前一阶段证据和配置 SHA。源码改动必须重新做门控。
- 检查为 HDF5 之后的前缀因果性；上游滤波/重采样仍为 `UPSTREAM_CAUSALITY_UNKNOWN`。
- diagnostics 中前三项加载原 ON checkpoint 并屏蔽振幅，是 `inference_intervention`。
  DiTing 训练前诊断标为 matched initial checkpoint。正式 OFF 的评价才是 `trained_ablation`。

## 6. 失败在哪里看；怎么恢复

所有 Slurm stdout/stderr 在 `$A01_OUTPUT_ROOT/slurm/`，文件名含阶段与作业ID。
环境阶段应先出现 `A01_ENV_READY`；Python 导入前失败时先看 module/conda 日志。
环境脚本先 `module purge`，逐项检查实际 `LOADEDMODULES`，再 source conda.sh 激活 zb；
保留已经验证的 DTK+MPI 组合，不重装 torch、不重跑旧 MPI 恢复脚本。

具体错误通常会指出：checkpoint SHA、encoder、epoch、config、训练源码、
数据 stat、请求人口、初始权重或哪个前置 gate 不匹配。不要删除校验或改旧 lock。
`a01.private.env` 中 DiTing encoder 的路径/SHA 来自已下载的 EVAL1 inventory；
若超算上该文件已移动，先确认同一 SHA 的原文件位置，再修改私有设置。

失败目录、日志、旧 checkpoint 均保留。重新执行使用新 batch ID/新输出根，
例如在私有 env 中将 `A01_BATCH_ID` 和 `A01_OUTPUT_ROOT` 改为新的对应值；
若 DiTing manifest 位于旧 batch 输出中，也同步改为新 batch 的路径。
新批次从 environment 开始依次提交。此版本没有自动续训或自动重提，
训练失败也不会从 ON 最终权重或 best checkpoint 偷换初始化。

查看结果根路径：

```bash
source a01.private.env
printf '%s\n' "$A01_OUTPUT_ROOT"
```

打包完成后，下载这个根路径中的 `review.tar.gz` 和 `review.tar.gz.sha256` 即可交回分析。
这里没有 test 入口；全部正式评价锁定 validation 共同十个 cell。
