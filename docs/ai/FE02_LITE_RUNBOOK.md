# FE02-Lite 超算运行手册

2026-11-24：用户报告Job35455202空数组启动错误，见
[保持原audit身份的修复/重提入口](../../compat/fe02_lite_bash_20261124/README.md)。
代理未代提交新任务。以下命令由用户在超算执行。登录节点只解析配置、复制
代码快照和 sbatch；真实数据扫描、权重加载、forward/update审计、训练、validation、
汇总全部在计算节点。无需升级 PyTorch 或重装集群环境。

## 环境与代码包

全新目录上传 `fe02_lite_source_20261124_v3.tar.gz` 和 sha256，校验后解压。
源码包包含所用 dtbench Python源与LICENSE，不含数据/大权重。

```bash
sha256sum -c fe02_lite_source_20261124_v3.tar.gz.sha256
mkdir team_pytorch_fe02_lite
tar -xzf fe02_lite_source_20261124_v3.tar.gz -C team_pytorch_fe02_lite
cd team_pytorch_fe02_lite

# 可选：先载入已经验证可用的旧FE02参数；只复用路径和环境。
source /path/to/existing/fe02_cluster.env
source fe02_lite_cluster.env.example

# 如未载入旧参数，填写实际外部位置。
export FE02_LITE_DATA_ROOT=/actual/corrected/acceleration/root
export FE02_LITE_SPLIT_MANIFEST=/actual/frozen/split_events.csv
export FE02_LITE_DITING_CHECKPOINT=/actual/MAE1200M/mp_rank_00_model_states.pt
export FE02_LITE_OUTPUT_ROOT=/actual/independent/fe02_lite_runs
export FE02_LITE_PARTITION=actual_partition
export FE02_LITE_CONDA_SH=/actual/miniconda/etc/profile.d/conda.sh
export FE02_LITE_CONDA_ENV=existing_working_environment
export FE02_LITE_MODULE_LOADS='existing verified compiler rocm mpi conda modules'
```

也可复制 example 到私人 `.env`，用 `FE02_LITE_ENV_FILE=/actual/lite.env` 指定。环境文件
应包含 example的全部变量/default定义。优先复用原 FE02 的 DTK/MPI/Conda 环境，避免
module purge后漏载MPI。Conda bootstrap显式关闭继承nounset、设置PS1，再恢复nounset。
若当前环境确实不需要模块，用 `FE02_LITE_MODULE_LOADS=none`。

默认4nodes×4devices、每rank microbatch8、有效batch128、23:50h、不写内存申请。
`NODES/DEVICES_PER_NODE/CPUS_PER_TASK/GRES_KIND/MEMORY/WALLTIME/ACCOUNT/QOS/PARTITION`
均以 `FE02_LITE_` 前缀配置。world×microbatch必须整除有效batch。默认concurrency1。
run目录只由rank 0检查/创建，再广播成功/失败，避免其他rank误判成已有目录。
training.distributed_timeout_seconds默认86400秒，以容许rank 0全量validation/诊断；
它不是额外训练预算，Slurm walltime与srun失败退出仍然生效。

默认fresh计算所有年度shard SHA，可能需要较长顺序读取。已有可信 FE01/FE02
`data_split_audit.json` 可显式设 `FE02_LITE_SHARD_IDENTITY_MANIFEST=/actual/file.json`；
仅复用路径/bytes/mtime完全一致的hash，新增shard仍计算，保存来源hash与复用状态，
不复用旧KNET cohort、目标归一化或模型。统计/哈希变化直接失败，不能靠改时间戳绕过。

## 三个默认正式任务

```bash
bash scripts/fe02_lite/submit_knet.sh
bash scripts/fe02_lite/submit_kiknet_borehole.sh
bash scripts/fe02_lite/submit_kiknet_surface.sh

# 等价的一次三任务提交；仅 main_d128 / seed42，默认总并发1。
bash scripts/fe02_lite/submit_all.sh
```

只选其中一种提交方式，避免同run重复排队。脚本默认实际调用sbatch并返回JobID。
三个视图各有独立job name和日志名；共享tasks索引，使用afterany依赖链限制批量并发，
默认前一任务退出后再运行下一视图。CONCURRENCY可增大，单任务失败不会伪装成另一个
视图的成功，也不会阻止后续独立视图产生自己的审计证据。
提交时创建 `OUTPUT_ROOT/launches/fe02-lite-*/`，保存源/依赖/config快照、完整环境、
tasks.json、精确sbatch命令、源码commit与文件hash、sbatch回执。queued worker只运行
快照；原上传目录后续编辑不影响该任务。env/task-map/source哈希变化会拒绝运行。

```bash
bash scripts/fe02_lite/submit_all.sh --dry-run
bash scripts/fe02_lite/submit_kiknet_surface.sh --dry-run
```

dry-run只打印任务映射、环境、计划命令，不创建快照、不加载torch、不扫描数据，
不调用sbatch。登录节点不校验大权重SHA，不做真实模型推理。

计算节点依次登记并核验共同DiTing资产 → 三视图对应data audit → 真实模型/冻结/梯度/
因果/query独立性审计（恢复初始化与RNG）→ 分布式训练 → best/last各自全固定validation
→ 离线汇总。每run flock锁，失败退出保留attempt日志；最后完整成功才写COMPLETED。
init/best/last固定样本诊断由训练阶段导出，默认2GiB上限；不会截断完整评价结果。

## 续训、仅评价与可选消融

```bash
# 必须仍使用相同输出根、数据、权重和代码版本。
bash scripts/fe02_lite/resume.sh --dataset-view kiknet_borehole
bash scripts/fe02_lite/eval_only.sh --dataset-view kiknet_surface

# 可选有限消融，逐次显式选择，不自动展开。
bash scripts/fe02_lite/submit.sh --dataset-view knet --variant amp16 --seed 42
bash scripts/fe02_lite/submit.sh --dataset-view knet --variant d64 --seed 42 --dry-run
bash scripts/fe02_lite/submit.sh --dataset-view knet --variant no_source_aux --seed 42
bash scripts/fe02_lite/submit.sh --dataset-view knet --seed 43
bash scripts/fe02_lite/submit.sh --dataset-view knet --seed 44
```

variants：main_d128、d64、amp16、amp32、pool4、no_source_aux。
`configs/fe02_lite/ablations/` 提供可解析配置；命令行可对任意一个视图显式启用。
source_weight和所有loss权重可配置，改变后属于新身份，不能resume既有run。

resume仅从last恢复，严格匹配world/effective batch/累积/计划预算；中断epoch从上个已
提交epoch边界重放。失败attempt和日志保留。已训练完成但最终评价失败，可以eval_only
补导出；COMPLETED标记存在时不能再训练覆盖。eval_only每次写新带时间戳的目录，不覆盖
原结果。新快照中相同相对YAML路径保持配置身份稳定，不以排队快照绝对路径改变resume。

可选短pilot，不在三个默认任务中。pilot用单节点单device，1epoch、最多2updates、
最多8train/4val事件，有效batch2；若支持cohort不足或无完整validation cell会明确失败。

```bash
FE02_LITE_NODES=1 FE02_LITE_DEVICES_PER_NODE=1 \
  bash scripts/fe02_lite/submit.sh --dataset-view knet --stage pilot
```

## 离线分析、打包和回滚

```bash
# CPU环境可运行，不重新跑DiTing。
python scripts/fe02_lite/run.py analyze --evaluation /actual/run/evaluations/best_TIMESTAMP \
  --run-dir /actual/run --output /actual/new_analysis
python scripts/fe02_lite/collect.py --root "$FE02_LITE_OUTPUT_ROOT" \
  --checkpoint best --output /actual/new_comparison
python scripts/fe02_lite/run.py diagnose --diagnostics /actual/run/diagnostics \
  --output /actual/attention_and_representations
python scripts/fe02_lite/pack_results.py --root "$FE02_LITE_OUTPUT_ROOT" \
  --output /actual/fe02_lite_light_results.tar.gz
# 加入完整逐目标NPZ（仍遵守明确给出的包大小上限，未收入文件均有外部hash索引）。
python scripts/fe02_lite/pack_results.py --root "$FE02_LITE_OUTPUT_ROOT" \
  --include-predictions --max-bytes 536870912 --output /actual/fe02_lite_evidence.tar.gz

# 在干净git checkout生成含真实dtbench源的代码包；不传weights-root。
bash scripts/fe02_lite/pack_source.sh --dtbench-root /actual/ditingbench \
  --output /actual/fe02_lite_source.tar.gz
```

`--include-predictions`只控制轻量传输包；原run评价永远完整保存，无存储cap截断。
传输包的 EXTERNAL_ARTIFACT_INDEX.json 保存未收入原文件的绝对路径、大小、hash与原因。
大权重、波形、全量激活不进Git。

回滚直接停止使用新入口或切回旧worktree；原FE01/FE02/RT55默认行为和checkpoint路径
没有迁移。站点留出和源区留出仅预留版本化配置字段，本轮非空会拒绝运行；普通事件
划分不称为未见场地泛化。held-out test入口未开放。
