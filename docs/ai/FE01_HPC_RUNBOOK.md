# FE01 V2 — 用户手动运行手册

任务 ID `20261001-fe01-native-window-random-time-v2`。本轮 **NOT_SUBMITTED**。
独立分支 `exp/fe01-feature-extractor-site-effects`，基点
`9c95dbfaf92f36b2673d816026cd3f25b91eec66`。工程交付后的精确 SHA 见 CODEX-RESULT
及源码包 SOURCE_IDENTITY.json。所有正式训练、设备审计、验证/test 和逐秒回放由用户
执行下面的命令。代码不会自动 sbatch、重提交流程、转换数据或删除 checkpoint。

## 1. 本地离线权重与源码包

本次已实际下载 PhaseNet/EQT 的官方 STEAD **v2** 权重，SeisBench **0.10.1**。
轻量清单见 reports/fe01_local_20261001/pretrained_manifest.json，二进制位于独立
工作目录 offline_weights/stead_v2。未取得真实 DiTing checkpoint；禁止随机初始化后
标作预训练成功。STEAD/DiTing 训练语料与 Japan 的重叠未知。

重新下载时（显式联网，仅本地准备阶段）：

```bash
.venv-fe01/bin/python scripts/fe01/download_weights.py \
  --output offline_weights/stead_v2_new --name stead --version 2 \
  --remote https://seisbench.gfz.de/mirror/
```

运行期禁止下载。以下在已提交且干净的 FE01 工作目录执行，包含 per-file 哈希锁定
dtbench 外部依赖及其 GPLv3 LICENSE。打包不会修改原工作目录：

```bash
FE01_PACK_PYTHON=.venv-fe01/bin/python bash scripts/fe01/pack_source.sh \
  --dtbench-root ../ditingbench --weights-root offline_weights/stead_v2 \
  --output artifacts/fe01/fe01_source_weights.tar.gz
sha256sum artifacts/fe01/fe01_source_weights.tar.gz
```

选择一个全新的超算目录，手动同步 tar.gz 与 .sha256，校验后解包。源码包自带
vendor/dtbench，不依赖超算的主线目录；不要解包覆盖 RT55、RT59、V01 或旧 FE01 输出。
包中的 offline_weights 是 PhaseNet/EQT 小权重；DiTing 可继续使用超算已存在的只读文件。

## 2. 环境和真实路径

复制 fe01_cluster.env.example 到仓库外的私人 env 文件；填写所有 __FILL__。
FE01_CODE_ROOT、FE01_OUTPUT_ROOT 为绝对独立目录；DATA_ROOT 包含
`2000/japan_2000.hdf5` 至 `2024/japan_2024.hdf5`；FE01_SPLIT_MANIFEST 必须是既有
RT55 frozen split_events.csv，不允许重新划分。FE01_WEIGHTS_ROOT 指向离线包目录。
FE01_SPATIAL_MANIFEST 可先指向将来创建的绝对文件，主实验不会读取它。
partition/account/qos/modules/节点/DCU/CPU/内存/时限均必须按真实账户填写。
未填写字段 fail-fast。FE01_GRES_KIND=dcu 来自旧启动经验，需核对本集群实际资源名称。

使用已有可工作的 PyTorch/ROCm Python 创建专用环境，避免替换 torch 或安装 CUDA：

```bash
# 用真实集群 Python 路径替换这里的 /path/to/working/python
/path/to/working/python -m venv --system-site-packages /path/to/fe01-venv
/path/to/fe01-venv/bin/python -m pip install --no-deps -r requirements-fe01.txt
/path/to/fe01-venv/bin/python -m pip check
```

SeisBench 0.10.1 requires Python >=3.9。项目本地验证 Python 3.11 / torch 2.11 CPU；
集群的已工作 torch/ROCm 版本必须保留并单独核验。若依赖缺失，按 pip check 输出在
专用环境补齐非 torch 包，不运行 pip install -U。设置 FE01_PYTHON 为该环境解释器。
module loads 填写实际 DTK/ROCm/Miniconda 模块；`none` 仅用于已准备好的环境。
联网安装不可用时，在与集群 Python/平台匹配的准备环境运行
`python -m pip download --no-deps -r requirements-fe01.txt -d wheels`，同步 wheel 目录后
使用 `--no-index --find-links wheels --no-deps` 安装。不要拿不匹配平台的本地 wheel 替换
torch/ROCm。只读环境检查：`scripts/fe01/check_environment.py --require-rocm`。

```bash
export FE01_ENV_FILE=/absolute/private/fe01_cluster.env
source "$FE01_ENV_FILE"
cd "$FE01_CODE_ROOT"
"$FE01_PYTHON" scripts/fe01/register_diting.py \
  --manifest "$FE01_WEIGHTS_ROOT/pretrained_manifest.json" \
  --checkpoint /absolute/existing/mae_1200m_checkpoint.pt \
  --source '填写实际来源、训练语料/时段；无法核验的部分写 unknown'
```

注册会保留原始 manifest 备份、计算 SHA256；不会训练，也不表示 keys/shapes 正确。
只有真实 audit 的 strict load 和 forward 通过才证明本地文件可用于该配置。
若需把 DiTing 一并打包，可在另一个离线目录用 `--copy` 注册为相对路径；该文件很大。

## 3. 审计和 pilot

配置已生成，可检查 configs/fe01/runs.tsv。prepare_runs 只写配置，不训练、不提交：

```bash
"$FE01_PYTHON" scripts/fe01/prepare_runs.py
export FE01_STAGE=pilot
bash scripts/fe01/print_submit_commands.sh
```

打印器先创建 FE01_OUTPUT_ROOT/slurm，再打印动态 sbatch 参数，不在 #SBATCH 中使用
shell 变量。用户复制其 **audit 命令** 手动提交。audit 在单设备逐模型检查原生前向、
冻结/公共结构、真实数据/权重 hash、因果边界、100000 sampler draws、query order/
chunk 数值一致性，以及 train-only 统计与共同 validation 人口。
审计会锁定可用事件、排除原因、目标/输入 ID、共同时间、模型支持域和有效配置。
任一项失败必须先解决，不能跳过 audit 开训。

四个 pilot 的索引：0 DiTing，1 TEAM scratch，2 PhaseNet，3 EQT，均 seed42。
audit 成功后，复制打印的 train 命令，将 `--array=0-3` 改成 `--array=0`，先运行单个
pilot 检查 DCU 内存/吞吐/训练下降，再逐个运行 1、2、3。这里只检查接口/资源，
不作正式模型优胜判断。pilot 16 train/8 validation 事件，最多四更新；完整 16 事件
和 batch16 通常产生三更新。保留实际更新/样本计数。

Slurm-direct 内部 srun 的 rank/world/local_rank 与原集群启动方式一致，不使用弹性
torchrun。有效 batch 必须被 microbatch×实际 world_size 整除。默认正式 microbatch8、
batch128 支持 world_size=1/2/4/8/16；pilot microbatch1/batch16。按内存调整 microbatch
时四组保持有效 batch/总更新不变；调整必须发生在 audit 之前并重锁协议。

## 4. 正式 12 训练和独立控制

```bash
export FE01_STAGE=formal
bash scripts/fe01/print_submit_commands.sh
```

同样先手动 audit，再检查每个 AUDIT_PASS 后手动复制 train 命令提交。formal 不继承
pilot 权重；四组公共下游重新初始化。默认 array=0-11：

| Index | Family | Seeds |
|---|---|---|
| 0,1,2 | diting_pretrained_frozen | 42,43,44 |
| 3,4,5 | team_original_scratch | 42,43,44 |
| 6,7,8 | phasenet_pretrained_frozen | 42,43,44 |
| 9,10,11 | eqt_pretrained_frozen | 42,43,44 |

每个 index 有唯一 run_id/audit/output 目录。预算固定 12 epochs、batch128、
3 samples/event/epoch、共同 Adam/cosine/原 loss；epoch 更新 floor(3N/128)，审计后
N 为锁定 cohort。common 1/3/5/10/20 s × normal/random 的 noninput 等权 MAE 选择 best，
并列取早期。PhaseNet prefix 超过25 s、EQT 超过54.99 s 为 unsupported_history；
40/90 s 不进入四组总排名。

只给额外实验显式设置 FE01_INCLUDE_CONTROLS=1：formal 12–14 amplitude_only、15–17
coords_only、18–20 diting_random_frozen，各 seed42/43/44。打印器拒绝自动扩展全矩阵；
用户显式复制单个 array index 的 audit/train 命令。不要默认运行 21 组。

大年度 HDF5 hash可复用已成功 audit 的 data_split_audit.json：设置
FE01_REUSE_DATA_AUDIT 为绝对路径。只允许同一 split 和空间协议、不变文件 size/mtime；
修改数据时重新 hash。首次仍必须完整 hash。真实权重 SHA 每次 build 都重新校验。

## 5. 输出与严格续训

非空输出默认拒绝。last.pth/best.pth/init.pth、optimizer/scheduler、各 rank RNG、
resolved_config、lock、common initial fingerprint、training_curves、sample_journals 均保留。
epoch checkpoint 原子发布；断点从 last 恢复到已完成 epoch，未发布 epoch 重新计算。
实际时间直方图仅统计 checkpoint 引用的 committed journals；失败尝试保留但不重复计数。
cfg/data/weights/code/协议/world_size 不一致拒绝续训，不能用 best 自动回退。

查看 stdout/stderr 和 scheduler 状态由用户执行 `squeue -u "$USER"` / `sacct -j JOBID`。
异常让 Slurm 失败，不会打印成功或自动重提。需要续训时：

```bash
export FE01_RESUME=1
# 手动再次提交同 stage/index/world_size 的 train 命令
# 完成后将 FE01_RESUME 设回 0，避免误用于新 run。
```

## 6. 验证、配对比较、概率和场地统计

默认 eval_job 是 validation，输出新 evaluation_fixed_val。用户手动复制 eval 命令，
把 array=0 改为需要的 indices；不会在训练后自动评价。
也可在用户自己的设备 allocation 中直接执行以下程序（不在登录节点跑大模型）：

```bash
cfg=configs/fe01/formal/formal__diting_pretrained_frozen__seed42.json
run="$FE01_OUTPUT_ROOT/formal__diting_pretrained_frozen__seed42"
"$FE01_PYTHON" scripts/fe01/run.py eval --config "$cfg" --checkpoint "$run/best.pth" \
  --output "$run/evaluation_fixed_val" --device cuda
"$FE01_PYTHON" scripts/fe01/run.py eval --config "$cfg" --checkpoint "$run/best.pth" \
  --random-times-manifest "$FE01_OUTPUT_ROOT/audits/formal__diting_pretrained_frozen__seed42/random_validation_times_manifest.csv" \
  --output "$run/evaluation_random_val" --device cuda
"$FE01_PYTHON" scripts/fe01/analyze.py training-histograms --run-dir "$run" \
  --output "$run/actual_training_time_histograms.csv"
"$FE01_PYTHON" scripts/fe01/analyze.py reference --config "$run/resolved_config.json" \
  --output "$run/train_reference.json"
"$FE01_PYTHON" scripts/fe01/analyze.py summarize --evaluations "$run/evaluation_fixed_val" \
  --reference "$run/train_reference.json" --output "$run/analysis_fixed_val"
```

锁定同 cohort 的一个 reference 可跨主模型复用。空间留出必须重新拟合自己的 reference。
compare 要求同 split/manifest/common population/window/stage 和完全相同的 query/role/input
IDs，缺失 common 决策、数值失败或不配对拒绝排名。给出四/十二个正式 fixed_val 目录：

```bash
"$FE01_PYTHON" scripts/fe01/analyze.py compare --evaluations \
  "$FE01_OUTPUT_ROOT/formal__diting_pretrained_frozen__seed42/evaluation_fixed_val" \
  "$FE01_OUTPUT_ROOT/formal__team_original_scratch__seed42/evaluation_fixed_val" \
  "$FE01_OUTPUT_ROOT/formal__phasenet_pretrained_frozen__seed42/evaluation_fixed_val" \
  "$FE01_OUTPUT_ROOT/formal__eqt_pretrained_frozen__seed42/evaluation_fixed_val" \
  --output "$FE01_OUTPUT_ROOT/comparison_seed42"
```

全 MDN 概率、site residual、稀疏站点数、bootstrap、分层、图源 CSV/PDF/PNG 均由原始
逐目标输出复核。random manifest 单独汇总，不用 fixed-time primary comparator 强行排名。

## 7. 预选案例、区域地图与逐秒回放

cases 在看预测前依据 validation metadata 震级分位、>=5站等规则冻结3–5个事件：

```bash
"$FE01_PYTHON" scripts/fe01/run.py cases --config "$run/resolved_config.json" \
  --output "$FE01_OUTPUT_ROOT/case_selection_manifest.json"
export FE01_CASES_MANIFEST="$FE01_OUTPUT_ROOT/case_selection_manifest.json"
export FE01_RUN_INDEX=0
# 复制打印的 replay_job.sbatch 命令，手动提交。
```

同一案例集用于四模型，不按模型误差挑案例。直接入口：

```bash
"$FE01_PYTHON" scripts/fe01/run.py replay --config "$cfg" --checkpoint "$run/best.pth" \
  --cases-manifest "$FE01_CASES_MANIFEST" --output "$run/replay_prefix" --device cuda
"$FE01_PYTHON" scripts/fe01/render_maps.py --replay "$run/replay_prefix" \
  --reference "$run/train_reference.json" --output "$run/maps_prefix"
```

replay 默认 first_p_pick、1–90每秒、20 km格点，模型查询没有新站点 ID 或伪波形。
保存各帧独立结果、mask NPZ、support、independent-cutoff consistency、整帧P50/P95、
encoder/下游/IO、冷启动、设备内存、理想零通信延迟结果时间。地图/动画只读这些结果；
不支持时点明确留空，最终 PGA 标签保持相同，不从相邻帧插值。
模型与网格/色标一致，真值仅画站点散点，coverage是距离启发式而非可信区保证。
CPU绘图也可由 site_map_job.sbatch 独立执行：设 FE01_REPLAY_DIR、FE01_MAP_OUTPUT、
FE01_REFERENCE_JSON。rendering/归档耗时不属于每帧模型推理时间。

震源时刻回放需加 `--reference origin_time` 并通过真实record/origin绝对时间轴检查。
可用 `--rolling` 输出另一个新目录作为 prefix-trained rolling OOD诊断；不得冒充原生
累计窗口或匹配训练。匹配 rolling 训练不在默认37个配置或首轮预算内。

## 8. 显式 test 与独立空间留出

测试前先完成 validation 选型/协议冻结。用户创建 test_exposure_ledger.json，至少包含
prior_project_exposure（已知历史或 unknown）、authorized_purpose、recorded_by。
只有用户手动显式执行才打开 test；本轮不执行。

```bash
lock_sha=$(sha256sum "$run/protocol.lock.json" | cut -d' ' -f1)
"$FE01_PYTHON" scripts/fe01/run.py eval --config "$cfg" --checkpoint "$run/best.pth" \
  --split test --allow-test --protocol-lock-sha256 "$lock_sha" \
  --test-exposure-ledger /absolute/private/test_exposure_ledger.json \
  --output "$run/evaluation_locked_test" --device cuda
```

空间实验 B 先提供唯一站点/坐标 station_catalog.csv，仅用非标签信息生成：

```bash
"$FE01_PYTHON" scripts/fe01/prepare_manifests.py spatial \
  --station-catalog /absolute/station_catalog.csv --output "$FE01_SPATIAL_MANIFEST"
export FE01_STAGE=spatial
bash scripts/fe01/print_submit_commands.sh
```

独立重新 audit/train，indices与formal主组相同，new run_ids为spatial__...；留出台站在
训练输入/标签/参考到时之前移除，train-only统计重算。validation/test分别只查其留出core，
输入仅train core，buffer20km；event split依旧独立。空空间core/早期支持不足必须报告，
不能改规则来追求好结果。先显式一seed试运行；三seed至少比较DiTing与validation预选
最强竞争者，其余未运行不得出现在空间泛化排名。默认不提交该矩阵。

## 9. 回传与结果封装

先用 `scripts/fe01/collect_runs.py --runs <四/十二个run目录> --output
"$FE01_OUTPUT_ROOT/runset_manifest"` 验证各seed公共初始化/结构、cohort/统计和实际更新/
样本预算并输出 runs_manifest.csv 与 downstream_initial_state_fingerprints.json。
不完整pilot仅可显式 --allow-incomplete，不得据此宣称12组全部完成。

```bash
"$FE01_PYTHON" scripts/fe01/pack_results.py --roots \
  "$FE01_OUTPUT_ROOT/audits" "$run" "$FE01_OUTPUT_ROOT/comparison_seed42" \
  "$FE01_OUTPUT_ROOT/runset_manifest" \
  --output "$FE01_OUTPUT_ROOT/fe01_review_seed42.tar.gz"
```

可多加其余run、spatial manifest所在目录和其他分析目录。包保留所有轻量 JSON/CSV/
manifest/图源/图，生成required evidence inventory（缺项标NOT_PROVIDED）及每文件SHA。
大逐目标 NPZ/CSV默认>10MB只记录外部hash，可用 --max-file-mb 显式放宽；完整checkpoint、
sample_journals、原始波形保留超算，不进审阅包/Git。回传包至少包含审计lock、resolved
configs、actual training time histogram、curves、完整概率/target export或下载位置+hash、
pairedbootstrap、site/站点counts、case manifest、replaygrid/latency/masks/metrics、figure_data。
只给摘要/图片不足以审阅。未知或未跑项保持未知，不补造 HPC 结果。
