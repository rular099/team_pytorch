# TEAM PyTorch 项目交接文档

更新时间：2026-09-30（Asia/Shanghai）

本文档是下一次全新 agent 会话的权威工程入口。当前主任务已从 RT57--RT61
模型结构研究切换到 **V01：速度波形输入与 P 前缺失/补零受控实验**。不要根据旧聊天、
旧 `PROJECT_CONTEXT.md` 的分支名、目录名或 checkpoint 文件名推断当前状态。

## 0. 快速定位和当前结论

- 工作区：`/home/zhangb/work/people/zhangbei/team_claude`
- 当前活跃仓库：
  `/home/zhangb/work/people/zhangbei/team_claude/team_pytorch_query_geometry_diagnostics`
- GitHub：`rular099/team_pytorch`
- 当前分支：`exp/v01-velocity-prep-padding-control`
- V01 基线提交：`9c95dbfaf92f36b2673d816026cd3f25b91eec66`
- V01 核心实现提交：`6a341fb936035d4654db4ddbba974a3cdd471352`
- 本交接前最新代码提交：`04a23a3aa74c81ba18d4625f09e4015d2dc8ae69`
- 远端同名分支已推送到上述最新代码提交。
- 当前工作树有用户未跟踪文件 `tmp.tar.gz`；不要删除、覆盖或提交它。
- 当前最重要事实：**V01 代码和提交脚本已就绪，但尚无一次确认成功的正式 Slurm
  提交，更没有 V01 训练或验证结果。**

超算实际路径：

```text
V01 代码：
  /public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_query_geometry_diagnostics_vel

速度 archive：
  /public/home/test_bigmodel/seismogram/zb/japan_data/hinet_data/archive

速度 catalog：
  /public/home/test_bigmodel/seismogram/zb/japan_data/hinet_data/catalog

加速度年度 HDF5：
  /public/home/test_bigmodel/seismogram/zb/origin_corrected_diting_vel_acc_vs30

RT55 历史运行目录：
  /public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team/weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42
```

当前上传模式应打印的源码清单 SHA-256：

```text
bccb84254bacdd6aa8878bec5deaa17aaa6033139277853bccab1c7184109afe
```

该哈希只适用于超算文件与提交 `04a23a3...` 的 source-manifest 文件逐字节一致时；
正式提交必须以超算 dry-run 实际打印值为准。

## 1. 项目当前目标

### 1.1 当前主任务：V01

V01 要回答两个受控问题：

1. 使用真实 Hi-net 速度计记录作为输入，能否在保持 RT55 模型计算和 PGA 标签定义
   不变的前提下训练可用模型？
2. 在完全相同的速度记录、事件、source/query 身份、空间坐标和决策时刻下，人工删除
   一段 P 前有效样本并正确标为 mask=False，会造成多大 PGA 预测损失？

正式准备三个训练臂：

| arm | 输入 | 目的 |
|---|---|---|
| `vfull` | Hi-net 速度记录，保留全部可用 P 前支持 | 完整速度前缀基线 |
| `vmissing` | 同一速度记录，仅删除模板指定的 P 前前缀 | 核心缺失干预 |
| `apair` | 与速度 source 配对的实测加速度记录 | 跨输入域桥接控制 |

两个速度 checkpoint 要在 `vfull`/`vmissing` 两个视图上交叉验证，并分别跑固定
normal/random validation。目标仍是原 KNET 查询台站 PGA，坐标为 `log10(m/s^2)`；
不是 PGV，也不是速度记录微分出的新标签。

### 1.2 当前不是要做的事

- 不继续 RT62；V01 明确暂停了上一轮独立 RT62 任务。
- 不继续相同设置训练 RT59/RT60/RT61。
- 不打开 held-out test 来选择 V01 协议、epoch、mask 强度或阈值。
- 不把 V01 当作 RT61 性能升级；它是独立的数据机制实验。

## 2. 当前完成状态

### 2.1 V01 已完成的实现

已实现并推送：

- `tools/velocity_waveform_backend.py`
  - 读取年度 Hi-net CNT/CH archive；
  - 每个事件一次解码所需通道；
  - 按 CH `counts_per_physical_unit` 将 raw counts 转成 m/s；
  - 明确标记 `response_correction=sensitivity_only`，不声称完整去仪器响应；
  - E/N/U 三分量顺序；进程独立 reader 和有界打开句柄。
- `tools/prep_padding_protocol.py`
  - train-only P 前支持模板；
  - event/source 稳定哈希分配；
  - 只删除严格早于 P 的样本；P 点和 P 后支持不变；
  - 填零并设置 mask=False。
- `tools/build_v01_paired_manifest.py`
  - 先继承冻结 RT55 split，再筛速度交集；
  - test event 在打开波形前排除；
  - source 与 query 使用各自真实传感器身份和坐标；
  - 同一 Hi-net source 去重，不能复制成多台输入；
  - 生成 velocity 和 A-pair 两套 derived cache；`vmissing` 是运行时视图，不复制第三套波形；
  - 保存单位、source hash、配对、计划和干预 provenance。
- `gemini_util_light.py`
  - 新 metadata-support、V01 干预和 source-role 路径均默认关闭；
  - 干预发生在中心化/归一化之前；
  - V01 可用显式 support mask，不用幅值阈值决定速度台站是否存在；
  - 导出 P 前/P 后有效秒数和 V01 provenance。
- `loader_light.py`
  - 新增显式 frozen split manifest；旧默认 splitter 不变。
- `train_light.py`、`eval_checkpoint.py`
  - 只增加 V01 split/provenance 透传；继续复用原模型、损失和正式评估路径。
- `pga_configs/v01_*.json`
  - 三臂同一 RT55 `model_params`、RT56 mixed geometry、seed 42、8 新 epoch、
    LR/adapter LR/TEAM LR 均为 1e-4。
- `tools/analyze_v01_padding_controls.py`
  - 汇总点误差；做 5,000 次 event-cluster paired bootstrap，seed 20260915。
- `tools/run_v01_prep_padding_controls_slurm.sh`
  - `ACTION=preflight|train|eval|analyze|all`；
  - `ARMS=vfull,vmissing,apair`；
  - 默认 dry-run，正式运行要求 `CONFIRM_V01=1 DRY_RUN=0`；
  - `all` 使用 `afterok` 串联 preflight、三个训练、十个 validation eval 和分析；
  - 非空输出拒绝覆盖，resume 必须显式开启；
  - 只跑 validation，不调 test。

`gemini_models.py`、DiTing 源码和所有 RT55--RT61 配置均未修改。V01 实现时记录：

```text
gemini_models.py SHA-256:
  cd8481286436342d09781888967dc757f4bde383f4d344033d866c6b06b7be84

canonical RT55/V01 model_params SHA-256:
  d0ee04ee99872c2eeed644c1ce05ec822b311eb610b58c732465d024bcc1ef6b
```

### 2.2 已核验的 V01 数据状态

本地速度归档快照在 2026-09-29 核验为：

- 外置盘 `/dev/sdd1` 以 `fuseblk,ro` 只读挂载；
- 没有实际 Hi-net 下载进程；最后更新时间为 2026-08-11；
- 14,153 requested，12,708 committed；其中 12,392 有波形，316 无匹配台站；
- 1,445 个事件未归档；
- archive 约 26 GiB，catalog 约 14 MiB，加速度数据约 60 GiB；
- 12 个完整年度：2006、2008、2009、2010、2011、2014、2015、2017、2019、
  2020、2021、2023；
- 13 个 partial 年度：2000--2005、2007、2012、2013、2016、2018、2022、2024；
- 2000--2003 没有可用波形，V01 配置只引用 2004--2024。

用户已明确报告 2004--2024 HDF5 archive 和 catalog 已上传到上述超算路径。
partial archive 在 V01 中被当作固定只读快照，并由哈希固定；不要边下载边训练。

真实 metadata-only V01 audit：

| item | count |
|---|---:|
| train events | 8,439 |
| validation events | 1,225 |
| source-event sensor rows | 129,636 |
| KNET query targets | 153,462 |
| KNET-compatible source rows | 2,094 |
| KiK/Hi-net bridge source rows | 127,542 |
| frozen test events excluded before waveform open | 2,483 |
| train-only P-prefix templates | 146,842 |

这些是 metadata/preflight 数，不是最终训练数；完整 materialization 可能因三分量或
channel 校验减少样本。模板保留 P 前支持的中位数为 7.21 s；小于 1/3/5 s 的比例约
3.21%/16.10%/32.78%，因此一部分样本的实际干预剂量较小，必须据实报告。

### 2.3 本地验证状态

V01 核心实现后实际运行：

```text
python -m unittest tests.test_v01_velocity_backend tests.test_v01_padding_controls
  -> 10 tests, PASS

python -m unittest discover -s tests -p 'test_*.py'
  -> 118 tests, PASS

python -m py_compile ...V01 and modified runtime files...
  -> PASS

bash -n tools/run_v01_prep_padding_controls_slurm.sh
  -> PASS

DRY_RUN=1 ACTION=all ARMS=vfull,vmissing,apair ...
  -> preflight + 3 train + 10 eval + analyze dependency graph, PASS

one-event 2024 cache materialization and generator read
  -> PASS
```

后续路径、arm 解析和资源修订只改了 launcher/docs；每次都重新执行了 `bash -n` 和
完整三臂 dry-run。没有在最后几次路径修订后重跑 118 项 Python 测试，因为 Python
运行时未变。

### 2.4 超算执行状态

截至 2026-09-30，用户遇到并依次报告：

1. `Unknown arm: vfull`
   - 原因：目标集群旧 Bash 对 `[[ -v 'assoc[key]' ]]` 支持不可靠；
   - 修复提交 `80a539e95bbe46708387738b66a53b6eb8e07222`；
   - 改用显式 `case`，单臂/双臂/三臂均 dry-run 通过。
2. RT55 split/中间结果默认指向了错误项目；
   - 正确项目为 `team_pytorch-zhangb-diting-backbone-attnpool-team`；
   - 修复提交 `492465c3013078d023f45b8cb95aefb84262c0cb`。
3. `Memory specification can not be satisfied`；
   - 原 preflight 默认请求 192000M；目标分区无法满足；
   - 修复提交 `04a23a3aa74c81ba18d4625f09e4015d2dc8ae69`；
   - 当前默认 `PREFLIGHT_CPUS=8`、`PREFLIGHT_MEM=102400M`。

第三个错误发生在第一个 `sbatch` 提交阶段，按脚本 `set -e` 逻辑后续训练不会提交。
但接手者仍须在超算用 `squeue`/`sacct` 核验，不能仅凭本地推断宣称无残留作业。
目前没有用户确认最新脚本已覆盖、最新 dry-run 哈希匹配或正式 preflight 已进入队列。

### 2.5 RT57--RT61 已完成研究链的结论

这些实验全部是固定 validation、单 seed 42；没有打开 held-out test。它们是 V01 的
研究背景，不是 V01 的 parent 选择依据。

| 实验 | 主要结果 | 决策 |
|---|---|---|
| RT57 GTNP v2 | random MAE 0.257229 -> 0.247857；normal 保留；单站空间范围明显改善但仍严重压缩；6 个 gate 中 5 个通过，slope 未过 0.40 | 保留成功结果，但未全 GO |
| RT58 waveform-anchor transfer | 相对 gamma=1.66 base 的新 anchor correction 反而使 random MAE +0.000458；单站 range ratio 下降 | 不继续当前 anchor 设计 |
| RT59 dual-objective transport v3 | random MAE 0.245578 -> 0.237828；normal non-input MAE 0.210371 -> 0.199301；空间 range 仍退化，22/29 gate | 当前最强 pointwise development parent；总体 NO-GO |
| RT60 final contrast readout | 单站空间指标有极小改善，但 normal remote 指标退化；机制 8/14，legacy 18/25 | 负结果，保留 RT59 |
| RT61 wave-geometry residual | random MAE 小幅改善到 0.237324；单站 pairwise 仅改善 0.385%，normal CI 不稳；机制 13/14、useful progress 0/4 | 不采用，保留 RT59 reference |

详细证据入口：

```text
docs/ai/CODEX_RESULT_20260912_RT57_GTNP_V2_VALIDATION.md
docs/ai/CODEX_RESULT_20260915_RT58_WATF_VALIDATION.md
docs/ai/CODEX_RESULT_20260920_RT59_DUAL_OBJECTIVE_TRANSPORT_V3_VALIDATION.md
docs/ai/CODEX_RESULT_20260923_RT60_CONTRAST_READOUT_VALIDATION.md
docs/ai/CODEX_RESULT_20260928_RT61_WAVE_GEOMETRY_VALIDATION.md
reports/rt57*  reports/rt58*  reports/rt59*  reports/rt60*  reports/rt61*
```

## 3. 最近的重要修改

### 3.1 V01 实现文件

| 文件 | 改动 | 原因 |
|---|---|---|
| `tools/velocity_waveform_backend.py` | archive 探测、CNT 解码、灵敏度转换、worker reader pool | 原训练 HDF5 与 Hi-net archive schema 不同 |
| `tools/prep_padding_protocol.py` | 稳定模板分配和 mask-first P 前删除 | 构造严格配对缺失干预 |
| `tools/build_v01_paired_manifest.py` | split 继承、source/query 分离、两套 cache、provenance | 防 split 漂移、坐标伪造和波形复制 |
| `gemini_util_light.py` | V01 opt-in intervention/support/source-role/provenance | 复用旧 generator 且保持 RT55 默认 |
| `loader_light.py` | frozen split manifest | 过滤前固定原事件划分 |
| `train_light.py`、`eval_checkpoint.py` | V01 透传和导出 | 复用训练/评估主链 |
| `pga_configs/v01_*.json` | 三臂和 normal/random validation 配置 | 固定实验合同 |
| `tools/analyze_v01_padding_controls.py` | point metrics 和 paired cluster CI | 固定基础统计 |
| `tools/run_v01_prep_padding_controls_slurm.sh` | 全流程 Slurm 编排 | 用户在超算一键提交 |
| `tests/test_v01_*.py` | 单位、mask、配对、cutoff、bootstrap 测试 | 防数据契约回归 |
| `docs/v01_velocity_prep_padding.md` | 数据路径、运行和回传说明 | 超算操作入口 |

### 3.2 最近四个 launcher 修复

| commit | 修复 |
|---|---|
| `0411b17` | 代码默认路径改为 `_vel`，速度根改为 `japan_data/hinet_data` |
| `80a539e` | 兼容旧 Bash 的 arm 解析 |
| `492465c` | RT55 产物根改到 `team_pytorch-zhangb-diting-backbone-attnpool-team` |
| `04a23a3` | preflight 从 16 CPU/192000M 降为 8 CPU/102400M |

路径或 launcher 再变化时，source-manifest 哈希也会变化。不得继续使用旧哈希
`a9ce...`、`72ca...`、`7cfc...` 或 `3179...`。

## 4. 当前架构和设计决策

### 4.1 不修改模型计算

V01 固定 RT55 模型参数和 forward；不修改 `gemini_models.py`、DiTing、attention、
readout、MDN、loss、内部幅值计算或 tensor shape。速度输入是数据协议变化，不是新模型。
所有 V01 开关默认关闭，因此旧 RT55 loading/inference 路径应保持不变。

### 4.2 source 和 query 是两个独立实体

- 输入 source 使用真实 Hi-net/配对加速度传感器 ID 和真实 source 坐标；
- PGA query 使用原 KNET 目标 ID、坐标和标签；
- 近邻匹配只表示 paired-site，不等于同一传感器或直接观测目标；
- query-only 行不能进入 waveform input slot；
- 同一速度波形不能复制成多个输入台站。

这是硬约束。不要为扩大样本量伪造坐标、扩大匹配半径或把 surface/borehole 当同一站。

### 4.3 单位和响应

Hi-net CH `counts_per_physical_unit` 只做灵敏度换算：输出是 m/s 的
`velocity_sensor_output`。仪器频率响应形状仍在；这不是完整反褶积后的宽频真实地动速度。
A-pair 与速度的比较还包含仪器、深度、场地、频响和输入域差异，不能单独归因于 padding。

### 4.4 干预和时间协议

- `vfull` 与 `vmissing` 共用同一 velocity cache；
- 模板只来自 frozen train split；
- 只删 `time < P - retained_preP` 的原有效样本；
- P 点、P 后样本和原 storage 缺口不改变；
- 删除发生在中心化/归一化之前，填零且 mask=False；
- 两视图共享 event/source/query、absolute cutoff 和 crop anchor；
- test event 不物化、不评估、不参与模板。

### 4.5 训练合同

- 同一 RT55 epoch-32 weight-only 起点；
- 新 optimizer/scheduler/epoch/best state；
- DiTing encoder 冻结，按 RT55/RT56 规则训练其余原模块；
- seed 42，8 个新 epoch，固定 epoch 8 比较；
- 50% normal-style + 50% causal-random train；
- LR/adapter LR/TEAM LR 都是 1e-4；
- 相同 batch/world-size/update 计划；
- validation normal/random，绝不自动触发 test。

### 4.6 缓存和 I/O tradeoff

preflight 会为 2004--2024 生成 velocity 和 A-pair 两套 derived HDF5，并计算源 archive
和加速度 shard 哈希。这样训练 I/O 可控且复现性强，但首次 preflight 可能耗时、占用大量
临时/永久空间。`vmissing` 不复制第三套大文件。不要在 preflight 未完成时手工启动训练。

## 5. 未完成事项（按优先级）

### P0：确认最新 launcher 已部署并成功提交 preflight

1. 在超算覆盖最新 `tools/run_v01_prep_padding_controls_slurm.sh`。
2. 核验 split、ep32 checkpoint、21 个 2004--2024 archive 和可用空间。
3. 检查是否有 V01 残留 job 或半成品输出。
4. 用 uploaded-sha256 模式 dry-run；实际 hash 必须为当前上传内容打印的值。
5. 正式提交 `ACTION=all ARMS=vfull,vmissing,apair`，保存所有 job IDs。

### P0：监控 preflight，而不是重复提交

- 检查 `v01-preflight` 的日志、运行时间、MaxRSS、磁盘增长；
- 如果因 23:50:00 超时，先保留 `.tmp`/日志并分析 builder 是否支持安全续跑；当前 builder
  默认拒绝覆盖，不要直接 `--overwrite` 或删目录；
- 如果内存仍不满足，可显式降至 `PREFLIGHT_MEM=96000M`，但先看分区节点配置；
- preflight 成功后核验 `protocol_lock.json`、`preflight_summary.json` 和实际 cohort counts。

### P0：训练和验证完成后回传产物

至少回传：

```text
V01_RUN_ROOT/derived_cache/protocol_lock.json
V01_RUN_ROOT/derived_cache/preflight_summary.json
V01_RUN_ROOT/derived_cache/cohort_counts_by_year.csv
V01_RUN_ROOT/derived_cache/cohort_audit.csv
V01_RUN_ROOT/weights_*/config.json
V01_RUN_ROOT/weights_*/split_events.csv
V01_RUN_ROOT/weights_*/split_stations.csv
V01_RUN_ROOT/weights_*/full_model_init.pth metadata/hash
V01_RUN_ROOT/weights_*/full_model_last.pth metadata/hash
V01_RUN_ROOT/weights_*/训练日志和 scalar CSV
V01_RUN_ROOT/eval/*.{npz,metrics.json,txt}
V01_RUN_ROOT/report/*
V01_RUN_ROOT/logs/*
sacct 表
```

不要把 raw Hi-net 波形或大 checkpoint 直接推到 GitHub；可把结果包放 `chaosuan_res/`，
Git 只保存轻量摘要、哈希、表和批准公开的固定案例。

### P1：补齐 V01 规范中的低成本诊断

当前尚未自动化：

- production FullModel masked-filler invariant；
- zero-as-valid diagnostic；
- P 前固定 0/1/3/5/full dose inference；
- A-historical stage-0 参考；
- 完整概率、空间 field、calibration 图表和全部分层表。

这些是明确缺口，不得在结果报告中写成已完成。优先先让核心三臂正常运行；不要重新加一串
smoke。核心结果回来后，根据主效应和 ChatGPT 审阅决定最小补充诊断。

### P1：完善分析器

当前 `tools/analyze_v01_padding_controls.py` 主要提供 MAE/RMSE/bias/tail/within-threshold
和两类 paired MAE CI。最终报告还需核验 eval NPZ 中的 MDN 输出、NLL/Brier/coverage、
common-remote 分组、field range/pairwise difference、干预剂量和 event-macro 稳健性。

### P2：数据下载

当前本地盘只读且下载停止。V01 已上传固定快照，不要在计算节点恢复下载。若未来补齐
1,445 个未归档事件，应先把盘安全重挂为可写并独立恢复 downloader；新快照必须用新
数据身份，不能静默替换正在使用的 V01 archive。

## 6. 已知问题和风险

1. **HPC 成功状态未知。** 最后一次用户反馈仍是 `sbatch` 内存规格失败；最新资源修复后
   没有收到成功 job ID。
2. **超算是上传目录，可能没有 `.git`。** 应使用 `SOURCE_IDENTITY_MODE=uploaded_sha256`，
   不能要求 `EXPECTED_GIT_COMMIT`。
3. **旧脚本哈希全部失效。** 当前预期是 `bccb...`，但仍以超算打印值为准。
4. **preflight 是重量级物化，不只是轻量审计。** 它会读/哈希大文件并写两套 cache；
   可能受时限和空间限制。
5. **partial archive 是不完整数据快照。** 允许使用不代表可称全量完整 Hi-net 数据。
6. **A-pair 主要是 KiK/Hi-net bridge。** 129,636 source rows 中只有 2,094 是
   KNET-compatible；不能称为历史 RT55 KNET-only 对照。
7. **单 seed、短预算。** V01 的 8 epoch 只比较受控条件，不证明速度或加速度各自充分
   训练后的性能上限。
8. **干预剂量不均。** 一些样本原本没有足够 P 前上下文，不能把 nominal 模板时长当
   实际删除时长。
9. **当前分析器不满足完整论文级 V01 规范。** 不能只凭 `cross_eval_metrics.csv` 宣称
   padding 是主要瓶颈。
10. **旧 `PROJECT_CONTEXT.md` 是 2026-09-02 的 RT55/RT56 快照。** 其指标仍可作为背景，
    但其中活跃分支和当前任务已过期。
11. **RT57--RT61 validation 被反复用于开发。** 不把这些 validation 结果写成独立泛化
    或 held-out test 证据。
12. **结果包常不含 checkpoint body。** 以前 RT58--RT61 报告只能核验导出和 metadata；
    V01 回传时要保存 checkpoint hash/epoch，不根据文件名认身份。
13. **可选 xFormers/Apex 警告通常不是致命错误。** 判断失败要看最终 traceback 和退出码。
14. **时间上限存在站点隐藏约束。** 过去 `3-00:00:00` 实际约一天被杀；V01 默认
    `23:50:00` 是有意规避，不要因 partition 显示 unlimited 就假设可跑多天。
15. **工作树有 `tmp.tar.gz`。** 它属于用户，保持未跟踪；禁止 `git add -A`。

## 7. 下一会话第一步

第一步不是改模型，而是在超算确认最新 V01 launcher、输入文件和队列状态。

```bash
cd /public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_query_geometry_diagnostics_vel

export WORKDIR=$PWD
export ACC_DATA_ROOT=/public/home/test_bigmodel/seismogram/zb/origin_corrected_diting_vel_acc_vs30
export VELOCITY_DATA_ROOT=/public/home/test_bigmodel/seismogram/zb/japan_data/hinet_data
export RT55_RUN_ROOT=/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team/weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42
export FROZEN_SPLIT_MANIFEST=$RT55_RUN_ROOT/split_events.csv
export RT55_EP32_CHECKPOINT=$RT55_RUN_ROOT/full_model_best_ep32.pth
export V01_RUN_ROOT=$WORKDIR/v01_velocity_prep_padding_seed42
export SOURCE_IDENTITY_MODE=uploaded_sha256

test -s "$FROZEN_SPLIT_MANIFEST"
test -s "$RT55_EP32_CHECKPOINT"
find "$VELOCITY_DATA_ROOT/archive" -maxdepth 1 -type f -name 'hinet_raw_*.h5' | sort
df -h "$WORKDIR" "$VELOCITY_DATA_ROOT"
squeue -u "$USER" | grep -E 'v01|JOBID' || true
find "$V01_RUN_ROOT" -maxdepth 2 -type f -printf '%p %s\n' 2>/dev/null | head -n 100
```

若没有运行作业且没有需要保留的非空 V01 输出，先 dry-run：

```bash
unset EXPECTED_SOURCE_MANIFEST_SHA256

DRY_RUN=1 ACTION=all ARMS=vfull,vmissing,apair \
  bash tools/run_v01_prep_padding_controls_slurm.sh
```

确认打印的 source manifest 和资源为：

```text
source_manifest_sha256=bccb84254bacdd6aa8878bec5deaa17aaa6033139277853bccab1c7184109afe
v01-preflight: cpus-per-task=8, mem=102400M, time=23:50:00
```

再正式提交：

```bash
export EXPECTED_SOURCE_MANIFEST_SHA256=bccb84254bacdd6aa8878bec5deaa17aaa6033139277853bccab1c7184109afe

CONFIRM_V01=1 DRY_RUN=0 ACTION=all ARMS=vfull,vmissing,apair \
  bash tools/run_v01_prep_padding_controls_slurm.sh
```

立即记录脚本打印的 preflight/train/eval/analyze job IDs。若目标只是先验证 cache，可用
`ACTION=preflight`；不要同时再提交 `ACTION=all` 造成重复物化。

## 8. 不要做什么

- 不要修改 `gemini_models.py`、DiTing、PGA 标签、MDN/loss 或 RT55--RT61 原配置来绕过
  V01 数据问题。
- 不要用 RT59/RT61 checkpoint 替换预注册的 RT55 ep32 V01 起点。
- 不要在过滤速度交集后重新随机 split；必须先继承 frozen RT55 split。
- 不要读取 test 波形建模板、调协议、挑 epoch、选阈值或写主结论。
- 不要把 validation 结果改名为 test。
- 不要把 raw counts 直接叫 m/s，也不要对 sensitivity-only 数据声称完整去响应。
- 不要把速度 source 坐标替换成 KNET query 坐标，不要复制一个速度波形成多台输入。
- 不要把 A-pair 冒充 KNET-only 或同仪器严格对照。
- 不要边下载边训练，不要上传/使用 `.lock`，不要原地改写 partial archive。
- 不要看到 preflight 超时就删除 cache 或加 `--overwrite`；先审计临时文件和可恢复性。
- 不要重复此前已经完成的 query-geometry smoke、waveform/station mismatch smoke 或 RT57--RT61
  大矩阵诊断。
- 不要自动增加 epoch、扫 LR/mask/dose/seed，或根据 validation 继续事后调参。
- 不要只报告总体 MAE；至少区分 protocol、时间、input/non-input/common-remote、事件数和
  target 数，并同时看概率与空间指标。
- 不要执行 `git reset --hard`、`git checkout --`、批量删除、`git add -A`。
- 不要删除或提交用户未跟踪的 `tmp.tar.gz`。

## 9. 关键上下文速记

### 9.1 新会话阅读顺序

1. `AGENTS.md`
2. 本 `SESSION_SUMMARY.md`
3. `docs/ai/V01_prompt.md`
4. `docs/ai/V01_VELOCITY_PREP_PADDING_CODEX_PROMPT_20260929.md`
5. `docs/ai/CODEX_RESULT_20260929_V01_VELOCITY_PADDING.md`
6. `docs/v01_velocity_prep_padding.md`
7. V01 launcher、config、builder、backend、tests

`docs/ai/PROJECT_CONTEXT.md` 只作为 RT55/RT56 历史快照；不要用其中旧分支覆盖本交接。

### 9.2 Repo 结构

```text
team_pytorch_query_geometry_diagnostics/
  AGENTS.md
  SESSION_SUMMARY.md
  train_light.py
  eval_checkpoint.py
  gemini_util_light.py
  gemini_models.py
  loader_light.py
  train_light_slurm.sh
  eval_checkpoint_slurm.sh
  pga_configs/v01_*.json
  tools/velocity_waveform_backend.py
  tools/prep_padding_protocol.py
  tools/build_v01_paired_manifest.py
  tools/analyze_v01_padding_controls.py
  tools/run_v01_prep_padding_controls_slurm.sh
  tests/test_v01_velocity_backend.py
  tests/test_v01_padding_controls.py
  docs/v01_velocity_prep_padding.md
  docs/ai/
  reports/
```

### 9.3 Launcher 常用变量

```text
WORKDIR, ACC_DATA_ROOT, VELOCITY_DATA_ROOT, RT55_RUN_ROOT
FROZEN_SPLIT_MANIFEST, RT55_EP32_CHECKPOINT, V01_RUN_ROOT, V01_CACHE_ROOT
SOURCE_IDENTITY_MODE, EXPECTED_SOURCE_MANIFEST_SHA256, EXPECTED_GIT_COMMIT
ACTION, ARMS, DRY_RUN, CONFIRM_V01, RESUME_V01, ALLOW_EXISTING_EVAL
SLURM_PARTITION, SLURM_GRES_RESOURCE, SLURM_ACCOUNT
PREFLIGHT_CPUS, PREFLIGHT_MEM, PREFLIGHT_TIME
TRAIN_NODES, TRAIN_GPUS_PER_NODE, TRAIN_TIME
EVAL_GPUS, EVAL_TIME, SLURM_CPUS_PER_TASK, SLURM_MEM
CONDA_ENV, MODULE_UNLOAD, MODULE_LOADS, DITING_CONFIG, DITING_PRETRAINED
```

默认资源：preflight 1 node/8 CPU/102400M/23:50；train 4 nodes x 4 DCU、
8 CPU/task、102400M/node、23:50；eval 1 DCU、12:00。

### 9.4 Resume 和输出保护

- `ACTION=all` 在 preflight 已生成非空 cache 时可能拒绝覆盖；不要盲目重复。
- 训练 resume 只在对应 arm 输出非空且存在 `full_model_last.pth`、`config.json` 时使用：

  ```bash
  CONFIRM_V01=1 DRY_RUN=0 ACTION=train ARMS=vfull RESUME_V01=1 \
    bash tools/run_v01_prep_padding_controls_slurm.sh
  ```

- resume 前要核验 protocol/parent/source/data hashes；当前脚本的 resume 身份保护仍不等于
  完整科学审计。

### 9.5 结果统计约定

- PGA/error 坐标：`log10(m/s^2)`；
- point estimate：该坐标上的 MDN predictive mixture mean；
- 核心时间：normal/random 的 1/3/5 s，同时保留 10/20/40/90 s 描述；
- bootstrap：event ID 聚类，5,000 draws，seed 20260915；
- 正 paired key 至少包含 event、decision time、query target 和 protocol；
- 同事件全部时刻/台站应一起重采样；不能按 target 独立 bootstrap；
- 单 seed CI 不代表训练 seed 不确定性。

### 9.6 Git 操作

```bash
git status --short --branch
git diff --check
git log -8 --oneline --decorate
```

当前远端活跃分支为 `exp/v01-velocity-prep-padding-control`。GitHub SSH 22 端口在本环境
不可用时，已验证可通过 `ssh.github.com:443` 推送。只暂存本任务文件，保留
`tmp.tar.gz` 未跟踪。

### 9.7 权威文档

```text
docs/ai/V01_prompt.md
docs/ai/V01_VELOCITY_PREP_PADDING_CODEX_PROMPT_20260929.md
docs/ai/V01_TASK_SOURCE_HASHES.md
docs/ai/CODEX_RESULT_20260929_V01_VELOCITY_PADDING.md
docs/v01_velocity_prep_padding.md
```

若实际超算文件、job 状态或结果与本文不同，优先相信可核验的 `squeue/sacct`、日志、
resolved config、checkpoint metadata、NPZ/metrics 和文件哈希，并立即更新本交接文档。
