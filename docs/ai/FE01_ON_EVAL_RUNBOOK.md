# FE01 ON V4：本地交付与超算操作手册

任务 `20261021-fe01-on-integrated-warning-map-v4`。实施在独立工作区
`team_pytorch_fe01_on_eval`、分支 `exp/fe01-on-warning-map-evaluation`；基点
`29ffb20ba0effc0c345a0329ba3f8c658dc8cef0`。本轮正式超算状态 **NOT_SUBMITTED**。

## 1. 哪些已经在本地做，哪些需要你在超算做

本地完成实现、已有 Japan validation 保存预测的 CPU 统计、聚焦测试、合成动态地图/TTUM/shadow。
本地没有生产 checkpoint、Japan HDF5 或 DCU；没有把合成数值当成 Japan 新前向结果。
新的 inventory、波形补数据、模型 smoke/pilot、逐秒回放、时延、render/pack 均由下面的 **sbatch 入口**在超算计算。
prepare/status 只在登录节点准备/检查，不提交作业，不导入 Torch。

你只需复制新源码 release 的 `.tar.gz` 和同名 `.sha256` 到项目根目录。
`SOURCE_RELEASE.json` 可一起复制便于核对版本。**不用复制权重或数据，也不用覆盖旧代码。**
源码包解压到独立 `on_eval_v4_<result_commit前12位>/`。下面的 `V4` 指向这一个目录。
最后交付消息会给出本批真实文件名和 V4 的准确赋值；不是让你猜 commit。

## 2. 超算登录节点：第一次准备

所有操作都在你既有的项目目录内：

```bash
cd /public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_fe01
conda activate zb
```

用交付消息中的实际包名执行 `sha256sum -c ...tar.gz.sha256`，再执行
`tar --keep-old-files --no-same-owner -xzf ...tar.gz`。`--keep-old-files` 会拒绝覆盖已有文件。
按交付消息设置 `V4="$PWD/on_eval_v4_<实际commit前12位>"` 后，执行：

```bash
bash "$V4/scripts/fe01_a01_compare/prepare.sh" \
  --env "$PWD/on_eval.private.env" --run-id on_v4_20261022 \
  --inherit-session "$PWD/outputs/a01_eval_20261014T102355__51fzm5p"

bash "$V4/scripts/fe01_a01_compare/status.sh" \
  --env "$PWD/on_eval.private.env" --run-id on_v4_20261022
```

首次 prepare 自动创建私有 `on_eval.private.env`，复用旧 session 的
`runtime.private.env`、`A01_CODE_ROOT/vendor`、既有 DTK/MPI/conda 参数及 `A01_SINGLE_GRES`。
不会改旧文件，不复制大权重，不安装软件。再次 prepare 只核对同一个 plan。
如果这两个私有文件实际不在上述位置，prepare 会明确停止；指定它们的真实位置，不能拿脱敏报告中的 PRIVATE_PATH 替代。

prepare 打印：1310 个事件、真实数组任务数、模型加载次数、Q/网格点数分布及预计文件数。
默认每事件块32个事件；完整 station 数值阶段328项，normal grid164项；每项只加载一次对应模型。
这些编号由程序管理，**不需要你手工选择数组下标**。最大209600是模型×事件×时刻×几何帧上界，不是独立样本数。
网格布局在看预测前确定，高程支撑由 CPU sidecar 补齐。最终6个 pilot/12个报告案例在 CPU 元数据齐全后、第一次新模型前向之前冻结，考虑K、实际记录长度、Q和质量等，不按预测误差选案例。

本批默认并发8个任务，每任务1个DCU。可以在手动 submit 前设
`ON_EVAL_MAX_CONCURRENT=16`，或按 pilot 调内存/时间；不改变科学人口或固定时刻。
`ON_EVAL_BLOCK_EVENTS` 只在首次 prepare 有效，之后任务表已冻结。

## 3. 先提交四个独立 CPU 阶段

以下每条命令只提交它自己指定的阶段，不会自动连跑后续：

```bash
bash "$V4/scripts/fe01_a01_compare/submit_stage.sh" --env "$PWD/on_eval.private.env" --run-id on_v4_20261022 --stage inventory
bash "$V4/scripts/fe01_a01_compare/submit_stage.sh" --env "$PWD/on_eval.private.env" --run-id on_v4_20261022 --stage compare_existing
bash "$V4/scripts/fe01_a01_compare/submit_stage.sh" --env "$PWD/on_eval.private.env" --run-id on_v4_20261022 --stage covariates
bash "$V4/scripts/fe01_a01_compare/submit_stage.sh" --env "$PWD/on_eval.private.env" --run-id on_v4_20261022 --stage crossing_truth
```

- inventory：CPU节点逐个读 checkpoint，核内部epoch、完整字节SHA、config/lock/实际tensor与encoder身份。TEAM核训练后的encoder，不要求回到初态。
- compare_existing：复用旧四ON selected/epoch12 CSV，主端点/配对/分层/交互；不等缺权重问题，也不需要DCU。
- covariates：41个CPU块，读取因果前缀，补SNR、mask有效时长、UTC/origin、共同输入计划、固定站点与网格元数据。
- crossing_truth：41个CPU块，完整目标记录只用于离线评分，核PGA物理定义与首次越阈/删失。

数据散列有新run内的共享锁/缓存；同一实际文件stat身份不会被82个CPU任务重复完整散列。
原数据仍只读。复制后mtime变化时先独立核对字节SHA，不用文件名代替身份。

## 4. 等门通过，再提交 DCU smoke 和6事件 pilot

先重复第2节的 status。inventory 和 covariates 全部 `COMPLETE_SEAL_VERIFIED` 后：

```bash
bash "$V4/scripts/fe01_a01_compare/submit_stage.sh" --env "$PWD/on_eval.private.env" --run-id on_v4_20261022 --stage model_smoke
```

model_smoke 四项分别核四ON；测试原snapshot重叠点、完整MDN、query顺序/分块/追加grid、同截止缓存、未来NaN/Inf/脉冲、合法最后采样和label-only控制。
重叠点输入计划不同会保存差异，不能冒充同协议复现。某个门失败要先看日志，不能跳过门。
DCU可见性只在真正分配的 `srun` step 内检查，登录节点没有DCU不影响 prepare。

model_smoke 四项均完整通过后：

```bash
bash "$V4/scripts/fe01_a01_compare/submit_stage.sh" --env "$PWD/on_eval.private.env" --run-id on_v4_20261022 --stage replay_pilot
```

pilot四项各加载一个模型，跑共同6事件、两种geometry、1–20秒及小网格。
看 `stages/replay_pilot/*/worker.json` 的实测时间、输出字节数、显存峰值和支持状态，再决定全量资源。
尚未运行 pilot 时，任何“预计需要多少小时/磁盘”都不是实测结果。

## 5. pilot通过后，逐项手动提交全量

```bash
ON_EVAL_MAX_CONCURRENT=16 bash "$V4/scripts/fe01_a01_compare/submit_stage.sh" --env "$PWD/on_eval.private.env" --run-id on_v4_20261022 --stage replay_stations
ON_EVAL_MAX_CONCURRENT=16 bash "$V4/scripts/fe01_a01_compare/submit_stage.sh" --env "$PWD/on_eval.private.env" --run-id on_v4_20261022 --stage replay_grid --geometry normal
bash "$V4/scripts/fe01_a01_compare/submit_stage.sh" --env "$PWD/on_eval.private.env" --run-id on_v4_20261022 --stage latency
```

station两种geometry都做。normal网格覆盖全部1310事件；无高程支撑的点明确NA，退化事件提供站点支撑展示。
只在同一cutoff/input IDs/坐标/mask/checkpoint内缓存station编码；跨cutoff重算。
所预测的始终是同一台站的**最终PGA**，不是截至当前累计峰值。

latency四项测同一metadata pilot的K/Q，分station-only和whole-map，2次warmup、3次warm重复，正确同步DCU。
保存原始分项、cold加载、设备型号/precision、P50/P95及实际CPU渲染/PNG发布时间；不同硬件记录不伪装成相同硬件。

## 6. 全量完成后，统计、出图、pack

station、normal grid和crossing_truth完整通过后：

```bash
bash "$V4/scripts/fe01_a01_compare/submit_stage.sh" --env "$PWD/on_eval.private.env" --run-id on_v4_20261022 --stage analyze
```

analyze同时补齐旧snapshot的物理质量分层/四类交互，并分析新的固定站点回放。
normal grid完成时才有该scope的实测整图数值交付时钟；没有random grid时不声称已完成random地图时延。

**analyze完成后再提交render**，以携带TTUM前后关键帧和shadow分类图；正常出图需完整normal grid：

```bash
bash "$V4/scripts/fe01_a01_compare/submit_stage.sh" --env "$PWD/on_eval.private.env" --run-id on_v4_20261022 --stage render --geometry normal
```

全部1310事件都有20帧低分辨率浏览、动画、统一色标/等值线和直接台站评分图。
有ffmpeg用MP4，失败或没有则GIF；HTML帧浏览始终存在。每事件另有可切换完整研究阈值的真实坐标shadow分类HTML。
台站真值/目录图层醒目标为回顾性。没有网格真值，没有把台站误差插值认证为连续地面误差。

compare_existing、analyze、render_normal、latency全部完整通过后，才执行：

```bash
bash "$V4/scripts/fe01_a01_compare/submit_stage.sh" --env "$PWD/on_eval.private.env" --run-id on_v4_20261022 --stage pack
```

pack是CPU作业。它另用全量normal的实测计算+渲染时间重放共享FIFO，产生数值/渲染交付场景的TTUM和shadow；通信仍UNKNOWN。
完整结果不会被覆盖。重复submit会略过已经成功或正在跑的项，不会重复算一遍。

## 7. 结果到底在哪，失败如何恢复

本批统一新目录：

```text
/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_fe01/outputs/fe01_on_eval_v4/on_v4_20261022/
  plan.json, analysis_contract.json, source_identity.json
  slurm/STAGE_JOBID_TASKID.out 或 .err   ← 环境初始化失败也在这里
  attempts/STAGE/TASKID_随机后缀/       ← 失败前的中间产物与FAILURE.json，永久保留
  stages/STAGE/TASKID/                 ← 只有成功才原子发布；SEAL.json核全部文件
  jobs/                               ← 每次提交的job号和真实task集合
  inputs/                             ← 只读数据身份缓存、metadata选例封存
```

检查仍是第2节同一条status命令。它完整核SHA，旧集群sacct不使用不支持的`--array`。
队列消失、`sbatch`返回job号、目录非空都不算完成；只有 `COMPLETE_SEAL_VERIFIED` 才算。
失败/未知项会显示具体模型、事件块、日志路径。

例如只恢复失败的station块：

```bash
bash "$V4/scripts/fe01_a01_compare/resume_failed.sh" --env "$PWD/on_eval.private.env" --run-id on_v4_20261022 --stage replay_stations
```

其他阶段同样替换`--stage`；random grid/render同时加`--geometry random`。
成功项保留，失败新建attempt，在跑项不重复提交。scheduler状态UNKNOWN时拒绝自动重提，先核真实作业状态及日志。
改科学输入路径/合同/源码需新run ID，不重签旧seal。仅资源变量可按实测调；不要修改正在使用的release源码。

pack成功后下载：

```text
.../stages/pack/0/fe01_on_eval_v4_review.tar.gz
.../stages/pack/0/fe01_on_eval_v4_review.tar.gz.sha256
```

本机打开的全量图库入口是 `.../stages/pack/0/full_visuals_index.html`，相对链接指向各render块。
review包只放12个metadata预选详细案例，其余全部图源/数值的相对路径、大小、SHA在full_artifact_index.json及full_visuals_index.json。
大grid/全视频留超算，按索引另取，不塞入Git。

如需仅收集失败证据，可明确使用
`ON_EVAL_PACK_PARTIAL=1 ON_EVAL_PACK_PARTIAL_REASON='具体原因'` 提交pack；产物标`PARTIAL_WITH_FAILURES`，不能当完整研究结果。

## 8. 三个可选scope，绝不自动连跑

```bash
bash "$V4/scripts/fe01_a01_compare/submit_stage.sh" --env "$PWD/on_eval.private.env" --run-id on_v4_20261022 --stage replay_grid --geometry random
# random grid完成、analyze完成后：
bash "$V4/scripts/fe01_a01_compare/submit_stage.sh" --env "$PWD/on_eval.private.env" --run-id on_v4_20261022 --stage render --geometry random

bash "$V4/scripts/fe01_a01_compare/submit_stage.sh" --env "$PWD/on_eval.private.env" --run-id on_v4_20261022 --stage origin_replay
bash "$V4/scripts/fe01_a01_compare/submit_stage.sh" --env "$PWD/on_eval.private.env" --run-id on_v4_20261022 --stage long_extension
```

origin按[5/10/15/20/30]秒绝对截止转换回原first-P reference，历史起点不变；时钟缺失、elapsed<1和无input分别保留状态。
40/90秒按原native_prefix能力标支持/NA，不偷偷切换rolling，不混入主排名。
本轮没有重训、续训、OFF扩展、test、重选checkpoint或调sigma。
