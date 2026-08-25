# TEAM PyTorch 项目交接文档

更新时间：2026-08-25（Asia/Shanghai）

本文档是下一次全新 agent 会话的权威入口。先读完本文，再检查超算端状态和 Git
工作区；不要根据旧聊天记录、文件名或本地旧 checkpoint 猜测当前状态。

## 0. 快速定位与最重要结论

- 工作区：`/home/zhangb/work/people/zhangbei/team_claude`
- 主仓库：`/home/zhangb/work/people/zhangbei/team_claude/team_pytorch`
- 当前分支：`zhangb/native-scale-adapter-scaling`
- 本次 rt56 修改前基线：`8dd3bb3 Sync rt55 formal evaluation and project handoff`
- GitHub：`github.com/rular099/team_pytorch`
- 主实验：`rt55`，Japan 2000–2024 KNET-only 全量实时 PGA 概率预测
- 本地结果：`../chaosuan_res/weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42`
- 超算仓库：`/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team`
- 超算权重目录：`weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42`
- 超算实际数据根：`/public/home/test_bigmodel/seismogram/zb/origin_corrected_diting_vel_acc_vs30`

当前结论：

1. rt55 已训练到 checkpoint epoch 34，不建议继续无变化续训。
2. 当前远端 `full_model_best.pth` 的已知元数据为 epoch 32、validation objective
   `0.0123629803`；`full_model_last.pth` 应为 epoch 34。
3. epoch 32 的正式 validation normal 与 waveform-station-roll 已完成。normal PGA
   为 MAE `0.11663`、RMSE `0.18055`、R² `0.75412`、NLL `-0.88674`。
4. roll 后 MAE 升至 `0.24898`、R² 降至 `0.20691`，证明模型显著依赖正确匹配的
   台站波形；“模型只看坐标”的担忧不成立。
5. 用户在 2026-08-23 报告超算已同步运行 epoch 20 和 epoch 32 formal test；本地尚无
   新 test 结果，完成状态和指标仍须从超算核验，不能根据 test 结果反向改实验协议。
6. rt56 random geometry 已实现：ep32 zero-shot random mask 与 50% rt55 / 50% causal
   random mixed fine-tuning 可独立并行提交；rt55 配置、模型结构和原加载/推理路径不变。
7. 极重要：本地 `full_model_best.pth` 和 `full_model_last.pth` 都仍是 epoch 20，不能
   用它们代替远端 epoch 32/34 checkpoint。最新结果包没有覆盖这两个大文件。

## 1. 项目当前目标

### 1.1 主线目标

使用冻结的 DiTing waveform encoder、可训练 station adapter 和 TEAM 风格多台站
Transformer，对 Japan K-NET 台站进行实时、多目标 PGA 概率预测，同时保留震级和
位置辅助任务。

当前阶段已经从“继续优化 validation”进入“一次性冻结模型并做 held-out test”：

1. 固定 epoch 32 为正式 test 的预选 checkpoint；
2. 在固定 test split 上跑 normal，输出 MAE、RMSE、R²、NLL、Brier、1σ/2σ coverage；
3. 如实验方案预先要求，再跑 test waveform-station-roll，不能根据 normal test 结果
   再决定模型或阈值；
4. 汇总 validation/test 的时间、台站数、目标类型、预警提前量、有效波形秒数等分桶；
5. 再决定是否开展多 seed、encoder 末层解冻或更强 station representation 实验。

### 1.2 次要工作流

- Hi-net 原始 CNT/CH 年度 HDF5 归档、断点续传和数据集审计；
- 技术交流 PPT 与图表；
- station representation collapse 与诊断统计修正。

这些次要工作流不能改变 rt55 的固定 split、checkpoint 或正式 test 协议。

## 2. 当前完成状态

### 2.1 2000–2024 数据与固定 split

全量源数据检查已经通过：

- 25 个年度 HDF5 shard（2000–2024）；
- 14,153 个事件，501,956 条 station rows；
- KNET 211,546 条，KiK-net 290,410 条；
- 100 Hz、三分量，必要波形、坐标、P pick、PGA、storage-valid 区间和 VS30 字段齐全；
- `pga` 坐标为 `log10(m/s²)`；
- 波形入模前转 float32，不需要另存一套 float32 HDF5；
- 97.37% KNET 记录有前置 storage padding，loader 已优先使用
  `record_start_sample + valid_n_samples` 生成显式有效区间。

固定 KNET split：

| split | events | station rows |
|---|---:|---:|
| train | 9,627 | 146,842 |
| validation | 1,383 | 22,336 |
| test | 2,769 | 42,368 |

划分是每个年度 shard 内 event-disjoint 的约 70%/10%/20%，不是 held-out-year 外推。
`split_events.csv` 和 `split_stations.csv` 已随远端运行目录生成，正式 test launcher 会
核验 test 数量。

### 2.2 rt55 结构与训练设置

rt55 的固定设计：

- KNET-only；
- legacy station adapter；
- 显式 waveform storage-valid mask；
- DiTing encoder 冻结；station adapter 和 TEAM 侧训练；
- DPK cache、DPK token weighting、PGA temporal residual 全关闭；
- PGA 三分量 MDN，主要优化 PGA，保留 magnitude/location 辅助头；
- PGA normalization：mean `-1.1228067351`，std `0.4312468402`，count `146842`；
- train 每个事件每 epoch 从 7 个实时 bin 中不放回抽 3 个；
- validation/test 固定 1/3/5/10/20/40/90 秒；
- 训练使用约 4 节点 × 4 DCU，batch size 8/卡，global batch 128；
- 初始 lr 为 `1e-3`，ReduceLROnPlateau factor `0.5`、patience `2`、min lr `1e-5`；
- 续训目标 epoch 是“总完成轮数”，不是额外轮数。

### 2.3 训练已经完成到 epoch 34

训练经历了 12、20、34 三个阶段。checkpoint 中的 `epoch=32` 表示已经完成 32 轮，
对应 TensorBoard/CSV 的零基 step 31；不要混淆这两个编号。

最新本地标量包记录的后十个零基 epoch：

| scalar step | 完成后的 checkpoint epoch | train loss | validation objective | lr |
|---:|---:|---:|---:|---:|
| 24 | 25 | 0.08814 | 0.02996 | 2.5e-4 |
| 25 | 26 | 0.08609 | 0.05430 | 2.5e-4 |
| 26 | 27 | 0.08017 | 0.02443 | 2.5e-4 |
| 27 | 28 | 0.07965 | 0.03731 | 2.5e-4 |
| 28 | 29 | 0.08732 | 0.04127 | 2.5e-4 |
| 29 | 30 | 0.06459 | 0.02421 | 2.5e-4 |
| 30 | 31 | 0.07478 | 0.06963 | 2.5e-4 |
| 31 | 32 | 0.06629 | **0.01236** | 2.5e-4 |
| 32 | 33 | **0.06007** | 0.02942 | 2.5e-4 |
| 33 | 34 | 0.06578 | 0.05684 | 2.5e-4 |

训练判断：

- epoch 32 是当前 validation objective 最优 checkpoint；
- epoch 33/34 连续回升，继续相同训练的边际收益低且会增加事后选择风险；
- epoch 32 后 train loss 仍低，但 validation objective 反弹，当前应停止并固定 test；
- epoch 20 到 32 的收益主要体现在 NLL、RMSE、R²和波形依赖证据，MAE 几乎持平；
- 本地标量包缺少 step 20–23，不应据此虚构这四轮的轨迹。

### 2.4 scheduler 恢复问题已修复

旧 checkpoint 在 `scheduler.step()` 之前保存，导致恢复后 ReduceLROnPlateau 内部状态
少观察一次当轮 validation loss。修复包括：

- 新 checkpoint 在 scheduler step 后保存；
- 写入 `scheduler_step_completed`、`scheduler_monitor`、
  `scheduler_monitor_loss`；
- 加载旧 checkpoint 时仅对缺少完成标记的 ReduceLROnPlateau 重放一次 monitor loss；
- 新 checkpoint 恢复时不重复 step；
- `tests/test_scheduler_checkpoint_resume.py` 覆盖旧 checkpoint 补 step、新 checkpoint
  不重复 step 和 plateau/best 两类状态。

本地 epoch 20 `.pth` 是修复前格式：optimizer lr 已是 `2.5e-4`，但 scheduler
`last_epoch=18` 且没有新标记。它可以由修复后的 loader 正确兼容，但它不是最新模型。
远端 epoch 32/34 checkpoint 是否带新标记尚未下载到本地，下一会话应在远端核验。

### 2.5 epoch 20 与 epoch 32 的正式 validation

统一协议：1,383 个 validation events、9,681 个 realtime samples、89,770 个有效 PGA
targets；坐标为 `log10(m/s²)`，point estimate 为 MDN predictive mixture mean；Brier
阈值为 `-1.2 log10(m/s²)`。

| checkpoint | pairing | MAE | RMSE | R² | NLL | Brier | coverage 1σ | coverage 2σ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| epoch 20 | normal | **0.11617** | 0.18120 | 0.75235 | -0.86667 | **0.09112** | 0.70168 | 0.95367 |
| epoch 20 | roll | 0.24466 | 0.32063 | 0.22459 | 1.68712 | 0.23719 | 0.39195 | 0.61143 |
| epoch 32 | normal | 0.11663 | **0.18055** | **0.75412** | **-0.88674** | 0.09344 | 0.67869 | 0.94927 |
| epoch 32 | roll | 0.24898 | 0.32426 | 0.20691 | 1.99375 | 0.24429 | 0.37067 | 0.59288 |

epoch 32 normal 的其他关键结果：

- correlation `0.87210`，slope `0.77200`，bias `+0.02883`；
- predictive sigma mean `0.13838`；1σ coverage `0.67869` 接近常见高斯参考值；
- n=1 station 时 MAE `0.2379`，n=16+ 时降至 `0.0829`；
- event-mean valid waveform 1–3 s 时 MAE `0.2262`，40 s+ 时降至 `0.0391`；
- post-P 0–1 s 时 MAE `0.2169`，40 s+ 时降至 `0.0369`；
- input targets MAE `0.0830`，triggered non-input `0.1567`，untriggered `0.2061`；
- lead 0–5 s MAE `0.1928`，5–20 s `0.2388`，20 s+ `0.3990`；最后一桶仅
  343 targets，解释时必须报告样本量；
- weak/strong（阈值 -1.2）MAE 分别 `0.1198/0.1122`，但 bias 分别
  `+0.0833/-0.0477`，仍存在弱震偏高、强震偏低的动态范围压缩。

roll 控制的关键解释：

- epoch 32 roll 相对 normal：MAE `+0.13235`、RMSE `+0.14371`、R² `-0.54721`、
  NLL `+2.88049`、Brier `+0.15085`；
- 长波形反而退化最明显：valid waveform 40 s+ 的 MAE 从 `0.0391` 升到 `0.2786`；
- n=1 时 roll 与 normal 相同是预期行为，因为只有一个有效 station 时循环置换不改变
  waveform-station 配对；
- 结论是模型确实利用 matched waveform，且随台站数/有效时长增加获得明显收益；
- 这不等于 raw station adapter 的 cosine collapse 已解决。波形幅值注入、坐标和 TEAM
  上下文仍可能承担主要区分能力。

模型选择约束：epoch 20 的 MAE/Brier 略优，epoch 32 的 RMSE/R²/NLL、slope 和 roll
依赖更强。已决定把 epoch 32 作为正式 test 的主 checkpoint；epoch 20 只能作为
validation 参考，不能在看到 test 后切换。

### 2.6 正式评估基础设施已完成

`eval_checkpoint.py` 现在支持：

- `--splits train,val,test`，别名规范化和去重；
- test 真正读取 `parts=(False, False, True)`；
- test 使用与 validation 相同的固定 7 时刻协议，不做 train 随机采样；
- waveform-station `none/roll/random` 对照；
- waveform、storage mask、amplitude 相关输入和 cached token weights 一起置换；
- formal MAE、RMSE、R²、unweighted MDN NLL、Brier、1σ/2σ coverage JSON；
- `--skip_diagnostics`，正式大 split 可跳过昂贵样例诊断；
- 独立 TXT、NPZ、metrics JSON 输出。

Slurm 入口：

- `tools/eval_rt55_validation_normal_roll_slurm.sh`：validation normal/roll 可分开提交，
  `ACTION=normal|roll|all`；之前组合任务超时后已经通过分开续跑完成。
- `tools/eval_rt55_test_formal_slurm.sh`：一次性 test，必须显式
  `CONFIRM_TEST_EVAL=1`；校验 test split 数量，默认禁止覆盖已有结果。

正式 test 结果截至本地 2026-08-20 更新包仍不存在。不要把 validation 的
`metrics_1.json` 当 test。

### 2.7 feature-collapse 状态

全量训练没有消除 legacy adapter 原始表征的高相似性：早期 12 轮中 raw adapter
station cosine 约 `0.987–0.988`，event-centered residual norm 只占公共向量几个百分点。
但是 epoch 32 的 roll 对照证明最终系统并非忽略 waveform。

已知诊断 bug 仍未修：`gemini_models.py` 外层逐 station 调同一个 adapter，adapter 的
`_last_token_pool_diag` 被最后 station slot 覆盖；第 25 个 slot 常是 padding，所以
`diag_station_pool_*_valid_token_count=0` 不能解释为所有波形被 mask。训练 forward 的
mask 实际正常。后续应按 `station_valid` 聚合全部有效 station，并在固定 eval split 上
统计，不要继续解释最后 slot 快照。

### 2.8 rt56 causal random geometry

rt56 从固定的 rt55 epoch 32 checkpoint 做 weight-only 初始化，optimizer、scheduler、
epoch 计数和 best-loss 状态全部重新开始；DiTing encoder 继续冻结，model parameters 与
rt55 完全一致。

- train：50% 保留 rt55 协议，50% 使用 causal random geometry；
- validation/zero-shot：100% random geometry，固定 seed 和 1/3/5/10/20/40/90 秒；
- 输入数从 1/3/5/8/12/16 中抽取上限，只从当前时刻已触发且有有效波形的完整事件台站
  集合中采样，发生在 25 台站截断之前；
- 随机分支至少保留一个有限 PGA 的非输入台站，PGA targets 明确排除输入台站；
- 未选输入 waveform 与 sample-valid mask 同时归零，避免坐标 slot 被误当作有效波形；
- 新输出会记录 random mask 是否生效、候选/请求/实际输入台站数直方图。

入口：`tools/run_rt56_random_geometry_slurm.sh`。`ACTION=all` 会提交 zero-shot 与
fine-tune 两个无依赖并行 job；默认只允许 dry-run，真实提交必须设置 `CONFIRM_RT56=1`。
截至本文更新，代码和脚本已准备完成，但本会话未替用户调用 `sbatch`。

### 2.9 Hi-net 年度原始归档与审计

Hi-net 工作流保留 CNT/CH 原始 bytes 的唯一永久副本，每年一个 HDF5，支持 SHA256
回读校验和事务恢复，不额外永久保存 MiniSEED/SAC/NPZ waveform shard。

2026-08-11 审计：

- 14,153 个源事件；已提交 12,708；下载波形 12,392；无匹配台站 316；未归档 1,445；
- 25 年中 12 年 complete；2000–2003 没有成功波形，主要是提供端无数据；
- 270,050 个请求台站行中，949 行缺竖直分量、1,086 行缺完整三分量；影响 551 个
  已下载事件；
- 当前数值是 raw counts，不是响应校正后的物理速度；
- 新下载器已增加 300 s timeout、最多 40 station/batch、整窗失败后的连续分钟回退、
  三分量 channel table 与 CNT 样点覆盖校验；任一批次失败时事件不提交；
- 新校验不追溯改写历史 551 个受影响事件，训练时必须使用台站级可用性 mask；
- 审计报告在 `reports/hinet_dataset_audit_20260811/`。

## 3. 最近的重要修改

### 3.1 全量训练、resume 与数据加载

| 文件 | 修改 | 原因 |
|---|---|---|
| `gemini_util_light.py` | storage-valid mask 回退使用 `record_start_sample` | 排除年度 HDF5 前置补零 |
| `loader_light.py` | versioned metadata cache、HDF5 identity、精简列、原子发布 | 25 shard 加速且避免 DDP 半写 cache |
| `train_light.py` | config `extends`/环境变量、多 shard generator、rank0 cache 预热、固定 DDP global index、resume-aware epoch sampling、epoch override | 支持 rt55 全量严格续训 |
| `train_light.py` | scheduler post-step checkpoint 标记与 legacy replay | 修复恢复时 ReduceLROnPlateau 少一步 |
| `train_light_slurm.sh` | 安全解析环境变量 weight path | 支持超算 launcher 覆盖路径 |
| `pga_configs/...rt55...json` | 25 年 KNET-only、legacy+mask、no DPK、固定 normalization | 固定 rt55 实验定义 |
| `tools/run_rt55_japan_full_2000_2024_slurm.sh` | smoke/full、last-only resume、epoch target、active-job guard、dry-run | 安全提交和续训 |
| `tools/validate_japan_full_training_data.py` | schema、全量/抽样、mask、cache、manifest 检查 | 上传后验证数据兼容性 |
| `docs/japan_full_training_2000_2024.md` | 数据、上传、训练、resume、eval 说明 | 固化操作协议 |

launcher 的旧 Bash 空数组 `resume_args[@]: unbound variable` 已修复：现在先构造必定非空
的 `train_args`，再追加可选参数。不要复制早期 launcher 写法。

### 3.2 formal eval 与 test

| 文件 | 修改 | 原因 |
|---|---|---|
| `eval_checkpoint.py` | test split、split 选择、formal metrics、roll permutation、skip diagnostics | 完成正式 validation/test 协议 |
| `eval_checkpoint_slurm.sh` | 单 checkpoint 输出、metrics JSON、参数白名单和输出检查 | 可靠提交独立 eval |
| `tools/eval_rt55_validation_normal_roll_slurm.sh` | normal/roll 分离、overwrite guard、dry-run | 解决长任务超时后的可续跑性 |
| `tools/eval_rt55_test_formal_slurm.sh` | test 数量审计、显式确认、固定输出名、禁止覆盖 | 防止误跑/重复使用 test |
| `tests/test_eval_checkpoint_formal.py` | split、NLL、valid target、roll 测试 | 锁定指标语义 |
| `tests/test_scheduler_checkpoint_resume.py` | legacy/new scheduler 恢复测试 | 防止恢复回归 |

### 3.3 rt56 random geometry

| 文件 | 修改 | 原因 |
|---|---|---|
| `gemini_util_light.py` | causal full-event random input mask、非输入 PGA target sampling、诊断字段 | 支持任意因果台站到任意非输入位置 PGA |
| `train_light.py` | nullable load/transfer path、generator 日志、继承合并后统一展开环境变量 | weight-only ep32 初始化且避免被子配置覆盖的父占位符提前报错 |
| `eval_checkpoint.py` | random geometry 元数据收集和 formal metrics | 审计实际随机台站数分布 |
| `pga_configs/...rt56...json` | ep32 初始化、50/50 mixed train、100% random val | 固定新实验协议 |
| `tools/run_rt56_random_geometry_slurm.sh` | 双任务、路径/输出/job guard、dry-run | 安全并行提交 zero-shot 和 fine-tune |
| `tests/test_causal_random_geometry.py` | helper、端到端 generator、rt55/rt56 config 兼容测试 | 防止泄漏和 rt55 回归 |

2026-08-25 首次超算提交暴露了父 rt55 的 `${JAPAN_FULL_WEIGHT_PATH}` 在 child merge 前
被展开的问题。加载器现已改为完整继承合并后统一展开，rt56 launcher 也显式导出该父级
占位符以兼容超算上的旧代码副本。失败发生在创建模型和权重目录之前，不是训练中断恢复。

### 3.4 Hi-net

| 文件 | 修改 |
|---|---|
| `download_hinet.sh`、`download_hinet_continue.sh` | 年度倒序、重试、续传、transport 参数 |
| `tools/download_hinet_velocity.py` | annual transaction、分批、分钟回退、错误根因和严格通道校验 |
| `tools/hinet_raw_archive.py` | byte-exact HDF5 writer/reader、SHA256、恢复和 worker-safe 读取 |
| `tools/plot_hinet_accel_velocity_qc.py`、`hinet_qc.sh` | 直接从年度 archive 做波形 QC |
| `tools/audit_acceleration_hinet_datasets.py` | 全量目录/归档审计与抽样波形 QC |
| `docs/hinet_velocity_download.md` | 下载、恢复、schema、DataLoader 使用说明 |
| `tests/test_hinet_raw_archive.py`、`tests/test_download_hinet_archive_flow.py` | 归档事务与下载回退测试 |

## 4. 当前架构与设计决策

### 4.1 主链路

```text
raw waveform
  -> storage-valid mask
  -> per-channel normalization（padding 保持 0）
  -> frozen DiTing encoder（f2/f3/f4/x）
  -> trainable legacy station adapter
  -> amplitude scale embedding
  -> coordinate fusion
  -> TEAM station transformer
  -> event cross-attention
  -> target cross-attention
  -> PGA 3-component MDN + magnitude/location auxiliary heads
```

`freeze_mode=none` 不代表 encoder 可训练；full model 初始化后 encoder 被显式
`requires_grad=False`。

### 4.2 mask 与物理时间

- storage-valid mask 只排除补零/记录外区域，不等于 post-P event mask；
- P 前可以是真噪声，也可以是 padding，不能混为一谈；
- 诊断必须同时报告 valid waveform seconds 和 post-P valid seconds；
- 不要只报告 token 数。

### 4.3 DPK 与 station adapter 选择

- 用户不接受手工 P/DET confidence gate 作为主融合方案；
- rt55 关闭所有 DPK cache/token/residual，减少迁移和置信度校准依赖；
- rt52 legacy、rt53 NLTA-S、rt54 NLTA-M 的 validation roll delta 都约 0.009，NLTA-S/M
  没有相对 legacy 的 Pareto 优势；
- 不要直接上 NLTA-L，也不要原样重复 event+residual 或可关闭 scalar gate；
- 若后续改结构，优先评估轻量解冻 DiTing 最后 blocks，backbone lr 取 adapter lr 的
  约 0.05–0.1，并做相同 split/seed 的严格消融。

### 4.4 数据与计算

- 保留 25 个年度 HDF5，不合并；
- metadata cache 可重建，不复制 waveform；
- corrected HDF5 不再 origin-correct，不转 float32，不生成 DPK cache；
- 登录节点不做全量 HDF5 扫描或正式推理；
- 精确续训应保持原 world size/global batch，但 rt55 当前不建议再续；
- resolved `weight_dir/config.json` 是 eval 的权威配置，不用原始含 `auto`/环境变量配置。

## 5. 未完成事项（按优先级）

### P0：完成正在运行的 formal test 并归档

1. 在超算读取 `full_model_best.pth` 和 `full_model_last.pth` 元数据，确认分别为 epoch
   32 和 34，并确认 epoch 32 loss 为 `0.012362980283796787`。
2. 若还没有固定副本，使用不覆盖方式归档：

   ```bash
   cp -n full_model_best.pth full_model_best_ep32.pth
   cp -n full_model_last.pth full_model_last_ep34.pth
   ```

3. 检查用户所述 epoch 20/32 test jobs 的 `squeue`、日志与输出；不要重复提交同一输出。
4. jobs 完成后下载 TXT、NPZ、metrics JSON 和新 checkpoint 元数据，不要只下载标量
   CSV；放入一个新的本地目录或保留原文件名，避免自动生成 `_1/_2` 后失去语义。
5. epoch 32 是预先固定的主结果；epoch 20 仅作事先声明的 sensitivity reference，不能
   根据两者 test 表现重新选择模型。roll 是稳健性对照，也不参与选择。

### P0：运行 rt56 zero-shot 与 mixed-random fine-tuning

1. 同步新 commit 后先在超算执行 `DRY_RUN=1 ACTION=all`；
2. 核对源 checkpoint 是远端 `full_model_best_ep32.pth`、25 个年度 shard 均存在、新权重
   目录为空，且 zero-shot 输出未存在；
3. 用 `CONFIRM_RT56=1 ACTION=all` 提交两个并行任务；
4. fine-tune 只按 random-geometry validation 选择 checkpoint；后续 retention eval 要用
   原 resolved rt55 config 加载所选 rt56 checkpoint，不能修改 rt55 正在运行的 test。

### P1：结果整理

- 汇总 validation/test overall 和 1/3/5/10/20/40/90 s；
- 同时报告 station count、target type、lead time、valid/post-P seconds、PGA 强弱和样本量；
- 校准结果必须说明 Brier threshold 与 coverage 定义；
- epoch 20 可列为 validation sensitivity reference，不在 test 上比较并重新选择；
- 正式论文至少补 2 seeds，理想 3 seeds；rt55 当前仅 seed 42。

### P1：修 station-pool diagnostics

- 按 `station_valid` 聚合所有有效 station，不再读取最后 slot；
- all-invalid station 不进入平均；
- 同时记录有效 token、valid seconds、post-P seconds；
- 添加“最后 slot 为 padding 仍能得到正确聚合”的回归测试；
- 在固定 eval split 上统计，不依赖每 epoch 单 batch 快照。

### P1：决定下一模型实验

formal test 和多 seed 计划确定后再选：

- 若当前性能满足 baseline 目标，冻结 rt55，把 collapse 作为独立 representation 研究；
- 若要改善早期/未触发目标，优先研究 encoder 末层解冻、物理时间上下文或更稳健的
  station-local learning objective；
- 不要仅因 raw cosine 高就否定系统，roll 已给出直接因果对照；
- 不要仅因 roll gap 大就声称 station representation 已充分多样化。

### P2：Hi-net

- 用新分批/分钟回退机制处理剩余技术失败；长期无数据应进入可审计终态，不无限重试；
- 训练用速度时先生成 station-row availability mask；
- 物理速度研究前完成响应/灵敏度校正和单位追踪；
- 历史已提交归档不原地改写，修复应新建版本或外置质量表。

## 6. 已知问题与风险

1. 本地两个 `.pth` 都是 epoch 20；最新 epoch 32/34 checkpoint 只在超算，或尚未被
   下载。文件名相同不代表内容最新。
2. 本地带 `_1` 的 eval 文件是 epoch 32；不带后缀的是 epoch 20。带 `_2` 的训练标量
   是后续 24–33 step。后缀来自重复解压/复制碰撞，不是实验编号。
3. 本地标量缺 step 20–23；不能画成连续完整曲线而不标注缺口。
4. 远端 epoch 32/34 checkpoint 的 scheduler 完成标记尚未在本地核验。
5. formal test 尚无本地结果；不能用 validation 代替 test，也不能在 test 后调模型。
6. test launcher 默认禁止覆盖。若已有文件，先鉴别是完整结果还是超时残留；不要直接
   `ALLOW_OVERWRITE=1`。
7. validation normal+roll 曾因单任务时限不足而超时；分开提交已解决。test 样本约为
   validation 两倍，建议 `SLURM_TIME=3-00:00:00`。
8. split 是每年内部固定划分，不是 unseen-year；不能声称时间外推泛化。
9. 137 个事件 correction provenance 缺失；严格数据质量消融应另建配置，不原地改 HDF5。
10. `station_pool_*` 诊断仍是最后 station slot，部分值为 0 只说明最后 slot 是 padding。
11. 多数训练 `diag_*` 是单 batch 快照，不是完整 split 均值。
12. rt55 只有一个 seed；单次 test 不支持稳健显著性结论。
13. 本地 worktree 曾混有 Hi-net、PPT 和训练多批文件。不要 reset、checkout 或批量删除。
14. `logs.zip`、PPT 生成目录、lock/state 和大图属于本地产物，不应无审查地加入源码提交。
15. GitHub SSH 端口 22 在 2026-08-21 曾被网络中间层关闭；HTTPS `ls-remote` 可用。
16. Hi-net archive identity 很严格；科学选择参数改变后使用新 archive，transport timeout/
    batch/fallback 可以在同一 partial archive 上调整。

## 7. 下一会话第一步

不要再改 rt55 模型，也不要继续 rt55 原设置训练。

第一步是在超算确认正在运行的 epoch 20/32 formal test 和 rt56 同步状态：

```bash
cd /public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team

RUN=weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42

python - <<'PY'
import torch
from pathlib import Path

root = Path("weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42")
for name in ("full_model_best.pth", "full_model_last.pth"):
    ckpt = torch.load(root / name, map_location="cpu")
    print(name, {
        "epoch": ckpt.get("epoch"),
        "loss": ckpt.get("loss"),
        "scheduler_step_completed": ckpt.get("scheduler_step_completed"),
        "scheduler_monitor_loss": ckpt.get("scheduler_monitor_loss"),
    })
PY

squeue -u "$USER" -n team-rt55-test
ls -lh "logs/$RUN"/eval_test_best_normal.* 2>/dev/null || true

DRY_RUN=1 ACTION=all bash tools/run_rt56_random_geometry_slurm.sh
```

只有在确认 formal test 实际未提交、没有输出且用户仍要求补交时，才使用原 rt55 test
launcher；不要在用户所述任务仍运行时重复提交。rt56 的真实提交命令为：

```bash
CONFIRM_RT56=1 ACTION=all bash tools/run_rt56_random_geometry_slurm.sh
```

## 8. 不要做什么

- 不要继续相同设置续训并在更多 validation 波动中挑 checkpoint。
- 不要用本地 epoch 20 `.pth` 冒充 epoch 32。
- 不要从 best 恢复训练；last 才是恢复点，best 是推理/选择点。
- 不要看到 test 后切换 epoch 20/32、阈值、seed 或分桶定义。
- 不要把 validation objective 当 MAE，也不要把 validation 指标当 test。
- 不要覆盖已有 test 输出，除非先证明它只是失败残留并保留审计记录。
- 不要把 station-pool token count 为 0 解读为所有波形被 mask。
- 不要只报告总体均值，忽略早期、未触发、长 lead-time 和样本量。
- 不要声称 roll 证明 adapter 本身不 collapse；它证明最终系统使用 matched waveform。
- 不要声称每年内部 test 是 held-out-year。
- 不要重新启用 DPK 或手工 confidence gate，除非设计独立、同 split 的新消融。
- 不要直接上 NLTA-L；NLTA-S/M 已无 Pareto 优势。
- 不要重新 origin-correct 已校正 HDF5，不要合并年度 HDF5，不要复制 float32 波形。
- Hi-net 不要同时永久保存 CNT、MiniSEED 和 decoded waveform shards。
- 不要把 Hi-net raw counts 称为物理速度。
- 不要执行 `git reset --hard`、`git checkout --`、批量删除或无审查 `git add -A`。

## 9. 关键上下文速记

### 9.1 formal validation 文件映射

本地目录：

```text
../chaosuan_res/weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42/
```

```text
eval_validation_best_normal.metrics.json              # epoch 20
eval_validation_best_waveform_station_roll.metrics.json # epoch 20
eval_validation_best_normal.metrics_1.json            # epoch 32
eval_validation_best_waveform_station_roll.metrics_1.json # epoch 32
train_epoch_loss_2.csv / val_epoch_loss_2.csv          # scalar step 24–33
full_model_best.pth / full_model_last.pth              # 两者均为本地旧 epoch 20
```

### 9.2 validation 重跑入口

```bash
ACTION=normal bash tools/eval_rt55_validation_normal_roll_slurm.sh
ACTION=roll bash tools/eval_rt55_validation_normal_roll_slurm.sh
```

当前 epoch 32 validation 已完成，不要无理由覆盖重跑。

### 9.3 test 输出

normal 预期生成：

```text
logs/$RUN/eval_test_best_normal.txt
logs/$RUN/eval_test_best_normal.npz
logs/$RUN/eval_test_best_normal.metrics.json
```

可选 roll 使用 `PERMUTATION=roll`，输出 stem 为
`eval_test_best_waveform_station_roll`。normal 是正式主结果。

### 9.4 当前快速测试

2026-08-23 本地执行通过 28 个测试：

```bash
python -m unittest -v \
  tests.test_causal_random_geometry \
  tests.test_scheduler_checkpoint_resume \
  tests.test_eval_checkpoint_formal \
  tests.test_hinet_raw_archive \
  tests.test_download_hinet_archive_flow

python -m py_compile \
  train_light.py eval_checkpoint.py loader_light.py gemini_util_light.py \
  tools/download_hinet_velocity.py tools/hinet_raw_archive.py \
  tools/plot_hinet_accel_velocity_qc.py \
  tools/audit_acceleration_hinet_datasets.py \
  tools/validate_japan_full_training_data.py

bash -n \
  check_hinet.sh download_hinet.sh download_hinet_continue.sh hinet_qc.sh \
  train_light_slurm.sh eval_checkpoint_slurm.sh \
  tools/run_rt55_japan_full_2000_2024_slurm.sh \
  tools/eval_rt55_validation_normal_roll_slurm.sh \
  tools/eval_rt55_test_formal_slurm.sh \
  tools/run_rt56_random_geometry_slurm.sh
```

测试输出中的 `Please 'pip install apex'` / `xformers` 是可选依赖提示，不是测试失败。

### 9.5 Repo 结构

```text
team_pytorch/
  SESSION_SUMMARY.md             # 本交接文档
  train_light.py                 # 训练、resume、scheduler、多 shard、split export
  eval_checkpoint.py             # train/val/test、roll、formal metrics
  gemini_models.py               # adapters、TEAM、readouts、diagnostics
  gemini_util_light.py           # generator、mask、realtime sampling
  loader_light.py                # HDF5 metadata/cache/split
  train_light_slurm.sh           # 通用训练 Slurm
  eval_checkpoint_slurm.sh       # 通用 eval Slurm
  pga_configs/                   # 实验配置
  tools/                         # launchers、validators、Hi-net
  tests/                         # scheduler/eval/Hi-net 回归测试
  reports/                       # 实验计划、数据审计、汇报源文件
  docs/                          # 操作文档
```

### 9.6 Hi-net 常用入口

```bash
export HINET_USER='...'
export HINET_PASSWORD='...'
bash download_hinet.sh
```

凭据只通过环境变量提供，不写入仓库。中断后重复同一命令恢复；不要对 corrected HDF5
再传 `--origin-corrections`。

### 9.7 Git 注意事项

- 提交前先看 `git status --short --branch` 和 `git diff --check`；
- GitHub SSH 如仍被关闭，可使用 HTTPS fetch/push，但不要在命令行明文写 token；
- 本地 `logs.zip`、PPTX、生成 slide 图片和 lock/state 文件需保留，但不属于默认源码同步；
- 任何新 agent 都要先确认这些本地产物是否仍未跟踪，再决定是否另建 artifact release。
