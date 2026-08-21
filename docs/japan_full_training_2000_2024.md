# 日本 2000–2024 全量训练（rt55）

## 当前状态（2026-08-21）

- 训练已完成到 checkpoint epoch 34，不再建议无变化续训；
- 当前远端 best 是 epoch 32，validation objective 为 `0.012362980283796787`；
- epoch 32 validation normal/roll 已完成；formal normal 指标为 MAE `0.11663`、
  RMSE `0.18055`、R² `0.75412`、NLL `-0.88674`、Brier `0.09344`、1σ/2σ coverage
  `0.67869/0.94927`；
- waveform-station-roll 的 MAE 为 `0.24898`、R² 为 `0.20691`，说明模型显著使用
  matched station waveform；
- 下一步是固定归档 epoch 32，并在 2,769-event test split 上进行一次性 formal eval；
- 本地结果目录中的 `full_model_best.pth`/`full_model_last.pth` 仍是旧 epoch 20，不能
  代替超算端 epoch 32/34。详细文件映射和风险见 `SESSION_SUMMARY.md`。

旧 checkpoint 在 scheduler step 前保存的问题已经修复：新 checkpoint 在 step 后写入
完成标记；恢复旧 ReduceLROnPlateau checkpoint 时只重放一次缺失的 monitor loss。相关
回归测试位于 `tests/test_scheduler_checkpoint_resume.py`。

## 已确认的数据兼容性

本地数据根目录：

```text
/run/media/zhangb/My Passport/knet_converted/origin_corrected_diting_vel_acc_vs30
```

当前检查结果：

- 25 个年度 HDF5 分片（2000–2024）均可打开；
- 14,153 个事件，事件 ID 全局无重复；
- 501,956 条台站记录，其中 KNET 211,546 条、KiK-net 290,410 条；
- 所有年度均为 100 Hz、三分量，包含 `waveforms`、`coords`、`p_picks`、`pga`、`record_start_sample`、`valid_n_samples`、网络/传感器类型和 VS30；
- `pga` 已是 `log10(PGA[m/s²])`，不需要重新计算标签；
- 波形在送入模型前会转为 float32，因此不需要为训练另存一套 float32 HDF5；
- HDF5 已保存校正后的 origin time 及其来源/status/delta，不再传 `--origin-corrections`；
- rt55 不读取 DPK probability/cache，也不构建 PGA temporal residual。

97.4% 的 KNET 台站记录具有正的 `record_start_sample`。代码已将
`record_start_sample + valid_n_samples` 作为显式有效存储区间，避免把前置补零计入
归一化和 pooling。

Origin-correction provenance 中 14,016 个事件标记为 `matched`，另有 137 个事件的
correction status/source 为缺失值（KNET split 中为 train/dev/test=99/7/28）。这些事件
仍具有最终 P-pick 和对齐波形，当前快速全量基线保留它们。上传后不能靠再次传
`--origin-corrections` 原地改变 HDF5；若以后决定剔除或重建，应作为单独的数据质量
消融，而不是在本次上传步骤中混入。

当前快速全量 split 是每个年度分片内的 event-disjoint 70%/10%/20%
train/dev/test；KNET 筛选后预计为 9,627/1,383/2,769 个事件。这是内部评估 split，
不是严格的 held-out-year 时间外推测试。

## 上传

建议把年度目录原样上传，不合并 HDF5，也不上传 diagnostics PNG（训练不读取它们）。
若需要完整保留数据版本证据，可另外保存 CSV 和 diagnostics。

示例目标路径：

```text
/public/home/test_bigmodel/seismogram/zb/team_pytorch/japan_data/origin_corrected_diting_vel_acc_vs30
```

可续传上传示例（源目录末尾的 `/` 很重要）：

```bash
rsync -avhP --partial --append-verify \
  --include='*/' \
  --include='japan_*.hdf5' \
  --include='japan_*_events.csv' \
  --include='japan_*_stations.csv' \
  --exclude='*' \
  '/run/media/zhangb/My Passport/knet_converted/origin_corrected_diting_vel_acc_vs30/' \
  USER@HOST:/public/home/test_bigmodel/seismogram/zb/team_pytorch/japan_data/origin_corrected_diting_vel_acc_vs30/
```

## 上传后的必要操作

进入超算仓库和训练环境后，先执行全 schema 检查并生成约几十 MB 的 station metadata
cache。该步骤只读 HDF5，不复制波形：

```bash
cd /public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team

export JAPAN_FULL_DATA_ROOT=/public/home/test_bigmodel/seismogram/zb/team_pytorch/japan_data/origin_corrected_diting_vel_acc_vs30
export JAPAN_FULL_WEIGHT_PATH=weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42

python tools/validate_japan_full_training_data.py \
  --data-root "$JAPAN_FULL_DATA_ROOT" \
  --scan all \
  --prepare-cache
```

预期末行包含：

```text
[PASS] full-data compatibility: events=14153, station_rows=501956, knet_rows=211546, split_after_knet={'train': 9627, 'dev': 1383, 'test': 2769}
```

训练启动时 rank 0 也会自动补齐缺失 cache，然后再放行其他 DDP ranks。因此手工
`--prepare-cache` 是节省多卡启动时间的推荐步骤，但遗漏后不会导致并发写坏 cache。

无需执行：

- 年度 HDF5 合并；
- 波形格式/float32 二次转换；
- origin correction；
- DPK prior cache 预计算；
- PGA 标签或 VS30 重建。

## Smoke run

先检查路径与提交参数：

```bash
JAPAN_FULL_DATA_ROOT="$JAPAN_FULL_DATA_ROOT" \
RUN_MODE=smoke DRY_RUN=1 \
bash tools/run_rt55_japan_full_2000_2024_slurm.sh
```

再提交 1 节点、4 DCU、1 epoch 的 limited-data smoke run（同时验证单节点 DDP）：

```bash
JAPAN_FULL_DATA_ROOT="$JAPAN_FULL_DATA_ROOT" \
RUN_MODE=smoke \
bash tools/run_rt55_japan_full_2000_2024_slurm.sh
```

必须确认：数据加载完成、mask shape 正确、forward/backward 无 NaN、初始 checkpoint
和 `full_model_last.pth` 均生成。

## 正式训练与续训

正式默认申请 4 节点 × 4 DCU，先训练到总 epoch=12：

```bash
JAPAN_FULL_DATA_ROOT="$JAPAN_FULL_DATA_ROOT" \
bash tools/run_rt55_japan_full_2000_2024_slurm.sh
```

脚本检测到 `full_model_last.pth` 后会自动续训，不删除已有 checkpoint。Slurm 超时后
直接重复同一命令即可。

若 12 epoch 时 validation 仍持续改善，把总目标延长到 20（不是再额外跑 20）：

```bash
JAPAN_FULL_DATA_ROOT="$JAPAN_FULL_DATA_ROOT" \
EPOCHS_FULL_MODEL=20 \
bash tools/run_rt55_japan_full_2000_2024_slurm.sh
```

如需强制要求存在续训点：

```bash
JAPAN_FULL_DATA_ROOT="$JAPAN_FULL_DATA_ROOT" \
REQUIRE_RESUME=1 EPOCHS_FULL_MODEL=20 \
bash tools/run_rt55_japan_full_2000_2024_slurm.sh
```

## rt55 的计算控制

- adapter：rt52 legacy + 显式 padding mask；
- DiTing encoder：冻结；
- DPK/cache/temporal residual：全部关闭；
- train：每个事件、每 epoch 从 7 个实时 bin 中无放回抽 3 个；
- validation：固定 1/3/5/10/20/40/90 s，各算一次；
- PGA normalization：只用 KNET train pool 固定计算，mean=-1.1228067351，std=0.4312468402；
- 第一段总 epoch：12，可通过 launcher 提高后严格续训；
- checkpoint 保存 optimizer 和 LR scheduler，超时后从上一个完整 epoch 末继续。
