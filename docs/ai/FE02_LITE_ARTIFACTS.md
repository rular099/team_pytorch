# FE02-Lite 产物 schema、单位和读取

每个run独立目录，名称包含视图/stage/variant/seed：

```text
OUTPUT_ROOT/
  launches/fe02-lite-*/{launch.env,tasks.json,submission.json,sbatch_receipt.txt,source/}
  offline_weights/pretrained_manifest.json              # encoder外部路径/bytes/SHA
  audits/RUN/
    protocol.lock.json, model_audit.json
    data/{data_audit.json,view_records.csv.gz,view_filter_counts.csv,exclusions.csv,
          validation_population.csv.gz,diagnostic_plan.json}
  RUN/
    resolved_config.json,protocol.lock.json,parameters.json,runtime_ATTEMPT.json
    init.pth,epoch0001.pth,...,best.pth,last.pth
    sample_journals/epochE_ATTEMPT_rankR.jsonl.gz
    step_logs/epochE_ATTEMPT_rankR.jsonl.gz
    training_curves.csv,TRAINING_COMPLETE.json,COMPLETED.json
    validation/epochE_ATTEMPT/{predictions/,sources/,statuses/,selection.json}
    evaluations/{best_TIMESTAMP,last_TIMESTAMP}/{predictions/,sources/,statuses/,analysis/}
    diagnostics/{TAG_SAMPLE.npz,TAG_SAMPLE.json,index.jsonl}
```

## 数据和配置身份

data_audit记录实际 train/val selected records、有效记录、included events/stations、
排除行和原因统计；view_records保存sensor/physical ID、层位、原HDF5行号、有效性、
cohort身份、原始坐标、PGA标签。view_filter_counts记录每event源总行数/视图选择数/
其他视图排除数；无split事件写exclusions。shards记录路径/bytes/mtime/SHA，若复用
已有hash会保留来源。view manifest逻辑内容hash与压缩文件字节hash分开，避免gzip
时间戳改变导致不同架构误判为不同数据。

target_normalization只从该视图included training事件的有限传感器标签拟合 population
mean/std；它与逐站当前前缀归一化无关。coordinates manifest固定全局转换和正负向，
井下深度缺失明确记录。源标签和最终PGA只作监督/评价。

每次runtime记录Git/source package、实际command、Python/PyTorch/依赖/设备、ROCm、
world/batch/累积/precision。parameters包含全部注册/可训练计数、分模块计数、optimizer
的精确参数名称与shape。model_audit另记录真实encoder类/feature/patch参数和严格资产
身份，不能把本地合成encoder报告混入正式审计。

## 表格式与评价字段

predictions/sources/statuses为压缩NPZ分片，每片最多4096行，每列为numeric或Unicode
数组，无pickle。manifest.json列出schema版本、列类型、行数、每片bytes/SHA和checkpoint
身份。完整评价无诊断存储上限，也不会采样掉checkpoint选择目标。

```python
from fe02_lite.artifacts import read_table
from fe02_lite.analysis import scored
frame = read_table('/actual/evaluation/predictions')   # 校验每片hash和行数
source = read_table('/actual/evaluation/sources')
recomputed = scored('/actual/evaluation')             # 从w/mu/sigma独立重算全部概率指标
```

| 字段组 | 定义 |
|---|---|
| 实验 | dataset_view,run_id,variant,seed,stage,split；checkpoint_epoch/step/sha；config_sha |
| 数据 | data_shards/split/view_manifest/cohort hash、encoder weights/YAML hash、code hash |
| 样本 | canonical_event_id,event_id,dataset_id,observation_id,input_set_id,record_id,station_id,physical_station_id |
| 几何 | sensor_class,target_role,input_count,input_ids,input_coords；normal/random |
| 时间 | elapsed_time/actual_elapsed_time、reference_sample、record_axis_start_utc、current_sample、cutout_exclusive、history_start_sample |
| UTC | absolute_cutoff_utc=公共轴+current_sample/100，为最后已到达采样时刻；absolute_exclusive_end_utc=公共轴+cutout_exclusive/100 |
| 位置 | target latitude/longitude degrees、height_km向上；目录event_latitude/longitude、depth km向下、magnitude仅标签 |
| 真值 | truth为log10(m/s²)；target_valid/status/reason；normalization mean/std/hash和coordinate hash |
| MDN | w0/w1/w2概率、mu0..2均值[dex]、sigma0..2标准差[dex]，全部已反PGA归一化 |

prediction是 log-space mixture mean E[log10(PGA)]，不是物理PGA算术均值。
linear_mixture_mean_mps2才是 E[PGA]，通过每分量lognormal矩计算；median另由混合分位数。
q025/q975与q16/q84是混合CDF反解的等尾分位数，区间在dex坐标。NLL为该dex坐标的
密度，CRPS也为dex，Brier为PGA>10^-1.2m/s²阈值事件，无量纲。PIT/coverage/width/sigma
均保留，可以脱离模型重算。R²在真值方差0时输出null，不制造数值。

source表每个“事件＋截止时刻＋输入几何＋protocol”仅一行，不随target或chunk重复。
location_mdn_normalized为 `[3,7]`，顺序logit、3mean、3sigma；另保存原坐标location_weights、
location_mu[3,3]（纬度deg/经度deg/深度km正下）、location_sigma[3,3]。
magnitude_mdn为 `[1,3]`（logit=0、M、sigma_M）。保存对应完整真值和观测身份。

## 训练与恢复

journals保存每个事件draw的epoch/step/rank/micro、事件/record/target ID、输入集合、
角色、时间轴、valid seconds、mask SHA和采样种子/规则的可重算索引；不重复保存原波形。
step logs保存所有loss原始/加权项、LR、clip前梯度范数、指定频率模块梯度/更新范数、
测量开销、最后microbatch的数据/计算wall time、样本/目标/台站计数、显存峰值和异常。
scope明确为rank累计update或最后micro快照；training_curves是all-rank epoch汇总。

last是完整epoch边界提交点，含task state、optimizer/scheduler、AMP=null、各rank RNG、
sampling_position和committed_files的hash。失败attempt未被last引用，离线分析只读取
committed日志。epoch未完成会重放，未声称支持任意microstep即时恢复。init与逐epoch
是轻量任务头快照；best/last保留恢复状态，但encoder均只通过外部路径/SHA严格引用。

## 固定诊断

在模型预测前按 SHA(seed,event) 选最多32个validation事件；1/20秒×normal/random×
单站/原完整输入，init/best/last导出。保存时间幅值feature、pool weights、token mask/
visible fraction、u、m0/m、e/source、station/target r(c)、query/h、event/target attention、
PGA分布；对应station/token/target映射、物理坐标、window和record轴token起点在JSON。
中间浮点默认FP16，mask保持原类型；默认2GiB/run，index记录skip和原因。完整逐目标
评价不受此cap约束。开启诊断不修改输出、RNG和正式训练抽样，已作对照测试。

默认没有原生1792×200特征。显式将diagnostics.native_feature_debug=true且max_events=1
才允许额外debug导出；仍受cap控制，不加入默认任务矩阵。

## 分析范围与配对

analyze导出训练曲线/梯度更新及性能、各时刻/normal-random/单多站/角色分层指标、
MAE/RMSE/bias/R²/NLL/CRPS/Brier/coverage/width、同事件centered MAE和pairwise-delta MAE、
预测/真实P95-P05及max-min范围、源误差和对应PGA误差关联。空间centered/pairwise
至少5个noninput目标，range ratio另要求真值P95-P05≥0.05dex；不足时保留数量与不适用原因。

compare/collect同时输出全部视图样本表和共同canonical事件子集指标；跨视图record
匹配要求canonical事件、可验证physical ID、**相同绝对截止时刻**和protocol，导出左右
原记录，保留各自真值/输入集合。不同网络没有已确认物理同址映射时不强行匹配。
共同事件指标不伪装为相同物理时刻的严格配对。

同视图架构配对要求data/split/cohort/encoder、实际时间、input ID/coordinates、target/
role/truth和归一化身份完全一致，缺证据则写refused。区间以canonical event簇bootstrap，
不把同事件多个target/time当独立样本。source相关性是描述统计，不作为已证明因果联系。
每张图旁边有完整绘图CSV、单位、筛选条件和数量JSON；注意力只作诊断，不解释为传播
射线或地下介质。普通event split不是未见站址泛化。
