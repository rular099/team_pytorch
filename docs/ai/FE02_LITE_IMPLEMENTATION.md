# FE02-Lite 实施、数据审计与张量接口

基点 `618661f99afb67f5c23c9f1a3fd713a0a156e7b5`，新分支
`exp/fe02-lite-event-spatial`。本轮只交付实现、验证、提交与分析入口，不启动正式训练。

## 实际复用与隔离

复用 FE01 `split_catalog`、`build_window`、`clock`、`TimeSampler`、`geometry`、
`balanced_targets/prepare_sample`、`collate`、严格预训练资产校验、DiTing 工厂以及
FE01 的 MDN 概率指标、spatial field 指标、事件簇 bootstrap。新增小 runner 是因为
原 `fe01/data.py` 固定 `source_network=knt`，原 factory 构造旧大型下游，原 checkpoint
包含整个 encoder，不能满足本任务。没有修改原 FE01/FE02/RT55 的 Python、配置或脚本。

训练协议仍为当前可见 native prefix、5秒 pre-P、50% normal/random、每事件每轮
3次 early-weighted 随机时刻、原目标角色/随机输入数。主任务12 epochs，有效batch128、
microbatch8、Adam 1e-3、固定更新数 cosine、clip5、FP32。不同视图 cohort 大小不同，
更新数和耗时会不同；日志记录实际数量，不称为相同计算预算。

## 数据语义：已查验代码，真实文件待审计

| 项目 | 已确认的存储/处理 |
|---|---|
| 定位 | frozen split 的 source_data_path basename → DATA_ROOT/year/japan_year.hdf5；group `data/EVENT` |
| 波形 | `waveforms[sensor_row,time,3]`，NS/EW/UD，即 NEZ；100Hz；物理 m/s² |
| KNET | `source_network=knt`, `sensor_class=single_surface`, suffix 空串或 `0` |
| KiK井下 | `source_network=kik`, `sensor_class=borehole`, suffix `1` |
| KiK地表 | `source_network=kik`, `sensor_class=surface`, suffix `2` |
| 传感器ID | `network:station_code:layer`；记录ID再加 shard 和 EVENT；物理站址ID `network:station_code` |
| PGA | 同一行 `pga` = log10(`pga_norm_resampled_mps2`+1e-10)；标签是 resampled 三分量向量模的时间最大值，不是最大单水平分量 |
| 有效性 | `[record_start_sample, record_start_sample+valid_n_samples)`，可选 NEZ component_valid / waveform_sample_valid 取交集 |
| P | `p_picks` 已对齐至 shard event sample axis；无效 P 不可触发；参考是所选视图有效 P 的最早值 |
| 绝对时间 | `record_start_time_jst - record_start_sample/100` 重建公共轴；跨传感器容许至多一个采样点误差；保存 UTC seconds |
| 接收点 | coords=[纬度deg,经度deg,Station Height(m)/1000]；高程正向UP；没有已确认的井下接收器深度字段 |

来源：`japan_dataset_builder.py` 的 COMPONENT_ORDER、_sensor_class、build_station_trace、
事件dataset写入；`build_japan_flat_training_data.py` 的 DIR_SENSOR_MAP、单位转换、
station-row 数据写入。raw archive KNET suffix 空串，flat CSV builder 用 `0`，两者均有
明确来源。raw NS1/EW1/UD1 为井下，NS2/EW2/UD2 为地表；当前 builder 已拆成独立行。
不会把原 six-component `japan.py` 数据当成本协议输入，也不会猜前三/后三通道。

runtime 审计校验 sampling_rate、suffix/layer 一致、PGA log/linear 一致、坐标和区间、
重复 sensor ID、三分量掩码、绝对时间轴；额外对每视图最多3条有效完整记录复算
vector-norm peak，捕获幅值单位、行/层位/标签不匹配。该完整读取仅用于审计标签，
不进入训练输入或模型统计。真实计数写入 `audits/RUN/data/`，本地没有这些真实报告。

审计导出全部 train/val 已筛选行、其他视图排除计数、无效行原因、cohort 排除列表、
实际事件/传感器/记录数和每视图 train-only PGA mean/std。未知 EVENT 没有 frozen
split 身份时记录并排除，绝不独立重分视图。同一 EVENT 在现有 manifest 中跨 split
将立即失败；本轮没有猜测基于源目录或近似时空阈值的事件别名。

井下使用原 station height 明确作为代理，不能解释成井下真实 receiver elevation。
若将来提供井深，需版本化坐标 manifest、重新审计并新建实验身份。

上游存在完整记录去均值、离线 resample/filter 与回顾性 P picks；本实现只证明
**给定存储波形后的当前前缀因果性**，不能据此认证原始采集到预测的在线因果性。
预训练语料重叠和原预训练归一化未知，均在资产登记中保留 unknown。

## 模型与接口

输入6张量：waveform `[B,N,3,10000]`，station coordinates `[B,N,3]`，station-valid
`[B,N]`，target coordinates `[B,Q,3]`，target-valid `[B,Q]`，received/storage mask
`[B,N,10000]`。没有有效观测的 batch member 立即返回明确 unsupported 异常。

| 阶段 | 主配置形状/定义 |
|---|---|
| 当前前缀 | 先与 storage mask 取交集，逐分量可见均值；三分量共同 peak 只用于 DiTing 归一化，未输入统计量 |
| DiTing | 真 MAE1200M，width1792/depth24；no_grad + eval + requires_grad=False；输出 `[B,N,1792,200]` |
| token对齐 | 原生 `backbone.W_p=Conv1d(3,1792,k50,s50,p0,dilation1)`，mask_ratio=0、ids_keep升序；实际属性和shape在audit核验 |
| 幅值 | 同一已掩码、去当前均值的 m/s² 波形；固定乘数1；Conv1d(3,8,50,50)+GELU；channel-major每块150元素；`[B,N,200,8]` |
| 拼接 | Linear(1792,120) + 8幅值维 → `[B,N,200,128]`；整块无效排除，部分块只编码已到达位置 |
| 学习池化 | 两个query，对LayerNorm后的key做点积score，raw token作value；输出u `[B,N,2,128]` |
| 坐标 | 固定全局 `(coord-[37,140,0])/[5,5,100]`；第三维km向上，catalog source depth先取负；保留绝对位置 |
| m0 | E_c:3→128→128；ψ:256→256→128；wave summary与坐标非线性融合 |
| event | 一个query、一层4-head cross-attention + residual + LN/128→256→128 FFN；e `[B,128]`，不依赖目标集合 |
| source | 128→32→21 三个三维 diagonal Gaussian；128→32→2 单Gaussian震级；不反馈分布参数 |
| spatial | 同一φ:256→256→128用于station和target；memory=m0+φ(e,E_c(station))；q=φ(e,E_c(target)) |
| target | 一层4-head cross-attention，h=q+attention；无target self-attention；384→256→9 MDN |

所有MLP隐藏激活GELU，attention dropout0，无额外gates。池化部分块的bool有效性与
visible fraction 均保留；fraction用于导出，不作为人工幅值输入。没有对raw value通路
再做逐样本幅值归一化。

主任务参数740,592；分模块完整实际计数和optimizer名称列表见本地 parameters.json。
模型参数按参数名独立 seed 初始化，相同形状的公共参数跨消融可配对。amp16/amp32
分别将 DiTing 投影改为112/96维，总宽度128；它们不是等参数对照。d64使相关宽度
按d缩放，幅值8、辅助头隐藏32、attention heads4不变。no_source_aux保留event/source
结构，只令source loss权重0，事件路径仍由PGA训练。

## loss、选择与恢复

MDN沿用 `[logits, D means, D softplus(raw_sigma)+1e-4]`。NLL复用既有公式；任务
权重PGA/location/magnitude=1/.02/.02除以权重和，distribution-mean smooth-L1
按同一权重、beta1再乘0.1，保持FE02数学约定。PGA loss在该视图训练集标准化坐标中，
定位在上述统一固定坐标中，震级保持目录量纲。原始及加权项均记录，正式PGA评分一律
转换回 log10(m/s²)。source项每batch事件观测计算一次，query chunk只用于评价读出。

validation十cell（normal/random×1/3/5/10/20秒）noninput MAE等权选择，平局取早期；
40/90秒只做最终诊断。每次选择保存完整逐目标分布和epoch head SHA，且核对审计锁定的
全部query/input/role身份。test不暴露在本轮CLI中。

init/best/last与逐epoch小头保留。last存optimizer、scheduler、AMP=null（明确FP32）、
各rank Python/NumPy/CPU/device RNG、下一epoch位置、提交journal/log路径及hash；只从
last续训。仅load task state不是继续训练入口。中断epoch从上次提交的epoch边界重放，
失败attempt日志和未提交epoch快照保留；不用best回滚，不重置已提交RNG。恢复在CPU
load，并对所有device RNG ByteTensor显式.cpu()，保留既有ROCm修复语义。

正式run身份锁包含config、代码/依赖、数据/划分/cohort、DiTing、world size、累积与
更新预算。冻结权重不反复保存。完成run禁止训练覆盖；评价写入新的独立目录。
