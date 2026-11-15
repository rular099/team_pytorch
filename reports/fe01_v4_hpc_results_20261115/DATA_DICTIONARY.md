# 数据字典与估计量边界

本批均为validation、seed42。selected依次为DiTing12/PhaseNet12/EQT10/TEAM11；旧预测另有epoch12敏感性。
所有PGA误差、bias、sigma、区间宽度的坐标为log10(m/s^2)（dex）；时钟以首个P到达为零点、单位秒。
除明确的origin列外，不使用发震时刻作默认零点。阈值−1.2和0.0也在该PGA坐标中。

`source_tables/`保存从本次review原样复制的小表；完整成员路径、SHA和字节数见SOURCE_MANIFEST。
`derived/`是从已下载的完整事件记录、时延记录和历史同SHA预测事件表重建的表；输入也在manifest中声明。
`metadata/`保存合同、checkpoint内部epoch/SHA、来源、预选案例及完整图库索引。
`method_sources/`是49个主程序pin中的方法快照，只供定义与审阅；本轮没有改变这些程序。

旧预测指标：

- `cross_encoder_metrics.csv`：family、checkpoint_policy、epoch、target_group、endpoint确定一行。
  主端点为10 cell内目标等权，再对cell等权。`equal_cell_rmse`是cell MSE均值开平方。
  `pooled_*`按所有请求合并，R2和斜率列为pooled；不能给它们套用equal-cell的权重解释。
- `all`包括输入目标；`noninput`为该时刻非输入；`K1_noninput`、`Kmulti_noninput`等具有独立人口。
  events为事件去重，requests为跨时刻/几何请求数。目标角色/可用目标可能随时间变化。
- `paired_effects.csv`的estimate为DiTing减comparator。MAE/RMSE/NLL/CRPS/Brier负值表示较小；coverage差值的方向需与名义覆盖率一起解释。
  NLL是该log-PGA坐标上的负对数密度，可以为负。Brier使用PGA阈值−1.2的真实超阈标签。
- `strata_support.csv`、`bootstrap_support.csv`及各效果表的events、supported_cells、effective_replicates、
  empty_cell_fraction必须与估计一并报告。SNR代理有明确Unknown，不能将Unknown合并进低/高SNR。
- `legacy_on_event_cells.csv.gz`含每事件/几何/时刻的targets及误差/概率指标分子。
  来源为先前已验证的A01 ON预测，其八份prediction SHA与本次导出验证逐一匹配，见metadata中的crosscheck。
  它可以复算旧主端点，不能充当新固定query的逐秒预测。

新固定panel与TTUM：

- `fixed_all_targets`是事件的固定全目标集合；`fixed_never_input_targets`排除1–20秒期间任何时刻成为输入的目标。
  它不是legacy逐时刻noninput。`fixed_high_PGA_ge_minus1p2`按回顾性最终PGA真值选取，属于回顾性分层。
- 至少5个固定目标才适格；不适格事件仍在total_cohort_events和ineligible列。normal never-input适格207/1310，
  random为539/1310；不能把207事件上的差异解释为全1310事件或未见训练台站泛化。
- station gate同时检查MAE、绝对bias以及绝对误差不超过两倍预算的目标比例；field gate再检查centered RMSE和P95−P05空间range误差。
  预算0.15/0.20/0.25 dex、三个连续秒帧及5/10/20秒报告点均为预先指定。
- `stable_confirmation`是首次出现三个连续通过帧时的第三帧。它没有要求之后一直通过。
  `stable_onset_retrospective`是该三帧的起点，不能作为当时已确认的实时交付时刻。
  `last_nonpassing_frame`晚于confirmation说明后续有已知不通过帧；派生degradation表保留这项描述性敏感性，不修改原端点。
- `ideal_cutoff_replay`按数据cutoff计时；`measured_compute_replay`按完整数值地图FIFO时钟；
  `measured_rendered_map_replay`按完整数值+共享比较图渲染FIFO场景计时。random只有理想场景。
- `ttum_support_recomputed.csv`中的fraction_lower是期限前已确认数除以全部适格事件；fraction_upper加上Unknown。
  这些是支持/未知界限，**不是统计置信区间**。不达标、缺失和未适格事件不得丢掉或填成成功。
- `ttum_paired_b020_recomputed.csv`保留三个panel、两个gate、全部已有几何/时钟；预算固定0.20。
  条件时间差只针对双方可确定的事件；CDF差值则使用全部共同适格事件、未知不算成功。
  `known_events`与`events`不同，必须报告。原HPC其他预算的配对区间保留在source_tables中。
- `ttum_event_records.csv.gz`保留全部377280条事件/模型/panel/budget/gate/scene记录和必要时钟。
  数据包中两个原TTUM事件文件名字不同但字节相同，不能当两份独立证据。

shadow与误报：

- 是最终PGA阈值条件下的有效预警失败比例；不是连续区域实测盲区。面积始终NA。
  阈值−1.2/0.0、p_alarm为0.1/0.3/0.5/0.7/0.9、tau为0/3/5/10秒，全网格保留，不按结果选阈值。
- 原始概率首次超过p_alarm后闩锁；使用同一首P时钟的标签一致瞬时幅度首次越阈真值。
  timely要求提前量严格大于0且至少tau。已知失败、timely和unknown之和等于affected_targets。
- 单事件failure_fraction_lower是已知失败除以affected；upper再加unknown。没有affected时为空，不填零。
  `shadow_paired_summary.csv`报告事件等权的差值界：DiTing下界减对照上界，以及DiTing上界减对照下界。
  两个界各自的bootstrap区间不能误读为单一确定差值的区间。
- `shadow_rendered_events.csv.gz`保存已下载渲染场景的全部事件汇总；包含312个有never-input目标的normal事件，
  其中214个在−1.2阈值有affected目标。shadow没有TTUM的至少5目标资格规则，分母不能混用。
- `shadow_rendered_summary.csv`同时列出event-equal与pooled（按affected目标加权）比例；
  false-positive率只对完整观测的unaffected目标计数，并保留不完整负例及已知误报数量。
  precision是告警目标中真实affected的比例，不是timely率。
- 固定目录事件内的空间误报率不能换算成每日连续运行误报次数；阈值/PGA群体改变时需重新报告分母。

时延、队列与图：

- `latency_recomputed.csv`每模型/每scope有6事件、36个条件，每条件2 warmup和3 measured重复（108记录）。
  P50/P95是同步后的完整station-only或whole-map数值服务分布，不是重复次数形成的置信区间。
  查询数、K、precision及device_name均保留；同名设备不证明不同分配的运行环境完全相同。
- cold_load单列，包含checkpoint认证/读取/构造/加载，开始于Torch导入与设备初始化之后。
  I/O和render/publish另列，不将单query分摊时间充当整帧耗时。
- `queue_event_frames.csv.gz`为normal的104800帧，1310事件×4模型×20cutoff。
  每事件、每模型一个顺序FIFO场景；前一帧渲染完成后下一帧才可在该场景开始。
  render_seconds是实际四模型比较图的CPU渲染+PNG发布成本，同帧统一计入每模型的场景，通信仍UNKNOWN。
  这是特定比较图工作流，不能直接当作单模型线上部署实测。
- `queue_quantiles.csv`的cutoff_20s是第20帧何时交付，不是TTUM何时首次达标。
  rendered队列中的numeric_ready也已受到前序render阻塞，不等于不含render的独立numeric队列。
- CASE_INDEX列出全部12个原元数据/hash预选案例，统一取5秒/20秒关键帧；没有按误差选图。
  灰色不支持网格、输入目标和回顾性台站真值需按原图例解释。完整动态图在原107MB包/超算图库中。

本地新脚本是只读结果整理。216个已有TTUM区间复现，预算0.20的rendered区间另行新增；
没有重算未下载的逐目标概率或连续网格误差。单seed/validation选轮、探索性多重比较和上游因果性未知的限制持续存在。
