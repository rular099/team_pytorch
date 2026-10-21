# FE01-A01 结果表口径

本目录均来自 `outputs/review.tar.gz` 的validation证据。所有模型seed42。
16份导出对应8个模型条件的两套checkpoint规则，不是16次独立训练。

## 主结果和人口

`model_summary.csv`：一行对应family/mode/checkpoint_policy/target_group。

- `mode`：ON=physical_scale_keep_duration，OFF=no_absolute_scale_keep_duration。
  OFF只移除显式绝对幅值通道，保留有效时长和归一化波形；不要称为删除所有幅值形状信息。
- `reused_old_ON`：TEAM/PhaseNet/EQT的ON复用旧训练权重；其保存run_id可包含off_reused_ON，真实条件以模式标签和身份记录为准。
- `checkpoint_policy`：selected是预定验证指标选轮；epoch12固定训练12轮。
- `epoch`：来自checkpoint_identity内部epoch记录，非目录名猜测。
- `equal_cell_mae`：每个支持的几何/时刻cell内目标等权，然后cell等权。noninput有10个共同cell，是主指标。
- `equal_cell_rmse_of_mse`：先平均每cell MSE再开方，命名与平均每cell RMSE不同。
- `equal_cell_nll/crps/brier/covered68/covered95/predictive_sigma/width95`：同样cell等权。
- `pooled_mae/rmse/r2/prediction_on_truth_slope`：合并该组全部请求后计算，不能当作主端点权重。
- `targets`：跨时间/几何的目标请求数；`events`是dataset/event联合去重；`decisions`是dataset/event/time/geometry去重。
- `supported_cells`：实际非空cell数。triggered_noninput的normal/1秒为空，只有9个cell。

`validation_by_time_geometry_role.csv`：960行，每行一个固定模型条件、目标组和时间/几何cell。
保留空cell，valid_targets=0时指标为N/A，不能填零。
`time_role_population.csv`去掉模型重复，显示各cell实际事件、decision和目标数量。
normal下部分目标为observed_input；random的当前查询均为noninput，目标总数随时间变化。

目标组：

- `all`：全部合法目标请求。
- `noninput`：所有非observed_input请求。
- `triggered_noninput`：已触发、未选作输入的请求。
- `untriggered_noninput`：尚未触发的请求。
- `K1_noninput`：该decision实际一个输入站，且目标为非输入站。
- `Kmulti_noninput`：该decision实际多输入站，且目标为非输入站。

组之间有包含和重叠关系，不能将样本量相加当作新事件。
跨时间的自然输入/角色/查询人口变化，不能仅凭时间曲线作同一目标轨迹的解释。

## 概率指标

点预测为MDN在log10(m/s²)坐标的混合均值。
NLL使用保存的原始logits作稳定logsumexp；CRPS/Brier按真实高斯混合分布计算。
Brier的超阈值坐标为-1.2；不是所有PGA阈值的平均Brier。
covered95为混合分布2.5%–97.5%分位区间的覆盖率；covered68对应15.8655%–84.1345%。
它们与均值±1/2个预测sigma的覆盖率是不同定义。
width95与predictive_sigma均为dex；NLL的密度坐标固定为log PGA，可能为负。

## 配对与重采样

`paired_mae_summary.csv`：每行给同家族ON/OFF的绝对值、差值、相对ON变化和原区间。
`paired_event_bootstrap.csv`：原8组比较，共672行，保留所有指标、子组及支持状态。

所有差值均为OFF减ON：

- 对MAE、MSE、NLL、CRPS、Brier、level/shape/field MSE、range_error、等距离误差，增大通常表示该误差量增加。
- 对sigma和宽度，增大表示预测分布更宽；不直接等于更好或更差。
- 对coverage，增大表示覆盖率提高；需与名义值、宽度及proper scores一起解释。

事件cluster为dataset/event，所有站点、时刻、几何随同一事件重采样；2000 draws，seed20261007。
目标端点是每cell目标等权后cell等权。场端点是每cell eligible field等权后cell等权。
`events/targets/fields`取决于子组与指标，不能套用主端点1310事件的分母到所有子组。
`fields`在目标指标行表示非空event/cell数，在场指标行表示eligible fields数，应结合weighting阅读。

`bootstrap_support_audit.csv`专门列出支持与有效重采样数量。
`SPARSE_AVAILABLE_CELLS_ONLY`表示只在实际支持cell上计算，不能冒充10-cell端点。
即便status=PASS，也要查empty_cell_fraction；K1 field的有效重复数少于2000。
这些区间不包含模型重训或选轮的不确定性，也没有进行多个次要端点的多重比较校正。

`event_cells_noninput.csv.gz`：每行一个模型条件及dataset/event/time/geometry，保存目标数和误差/概率量的和。
请用sum除以targets恢复cell目标均值，再按cell等权；不要先平均事件均值来代替原目标加权。
该表用于复算原主端点与其事件区间；不含逐目标数据或K1分组分解。

## 空间场

优先读`paired_field_levels.csv`和`paired_field_by_cell.csv`：覆盖全部8个ON/OFF模型条件，
来自原比较中的spatial_fields，双方严格使用相同的eligible fields和目标数。
只有至少5个目标的event/time/geometry field进入这些场端点。

- `level_mse`：field平均误差的平方，单位dex²。
- `shape_mse`：扣除field平均误差后的误差均方，单位dex²。
- `field_mse`：整个field的误差均方；等于上述两部分之和，已按数据验证。
- `truth_range/prediction_range`：field内95%与5%分位差，单位dex。
- `range_error`：两种range的绝对差；`range_ratio`仅在truth_range至少0.05dex时定义。
- `equal_distance_delta_mae`：震中距差不超过5km的站点对，其预测差与真实差之间的MAE，单位dex。
  它是相对空间结构端点，不能直接等同于独立场地效应识别。
- `fields`和`targets`仅属于eligible field人口；`*_fields`列记录该指标非空field分母。

`field_summary.csv`、`field_by_time_geometry_role.csv`来自五组新训导出的原生fields表，
沿用FE01-review的分组名，包括observed_input和actual_single_input_noninput。
与上述A01配对表的六组命名范围有差别，不要直接叠加或重复计数。

## 机制与边界诊断

`diagnostic_*`每行均保留index和evidence。索引0–2为旧ON权重推理时OFF干预；
索引3–4为未训练DiTing匹配初态。
自然probe每组78个decision，最多每stratum两个，不是完整validation空间性能。

`diagnostic_query_stages.csv`汇总每个stage的跨query离散度及分支量。
`multiple_query_*`只取至少两个有效query；原query只有一个时零离散度没有诊断意义。
不同stage的向量维度和单位不同，不把它们的原始范数直接解释为统一信息量。

`diagnostic_query_sensitivity.csv`为局部坐标梯度和0.01/0.1/1km有限差分摘要。
这些是敏感性，不是预测正确性；大步长的非线性差异也不等于梯度计算错误。

`diagnostic_nested_stages.csv`按各K实际支持分别汇总，cohort会变。
`diagnostic_nested_matched.csv`仅用共同支持K=1/3/5/8、至少两个query的相同18个decision，
事件、时刻、查询台站固定；其输出离散度仍不是相对标签的准确度。
`diagnostic_nested_support.csv`记录可用/不可用K，不是额外预测成绩。

scale/future/permutation检查只支持相应前缀、gain和机制控制范围。
UNSUPPORTED病例不伪装成PASS；上游离线预处理保持UPSTREAM_CAUSALITY_UNKNOWN。

## 身份和来源

`method_sources/`保存六份指标、配对、诊断和模式源码文本，并核对原HPC声明的文件SHA。
用于检查定义和状态选择，完整执行仍须在原工程环境中完成。

`source_inventory.csv`是373个tar成员的字节SHA清单；`source_seal_checks.csv`区分原字节匹配与脱敏JSON。
`evaluation_execution_sources.csv`记录原执行、TEAM修复、DiTing修复的各项SHA。
`source_metadata/`保存轻量原记录，路径已由HPC pack脱敏，不是可直接在超算运行的配置。
原始权重、波形、sample journals、trace arrays及原私有plan不在包内。
因此保存的checkpoint字节SHA是超算导出身份记录，本地核对的是该记录和预测，并未重新载入权重。
