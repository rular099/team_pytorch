# FE01 ON V4 数值阅读表（不含 Codex 判断）

仓库 rular099/team_pytorch；整理基线99d9b1c4bb5c47fc26125df7eea2648917e90462。
全部为validation、seed42、ON；selected依次DiTing12/PhaseNet12/EQT10/TEAM11。
PGA坐标为log10(m/s^2)。原始归档SHA与复算范围见verification.json、local_numerical_checks.json。

旧snapshot主端点：1310事件、139440非输入目标请求；normal/random × 1/3/5/10/20秒的10 cell先各自目标等权，再cell等权。
下面仅列原表数值；这是legacy_snapshot_v1，不是新逐秒固定query人口。pooled R2/slope与equal-cell MAE权重不同，保留原列名。

| family | epoch | events | requests | equal_cell_mae | equal_cell_rmse | equal_cell_nll | equal_cell_brier | equal_cell_predictive_sigma | equal_cell_covered68 | equal_cell_covered95 | pooled_r2 | pooled_prediction_on_truth_slope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DiTing | 12 | 1310 | 139440 | 0.222859 | 0.296500 | 0.115770 | 0.169824 | 0.248804 | 0.614525 | 0.914884 | 0.432956 | 0.480508 |
| PhaseNet | 12 | 1310 | 139440 | 0.225697 | 0.300372 | 0.150267 | 0.173825 | 0.248571 | 0.607498 | 0.913038 | 0.412157 | 0.447309 |
| EQT | 10 | 1310 | 139440 | 0.236407 | 0.312033 | 0.167722 | 0.180948 | 0.268578 | 0.628063 | 0.922287 | 0.377906 | 0.416067 |
| TEAM | 11 | 1310 | 139440 | 0.245159 | 0.323800 | 0.217655 | 0.188617 | 0.268215 | 0.605038 | 0.911661 | 0.329485 | 0.380688 |

对应的全目标主端点（194265请求）见同一源表的target_group=all；非输入和全目标须分别报告。

主MAE配对差为DiTing减对照，单位dex。2000次固定模型事件cluster bootstrap、seed20261007，未校正多重比较。

| checkpoint_policy | comparator | metric | target_group | endpoint | events | requests | estimate | ci_low | ci_high | effective_replicates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selected | PhaseNet | mae | noninput | primary_10cell | 1310 | 139440 | -0.002838 | -0.005120 | -0.000580 | 2000 |
| selected | EQT | mae | noninput | primary_10cell | 1310 | 139440 | -0.013548 | -0.016358 | -0.010949 | 2000 |
| selected | TEAM | mae | noninput | primary_10cell | 1310 | 139440 | -0.022299 | -0.026050 | -0.018831 | 2000 |
| epoch12 | PhaseNet | mae | noninput | primary_10cell | 1310 | 139440 | -0.002838 | -0.005120 | -0.000580 | 2000 |
| epoch12 | EQT | mae | noninput | primary_10cell | 1310 | 139440 | -0.014012 | -0.016638 | -0.011437 | 2000 |
| epoch12 | TEAM | mae | noninput | primary_10cell | 1310 | 139440 | -0.022472 | -0.026130 | -0.019073 | 2000 |

新固定query TTUM的预定预算0.20 dex、field_pass、20秒数值例子如下。全预算0.15/0.20/0.25、两个gate、三个panel及5/10/20秒均保留在完整CSV中。下列unknown为支持状态界限，不是置信区间。

| geometry | panel | family | eligible_events | cohort_events | ideal_20s | numeric_20s | rendered_20s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| normal | fixed_all_targets | DiTing | 1057 | 1310 | 676/1057; unknown=27 | 455/1057; unknown=27 | 78/1057; unknown=27 |
| normal | fixed_all_targets | EQT | 1057 | 1310 | 597/1057; unknown=40 | 568/1057; unknown=40 | 88/1057; unknown=40 |
| normal | fixed_all_targets | PhaseNet | 1057 | 1310 | 804/1057; unknown=37 | 783/1057; unknown=37 | 125/1057; unknown=37 |
| normal | fixed_all_targets | TEAM | 1057 | 1310 | 491/1057; unknown=8 | 473/1057; unknown=8 | 97/1057; unknown=8 |
| normal | fixed_never_input_targets | DiTing | 207 | 1310 | 88/207; unknown=6 | 49/207; unknown=6 | 15/207; unknown=6 |
| normal | fixed_never_input_targets | EQT | 207 | 1310 | 51/207; unknown=4 | 46/207; unknown=4 | 10/207; unknown=4 |
| normal | fixed_never_input_targets | PhaseNet | 207 | 1310 | 73/207; unknown=5 | 69/207; unknown=5 | 16/207; unknown=5 |
| normal | fixed_never_input_targets | TEAM | 207 | 1310 | 66/207; unknown=4 | 61/207; unknown=4 | 7/207; unknown=4 |
| random | fixed_all_targets | DiTing | 1057 | 1310 | 358/1057; unknown=15 | not measured | not measured |
| random | fixed_all_targets | EQT | 1057 | 1310 | 322/1057; unknown=17 | not measured | not measured |
| random | fixed_all_targets | PhaseNet | 1057 | 1310 | 423/1057; unknown=28 | not measured | not measured |
| random | fixed_all_targets | TEAM | 1057 | 1310 | 271/1057; unknown=1 | not measured | not measured |
| random | fixed_never_input_targets | DiTing | 539 | 1310 | 114/539; unknown=3 | not measured | not measured |
| random | fixed_never_input_targets | EQT | 539 | 1310 | 81/539; unknown=5 | not measured | not measured |
| random | fixed_never_input_targets | PhaseNet | 539 | 1310 | 106/539; unknown=5 | not measured | not measured |
| random | fixed_never_input_targets | TEAM | 539 | 1310 | 80/539; unknown=3 | not measured | not measured |

warm数值时延：每模型/每scope六个元数据预选事件，normal/random × 1/10/20秒共36条件，每条件2 warmup、3 measured重复（108记录）。硬件记录同名不证明物理运行条件严格相同。单位秒。

| family | scope | repeats | events | p50_seconds | p95_seconds | maximum_seconds | cold_load_seconds | device_names |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DiTing | station_only | 108 | 6 | 0.413123 | 3.432414 | 4.449764 | 101.278916 | Device 66a1 |
| DiTing | whole_map | 108 | 6 | 0.584978 | 4.500733 | 5.310486 | 101.278916 | Device 66a1 |
| EQT | station_only | 108 | 6 | 0.116195 | 0.714546 | 0.774206 | 43.492946 | Device 66a1 |
| EQT | whole_map | 108 | 6 | 0.292316 | 1.676303 | 1.825012 | 43.492946 | Device 66a1 |
| PhaseNet | station_only | 108 | 6 | 0.076651 | 0.388249 | 0.423699 | 42.583515 | Device 66a1 |
| PhaseNet | whole_map | 108 | 6 | 0.234989 | 1.278129 | 1.379857 | 42.583515 | Device 66a1 |
| TEAM | station_only | 108 | 6 | 0.081696 | 0.425355 | 0.454760 | 22.479288 | Device 66a1 |
| TEAM | whole_map | 108 | 6 | 0.271248 | 1.495653 | 1.594857 | 22.479288 | Device 66a1 |

下表是渲染交付shadow的一个预定网格例子：normal、PGA阈值−1.2 log10(m/s^2)、p_alarm=0.5、tau=3秒。所有阈值/概率/提前量组合仍保留。pooled为目标加权；paired原表为事件等权，二者不能混作同一估计量。

| family | panel | events | events_with_affected_targets | affected_targets | timely | failures | unknown | pooled_failure_lower | pooled_failure_upper | complete_negative_targets | false_positives | pooled_false_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DiTing | fixed_all_targets | 1310 | 1193 | 11111 | 5083 | 5945 | 83 | 0.535055 | 0.542525 | 11152 | 4891 | 0.438576 |
| DiTing | fixed_never_input_targets | 312 | 214 | 2336 | 1570 | 717 | 49 | 0.306935 | 0.327911 | 2507 | 1210 | 0.482649 |
| EQT | fixed_all_targets | 1310 | 1193 | 11111 | 5177 | 5870 | 64 | 0.528305 | 0.534065 | 11152 | 5157 | 0.462428 |
| EQT | fixed_never_input_targets | 312 | 214 | 2336 | 1611 | 690 | 35 | 0.295377 | 0.310360 | 2507 | 1465 | 0.584364 |
| PhaseNet | fixed_all_targets | 1310 | 1193 | 11111 | 5310 | 5721 | 80 | 0.514895 | 0.522095 | 11152 | 4781 | 0.428712 |
| PhaseNet | fixed_never_input_targets | 312 | 214 | 2336 | 1578 | 708 | 50 | 0.303082 | 0.324486 | 2507 | 1225 | 0.488632 |
| TEAM | fixed_all_targets | 1310 | 1193 | 11111 | 4901 | 6100 | 110 | 0.549005 | 0.558906 | 11152 | 5286 | 0.473996 |
| TEAM | fixed_never_input_targets | 312 | 214 | 2336 | 1510 | 773 | 53 | 0.330908 | 0.353596 | 2507 | 1312 | 0.523335 |

来源：source_tables/analyze/0/legacy_with_physical_covariates/；derived/ttum_support_recomputed.csv；derived/latency_recomputed.csv；derived/shadow_rendered_summary.csv。复算和估计量定义见build_review.py与DATA_DICTIONARY.md。
