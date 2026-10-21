# FE01-A01 实际超算结果：独立分析入口（2026-10-21）

仓库 `rular099/team_pytorch`；分支 `exp/fe01-a01-absolute-amplitude-ablation`。
整理基线 `59f3eb30d6c955ffe2904aee40693af71dc0a440`，这是结果整理前的代码基线；
请以用户提供的本次 result commit 为准回报实际读取版本。

本目录提供事实表、来源和复算记录，供 ChatGPT 独立分析。
Codex 的独立判断另外封存于本地，待 ChatGPT 形成结论后由用户对照。
阅读任务见 [CHATGPT_ANALYSIS_PROMPT.md](CHATGPT_ANALYSIS_PROMPT.md)。

## 先读什么

1. [evidence_checks.json](evidence_checks.json)：归档、导出人口、指标及区间复算。
2. [paired_mae_summary.csv](paired_mae_summary.csv)、[model_summary.csv](model_summary.csv)：主指标和全目标/非输入目标分组。
3. [validation_by_time_geometry_role.csv](validation_by_time_geometry_role.csv)：逐时刻/几何/角色，共960个模型条件cell，含空cell。
4. [paired_event_bootstrap.csv](paired_event_bootstrap.csv)、[bootstrap_support_audit.csv](bootstrap_support_audit.csv)：672个原配对端点及支持情况。
5. [paired_field_levels.csv](paired_field_levels.csv)、[paired_field_by_cell.csv](paired_field_by_cell.csv)：同一有效field人口上的ON/OFF绝对空间场量。
6. [diagnostic_query_stages.csv](diagnostic_query_stages.csv)、[diagnostic_nested_matched.csv](diagnostic_nested_matched.csv)、[diagnostic_invariance_checks.csv](diagnostic_invariance_checks.csv)：机制与边界诊断，必须保留 evidence 字段。
7. 需复算时读 [event_cells_noninput.csv.gz](event_cells_noninput.csv.gz)；需原配置和身份时读 `source_metadata/`。

若需核对定义，`method_sources/`提供六份与原HPC源码pin一致的指标、配对、空间场、人口锁、
诊断状态入口和幅值模式源码快照，SHA见其中的SOURCE_SHA256.json；用于阅读定义。

## 评价身份与端点

- 全部为 **validation、seed42**。共有5组新训模型、3组复用旧ON模型，共8个模型条件。
- 每个模型条件提供预定验证规则选出的 selected 和固定 epoch12 两套导出，共16份；二者可为同一模型状态，不能当作独立重复实验。
- TEAM/PhaseNet/EQT ON 复用旧训练、冻结选轮分别11/12/10；三组OFF为匹配的新训练。
  DiTing ON/OFF为新训匹配对，区别于历史RT55/RT61完整系统。
- 五组新训练均12轮、2544 updates、world16、global batch128、microbatch8。
- 每份导出同一批1310个事件、194265条目标请求；其中非输入目标139440条。
  目标角色数量：observed_input=54825、triggered_noninput=43850、untriggered_noninput=95590。
- 主端点是非输入目标在 normal/random × 1/3/5/10/20秒共10个cell内分别计算MAE，再对10个cell等权平均。
  MAE和区间单位均为 `log10(m/s²)`（dex）；目标数是跨决策请求数，不是独立台站或独立事件数。
- `pooled_*`按该组所有请求合并计算，与主端点权重不同。
  `triggered_noninput` 的 normal/1秒为空，只有9个支持cell；N/A不填零。
- K1_noninput共有660事件/28535请求，Kmulti_noninput共有1219事件/110905请求；事件集可以重叠。
- 区间为固定模型的事件cluster bootstrap，所有台站/时间/几何行随同一事件抽取。
  主端点2000次、seed20261007。区间条件于本次训练和选轮，不涵盖训练seed变化或验证集选轮过程。

## selected 原始主端点数值

1310事件、139440非输入目标请求，10个共同cell等权。差值定义为OFF减ON。
下表只列数值；解释与研究判断请由分析者独立形成。

| 家族 | ON/OFF epoch | ON MAE | OFF MAE | OFF−ON | 事件配对95%区间 | 相对ON变化 |
| --- | --- | --- | --- | --- | --- | --- |
| TEAM | 11/12 | 0.245159 | 0.286003 | +0.040844 | [0.030898, 0.052383] | 16.66% |
| PhaseNet | 12/11 | 0.225697 | 0.246947 | +0.021250 | [0.016876, 0.025910] | 9.42% |
| EQT | 10/11 | 0.236407 | 0.268277 | +0.031870 | [0.024962, 0.039194] | 13.48% |
| DiTing | 12/11 | 0.222859 | 0.248377 | +0.025517 | [0.020964, 0.030614] | 11.45% |


## 固定 epoch12 数值

| 家族 | ON/OFF epoch | ON MAE | OFF MAE | OFF−ON | 事件配对95%区间 | 相对ON变化 |
| --- | --- | --- | --- | --- | --- | --- |
| TEAM | 12/12 | 0.245331 | 0.286003 | +0.040672 | [0.030544, 0.052382] | 16.58% |
| PhaseNet | 12/12 | 0.225697 | 0.247157 | +0.021460 | [0.017243, 0.026100] | 9.51% |
| EQT | 12/12 | 0.236871 | 0.268639 | +0.031767 | [0.024883, 0.039134] | 13.41% |
| DiTing | 12/12 | 0.222859 | 0.248573 | +0.025714 | [0.021278, 0.030649] | 11.54% |


来源：[paired_mae_summary.csv](paired_mae_summary.csv)。

## selected 概率端点

仍为非输入目标、10个cell等权。覆盖率来自真实混合分布分位区间，名义95%；
NLL由原始logits精确计算。Brier阈值为log10(PGA)=-1.2，PGA单位m/s²。
宽度单位dex，NLL是该坐标上的负对数密度，可为负。

| 家族 | 模式 | epoch | NLL | CRPS | Brier | 95%覆盖率 | 95%区间宽度 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TEAM | ON | 11 | 0.217655 | 0.174864 | 0.188617 | 91.17% | 1.051705 |
| TEAM | OFF | 12 | 0.410115 | 0.208984 | 0.225917 | 90.82% | 1.222125 |
| PhaseNet | ON | 12 | 0.150267 | 0.162109 | 0.173825 | 91.30% | 0.976773 |
| PhaseNet | OFF | 11 | 0.217039 | 0.176895 | 0.189234 | 91.64% | 1.063337 |
| EQT | ON | 10 | 0.167722 | 0.168356 | 0.180948 | 92.23% | 1.051932 |
| EQT | OFF | 11 | 0.292193 | 0.191923 | 0.207324 | 92.87% | 1.199336 |
| DiTing | ON | 12 | 0.115770 | 0.159220 | 0.169824 | 91.49% | 0.975356 |
| DiTing | OFF | 11 | 0.228030 | 0.177574 | 0.189146 | 91.35% | 1.081741 |


来源：[model_summary.csv](model_summary.csv)及[数据字典](DATA_DICTIONARY.md)。

## 诊断证据范围

`training/index0/1/2/diagnostics` 的 evidence 为 `real_ON_checkpoint_inference_intervention`：
旧训练ON权重、推理时OFF干预，不能称为新训OFF checkpoint的诊断。
`training/index3/4/diagnostics` 为 `matched_initial_checkpoint_NOT_TRAINED`：DiTing匹配初态，
不能作为训练后DiTing信息流的结论。每组78个自然probe，cap_per_stratum=2；
嵌套对照另按实际支持记录，不增加独立事件证据。

`diagnostic_nested_matched.csv` 固定完整支持K=1/3/5/8的同一decision cohort；每组18个decision。
普通 `diagnostic_nested_stages.csv` 的不同K支持cohort数量不同，比较其组中位数时须注意这一点。
跨query差异的汇总保留>=2 query的专门列；一个query的零离散度没有判别意义。
原始诊断图沿用各自evidence标识，位于 `source_metadata/training/index*/diagnostics/figures/`。

## 支持与精度记录

- 373个tar成员，其中372条归档manifest记录，全部SHA验证通过。
- 16份生产人口/标签/输入/时钟哈希一致，16份导出条件标签检查通过。
- 960个score cell从逐目标数据重算，与保存表在所核验指标上的最大绝对差为0。
- 1040条确定性抽样MDN记录重算，最大绝对差约8.9e-16；该抽样不替代全量模型重新推理。
- 8个主端点95%区间按逐事件表重新抽样，均与原表一致；240个空间配对点估计从绝对场量复核通过。
- 原始eval seal共126个文件引用，108个字节相同，18个JSON经封装的路径脱敏/再序列化后字节不同。
  脱敏JSON按归档manifest验证；不要把它们与原私有seal的字节差异当作预测文件损坏。
- 本地逐文件验证46个认证核心文件及50个原训练文件。原115文件清单指纹相符；65个dtbench依赖源文件本地未提供，只核对归档清单指纹。
- 模型权重、私有原plan、波形和sample journals未随包提供；checkpoint epoch/SHA来自真实超算导出的身份记录，权重字节没有在本地重新读取。
- sacct状态和原slurm out/err未被本包封装；`jobs.tsv`记录作业号，不能单独当退出码证据。
  本批成功证据来自完整结果seal、manifest、实际预测、比较与归档，而非旧失败日志。

重要支持细节见 [bootstrap_support_audit.csv](bootstrap_support_audit.csv)：
triggered_noninput缺1个cell；K1的空间field端点只有410事件/1413个eligible fields，
2000次中1253次有效（37.35%因缺cell被排除），等距离端点1289次有效。
K1主MAE端点为1999次有效。请在解释相应区间时报告这些差异。

## 来源与复算

输入 `outputs/review.tar.gz`，SHA256：
`7a6ecf2260f1fd8d7f1d188c48d6b0f19bbdc433bb861879fb51ac549f209986`。
原HPC核心source commit `3ecf060a3334d1c7b48662cf4395a3bab07723ff`；
认证核心内容SHA `3e1e4398b01d724f793f88b19be2a2c03ca67a71e881d876c173a8fbc5b8177f`。
索引0的评价修复来源对应f29ecdcb，索引4及pack对应59f3eb30；执行SHA和revision逐项保留。
DiTing续训的RNG兼容循环另有执行身份，见 `source_metadata/training_execution/`，
不能把核心SHA理解为完整训练执行源码SHA。

CSV往返精度修复的实际证据：4号两套ON读取检查都记录默认解析有174个truth值变化，
round_trip已通过原严格人口门；两份JSON保存在 `source_metadata/results/index4/paired_input_checks/`。
本地用round_trip读取，未替换标签或放宽生产人口门。

可在有numpy/pandas/scipy的Python环境用新目录复算：

```bash
python scripts/fe01_a01_ops/review_archive.py --source /path/review.tar.gz --destination /new/cache/raw
python scripts/fe01_a01_ops/summarize_review.py --source /new/cache/raw --output /new/summary
```

原始归档和约1.1GB逐目标导出保留本地 `artifacts/fe01_a01/review_20261021/raw`，不进入Git。
本次仅阅读与整理已保存的validation证据，没有加载模型、波形或访问held-out test，也未提交新超算任务。
