# FE01-EVAL1本地真实分析：供ChatGPT审阅

本轮已完成九组原始validation CSV再分析、来源/人口核验、5000次事件簇bootstrap、固定目标与空间形状诊断，并提供独立手动sbatch脚本。**生产权重前向、两个旧系统同人口成绩、train-only场地参照及3例回放未执行。** 所有数字来自真实已有CSV；合成测试没有充当Japan结果。

仓库`rular099/team_pytorch`，分支`exp/fe01-feature-extractor-site-effects`，base `b2756825190f93034ece436fab778e8a69adb41c`。原训练源码commit为`73ae9dec50713805c878b5e4efeb8264c01d0c76`，115文件SHA仍为`3e537ff6693971d92b139e517f370b998237bd8b29989194e9ddcc06cc9dcfea`。新工具在旧哈希范围之外；Git提交前运行的工作区状态和各文件SHA分别保存在`provenance.json`、`publication_provenance.json`。后者记录轻量导出和新增早期概率CI，不能把它当成一次新前向。

## 数据与比较对象

validation共同cohort1310事件。每run194265目标记录，其中noninput139440、observed_input54825、triggered_noninput43850、untriggered_noninput95590。时间参考first_p_pick，固定1/3/5/10/20秒×normal/random，单位log10(m/s²)。每cell先对目标算MAE，然后十cell等权；三个固定模型loss均值与seed样本标准差分列，**没有平均预测组成ensemble**。每run的epoch为：TEAM42/43/44=11/11/11；PhaseNet=12/10/12；EQT=10/10/10。epoch来自冻结CSV/曲线，内部checkpoint仍待核验。

九组原字节SHA在`reused_result_manifest.csv`；人口/标签/时钟严格匹配原锁。`checkpoint_inventory.json`逐项列出本地可见身份：九组selected/init/last权重均未在上传目录中找到；RT55实际可见候选内部epoch20，不能替代要求的32；RT61上传包没有checkpoint。11个候选本地均BLOCKED；CSV_IDENTITY_PASS不等于生产forward PASS。

## 已核验结果

| 家族 | 主noninput MAE：mean ± seed SD | 早期1/3/5秒MAE | 未触发MAE | 实际单输入站的noninput MAE |
|---|---:|---:|---:|---:|
| PhaseNet | 0.225818 ± 0.000125 | 0.232506 | 0.251068 | 0.280194 |
| EQT | 0.235296 ± 0.001310 | 0.240830 | 0.261969 | 0.288085 |
| TEAM | 0.241817 ± 0.003011 | 0.248279 | 0.268943 | 0.294674 |

表中都是各家族三个固定seed的cell等权平均，MAE单位dex；未触发和单输入站分母是实际子群，不能照抄139440。逐run/time/geometry/role的counts、MAE/RMSE/R²/slope/bias/NLL/CRPS/Brier/sigma/coverage/width在`metrics_by_time_geometry_role.csv`。正确pooled RMSE与mean(cell RMSE)在`reused_result_manifest.csv`分列。主分数逐项与上一轮报告核对至1e−12，没有改变历史主表。

**反例：1秒点预测优势不能替代概率评分。** normal/random两cell等权、三seed平均：PhaseNet MAE0.263840、NLL0.324149、95%coverage0.910757；EQT MAE0.266483、NLL0.296632、coverage0.926425；TEAM MAE0.271501、NLL0.323185、coverage0.926293。1秒noninput分母为normal19160目标/1292事件、random19635目标/1310事件。两种分母都包含同一事件多个站点，不是独立样本。

PhaseNet−EQT的1秒NLL配对差及事件CI：seed42为0.036443[−0.000699,0.084110]；seed43为0.006715[−0.023013,0.045347]；seed44为0.039391[0.007570,0.076089]。前两区间跨0，因此不能把平均反例写成三seed均统计显著。详见`paired_early_probability_event_ci.csv`，同时包含早期MAE/NLL/CRPS/Brier。没有调整sigma或阈值。

**空间诊断：整体偏移与形状均需单独看。** noninput、每field≥5目标，先field等权、再time/geometry及固定seed等权：

| 家族 | level MSE（dex²） | shape MSE（dex²） | 近等距离站对差值MAE（dex） |
|---|---:|---:|---:|
| PhaseNet | 0.023428 | 0.053053 | 0.227051 |
| EQT | 0.025820 | 0.056208 | 0.233187 |
| TEAM | 0.031162 | 0.055367 | 0.232242 |

这支持PhaseNet在现有固定时刻的空间形状误差上也更低。三个seed中，PhaseNet相对TEAM/EQT的shape MSE配对CI都在改善方向；逐field及近距离5km/符号可判定分母见`field_summary.csv`和`paired_field_event_bootstrap.csv`。仍不能据此证明可重复场地效应或地质成因，缺共同train-only参照与跨事件台站恢复。

**固定目标揭示人口差别。** normal全目标22263站-event组合在五时刻均存在，但角色随输入增加迁移；全过程保持noninput的面板只有4843目标/312事件。random原目标由19635降至14822；五时刻交集11256目标/1015事件，排除了295事件。排除数逐时刻在`fixed_panel_population.csv`，角色迁移在`role_migrations.csv`。PhaseNet normal始终noninput面板MAE由1秒0.276864降至20秒0.185852；random面板5秒0.238753、10秒0.240703，仍有轻微反弹。固定目标没有把所有变化都消除，不能据此归因于encoder。固定输入S0反事实尚未执行。

## 图怎么读

- `figures/*_normal_density_pit.png`及`*_random_density_pit.png`：展示每家族seed42的五时刻noninput预测。上排共同色标的truth/prediction密度、对角线和实际N；下排PIT与理想均匀线。尾部堆积提示概率分布问题；不是地质图。源为`density_bins.csv.gz`、`pit_bins.csv.gz`，其余seed的分箱也保存。
- `figures/*_calibration_counts.png`：可靠性、95%coverage-width及目标数量。可靠性展示明确按原计数pool两geometry，其他表仍保留geometry；均值±sigma覆盖率在主CSV另列，不冒充分位数coverage。源为`reliability_bins.csv.gz`和metrics表。
- `figures/level_shape_equal_distance.png`：level、shape和近等距离站对差值误差的时间变化。它分解空间误差，不判定地质成因；参照控制和台站重复残差缺失。源为`field_summary.csv`、逐field加总。
- `figures/input_and_untriggered_counts.png`：左边是去重后的decision输入站数分层；右边是未触发目标记录数量。两个纵轴的统计单位不同。源为`input_decision_counts.csv`和metrics表。

每图另附`figures/CAPTIONS.md`。地图、场地恢复图和动画没有生成，不能把未来输出路径当作已有图。网格任务只在3个预选事件做1–20秒区域演示。

## 来源、分母与缺项

`event_fields_primary.csv.gz`保留175797个noninput/untriggered field尝试及各自资格、目标数、MSE分解、范围/站对指标；`event_losses_primary.csv.gz`保留相同两种主分析群体的逐event/time/geometry误差和概率loss加总、目标数。其他角色的全量逐field/逐event表留在本地完整结果包，各角色aggregate与CI仍在本目录。`publication_provenance.json`给出全表SHA与轻量导出的范围；不能将子群文件误当全集。原始194265行×9的预测留原目录，不重复上传Git。

5000次bootstrap，RNG20261001，以(dataset_id,event_id)整事件簇移动；点loss按cell内目标权重，field按cell内合格事件field等权。家族mean-loss CI仅对三个固定模型条件化，不包括训练随机性/选epoch不确定性；seed SD另列。子群/field资格改变有效事件数，各CI记录实际分母、空cell replicate和排除数，不称1310个独立台站。

共同随机时刻原SHA九组相同；3930draw/3926快照/4重复，映射见`random_draw_mapping.csv`。预选案例不依预测：`japan_2000.hdf5/20001009022000`（M3.6）、`japan_2000.hdf5/20001004200000`（M4.0）、`japan_2006.hdf5/20060909193600`（M4.9）。科学合同、旧桥接和提交步骤见[协议](../../docs/ai/FE01_EVAL1_PROTOCOL.md)及[超算手册](../../docs/ai/FE01_EVAL1_HPC_RUNBOOK.md)。

`gates.csv`与`summary.json`区分已通过的CSV人口/历史主端点核验、NOT_RUN的checkpoint前向/旧系统比较/随机长时刻回放、INCOMPLETE的train-only/site证据。HPC状态为NOT_SUBMITTED_THIS_TASK。实际50项聚焦测试通过，另有近期runner/请求/调度器11项复验通过；只有已有timm弃用警告。原115源码锁未改变。

限制仍包括validation已用于选epoch、系统的预训练/原生长度/曝光/前处理/adapter混杂、C1右对齐prefix对旧权重的分布变化、没有真正训练期空间隔离，以及上游离线滤波和触发/通信延迟尚未认证。请依据证据决定后续研究；本轮不自动增加训练。
