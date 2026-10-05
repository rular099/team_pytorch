# FE01-EVAL1：已有权重的 validation 复评合同

本轮不训练、不续训、不重新选择 epoch、不开 test、不校准分布、不自动提交作业。新增代码位于 `fe01_review/`；原 `fe01/`、legacy模型、115文件源码锁、旧配置、权重及结果均不修改。训练源码 SHA 必须仍为 `3e537ff6693971d92b139e517f370b998237bd8b29989194e9ddcc06cc9dcfea`。新评价源码另记模块 SHA、Git身份及工作区是否有未提交新工具；两者不能混用。

| 系统 | seed42 | seed43 | seed44 | 长时刻能力 |
|---|---:|---:|---:|---|
| TEAM | 11 | 11 | 11 | 40、90秒 |
| PhaseNet | 12 | 10 | 12 | 40、90秒均N/A |
| EQT | 10 | 10 | 10 | 40秒；90秒N/A |
| RT55完整系统 | epoch32，只有一份参考 | — | — | 40、90秒 |
| RT61增强完整系统 | epoch8，只有一份参考 | — | — | 40、90秒 |

九组旧CSV的 SHA、模型/seed、共同人口和历史分数已经能够本地核验；CSV本身没有内部checkpoint epoch字段，真实权重身份仍需单独核验。本地RT55候选的内部epoch确实是20，因此BLOCKED，不用它替代32。RT61 zip中没有checkpoint。缺项逐模型隔离。

## 请求与主指标

共同validation为1310事件。固定时刻1/3/5/10/20秒、normal/random十cell，每run194265行，其中noninput139440、observed_input54825、triggered_noninput43850、untriggered_noninput95590。人口SHA为`e96284780cb5f72b952c1697d4d264d307e7139a153201fa222566015aec7f69`，标签/输入/时钟SHA为`f204deca06725af5513895416d5b0691c75c12221127854c32d1d497855155bb`。记录数不是独立样本数。

```text
Primary MAE = (1/10) × Σ_cell [Σ_target |prediction − truth| / N_cell]
```

原始LaTeX：
```latex
\mathrm{MAE}_{\mathrm{primary}}=\frac1{10}\sum_{c=1}^{10}\frac1{N_c}\sum_{i\in c}|\hat y_i-y_i|.
```

单位是log10(m/s²)，预测是该坐标下的mixture mean。总RMSE先汇总平方误差再开方；mean(cell RMSE)另列。1秒、早期1/3/5秒、实际input_count=1、未触发子群均是独立次端点，不能替换主端点。概率指标使用完整混合分布；Brier阈值固定−1.2，分位数coverage与mean±sigma coverage分列。

共同请求先于预测锁定；禁止每个模型自由采样。固定请求来自原TEAM seed42 epoch11 CSV元数据，不含预测。CPU阶段从经原lock验证的HDF5补锁真实查询高程、storage起点和长度，原CSV没有的高程不能补0。新增随机、长时刻、案例请求保存实际ID顺序、三维坐标、绝对/样本时钟、输入及角色。每次前向导出物理前缀、模型输入指纹和preprocessor ID。模型自身的前处理可以不同，但物理可见前缀必须相同。人口失败输出BLOCKED，不静默取交集。

原随机manifest九组字节相同：3930draw、3926unique snapshots、4duplicate draws。保留原文件，计算快照后映射回draw。次指标先目标平均、再按原draw在事件内平均、再事件等权、geometry等权；重复draw不当独立事件。若某子群的原draw没有合格目标，记录INCOMPLETE，不补0或悄悄重定义权重。新随机时刻仍来自同一validation事件，不是test。

## 旧系统桥接与验证门

RT55/RT61通过原`eval_checkpoint.build_model_and_load`构造原类、严格加载原非encoder state，并只允许原`waveform_model.0.`缺失前缀。其原external DiTing encoder必须存在于checkpoint记录的源路径，CPU阶段另钉实际文件SHA；不能用其他encoder或随机初始化替代。RT61还核验epoch8父模型、任务ID和不可变readout内容SHA，通过原恢复函数恢复payload。

L0直接原模型与包装器对**同一六个tensor**比较，证明包装没有改读出；不是旧历史loader整条重现。C1先严格截原始prefix，再仅用有效前缀做loader去均值；随后保留旧FullModel自身masked per-channel std、原自然对数尺度定义，以及原PGA反标准化。禁止替换为FE01的joint-peak/log10尺度。结果命名`rt55_strict_prefix_replay`和`rt61_strict_prefix_replay`，是新因果输入协议下的完整系统复评，不填历史标量分数。源配置、单位、权重SHA分别记录。

C1明确采用共同`build_window`的右对齐native prefix：科学历史保留完整，左侧补至旧系统10000点；不会为短模型截去应见历史。它不声称复现历史loader的时间布局，旧权重对此输入分布的适应性属于本轮复评限制。L0同tensor检查不能消除这个限制。

六个forward输入只含输入站波形、坐标、有效性、query坐标/有效性和sample mask；震级、震源位置和最终PGA只用于评分或事后oracle，不进入网络。原relative-coordinate行为保留，每帧记录输入物理centroid和模型相对坐标中心。

共同probe依据元数据哈希，normal/random×1/20秒各8个decision，共32；需要时补实际单/多输入，总数不超过40。每个decision全部query验证与原CSV的mean、weights、mu、sigma；小子集另验单点/分块/逆序/追加无关query、缓存开关、不同T拒绝缓存、未来NaN/大脉冲下前处理/encoder features/完整MDN不变、全eval/frozen且buffer不变。预先固定atol=rtol=1e−5，不自动放宽。小样本失败隔离该模型，不全量重跑九组固定时刻。上游离线滤波因果性仍uncertified。

## 空间与统计定义

```text
error_i   = prediction_i − truth_i
field MSE = mean(error)² + mean[(error_i − mean(error))²]
```

原始LaTeX：
```latex
\frac1n\sum_i e_i^2=\bar e^2+\frac1n\sum_i(e_i-\bar e)^2.
```

每event/T/geometry/role field至少5目标；level与shape用同一集合。未触发群体在自身群体重中心化。报告centered MAE/RMSE、所有站对差值MAE、P95−P05范围误差/ratio；truth range<.05 dex的ratio缺失并保留分母。近等距离站对用epicentral半径差≤5km，符号分母要求truth差绝对值>1e−8；先field内平均，不能把共享站点的站对当独立样本。

共同train-only参考复用`fe01.spatial.fit_reference`：9084训练事件，146299最终event-station标签各一次；intercept、M−4、(M−4)²、log10(max(hypocentral R,1))、R/100、depth/100，ridge=.1，site shrinkage10，20迭代。no-site与train-site是所有系统的共同事后oracle。缺训练HDF5则reference/site证据INCOMPLETE，不能把普通PGA误差改名为场地误差。

台站残差在每T/geometry/角色自身event field中中心化，再跨事件平均；主台站群≥10唯一事件，稀疏台站保留。按事件bootstrap后重新计算台站均值，报告台站等权MAE、corr、slope、sign和CI。方位/强度表保留T与geometry。原selected checkpoint提交的journal记录实际input/query训练曝光；缺日志身份写UNKNOWN。训练标签cohort、实际模型曝光、评价缺input波形是三种不同身份，不声称spatial holdout。

bootstrap5000次，seed20261001，按(dataset_id,event_id)整事件移动全部站/时间/协议。FE01三seed样本标准差与事件CI分列。家族mean是固定模型逐目标loss平均，不是预测平均ensemble；其CI不含训练随机性或epoch选择不确定性。同一个RT参考逐个与九组配对，不复制为三个seed模型。配对空间结果仅为预设/探索性诊断，不作为多重比较校正后的地质结论。

固定面板对各geometry保留所有五时刻交集、实际角色和排除数；normal始终noninput面板仅312事件，random交集1015事件。两种面板都只反映自然输入变化。固定S0反事实只在3个预选案例新增前向，query固定且排除S0；每T重新编码，只在同T缓存。两条路径都noninput的子集另评分，角色迁移和坐标中心随帧保存。

## 有限回放与证据边界

案例在共同cohort震级.25/.50/.90分位附近按元数据预选，平局按dataset/event，不看误差。3个案例锁定为M3.6、4.0、4.9；1–20秒逐秒、normal/random、自然输入/固定S0，seed42三模型和可用两个参考。网格10km、站点外扩50km、统一边界和色标；无DEM时网格高程0只用于演示，网格没有ground truth。真实站点评分与网格展示分开。图示1/5/20秒，其余逐秒数值保留；没有生成动画时明确不称已生成动画。

整帧同步计时包括读prefix、预处理、encoder、全部station/grid query、分布评分、帧文件写入；p50/p95/max和>1秒比例原样保存。冷启动、独立snapshot复验、事后oracle分析和render另列。100km输入距离遮罩是披露的展示支持约定，域外预测仍保留，不据此筛掉定量误差。真实触发/通信延迟未测，称理想触发因果回放。

本地当前证据只能支持九组旧CSV的validation诊断。RT系统同人口成绩、原checkpoint的生产forward、train-only参考、随机/长时刻及3例回放都需手动HPC补证；没有显著提升是有效结果，不能据此自动改训练。
