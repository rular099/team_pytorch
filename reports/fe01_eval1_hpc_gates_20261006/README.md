# FE01-EVAL1：实际超算身份、前向核验与训练集参照

本批下载结果的结论是：**11组固定checkpoint身份通过；8组小样本前向核验通过；TEAM scratch的3个seed被评价入口误用的预训练清单检查拦住；共同train-only参照已生成。** 当前还没有本轮RT55/RT61同人口预测成绩、随机/长时刻评价或回放。这份报告不能作为新的模型性能排行榜。

仓库 `rular099/team_pytorch`，分支 `exp/fe01-feature-extractor-site-effects`；本轮整理基点 `221f8b24483148d630e0eb85a179d763f8c580ac`。超算报告的评价source为 `4c9c1757f726e571022a398cb5a217ca6c325cbc`，评价模块SHA为 `e65a7dd017828da10adcf5c45c4aa32d1d702833910972556d4eda5b6a640564`。二者分别表示部署source身份和实际模块内容；不能把本次文档提交称为超算执行的代码版本。

## 实际结果与前置门

| 模型 | seed | 固定内部epoch | checkpoint身份 | 前向核验 |
|---|---|---|---|---|
| TEAM scratch | 42 / 43 / 44 | 11 / 11 / 11 | 三组均IDENTITY_PASS | 三组均BLOCKED，尚未执行前向 |
| PhaseNet frozen | 42 / 43 / 44 | 12 / 10 / 12 | 三组均IDENTITY_PASS | 三组均PASS |
| EQT frozen | 42 / 43 / 44 | 10 / 10 / 10 | 三组均IDENTITY_PASS | 三组均PASS |
| RT55完整系统 | 原固定权重 | 32 | IDENTITY_PASS | PASS |
| RT61增强系统 | 原固定权重 | 8 | IDENTITY_PASS | PASS |

内部epoch、checkpoint字节SHA、原config/lock身份来自超算CPU inventory，不根据文件名推断。本地下载不含权重字节，因此本次没有独立重载checkpoint。RT55/RT61的外部encoder在超算核验中按固定SHA检查；保留原加载、归一化与RT61父readout，不把旧系统视为新训同公共下游的DiTing encoder消融。完整逐模型数值见 [model_gates.csv](model_gates.csv)，原始身份记录的脱敏版本在 [checkpoint_inventory.json](evidence/identity/checkpoint_inventory.json)。

八组PASS均使用同一冻结验证计划：validation的32个事件/32个decision、544条目标记录，时间为1秒和20秒，normal/random各16个decision；这些是小样本核验人口，并非固定时刻正式评价的1310事件/139440条noninput记录。完整prediction和MDN的CSV再现只适用于六组PhaseNet/EQT；RT55/RT61没有本轮旧CSV逐行差值。

六组PhaseNet/EQT与其选定epoch的原CSV相比，点预测最大绝对差为 `2.0758942215204357e-7`，单位 `log10(m/s²)`，发生在EQT seed44 epoch10。每组各544目标，绝对和相对容差均为 `1e-5`。MDN weights/mu/sigma差值逐组列于CSV；weights无量纲，mu/sigma沿用反标准化PGA坐标。差值验证了这些探针上的复现，不能代替整体MAE、校准或新数据泛化成绩。

八组的时钟/标签检查覆盖全部32个decision；额外原模型/包装器同输入等价、query倒序/单个/追加、同cutoff缓存与不同cutoff拒绝覆盖前4个decision；future NaN/large pulse覆盖前2个decision，检查预处理、encoder特征和MDN。模型buffers保持不变。**L0是同输入tensor上的包装器等价，不是历史loader重建；上游离线滤波的在线因果性仍未认证。** 逐decision证据在 `evidence/verification/<run_id>/verification.json`。

## TEAM为何失败：已核验事实与修复建议

真实错误为 `Original pretrained manifest byte SHA changed`，见三组 `failure.json` 和 [seed42错误日志](evidence/scheduler_logs/fe01e1_verify_29354874_0.err)。错误发生于 [fe01_review/runners.py](../../fe01_review/runners.py:124) 的无条件检查，在 `build_model` 和模型前向之前。

从此前真实训练目录读取的三份 `protocol.lock.json` 均满足：`model_family=team_original_scratch`，`pretrained_manifest_sha256=null`。这三份原文件的字节SHA与本次超算inventory记录的父lock分别完全相同：

| seed | 原lock SHA256 |
|---|---|
| 42 | `15143a7c0cfc4dc0aacfd35ad489805f09476c75fdc6bbaa4168e97f9f394e54` |
| 43 | `6dbdcfb652aa3e6779b82c29e86443ebd3853963b39da5d52e0f0b079834cfcd` |
| 44 | `a0f6458677076243ec1fbb319ccd20ee4bd69fbdf5471d9dbcb10e71f8cece88` |

原训练审计 [fe01/engine.py](../../fe01/engine.py:359) 只为pretrained家族保存清单SHA；TEAM模型构造 [fe01/model.py](../../fe01/model.py:130) 直接使用OriginalTEAM。评价入口把实际清单的SHA字符串与 `null` 比较，必然失败。因此，日志没有证明清单字节发生变化、TEAM权重损坏或TEAM前向数值不一致。对应证据字段另存 `evidence/training_locks/`。

**建议，尚未实施：** 仅让TEAM scratch按其真实模型家族遵循“无需预训练清单”的原合同；pretrained家族仍须非空有效SHA且严格匹配。保留原训练源码、lock、权重和全部其它身份检查。修复评价模块后应使用新source pin和新run ID重新核验；不能沿用旧模块SHA下的PASS或把本批TEAM结果改写成PASS。本轮只整理和同步证据，没有修改runner。

## 训练集参照与台站曝光

共同参照在train上拟合一次：9084个事件、960个台站、146299条最终PGA标签；标签按dataset/event/station去重，全部 `split=train`，单位 `log10(m/s²)`。本地逐行复核人数、去重和原归一化一致，均值 `-1.122462063306821`、人口标准差 `0.43132895149034495`。完整参照见 [train_only_reference.json](evidence/reference/train_only_reference.json)。

参照使用回顾性的catalog震级、震源位置和深度，包含无site回归及固定惩罚/收缩的event/site分解；penalty=0.1、shrinkage=10、20次迭代。它为所有系统提供同一残差锚点，**不是可在线获得catalog元数据的实时预测竞争模型**。本批尚未输出validation的台站残差恢复、场地评分或空间图，因此不能从已拟合的site effects宣称任一预测系统学会场地效应。

九组FE01均提供与所选checkpoint绑定的训练曝光身份：记录的committed journal数量分别为TEAM 176/176/176，PhaseNet 192/160/192，EQT 160/160/160；每组9084个训练事件、960个台站均曾作输入和query。本地只核对记录与checkpoint inventory的SHA/count及导出CSV，原jsonl未包含在本次下载中，未独立重算其字节SHA。RT55/RT61曝光状态仍为UNKNOWN，不能从FE01的train cohort推断。

本轮validation固定台站元数据含22263个event/station组合、1310个事件、915个不同台站；915台站均在train参照及九组FE01实际曝光表中出现，作为input/query的未见台站数均为0。因此这批结果支持事件互斥validation，不支持未见台站泛化声明。详见 [training_exposure_summary.csv](training_exposure_summary.csv) 与完整 [actual_training_station_exposure.csv](evidence/reference/actual_training_station_exposure.csv)。

## 已准备的请求与尚缺结果

| scope | validation事件 | decision数 | 目标记录 | noninput记录 | 当前证据 |
|---|---:|---:|---:|---:|---|
| random | 1310 | 7852 | 113226 | 70558 | 仅请求；实际时间1.01–19.98秒 |
| long | 1310 | 5240 | 74116 | 39106 | 仅请求；40/90秒 |
| replay | 3 | 240 | 4800 | 3615 | 仅请求；1–20秒、natural/fixed_s0、normal/random |

这里decision按dataset/event/time/geometry/input_mode去重；目标记录按decision/station去重，noninput为triggered_noninput和untriggered_noninput之和。random原3930draw中有4个重复快照，3926快照乘两geometry为7852decision；重复draw的原计划仍保留，不当作新快照。long/replay的资格和模型原生能力须在实际执行中判断，这些人口不是任何模型已完成的预测分母。详见 [request_population_summary.csv](request_population_summary.csv)。

`jobs_manifest.json`仅记录三个阶段的实际提交回执：identity **29351785**，verify数组 **29354874**（0–10，最多8并发），reference **29354884**。本目录名的29351034是上一次环境诊断Job，不能用它查本批三个作业。日志显示MPI/DTK初始化、Torch导入已成功：Torch `1.13.1+git55d300e.abi0.dtk2304`，ROCm `5.4.23191-0e6bf66b`；CPU identity/reference无加速器正常，verify各元素有加速器。TEAM错误发生于环境准备之后。

本批没有 `sacct` 文件，所以不写Slurm `COMPLETED`、ExitCode或运行时长；`SUBMITTED`仅为提交成功回执。identity/request/reference的阶段产物及SHA一致；verify按模型区分8 PASS/3 BLOCKED。`legacy_fixed`、`random`、`long`、`replay`、`analyze`、`pack`均未出现在提交manifest，也没有相应正式结果。

后续可让通过核验的8组沿当前source继续既有评价入口；TEAM会按门自动跳过，无法形成包含TEAM的完整比较。若选择先修复TEAM，应按前述新source/new run ID处理，不删除失败目录或重签旧lock。提交步骤见 [超算运行手册](../../docs/ai/FE01_EVAL1_HPC_RUNBOOK.md)。本轮没有新训练、checkpoint重选、阈值调整、test访问或远程任务提交。

## 文件完整性与复核入口

下载目录共84文件、22659628字节；与用户提供的同名tar.gz全部文件逐字节SHA一致。14份原阶段seal中的30个文件全部通过；当前31个评价源码文件与超算保存的SHA逐文件相同；原115文件训练源码指纹仍为 `3e537ff6693971d92b139e517f370b998237bd8b29989194e9ddcc06cc9dcfea`。HDF5只具有超算沿用原audit的bytes/mtime与SHA记录，本地未重算大型数据字节。

原始权重、大型request/label表、cache和下载目录留在本地。本目录保存脱敏JSON、逐模型/人口/曝光CSV及必要小型日志：绝对私有路径替换为变量，数字、失败原因和记录中的SHA保留。`source_inventory.csv`区分原文件SHA与导出位置；脱敏文件不能拿原SHA直接校验。`artifact_manifest.sha256`校验本报告的实际导出字节。

机器可读总入口 [summary.json](summary.json)，本地一致性检查 [integrity_checks.json](integrity_checks.json)，给ChatGPT的阅读任务 [CHATGPT_ANALYSIS_PROMPT.md](CHATGPT_ANALYSIS_PROMPT.md)。前轮 [CSV分析](../fe01_eval1_review_20261005/README.md) 是历史性能证据；本批新增的是门、参照和曝光，不能把两种证据混写为新完整系统成绩。

数据汇总可复现（纯标准库，不导入Torch、不读取HDF5、不调用调度器；output必须是新目录）：

```bash
.venv-fe01/bin/python scripts/fe01_diagnostics/summarize_eval1_gate_results.py \
  --input artifacts/fe01_eval1_mpi_29351034 \
  --archive artifacts/fe01_eval1_mpi_29351034.tar.gz \
  --training-root ../chaosuan_res/fe01_runs_20261002 \
  --plan artifacts/fe01/eval1_local_plan_20261005 \
  --output artifacts/fe01/gates_reproduced_20261006
```

README/ChatGPT说明为本轮人工核对的报告文字；脚本复现JSON、CSV及脱敏证据。
