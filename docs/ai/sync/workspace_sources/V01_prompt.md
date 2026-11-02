请先暂停上一轮独立 RT62 任务，执行 V01：
速度波形训练与 P 前缺失/补零受控实验。

先阅读随附文件：
V01_VELOCITY_PREP_PADDING_CODEX_PROMPT_20260929.md
该文件是本轮完整的实验、实现、测试和结果交付规范。

[AI-HANDOFF]
task_id: 20260929-v01-velocity-prep-padding-control
repo: rular099/team_pytorch
branch: exp/v01-velocity-prep-padding-control
base_commit: 9c95dbfaf92f36b2673d816026cd3f25b91eec66

goal:
  实现保持原加速度模型计算不变的速度输入训练，
  通过完整/人工P前缺失对照检验缺失代价，
  并交付用户可在超算直接执行的提交脚本。

verified_facts:
  - RT55已有storage-valid mask和masked normalization。
  - DiTing encoder调用未传时间mask，后续adapter才使用mask。
  - 现有训练HDF5与Hi-net年度CNT/CH归档schema不同。
  - tools/hinet_raw_archive.py已有可复用reader和worker-safe接口。
  - RT56保持RT55模型结构，使用normal/random混合采样。
  - 旧Hi-net审计是raw counts，且近邻匹配不保证同传感器/深度。

inferences:
  - P前缺失可能造成部分性能损失，但尚不能认定为主因。
  - 速度优于加速度不能单独证明补零有害。
  - 数据入口需要独立适配，建议新分支隔离，模型代码不分叉。

files_to_inspect:
  - AGENTS.md、docs/ai/PROJECT_CONTEXT.md、docs/ai/README.md
  - SESSION_SUMMARY.md
  - docs/hinet_velocity_download.md
  - reports/hinet_dataset_audit_20260811/*
  - tools/hinet_raw_archive.py
  - loader_light.py、gemini_util_light.py
  - train_light.py、eval_checkpoint.py、gemini_models.py
  - RT55/RT56配置、Slurm入口及对应tests
  - 用户实际速度数据文件、元数据和校正provenance

constraints:
  - 核验实际HEAD；有差异先报告，保留用户已有工作。
  - 新建建议分支；禁止破坏性Git操作和覆盖旧实验。
  - 不修改gemini_models.py、DiTing计算源码、模型张量结构。
  - 不改变attention/readout/MDN、内部幅值计算或损失公式。
  - 旧RT55–RT61配置、默认加载和推理行为保持不变。
  - 标签仍为原查询加速度站PGA，log10(m/s²)，不得替换为PGV。
  - 原split先继承再筛交集，禁止过滤后重新划分。
  - 不读取test波形做实验，不调test，不自动续训或扫参。
  - 输入速度站与查询加速度站使用各自真实身份和坐标。
  - 不把同一速度波形复制成多个独立输入台站。
  - 不把同seed误当作同物理窗口和同采样计划。

proposed_change:
  - 新增独立velocity backend，探测真实schema与units；
    复用年度archive reader，使用进程独立handle和有界缓存。
  - raw CNT按通道表处理；已校正或SAC数据防止重复去灵敏度。
    明确记录sensitivity-only和完整响应校正的区别。
  - 新增source/query契约、固定采样计划及P前缺失干预工具。
  - 准备A_pair、V_full、V_missing三臂；
    两个V臂使用同一批速度记录，只有P前支持不同。
  - 缺失模板从原加速度TRAIN生成，固定到event/source；
    填零且mask=False，不删除P点及P后样本。
  - 三臂使用相同RT55结构、相同RT55ep32初始权重，
    冻结DiTing，按RT55/RT56规则训练adapter/TEAM/原heads。
  - 统一seed42、8个新epoch、固定LR1e-4，
    相同batch/world-size/update计划，固定最终epoch8评价。
  - 两个V checkpoint各在full/missing视图上评估，共四格。
  - 新数据路径统一采用合法前缀统计，旧默认不变；
    不要求先执行完整RT62桥接。
  - 原训练/评估入口仅作必要透传与provenance导出。

acceptance_checks:
  - 核验三分量、采样率、时间轴、units、response与source hashes。
  - 报告KNET兼容配对及KiK-net/Hi-net配对各自真实样本数。
  - 不能匹配时如实标注，不扩大距离或换标签来凑数量。
  - 三臂model_params、模型源码、初始tensor hash、
    trainable名单、loss配置和优化计划一致。
  - 两个V视图同event/source/query/absolute cutoff/crop；
    单位转换后的raw post-P值逐样本一致。
  - future/deleted-prefix变形、masked filler、mask语义、
    worker/DDP、断点恢复和旧RT55兼容测试通过。
  - 新cohort实际counts据实报告，不套用旧formal目标数。
  - 保存三臂和四格逐目标MDN结果、缺失剂量和完整配对身份。
  - 联合报告点误差、概率、尾部及空间指标；
    使用5000次event-cluster CI，seed20260915。
  - bash -n、py_compile和针对性tests实际执行。
    未执行项目明确标NOT RUN。

hpc_followup:
  - 必须交付真实：
    tools/run_v01_prep_padding_controls_slurm.sh
  - ACTION支持preflight/train/eval/analyze/all；
    ARMS支持vfull,vmissing,apair。
  - 默认DRY_RUN=1，正式提交需CONFIRM_V01=1 DRY_RUN=0。
  - 复用现有train_light_slurm.sh和eval_checkpoint_slurm.sh，
    适配真实GPU/DCU环境和资源变量，不新增硬编码私有路径。
  - all使用afterok依赖串联preflight、训练、验证和分析。
  - 新输出目录，禁止覆盖；resume需显式开启并核验身份。
  - 只评价val，checkpoint必须核验epoch8，不能只看best/last名称。
  - 由用户在超算提交，本轮不要自行提交训练。

risks:
  - 速度与加速度的仪器、场地、频响和样本组成不同。
  - A_pair可能不是历史KNET-only输入，必须标明桥接性质。
  - 两个V组能识别速度域内的缺失代价，
    不能据此解释原加速度模型的全部误差。
  - 加速度预训练起点与8轮预算不代表速度模型的性能上限。
  - 单seed、开发validation不等于独立泛化证据。

open_questions:
  - 真实格式/覆盖/父权重先从本地preflight解析，不凭记忆假定。
  - 若必须改变模型计算、查询网络或PGA定义才能实施，
    报告该阻塞，禁止静默越界。
[/AI-HANDOFF]

最终返回[CODEX-RESULT]，必须包含：
精确base/result commit、分支、修改文件、真实测试命令与结果、
RT55兼容性、实际数据计数、未完成项、HPC状态。

同时附：
1. 实际超算提交脚本的完整内容；
2. 所有环境变量与资源设置说明；
3. 可复制的dry-run和正式提交命令；
4. 完整三臂及仅两个速度核心臂的运行方式；
5. 训练后需要回传/同步的结果清单。

不要只写“建议运行sbatch”，也不要把未执行训练写为完成。
