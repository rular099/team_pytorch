# FE01-EVAL1实际交接（2026-10-05）

结果代码与真实CSV分析固定在commit `478475f9381a462212affb33c8a9a0adeda70f1d`。本文件是后续交接记录，不改变该结果commit中的代码、指标、旧runtime或选择。

```text
[CODEX-RESULT]
task_id: 20261004-fe01-eval1-existing-model-comparison
base_commit: b2756825190f93034ece436fab778e8a69adb41c
result_commit: 478475f9381a462212affb33c8a9a0adeda70f1d
branch: exp/fe01-feature-extractor-site-effects
changed_files:
  - fe01_review/: 只读权重inventory、冻结请求、独立系统runner、统计/空间诊断、reference、有限replay、pack与独立provenance。
  - scripts/fe01_review/: 9动作CLI、只打印命令的准备器、6个CPU/DCU sbatch入口、conda zb初始化、轻量报告及HPC overlay打包工具。
  - configs/fe01_review/: 原115文件pin、固定评价合同、11模型及任务manifest模板、无私有路径的环境模板。
  - tests/test_fe01_review_{statistics,runners,scheduler,requests}.py: 数学/权重身份/桥接/未来/query/cache/请求及fake scheduler检查。
  - docs/ai/FE01_EVAL1_{PROTOCOL,HPC_RUNBOOK}.md: 科学边界、旧桥接和逐阶段中文手册。
  - reports/fe01_eval1_review_20261005/: 真实轻量表、逐event/field主群体加总、11张图、图源、CI、counts、缺项与SHA。
verification:
  - PASS exit0: .venv-fe01/bin/python -m compileall -q fe01_review scripts/fe01_review
  - PASS exit0: bash -n scripts/fe01_review/{env.sh,print_submit_commands.sh,identity_job.sbatch,evaluate_job.sbatch,reference_job.sbatch,replay_job.sbatch,analyze_job.sbatch,pack_job.sbatch}
  - PASS: Python3.9语法解析17个新Python文件；实际本地解释器Python3.11，HPC Python3.9尚未执行。
  - PASS exit0 50 tests: env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-fe01/bin/python -m pytest -q tests/test_fe01_review_statistics.py tests/test_fe01_review_runners.py tests/test_fe01_review_scheduler.py tests/test_fe01_review_requests.py tests/test_fe01_causal_replay.py tests/test_fe01_statistics.py tests/test_fe01_native_windows.py tests/test_eval_checkpoint_formal.py tests/test_rt61_wave_geometry.py
  - PASS: fake sbatch/srun marker未生成；DRY_RUN=0拒绝；train/resume/test动作拒绝。
  - PASS: 原始CSV字节SHA、194265行人口/139440 noninput、原标签/输入/时钟SHA；历史九组主分数到1e-12不变。
  - PASS: 5000次事件簇CI及早期概率CI；图从保存分箱/加总渲染，实际预览density/PIT图；轻量pack及artifact manifest验证。
  - PASS: Git暂存diff --check；源码/报告私有路径扫描（pytest绝对site-packages前缀已脱敏，原log SHA保留）。
  - NOT_RUN: Japan生产权重L0/C1、小样本CSV复算、随机/长时刻及案例前向、train-only参考；本地无所需权重/HDF5。
compatibility:
  - 原115文件/code SHA仍为3e537ff6693971d92b139e517f370b998237bd8b29989194e9ddcc06cc9dcfea；旧runtime/config/lock/result没有修改。
  - 原RT55类、state schema和默认源码不变；合成原模型直接/包装输出、物理单位解码及相关旧回归通过。不能称真实RT55 ep32前向已通过。
  - 新工具模块SHA、Git开发工作区标记、legacy loader SHA另记。旧参考保留其原归一化、ln尺度和反标准化；C1右对齐prefix分布变化公开保留。
local_results:
  - reports/fe01_eval1_review_20261005/README.md及artifact_manifest.sha256是完整轻量审阅入口；图源及逐field/逐event主群体加总随Git保存。
  - artifacts/fe01/eval1_full_local_final_20261005/保留完整所有角色逐field/逐event表；原始预测/大权重没有重复上传Git。
  - validation主MAE家族mean±seed SD: PhaseNet0.225818±0.000125；EQT0.235296±0.001310；TEAM0.241817±0.003011 dex。
  - 1秒NLL: PhaseNet0.324149、EQT0.296632；前者较高。逐seed配对CI前两seed跨0、seed44不跨0；不声称三seed均显著。
  - 未触发MAE: PhaseNet0.251068、EQT0.261969、TEAM0.268943。level/shape分解及5km站对真实结果保留；site重复残差INCOMPLETE。
  - normal始终noninput面板仅312事件；random固定面板1015事件，排除数和角色迁移完整保存。固定S0反事实尚未运行。
  - 共随机3930draw/3926snapshot/4重复，九组manifest字节相同；3个元数据案例及32个共同probe已锁定。
checkpoint_status:
  - TEAM seed42 ep11: BLOCKED selected checkpoint locally missing; CSV_IDENTITY_PASS; forward NOT_RUN.
  - TEAM seed43 ep11: BLOCKED selected checkpoint locally missing; CSV_IDENTITY_PASS; forward NOT_RUN.
  - TEAM seed44 ep11: BLOCKED selected checkpoint locally missing; CSV_IDENTITY_PASS; forward NOT_RUN.
  - PhaseNet seed42 ep12: BLOCKED selected checkpoint locally missing; CSV_IDENTITY_PASS; forward NOT_RUN.
  - PhaseNet seed43 ep10: BLOCKED selected checkpoint locally missing; CSV_IDENTITY_PASS; forward NOT_RUN.
  - PhaseNet seed44 ep12: BLOCKED selected checkpoint locally missing; CSV_IDENTITY_PASS; forward NOT_RUN.
  - EQT seed42 ep10: BLOCKED selected checkpoint locally missing; CSV_IDENTITY_PASS; forward NOT_RUN.
  - EQT seed43 ep10: BLOCKED selected checkpoint locally missing; CSV_IDENTITY_PASS; forward NOT_RUN.
  - EQT seed44 ep10: BLOCKED selected checkpoint locally missing; CSV_IDENTITY_PASS; forward NOT_RUN.
  - RT55 ep32: BLOCKED accessible candidate internal epoch20; actual file bytes/SHA recorded; no replacement allowed.
  - RT61 ep8: BLOCKED no checkpoint in locally uploaded zip/tar; original config available; parent/readout production identity NOT_RUN.
hpc_status:
  - NOT_SUBMITTED_THIS_TASK. No remote HPC execution, sbatch/srun submission, training, resume, calibration, download or test access.
manual_next_commands:
  - 将本地artifacts/fe01/fe01_eval1_hpc_20261005.tar.gz及.sha256拷到超算原team_pytorch_fe01项目根目录。
  - 在该目录: sha256sum -c fe01_eval1_hpc_20261005.tar.gz.sha256
  - tar -xzf fe01_eval1_hpc_20261005.tar.gz
  - sha256sum -c artifacts/fe01/eval1_overlay.sha256
  - export FE01_REVIEW_ENV="$PWD/scripts/fe01_review/cluster_zb.env"
  - source "$FE01_REVIEW_ENV"
  - bash scripts/fe01_review/print_submit_commands.sh "$FE01_REVIEW_ENV" > fe01_eval1_submit_commands.txt
  - cat fe01_eval1_submit_commands.txt
  - 只先手动复制identity的完整sbatch行；待其结束检查inventory与requests。随后verify逐模型通过，再按手册推进legacy_fixed/random/long/reference/analyze/replay/pack。
remaining_risks:
  - 权重、原HDF5与DCU上的实际前向验证仍待用户手动任务。epoch不匹配单独隔离，不重选或重训。
  - validation已选epoch；时间补评不成为独立test。系统预训练/原生长度/归一化/曝光/adapter混杂，不能推纯encoder贡献。
  - C1保留旧尺度但采用共同右对齐prefix；并非历史完整pipeline原样成绩。
  - site train-only参考与稳定台站证据未完成；没有真正spatial holdout，不做地质成因断言。
  - 真触发/通信延迟、上游offline滤波因果性未认证。图与数值回放可用才生成，未生成动画不称完成。
  - 随机子群若原draw没有合格目标，输出INCOMPLETE，不补0或静默改draw权重。
review_request:
  - 请ChatGPT按result_commit核查legacy桥接、原115源码锁保持、指标/人口/事件bootstrap/固定面板和手动提交边界；生产前向与site结论暂缓。
[/CODEX-RESULT]
```

HPC包的私有env只留本地交付包，从既有launcher、原config和checkpoint解析已知路径；Git模板不包含私有账户/路径。打印器默认并发8模型，每个GPU任务1节点1DCU；CPU任务无需DCU，没有无依据的运行时间估计。

交付包13585037 bytes，SHA256 `419d12335cdd291d7a77f7dc824ee84260a15ef961fe1b232c997175a316ee13`。实际在全新临时目录先解原`fe01_source_weights.tar.gz`、再解新overlay：原115源码SHA不变、新模块SHA为`8e99531c4cb76464f233ab2c61100ada6f6087a80a64bc2e67200cf2b8798ae4`、legacy loader字节与打包声明一致、overlay逐文件SHA全部通过。记录见`FE01_EVAL1_OVERLAY_VERIFICATION.json`。这验证交付包组合，不等于超算执行通过。
