# FE02 现有超算结果提取与独立分析

已将用户提供的真实结果整理为
[报告与轻量证据](../../reports/fe02_hpc_results_20261104/README.md)，并提供
[ChatGPT 独立分析任务](../../reports/fe02_hpc_results_20261104/CHATGPT_ANALYSIS_PROMPT.md)。
本轮分析运行于用户安装依赖的 general 环境；没有训练/前向或新超算提交。

[CODEX-RESULT]
task_id: 20261104-fe02-existing-results-review
repo: rular099/team_pytorch
base_commit: 42b3d5405d9fe9672f0a0925275ca4f7e9e6a9e7
result_commit: 本提交；完整SHA以最终交接回执和git log为准
branch: exp/fe02-event-fusion-readout
changed_files:
  - reports/fe02_hpc_results_20261104/: 真实源文件SHA索引、可移植配置/audit/回执、点/概率/空间/配对统计、复算脚本、SVG和独立分析
  - docs/ai/PROJECT_CONTEXT.md, docs/ai/README.md, SESSION_SUMMARY.md: 添加带日期的当前结果入口，历史状态保留
  - docs/ai/CODEX_RESULT_20261104_FE02_RESULTS.md: 本回执
verification:
  - 原fe02.analysis.compare: 五组seed42固定validation，5000次bootstrap，PASS
  - 原tests.test_fe02_analysis: 两项PASS（CPU合成分析契约，不是模型/设备测试）
  - 464个原始文件SHA，源码包成员SHA，115 backend/42 experiment身份和92个当前本地科研文件核对
  - 五组同split/cohort/data记录/encoder/公共初始化/预算；60份每轮validation分数独立复算
  - 最终人口/逐目标身份/标签/输入/可用截断元数据配对，原指标表重建，512 MDN行/组独立评分
  - verify_summary.py: 从轻量事件表重建5行主分数和63行bootstrap，PASS
  - Python编译、SVG生成/目视检查、git diff --check；具体最终状态见会话回执
compatibility:
  - 只新增报告脚本/证据并更新入口文档；原RT55/FE01/FE02模型、训练/评价、配置和启动器未改
  - 不改旧source/lock/checkpoint、原始结果和用户未跟踪接手报告；未运行新的模型兼容回归
hpc_status:
  - 本轮无sbatch/srun/重训/推理/test；分析本地已下载的训练和validation产物
  - 五组日志记录完成12轮与validation；无sacct，调度器State/ExitCode未核验
remaining_risks:
  - 单seed、validation用于checkpoint选择；CI未校正多项比较且不涵盖跨seed/选择过程不确定性
  - 缺权重字节/committed training journals/sacct；最后R0/A/B/C launcher SHA不同且原脚本字节未取得
  - 单台晚时刻normal人口极少，bootstrap存在空cell；概率仍欠覆盖，C不是等容量对照
  - 预训练数据重叠、上游在线因果性、未见台站泛化均未认证
review_request:
  - ChatGPT在精确结果commit上独立核验来源与分层；比较本报告结论，给出一致点/分歧与最小后续证据需求
  - proposed_change=none；不得把本次validation当held-out test或据此调test方案
[/CODEX-RESULT]
