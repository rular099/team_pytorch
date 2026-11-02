请执行当前目录中的：

CODEX_TASK_FE01_EVAL1_20261004.md

这是 FE01-EVAL1：复用现有权重的同人口复评与空间诊断。
研究协议、优先级、指标和验收标准已经由 ChatGPT 定稿，
请按任务书实施，不重新扩大实验范围。

[AI-HANDOFF]
task_id: 20261004-fe01-eval1-existing-model-comparison
repo: rular099/team_pytorch
branch: exp/fe01-feature-extractor-site-effects
base_commit: b2756825190f93034ece436fab778e8a69adb41c

goal:
  不重训、不开 test，利用现有结果和权重完成可比评价工具、
  本地可执行分析及用户手动提交的超算评价脚本。

verified_facts:
  - 九组已有结果都是 validation。
  - PhaseNet seed42/43/44 固定 epoch12/10/12。
  - EQT seed42/43/44 固定 epoch10/10/10。
  - TEAM seed42/43/44 固定 epoch11/11/11。
  - 上述 epoch 来自曲线/CSV，仍须核验真实 checkpoint 内部身份。
  - 当前缺 RT55/RT61 同人口复评及正式空间/回放证据。
  - 原 code_identity 包含 fe01/*.py 和 tools/**/*.py，
    修改这些文件会改变旧训练源码指纹。

inferences:
  - 当前优先级是补评价证据，不是追加训练。
  - 总体误差改善不自动等于极早期、未触发或空间形状改善。
  - 旧 RT55/RT61 是完整系统参考，不是新训同公共下游的
    DiTing encoder 对照。

files_to_inspect:
  - 先读 AGENTS.md、docs/ai/PROJECT_CONTEXT.md、
    docs/ai/README.md、SESSION_SUMMARY.md。
  - 再读完整任务书第1节列出的代码、结果、config、lock、
    原始 CSV 与 provenance。
  - 先只读定位本地已有原始结果和权重，不要求用户重复提供
    已经存在且可解析的信息。

constraints:
  - 只在既有 FE01 独立 worktree 工作。
  - HEAD 不同先核对 diff，不 reset/clean，不覆盖未提交改动。
  - 不训练、不续训、不重选 epoch、不调阈值或 sigma、
    不开 test，不自动 sbatch/srun，不自动下载。
  - 原 runtime、配置、lock、checkpoint 和结果保持不变。
  - 新工具放在 fe01_review/、scripts/fe01_review/、
    configs/fe01_review/ 及对应 tests/docs。
  - 不删除旧 code hash 校验，不重签旧 lock 绕过检查。

proposed_change:
  1. 实现只读 checkpoint inventory 和结果来源核验。
     核验九组已选权重、RT55 ep32、RT61 ep8；
     缺项逐模型标注，不能凭文件名猜身份。

  2. 实现共同评价请求 manifest 与独立系统 runner。
     RT55/RT61 必须先通过同输入下的原模型数值等价检查，
     再做共同严格 prefix 复评。
     保留旧模型自己的归一化、尺度和反标准化；
     不直接替换成 FE01 的新尺度定义。

  3. 优先复用九组选中 epoch 的已有 fixed-time CSV。
     按任务书固定规则做小样本前向核验；
     不默认完整重跑九组固定时刻评价。
     新前向优先补 RT55/RT61、共同随机时刻、
     原生能力范围内40/90秒及有限案例回放。

  4. 完成本地已有 CSV 能支持的真实分析：
     all/noninput/triggered/untriggered、实际单输入站、
     1秒概率评分、固定目标面板、事件level/shape误差。
     场地分析使用共同train-only参照、近等距离站对、
     跨事件台站残差恢复；不能只改名“残差MAE”。

  5. 准备3个元数据规则预选validation事件、
     seed42三模型加可用RT55/RT61的1–20秒逐秒回放。
     固定输入与自然新增输入分开，真实站点评分与网格展示分开。
     没有训练期空间隔离，不宣称unseen-station泛化。

  6. 提供独立手动超算脚本、任务manifest和运行手册。
     准备阶段只打印命令；不执行远端。
     输出全新目录，旧结果只读。

acceptance_checks:
  - 执行完整任务书第9节的身份、人口、因果、query/cache、
    数值、统计定义、脚本与RT55回归检查。
  - 主表仍为共同1/3/5/10/20秒×normal/random、
    十cell等权noninput MAE，单位log10(m/s²)。
  - 原固定人口每run1310事件、139440条noninput记录；
    不匹配时输出差异并阻止正式配对，不静默删行。
  - 事件bootstrap与seed标准差分开，
    同一旧参考不能复制成三个独立DiTing seed。
  - 缺权重/HDF5/设备的项目写NOT_RUN或BLOCKED及原因，
    不用合成测试冒充Japan结果。
  - 原runtime指纹保持不变，新评价源码身份另行保存。
  - 使用fake scheduler测试确认准备流程没有自动提交。

hpc_followup:
  - 由用户手动执行identity、reference复评、随机/长时刻补评、
    train-only参考和案例回放任务。
  - 给出准确命令、输入、输出、依赖与实际缺项，
    不仅给一个模糊的“请运行评估”。

risks:
  - validation已经用于选checkpoint，不是独立test。
  - 原生窗口、训练曝光、预训练和adapter仍存在归因混杂。
  - RT55/RT61严格prefix复评不是历史pipeline原样成绩。
  - 稀疏未触发群体、上游offline处理因果性、
    实际触发/通信延迟仍需明确限制。

open_questions:
  - 本轮科学范围已经确定，无需重新询问是否重训或开test。
  - 账户/路径/资源仅在本地已有资料无法解析时由用户补填。
[/AI-HANDOFF]

请实际完成本地可执行的工具、测试和结果分析，再提交/推送本任务文件。
不要只返回另一份规划，也不要因缺少某个超算文件而停止全部工作。

最后按完整任务书返回 [CODEX-RESULT]：
包含精确 result_commit、实际修改、实际测试、本地分析产物、
各checkpoint身份状态、HPC未执行项、用户手动命令和剩余风险。
