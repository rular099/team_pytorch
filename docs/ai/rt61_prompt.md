请执行 RT61：波形—几何交互残差条件化实验。

先阅读随附任务书：
RT60_REVIEW_RT61_CODEX_PROMPT_20260925.md
该文件是本轮实施、验证、停止和结果交付的完整规范。

[AI-HANDOFF]
task_id: 20260925-rt61-wave-geometry-residual-conditioning
repo: rular099/team_pytorch
branch: rt61-wave-geometry-residual-conditioning
base_commit: 00d624d2fb01c6dd98cac150043dd15b7d8366ff

goal:
  只开展一个受控实验，检验补充波形/事件状态与查询几何的交互，
  能否同时改善实际单输入台站的空间差分和 normal 非输入台站精度。

verified_facts:
  - RT60 机制门控通过 8/14，legacy required 通过 18/25。
  - 单站 pairwise-delta MAE 只改善约 0.17%。
  - normal 非输入 MAE/RMSE 显著退化，不能采用 RT60。
  - RT60 reference 使用共享 residual_input；
    修改表示时必须避免 reference 随 student 漂移。

inferences:
  - 冻结 pair representation 可能限制波形—几何交互的可读性，
    但尚未证明它是唯一瓶颈。本轮只检验此假设。

files_to_inspect:
  - AGENTS.md、docs/ai/README.md、PROJECT_CONTEXT.md、SESSION_SUMMARY.md
  - RT60 结果报告、summary、分组指标、gates 和训练记录
  - gemini_models.py 中 RT59 transport/reference/route 路径
  - tools/rt60_contrast_objective.py 及 RT59/RT60 分析器
  - 对应 trainability、配置、checkpoint/export 和测试代码

constraints:
  - 先核对实际 HEAD；如有变化，报告差异并保留用户工作区。
  - 代码基于上述 HEAD，权重从精确 RT59-v3 epoch-8 parent 初始化。
  - 不从 RT60 candidate 权重继续训练。
  - RT55–RT60 原配置、默认行为和 checkpoint 兼容性不得改变。
  - 不改变 sampler、split、cutoff、loss 权重、MDN 概率形状。
  - 不读取 held-out test，不选中间 epoch，不做 seed/rank/权重扫描。

proposed_change:
  - 新增 opt-in、rank=16 的 residual-only 波形—几何交互旁路。
  - h1 = h0 + Wo(tanh(Ww LN(z)) * tanh(Wg geometry))。
  - Wo 零初始化；Ww/Wg 非零初始化；起点与 RT59 数值一致。
  - 只训练该旁路与既有 residual_head，其余参数和 buffer 冻结。
  - reference 和原 station weighting 必须继续读取 h0。
  - student residual_head 读取 h1；normal observed 路由保持不变。
  - 不采用依赖 query 集合均值的部署修正，不使用目标标签构造推理特征。

acceptance_checks:
  - 验证真正 reference 输出不漂移，而不只验证其参数哈希。
  - 检查起点恒等、训练白名单、梯度可达、NaN/mask、单站行为。
  - 检查台站重排及 query 增删/分块后的预测一致性。
  - 验证 checkpoint 保存、恢复及 trainable-delta 重建。
  - 完整报告原 14 个机制 gates 和 25 个 legacy required gates。
  - 新增 useful_joint_progress：
    单站 pairwise-delta MAE 相对 RT59 至少改善 2%，
    且 normal 非输入 MAE/RMSE 配对 CI 上界均小于 0，
    同时通过原机制 gates。
  - 仅有微小空间改善或 normal 再次退化，不视为模型升级。

hpc_followup:
  - 固定 seed42、原 25 年度 shards 的既有 train split。
  - 恰好训练 8 个新 epoch，固定最终 epoch8 评价。
  - LR 与 RT60 相同：1e-4×4、5e-5×2、2.5e-5×2。
  - 只在已有明确授权时执行该单一正式 run；
    否则提交实现、测试与 ready-to-run 命令，标明 not submitted。
  - 禁止自动续训、补 seed、扫参或打开 test。

risks:
  - 同一 development validation 已反复使用；本轮仍非独立泛化证据。
  - normal 保护仍为软惩罚，新增表示的收益不确定。
  - 少站早期观测存在信息限制，不能把空间范围扩大等同精度提升。

open_questions:
  - none；执行环境或父权重缺失时，完成不依赖它的部分并如实标注。
[/AI-HANDOFF]

结果必须同步到 GitHub：
精确代码/数据/权重身份、resolved config、逐组与逐场指标、
event-cluster CI、全部 gates、误差分解、reference 不变性证据、
训练曲线、统一坐标的真值—预测图、固定规则选取的成功/失败样例，
以及可在精确父权重上恢复的轻量 trainable-delta 包。

结束时按仓库约定返回 [CODEX-RESULT]。
未执行的训练、测试或推送不得写成已完成。
