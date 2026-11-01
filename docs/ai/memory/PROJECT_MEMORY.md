# 项目长期记忆

Repository：`rular099/team_pytorch`。本文件保存已核验的稳定约束；动态状态看
[CURRENT_STATUS.md](CURRENT_STATUS.md)，新对话摘要看 [CHAT_HANDOFF.md](CHAT_HANDOFF.md)。

## 科研目标与任务

从截至决策时刻可用的多台站三分量波形、输入与查询台站位置，预测查询台站最终
PGA的概率分布，研究非输入/未触发目标的空间辨识、动态范围、校准及实时可用性。
当前体系采用冻结预训练波形前端、可训练station接口与TEAM风格融合/PGA读出；
FE01比较前端及幅值路径，FE02独立研究事件融合，V01研究速度与P前缺失干预。
具体新架构由对应协议定义，不能将某条线配置强制推广为所有实验线的默认值。
模型必须学习波形与空间信息；尚未授权的新人工传播/GMPE/地质修正不能伪装成原设计。

输入时钟、mask、storage-valid区间与实际forward路径须追溯，注册了模块不等于路径调用它。
原系统还保留震级/位置辅助任务；概率PGA采用MDN。报告坐标为 `log10(m/s²)`，
不能把该坐标的误差当成线性加速度误差。原始三分量波形、物理单位、原生窗口和
预处理的在线可用性按协议分别核验。

## 科学评价与兼容性

- RT55原checkpoint加载、参数命名/形状、默认配置和推理保持兼容；后续能力用独立配置/显式开关。
- train、validation/dev、held-out test分开。不得用test选择模型、采样、阈值或汇报口径。
- 原RT55按年度内event-disjoint划分，不是held-out-year；新线实际支持cohort以其audit为准。
- 不能用输入台站的容易目标掩盖非输入目标。normal/random、目标角色、实际K、时刻和空间field人口分别报告。
- 点误差与NLL/CRPS/Brier、覆盖率/宽度一起判断；空间range变大、attention图或坐标梯度不单独证明准确空间预测。
- 预注册门、固定训练预算与选轮规则不因结果修改。联合门失败可与局部点指标进展并存。
- 事件簇bootstrap条件于固定模型与选轮，不代替训练seed重复或消除重复使用validation的选择偏差。
- 缺失/空cell/弃权/Unknown保留分母；缺同步不等于未完成，SUBMITTED或队列消失不等于成功。
- 历史source lock、checkpoint、原seal及已发布结果不重签、不覆盖。缺原始权重时区分导出元数据核验与张量重载。

## 协作与证据

用户决定目标与科研取舍；ChatGPT做独立解释/审阅；Codex核对真实文件并实施受托改动；
超算保存大型数据、权重和正式运行产物。Git保存轻量报告/身份/索引。当前任务只做文档和维护工具，
不触发训练、评价推理、sbatch/srun或远程计算。不得声称读取了全部历史聊天。

事实核验优先看精确commit下代码、resolved config、split/lock、checkpoint与结果provenance；
其次是有证据的带日期报告，再次是历史交接/聊天补充，最后才是明确标注的推断。
该顺序用于事实冲突，不削弱当前用户指令的授权优先级。来源冲突保留两方和差异，不能静默覆盖。
没有原讨论不能补造用户决定或理由，标 `needs_history_review`。

成功标准：新对话用三份短入口回答目标/实验线/可信结论/待办，并能从ID定位精确报告、
配置、结果和commit；不同实验线可并行，科研结果和用户未提交工作不受影响。

## 权威入口

- [AGENTS.md](../../../AGENTS.md)：兼容性、安全、证据与交接规则。
- [历史上下文](../PROJECT_CONTEXT.md)：2026-09-02 RT55/RT56快照，不能作为全局最新状态。
- [逐线实验索引](EXPERIMENT_INDEX.md)、[决策记录](DECISION_LOG.md)：固定commit的证据链接。
- [并行规范](PARALLEL_WORKFLOW.md)：所有权、产物路径与故障恢复。

以上长期信息来源于原AGENTS、RT55/56上下文、FE01/FE02与V01协议。完整来源commit见各线记录。
