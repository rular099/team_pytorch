# FE02-Lite 审阅交接（2026-11-23）

task_id: `20261123-fe02-lite-event-spatial`

仓库 `rular099/team_pytorch`，新分支 `exp/fe02-lite-event-spatial`，独立 worktree
`team_pytorch_fe02_lite`。实际基点 `618661f99afb67f5c23c9f1a3fd713a0a156e7b5`；
本地与 GitHub FE02 ref 均确认是该基点，无需叠加后续改动。结果 commit 由最终
CODEX-RESULT 与代码包 SOURCE_IDENTITY.json 标明，避免在自身 commit 中嵌入循环 SHA。
原 FE02 worktree 的未跟踪 `FE02_jieshou_2026-11-03.md` 保留。

实现已落地，HPC 正式任务 **NOT_SUBMITTED**。没有日本三视图真实数据计数、真实
DiTing forward、ROCm 恢复、训练性能或科学结果；这些检查由计算节点启动器执行，
失败即退出。不能把本地 synthetic/mock 的 PASS 写成真实 DiTing 或集群 PASS。

主要入口：

- [实现/数据语义及模型接口](docs/ai/FE02_LITE_IMPLEMENTATION.md)
- [超算运行手册](docs/ai/FE02_LITE_RUNBOOK.md)
- [产物 schema 与离线分析](docs/ai/FE02_LITE_ARTIFACTS.md)
- [本地验证记录](reports/fe02_lite_local_20261123/verification.json)
- [实际任务参数量](reports/fe02_lite_local_20261123/parameters.json)
- [用户完整任务](docs/ai/FE02_LITE_TASK_PROMPT.md)

主网络真实任务参数 **740,592**；d64=237,168，amp16=727,456，amp32=701,184，
pool4=740,848，no_source_aux=740,592。冻结 encoder 的注册参数量、总参数量和实际
类由真实 audit 另报，不以合成 encoder 的大小替代。

审阅顺序：`fe02_lite/data.py → model.py → losses.py → engine.py → evaluation.py →
analysis.py`，再查看 `scripts/fe02_lite/`、三个主配置和聚焦测试。重点核查：

1. 三视图网络/层位/suffix/PGA/mask 同行选择；KNET 的空 suffix 与 `0` 两种
   写法有不同构建器的明确来源。六分量存储会拒绝读取，禁止猜测通道。
2. KiK-only 事件如不在原 frozen split 中，明确排除并导出；不独立随机分割。
   本地不能确认原 manifest 的覆盖率，不能据历史 KNET 数量填三视图计数。
3. 井深缺失，station height 是明确的代理坐标。统一正向 UP 与 source depth
   DOWN 的固定转换；不会凭空提供井下深度或宣称场地泛化。
4. amplitude value 没有 LayerNorm；score/key 可归一化。没有人工振幅统计输入、
   source 参数反馈、额外 gates、时间/台站 Transformer 或旧大下游网络。
5. 每次 validation 逐目标完整分布可重算十 cell selection；source 按观测实例保存一次。
   学习任务头的 source 辅助项不因 query chunk 或目标个数重复加权。
6. checkpoint 只保存任务头；last 保存 optimizer/scheduler/各 rank RNG/epoch 边界
   提交位置。中断 epoch 重放，失败日志保留。真实 ROCm 续训仍需计算节点验证。
7. 两个层位即使相对首 P 时间相同也不是严格配对。跨视图共同事件统计与严格
   绝对截止时刻/物理站址相交表分开，保留不同真值和输入几何。

当前代码、所有主配置/脚本/合成测试均可审阅；没有正式训练成绩可供模型排名。
