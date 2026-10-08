# 请 ChatGPT 独立审阅 FE02 实现与后续结果

读取仓库 `rular099/team_pytorch` 的 `exp/fe02-event-fusion-readout` 分支，
实现基点 `ba4fa7740d9fde5fde87a7b0ae0397037209baf4`，实现提交精确 SHA 见
用户转交的 CODEX-RESULT / 本文件所在提交。不要混读旧FE01或V01分支。

先读 FE02_TASK_PROMPT、FE02_PROTOCOL、FE02_HPC_RUNBOOK、CODEX_RESULT_20261008_FE02。
检查 gemini_models.py 的主 PGA 独立 dims 和 opt-in event-memory、fe02/config.py
的冲突/未知参数检查、fe02/model.py 的真实前端与共享初始化、FE01 engine显式
hooks及默认兼容路径、单台多台/概率配对汇总、脚本从last严格resume。

本地单元与合成数据证据不代表真实DiTing/HPC审计通过。当前没有训练成绩，
不能选胜者或说单台问题解决。正式计算由用户手动提交，默认五组seed42。
若发现阻断正确性的工程问题，具体给出文件/分支/验收点；不默认新增encoder、
解冻、gate/宽度/损失搜索、更多smoke或test。

真实结果返回后，先认证实际encoder类/adapter/YAML/nativefeature/冻结边界与
同一checkpoint SHA、cohort/sampling/训练预算/所选epoch/MDN概率。用同目标配对：
R0→A分离非线性解码作用，A→B与A/B→M分离事件摘要及注入时机，B/M→C判断
事件摘要是否足够。C不是等容量消融。重点同时看normal/random、single/multi、
noninput/untriggered、早期时间、MAE/RMSE/bias/slope、NLL/Brier/coverage以及
centered/pairwise空间误差；不要仅看动态范围或attention。

请分别给出verified fact / inference / proposal，优先判断本轮证据是否已能回答
问题、哪一种架构值得保留，以及至多一项真正影响结论的必要后续工作。
单seed、预训练重叠unknown、catalog触发/离线预处理、地质机制不可识别等限制
不得省略；没有完整结果或sacct保持未知。
