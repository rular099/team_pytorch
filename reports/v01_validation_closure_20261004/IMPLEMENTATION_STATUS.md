# V01 validation closure 实施状态（2026-10-04）

repo `rular099/team_pytorch`，branch `exp/v01-velocity-prep-padding-control`，
base commit `74c55aa5442b4200961c88ceee3d11af5033275d`。
result commit 为本文件所在提交，精确 SHA 随最终交付提供。

已核验：HEAD 与任务书基线一致；用户未跟踪 `tmp.tar.gz` 和两份技术交流目录未触碰。
AA 15 组 event/time 重复共 45 额外行，各组至少一项预测/概率字段不同；不去重旧 NPZ。
完整字段/行号/请求 index/文件 SHA 见 `aa_offline_duplicate_audit.json`。
这满足增加且仅增加一次 AA random validation 的必要性，不增加速度 random 或训练。

代码：V01 validation-only clock mask copy、strict request iterator、真实 cache sensor IDs、
UTC cutoff 与 outcomes ledger；五格 normal 提交器、可选一格 AA random；真实 idx7 审计门禁；
旧四格 random 身份 sidecar + 输入/label 签名；新分层 CI、MSE 分解、outer-match 分母与图源。
模型和 train_light.py 未修改；旧 V01 配置/统计器/旧报告不变；RT55 默认保持旧数据路径。

真实 cache/preflight 文件仅在超算：本地尚无真实 idx7 before/after trace，
不能声称已确认实际失败根因或得到新 normal 指标。当前只验证生产类 fixture 中的机制。
超算 audit 必须确认真实错误 + 相同 clock + 新 query 恢复，失败则依赖评估不得启动。

测试见 `docs/ai/CODEX_RESULT_20261004_V01_VALIDATION_CLOSURE.md`。
执行说明/完整提交脚本见 `docs/v01_validation_closure.md`、
`tools/complete_v01_validation_slurm.sh`。

新增协议明确报告恢复的 query、弃权/未匹配目标和选择偏差。
真实 physical-site remote 无充分 cache 身份时标未知；paired sensor-ID 排除 proxy 单独报告。
全记录去均值/继承的 cutout+1 不变，仍不是严格在线因果性认证。
本轮没有提交 Slurm、重训或访问 held-out test。

新增纯离线 P1 材料在 `offline_random/`：4 原 velocity random NPZ、5000 次
event-cluster CI、seed20260915；主比较总体 MAE MM−FF=-0.007620（CI不跨零），
above -1.2 MAE=+0.009195（CI不跨零），centered variance 差 CI 跨零。
这里的配对是 exact coordinates/time、truth/current/first-pick 验证，不是新的 sensor-ID/UTC 认证。
normal/A/V 矩阵仍未完成。固定坐标密度/残差图、图源计数和主比较 event sufficient table 已保存。
