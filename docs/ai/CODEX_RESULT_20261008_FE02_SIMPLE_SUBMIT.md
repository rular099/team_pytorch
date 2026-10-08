# FE02 一条命令提交入口

用户只运行 `bash scripts/fe02/submit.sh`，直接提交并返回Slurm JobID，不再复制命令。
`--dry-run`为可选预览；默认明确为真实提交。最新用户指令授权此行为。
入口自动冻结当前环境配置；计算任务执行权重登记/同一文件SHA核验、必要audit、
原正式12ep训练和normal/random固定validation。默认仅seed42五组，数组并发1。
没有新增smoke、pilot、多seed或test，也没有替用户提交真实超算任务。

[CODEX-RESULT]
task_id: 20261008-fe02-direct-submit
repo: rular099/team_pytorch
base_commit: 9596340b36364d8d5b0c433f0bb2e35756eeec36
result_commit: 本提交完整SHA记录于新版源码包SOURCE_IDENTITY.json
branch: exp/fe02-event-fusion-readout
changed_files:
  - scripts/fe02/submit.sh: 无参数直接sbatch；可选--dry-run；无需登录节点Python/torch或私有env
  - scripts/fe02/job.sh: allocation内串联登记/audit/train/eval，同run及共享登记加锁
  - scripts/fe02/register_diting.py: 明确reuse-existing，只复用相同路径/SHA/bytes的权重
  - fe02_cluster.env.example: 默认自动加载，覆盖可选
  - tests/test_fe02_simple_submit.py: 直接提交/返回JobID/提交失败及原编排等10项合成测试
  - docs/ai/FE02_HPC_RUNBOOK.md、README.md、SESSION_SUMMARY.md: 默认入口改为一条命令
verification:
  - 10项纯CPU/合成launcher测试PASS；含默认提交一次、失败不重试及Slurm spool回归
  - FE02已有配置/模型/分析回归PASS；shell语法/编译/whitespace PASS
  - 新源码包解压后逐文件SHA核验PASS；源码包不含权重和数据
compatibility:
  - 本次不改RT55/FE01模型、加载、推理、配置、采样、训练器或评价器
  - 原训练器继续检查AUDIT_PASS和source/config/data/encoder/world身份；仅last恢复
hpc_status:
  - NOT_SUBMITTED；本地仅用假sbatch测试，本会话没有访问超算提交真实任务
remaining_risks:
  - 超算权重/模块路径沿用已给默认值，远端存在性和性能仍以真实任务为准
  - 各组allocation现在包含audit与validation时间，23:50h超时须原身份从last恢复
  - 旧非空训练/评价目录不覆盖；不要将新版脚本散补到旧包并沿用旧source/audit身份
review_request:
  - 确认单入口没有实验定义变化；真实结果出来后另行科学分析
[/CODEX-RESULT]
