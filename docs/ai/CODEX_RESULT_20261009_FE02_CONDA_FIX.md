# FE02 Conda 非交互启动修复

用户报告：`/public/software/apps/miniconda3/etc/profile.d/conda.sh: line 55: PS1: unbound variable`。
真实job.sh以set -euo启动并source env.sh；后者的set -eo并不会取消继承的nounset。
在本地使用真实env.sh、开启nounset且unset PS1的调用方及合成Conda函数复现同类错误。
修复仅关闭bootstrap期间nounset、初始化PS1、禁止Conda改prompt；激活后仍恢复set -u。
错误退出检查set -e/pipefail和来源校验保留，模型与实验定义不变。

上传 `artifacts/fe02/fe02_source_20261009_conda_fix.tar.gz` 到新源码目录后直接运行
`bash scripts/fe02/submit.sh`。失败发生在audit/训练前，不需要resume。
不覆盖旧日志/训练目录，不散补脚本后跳过来源校验。

[CODEX-RESULT]
task_id: 20261009-fe02-conda-nounset-fix
repo: rular099/team_pytorch
base_commit: ef274d0d4aa0ddc775848347b68ea68d7289ddc1
result_commit: 本提交精确SHA见修复包SOURCE_IDENTITY.json
branch: exp/fe02-event-fusion-readout
changed_files:
  - scripts/fe02/env.sh: set +u、PS1默认空值、CONDA_CHANGEPS1=false；原激活后set -u保留
  - tests/test_fe02_cluster_env.py: 实际shell bootstrap的3项合成回归，不用真实Torch/Conda/GPU
  - SESSION_SUMMARY.md、docs/ai/README.md、FE02_HPC_RUNBOOK.md: 当前失败/修复/单行重提
verification:
  - 修复前3项bootstrap测试失败，明确复现PS1 unbound
  - 修复后3项bootstrap+10项既有提交编排=13项PASS
  - bash -n env.sh/job.sh/submit.sh PASS；git diff --check PASS
compatibility:
  - 不改任何RT55/FE01模型、checkpoint加载、推理、训练配置或数据协议
  - 保留直接sbatch提交、audit失败阻断、严格身份及last-only恢复
hpc_status:
  - 用户报告原任务初始化失败，无JobID/sacct可核验；本会话未提交超算任务
  - 修复版真实超算环境/前向/训练/validation尚待执行，不由合成本地测试代替
remaining_risks:
  - 已解决报告的nounset/PS1问题；其他远端模块/权重/设备问题仍须实际日志验证
review_request:
  - 复核仅在bootstrap期间关闭nounset；激活错误不吞掉，激活后恢复严格检查
[/CODEX-RESULT]
