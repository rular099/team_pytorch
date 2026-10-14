# FE02 已有5轮checkpoint的无重训续跑修复

证据：用户提供chaosuan_res/fe02-run-29934377_4.out/.err及
fe02-run-30444656_4.err。原来源为9f70371838ec31044e559096a939f023d2924f37。
M/seed42完成audit和5轮，第一次因Slurm时限退出；续跑通过原身份/参数加载检查，
失败于fe01/engine.py:501的CUDA RNG恢复，不是权重strict-load失败。

`torch.load(..., map_location=device)`同时将rank_rng中的ByteTensor映射到DCU，
当前PyTorch1.13恢复接口需要CPU ByteTensor。直接改engine会改变原code_identity，
破坏已有audit/resume/评价的严格身份，所以本次不改任何受锁定源码或锁。

新增opt-in compatibility层，只在明确启用的training rank启动时安装，对恢复输入
逐个执行detach/cpu/contiguous，拒绝其他dtype/shape，不改变原始uint8字节/顺序。
原源码仍逐文件验证，原code_identity相同；运行时兼容层不伪装成原源码，独立SHA
由提交快照固定、每rank核验，并在成功恢复后记录每状态SHA和版本/rank/job。
不跳过/重写旧lock，不改checkpoint，不重新设seed或减少预算。

交付小补丁包仅有3个新增文件（入口、sitecustomize、说明）；解压到原代码目录，
`bash resume_fe02_rng_cpu.sh`默认直接提交M/index4，原pipeline从last恢复。
不要解压到新源码目录（config中包含原代码绝对路径），不要替换完整源码包。
此兼容层只用于已知9f70371来源任务；登录/辅助进程不因此import torch。

[CODEX-RESULT]
task_id: 20261014-fe02-rng-cpu-resume
repo: rular099/team_pytorch
base_commit: 9f70371838ec31044e559096a939f023d2924f37
result_commit: 本提交，精确SHA以git log及交付时记录为准
branch: exp/fe02-event-fusion-readout
changed_files:
  - resume_fe02_rng_cpu.sh: 验证原来源/compat SHA，默认M从last直接提交，复用原入口
  - compat/fe02_rng_cpu/sitecustomize.py: rank-only、显式开关、hash-pinned CPU ByteTensor恢复
  - FE02_RNG_RESUME_README.md: 小补丁上传到原目录及唯一提交入口
  - tests/test_fe02_rng_resume_compat.py: legacy接口复现/字节序列/严格拒绝/启用范围/hash/原code_identity/入口
  - SESSION_SUMMARY/docs: 当前真实进度、失败与严格无重训修复
verification:
  - 6项compat+3项bootstrap+10项提交编排=19项PASS；仅CPU/合成，不是实际DCU恢复
  - 合成已映射张量复现原ByteTensor错误，CPU归一化后原字节与后续随机序列一致
  - 当前code_identity与git show基点全部被锁定文件逐一重建的identity一致
  - bash语法/编译/whitespace PASS；小包叠加原源码包后原SOURCE_IDENTITY仍PASS
compatibility:
  - RT55/FE01/FE02模型、训练器、配置、采样、checkpoint/lock和源校验一字节未改
  - 原source/config/data/encoder/world身份检查与原last-only恢复完整保留
hpc_status:
  - 原M训练5/12后TIME LIMIT；30444656_4恢复失败，尚未进入新一轮
  - 本会话未提交真实超算任务；补丁真实ROCｍ/torch1.13恢复仍待用户执行
remaining_risks:
  - 仅有M组日志，其他组状态未知；完整checkpoint与逐目标结果未下载核验
  - 外加运行时兼容代码以独立receipt明确记录，审阅/汇总时一并保留
  - 作业时限不因本补丁改变；需继续分段从last恢复，不自动重提/改预算
review_request:
  - 检查只修复恢复张量设备，不重置随机状态或削弱原身份；核验补丁receipt与后续epoch
[/CODEX-RESULT]
