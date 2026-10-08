# FE02 本地实现证据（2026-10-08）

仓库 `rular099/team_pytorch`；基点 `ba4fa7740d9fde5fde87a7b0ae0397037209baf4`；
分支 `exp/fe02-event-fusion-readout`；cwd
`/home/zhangb/work/people/zhangbei/team_claude/team_pytorch_fe02`。
从指定base新建worktree，开始时tracked/staged/untracked均为空；FE01源分支HEAD
精确等于base（相对origin本地ahead7，不另吸收新改动）。源FE01的6项untracked
用户材料、V01的2个PPT结果目录及tmp.tar.gz、主RT55既有untracked全部保留。
FE01-A01原HEAD `3ecf060a3334d1c7b48662cf4395a3bab07723ff` 且干净；
V01 `d8f2436273fbe990e77f7151d86a6b629e99c5c1`；主RT55 `8bde65c`。

## 验证范围

最终聚焦回归 **128项全部通过、0跳过**；其中FE02专用14项。编译、8个shell/Slurm
脚本语法、git whitespace与指令原文cmp通过。没有用全量生产数据运行本地实验。

默认工厂使用指定base的真实gemini_models.py源码在独立namespace构建，
比较state_dict全部keys/shapes/values并strict load，单台与三台输入输出逐张量
完全一致。使用小合成前端隔离读出实现，**不是**生产RT55大checkpoint复评。

新五组真实下游读出已做：单台/多台、padding扰动、query反序、memory key/mask、
mapper梯度、encoder无梯度、C不调用PGA attention、B gate作用、主PGA ReLU及
不同末维与point/gaussian/MDN输出、全局辅助头结构保持、无输入/冲突拒绝。
使用实际1000维公共下游+显式mock encoder 验证同seed未改动tensor初值和PGA形状。
合成HDF5/小前端的一次CPU优化更新、state/RNG回滚、完整checkpoint/严格resume、
重载val导出、variant/probability/空间统计和不配对拒绝均为unit证据，不能作Japan成绩。

登录节点准备入口不import torch；所有20配置（正式15+可选pilot5）通过严格验证；
默认打印只有0–4(seed42)，不调用sbatch/srun、不硬写内存、不包含test。
真实前端审计、真实Japan data加载/冻结验证、DCU/DDP/吞吐/训练/评价/结果及
Slurm State/ExitCode均 **NOT_RUN / NOT_SUBMITTED**，资源在超算且用户手动运行。
本地未下载或构造1.2B模型冒充预训练验证。

原 `pga_configs/`、`configs/fe01/`、`train_light.py`、`eval_checkpoint.py`、
`gemini_util_light.py`、FE01 config/model/extractors/data/windows/analysis相对base字节不变。
FE01 engine只有默认None的显式experiment hook和仅FE02启用的证据字段/门；
pack_source的可选report-prefix默认为空，不改旧使用行为。

## 复现命令

本地解释器：`../team_pytorch_fe01/.venv-fe01/bin/python`，只读复用旧环境；
CPU测试设置 `CUDA_VISIBLE_DEVICES=''`，本机旧GPU不支持当前torch，不作为设备证据。

```bash
CUDA_VISIBLE_DEVICES='' ../team_pytorch_fe01/.venv-fe01/bin/python -m unittest -v \
  tests.test_fe02_event_fusion tests.test_fe02_analysis \
  tests.test_causal_random_geometry tests.test_scheduler_checkpoint_resume \
  tests.test_eval_checkpoint_formal tests.test_query_geometry_diagnostics \
  tests.test_rt57_gtnp_v2_station_distinctive tests.test_rt58_waveform_anchor_transfer \
  tests.test_rt59_dual_objective_transport tests.test_rt60_contrast_readout tests.test_rt61_wave_geometry \
  tests.test_fe01_time_sampling tests.test_fe01_native_windows tests.test_fe01_statistics \
  tests.test_fe01_causal_replay
python -m compileall -q fe02 scripts/fe02 fe01/engine.py gemini_models.py scripts/fe01/pack_source.py
bash -n scripts/fe02/env.sh scripts/fe02/job.sh scripts/fe02/print_submit_commands.sh \
  scripts/fe02/pack_source.sh scripts/fe02/audit_job.sbatch scripts/fe02/train_array.sbatch \
  scripts/fe02/eval_job.sbatch scripts/fe02/collect_job.sbatch
git diff --check
```

量化验证状态见 [verification.json](verification.json)。没有FE02预测图/精度指标。
