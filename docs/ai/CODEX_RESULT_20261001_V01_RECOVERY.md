# V01 失败诊断与最小补跑交付

日期：2026-10-01（Asia/Shanghai）

2026-10-03 更新：本文保留当时的交付状态作为历史记录。上述修复已归档为
`8829029f7a1a3ef49cedeccef9385298c683ac01`；用户已回传三臂真实 epoch-8 权重与五份
random 结果，normal 五份发生新的 generator 错误。当前入口为
`CODEX_RESULT_20261003_V01_PARTIAL_VALIDATION.md`，不要重复执行本文的整体 recovery。

## 版本和边界

- repo: `rular099/team_pytorch`
- branch: `exp/v01-velocity-prep-padding-control`
- base commit: `8f54477ed11bf2623f40121648f7fcc756ada1a2`
- 本轮是本地工作树修复；尚未 commit/push，也未代用户提交超算任务。
- 没有修改模型、DiTing、训练/验证划分、PGA 标签、loss、V01 配置、seed、LR 或 epoch 数。

## 已核验事实

1. 用户报告 `v01-preflight`、`v01-vmissing-train`、`v01-apair-train` 完成。
   用户回传文件清单确认后两臂存在 `full_model_init.pth`（225399883 bytes）、
   `full_model_last.pth` 和 `full_model_best.pth`（各 547975895 bytes）。
   这证明文件存在，不代替 checkpoint epoch 和内容审计。
2. `../chaosuan_res/v01-vmissing-vfull-normal-28899994.out` 确认该评估读取的 checkpoint
   epoch 为 8；其 `.err` 显示启动脚本原始 JSON 读取 `weight_path` 时发生 KeyError。
   原验证配置通过 `extends` 继承该字段，启动器没有解析继承。
3. `../chaosuan_res/v01-vfull-train-28899987.err` 显示 rank 0 在构建 2004 年台站 CSV
   缓存时，`os.replace(tmp_path, event_metadata_path)` 发生 FileExistsError。
   `.out` 停在 Loading data，尚未建立模型或开始 epoch。该次运行无进度可续训。
4. vfull/vmissing 原来使用同一 velocity HDF5 和 metadata_cache 目录；跨作业并发发布缓存
   是上述 EEXIST 的高概率原因，但没有宿主文件系统跟踪证据，不声称已证明底层 FS 行为。
5. 原 vfull random 的两份上传文件只有 `File does not exist!` JSON，并非有效 Slurm 日志。

## 修复

- `tools/launcher_config.py`：Torch-free JSON/YAML 继承、深合并、最终环境展开；
  仅遇到 YAML 时导入 PyYAML。JSON 路径在 Python `-S` 环境也通过验证。
- 两个通用 shell launcher 通过该工具读 weight_path 和 single_station enabled，
  不修改训练/评估 Python 的模型路径。
- `loader_light.py`：EEXIST 时仅在现有 CSV 与临时 CSV 字节一致的情况下接受并复用；
  内容冲突、权限等异常仍报错，临时文件清理保持原流程。
- V01 launcher：各训练臂、各 checkpoint/view/protocol 验证使用独立的轻量台站缓存和
  resolved config；只覆盖 metadata_cache_dir。原始 JSON、derived waveform cache 不变。
- 验证配置副本目录按 checkpoint/view/protocol 分离，避免十个任务覆盖同一个 config.json。
- 新 `ACTION=recover`：跳过 preflight，只训练 vfull，复用两臂已有 checkpoint，运行原定
  10 个 validation，再 afterok 汇总。不新增 smoke 或 held-out test。
- 训练/验证输出冲突在任何 sbatch 前检查；恢复入口使用提交预留目录阻止重复点击，记录 IDs。

## 提交入口

使用 `bash tools/recover_v01_prep_padding_controls_slurm.sh`，不是对该入口直接 sbatch。
登录节点不需要 torch；checkpoint epoch 校验在计算节点执行。

默认路径（位于原 V01_RUN_ROOT 下）：

```text
复用：derived_cache、weights_vmissing、weights_apair
保留：weights_vfull、eval、report、原 logs
新增：weights_vfull_retry1、eval_retry1、report_retry1
记录：submissions/recover_retry1/submitted_jobs.txt
```

运行步骤：先把补丁包解压到原 `_vel` 代码目录。覆盖的是代码，不是权重或数据。

```bash
cd /public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_query_geometry_diagnostics_vel
tar -xzf v01_recovery_patch_20261001.tar.gz
export WORKDIR=$PWD
unset EXPECTED_SOURCE_MANIFEST_SHA256
DRY_RUN=1 bash tools/recover_v01_prep_padding_controls_slurm.sh

# 确认打印的 hash 与下方一致；否则停下检查上传内容，不要绕过核验。
export EXPECTED_SOURCE_MANIFEST_SHA256=a06164c0243ac920d78202e2d1455226f8a6f9103117c28de9337739e0d94592
CONFIRM_V01=1 DRY_RUN=0 bash tools/recover_v01_prep_padding_controls_slurm.sh
```

应该只打印 1 个 train、10 个 eval、1 个 analyze job ID。vfull 的四个验证等待其训练成功；
已有 vm/apair 的六个验证可同步开始；汇总等待十个验证全部成功。默认资源不变。

不要再次 ACTION=all，不要删除旧 weights_vfull 或 metadata_cache，不要重新提交两臂训练。
预留目录存在时不要直接删目录重提：先查 submitted_jobs.txt 和 squeue/sacct，确认没有运行
作业或重复提交。预留目录也可能代表部分提交失败，不等于 12 个作业都提交成功。
若只想补单独阶段，入口支持 ACTION=train|eval|analyze；train 默认只选择 vfull。
显式续训要使用 RESUME_V01=1，且已有 retry1/full_model_last.pth；8 是总 epoch 目标，
不是再追加 8 个 epoch。部分完成的验证不要直接覆盖，应保留产物、根据实际状态另定补跑。

## 本地验证

- `python -m unittest discover -s tests -p 'test_*.py'`: **130 tests PASS**。
- 新增 12 项测试覆盖配置解析一致性、无 torch CLI、缓存竞态、冲突保留、精确任务图、
  afterok、独立缓存、已有输出保护和重复提交预留。正式 sbatch 分支使用 mock，不是真实超算。
- 四个 shell 文件 `bash -n`: PASS。
- 修改 Python 文件 `py_compile`: PASS。
- `git diff --check`: PASS。
- `gemini_models.py` SHA-256 仍为
  `cd8481286436342d09781888967dc757f4bde383f4d344033d866c6b06b7be84`；
  `train_light.py`、`eval_checkpoint.py` 和全部 V01/RT55 模型配置未改。
- 新 source manifest:
  `a06164c0243ac920d78202e2d1455226f8a6f9103117c28de9337739e0d94592`。

## 仍待超算验证

真实 FS、Slurm、ROCm/PyTorch 1.13 环境下的补跑尚未执行。已有 apair checkpoint 的 epoch
仅有用户 COMPLETED 报告，补跑评估会再次强制核验 epoch 8；没有本地真实 checkpoint body。
本轮不增加研究干预、不查看 held-out test，不将本地测试声称为正式训练成功。

```text
[CODEX-RESULT]
task_id: 20261001-v01-recovery
base_commit: 8f54477ed11bf2623f40121648f7fcc756ada1a2
result_commit: none (local worktree only)
branch: exp/v01-velocity-prep-padding-control
changed_files:
  - loader_light.py: guarded identical-cache reuse on EEXIST
  - train_light_slurm.sh and eval_checkpoint_slurm.sh: inheritance-aware configuration reads
  - tools/launcher_config.py: torch-free configuration resolver
  - tools/run_v01_prep_padding_controls_slurm.sh: isolated caches, recover action, output/submission guards
  - tools/recover_v01_prep_padding_controls_slurm.sh: minimal recovery entrypoint
  - tests/test_v01_recovery_launchers.py and project docs
verification:
  - full unittest suite: 130 tests PASS; syntax/compile/diff checks PASS
compatibility:
  - RT55 configuration-loader equality and existing regression suite PASS; model/runtime configs unchanged
hpc_status:
  - not submitted; user executes the supplied recovery entrypoint
remaining_risks:
  - actual HPC FS/Slurm behavior and checkpoint provenance need runtime confirmation
review_request:
  - review cache conflict handling, scientific-contract preservation, and exact recovery dependencies
[/CODEX-RESULT]
```
