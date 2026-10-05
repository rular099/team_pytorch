# CODEX-RESULT：V01 closure HDF5 标量读取修复

```text
[CODEX-RESULT]
task_id: 20261005-v01-closure-scalar-fix
base_commit: 632cc63ae82d9c29c238918d4a1ffa5ec92aa3b2
result_commit: 本文所在提交；最终回复和 git log 提供精确40位 SHA
branch: exp/v01-velocity-prep-padding-control
changed_files:
  - gemini_util_light.py: 仅 closure describe_request 的标量读取改用 [()]
  - tools/audit_v01_validation_contract.py: plan_and_sidecar 使用相同 reader
  - tools/v01_validation_contract.py: scalar/singleton reader，拒绝多值/非整数/非有限参考点
  - tests/test_v01_validation_closure.py: fixture 对齐 builder 的 0-D int64，新增3项回归
  - tools/v01_closure_source_manifest.json: 更新3个 runtime 文件的 SHA；共180个文件
  - docs/v01_validation_closure.md / docs/ai/README.md / SESSION_SUMMARY.md: 失败记录和 retry1 操作
verification:
  - before runtime fix: production scalar fixture reproduced user's exact ValueError in 2 tests
  - focused closure + analysis unittest: PASS, 24 tests
  - full unittest discovery: PASS, 159 tests, 0 skipped
  - production generator idx7 mechanism trace with scalar fixture: PASS
  - real evaluator -> NPZ -> plan/identity sidecar -> report rows with scalar fixture: PASS
  - scalar/singleton same request/cutoff/input/label tensors: PASS
  - scalar requested cutoff equals actual generator cutoff at 1/3/5s: PASS
  - malformed reference metadata aborts without silently moving cutoff: PASS
  - login prepare through python -S: PASS (included in focused/full tests)
  - py_compile / Python3.8 AST grammar / bash -n / git diff --check: PASS
  - all180 runtime source hashes: PASS
compatibility:
  - RT55 default code path unchanged; reader imported only inside explicit closure path
  - legacy _get_one already reads scalar via [()]; no change made to its clock or training semantics
  - model/train/eval checkpoint entry/loss/DiTing/config/cache builder bytes unchanged from base
  - existing RT55–RT61 regression suite PASS; no checkpoint/cache/result rewritten
hpc_status:
  - user reports original CPU audit FAILED; job IDs and dependency states not provided
  - fixed code NOT SUBMITTED by Codex; existing launcher supplied for user retry
remaining_risks:
  - real HPC derived cache idx7 gate and downstream metrics pending; synthetic trace is not certification
  - old pending afterok jobs may remain DependencyNeverSatisfied; inspect exact IDs before patching source
  - old source snapshot must remain unchanged; prepare new validation_closure_v1_retry1
  - Python3.8 grammar was checked, not actual HPC Py3.8/torch1.13 execution
review_request:
  - review fix against builder's scalar dataset schema, two read sites, and same cutoff contract
  - retain validation-only scope and existing epoch8 weights; do not request retraining or cache rebuild
  - wait for real closure result package before drawing new normal or paired A/V conclusions
[/CODEX-RESULT]
```

## 根因与边界

`tools/build_v01_paired_manifest.py` 写入：

```python
event_group.create_dataset(
    "v01_reference_p_pick", data=np.asarray(int(source_picks.min()), dtype=np.int64)
)
```

这是事件级标量，shape=()。10-04 新增审计两次使用 `[0]`，h5py 禁止对标量做这种索引。
原 generator 读元数据使用 `[()]`，因此旧训练成功不矛盾；数据没有坏，不需重建。
之前测试误用 shape=(1,) 掩盖了错误；本轮 fixture 改为真实 schema，先复现再修复。
所有 `v01_reference_p_pick` 使用处已检查； `_get_one` 中 `data[key][0]` 是事件 list 的索引，
不是 HDF5 dataset 索引，不应机械替换。

xFormers 警告不是该 traceback 的原因；不要求重装依赖。本轮不改变时钟值、采样/协议、
固定三臂 epoch8/step1496 的权重、split、PGA 标签或旧 NPZ。

## 补丁与重新提交

`../v01_closure_scalar_fix_20261005.tar.gz` 是基于上述 base 的小补丁包：
仅三个修复的 runtime Python、更新 manifest、测试与文档；不含 checkpoint、cache、
旧输出或未跟踪文件。解压到原 `_vel` 代码根目录。若 prepare 报其他源文件不匹配，
需同步完整代码而非修改 manifest 绕过校验。

新 source manifest SHA256：
`88d4570c2f87c3f0daf00eece6e87dc0a733c78a4d86963645152d1538d9fc86`。

先在超算检查原失败流水线的 `validation_closure_v1/submitted_jobs.tsv`，
用其中列出的明确 ID 查看 squeue/sacct；若仍有依赖失败的挂起 closure 作业，
只取消这些 ID，不取消其他任务。不重新 release/requeue 旧 job，它绑定旧源码快照。

将补丁包上传到代码根目录，随后在登录节点运行：

```bash
cd /public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_query_geometry_diagnostics_vel
tar -xzf v01_closure_scalar_fix_20261005.tar.gz
export WORKDIR="$PWD"
export V01_RUN_ROOT="$WORKDIR/v01_velocity_prep_padding_seed42"
export V01_CLOSURE_ROOT="$V01_RUN_ROOT/validation_closure_v1_retry1"
export DITING_CONFIG="$WORKDIR/diting/config/diting_1200m_backbone_attnpool.yml"
export DITING_PRETRAINED="/public/home/test_bigmodel/seismogram/mx/results/scaling_diting_1b/scaling_diting_1200M/checkpoint_pt_epoch_70/mp_rank_00_model_states.pt"

INCLUDE_AA_RANDOM=1 bash tools/complete_v01_validation_slurm.sh
DRY_RUN=0 CONFIRM_V01_CLOSURE=1 INCLUDE_AA_RANDOM=1 \
  bash tools/complete_v01_validation_slurm.sh
```

仍是 CPU audit -> 5 normal + 1必要 AA random -> CPU analysis；
四格速度 random 复用，三臂训练和 preflight 均不提交。
若 retry1 目录已经存在，应先检查，另选新目录；不要删除或覆盖原失败证据。
无需登录节点 torch 或 `.git`。后续运行前/后权重哈希门禁保持不变。

作业成功后回传 `validation_closure_v1_retry1`（排除 metadata_cache）和 sacct 文本。
若 audit 再失败，回传其 `.out/.err` 与生成的 `audit/*.idx7_trace.json`，不要绕过门禁。
