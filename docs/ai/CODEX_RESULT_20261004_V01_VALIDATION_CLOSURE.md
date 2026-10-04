# CODEX-RESULT：V01 validation closure 实施与待超算执行

```text
[CODEX-RESULT]
task_id: 20261004-v01-validation-closure
base_commit: 74c55aa5442b4200961c88ceee3d11af5033275d
result_commit: 本文所在提交；精确40位SHA在最终回复与git log中提供
branch: exp/v01-velocity-prep-padding-control
changed_files:
  - gemini_util_light.py / eval_checkpoint.py: validation-only clock copy、strict request ledger、sensor IDs/UTC
  - tools/audit_v01_validation_contract.py: 实际idx7门禁、cache/身份/旧random/AA审计，torch-free登录准备
  - tools/analyze_v01_validation_closure.py: 闭环统计、分层CI、MSE分解、outer match、coverage与可复算图
  - tools/complete_v01_validation_slurm.sh / v01_validation_closure_job.sh: 只补验证、默认5 normal、条件性1 AA random
  - tools/v01_validation_contract.py / v01_closure_checkpoint_guard.py / v01_closure_source_manifest.json: 固定权重、来源/请求合同
  - tests/test_v01_validation_closure.py / test_v01_closure_analysis.py: 21个新增回归
  - docs/v01_validation_closure.md: 完整运行/回传步骤；本文件附完整提交脚本
  - reports/v01_validation_closure_20261004/: AA不可去重证据、纯离线P1统计/图/源计数
verification:
  - unittest discover tests: PASS, 156 tests, 0 skipped (135 baseline + 21 new)
  - py_compile: PASS, 2 production entry points + 4 new Python modules
  - Python3.8 ast grammar: PASS, 6 modified/new runtime Python files; not actual Python3.8 execution
  - bash -n two new Slurm scripts: PASS
  - default dry-run 5 normal / exception dry-run 6 eval + dependency graph: PASS; no sbatch executed
  - python -S login prepare with fixture weights/configs: PASS; no torch/numpy required
  - 180 runtime-source SHA manifest and two original prompt byte cmp: PASS
  - real evaluator -> NPZ -> production cache sensor identity -> report table fixture roundtrip: PASS
  - local AA duplicate audit: PASS, 15 nonidentical groups / 45 extra rows, original SHA preserved
  - offline P1: completed on 4 original velocity random NPZ, 5000 event-cluster draws, seed20260915
  - actual HPC cache idx7 trace/full model/normal inference: NOT RUN (cache only on HPC)
compatibility:
  - RT55 model/DiTing/train/loss/config/checkpoint layout unchanged; gemini_models.py/train_light.py bytes unchanged
  - legacy alias/substitution remain defaults; new flag rejects realtime training and test/overfit reads
  - RT55/RT56 config, tiny checkpoint-load/forward, realtime/padding, RT57–RT61 regressions all PASS
  - V01 production random/mixed-enabled fixture inputs/labels exactly equal with closure switch off/on
  - mixed training remains original path; no saved weight or old result was written
hpc_status:
  - NOT SUBMITTED; no job IDs and no new HPC metrics
  - deliver user-run evaluation-only orchestrator; CPU audit afterok gates all GPU evals
remaining_risks:
  - actual idx7 root cause not yet certified; synthetic mechanism tests are not real failure evidence
  - velocity random physical sensor/absolute UTC certification pending HPC sidecars; local P1 uses verified exact coordinates/time/labels/current/first pick
  - AA strict availability/missing denominator pending real cache; nonidentical duplicate outputs cannot be deduplicated
  - encoder/parent/source current hashes cannot be claimed as recorded historical hashes
  - sensor-ID remote exclusion is only proxy, not certified true physical-site separation
  - validation reused for development; one seed/8 epochs; A/V different input systems and selection bias
  - inherited full-record centering and cutout+1 unchanged; no strict-online causality certification
review_request:
  - Review explicit V01-only scope, no-substitution ledger, actual idx7 gate, old random signature audit and AA one-cell exception
  - Review offline above-threshold deterioration + bias/variance decomposition; do not infer regularization/padding causality
  - Wait for closure artifact package before claiming normal/A/V matrix completion or proposing additional experiments
[/CODEX-RESULT]
```

## 已核实的纯离线 P1 结果

全部为已有 **validation random**、fixed epoch8/step1496、`log10(m/s²)`。
主比较 MM−FF（vmissing 模型/视图减 vfull 模型/视图），没有 bias correction。
原四格在 exact query coordinates + event/requested time 上一对一 outer merge，
truth/current/first-pick 完全一致；每组匹配 84,259，左右 unmatched=0。
这不是 sensor-ID/absolute-UTC 认证，后者仍须真实 cache gate。

| 群体 / metric | MM−FF | 95% event-cluster CI | targets / events |
| --- | ---: | --- | --- |
| all7 MAE | -0.007620 | [-0.008285, -0.006954] | 84,259 / 1,194 |
| all7 RMSE | -0.006504 | [-0.007293, -0.005698] | 84,259 / 1,194 |
| all7 NLL | -0.030090 | [-0.033578, -0.026539] | 84,259 / 1,194 |
| all7 Brier | -0.008153 | [-0.008984, -0.007359] | 84,259 / 1,194 |
| above -1.2 MAE | +0.009195 | [+0.008323, +0.010055] | 35,612 / 1,051 |
| above -1.2 NLL | +0.041495 | [+0.037650, +0.045329] | 35,612 / 1,051 |
| above -1.2 Brier | +0.019042 | [+0.018029, +0.020109] | 35,612 / 1,051 |
| below -1.2 MAE | -0.019930 | [-0.020778, -0.019105] | 48,647 / 1,158 |

“above”指目标 PGA >= -1.2，不是大震事件。探索性分层 CI 没有多重比较修正，不用于调参数。

```text
MSE = bias² + centered error variance
```

原始 LaTeX：`\mathrm{MSE}=\mathrm{bias}^{2}+\mathrm{Var}(\hat y-y)`。

| MSE 分解 | FF | MM | MM−FF 的 95% CI |
| --- | ---: | ---: | --- |
| MSE | 0.096277 | 0.092284 | [-0.004473, -0.003511] |
| bias² | 0.008384 | 0.004326 | [-0.004403, -0.003698] |
| centered error variance | 0.087893 | 0.087958 | [-0.000243, +0.000362] |

总体 MSE 减少主要来自 bias² 减少，不能描述为误差云明显收紧、删除前缀普遍有益、
已确认正则化机制或 padding 主瓶颈。模型原预测未做任何偏差校正。

文件位于 `reports/v01_validation_closure_20261004/offline_random/`：
`metrics_by_stratum.csv`、`paired_event_cluster_ci_and_mse_decomposition.csv`、
`primary_ff_mm_event_sufficient_statistics.csv`（主比较全部五分层、all7/early135可复算）、
source SHA、outer-match audit、fixed-axis scatter-density/residual SVG/PNG 和全部 bin counts。
完整多比较 event sufficient table 是本地产物，不强推 45MB CSV 到 Git。

图表按 figure-designer 指引使用共同固定坐标、连续密度色标、残差双编码和可复算分母；
当前实际 random 图已检查字体、单位、坐标、计数，统计中无视野外目标。
残差图 legend 可能局部靠近峰顶，属于演示排版小问题，不影响数值；不是论文定稿。
normal 图须等 HPC 结果，不冒称已经产出。

## 如何执行、回传

完整可复制命令、已有数据/权重绝对路径、CPU审计门禁、GPU单卡资源、默认不申请固定内存、
唯一目录和结果回传打包步骤见 `docs/v01_validation_closure.md`。
当前建议 `INCLUDE_AA_RANDOM=1`（已核实不可去重的唯一例外），由用户手动执行。
仅传代码不重传 raw archive/weights；必须使用新 source manifest 对应的源码。

## 完整提交脚本

以下为 `tools/complete_v01_validation_slurm.sh` 的完整内容（运行时还需要新 compute job
runner、Python合同/审计/分析模块及 source manifest，所以请上传完整交付源码，不只复制这个脚本）：
```bash
#!/usr/bin/env bash
# Run with bash on the login node. Evaluation only; no preflight or training.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export WORKDIR=${WORKDIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
export V01_RUN_ROOT=${V01_RUN_ROOT:-$WORKDIR/v01_velocity_prep_padding_seed42}
export V01_CLOSURE_ROOT=${V01_CLOSURE_ROOT:-$V01_RUN_ROOT/validation_closure_v1}
export DITING_CONFIG=${DITING_CONFIG:-$WORKDIR/diting/config/diting_1200m_backbone_attnpool.yml}
export CONDA_ENV=${CONDA_ENV:-lsm_env}
export MODULE_UNLOAD=${MODULE_UNLOAD:-compiler/rocm/2.9}
export MODULE_LOADS=${MODULE_LOADS:-"compiler/rocm/dtk-23.04 apps/miniconda/3"}
DRY_RUN=${DRY_RUN:-1}
INCLUDE_AA_RANDOM=${INCLUDE_AA_RANDOM:-0}
[[ "$DRY_RUN" =~ ^[01]$ && "$INCLUDE_AA_RANDOM" =~ ^[01]$ ]] || { echo 'Flags must be 0 or 1.' >&2; exit 2; }
if [[ "$DRY_RUN" == 0 && "${CONFIRM_V01_CLOSURE:-0}" != 1 ]]; then
    echo 'Formal submission requires DRY_RUN=0 CONFIRM_V01_CLOSURE=1.' >&2; exit 2
fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo 'Use bash on the login node, not sbatch for this orchestrator.' >&2; exit 2
fi
cells=(vfull__vfull__normal vfull__vmissing__normal vmissing__vfull__normal vmissing__vmissing__normal apair__apair__normal)
prepare=(python "$WORKDIR/tools/audit_v01_validation_contract.py" prepare --run-root "$V01_RUN_ROOT" --output "$V01_CLOSURE_ROOT")
if [[ "$INCLUDE_AA_RANDOM" == 1 ]]; then
    cells+=(apair__apair__random)
    prepare+=(--include-aa)
    echo '[AA exception] 15 duplicate groups / 45 extra rows have nonidentical predictions; one AA random rerun is necessary.'
fi
base=(sbatch --parsable --partition="${SLURM_PARTITION:-diting}" --nodes=1 --ntasks=1
      --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" --chdir="$WORKDIR" --export=ALL
      --output="$V01_CLOSURE_ROOT/logs/%x-%j.out" --error="$V01_CLOSURE_ROOT/logs/%x-%j.err")
[[ -z "${SLURM_ACCOUNT:-}" ]] || base+=(--account="$SLURM_ACCOUNT")
# Omit memory by default: the previous site's fixed request was unsatisfiable.
[[ -z "${SLURM_MEM:-}" ]] || base+=(--mem="$SLURM_MEM")
submit() {
    local name=$1 stage=$2 dependency=$3 cell=${4:-}
    local command=("${base[@]}" --job-name="$name")
    if [[ "$stage" == eval ]]; then
        command+=(--time="${EVAL_TIME:-23:50:00}" --gres="${SLURM_GRES_RESOURCE:-dcu}:1")
    else
        command+=(--time="${AUDIT_TIME:-04:00:00}")
    fi
    [[ -z "$dependency" ]] || command+=(--dependency="afterok:$dependency")
    command+=("$WORKDIR/tools/v01_validation_closure_job.sh" "$stage" "$cell")
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '[DRY-RUN] ' >&2; printf '%q ' "${command[@]}" >&2; printf '\n' >&2
        printf 'dry_%s\n' "$name"
    else
        local reply job_id
        reply=$("${command[@]}")
        job_id=${reply%%;*}
        [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid job ID: $reply" >&2; exit 1; }
        printf '%s\t%s\t%s\n' "$name" "$job_id" "$dependency" >> "$V01_CLOSURE_ROOT/submitted_jobs.tsv"
        printf '%s\n' "$job_id"
    fi
}
if [[ "$DRY_RUN" == 1 ]]; then
    printf '[DRY-RUN] '; printf '%q ' "${prepare[@]}"; printf '\n'
else
    : "${DITING_PRETRAINED:?Export the existing DiTing checkpoint path; do not download another encoder.}"
    export DITING_PRETRAINED
    [[ -f "$DITING_CONFIG" && -f "$DITING_PRETRAINED" ]] || { echo 'DiTing inputs not found.' >&2; exit 1; }
    "${prepare[@]}"
fi
audit_id=$(submit v01-closure-audit audit '')
ids=()
for cell in "${cells[@]}"; do
    ids+=("$(submit "v01-close-$cell" eval "$audit_id" "$cell")")
done
dependencies=$(IFS=:; echo "${ids[*]}")
analysis_id=$(submit v01-closure-analysis analyze "$dependencies")
printf '[V01 closure] audit=%s eval=%s analysis=%s dry_run=%s\n' "$audit_id" "$dependencies" "$analysis_id" "$DRY_RUN"
printf 'Output: %s\n' "$V01_CLOSURE_ROOT"
if [[ "$INCLUDE_AA_RANDOM" == 0 ]]; then
    echo 'AA random omitted by default. Its old export is NOT valid for paired A/V comparison; use INCLUDE_AA_RANDOM=1 for the verified nonidentical-duplicate exception.'
fi
```
