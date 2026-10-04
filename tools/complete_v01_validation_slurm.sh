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
