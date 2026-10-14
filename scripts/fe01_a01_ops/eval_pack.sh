#!/usr/bin/env bash
# LOGIN: prepare | eval_first | eval_diting_off | pack. One explicit stage only.
set -Eeuo pipefail
mode=${1:?Usage: prepare | eval_first | eval_diting_off | pack}
tool_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ "$mode" != _* ]]; then
  if [[ -f "$tool_dir/SHA256SUMS" ]]; then
    (cd "$tool_dir" && sha256sum -c SHA256SUMS)
  fi
  log="$tool_dir/eval_pack_${mode}_$(date +%Y%m%dT%H%M%S)_$$.log"
  (umask 077;set -o noclobber;: > "$log")
  exec > >(tee -a "$log") 2> >(tee -a "$log" >&2)
  printf 'SUBMISSION_LOG=%s\n' "$log" >&2
  if [[ "$mode" == prepare ]]; then
    : "${A01_ENV_FILE:?Set A01_ENV_FILE to the existing private settings}"
    source "$A01_ENV_FILE"
    exec "${A01_CONFIG_PYTHON:-python3}" "$tool_dir/eval_pack.py" prepare \
      --project "${A01_PROJECT_ROOT:-$(dirname "$tool_dir")}" --tool-dir "$tool_dir" --env-file "$A01_ENV_FILE"
  fi
  session=${A01_EVAL_SESSION:-}
  if [[ -z "$session" ]]; then
    [[ -f "$tool_dir/eval_session.path" ]] || { echo 'Run prepare first.' >&2;exit 2; }
    IFS= read -r session < "$tool_dir/eval_session.path"
  fi
  export A01_EVAL_SESSION="$session"
fi
: "${A01_EVAL_SESSION:?}"
export A01_ENV_FILE="$A01_EVAL_SESSION/runtime.private.env"
source "$A01_ENV_FILE"
shared="$A01_EVAL_SESSION/tools/eval_pack.sh"
helper="$A01_EVAL_SESSION/tools/eval_pack.py"
case "$mode" in
  _rank_eval)
    # Initialize DCU inside the actual srun step, never in the batch shell.
    source "$A01_CODE_ROOT/scripts/fe01_a01/env.sh"
    export A01_WORLD_SIZE=1
    exec bash "$A01_CODE_ROOT/scripts/fe01_a01/rank_exec.sh" "$A01_PYTHON" "$helper" evaluate \
      --session "$A01_EVAL_SESSION" --index "${SLURM_ARRAY_TASK_ID:?}"
    ;;
  _job_eval)
    : "${SLURM_JOB_ID:?}" "${SLURM_ARRAY_TASK_ID:?}"
    exec 9>"$A01_EVAL_SESSION/index${SLURM_ARRAY_TASK_ID}.lock"
    flock -n 9 || { echo 'This index already has an evaluation running.' >&2;exit 2; }
    srun --nodes=1 --ntasks=1 --gres="$A01_SINGLE_GRES" --kill-on-bad-exit=1 bash "$shared" _rank_eval
    exit 0
    ;;
  _job_pack)
    : "${SLURM_JOB_ID:?}"
    exec 9>"$A01_EVAL_SESSION/pack.lock"
    flock -n 9 || { echo 'Pack already running.' >&2;exit 2; }
    # Stdlib/config/package helpers only: this job requests CPU, no DCU.
    exec "${A01_CONFIG_PYTHON:-python3}" "$helper" pack --session "$A01_EVAL_SESSION"
    ;;
  eval_first|eval_diting_off|pack) ;;
  *) echo 'Choose prepare, eval_first, eval_diting_off or pack.' >&2;exit 2;;
esac
indices=$("${A01_CONFIG_PYTHON:-python3}" "$helper" preflight --session "$A01_EVAL_SESSION" --stage "$mode")
if [[ "$mode" != pack && -z "$indices" ]]; then
  echo 'All indices in this stage already completed; no sbatch submitted.'
  exit 0
fi
args=(--parsable --export=ALL --chdir="$A01_CODE_ROOT" --job-name="a01_$mode" --nodes=1 \
  --ntasks=1 --cpus-per-task=4 --output="$A01_EVAL_SESSION/slurm/${mode}_%A_%a.out" \
  --error="$A01_EVAL_SESSION/slurm/${mode}_%A_%a.err")
[[ -z "${A01_PARTITION:-}" ]] || args+=(--partition="$A01_PARTITION")
[[ -z "${A01_EXCLUDE_NODES:-}" ]] || args+=(--exclude="$A01_EXCLUDE_NODES")
if [[ "$mode" == pack ]]; then
  args+=(--time="${A01_PACK_TIME:-02:00:00}" --mem="${A01_PACK_MEM:-8G}")
  action=_job_pack
else
  args+=(--array="$indices%4" --gres="$A01_SINGLE_GRES" --mem="${A01_SINGLE_MEM:-64G}" --time="${A01_EVAL_TIME:-24:00:00}")
  action=_job_eval
fi
job=$(sbatch "${args[@]}" "$shared" "$action")
printf '%s\t%s\t%s\n' "$mode" "$indices" "$job" >> "$A01_EVAL_SESSION/jobs.tsv"
printf 'SUBMITTED stage=%s indices=%s job=%s\nLogs=%s/slurm\nResults=%s/results\nPack=%s/review.tar.gz\n' \
  "$mode" "$indices" "$job" "$A01_EVAL_SESSION" "$A01_EVAL_SESSION" "$A01_EVAL_SESSION"
