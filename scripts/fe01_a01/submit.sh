#!/usr/bin/env bash
# User-invoked LOGIN NODE entry. Submits exactly one chosen stage, never chains.
set -euo pipefail
stage=${1:?Usage: bash scripts/fe01_a01/submit.sh STAGE [INDEX_OR_RANGE]}
indices=${2:-0-4}
code=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "${A01_ENV_FILE:-$code/a01.private.env}"
case "$stage" in environment|audit|diagnostics|pilot|train|eval|pack) ;; *) echo 'Invalid stage' >&2;exit 2;;esac
[[ "$indices" =~ ^[0-4](-[0-4])?$ ]] || { echo 'Indices must be 0..4 or ascending range, e.g. 0-2' >&2;exit 2; }
[[ "${indices:0:1}" -le "${indices: -1}" ]] || { echo 'Descending range forbidden' >&2;exit 2; }
[[ "$A01_CODE_ROOT" == "$code" ]] || { echo 'Code root mismatch' >&2;exit 2; }
mkdir -p "$A01_OUTPUT_ROOT/slurm"
args=(--parsable --chdir="$code" --export=ALL --job-name="a01_$stage" --output="$A01_OUTPUT_ROOT/slurm/${stage}_%A_%a.out" --error="$A01_OUTPUT_ROOT/slurm/${stage}_%A_%a.err")
[[ -z "${A01_PARTITION:-}" ]] || args+=(--partition="$A01_PARTITION")
if [[ "$stage" == train ]]; then
  # Three tasks at 4 nodes each use at most12 nodes. Remaining tasks wait.
  args+=(--array="$indices%3" --nodes="$A01_TRAIN_NODES" --ntasks-per-node="$A01_DEVICES_PER_NODE" --cpus-per-task="$A01_CPUS_PER_TASK" --gres="$A01_GRES" --time="$A01_TRAIN_TIME")
  args+=(--mem="${A01_TRAIN_MEM:-64G}")
else
  args+=(--nodes=1 --ntasks=1 --cpus-per-task=4 --gres="$A01_SINGLE_GRES" --time="$A01_GATE_TIME")
  args+=(--mem="${A01_SINGLE_MEM:-64G}")
  [[ "$stage" == environment || "$stage" == pack ]] || args+=(--array="$indices%4")
fi
job=$(sbatch "${args[@]}" "$code/scripts/fe01_a01/job.sbatch" "$stage")
printf 'SUBMITTED stage=%s index=%s job=%s\nOutputs=%s\n' "$stage" "$indices" "$job" "$A01_OUTPUT_ROOT"
printf '%s\t%s\t%s\n' "$stage" "$indices" "$job" >> "$A01_OUTPUT_ROOT/jobs.tsv"
