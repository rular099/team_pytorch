#!/usr/bin/env bash
# PRINT ONLY. Never executes sbatch/srun or imports torch on the login node.
set -euo pipefail
: "${FE02_ENV_FILE:?Set the absolute path of a private FE02 env file}"
source "$FE02_ENV_FILE"
for key in FE02_CODE_ROOT FE02_OUTPUT_ROOT FE02_DATA_ROOT FE02_SPLIT_MANIFEST FE02_WEIGHTS_ROOT FE02_PARTITION FE02_NODES FE02_DEVICES_PER_NODE FE02_CPUS_PER_TASK FE02_WALLTIME FE02_GRES_KIND; do
    value=${!key:-}
    [[ -n "$value" && "$value" != *'__FILL__'* ]] || { echo "Missing $key" >&2; exit 2; }
done
[[ "$FE02_CODE_ROOT" = /* && "$FE02_OUTPUT_ROOT" = /* && "$FE02_ENV_FILE" = /* ]] || { echo 'Use absolute roots/env path' >&2; exit 2; }
[[ "$FE02_CODE_ROOT" != "$FE02_OUTPUT_ROOT" ]] || { echo 'Output root must be independent of source' >&2; exit 2; }
[[ -f "$FE02_WEIGHTS_ROOT/pretrained_manifest.json" ]] || { echo 'Register the real DiTing checkpoint first' >&2; exit 2; }
indices=${FE02_INDICES:-0-4}
[[ "$indices" =~ ^[0-9,-]+$ && "${FE02_ARRAY_CONCURRENCY:-1}" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid array selection/concurrency' >&2; exit 2; }
stage=${FE02_STAGE:-formal}
[[ "$stage" == formal || "$stage" == pilot ]] || { echo 'Unknown stage' >&2; exit 2; }
log_dir="$FE02_OUTPUT_ROOT/slurm";mkdir -p "$log_dir"
common=(sbatch --partition="$FE02_PARTITION" --cpus-per-task="$FE02_CPUS_PER_TASK" \
    --time="$FE02_WALLTIME" --chdir="$FE02_CODE_ROOT" --export=ALL \
    --output="$log_dir/%x-%A_%a.out" --error="$log_dir/%x-%A_%a.err")
[[ -z "${FE02_ACCOUNT:-}" ]] || common+=(--account="$FE02_ACCOUNT")
[[ -z "${FE02_QOS:-}" ]] || common+=(--qos="$FE02_QOS")
[[ -z "${FE02_MEMORY:-}" ]] || common+=(--mem="$FE02_MEMORY")
array=(--array="$indices%${FE02_ARRAY_CONCURRENCY:-1}")
single=(--nodes=1 --ntasks=1 --gres="$FE02_GRES_KIND:1")
printf '# FE02 %s. Default 0-4 = seed42 R0/A/B/C/M; 5-9=43, 10-14=44 (formal only).\n' "$stage"
printf '# Copy audit first; train only after AUDIT_PASS. No commands are being submitted.\n'
printf '%q ' "${common[@]}" --job-name=fe02-audit "${single[@]}" "${array[@]}" "$FE02_CODE_ROOT/scripts/fe02/audit_job.sbatch";printf '\n'
dependency=()
if [[ -n "${FE02_AUDIT_JOB_ID:-}" ]]; then
    [[ "$FE02_AUDIT_JOB_ID" =~ ^[0-9]+$ ]] || exit 2
    dependency+=(--dependency="aftercorr:$FE02_AUDIT_JOB_ID")
fi
printf '%q ' "${common[@]}" --job-name=fe02-train --nodes="$FE02_NODES" \
    --ntasks-per-node="$FE02_DEVICES_PER_NODE" --gres="$FE02_GRES_KIND:$FE02_DEVICES_PER_NODE" \
    "${array[@]}" "${dependency[@]}" "$FE02_CODE_ROOT/scripts/fe02/train_array.sbatch";printf '\n'
printf '# Eval only after all selected training runs have completed; --resume uses last, never best.\n'
printf '%q ' "${common[@]}" --job-name=fe02-eval "${single[@]}" "${array[@]}" "$FE02_CODE_ROOT/scripts/fe02/eval_job.sbatch";printf '\n'
printf '# Collect after seed42 fixed_val evaluations. Actual artifacts only, no test.\n'
printf '%q ' "${common[@]}" --job-name=fe02-collect "${single[@]}" "$FE02_CODE_ROOT/scripts/fe02/collect_job.sbatch";printf '\n'
