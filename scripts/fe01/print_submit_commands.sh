#!/usr/bin/env bash
# Only PRINTS scheduler commands, even when sbatch/srun is present.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/cluster_common.sh"
for key in FE01_PARTITION FE01_ACCOUNT FE01_QOS FE01_NODES FE01_DEVICES_PER_NODE FE01_CPUS_PER_TASK FE01_MEMORY FE01_WALLTIME; do
    value=${!key:-}
    [[ -n "$value" && "$value" != *'__FILL__'* ]] || { printf 'Missing scheduler field: %s\n' "$key" >&2; exit 2; }
done
stage=${FE01_STAGE:-formal}
case "$stage" in pilot) last=3;; formal|spatial) last=11;; *) exit 2;; esac
[[ "${FE01_INCLUDE_CONTROLS:-0}" == 0 ]] || { echo 'Controls require explicit individual index selection (runbook)' >&2; exit 2; }
log_dir="$FE01_OUTPUT_ROOT/slurm"
mkdir -p "$log_dir"
common=(sbatch --partition="$FE01_PARTITION" --account="$FE01_ACCOUNT" --qos="$FE01_QOS" \
    --cpus-per-task="$FE01_CPUS_PER_TASK" --mem="$FE01_MEMORY" --time="$FE01_WALLTIME" \
    --chdir="$FE01_CODE_ROOT" --export=ALL --output="$log_dir/%x-%A_%a.out" --error="$log_dir/%x-%A_%a.err")
printf '# Copy ONE command to submit. Audit must pass before training. Stage=%s\n' "$stage"
printf '%q ' "${common[@]}" --job-name=fe01-audit --nodes=1 --ntasks=1 --gres="$FE01_GRES_KIND:1" \
    --array="0-$last" "$FE01_CODE_ROOT/scripts/fe01/audit_job.sbatch"; printf '\n'
printf '%q ' "${common[@]}" --job-name=fe01-train --nodes="$FE01_NODES" --ntasks-per-node="$FE01_DEVICES_PER_NODE" \
    --gres="$FE01_GRES_KIND:$FE01_DEVICES_PER_NODE" --array="0-$last" "$FE01_CODE_ROOT/scripts/fe01/train_array.sbatch"; printf '\n'
for job in eval replay site_map; do
    printf '%q ' "${common[@]}" --job-name="fe01-$job" --nodes=1 --ntasks=1 --gres="$FE01_GRES_KIND:1" \
        --array=0 "$FE01_CODE_ROOT/scripts/fe01/${job}_job.sbatch"; printf '\n'
done
