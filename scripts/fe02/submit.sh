#!/usr/bin/env bash
# Login node: submit directly. Compute node: execute the complete run.
set -euo pipefail

if [[ "${1:-}" == --worker ]]; then
    [[ $# == 3 ]] || { echo 'Missing launch environment/hash' >&2; exit 2; }
    : "${SLURM_JOB_ID:?Run bash scripts/fe02/submit.sh to submit}"
    [[ "$(sha256sum -- "$2" | cut -d ' ' -f 1)" == "$3" ]] || {
        echo 'Launch environment changed after command generation' >&2; exit 2;
    }
    # Slurm copies this file to its spool: BASH_SOURCE is NOT the code path.
    source "$2"
    export FE02_ENV_FILE="$2"
    exec bash "${FE02_CODE_ROOT:?}/scripts/fe02/job.sh" run
fi
dry_run=0
if [[ $# == 1 && "$1" == --dry-run ]]; then
    dry_run=1
elif [[ $# != 0 ]]; then
    echo 'Usage: bash scripts/fe02/submit.sh [--dry-run]' >&2; exit 2
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

# No Python, torch, modules, or checkpoint hashing on the login node.
if [[ "$dry_run" == 0 ]]; then
    command -v sbatch >/dev/null || { echo 'sbatch unavailable: run this on the supercomputer login node' >&2; exit 2; }
fi
export FE02_CODE_ROOT="$root"
source "${FE02_ENV_FILE:-$root/fe02_cluster.env.example}"
[[ "$FE02_CODE_ROOT" == "$root" ]] || { echo 'FE02_CODE_ROOT must match this uploaded source directory' >&2; exit 2; }
[[ "$FE02_OUTPUT_ROOT" == /* && "$FE02_OUTPUT_ROOT" != / && "$FE02_OUTPUT_ROOT" != "$root" ]] || {
    echo 'Use an independent absolute FE02_OUTPUT_ROOT' >&2; exit 2;
}
[[ "$FE02_STAGE" == formal || "$FE02_STAGE" == pilot ]] || { echo 'Invalid stage' >&2; exit 2; }
[[ "$FE02_RESUME" == 0 || "$FE02_RESUME" == 1 ]] || { echo 'FE02_RESUME must be 0 or 1' >&2; exit 2; }
for key in FE02_NODES FE02_DEVICES_PER_NODE FE02_CPUS_PER_TASK FE02_ARRAY_CONCURRENCY; do
    [[ "${!key}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid $key" >&2; exit 2; }
done
world=$((FE02_NODES * FE02_DEVICES_PER_NODE))
((128 % (8 * world) == 0)) || { echo 'Total devices must divide the fixed global batch 128 / microbatch 8' >&2; exit 2; }
[[ "$FE02_INDICES" =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]] || { echo 'Invalid array indices' >&2; exit 2; }
limit=14; [[ "$FE02_STAGE" != pilot ]] || limit=4
IFS=, read -ra selections <<< "$FE02_INDICES"
for selection in "${selections[@]}"; do
    first=${selection%%-*}; last=${selection##*-}
    ((10#$first <= 10#$last && 10#$last <= limit)) || { echo 'Array index outside the selected stage' >&2; exit 2; }
done

mkdir -p "$FE02_OUTPUT_ROOT/slurm" "$FE02_OUTPUT_ROOT/launches"
umask 077
snapshot=$(mktemp "$FE02_OUTPUT_ROOT/launches/fe02-XXXXXX.env")
# Freeze this invocation's overrides for the queued compute task.
while IFS= read -r key; do
    [[ "$key" != FE02_ENV_FILE ]] || continue
    printf 'export %s=%q\n' "$key" "${!key}"
done < <(compgen -A variable FE02_) > "$snapshot"
chmod 400 "$snapshot"
snapshot_sha=$(sha256sum -- "$snapshot" | cut -d ' ' -f 1)
command=(sbatch --job-name=fe02-run --partition="$FE02_PARTITION"
    --nodes="$FE02_NODES" --ntasks-per-node="$FE02_DEVICES_PER_NODE"
    --cpus-per-task="$FE02_CPUS_PER_TASK" --gres="$FE02_GRES_KIND:$FE02_DEVICES_PER_NODE"
    --time="$FE02_WALLTIME" --array="$FE02_INDICES%$FE02_ARRAY_CONCURRENCY"
    --chdir="$root" --export=ALL
    --output="$FE02_OUTPUT_ROOT/slurm/%x-%A_%a.out"
    --error="$FE02_OUTPUT_ROOT/slurm/%x-%A_%a.err")
[[ -z "${FE02_MEMORY:-}" ]] || command+=(--mem="$FE02_MEMORY")
[[ -z "${FE02_ACCOUNT:-}" ]] || command+=(--account="$FE02_ACCOUNT")
[[ -z "${FE02_QOS:-}" ]] || command+=(--qos="$FE02_QOS")
if [[ "$dry_run" == 1 ]]; then
    printf '%q ' "${command[@]}" "$root/scripts/fe02/submit.sh" --worker "$snapshot" "$snapshot_sha"
    printf '\n'
else
    exec "${command[@]}" "$root/scripts/fe02/submit.sh" --worker "$snapshot" "$snapshot_sha"
fi
