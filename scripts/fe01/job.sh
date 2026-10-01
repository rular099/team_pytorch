#!/usr/bin/env bash
# Invoked ONLY inside a user-created Slurm allocation. Never auto-submits.
set -euo pipefail
action=${1:?audit|train|eval|replay|site-map}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/cluster_common.sh"
: "${SLURM_JOB_ID:?Manually submit this job with sbatch first}"
stage=${FE01_STAGE:-formal}
index=${SLURM_ARRAY_TASK_ID:-${FE01_RUN_INDEX:-0}}
selection_args=(--root "$FE01_CODE_ROOT" --stage "$stage" --index "$index")
if [[ "${FE01_INCLUDE_CONTROLS:-0}" == 1 ]]; then selection_args+=(--controls); fi
mapfile -t selection < <("$FE01_PYTHON" scripts/fe01/select_run.py "${selection_args[@]}")
[[ ${#selection[@]} == 2 ]] || { echo 'Invalid run selection' >&2; exit 2; }
config=${selection[0]}
run_id=${selection[1]}
audit_dir="$FE01_OUTPUT_ROOT/audits/$run_id"
run_dir="$FE01_OUTPUT_ROOT/$run_id"
case "$action" in
    audit)
        args=(audit --config "$config" --output "$audit_dir" --device "${FE01_DEVICE:-cuda}")
        if [[ -n "${FE01_REUSE_DATA_AUDIT:-}" ]]; then args+=(--reuse-data-audit "$FE01_REUSE_DATA_AUDIT"); fi
        exec "$FE01_PYTHON" scripts/fe01/run.py "${args[@]}"
        ;;
    train)
        : "${FE01_DEVICES_PER_NODE:?}" "${FE01_MASTER_PORT:?}"
        export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}
        export MASTER_PORT=$FE01_MASTER_PORT
        export FE01_WORLD_SIZE=$((SLURM_JOB_NUM_NODES * FE01_DEVICES_PER_NODE))
        args=(train --config "$config" --audit-dir "$audit_dir")
        if [[ "${FE01_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
        # Reuse the existing cluster's Slurm-direct DCU launch convention.
        exec srun --kill-on-bad-exit=1 --ntasks="$FE01_WORLD_SIZE" --ntasks-per-node="$FE01_DEVICES_PER_NODE" \
            bash -c 'export RANK="${SLURM_PROCID:?}" WORLD_SIZE="${FE01_WORLD_SIZE:?}" LOCAL_RANK="${SLURM_LOCALID:-0}"; exec "$@"' \
            fe01-rank "$FE01_PYTHON" scripts/fe01/run.py "${args[@]}"
        ;;
    eval)
        checkpoint=${FE01_CHECKPOINT:-$run_dir/best.pth}
        split=${FE01_EVAL_SPLIT:-val}
        args=(eval --config "$config" --checkpoint "$checkpoint" --split "$split" \
            --output "$run_dir/evaluation_${FE01_EVAL_TAG:-fixed_val}" --device "${FE01_DEVICE:-cuda}")
        if [[ -n "${FE01_RANDOM_TIMES_MANIFEST:-}" ]]; then args+=(--random-times-manifest "$FE01_RANDOM_TIMES_MANIFEST"); fi
        if [[ "$split" == test ]]; then
            [[ "${FE01_ALLOW_TEST:-0}" == 1 ]] || { echo 'Test is locked' >&2; exit 2; }
            : "${FE01_PROTOCOL_LOCK_SHA256:?}" "${FE01_TEST_EXPOSURE_LEDGER:?}"
            args+=(--allow-test --protocol-lock-sha256 "$FE01_PROTOCOL_LOCK_SHA256" --test-exposure-ledger "$FE01_TEST_EXPOSURE_LEDGER")
        fi
        exec "$FE01_PYTHON" scripts/fe01/run.py "${args[@]}"
        ;;
    replay)
        : "${FE01_CASES_MANIFEST:?}"
        args=(replay --config "$config" --checkpoint "${FE01_CHECKPOINT:-$run_dir/best.pth}" \
            --cases-manifest "$FE01_CASES_MANIFEST" --output "$run_dir/replay_${FE01_REPLAY_TAG:-prefix}" \
            --device "${FE01_DEVICE:-cuda}" --reference "${FE01_REFERENCE:-first_p_pick}")
        if [[ "${FE01_ROLLING:-0}" == 1 ]]; then args+=(--rolling); fi
        exec "$FE01_PYTHON" scripts/fe01/run.py "${args[@]}"
        ;;
    site-map)
        : "${FE01_REPLAY_DIR:?}" "${FE01_MAP_OUTPUT:?}" "${FE01_REFERENCE_JSON:?}"
        exec "$FE01_PYTHON" scripts/fe01/render_maps.py --replay "$FE01_REPLAY_DIR" \
            --reference "$FE01_REFERENCE_JSON" --output "$FE01_MAP_OUTPUT"
        ;;
    *) echo 'Unknown action' >&2; exit 2;;
esac
