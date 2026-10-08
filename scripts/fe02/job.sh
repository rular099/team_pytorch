#!/usr/bin/env bash
set -euo pipefail
action=${1:?run|audit|train|eval|collect}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/env.sh"
stage=${FE02_STAGE:-formal}
index=${SLURM_ARRAY_TASK_ID:-0}
mapfile -t selection < <("$FE02_PYTHON" scripts/fe02/select_run.py --root "$FE02_CODE_ROOT" --stage "$stage" --index "$index")
[[ ${#selection[@]} == 2 ]] || { echo 'Invalid FE02 selection' >&2; exit 2; }
config=${selection[0]};run_id=${selection[1]}
audit_dir="$FE02_OUTPUT_ROOT/audits/$run_id"
run_dir="$FE02_OUTPUT_ROOT/$run_id"
case "$action" in
    run)
        mkdir -p "$FE02_OUTPUT_ROOT/locks" "$FE02_WEIGHTS_ROOT"
        # Protect an individual run for the whole pipeline and a shared manifest
        # during registration, including when explicit array concurrency > 1.
        exec 9>"$FE02_OUTPUT_ROOT/locks/$run_id.lock"
        flock -n 9 || { echo "Another task is already running $run_id" >&2; exit 2; }
        if [[ "${FE02_RESUME:-0}" == 1 ]]; then
            [[ -f "$run_dir/last.pth" ]] || { echo "Resume requires $run_dir/last.pth" >&2; exit 2; }
            [[ -f "$audit_dir/protocol.lock.json" ]] || { echo 'Resume requires the original successful audit' >&2; exit 2; }
        elif [[ -d "$run_dir" && -n "$(ls -A -- "$run_dir")" ]]; then
            echo "Preserving existing run: $run_dir. Use FE02_RESUME=1 with last.pth." >&2; exit 2
        fi
        (
            flock -x 8
            "$FE02_PYTHON" scripts/fe02/register_diting.py \
                --checkpoint "$FE02_DITING_CHECKPOINT" \
                --output "$FE02_WEIGHTS_ROOT/pretrained_manifest.json" \
                --source 'Existing DiTing MAE1200M pretraining checkpoint; corpus overlap unknown' \
                --reuse-existing
        ) 8>"$FE02_WEIGHTS_ROOT/register.lock"
        # GPU work stays inside this allocation, not on the login node.
        unset RANK WORLD_SIZE LOCAL_RANK
        if [[ ! -f "$audit_dir/protocol.lock.json" ]]; then
            args=(audit --config "$config" --output "$audit_dir" --device cuda)
            if [[ -n "${FE02_REUSE_DATA_AUDIT:-}" && -f "$FE02_REUSE_DATA_AUDIT" ]]; then
                args+=(--reuse-data-audit "$FE02_REUSE_DATA_AUDIT")
            fi
            "$FE02_PYTHON" scripts/fe02/run.py "${args[@]}"
        fi
        # Existing training code enforces the audit lock, last-only resume, and
        # fixed config/data/source/world identity. Failed audit never trains.
        bash "$script_dir/job.sh" train
        "$FE02_PYTHON" scripts/fe02/run.py eval --config "$config" \
            --checkpoint "$run_dir/best.pth" --output "$run_dir/evaluation_fixed_val" --device cuda
        echo "FE02 completed training and normal/random validation: $run_id"
        ;;
    audit)
        args=(audit --config "$config" --output "$audit_dir" --device cuda)
        if [[ -n "${FE02_REUSE_DATA_AUDIT:-}" ]]; then args+=(--reuse-data-audit "$FE02_REUSE_DATA_AUDIT"); fi
        exec "$FE02_PYTHON" scripts/fe02/run.py "${args[@]}";;
    train)
        args=(train --config "$config" --audit-dir "$audit_dir")
        if [[ "${FE02_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
        fe02_tasks=${SLURM_NTASKS_PER_NODE:?};fe02_tasks=${fe02_tasks%%(*}
        [[ "$fe02_tasks" == "$FE02_DEVICES_PER_NODE" ]] || { echo 'Allocation/tasks-per-node mismatch' >&2; exit 2; }
        export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
        export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
        # Prevent inherited single-DCU step GRES from conflicting with multi-node allocation.
        unset SLURM_STEP_GRES
        exec srun --kill-on-bad-exit=1 --ntasks="$SLURM_NTASKS" --ntasks-per-node="$FE02_DEVICES_PER_NODE" \
            bash -c 'export RANK="${SLURM_PROCID:?}" WORLD_SIZE="${SLURM_NTASKS:?}" LOCAL_RANK="${SLURM_LOCALID:?}"; exec "$@"' \
            fe02-rank "$FE02_PYTHON" scripts/fe02/run.py "${args[@]}";;
    eval)
        args=(eval --config "$config" --checkpoint "$run_dir/best.pth" \
            --output "$run_dir/evaluation_${FE02_EVAL_TAG:-fixed_val}" --device cuda)
        if [[ "${FE02_EVAL_TAG:-fixed_val}" == random_val ]]; then
            args+=(--random-times-manifest "$audit_dir/random_validation_times_manifest.csv")
        elif [[ "${FE02_EVAL_TAG:-fixed_val}" != fixed_val ]]; then echo 'Unknown eval tag' >&2; exit 2; fi
        exec "$FE02_PYTHON" scripts/fe02/run.py "${args[@]}";;
    collect)
        exec "$FE02_PYTHON" scripts/fe02/collect_runs.py --root "$FE02_OUTPUT_ROOT" \
            --stage "$stage" --output "$FE02_OUTPUT_ROOT/review_seed42";;
    *) echo 'Unknown action' >&2; exit 2;;
esac
