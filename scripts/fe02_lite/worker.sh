#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 ]] || { echo 'worker requires environment, environment SHA, task-map SHA' >&2; exit 2; }
: "${SLURM_JOB_ID:?Use submit scripts on the login node}"
[[ "$(sha256sum -- "$1" | cut -d ' ' -f 1)" == "$2" ]] || { echo 'Launch environment changed' >&2; exit 2; }
source "$1"
[[ "$(sha256sum -- "$FE02_LITE_CODE_ROOT/SOURCE_IDENTITY.json" | cut -d ' ' -f 1)" == "$FE02_LITE_SOURCE_IDENTITY_SHA256" ]] || { echo 'Source identity changed' >&2; exit 2; }
[[ "$(sha256sum -- "$FE02_LITE_TASK_MAP" | cut -d ' ' -f 1)" == "$3" ]] || { echo 'Task map changed' >&2; exit 2; }
source "$FE02_LITE_CODE_ROOT/scripts/fe02_lite/env.sh"
selection=$("$FE02_LITE_PYTHON" - "$FE02_LITE_TASK_MAP" "${SLURM_ARRAY_TASK_ID:?}" <<'PY'
import json,sys,shlex
row=json.load(open(sys.argv[1]))[int(sys.argv[2])]
for key in ('dataset_view','variant','seed','stage','run_id'):
    print('lite_'+key+'='+shlex.quote(str(row[key])))
PY
)
eval "$selection"
run_dir="$FE02_LITE_OUTPUT_ROOT/$lite_run_id"
audit_dir="$FE02_LITE_OUTPUT_ROOT/audits/$lite_run_id"
mkdir -p "$FE02_LITE_OUTPUT_ROOT/locks" "$FE02_LITE_OUTPUT_ROOT/offline_weights"
exec 9>"$FE02_LITE_OUTPUT_ROOT/locks/$lite_run_id.lock"
flock -n 9 || { echo "Run is already active: $lite_run_id" >&2; exit 2; }
args=(--config "configs/fe02_lite/$lite_dataset_view.json" --dataset-view "$lite_dataset_view"
    --variant "$lite_variant" --seed "$lite_seed" --stage "$lite_stage" --audit-dir "$audit_dir")
if [[ "$FE02_LITE_ACTION" != eval ]]; then
    [[ ! -f "$run_dir/COMPLETED.json" ]] || { echo 'Completed run is preserved' >&2; exit 2; }
    if [[ "$FE02_LITE_ACTION" == resume ]]; then
        [[ -f "$run_dir/last.pth" && -f "$audit_dir/protocol.lock.json" ]] || { echo 'Resume needs last and original audit' >&2; exit 2; }
    else
        [[ ! -e "$run_dir" ]] || { echo 'Run exists; use resume' >&2; exit 2; }
    fi
    (
        flock -x 8
        "$FE02_LITE_PYTHON" scripts/fe02/register_diting.py --checkpoint "$FE02_LITE_DITING_CHECKPOINT" \
            --output "$FE02_LITE_OUTPUT_ROOT/offline_weights/pretrained_manifest.json" \
            --source 'Existing frozen DiTing MAE1200M; pretraining overlap unknown' --reuse-existing
    ) 8>"$FE02_LITE_OUTPUT_ROOT/offline_weights/register.lock"
    unset RANK WORLD_SIZE LOCAL_RANK
    "$FE02_LITE_PYTHON" scripts/fe02_lite/run.py audit "${args[@]}" --device cuda
    train_args=()
    [[ "$FE02_LITE_ACTION" != resume ]] || train_args+=(--resume)
    export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST:?}" | head -n 1)
    export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
    unset SLURM_STEP_GRES
    srun --kill-on-bad-exit=1 --ntasks="${SLURM_NTASKS:?}" --ntasks-per-node="$FE02_LITE_DEVICES_PER_NODE" \
        bash -c 'export RANK="${SLURM_PROCID:?}" WORLD_SIZE="${SLURM_NTASKS:?}" LOCAL_RANK="${SLURM_LOCALID:?}"; exec "$@"' \
        fe02-lite-rank "$FE02_LITE_PYTHON" scripts/fe02_lite/run.py train "${args[@]}" "${train_args[@]}"
fi
unset RANK WORLD_SIZE LOCAL_RANK
for checkpoint in best last; do
    "$FE02_LITE_PYTHON" scripts/fe02_lite/run.py eval "${args[@]}" --checkpoint "$checkpoint" --device cuda
done
if [[ "$FE02_LITE_ACTION" != eval ]]; then
    "$FE02_LITE_PYTHON" - "$run_dir" <<'PY'
from pathlib import Path
import json,sys
p=Path(sys.argv[1])/'COMPLETED.json'
if p.exists(): raise FileExistsError(p)
p.write_text(json.dumps({'status':'training and best/last fixed validation completed'})+'\n')
PY
fi
echo "FE02-Lite completed: $lite_run_id"
