#!/usr/bin/env bash
# Internal compute-node entry point; use complete_v01_validation_slurm.sh.
set -euo pipefail
stage=${1:?audit|eval|analyze}
cell=${2:-}
: "${WORKDIR:?}" "${V01_RUN_ROOT:?}" "${V01_CLOSURE_ROOT:?}"
cd "$WORKDIR"
set +u
[[ ! -f /etc/profile ]] || source /etc/profile
[[ ! -f /etc/profile.d/modules.sh ]] || source /etc/profile.d/modules.sh
if command -v module >/dev/null 2>&1; then
    [[ -z "${MODULE_UNLOAD:-}" ]] || module unload "$MODULE_UNLOAD" || true
    for entry in ${MODULE_LOADS:-}; do module load "$entry"; done
fi
export PS1=${PS1:-}
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-lsm_env}"
else
    source activate "${CONDA_ENV:-lsm_env}"
fi
set -u
audit=(python tools/audit_v01_validation_contract.py --run-root "$V01_RUN_ROOT" --output "$V01_CLOSURE_ROOT")
case "$stage" in
    audit)
        "${audit[@]}" audit
        ;;
    eval)
        case "$cell" in
            vfull__vfull__normal|vfull__vmissing__normal) arm=vfull; directory=weights_vfull_retry1 ;;
            vmissing__vfull__normal|vmissing__vmissing__normal) arm=vmissing; directory=weights_vmissing ;;
            apair__apair__normal|apair__apair__random) arm=apair; directory=weights_apair ;;
            *) echo "Unsupported validation cell: $cell" >&2; exit 2 ;;
        esac
        "${audit[@]}" verify
        checkpoint="$V01_RUN_ROOT/$directory/full_model_last.pth"
        python -m tools.v01_closure_checkpoint_guard "$arm" "$checkpoint"
        stem="$V01_CLOSURE_ROOT/eval/$cell"
        [[ ! -e "$stem.npz" && ! -e "$stem.txt" && ! -e "$stem.npz.val.requests.jsonl" ]] || { echo 'Refusing existing eval outputs.' >&2; exit 1; }
        # One process directly in the batch allocation: no extra step gres.
        python eval_checkpoint.py --config "$V01_CLOSURE_ROOT/configs/$cell.json" \
            --diting_config "$DITING_CONFIG" --diting_pretrained "$DITING_PRETRAINED" \
            --checkpoint "$checkpoint" --output "$stem.npz" --splits val \
            --skip_diagnostics --skip_single_station > "$stem.txt" 2>&1
        python -m tools.v01_closure_checkpoint_guard "$arm" "$checkpoint"
        ;;
    analyze)
        "${audit[@]}" verify
        python tools/analyze_v01_validation_closure.py --run-root "$V01_RUN_ROOT" --closure-root "$V01_CLOSURE_ROOT"
        ;;
    *) echo 'Only audit, eval and analyze are allowed.' >&2; exit 2 ;;
esac
