#!/usr/bin/env bash
# Run with bash on the login node, NOT sbatch.  Only submits missing work:
# one fresh vfull training, ten validation jobs, then one analysis job.
# No torch is required on the login node.  No preflight or trained-arm restart.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKDIR=${WORKDIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
V01_RUN_ROOT=${V01_RUN_ROOT:-$WORKDIR/v01_velocity_prep_padding_seed42}
V01_RECOVERY_TAG=${V01_RECOVERY_TAG:-retry1}
[[ "$V01_RECOVERY_TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Unsafe V01_RECOVERY_TAG." >&2; exit 2; }

export WORKDIR V01_RUN_ROOT
export V01_VFULL_WEIGHT_PATH=${V01_VFULL_WEIGHT_PATH:-$V01_RUN_ROOT/weights_vfull_$V01_RECOVERY_TAG}
export V01_VMISSING_WEIGHT_PATH=${V01_VMISSING_WEIGHT_PATH:-$V01_RUN_ROOT/weights_vmissing}
export V01_APAIR_WEIGHT_PATH=${V01_APAIR_WEIGHT_PATH:-$V01_RUN_ROOT/weights_apair}
export V01_EVAL_ROOT=${V01_EVAL_ROOT:-$V01_RUN_ROOT/eval_$V01_RECOVERY_TAG}
export V01_REPORT_ROOT=${V01_REPORT_ROOT:-$V01_RUN_ROOT/report_$V01_RECOVERY_TAG}
export SOURCE_IDENTITY_MODE=${SOURCE_IDENTITY_MODE:-uploaded_sha256}
export ACTION=${ACTION:-recover}
case "$ACTION" in
    recover|eval|analyze) export ARMS=vfull,vmissing,apair ;;
    train) export ARMS=vfull ;;
    *) echo "Recovery wrapper only accepts ACTION=recover|train|eval|analyze." >&2; exit 2 ;;
esac
if [[ "$ACTION" == recover ]]; then
    # Reserve the invocation on the login node so a second click cannot queue
    # another batch before the first training job creates its output files.
    export V01_SUBMISSION_GUARD_DIR="$V01_RUN_ROOT/submissions/recover_$V01_RECOVERY_TAG"
fi
if [[ "${ALLOW_EXISTING_EVAL:-0}" != 0 || "${RESET_WEIGHT_PATH:-0}" != 0 ]]; then
    echo "Recovery refuses destructive/overwrite options." >&2; exit 2
fi

echo "[RECOVERY] stage=$ACTION tag=$V01_RECOVERY_TAG"
echo "[RECOVERY] vfull_output=$V01_VFULL_WEIGHT_PATH"
echo "[RECOVERY] reusing_vmissing=$V01_VMISSING_WEIGHT_PATH"
echo "[RECOVERY] reusing_apair=$V01_APAIR_WEIGHT_PATH"
echo "[RECOVERY] eval_output=$V01_EVAL_ROOT report_output=$V01_REPORT_ROOT"
exec bash "$WORKDIR/tools/run_v01_prep_padding_controls_slurm.sh" "$@"
