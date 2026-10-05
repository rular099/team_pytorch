#!/bin/bash
#SBATCH --job-name=fe01e1_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH --mem=32G
set -eo pipefail
source "${SLURM_SUBMIT_DIR:?}/scripts/fe01_review/env.sh"
fe01_review_action="${1:?verify or evaluate}"
fe01_review_scope="${2:-random}"
fe01_review_model="$($FE01_REVIEW_PYTHON -c 'import json,os; m=json.load(open(os.environ["FE01_REVIEW_RUN_ROOT"]+"/identity/frozen_checkpoint_manifest.json")); print(m["models"][int(os.environ["SLURM_ARRAY_TASK_ID"])]["run_id"])')"
fe01_review_args=("$fe01_review_action" --manifest "$FE01_REVIEW_RUN_ROOT/identity/frozen_checkpoint_manifest.json" --run-id "$fe01_review_model" --plan "$FE01_REVIEW_PLAN" --request-root "$FE01_REVIEW_RUN_ROOT/requests" --source-root "$FE01_TRAIN_OUTPUT_ROOT" --device cuda)
if [ "$fe01_review_action" = verify ]; then
  fe01_review_args+=(--output "$FE01_REVIEW_RUN_ROOT/verification/$fe01_review_model")
elif [ "$fe01_review_action" = evaluate ]; then
  fe01_review_args+=(--scope "$fe01_review_scope" --verification "$FE01_REVIEW_RUN_ROOT/verification/$fe01_review_model" --output "$FE01_REVIEW_RUN_ROOT/evaluations/$fe01_review_model/$fe01_review_scope")
else
  echo 'Only verify/evaluate allowed' >&2; exit 2
fi
exec "$FE01_REVIEW_PYTHON" scripts/fe01_review/run.py "${fe01_review_args[@]}"
