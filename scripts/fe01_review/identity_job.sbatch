#!/bin/bash
#SBATCH --job-name=fe01e1_identity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -eo pipefail
source "${SLURM_SUBMIT_DIR:?}/scripts/fe01_review/env.sh"
"$FE01_REVIEW_PYTHON" scripts/fe01_review/run.py inventory --manifest "$FE01_REVIEW_MANIFEST" --output "$FE01_REVIEW_RUN_ROOT/identity"
"$FE01_REVIEW_PYTHON" scripts/fe01_review/run.py plan --hydrate --plan "$FE01_REVIEW_PLAN" --source-root "$FE01_TRAIN_OUTPUT_ROOT" --output "$FE01_REVIEW_RUN_ROOT/requests"
echo 'IDENTITY_AND_REQUEST_PLANNING_FINISHED: inspect each checkpoint status; this is not a forward PASS'
