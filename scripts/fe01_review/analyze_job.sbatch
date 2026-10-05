#!/bin/bash
#SBATCH --job-name=fe01e1_stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -eo pipefail
source "${SLURM_SUBMIT_DIR:?}/scripts/fe01_review/env.sh"
exec "$FE01_REVIEW_PYTHON" scripts/fe01_review/run.py analyze --source-root "$FE01_TRAIN_OUTPUT_ROOT" --reference "$FE01_REVIEW_RUN_ROOT/reference/train_only_reference.json" --verification "$FE01_REVIEW_RUN_ROOT/verification" --output "$FE01_REVIEW_RUN_ROOT/analysis"
