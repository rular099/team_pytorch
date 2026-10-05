#!/bin/bash
#SBATCH --job-name=fe01e1_reference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -eo pipefail
source "${SLURM_SUBMIT_DIR:?}/scripts/fe01_review/env.sh"
exec "$FE01_REVIEW_PYTHON" scripts/fe01_review/run.py reference --source-root "$FE01_TRAIN_OUTPUT_ROOT" --manifest "$FE01_REVIEW_RUN_ROOT/identity/frozen_checkpoint_manifest.json" --output "$FE01_REVIEW_RUN_ROOT/reference"
