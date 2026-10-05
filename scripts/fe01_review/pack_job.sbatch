#!/bin/bash
#SBATCH --job-name=fe01e1_pack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
set -eo pipefail
source "${SLURM_SUBMIT_DIR:?}/scripts/fe01_review/env.sh"
exec "$FE01_REVIEW_PYTHON" scripts/fe01_review/run.py pack --input "$FE01_REVIEW_RUN_ROOT" --output "$FE01_REVIEW_RUN_ROOT/return_package"
