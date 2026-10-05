#!/bin/bash
set -euo pipefail
if [ "${DRY_RUN:-1}" != 1 ]; then echo 'Preparation only supports DRY_RUN=1; submit printed commands manually.' >&2; exit 2; fi
fe01_review_env="${1:?Usage: bash scripts/fe01_review/print_submit_commands.sh /absolute/cluster_zb.env}"
export FE01_REVIEW_ENV="$fe01_review_env"
source "$fe01_review_env"
export FE01_REVIEW_ENV
python3 "$FE01_CODE_ROOT/scripts/fe01_review/prepare.py"
