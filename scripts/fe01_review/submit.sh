#!/bin/bash
# Run manually on the login node. One invocation submits exactly one stage.
set -euo pipefail
fe01_review_stage="${1:-identity}"
fe01_review_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fe01_review_env="${2:-$fe01_review_script_dir/cluster_zb.env}"
if [ ! -f "$fe01_review_env" ]; then
  echo "Environment file missing: $fe01_review_env" >&2
  exit 2
fi
export FE01_REVIEW_ENV="$(cd -- "$(dirname -- "$fe01_review_env")" && pwd)/$(basename -- "$fe01_review_env")"
source "$FE01_REVIEW_ENV"
cd "${FE01_CODE_ROOT:?}"
fe01_review_python="${FE01_SUBMIT_PYTHON:-python3}"
if [ -z "${FE01_SUBMIT_PYTHON:-}" ] && [ -x "$HOME/.conda/envs/${FE01_CONDA_ENV:-zb}/bin/python" ]; then
  fe01_review_python="$HOME/.conda/envs/${FE01_CONDA_ENV:-zb}/bin/python"
fi
exec "$fe01_review_python" scripts/fe01_review/submit.py "$fe01_review_stage"
