#!/usr/bin/env bash
set -Eeuo pipefail
session=${1:?};export ON_EVAL_ENV_FILE=${2:?};fix_root=${3:?}
source "$ON_EVAL_ENV_FILE"
release_root=${ON_EVAL_RELEASE_ROOT:?}
source "$release_root/scripts/fe01_a01_compare/compute_env.sh"
cd "$release_root"
exec "$ON_EVAL_PYTHON" -B "$fix_root/scripts/fe01_v4_team_smoke_fix/team_smoke_runtime.py" --session "$session"
