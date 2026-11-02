#!/usr/bin/env bash
# LOGIN: explicit submission of TEAM model_smoke task 3 only.
set -Eeuo pipefail
tool_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
env_file="$PWD/on_eval_v4_inventory_fix.private.env"
args=("$@")
for ((i=0;i<${#args[@]};i++)); do
  if [[ "${args[$i]}" == --env ]]; then env_file=${args[$((i+1))]:?}; fi
done
[[ -f "$env_file" ]] || { echo "Private environment not found: $env_file" >&2; exit 1; }
set +u
source "$env_file"
set -u
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ON_EVAL_RELEASE_ROOT:?}:${ON_EVAL_INHERITED_CODE_ROOT:?}/vendor${PYTHONPATH:+:$PYTHONPATH}"
exec "${ON_EVAL_CONFIG_PYTHON:-python3}" -B "$tool_dir/submit.py" \
  --env "$env_file" --run-id on_v4_20261025_inventory_fix "${args[@]}"
