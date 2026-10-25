#!/usr/bin/env bash
# LOGIN: explicitly submits only the repaired inventory; never chains later stages.
set -Eeuo pipefail
tool_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
release_root=$(cd "$tool_dir/../.." && pwd)
old_env="$PWD/on_eval.private.env"
for ((i=1;i<=$#;i++)); do
  if [[ "${!i}" == --old-env ]]; then
    j=$((i+1))
    old_env=${!j:?--old-env needs a path}
  fi
done
if [[ ! -f "$old_env" ]]; then
  echo "Original private file not found: $old_env" >&2
  exit 1
fi
set +u
source "$old_env"
set -u
export ON_EVAL_RELEASE_ROOT="$release_root"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$release_root:${ON_EVAL_INHERITED_CODE_ROOT:-$release_root}/vendor${PYTHONPATH:+:$PYTHONPATH}"
exec "${ON_EVAL_CONFIG_PYTHON:-python3}" -B -m fe01_a01_compare.recovery \
  --old-env "$old_env" --env "$PWD/on_eval_v4_inventory_fix.private.env" \
  --submit-inventory "$@"
