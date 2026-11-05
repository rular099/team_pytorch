#!/usr/bin/env bash
# LOGIN: explicit submission of TEAM model_smoke task 3 only.
set -Eeuo pipefail
team_smoke_tool_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
team_smoke_env_file="$PWD/on_eval_v4_inventory_fix.private.env"
team_smoke_expect_env=0
# "$@" is safe with zero arguments under nounset, including older Bash.
# Avoid a generic args array: the trusted private file may use that name itself.
for team_smoke_arg in "$@"; do
  if [[ "$team_smoke_expect_env" == 1 ]]; then
    team_smoke_env_file=$team_smoke_arg
    team_smoke_expect_env=0
  else
    case "$team_smoke_arg" in
      --env) team_smoke_expect_env=1;;
      --env=*) team_smoke_env_file=${team_smoke_arg#--env=};;
    esac
  fi
done
if [[ "$team_smoke_expect_env" == 1 ]]; then
  echo '--env requires a path' >&2
  exit 2
fi
[[ -f "$team_smoke_env_file" ]] || { echo "Private environment not found: $team_smoke_env_file" >&2; exit 1; }
set +u
source "$team_smoke_env_file"
set -u
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ON_EVAL_RELEASE_ROOT:?}:${ON_EVAL_INHERITED_CODE_ROOT:?}/vendor${PYTHONPATH:+:$PYTHONPATH}"
exec "${ON_EVAL_CONFIG_PYTHON:-python3}" -B "$team_smoke_tool_dir/submit.py" \
  --env "$team_smoke_env_file" --run-id on_v4_20261025_inventory_fix "$@"
