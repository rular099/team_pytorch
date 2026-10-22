#!/usr/bin/env bash
# LOGIN: only explicit submit/resume can call sbatch.
set -Eeuo pipefail
mode=${1:?}
shift
tool_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
release_root=$(cd "$tool_dir/../.." && pwd)
args=("$@")
env_file=''
inherit_session=''
for ((i=0;i<${#args[@]};i++)); do
  case "${args[$i]}" in
    --env) env_file=${args[$((i+1))]};;
    --inherit-session) inherit_session=${args[$((i+1))]};;
  esac
done
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$release_root${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$mode" == build-bundle ]]; then
  exec "${ON_EVAL_CONFIG_PYTHON:-python3}" -B -m fe01_a01_compare.cli "$mode" "${args[@]}"
fi
: "${env_file:?--env is required}"
if [[ "$mode" == prepare && ! -f "$env_file" ]]; then
  : "${inherit_session:?First prepare requires --inherit-session pointing to the completed A01 eval session}"
  "${ON_EVAL_CONFIG_PYTHON:-python3}" -B -m fe01_a01_compare.cli init-env --env "$env_file" --inherit-session "$inherit_session"
fi
# The user explicitly chooses this trusted private shell file; never source discovered experiment files.
set +u
source "$env_file"
set -u
export ON_EVAL_RELEASE_ROOT="$release_root"
export PYTHONPATH="$release_root:${ON_EVAL_INHERITED_CODE_ROOT:-$release_root}/vendor${PYTHONPATH:+:$PYTHONPATH}"
filtered=()
for ((i=0;i<${#args[@]};i++)); do
  if [[ "${args[$i]}" == --inherit-session ]]; then
    i=$((i+1))
  else
    filtered+=("${args[$i]}")
  fi
done
# Prepare/status do not initialize CUDA or call sbatch/srun.
exec "${ON_EVAL_CONFIG_PYTHON:-python3}" -B -m fe01_a01_compare.cli "$mode" "${filtered[@]}"
