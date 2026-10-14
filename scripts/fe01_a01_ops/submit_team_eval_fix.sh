#!/usr/bin/env bash
# LOGIN: default retries only index0. Later call with pack for the same session.
set -Eeuo pipefail
mode=${1:-eval_team}
[[ "$mode" == eval_team || "$mode" == pack ]] || { echo 'Usage: submit_team_eval_fix.sh [eval_team|pack]' >&2;exit 2; }
tool_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -f "$tool_dir/SHA256SUMS" ]]; then
  (cd "$tool_dir" && sha256sum -c SHA256SUMS)
fi
log="$tool_dir/team_fix_${mode}_$(date +%Y%m%dT%H%M%S)_$$.log"
(umask 077;set -o noclobber;: > "$log")
exec > >(tee -a "$log") 2> >(tee -a "$log" >&2)
printf 'SUBMISSION_LOG=%s\n' "$log" >&2
session=${A01_EVAL_SESSION:-}
if [[ -z "$session" && -f "$tool_dir/../a01_eval_pack_20261014/eval_session.path" ]]; then
  IFS= read -r session < "$tool_dir/../a01_eval_pack_20261014/eval_session.path"
fi
: "${session:?Set A01_EVAL_SESSION to the existing evaluation session}"
export A01_EVAL_SESSION="$session" A01_ENV_FILE="$session/runtime.private.env"
source "$A01_ENV_FILE"
revision=$("${A01_CONFIG_PYTHON:-python3}" "$tool_dir/eval_revision.py" register \
  --session "$A01_EVAL_SESSION" --tool-dir "$tool_dir")
export A01_EVAL_REVISION_DIRECTORY="$revision"
exec bash "$revision/eval_pack.sh" "$mode"
