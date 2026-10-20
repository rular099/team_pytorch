#!/usr/bin/env bash
# LOGIN: retry only index4; status is read-only; pack is a separate CPU sbatch.
set -Eeuo pipefail
mode=${1:-eval_diting_off}
[[ "$mode" == eval_diting_off || "$mode" == status || "$mode" == pack ]] || {
  echo 'Usage: submit_diting_eval_fix.sh [eval_diting_off|status|pack]' >&2;exit 2;
}
tool_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ "$mode" == status ]]; then
  exec bash "$tool_dir/check_eval_status.sh"
fi
if [[ -f "$tool_dir/SHA256SUMS" ]]; then
  (cd "$tool_dir" && sha256sum -c SHA256SUMS)
fi
log="$tool_dir/diting_fix_${mode}_$(date +%Y%m%dT%H%M%S)_$$.log"
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
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
revision=$("${A01_CONFIG_PYTHON:-python3}" -B "$tool_dir/eval_revision.py" register \
  --policy diting_off --session "$A01_EVAL_SESSION" --tool-dir "$tool_dir")
export A01_EVAL_REVISION_DIRECTORY="$revision"
exec bash "$revision/eval_pack.sh" "$mode"
