#!/usr/bin/env bash
# LOGIN: read-only status; never sbatch, srun, or import torch.
set -Eeuo pipefail
tool_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -f "$tool_dir/SHA256SUMS" ]]; then
  (cd "$tool_dir" && sha256sum -c SHA256SUMS)
fi
session=${A01_EVAL_SESSION:-}
if [[ -z "$session" && -f "$tool_dir/../a01_eval_pack_20261014/eval_session.path" ]]; then
  IFS= read -r session < "$tool_dir/../a01_eval_pack_20261014/eval_session.path"
fi
: "${session:?Set A01_EVAL_SESSION to the existing evaluation session}"
[[ -f "$session/runtime.private.env" ]] || { echo "Missing session: $session" >&2;exit 2; }
source "$session/runtime.private.env"
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
exec "${A01_CONFIG_PYTHON:-python3}" -B -u "$tool_dir/eval_status.py" --session "$session"
