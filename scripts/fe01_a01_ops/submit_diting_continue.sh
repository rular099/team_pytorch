#!/usr/bin/env bash
# LOGIN NODE: one explicit sbatch array through the operations launcher.
set -euo pipefail
ops_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -f "$ops_dir/SHA256SUMS" ]]; then
  (cd "$ops_dir" && sha256sum -c SHA256SUMS)
fi
exec bash "$ops_dir/dcu_recovery.sh" continue "${1:-3,4}"
