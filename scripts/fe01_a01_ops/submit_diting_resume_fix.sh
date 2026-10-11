#!/usr/bin/env bash
# LOGIN NODE: verified RNG compatibility entry, then one real sbatch array.
# A failed OFF directory is preserved by rename only if it has no checkpoint,
# epoch curve, or nonempty sample journal. Every other nonempty fresh run blocks.
set -euo pipefail
ops_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -f "$ops_dir/SHA256SUMS" ]]; then
  (cd "$ops_dir" && sha256sum -c SHA256SUMS)
fi
export A01_ARCHIVE_UNCOMMITTED_DITING_OFF=1
exec bash "$ops_dir/dcu_recovery.sh" continue "${1:-3,4}"
