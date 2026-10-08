#!/usr/bin/env bash
# Reuse the clean-commit archiver/vendor SHA schema; do not include weights.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
exec "${FE02_PACK_PYTHON:-python}" "$root/scripts/fe01/pack_source.py" --include-report-prefix reports/fe02_local_20261008/ "$@"
