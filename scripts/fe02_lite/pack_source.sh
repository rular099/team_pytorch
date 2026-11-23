#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
exec "${FE02_LITE_LOGIN_PYTHON:-python3}" "$root/scripts/fe01/pack_source.py" \
    --include-report-prefix reports/fe02_lite_local_20261123/ "$@"
