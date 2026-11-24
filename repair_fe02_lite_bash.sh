#!/usr/bin/env bash
# Existing v2 snapshots/audit locks stay byte-identical; only the submitted worker differs.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${FE02_LITE_ENV_FILE:-$root/fe02_lite_cluster.env.example}"
exec "${FE02_LITE_LOGIN_PYTHON:-python3}" "$root/compat/fe02_lite_bash_20261124/recover.py" "$@"
