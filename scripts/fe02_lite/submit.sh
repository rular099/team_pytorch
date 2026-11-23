#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "${FE02_LITE_ENV_FILE:-$root/fe02_lite_cluster.env.example}"
exec "${FE02_LITE_LOGIN_PYTHON:-python3}" "$root/scripts/fe02_lite/submit.py" "$@"
