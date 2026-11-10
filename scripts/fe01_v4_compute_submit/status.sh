#!/usr/bin/env bash
# LOGIN: status verification itself is a CPU compute job.
set -Eeuo pipefail
v4s_tool=$(cd -- "${BASH_SOURCE[0]%/*}" && pwd -P)
exec bash "$v4s_tool/submit.sh" --status "$@"
