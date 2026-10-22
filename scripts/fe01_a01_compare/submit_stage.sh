#!/usr/bin/env bash
set -Eeuo pipefail
tool_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$tool_dir/dispatch.sh" submit "$@"
