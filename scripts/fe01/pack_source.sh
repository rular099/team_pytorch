#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${FE01_PACK_PYTHON:-python}" "$script_dir/pack_source.py" "$@"
