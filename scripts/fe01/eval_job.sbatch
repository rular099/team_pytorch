#!/usr/bin/env bash
set -euo pipefail
exec bash "${FE01_CODE_ROOT:?}/scripts/fe01/job.sh" eval
