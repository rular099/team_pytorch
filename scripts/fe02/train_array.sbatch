#!/usr/bin/env bash
set -euo pipefail
exec bash "${FE02_CODE_ROOT:?}/scripts/fe02/job.sh" train
