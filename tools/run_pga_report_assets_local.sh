#!/usr/bin/env bash

# Generate academic-report figures/tables on a single machine.
# Usage:
#   bash tools/run_pga_report_assets_local.sh <results_root> [output_dir]
#
# Example:
#   bash tools/run_pga_report_assets_local.sh chaosuan_res reports/pga_academic_report_assets

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

RESULTS_ROOT=${1:-"$REPO_ROOT/chaosuan_res"}
OUTPUT_DIR=${2:-"$REPO_ROOT/reports/pga_academic_report_assets"}

cd "$REPO_ROOT"
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-cache}

python tools/generate_pga_report_assets.py \
    --results-root "$RESULTS_ROOT" \
    --pattern "${REPORT_RESULT_PATTERN:-weights_japan*_pga15*}" \
    --output-dir "$OUTPUT_DIR" \
    --primary "${REPORT_PRIMARY_MODEL:-}" \
    --selected-models "${REPORT_SELECTED_MODELS:-}" \
    ${REPORT_EXTRA_ARGS:-}

