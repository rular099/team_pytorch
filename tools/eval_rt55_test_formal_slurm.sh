#!/usr/bin/env bash

# One-time formal rt55 test evaluation.  The default evaluates
# full_model_best.pth on the held-out test partition and writes:
#   eval_test_best_normal.txt
#   eval_test_best_normal.npz
#   eval_test_best_normal.metrics.json
#
# The JSON contains formal PGA MAE, RMSE, R2, unweighted MDN NLL, Brier score,
# and predictive 1-sigma/2-sigma coverage in the raw log10(m/s^2) coordinate.
# Test uses the validation fixed-time protocol (1/3/5/10/20/40/90 seconds).
#
# Usage:
#   CONFIRM_TEST_EVAL=1 bash tools/eval_rt55_test_formal_slurm.sh
#   DRY_RUN=1 bash tools/eval_rt55_test_formal_slurm.sh
#
# Optional waveform-station control on test (not needed for the primary table):
#   CONFIRM_TEST_EVAL=1 PERMUTATION=roll bash tools/eval_rt55_test_formal_slurm.sh

set -euo pipefail

WORKDIR=${WORKDIR:-/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch-zhangb-diting-backbone-attnpool-team}
WEIGHT_NAME=${WEIGHT_NAME:-weights_japan_full_2000_2024_rt55_knet_legacy_paddingmask_no_dpk_seed42}
WEIGHT_DIR=${WEIGHT_DIR:-$WORKDIR/$WEIGHT_NAME}
CONFIG=${CONFIG:-$WEIGHT_DIR/config.json}
CHECKPOINT=${CHECKPOINT:-$WEIGHT_DIR/full_model_best.pth}
EVAL_SCRIPT=${EVAL_SCRIPT:-$WORKDIR/eval_checkpoint_slurm.sh}
RUN_LOG_DIR=${RUN_LOG_DIR:-$WORKDIR/logs/$WEIGHT_NAME}
PERMUTATION=${PERMUTATION:-none}
PERMUTATION_SEED=${PERMUTATION_SEED:-12345}
SLURM_TIME=${SLURM_TIME:-24:00:00}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
CONFIRM_TEST_EVAL=${CONFIRM_TEST_EVAL:-0}
SKIP_SPLIT_CSV_CHECK=${SKIP_SPLIT_CSV_CHECK:-0}
EXPECTED_TEST_EVENTS=${EXPECTED_TEST_EVENTS:-2769}
EXPECTED_TEST_STATIONS=${EXPECTED_TEST_STATIONS:-42368}
ALLOW_OVERWRITE=${ALLOW_OVERWRITE:-0}
DRY_RUN=${DRY_RUN:-0}

require_file() {
    local path=$1
    local label=$2
    if [[ -s "$path" ]]; then
        return 0
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY-RUN WARN] $label is not visible: $path" >&2
        return 0
    fi
    echo "$label is missing or empty: $path" >&2
    exit 1
}

case "$PERMUTATION" in
    none)
        output_label=normal
        job_name=team-rt55-test
        permutation_enabled=0
        ;;
    roll|random)
        output_label="waveform_station_${PERMUTATION}"
        job_name="team-rt55-test-${PERMUTATION}"
        permutation_enabled=1
        permutation_args=(
            --waveform_station_permutation "$PERMUTATION"
            --waveform_station_permutation_seed "$PERMUTATION_SEED"
        )
        ;;
    *)
        echo "PERMUTATION must be none, roll, or random; got: $PERMUTATION" >&2
        exit 2
        ;;
esac

require_file "$EVAL_SCRIPT" "Eval launcher"
require_file "$CONFIG" "Resolved rt55 run config"
require_file "$CHECKPOINT" "rt55 best checkpoint"

if [[ "$DRY_RUN" != "1" && "$CONFIRM_TEST_EVAL" != "1" ]]; then
    echo "Formal test evaluation requires explicit confirmation." >&2
    echo "Run: CONFIRM_TEST_EVAL=1 bash tools/eval_rt55_test_formal_slurm.sh" >&2
    exit 2
fi

if [[ "$SKIP_SPLIT_CSV_CHECK" != "1" ]]; then
    split_events="$WEIGHT_DIR/split_events.csv"
    split_stations="$WEIGHT_DIR/split_stations.csv"
    require_file "$split_events" "Training split_events.csv"
    require_file "$split_stations" "Training split_stations.csv"
    if [[ -s "$split_events" && -s "$split_stations" ]]; then
        read -r actual_test_events actual_test_stations < <(
            python -c '
import csv, sys
def count(path):
    with open(path, newline="") as f:
        return sum(1 for row in csv.DictReader(f) if row.get("split") == "test")
print(count(sys.argv[1]), count(sys.argv[2]))
' "$split_events" "$split_stations"
        )
        if [[ "$actual_test_events" != "$EXPECTED_TEST_EVENTS" ]]; then
            echo "Unexpected test event count: $actual_test_events (expected $EXPECTED_TEST_EVENTS)" >&2
            exit 1
        fi
        if [[ "$actual_test_stations" != "$EXPECTED_TEST_STATIONS" ]]; then
            echo "Unexpected test station-row count: $actual_test_stations (expected $EXPECTED_TEST_STATIONS)" >&2
            exit 1
        fi
        echo "[OK] held-out split audit: test_events=$actual_test_events test_station_rows=$actual_test_stations"
    fi
else
    echo "[WARN] SKIP_SPLIT_CSV_CHECK=1; held-out split archive was not audited." >&2
fi

output_stem="$RUN_LOG_DIR/eval_test_best_${output_label}"
output_txt="${output_stem}.txt"
output_npz="${output_stem}.npz"
output_metrics="${output_stem}.metrics.json"
if [[ "$ALLOW_OVERWRITE" != "1" ]] && {
    [[ -s "$output_txt" ]] || [[ -s "$output_npz" ]] || [[ -s "$output_metrics" ]]
}; then
    echo "Formal test output already exists; refusing to overwrite: $output_stem" >&2
    echo "Set ALLOW_OVERWRITE=1 only when rerunning intentionally." >&2
    exit 1
fi

eval_args=(--splits test --skip_single_station --skip_diagnostics)
if [[ "$permutation_enabled" == "1" ]]; then
    eval_args+=("${permutation_args[@]}")
fi

echo "[INFO] rt55 formal test evaluation"
echo "[INFO] checkpoint: $CHECKPOINT"
echo "[INFO] permutation: $PERMUTATION"
echo "[INFO] output stem: $output_stem"
if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY-RUN] WORKDIR=%q JOB_NAME=%q EVAL_CHECKPOINT=%q bash %q %q' \
        "$WORKDIR" "$job_name" "$CHECKPOINT" "$EVAL_SCRIPT" "$CONFIG"
    printf ' %q' "${eval_args[@]}"
    printf '\n'
    exit 0
fi

WORKDIR="$WORKDIR" \
RUN_LOG_DIR="$RUN_LOG_DIR" \
JOB_NAME="$job_name" \
SLURM_TIME="$SLURM_TIME" \
SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" \
EVAL_CHECKPOINT="$CHECKPOINT" \
EVAL_OUTPUT_TXT="$output_txt" \
EVAL_OUTPUT_NPZ="$output_npz" \
AUTO_SBATCH=1 \
bash "$EVAL_SCRIPT" "$CONFIG" "${eval_args[@]}"

echo "[INFO] formal test job submitted; metrics will be written to $output_metrics"
