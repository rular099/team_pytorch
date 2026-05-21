#!/usr/bin/env bash
set -euo pipefail

# Rebuild the Japan K-NET/KiK-net training HDF5 with JMA second-level origin
# corrections. Defaults target the local 2024 dataset; override variables from
# the shell when needed.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

YEAR=${YEAR:-2024}
# Must point to japan_dataset_builder.py's component-text archive layout:
#   <root>/<year>/<event>.tar containing *.knt.tar.gz / *.kik.tar.gz
# The local raw *.knt.kwin.tar / *.kik.kwin.tar archives need conversion first.
WAVEFORM_ROOT=${WAVEFORM_ROOT:-/opt/zb/data/japan}
OUTPUT_DIR=${OUTPUT_DIR:-/run/media/zhangb/My Passport/knet_converted_origin_corrected}
REFERENCE_HDF5=${REFERENCE_HDF5:-/run/media/zhangb/My Passport/knet_converted/japan_${YEAR}.hdf5}
ORIGIN_CORRECTIONS_CSV=${ORIGIN_CORRECTIONS_CSV:-$SCRIPT_DIR/jma_origin_corrections/japan_${YEAR}_origin_corrections.csv}
JMA_CATALOG_CSV=${JMA_CATALOG_CSV:-$SCRIPT_DIR/jma_origin_corrections/jma_${YEAR}_daily_catalog.csv}
JMA_SUMMARY_JSON=${JMA_SUMMARY_JSON:-$SCRIPT_DIR/jma_origin_corrections/japan_${YEAR}_origin_corrections_summary.json}
JMA_CACHE_DIR=${JMA_CACHE_DIR:-$SCRIPT_DIR/jma_origin_corrections/cache}
JMA_TRAVEL_TIME_ZIP=${JMA_TRAVEL_TIME_ZIP:-$SCRIPT_DIR/resources/jma_travel_times/tjma2001h.zip}

TARGET_SAMPLING_RATE=${TARGET_SAMPLING_RATE:-100}
MIN_STATIONS=${MIN_STATIONS:-3}
COMPRESSION_LEVEL=${COMPRESSION_LEVEL:-4}
LIMIT_EVENTS=${LIMIT_EVENTS:-}
OVERWRITE=${OVERWRITE:-1}

PICK_MODE=${PICK_MODE:-travel_time}
FINAL_PICK=${FINAL_PICK:-diting_vel}
TRAVEL_TIME_MODEL=${TRAVEL_TIME_MODEL:-jma2001a}
JMA_SEARCH_MARGIN_SECONDS=${JMA_SEARCH_MARGIN_SECONDS:-10}
JMA_SEARCH_MARGIN_PER_KM=${JMA_SEARCH_MARGIN_PER_KM:-0.03}
JMA_SEARCH_MARGIN_MAX_SECONDS=${JMA_SEARCH_MARGIN_MAX_SECONDS:-60}
FALLBACK_SEARCH_HALF_WINDOW_SECONDS=${FALLBACK_SEARCH_HALF_WINDOW_SECONDS:-60}

STALTA_PRE_SECONDS=${STALTA_PRE_SECONDS:-10}
STALTA_POST_SECONDS=${STALTA_POST_SECONDS:-10}
STALTA_STA_SECONDS=${STALTA_STA_SECONDS:-0.2}
STALTA_LTA_SECONDS=${STALTA_LTA_SECONDS:-1.0}
STALTA_THRESHOLD_RATIO=${STALTA_THRESHOLD_RATIO:-2.5}
STALTA_FEATURE=${STALTA_FEATURE:-vertical}
STALTA_HIGHPASS_HZ=${STALTA_HIGHPASS_HZ:-0.5}

RUN_DITING=${RUN_DITING:-1}
DITINGBENCH_ROOT=${DITINGBENCH_ROOT:-/run/media/zhangb/My Passport/DiTing_project/ditingbench}
DITING_MODEL_NAME=${DITING_MODEL_NAME:-diting1200m}
DITING_WEIGHTS=${DITING_WEIGHTS:-/run/media/zhangb/My Passport/DiTing_project/pingce/ditingbenche/pde2024/diting1200m.tar}
DITING_DEVICE=${DITING_DEVICE:-cuda:0}
DITING_BATCH_SIZE=${DITING_BATCH_SIZE:-100}
DITING_P_TH=${DITING_P_TH:-0.1}
DITING_S_TH=${DITING_S_TH:-0.1}
DITING_D_TH=${DITING_D_TH:-0.3}
DITING_TARGET_HALF_WINDOW_SECONDS=${DITING_TARGET_HALF_WINDOW_SECONDS:-10}
DITING_WINDOW_SECONDS=${DITING_WINDOW_SECONDS:-100}
VELOCITY_HIGHPASS_HZ=${VELOCITY_HIGHPASS_HZ:-0.05}

DIAGNOSTICS_DIR=${DIAGNOSTICS_DIR:-$OUTPUT_DIR/diagnostics_${YEAR}}
N_DIAGNOSTIC_PLOTS=${N_DIAGNOSTIC_PLOTS:-24}
DIAGNOSTIC_RANDOM_SEED=${DIAGNOSTIC_RANDOM_SEED:-2024}

FETCH_ORIGIN_CORRECTIONS=${FETCH_ORIGIN_CORRECTIONS:-1}
USE_UNACCEPTED_ORIGIN_CORRECTIONS=${USE_UNACCEPTED_ORIGIN_CORRECTIONS:-0}

if [[ ! -d "$WAVEFORM_ROOT/$YEAR" ]]; then
  echo "[ERROR] waveform year directory not found: $WAVEFORM_ROOT/$YEAR" >&2
  exit 1
fi

if [[ ! -f "$JMA_TRAVEL_TIME_ZIP" ]]; then
  echo "[ERROR] JMA travel-time table not found: $JMA_TRAVEL_TIME_ZIP" >&2
  exit 1
fi

if [[ ! -f "$ORIGIN_CORRECTIONS_CSV" ]]; then
  if [[ "$FETCH_ORIGIN_CORRECTIONS" == "1" ]]; then
    if [[ ! -f "$REFERENCE_HDF5" ]]; then
      echo "[ERROR] origin correction CSV is missing and reference HDF5 is unavailable:" >&2
      echo "        $ORIGIN_CORRECTIONS_CSV" >&2
      echo "        $REFERENCE_HDF5" >&2
      exit 1
    fi
    echo "[INFO] origin correction CSV not found; fetching JMA daily hypocenters"
    python tools/fetch_jma_hypocenters.py \
      --hdf5 "$REFERENCE_HDF5" \
      --output-csv "$ORIGIN_CORRECTIONS_CSV" \
      --catalog-csv "$JMA_CATALOG_CSV" \
      --summary-json "$JMA_SUMMARY_JSON" \
      --cache-dir "$JMA_CACHE_DIR"
  else
    echo "[ERROR] origin correction CSV not found: $ORIGIN_CORRECTIONS_CSV" >&2
    exit 1
  fi
fi

if [[ "$RUN_DITING" == "1" ]]; then
  if [[ ! -d "$DITINGBENCH_ROOT" ]]; then
    echo "[ERROR] DITINGBENCH_ROOT not found: $DITINGBENCH_ROOT" >&2
    exit 1
  fi
  if [[ ! -f "$DITING_WEIGHTS" ]]; then
    echo "[ERROR] DITING_WEIGHTS not found: $DITING_WEIGHTS" >&2
    echo "        Set RUN_DITING=0 FINAL_PICK=stalta to rebuild without DiTing picks." >&2
    exit 1
  fi
elif [[ "$FINAL_PICK" == diting_* ]]; then
  echo "[ERROR] FINAL_PICK=$FINAL_PICK requires RUN_DITING=1." >&2
  echo "        Use FINAL_PICK=stalta when RUN_DITING=0." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$DIAGNOSTICS_DIR"

CMD=(
  python build_japan_training_data.py
  --waveform_root "$WAVEFORM_ROOT"
  --output_dir "$OUTPUT_DIR"
  --year "$YEAR"
  --min_stations "$MIN_STATIONS"
  --target_sampling_rate "$TARGET_SAMPLING_RATE"
  --compression_level "$COMPRESSION_LEVEL"
  --pick_mode "$PICK_MODE"
  --final_pick "$FINAL_PICK"
  --travel_time_model "$TRAVEL_TIME_MODEL"
  --jma_travel_time_zip "$JMA_TRAVEL_TIME_ZIP"
  --jma_search_margin_seconds "$JMA_SEARCH_MARGIN_SECONDS"
  --jma_search_margin_per_km "$JMA_SEARCH_MARGIN_PER_KM"
  --jma_search_margin_max_seconds "$JMA_SEARCH_MARGIN_MAX_SECONDS"
  --fallback_search_half_window_seconds "$FALLBACK_SEARCH_HALF_WINDOW_SECONDS"
  --stalta_pre_seconds "$STALTA_PRE_SECONDS"
  --stalta_post_seconds "$STALTA_POST_SECONDS"
  --stalta_sta_seconds "$STALTA_STA_SECONDS"
  --stalta_lta_seconds "$STALTA_LTA_SECONDS"
  --stalta_threshold_ratio "$STALTA_THRESHOLD_RATIO"
  --stalta_feature "$STALTA_FEATURE"
  --stalta_highpass_hz "$STALTA_HIGHPASS_HZ"
  --origin_corrections_csv "$ORIGIN_CORRECTIONS_CSV"
  --velocity_highpass_hz "$VELOCITY_HIGHPASS_HZ"
  --diagnostics_dir "$DIAGNOSTICS_DIR"
  --n_diagnostic_plots "$N_DIAGNOSTIC_PLOTS"
  --diagnostic_random_seed "$DIAGNOSTIC_RANDOM_SEED"
)

if [[ -n "$LIMIT_EVENTS" ]]; then
  CMD+=(--limit_events "$LIMIT_EVENTS")
fi

if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi

if [[ "$USE_UNACCEPTED_ORIGIN_CORRECTIONS" == "1" ]]; then
  CMD+=(--use_unaccepted_origin_corrections)
fi

if [[ "$RUN_DITING" == "1" ]]; then
  CMD+=(
    --run_diting
    --ditingbench_root "$DITINGBENCH_ROOT"
    --diting_model_name "$DITING_MODEL_NAME"
    --diting_weights "$DITING_WEIGHTS"
    --diting_device "$DITING_DEVICE"
    --diting_batch_size "$DITING_BATCH_SIZE"
    --diting_p_th "$DITING_P_TH"
    --diting_s_th "$DITING_S_TH"
    --diting_d_th "$DITING_D_TH"
    --diting_target_half_window_seconds "$DITING_TARGET_HALF_WINDOW_SECONDS"
    --diting_window_seconds "$DITING_WINDOW_SECONDS"
  )
fi

echo "[INFO] year: $YEAR"
echo "[INFO] waveform_root: $WAVEFORM_ROOT"
echo "[INFO] output_dir: $OUTPUT_DIR"
echo "[INFO] origin_corrections: $ORIGIN_CORRECTIONS_CSV"
echo "[INFO] run_diting: $RUN_DITING"
echo "[INFO] final_pick: $FINAL_PICK"
printf '[INFO] command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

python - "$OUTPUT_DIR/japan_${YEAR}.hdf5" <<'PY'
import sys
import h5py

path = sys.argv[1]
with h5py.File(path, "r") as f:
    n_events = len(f.get("data", {}))
if n_events <= 0:
    raise SystemExit(f"[ERROR] rebuilt HDF5 has no events: {path}")
print(f"[INFO] rebuilt HDF5 event count: {n_events}")
PY
