#!/usr/bin/env bash
# Download all matched Hi-net event windows into one raw-byte HDF5 archive per
# year. Years and events are processed newest first. Re-running this script is
# safe: verified, committed events are skipped and incomplete annual archives
# resume from *.partial.h5.

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

PYTHON_BIN=${PYTHON_BIN:-python}
HDF5_ROOT=${HDF5_ROOT:-'/run/media/zhangb/My Passport/knet_converted/origin_corrected_diting_vel_acc_vs30'}
OUTPUT_ROOT=${OUTPUT_ROOT:-'/run/media/zhangb/My Passport/hinet_data'}
START_YEAR=${START_YEAR:-2000}
END_YEAR=${END_YEAR:-2024}
MATCH_DISTANCE_KM=${MATCH_DISTANCE_KM:-0.5}
PRE_SECONDS=${PRE_SECONDS:-120}
POST_SECONDS=${POST_SECONDS:-120}
SLEEP_SECONDS=${SLEEP_SECONDS:-0}
MAX_YEAR_ATTEMPTS=${MAX_YEAR_ATTEMPTS:-3}
RETRY_DELAY_SECONDS=${RETRY_DELAY_SECONDS:-60}
ARCHIVE_CHUNK_BYTES=${ARCHIVE_CHUNK_BYTES:-1048576}
HINET_TIMEOUT_SECONDS=${HINET_TIMEOUT_SECONDS:-300}
HINET_RETRIES=${HINET_RETRIES:-3}
STATION_BATCH_SIZE=${STATION_BATCH_SIZE:-40}
HINET_DOWNLOAD_THREADS=${HINET_DOWNLOAD_THREADS:-1}
MINUTE_FALLBACK=${MINUTE_FALLBACK:-1}
FALLBACK_SPAN_MINUTES=${FALLBACK_SPAN_MINUTES:-1}
SUBREQUEST_SLEEP_SECONDS=${SUBREQUEST_SLEEP_SECONDS:-0}

if [[ ! "$START_YEAR" =~ ^[0-9]{4}$ || ! "$END_YEAR" =~ ^[0-9]{4}$ ]]; then
  echo "[ERROR] START_YEAR and END_YEAR must be four-digit years" >&2
  exit 2
fi
if (( START_YEAR > END_YEAR )); then
  echo "[ERROR] START_YEAR=$START_YEAR is newer than END_YEAR=$END_YEAR" >&2
  exit 2
fi
if [[ -z ${HINET_USER:-} || -z ${HINET_PASSWORD:-} ]]; then
  echo "[ERROR] Export HINET_USER and HINET_PASSWORD before starting the download" >&2
  exit 2
fi
if [[ "$MINUTE_FALLBACK" != "0" && "$MINUTE_FALLBACK" != "1" ]]; then
  echo "[ERROR] MINUTE_FALLBACK must be 0 or 1" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/archive" "$OUTPUT_ROOT/catalog" "$OUTPUT_ROOT/logs"

failed_years=()
completed_years=()

for ((year=END_YEAR; year>=START_YEAR; year--)); do
  hdf5_path="$HDF5_ROOT/$year/japan_${year}.hdf5"
  archive_path="$OUTPUT_ROOT/archive/hinet_raw_${year}.h5"
  inventory_path="$OUTPUT_ROOT/catalog/hinet_stations.csv"
  match_path="$OUTPUT_ROOT/catalog/hinet_kiknet_station_matches_${year}.csv"
  log_path="$OUTPUT_ROOT/logs/download_hinet_${year}.log"

  if [[ ! -f "$hdf5_path" ]]; then
    echo "[ERROR] missing source HDF5 for year $year: $hdf5_path" | tee -a "$log_path" >&2
    failed_years+=("$year")
    continue
  fi

  attempt=1
  year_complete=0
  while :; do
    echo "[INFO] year=$year attempt=$attempt archive=$archive_path" | tee -a "$log_path"
    command=(
      "$PYTHON_BIN" -u tools/download_hinet_velocity.py
      --hdf5 "$hdf5_path"
      --year "$year"
      --mode all
      --storage-mode annual-hdf5
      --archive-path "$archive_path"
      --archive-chunk-bytes "$ARCHIVE_CHUNK_BYTES"
      --output-root "$OUTPUT_ROOT"
      --inventory-csv "$inventory_path"
      --match-csv "$match_path"
      --match-distance-km "$MATCH_DISTANCE_KM"
      --pre-seconds "$PRE_SECONDS"
      --post-seconds "$POST_SECONDS"
      --hinet-timeout-seconds "$HINET_TIMEOUT_SECONDS"
      --hinet-retries "$HINET_RETRIES"
      --station-batch-size "$STATION_BATCH_SIZE"
      --hinet-download-threads "$HINET_DOWNLOAD_THREADS"
      --fallback-span-minutes "$FALLBACK_SPAN_MINUTES"
      --subrequest-sleep-seconds "$SUBREQUEST_SLEEP_SECONDS"
      --sleep-seconds "$SLEEP_SECONDS"
      --no-write-mseed
      --response-mode none
    )
    if [[ "$MINUTE_FALLBACK" == "1" ]]; then
      command+=(--minute-fallback)
    else
      command+=(--no-minute-fallback)
    fi

    "${command[@]}" 2>&1 | tee -a "$log_path"
    rc=${PIPESTATUS[0]}
    if (( rc == 0 )); then
      year_complete=1
      completed_years+=("$year")
      break
    fi

    echo "[WARN] year=$year attempt=$attempt exited with status $rc; committed events will be reused" | tee -a "$log_path" >&2
    if (( MAX_YEAR_ATTEMPTS > 0 && attempt >= MAX_YEAR_ATTEMPTS )); then
      break
    fi
    attempt=$((attempt + 1))
    if (( RETRY_DELAY_SECONDS > 0 )); then
      sleep "$RETRY_DELAY_SECONDS"
    fi
  done

  if (( year_complete == 0 )); then
    failed_years+=("$year")
  fi
done

echo "[INFO] completed years (${#completed_years[@]}): ${completed_years[*]:-none}"
if (( ${#failed_years[@]} > 0 )); then
  echo "[ERROR] incomplete years (${#failed_years[@]}): ${failed_years[*]}" >&2
  echo "[ERROR] rerun the same script to resume only the missing events" >&2
  exit 1
fi

echo "[OK] Hi-net annual archives are complete for $END_YEAR down to $START_YEAR"
