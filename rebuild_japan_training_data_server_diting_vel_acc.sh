#!/usr/bin/env bash
set -euo pipefail

# Server-side preset for rebuilding a Japan training dataset whose training
# p_picks come only from DiTing picks:
#   1. use p_pick_diting_vel_aligned when available;
#   2. otherwise use p_pick_diting_acc_aligned;
#   3. drop stations without either DiTing pick before writing HDF5.
#
# The full rebuild logic, paths, origin-correction fetch, and HDF5 sanity check
# live in rebuild_japan_training_data_server.sh. Override any variable before
# calling this script when the server paths differ.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONVERTED_ROOT=${CONVERTED_ROOT:-/public/home/zhangbei/work_dir/zhangbei/japan_knet_converted}
OUTPUT_VARIANT=${OUTPUT_VARIANT:-origin_corrected_diting_vel_acc_vs30}
if [[ -z "${YEARS:-}" ]]; then
  if [[ -n "${YEAR:-}" ]]; then
    YEARS="$YEAR"
  else
    YEARS=$(seq -s ' ' 2000 2024)
  fi
fi

export YEARS
export CONVERTED_ROOT
export OUTPUT_VARIANT
export RUN_DITING=${RUN_DITING:-1}
export FINAL_PICK=${FINAL_PICK:-diting_vel_then_acc}
export N_DIAGNOSTIC_PLOTS=${N_DIAGNOSTIC_PLOTS:-48}
export VS30_CSV=${VS30_CSV:-$SCRIPT_DIR/resources/vs30/japan_vs30_jshis_station_cache.csv}
export VS30_MAX_DISTANCE_KM=${VS30_MAX_DISTANCE_KM:-1.0}
export REQUIRE_VS30=${REQUIRE_VS30:-1}

if [[ "$RUN_DITING" != "1" ]]; then
  echo "[ERROR] this preset requires RUN_DITING=1 because FINAL_PICK=$FINAL_PICK depends on DiTing picks." >&2
  exit 1
fi

"$SCRIPT_DIR/rebuild_japan_training_data_server.sh"

python - "$CONVERTED_ROOT" "$OUTPUT_VARIANT" "$YEARS" "${OUTPUT_DIR:-}" <<'PY'
import sys
import h5py
from pathlib import Path

converted_root = Path(sys.argv[1])
output_variant = sys.argv[2]
years = sys.argv[3].split()
explicit_output_dir = sys.argv[4]

for year in years:
    output_dir = Path(explicit_output_dir) if explicit_output_dir else converted_root / output_variant / year
    path = output_dir / f"japan_{year}.hdf5"
    with h5py.File(path, "r") as f:
        station_meta = f["metadata/station_metadata"]
        methods = station_meta["p_pick_refine_method"][()]
        sources = station_meta["p_pick_refined_source"][()]
        has_vs30_value = "vs30_mps" in station_meta or "vs30" in station_meta
        has_vs30_valid = "vs30_valid" in station_meta
        vs30_valid_count = int(station_meta["vs30_valid"][()].sum()) if has_vs30_valid else 0
        n_stations = len(methods)
        missing_event_vs30 = [
            event_id
            for event_id, group in f["data"].items()
            if "vs30" not in group or "vs30_valid" not in group
        ]

    decoded_methods = {m.decode() if isinstance(m, bytes) else str(m) for m in methods}
    decoded_sources = {s.decode() if isinstance(s, bytes) else str(s) for s in sources}
    if decoded_methods != {"diting_vel_then_acc"}:
        raise SystemExit(f"[ERROR] {year}: unexpected refine methods: {sorted(decoded_methods)}")
    if not decoded_sources.issubset({"diting_vel", "diting_acc"}):
        raise SystemExit(f"[ERROR] {year}: unexpected refine sources: {sorted(decoded_sources)}")
    if n_stations <= 0:
        raise SystemExit(f"[ERROR] {year}: no station rows in rebuilt dataset: {path}")
    if not has_vs30_value or not has_vs30_valid:
        raise SystemExit(f"[ERROR] {year}: VS30 metadata missing; check VS30_CSV and rebuild script arguments.")
    if vs30_valid_count != n_stations:
        raise SystemExit(f"[ERROR] {year}: VS30 is not complete: {vs30_valid_count}/{n_stations} valid station rows.")
    if missing_event_vs30:
        raise SystemExit(f"[ERROR] {year}: {len(missing_event_vs30)} event groups are missing VS30 datasets.")
    print(f"[INFO] {year}: strict DiTing pick station rows: {n_stations}")
    print(f"[INFO] {year}: strict DiTing pick sources: {sorted(decoded_sources)}")
    print(f"[INFO] {year}: valid VS30 station rows: {vs30_valid_count}/{n_stations}")
PY
