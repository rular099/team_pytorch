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
YEAR=${YEAR:-2024}

export YEAR
export RUN_DITING=${RUN_DITING:-1}
export FINAL_PICK=${FINAL_PICK:-diting_vel_then_acc}
export OUTPUT_DIR=${OUTPUT_DIR:-/opt/zb/data/japan_origin_corrected_diting_vel_acc}
export DIAGNOSTICS_DIR=${DIAGNOSTICS_DIR:-$OUTPUT_DIR/diagnostics_${YEAR}}
export N_DIAGNOSTIC_PLOTS=${N_DIAGNOSTIC_PLOTS:-48}
export VS30_CSV=${VS30_CSV:-$SCRIPT_DIR/resources/vs30/japan_vs30_jshis_station_cache.csv}
export VS30_MAX_DISTANCE_KM=${VS30_MAX_DISTANCE_KM:-1.0}
export REQUIRE_VS30=${REQUIRE_VS30:-0}

if [[ "$RUN_DITING" != "1" ]]; then
  echo "[ERROR] this preset requires RUN_DITING=1 because FINAL_PICK=$FINAL_PICK depends on DiTing picks." >&2
  exit 1
fi

"$SCRIPT_DIR/rebuild_japan_training_data_server.sh"

python - "$OUTPUT_DIR/japan_${YEAR}.hdf5" <<'PY'
import sys
import h5py

path = sys.argv[1]
with h5py.File(path, "r") as f:
    station_meta = f["metadata/station_metadata"]
    methods = station_meta["p_pick_refine_method"][()]
    sources = station_meta["p_pick_refined_source"][()]
    has_vs30 = "vs30" in station_meta and "vs30_valid" in station_meta
    vs30_valid_count = int(station_meta["vs30_valid"][()].sum()) if has_vs30 else 0
    n_stations = len(methods)

decoded_methods = {m.decode() if isinstance(m, bytes) else str(m) for m in methods}
decoded_sources = {s.decode() if isinstance(s, bytes) else str(s) for s in sources}
if decoded_methods != {"diting_vel_then_acc"}:
    raise SystemExit(f"[ERROR] unexpected refine methods: {sorted(decoded_methods)}")
if not decoded_sources.issubset({"diting_vel", "diting_acc"}):
    raise SystemExit(f"[ERROR] unexpected refine sources: {sorted(decoded_sources)}")
if n_stations <= 0:
    raise SystemExit(f"[ERROR] no station rows in rebuilt dataset: {path}")
if not has_vs30:
    raise SystemExit("[ERROR] VS30 metadata missing; check VS30_CSV and rebuild script arguments.")
if vs30_valid_count <= 0:
    raise SystemExit("[ERROR] no valid VS30 rows in rebuilt dataset.")
print(f"[INFO] strict DiTing pick station rows: {n_stations}")
print(f"[INFO] strict DiTing pick sources: {sorted(decoded_sources)}")
print(f"[INFO] valid VS30 station rows: {vs30_valid_count}/{n_stations}")
PY
