#!/usr/bin/env bash
# LOGIN: shell submission only. Never source private env, run Python or conda,
# scan result seals, hash inputs, or select logical tasks here.
set -Eeuo pipefail
v4s_tool=$(cd -- "${BASH_SOURCE[0]%/*}" && pwd -P)
v4s_root=$(cd -- "$v4s_tool/../.." && pwd -P)
v4s_project=${FE01_V4_PROJECT_ROOT:-$PWD}
v4s_env=${FE01_V4_ENV_FILE:-$v4s_project/on_eval_v4_inventory_fix.private.env}
v4s_run=${FE01_V4_RUN_ID:-on_v4_20261025_inventory_fix}
v4s_logs="$v4s_project/outputs/fe01_on_eval_v4/$v4s_run/operational_submit/logs"
[[ "$v4s_run" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$ ]] || { echo 'Unsafe run ID' >&2; exit 2; }
[[ -f "$v4s_env" ]] || { echo "Existing private env not found: $v4s_env" >&2; exit 2; }
mkdir -p "$v4s_logs"
v4s_args=(--parsable --export=ALL --nodes=1 --ntasks=1 --cpus-per-task=4
  "--mem=${FE01_V4_CONTROL_MEM:-8G}" "--time=${FE01_V4_CONTROL_TIME:-04:00:00}"
  "--chdir=$v4s_project" --job-name=onv4_compute_control
  "--output=$v4s_logs/control_%j.out" "--error=$v4s_logs/control_%j.err")
[[ -z "${FE01_V4_CONTROL_PARTITION:-}" ]] || v4s_args+=("--partition=$FE01_V4_CONTROL_PARTITION")
[[ -z "${FE01_V4_CONTROL_ACCOUNT:-}" ]] || v4s_args+=("--account=$FE01_V4_CONTROL_ACCOUNT")
[[ -z "${FE01_V4_CONTROL_GRES:-}" ]] || v4s_args+=("--gres=$FE01_V4_CONTROL_GRES")
echo "Compute-control logs: $v4s_logs/control_JOBID.out/.err"
exec sbatch "${v4s_args[@]}" "$v4s_tool/control.sbatch" "$v4s_env" "$v4s_run" "$v4s_root" "$@"
