#!/usr/bin/env bash
# LOGIN NODE ONLY: derive private paths; no torch, scheduler or data writes.
set -euo pipefail
legacy=${1:?Usage: bash scripts/fe01_a01/configure.sh ORIGINAL_FE01_CODE [OLD_RUNS_ROOT] [NEW_BATCH_ID]}
legacy=$(cd "$legacy" && pwd)
code=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
old_runs=${2:-$(dirname "$legacy")/fe01_runs_20261002}
batch=${3:-a01_$(date +%Y%m%dT%H%M%S)}
[[ "$batch" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid batch ID' >&2; exit 2; }
[[ "$code" != "$legacy" ]] || { echo 'Extract A01 to its own directory first' >&2; exit 2; }
env_file="$code/a01.private.env"
[[ ! -e "$env_file" ]] || { echo 'Private env exists; preserve it or use a new source directory' >&2; exit 2; }
[[ -f "$old_runs/formal__team_original_scratch__seed42/resolved_config.json" ]] || { echo 'Original results not found' >&2; exit 2; }
# Stdlib only. Resolve actual data/weights from the original resolved config.
python_bin=${A01_CONFIG_PYTHON:-python}
"$python_bin" - "$code" "$legacy" "$old_runs" "$batch" "$env_file" <<'PY'
import json,sys,shlex
from pathlib import Path
code,legacy,runs,batch,dest=map(Path,sys.argv[1:])
cfg=json.loads((runs/'formal__team_original_scratch__seed42/resolved_config.json').read_text())
weights=Path(cfg['pretrained_manifest']).parent
original_manifest=json.loads((weights/'pretrained_manifest.json').read_text())
out=code/'outputs'/str(batch)
diting_manifest=(weights/'pretrained_manifest.json') if 'diting' in original_manifest['models'] else out/'weights/diting_manifest.json'
values=dict(A01_CODE_ROOT=str(code),A01_OUTPUT_ROOT=str(code/'outputs'/str(batch)),A01_BATCH_ID=str(batch),
 FE01_CODE_ROOT=str(legacy),FE01_OUTPUT_ROOT=str(runs),FE01_TRAIN_OUTPUT_ROOT=str(runs),
 DATA_ROOT=cfg['data']['root'],FE01_SPLIT_MANIFEST=cfg['data']['split_manifest'],
 FE01_WEIGHTS_ROOT=str(weights),FE01_SPATIAL_MANIFEST=cfg['spatial']['manifest'],
 A01_CONDA_ENV='zb',A01_MODULES='compiler/devtoolset/7.3.1 compiler/rocm/dtk-23.04 mpi/hpcx/2.11.0/gcc-7.3.1 apps/miniconda/3',
 A01_PARTITION='',A01_TRAIN_NODES='4',A01_DEVICES_PER_NODE='4',A01_CPUS_PER_TASK='2',
 A01_GRES='dcu:4',A01_SINGLE_GRES='dcu:1',A01_TRAIN_MEM='64G',A01_SINGLE_MEM='64G',
 A01_TRAIN_TIME='24:00:00',A01_GATE_TIME='08:00:00',A01_MASTER_PORT='29607',
 A01_DITING_MANIFEST=str(diting_manifest),A01_DITING_CHECKPOINT='',A01_DITING_ENCODER_SHA256='')
Path(dest).write_text(''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in values.items()))
Path(dest).chmod(0o600)
print('Private settings:',dest)
print('NEW outputs:',values['A01_OUTPUT_ROOT'])
print('Check only resource fields against your allocation, then run prepare.sh.')
PY
