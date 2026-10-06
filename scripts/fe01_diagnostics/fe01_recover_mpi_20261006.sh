#!/bin/bash
# Run manually from the FE01 project root. Restores compiler/MPI modules
# recorded in the diagnostic, saves the old private env, then submits identity.
set -euo pipefail
fe01_recovery_log="${1:?Usage: bash fe01_recover_mpi_20261006.sh fe01_envdiag_JOBID.out}"
fe01_recovery_env="${2:-$PWD/scripts/fe01_review/cluster_zb.env}"
export FE01_MPI_RECOVERY_ENV="$fe01_recovery_env"
source "$fe01_recovery_env"
fe01_recovery_python="${FE01_SUBMIT_PYTHON:-python3}"
if [ -z "${FE01_SUBMIT_PYTHON:-}" ] && [ -x "$HOME/.conda/envs/${FE01_CONDA_ENV:-zb}/bin/python" ]; then
  fe01_recovery_python="$HOME/.conda/envs/${FE01_CONDA_ENV:-zb}/bin/python"
fi
"$fe01_recovery_python" - "$fe01_recovery_log" <<'PY'
import hashlib
import json
import os
import re
import shlex
import sys
from pathlib import Path

try:
    log = Path(sys.argv[1])
    lines = log.read_text().splitlines()
    values = dict(line.split('=', 1) for line in lines if line.startswith('FE01_DIAG_') and '=' in line)
    if values.get('FE01_DIAG_VERSION') != '20261006-1':
        raise ValueError('Unrecognized diagnostic version')
    job = values['FE01_DIAG_JOB'].split()[0]
    if not re.fullmatch(r'\d+', job):
        raise ValueError('Diagnostic Job ID must be numeric')
    root = Path(os.environ['FE01_CODE_ROOT']).resolve()
    env_file = Path(os.environ['FE01_MPI_RECOVERY_ENV']).resolve()
    if Path(values['FE01_DIAG_CODE_ROOT']).resolve() != root or Path(values['FE01_DIAG_ENV_FILE']).resolve() != env_file:
        raise ValueError('Diagnostic root/private env does not match this invocation')
    verified = set()
    filename = None
    for line in lines:
        if line.startswith('FE01_DIAG_FILE='):
            filename = Path(line.split('=', 1)[1]).resolve()
        if line.startswith('FE01_DIAG_EXPECTED_SHA='):
            match = re.fullmatch(r'FE01_DIAG_EXPECTED_SHA=([0-9a-f]{64}) ACTUAL_SHA=([0-9a-f]{64})', line)
            if not match or filename is None or filename.parent != root / 'scripts/fe01_review' or filename.name not in ('env.sh', 'identity_job.sbatch'):
                raise ValueError('Unexpected diagnostic source record')
            expected, actual = match.groups()
            if expected != actual or hashlib.sha256(filename.read_bytes()).hexdigest() != actual:
                raise ValueError('Installed evaluation script differs from the diagnosed envfix version')
            verified.add(filename.name)
    if verified != {'env.sh', 'identity_job.sbatch'}:
        raise ValueError('Two verified evaluation source records are required')
    desired = os.environ['FE01_MODULE_LOADS'].split()
    if set(desired) != set(values['FE01_DIAG_MODULES_READY'].split(':')):
        raise ValueError('Declared modules changed since the diagnostic')
    inherited = values['FE01_DIAG_MODULES_INHERITED'].split(':')
    compiler = [m for m in inherited if m.startswith('compiler/') and not m.startswith('compiler/rocm/')]
    mpi = [m for m in inherited if m.startswith('mpi/')]
    if not mpi:
        raise ValueError('Diagnostic contains no original MPI module; do not guess a replacement')
    merged = list(dict.fromkeys(compiler + [m for m in desired if m.startswith('compiler/')] + mpi +
                                [m for m in desired if not m.startswith('compiler/')]))
    if not all(re.fullmatch(r'[A-Za-z0-9_./+\-]+', m) for m in merged):
        raise ValueError('Invalid module name in diagnostic')
    run_id = 'fe01_eval1_mpi_' + job
    output = Path(os.environ['FE01_REVIEW_OUTPUT_ROOT']) / run_id
    if output.exists():
        raise ValueError('Recovery output already exists; use the regular stage submission scripts')
    original = env_file.read_text()
    backup = Path(str(env_file) + '.before_mpi_' + job)
    receipt = Path(str(env_file) + '.mpi_recovery_' + job + '.receipt')
    if backup.exists() or receipt.exists():
        raise ValueError('MPI recovery already prepared; use submit.sh for subsequent stages')
    # Keep every existing setting; append only the module and fresh run ID overrides.
    with backup.open('x') as handle:
        handle.write(original)
    updated = original.rstrip() + '\n\n# MPI recovery from diagnostic Job ' + job + '\n'
    updated += 'export FE01_MODULE_LOADS=' + shlex.quote(' '.join(merged)) + '\n'
    updated += 'export REVIEW_RUN_ID=' + shlex.quote(run_id) + '\n'
    temp = Path(str(env_file) + '.mpi_recovery_' + job + '.tmp')
    with temp.open('x') as handle:
        handle.write(updated)
    temp.chmod(env_file.stat().st_mode & 0o777)
    os.replace(temp, env_file)
    receipt.write_text(json.dumps(dict(diagnostic_job=job,diagnostic_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
        original_env_sha256=hashlib.sha256(original.encode()).hexdigest(),updated_env_sha256=hashlib.sha256(updated.encode()).hexdigest(),
        restored_modules=merged,review_run_id=run_id),indent=2)+'\n')
    print('FE01_MPI_RECOVERY_MODULES=' + ' '.join(merged))
    print('FE01_MPI_RECOVERY_RUN=' + run_id)
    print('FE01_MPI_RECOVERY_BACKUP=' + str(backup))
except (KeyError, ValueError, OSError) as exc:
    print('FE01_MPI_RECOVERY_STOPPED: ' + str(exc), file=sys.stderr)
    sys.exit(2)
PY
exec bash "$FE01_CODE_ROOT/scripts/fe01_review/submit.sh" identity "$fe01_recovery_env"
