#!/usr/bin/env bash
# Local/CPU only. Does not initialize a model, access production data or submit jobs.
set -Eeuo pipefail
python_bin=${ON_EVAL_TEST_PYTHON:-python3}
release_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$release_root"
export PYTHONPATH="${ON_EVAL_TEST_DEPS:+$ON_EVAL_TEST_DEPS:}$release_root${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/fe01-on-eval-mpl}"
export PYTHONDONTWRITEBYTECODE=1
"$python_bin" -B -m compileall -q fe01_a01_compare
"$python_bin" -B - <<'PY'
import importlib,pkgutil,sys
import fe01_a01_compare
for module in pkgutil.iter_modules(fe01_a01_compare.__path__):
    importlib.import_module('fe01_a01_compare.'+module.name)
assert 'torch' not in sys.modules, 'CPU module imports must not initialize model libraries'
print('IMPORT_ALL_TORCH_FREE_PASS')
PY
while IFS= read -r -d '' script; do bash -n "$script"; done < <(find scripts/fe01_a01_compare -type f \( -name '*.sh' -o -name '*.sbatch' \) -print0)
echo BASH_SYNTAX_PASS
"$python_bin" -B -m pytest -q tests/test_fe01_on_*.py
