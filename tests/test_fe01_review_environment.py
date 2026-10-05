"""Exercise inherited module conflicts and fail-fast setup without HPC access."""
import os
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cluster_shell(tmp_path):
    trace = tmp_path / 'trace'
    prefix = tmp_path / 'conda'
    (prefix / 'bin').mkdir(parents=True)
    interpreter = prefix / 'bin/python'
    interpreter.write_text('''#!/bin/bash
echo python >> "$FAKE_TRACE"
if [ "${FAKE_IMPORT_FAILURE:-0}" = 1 ]; then
  echo 'ImportError: libgalaxyhip.so.5 missing (fixture)' >&2
  exit 1
fi
echo 'FE01_RUNTIME_READY: fixture'
''')
    interpreter.chmod(0o755)
    conda_sh = tmp_path / 'conda.sh'
    conda_sh.write_text('''conda() {
  echo "conda $*" >> "$FAKE_TRACE"
  export CONDA_PREFIX="$FAKE_CONDA_PREFIX"
}
''')
    env_file = tmp_path / 'cluster.env'
    values = dict(FE01_CODE_ROOT=str(ROOT), FE01_REVIEW_OUTPUT_ROOT=str(tmp_path / 'review'),
                  REVIEW_RUN_ID='fixture', FE01_MODULE_LOADS='compiler/rocm/dtk-23.04 apps/miniconda/3',
                  FE01_CONDA_SH=str(conda_sh), FE01_CONDA_ENV='zb')
    env_file.write_text('\n'.join('export ' + k + '=' + shlex.quote(v) for k, v in values.items()))
    driver = tmp_path / 'driver.sh'
    driver.write_text('''#!/bin/bash
module() {
  echo "module $*" >> "$FAKE_TRACE"
  case "$1" in
    purge)
      if [ "${FAKE_PURGE_FAILURE:-0}" = 1 ]; then return 1; fi
      export LOADEDMODULES=''
      ;;
    load)
      if [ "${FAKE_LOAD_FAILURE:-}" = "$2" ]; then
        echo 'ERROR: Tcl module conflict (fixture)' >&2
        return "${FAKE_LOAD_EXIT:-0}"
      fi
      export LOADEDMODULES="${LOADEDMODULES:+$LOADEDMODULES:}$2"
      ;;
  esac
}
source "$FAKE_ENV_SCRIPT"
echo workflow >> "$FAKE_TRACE"
''')
    env = dict(os.environ, FE01_REVIEW_ENV=str(env_file), FAKE_ENV_SCRIPT=str(ROOT / 'scripts/fe01_review/env.sh'),
               FAKE_TRACE=str(trace), FAKE_CONDA_PREFIX=str(prefix), SLURM_JOB_ID='12345',
               LOADEDMODULES='compiler/rocm/2.9')

    def run(**changes):
        return subprocess.run(['bash', str(driver)], env=dict(env, **changes), text=True, capture_output=True)

    def events():
        return trace.read_text().splitlines() if trace.exists() else []

    return run, events


def test_inherited_rocm_is_purged_before_loading_and_workflow(cluster_shell):
    run, events = cluster_shell
    result = run()
    assert result.returncode == 0, result.stderr
    assert events() == ['module purge', 'module load compiler/rocm/dtk-23.04',
                        'module load apps/miniconda/3', 'conda activate zb', 'python', 'workflow']
    modules_line = next(line for line in result.stdout.splitlines() if line.startswith('FE01_MODULES_READY:'))
    assert 'compiler/rocm/dtk-23.04' in modules_line and 'compiler/rocm/2.9' not in modules_line


@pytest.mark.parametrize('exit_code', ['0', '1'])
def test_module_error_stops_even_when_tcl_wrapper_returns_zero(cluster_shell, exit_code):
    run, events = cluster_shell
    result = run(FAKE_LOAD_FAILURE='compiler/rocm/dtk-23.04', FAKE_LOAD_EXIT=exit_code)
    assert result.returncode != 0 and 'FE01_ENV_ERROR' in result.stderr
    assert events() == ['module purge', 'module load compiler/rocm/dtk-23.04']


def test_conda_module_failure_stops_before_activation(cluster_shell):
    run, events = cluster_shell
    result = run(FAKE_LOAD_FAILURE='apps/miniconda/3')
    assert result.returncode != 0 and 'requested module is not loaded' in result.stderr
    assert not any(event.startswith('conda') or event in ('python', 'workflow') for event in events())


def test_purge_failure_stops_before_new_modules(cluster_shell):
    run, events = cluster_shell
    result = run(FAKE_PURGE_FAILURE='1')
    assert result.returncode != 0 and 'module purge failed' in result.stderr
    assert events() == ['module purge']


def test_cpu_pytorch_library_failure_stops_before_workflow(cluster_shell):
    run, events = cluster_shell
    result = run(FAKE_IMPORT_FAILURE='1')
    assert result.returncode != 0 and 'libgalaxyhip.so.5' in result.stderr
    assert 'evaluation has not started' in result.stderr
    assert events()[-1] == 'python' and 'workflow' not in events()
