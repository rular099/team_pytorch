"""Validate standalone troubleshooting without a cluster or changing eval pins."""
import os
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/fe01_diagnostics/fe01_check_environment_20261006.sbatch'


@pytest.fixture
def diagnostic(tmp_path):
    project = tmp_path / 'project'
    scripts = project / 'scripts/fe01_review'
    scripts.mkdir(parents=True)
    for name in ('env.sh', 'identity_job.sbatch'):
        (scripts / name).write_bytes((ROOT / 'scripts/fe01_review' / name).read_bytes())
    trace = tmp_path / 'trace'
    prefix = tmp_path / 'conda'
    (prefix / 'bin').mkdir(parents=True)
    interpreter = prefix / 'bin/python'
    interpreter.write_text('''#!/bin/bash
echo torch_probe >> "$DIAG_TRACE"
if [ "${DIAG_IMPORT_FAIL:-0}" = 1 ]; then
  echo 'ImportError: libgalaxyhip.so.5 (fixture)' >&2
  exit 1
fi
echo 'FE01_DIAG_TORCH=fixture'
''')
    interpreter.chmod(0o755)
    conda_sh = tmp_path / 'conda.sh'
    conda_sh.write_text('conda() { export CONDA_PREFIX="$DIAG_CONDA_PREFIX"; }\n')
    env_file = scripts / 'cluster_zb.env'
    env_file.write_text('\n'.join('export '+key+'='+shlex.quote(value) for key,value in dict(
        FE01_CODE_ROOT=str(project),FE01_CONDA_SH=str(conda_sh),FE01_CONDA_ENV='zb',
        FE01_MODULE_LOADS='compiler/rocm/dtk-23.04 apps/miniconda/3').items()))
    driver = tmp_path / 'driver.sh'
    driver.write_text('''#!/bin/bash
module() {
  echo "module $*" >> "$DIAG_TRACE"
  case "$1" in
    purge)
      if [ "${DIAG_PURGE_RETAINS:-0}" != 1 ]; then export LOADEDMODULES=''; fi
      ;;
    unload)
      if [ "${DIAG_UNLOAD_FAIL:-0}" != 1 ]; then export LOADEDMODULES=''; fi
      ;;
    load)
      if [ "${DIAG_MODULE_FAIL:-}" = "$2" ]; then
        echo 'ERROR: Tcl conflict (fixture)' >&2
        return 0
      fi
      export LOADEDMODULES="${LOADEDMODULES:+$LOADEDMODULES:}$2"
      ;;
  esac
}
source "$DIAG_SCRIPT"
''')
    env = dict(os.environ,SLURM_SUBMIT_DIR=str(project),SLURM_JOB_ID='fixture',
        FE01_CODE_ROOT=str(project),FE01_REVIEW_ENV=str(env_file),DIAG_TRACE=str(trace),
        DIAG_CONDA_PREFIX=str(prefix),DIAG_SCRIPT=str(SCRIPT),LOADEDMODULES='compiler/rocm/2.9')

    def run(**changes):
        return subprocess.run(['bash',str(driver)],env=dict(env,**changes),text=True,capture_output=True)

    def events():
        return trace.read_text().splitlines() if trace.exists() else []

    return run, events, scripts


def test_matching_files_and_clean_runtime_report_pass(diagnostic):
    run,events,_ = diagnostic
    result=run()
    assert result.returncode==0,result.stderr
    assert 'FE01_DIAG_VERSION=20261006-1' in result.stdout
    assert result.stdout.count('FE01_DIAG_FILE_MATCH=YES')==2
    assert 'FE01_DIAG_MODULES_INHERITED=compiler/rocm/2.9' in result.stdout
    assert 'FE01_DIAG_TORCH_IMPORT=PASS' in result.stdout and 'FE01_DIAG_RESULT=PASS' in result.stdout
    assert events()==['module purge','module load compiler/rocm/dtk-23.04','module load apps/miniconda/3','torch_probe']


def test_old_env_file_is_reported_without_executing_it(diagnostic):
    run,events,scripts=diagnostic
    (scripts/'env.sh').write_text('echo old_env_was_executed >> "$DIAG_TRACE"\n')
    result=run()
    assert result.returncode==3
    assert 'FE01_DIAG_FILE_MATCH=NO' in result.stdout and 'FE01_DIAG_RESULT=SCRIPT_VERSION_MISMATCH' in result.stdout
    assert 'old_env_was_executed' not in events()


def test_tcl_zero_return_conflict_stops_before_torch(diagnostic):
    run,events,_=diagnostic
    result=run(DIAG_MODULE_FAIL='compiler/rocm/dtk-23.04')
    assert result.returncode==2 and 'requested module absent' in result.stderr
    assert 'torch_probe' not in events()


def test_rocm_retained_after_purge_is_explicitly_unloaded(diagnostic):
    run,events,_=diagnostic
    result=run(DIAG_PURGE_RETAINS='1')
    assert result.returncode==0,result.stderr
    assert 'FE01_DIAG_EXPLICIT_UNLOAD=compiler/rocm/2.9' in result.stdout
    assert events()[:2]==['module purge','module unload compiler/rocm/2.9']


def test_unload_that_returns_zero_but_retains_module_stops(diagnostic):
    run,events,_=diagnostic
    result=run(DIAG_PURGE_RETAINS='1',DIAG_UNLOAD_FAIL='1')
    assert result.returncode==2 and 'remains loaded' in result.stderr
    assert not any(event.startswith('module load ') or event=='torch_probe' for event in events())


def test_shared_library_failure_is_only_a_probe_failure(diagnostic):
    run,events,_=diagnostic
    result=run(DIAG_IMPORT_FAIL='1')
    assert result.returncode==2 and 'libgalaxyhip.so.5' in result.stderr
    assert 'FE01_DIAG_RESULT=PASS' not in result.stdout
    assert events()[-1]=='torch_probe'
