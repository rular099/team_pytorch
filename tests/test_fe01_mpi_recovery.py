"""Test configuration-only recovery and submission boundaries without HPC."""
import hashlib
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECOVER = ROOT / 'scripts/fe01_diagnostics/fe01_recover_mpi_20261006.sh'


@pytest.fixture
def cluster(tmp_path):
    root = tmp_path / 'project'
    scripts = root / 'scripts/fe01_review'
    scripts.mkdir(parents=True)
    for name in ('env.sh', 'identity_job.sbatch'):
        (scripts / name).write_bytes((ROOT / 'scripts/fe01_review' / name).read_bytes())
    calls = tmp_path / 'calls'
    submit = scripts / 'submit.sh'
    submit.write_text('''#!/bin/bash
set -eu
source "$2"
printf '%s|%s|%s|%s\\n' "$1" "$REVIEW_RUN_ID" "$FE01_MODULE_LOADS" "$DATA_ROOT" >> "$RECOVERY_TEST_CALLS"
echo 'SUBMITTED identity JobID=fixture'
''')
    env_file = scripts / 'cluster_zb.env'
    original = '\n'.join('export '+key+'='+shlex.quote(value) for key,value in dict(
        FE01_CODE_ROOT=str(root),FE01_REVIEW_OUTPUT_ROOT=str(root/'review'),REVIEW_RUN_ID='failed_original',
        FE01_SUBMIT_PYTHON=sys.executable,FE01_MODULE_LOADS='compiler/rocm/dtk-23.04 apps/miniconda/3',
        DATA_ROOT='user customized data path').items())+'\n'
    env_file.write_text(original)
    log = root / 'fe01_envdiag_29351034.out'
    lines = ['FE01_DIAG_VERSION=20261006-1','FE01_DIAG_JOB=29351034 HOST=fixture SUBMIT_DIR='+str(root),
             'FE01_DIAG_CODE_ROOT='+str(root),'FE01_DIAG_ENV_FILE='+str(env_file)]
    for name in ('env.sh','identity_job.sbatch'):
        path=scripts/name;digest=hashlib.sha256(path.read_bytes()).hexdigest()
        lines.extend(['FE01_DIAG_FILE='+str(path),'FE01_DIAG_EXPECTED_SHA='+digest+' ACTUAL_SHA='+digest,
                      'FE01_DIAG_FILE_MATCH=YES'])
    lines.extend(['FE01_DIAG_MODULES_INHERITED=compiler/devtoolset/7.3.1:compiler/rocm/2.9:mpi/hpcx/2.11.0/gcc-7.3.1:apps/tmux/gcc-7.3.1:apps/miniconda/3',
                  'FE01_DIAG_MODULES_READY=compiler/rocm/dtk-23.04:apps/miniconda/3'])
    log.write_text('\n'.join(lines)+'\n')

    def run():
        return subprocess.run(['bash',str(RECOVER),str(log)],cwd=root,
            env=dict(os.environ,RECOVERY_TEST_CALLS=str(calls)),text=True,capture_output=True)

    return dict(run=run,root=root,scripts=scripts,env=env_file,original=original,log=log,calls=calls)


def test_restore_diagnosed_mpi_preserves_user_settings_and_submits_identity(cluster):
    c=cluster;result=c['run']();assert result.returncode==0,result.stderr
    backup=Path(str(c['env'])+'.before_mpi_29351034')
    assert backup.read_text()==c['original']
    assert c['env'].read_text().startswith(c['original'])
    records=c['calls'].read_text().splitlines();assert len(records)==1
    action,run_id,modules,data=records[0].split('|')
    assert action=='identity' and run_id=='fe01_eval1_mpi_29351034'
    assert modules.split()==['compiler/devtoolset/7.3.1','compiler/rocm/dtk-23.04','mpi/hpcx/2.11.0/gcc-7.3.1','apps/miniconda/3']
    assert data=='user customized data path' and 'compiler/rocm/2.9' not in modules
    assert c['run']().returncode!=0 and len(c['calls'].read_text().splitlines())==1


def test_changed_evaluation_script_stops_before_private_config_write(cluster):
    c=cluster;(c['scripts']/'env.sh').write_text('changed source')
    result=c['run']();assert result.returncode!=0 and 'differs' in result.stderr
    assert c['env'].read_text()==c['original'] and not c['calls'].exists()
    assert not list(c['scripts'].glob('*.before_mpi_*'))


def test_wrong_diagnostic_root_does_not_modify_configuration(cluster):
    c=cluster;text=c['log'].read_text()
    c['log'].write_text(text.replace('FE01_DIAG_CODE_ROOT='+str(c['root']), 'FE01_DIAG_CODE_ROOT=/unrelated'))
    result=c['run']();assert result.returncode!=0 and 'does not match' in result.stderr
    assert c['env'].read_text()==c['original'] and not c['calls'].exists()


def test_no_original_mpi_stops_without_guessing_module(cluster):
    c=cluster;c['log'].write_text(c['log'].read_text().replace(':mpi/hpcx/2.11.0/gcc-7.3.1',''))
    result=c['run']();assert result.returncode!=0 and 'no original MPI' in result.stderr
    assert c['env'].read_text()==c['original'] and not c['calls'].exists()


def test_existing_recovery_output_is_not_reused(cluster):
    c=cluster;(c['root']/'review/fe01_eval1_mpi_29351034').mkdir(parents=True)
    result=c['run']();assert result.returncode!=0 and 'already exists' in result.stderr
    assert c['env'].read_text()==c['original'] and not c['calls'].exists()


def test_untrusted_module_name_cannot_become_shell_code(cluster):
    c=cluster;c['log'].write_text(c['log'].read_text().replace('mpi/hpcx/2.11.0/gcc-7.3.1','mpi/x;touch'))
    result=c['run']();assert result.returncode!=0 and 'Invalid module name' in result.stderr
    assert c['env'].read_text()==c['original'] and not c['calls'].exists()
