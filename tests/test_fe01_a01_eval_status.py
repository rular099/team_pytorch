"""Status must distinguish scheduler acceptance from authenticated completion."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import pytest

from scripts.fe01_a01_ops import eval_status as status, eval_pack as ep, eval_revision
from test_fe01_a01_eval_pack_ops import fixture, prepare, seal_result
from test_fe01_a01_team_eval_fix import legacy_session, revised_result, pins
from test_fe01_a01_scheduler import ROOT
from fe01.config import sha256


def report(session, states=None):
    lines = []
    result = status.inspect(session, ep=ep, query=lambda _: (states or {}, []), emit=lines.append)
    return result, '\n'.join(lines)


def test_latest_TEAM_retry_overrides_old_failure_and_OFF_waits_for_ON(tmp_path):
    env, log, project, tool, *_ = fixture(tmp_path)
    session = prepare(env, project, tool)
    (session/'jobs.tsv').write_text('eval_first\t0,1,2,3\t100;local\neval_team\t0\t200\n')
    for index in (1, 2):
        seal_result(session, index)
    with patch.dict(os.environ, env):
        value, output = report(session, {'100_0': {'state': 'FAILED'}, '200_0': {'state': 'RUNNING'},
                                         '100_3': {'state': 'PENDING'}})
    assert [row['status'] for row in value['rows']] == ['RUNNING', 'COMPLETE', 'COMPLETE', 'QUEUED', 'NO_RECORD']
    assert value['rows'][0]['submission']['task'] == '200_0'
    assert not value['pack_ready'] and 'PACK_READY=NO' in output
    assert '完整通过后，再提交' in output and not log.exists()


def test_partial_exports_and_scheduler_completed_never_mean_success(tmp_path):
    env, log, project, tool, *_ = fixture(tmp_path)
    session = prepare(env, project, tool)
    (session/'jobs.tsv').write_text('eval_first\t0,1,2,3\t100\n')
    (session/'attempts/partial/selected').mkdir(parents=True)
    (session/'attempts/partial/selected/scores.csv').write_text('SYNTHETIC_PARTIAL')
    (session/'results/index1').mkdir(parents=True)
    with patch.dict(os.environ, env):
        value, output = report(session, {'100_0': {'state': 'COMPLETED'}, '100_1': {'state': 'COMPLETED'}})
    assert value['rows'][0]['status'] == 'UNSEALED'
    assert value['rows'][1]['status'] == 'INVALID'
    assert value['rows'][2]['status'] == 'UNKNOWN'
    assert 'eval_first_100_0.err' in output and 'PACK_READY=NO' in output and not log.exists()


def test_ON_complete_unlocks_only_OFF_advice_then_all_five_unlock_pack(tmp_path):
    env, log, project, tool, *_ = fixture(tmp_path)
    session = prepare(env, project, tool)
    for index in range(4):
        seal_result(session, index)
    with patch.dict(os.environ, env):
        value, output = report(session)
        assert not value['pack_ready'] and status.OFF_COMMAND in output and status.PACK_COMMAND not in output
        seal_result(session, 4)
        value, output = report(session)
        assert value['pack_ready'] and 'PACK_READY=YES' in output and status.PACK_COMMAND in output
        plan = ep.load_plan(session)
        target = session/'results/index2/evaluation'/plan['runs'][2]['run_id']/'epoch12/scores.csv'
        target.write_text('CORRUPTED_AFTER_SEAL')
        value, output = report(session)
    assert not value['pack_ready'] and value['rows'][2]['status'] == 'INVALID'
    assert status.PACK_COMMAND not in output and not log.exists()


def test_corrected_TEAM_and_four_original_seals_are_readonly_pack_ready(tmp_path):
    env, log, session, tool, _ = legacy_session(tmp_path)
    with patch.dict(os.environ, env):
        directory = eval_revision.register(session, tool)
        for index in (1, 2, 3, 4):
            seal_result(session, index)
        revised_result(session, directory)
        before = pins(session)
        value, output = report(session)
    assert value['pack_ready'] and status.PACK_COMMAND in output
    assert pins(session) == before and not log.exists()


@pytest.mark.parametrize('state', ['PENDING', 'RUNNING', 'FAILED'])
def test_existing_pack_job_is_never_advised_to_submit_again(tmp_path, state):
    env, log, project, tool, *_ = fixture(tmp_path)
    session = prepare(env, project, tool)
    for index in range(5):
        seal_result(session, index)
    (session/'jobs.tsv').write_text('pack\t\t300\n')
    with patch.dict(os.environ, env):
        value, output = report(session, {'300': {'state': state}})
    assert value['pack_ready'] and status.PACK_COMMAND not in output
    assert value['pack_status'] == {'PENDING': 'QUEUED', 'RUNNING': 'RUNNING', 'FAILED': 'FAILED'}[state]
    assert not log.exists()


def test_archive_needs_matching_checksum_and_partial_archive_is_preserved(tmp_path):
    env, log, project, tool, *_ = fixture(tmp_path)
    session = prepare(env, project, tool)
    archive = session/'review.tar.gz';archive.write_bytes(b'SYNTHETIC_ARCHIVE_NOT_HPC_RESULTS')
    with patch.dict(os.environ, env):
        value, output = report(session)
        assert value['pack_status'] == 'INVALID' and status.PACK_COMMAND not in output
        (session/'review.tar.gz.sha256').write_text(sha256(archive)+'  review.tar.gz\n')
        value, output = report(session)
        assert value['pack_status'] == 'COMPLETE' and not value['pack_ready']
        archive.write_bytes(b'TAMPERED')
        value, output = report(session)
    assert value['pack_status'] == 'INVALID' and archive.read_bytes() == b'TAMPERED' and not log.exists()


def test_scheduler_uses_array_task_ids_and_live_queue_overrides_accounting(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert kwargs['timeout'] == 10
        output = '100_0|FAILED|1:0\n100_1|COMPLETED|0:0\n' if command[0] == 'sacct' else '100_0|RUNNING\n'
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr='')
    monkeypatch.setattr(status.subprocess, 'run', run)
    value, warnings = status.scheduler({0: {'job': '100', 'task': '100_0'}})
    assert value['100_0']['state'] == 'RUNNING' and value['100_1']['state'] == 'COMPLETED'
    assert '--array' not in calls[0] and '-r' in calls[1] and not warnings


def test_unavailable_scheduler_keeps_submitted_tasks_unknown(monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError('Slurm command unavailable')
    monkeypatch.setattr(status.subprocess, 'run', unavailable)
    value, warnings = status.scheduler({0: {'job': '100', 'task': '100_0'}})
    assert len(warnings) == 2 and status.pending_status({'task': '100_0'}, value) == 'UNKNOWN'


def test_login_entrypoint_never_imports_torch_submits_jobs_or_changes_session(tmp_path):
    env, log, project, tool, *_ = fixture(tmp_path)
    session = prepare(env, project, tool)
    for index in range(5):
        seal_result(session, index)
    # A dedicated deployed bundle reuses byte-identical gate helpers, not live tools.
    status_dir = tmp_path/'status';status_dir.mkdir()
    for name in ('check_eval_status.sh', 'eval_status.py', 'eval_pack.py', 'eval_revision.py'):
        shutil.copy2(ROOT/'scripts/fe01_a01_ops'/name, status_dir/name)
    deny = tmp_path/'deny';deny.mkdir()
    (deny/'torch.py').write_text('raise RuntimeError("STATUS_IMPORTED_TORCH")\n')
    before = {str(p):sha256(p) for p in session.rglob('*') if p.is_file()}
    env.update(A01_EVAL_SESSION=str(session), A01_CONFIG_PYTHON=sys.executable, PYTHONPATH=str(deny))
    result = subprocess.run(['bash', str(status_dir/'check_eval_status.sh')], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'PACK_READY=YES' in result.stdout and status.PACK_COMMAND in result.stdout
    assert {str(p):sha256(p) for p in session.rglob('*') if p.is_file()} == before
    assert not list(status_dir.rglob('__pycache__')) and not log.exists()


def test_malformed_job_record_blocks_uncertain_status(tmp_path):
    (tmp_path/'jobs.tsv').write_text('eval_team\t0\t$(BAD)\n')
    with pytest.raises(ValueError, match='作业号无效'):
        status.submissions(tmp_path)
