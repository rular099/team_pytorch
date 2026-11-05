"""Operational controls, not production model/DCU verification."""
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'scripts/fe01_v4_team_smoke_fix'
sys.path.insert(0, str(TOOLS))
import fix_support as support
import submit as repair_submit


def test_callback_binding_reuses_bytecode_and_preserves_original_globals():
    namespace = {'verify_smoke': lambda: 'original', 'sentinel': object()}
    exec('def runner_task():\n    return verify_smoke(), sentinel\n', namespace)
    original = namespace['runner_task']; before = dict(original.__globals__)
    revised = support.bind_smoke_callback(original, lambda: 'controlled')
    assert revised.__code__ is original.__code__
    assert original()[0] == 'original' and revised()[0] == 'controlled'
    assert all(original.__globals__[k] is v for k, v in before.items())
    assert all(revised.__globals__[k] is v for k, v in before.items() if k != 'verify_smoke')
    from fe01_a01_compare import replay
    binding = replay.verify_smoke
    real_copy = support.bind_smoke_callback(replay.runner_task, lambda *args: None)
    assert real_copy.__code__ is replay.runner_task.__code__
    assert replay.runner_task.__globals__['verify_smoke'] is binding


@pytest.mark.parametrize('a,b,passed', [
    ([np.array([1., 2.], np.float32)], [np.array([1., 2.], np.float32)], True),
    ([np.array([1., 2.])], [np.array([1.001, 2.])], False),
    ([np.array([1e8])], [np.array([1e8 + .1])], False),  # No relative tolerance waiver.
    ([np.array([np.nan])], [np.array([np.nan])], False),
    ([np.array([np.inf])], [np.array([np.inf])], False),
    ([np.array([1.])], [], False),
    ([], [], False)])
def test_feature_comparison_keeps_absolute_threshold_and_rejects_missing_nonfinite(a, b, passed):
    assert support.feature_comparison(a, b)['passed'] is passed


def fake_torch_and_original_paired_execution(monkeypatch):
    torch = SimpleNamespace(
        backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=True, deterministic=False, allow_tf32=True),
                                 cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))),
        device=lambda name: SimpleNamespace(type='cuda'),
        cuda=SimpleNamespace(device_count=lambda: 1, is_initialized=lambda: False),
        random=SimpleNamespace(fork_rng=lambda **kwargs: contextlib.nullcontext()))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    spec = importlib.util.spec_from_file_location('unchanged_a01_equivalence_test', ROOT / 'fe01_a01/equivalence.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return torch, module.paired_execution


def frame():
    return pd.DataFrame([dict(dataset_id='synthetic.hdf5', event_id='SYNTHETIC', geometry_protocol='normal',
                              elapsed_time=1, station_id='S00', prediction=.1, mdn_logits='[0.0]',
                              mdn_mu='[0.1]', mdn_sigma='[1.0]')])


class FakeForward:
    """A transparent synthetic feature source; not a production model stub PASS."""
    def __init__(self, unstable=False):
        self.device = 'cuda'; self.calls = 0; self.unstable = unstable
        self.runner = SimpleNamespace(capture_features=self.capture_features)
        self.active = None

    @contextlib.contextmanager
    def capture_features(self):
        assert self.active is None
        self.active = []
        try:
            yield self.active
        finally:
            self.active = None

    def predict(self, event, group):
        self.calls += 1
        value = event.get('feature', 1.) + (self.calls * .1 if self.unstable else 0.)
        if self.active is not None:
            self.active.append(np.array([[value]], dtype=np.float32))
        return frame(), {}


def test_controlled_audit_calls_original_gate_and_restores_backend(monkeypatch, tmp_path):
    torch, paired = fake_torch_and_original_paired_execution(monkeypatch)
    before = support.backend_settings(torch); called = []
    def original(forward, event, group, legacy, family, checks):
        called.append(1)
        assert support.backend_settings(torch) == dict(cudnn_benchmark=False, cudnn_deterministic=True,
                                                       cudnn_allow_tf32=False, matmul_allow_tf32=False)
        for i in range(4):
            with forward.runner.capture_features():
                forward.predict(event, group)
        checks.append({'original_gates': 'executed unchanged callback'})
    verify = support.controlled_smoke(original, paired, torch, tmp_path)
    checks = []; verify(FakeForward(), {}, frame(), None, 'TEAM', checks)
    assert called == [1] and checks[-1]['check'] == 'TEAM_controlled_same_input_repeat'
    assert support.backend_settings(torch) == before
    directory = next(tmp_path.iterdir())
    assert json.loads((directory / '99_pass.json').read_text())['backend_restored']
    assert len(list(directory.glob('*future_*.json'))) == 3


def test_blank_repeat_failure_does_not_claim_future_leak_or_call_original(monkeypatch, tmp_path):
    torch, paired = fake_torch_and_original_paired_execution(monkeypatch)
    before = support.backend_settings(torch); called = []
    verify = support.controlled_smoke(lambda *args: called.append(1), paired, torch, tmp_path)
    with pytest.raises(ValueError, match='SAME_INPUT_REPEATABILITY_FAILED'):
        verify(FakeForward(unstable=True), {}, frame(), None, 'TEAM', [])
    assert not called and support.backend_settings(torch) == before
    directory = next(tmp_path.iterdir())
    report = json.loads((directory / '02_same_input_repeat.json').read_text())
    assert report['features']['max_abs'] > support.ATOL
    assert json.loads((directory / '99_failure.json').read_text())['backend_restored']
    assert not (directory / '99_pass.json').exists()


def test_future_failure_preserves_original_error_and_diagnostics(monkeypatch, tmp_path):
    torch, paired = fake_torch_and_original_paired_execution(monkeypatch)
    before = support.backend_settings(torch)
    def original(forward, event, group, legacy, family, checks):
        with forward.runner.capture_features():
            forward.predict(event, group)
        with forward.runner.capture_features():
            forward.predict({'feature': 1.1}, group)
        raise ValueError('Future changed station encoder features')
    verify = support.controlled_smoke(original, paired, torch, tmp_path)
    with pytest.raises(ValueError, match='Future changed station encoder features'):
        verify(FakeForward(), {}, frame(), None, 'TEAM', [])
    directory = next(tmp_path.iterdir())
    assert not json.loads((directory / '04_future_NaN.json').read_text())['passed']
    assert not (directory / '99_pass.json').exists()
    assert support.backend_settings(torch) == before


def test_wrong_original_source_or_task_mapping_rejected(monkeypatch):
    plan = dict(source_identity=dict(result_commit=support.ORIGINAL_COMMIT, execution_sha256=support.ORIGINAL_EXECUTION),
                tasks={'model_smoke': [{'index': i, 'family': f} for i, f in enumerate(['DiTing', 'PhaseNet', 'EQT', 'TEAM'])]})
    support.validate_plan(plan, ROOT)
    plan['source_identity']['result_commit'] = 'wrong'
    with pytest.raises(ValueError, match='exact 50672e1'):
        support.validate_plan(plan, ROOT)
    plan['source_identity']['result_commit'] = support.ORIGINAL_COMMIT
    plan['tasks']['model_smoke'][3]['family'] = 'EQT'
    with pytest.raises(ValueError, match='task mapping differs'):
        support.validate_plan(plan, ROOT)


def fake_submit_setup(tmp_path, monkeypatch):
    from fe01_a01_compare import orchestration as ops
    from test_fe01_on_runtime import fake_scheduler
    log = fake_scheduler(tmp_path, monkeypatch)
    session = tmp_path / 'run'; session.mkdir(); (session / 'jobs').mkdir(); (session / 'slurm').mkdir()
    env = tmp_path / 'private.env'; env.write_text('# fake trusted operational paths\n')
    plan = {'env_file': str(env)}
    monkeypatch.setenv('ON_EVAL_RELEASE_ROOT', str(ROOT))
    monkeypatch.setattr(repair_submit, 'validate_plan', lambda *args: None)
    monkeypatch.setattr(repair_submit, 'operational_identity', lambda: dict(revision_sha256='revision', commit='repair-commit'))
    monkeypatch.setattr(ops, 'verify_plan', lambda s: plan)
    monkeypatch.setattr(ops, 'prerequisites', lambda *args: None)
    monkeypatch.setattr(ops, 'completed', lambda *args: False)
    return ops, log, session, env


def test_fake_scheduler_submits_only_three_and_records_revision(tmp_path, monkeypatch):
    ops, log, session, env = fake_submit_setup(tmp_path, monkeypatch)
    assert repair_submit.submit(session, env) == '999'
    assert '--array=3%1' in log.read_text()
    job = json.loads(next((session / 'jobs').glob('*.json')).read_text())
    assert job['stage'] == 'model_smoke' and job['indices'] == [3]
    assert job['runtime_revision_sha256'] == 'revision'
    monkeypatch.setenv('FAKE_QUEUE', '999_3|RUNNING')
    assert repair_submit.submit(session, env) is None
    assert len(log.read_text().splitlines()) == 1
    monkeypatch.setattr(ops, 'completed', lambda *args: True)
    assert repair_submit.submit(session, env) is None
    assert len(log.read_text().splitlines()) == 1


def test_unknown_scheduler_state_cannot_trigger_duplicate_repair(tmp_path, monkeypatch):
    ops, log, session, env = fake_submit_setup(tmp_path, monkeypatch)
    repair_submit.submit(session, env)
    with pytest.raises(ValueError, match='UNKNOWN'):
        repair_submit.submit(session, env)
    assert len(log.read_text().splitlines()) == 1


def test_small_release_identity_rejects_tampered_operational_source(tmp_path):
    root = tmp_path / 'release'
    f = root / 'scripts/fe01_v4_team_smoke_fix/runtime.py'; f.parent.mkdir(parents=True); f.write_text('# original\n')
    marker = dict(commit='repair-commit', original_commit=support.ORIGINAL_COMMIT,
                  original_execution_sha256=support.ORIGINAL_EXECUTION,
                  files_sha256={str(f.relative_to(root)): support.digest(f)})
    (root / 'TEAM_SMOKE_FIX_RELEASE.json').write_text(json.dumps(marker))
    assert support.operational_identity(root)['commit'] == 'repair-commit'
    f.write_text('# altered\n')
    with pytest.raises(ValueError, match='source changed'):
        support.operational_identity(root)


def test_new_runtime_evidence_is_sealed_and_original_status_keys_still_validate(tmp_path):
    from fe01_a01_compare.io import attempt, verify_seal, write_json
    original = dict(stage='model_smoke', task=3, execution_sha256=support.ORIGINAL_EXECUTION)
    revised = dict(original, runtime_revision_sha256='independent-repair-source')
    with attempt(tmp_path, 'model_smoke', 3, revised) as out:
        write_json(out / 'operational_execution.json', {'revision_sha256': 'independent-repair-source'})
    path = tmp_path / 'stages/model_smoke/3'
    assert verify_seal(path, original)['runtime_revision_sha256'] == 'independent-repair-source'
    (path / 'operational_execution.json').write_text('tampered')
    with pytest.raises(ValueError, match='Sealed files changed'):
        verify_seal(path, original)


def test_scripts_parse_and_import_does_not_initialize_torch():
    for path in TOOLS.iterdir():
        if path.suffix in ('.sh', '.sbatch'):
            subprocess.run(['bash', '-n', str(path)], check=True)
    code = 'import sys;sys.path.insert(0,' + repr(str(TOOLS)) + ');import fix_support,submit,team_smoke_runtime;assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', code], check=True)


@pytest.mark.parametrize('mode', ['no_arguments', 'override_run', 'custom_env', 'equals_env'])
def test_submit_shell_preserves_arguments_with_empty_or_clobbered_args_variable(tmp_path, mode):
    # Exercise the actual shell entrypoint. The interpreter replacement only
    # records argv; it cannot submit a scheduler task or import a model.
    capture = tmp_path / 'argv.json'
    fake_python = tmp_path / 'record_python'
    fake_python.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
        'from pathlib import Path\nPath(os.environ["TEAM_SMOKE_CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n')
    fake_python.chmod(0o755)
    env_file = tmp_path / ('private file.env' if mode in ('custom_env', 'equals_env')
                           else 'on_eval_v4_inventory_fix.private.env')
    env_file.write_text('export ON_EVAL_RELEASE_ROOT=/original/release\n'
        'export ON_EVAL_INHERITED_CODE_ROOT=/original/inherited\n'
        'export ON_EVAL_CONFIG_PYTHON=' + shlex.quote(str(fake_python)) + '\n'
        'args=("unrelated private environment argument")\n'
        'tool_dir=/unrelated/env/value\nenv_file=/unrelated/env/value\n')
    supplied = []
    if mode == 'override_run':
        supplied = ['--run-id', 'explicit_run']
    elif mode == 'custom_env':
        supplied = ['--env', str(env_file), '--run-id', 'explicit_run']
    elif mode == 'equals_env':
        supplied = ['--env=' + str(env_file)]
    env = dict(os.environ, TEAM_SMOKE_CAPTURE=str(capture))
    subprocess.run(['bash', str(TOOLS / 'submit_team_smoke.sh'), *supplied], cwd=tmp_path, env=env,
                   capture_output=True, text=True, check=True)
    assert json.loads(capture.read_text()) == ['-B', str(TOOLS / 'submit.py'), '--env', str(env_file),
                                              '--run-id', 'on_v4_20261025_inventory_fix', *supplied]


def test_submit_shell_missing_env_value_has_clear_error_before_interpreter(tmp_path):
    result = subprocess.run(['bash', str(TOOLS / 'submit_team_smoke.sh'), '--env'], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 2 and '--env requires a path' in result.stderr
    assert 'unbound variable' not in result.stderr
