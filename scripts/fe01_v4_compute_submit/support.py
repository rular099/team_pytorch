"""External operational identity; no changes to the frozen V4 execution tree."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[2]
MAIN = '50672e131d876a604432b283162d5b0189d40cf0'
EXECUTION = 'b9dd0368adc7041bb0c1e36a6899d49014ac9545ae7f44850274099bab6dc15c'
ALLOWED = ('replay_stations', 'replay_grid_normal', 'latency', 'analyze', 'render_normal', 'pack')
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'RESIZING', 'REQUEUED'}
RETRY = {'NOT_SUBMITTED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def allocated():
    require(bool(os.environ.get('SLURM_JOB_ID')) and bool(os.environ.get('SLURM_STEP_ID')),
            'All Python operations require an allocated srun compute step')


def revision(root=None):
    root = Path(root or ROOT)
    marker = root / 'COMPUTE_SUBMIT_RELEASE.json'
    declared = read(marker)
    actual = {str(p.relative_to(root)): digest(p)
              for p in sorted((root / 'scripts/fe01_v4_compute_submit').rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts}
    require(actual == declared['files_sha256'], 'Operational submission source changed')
    require(declared['main_commit'] == MAIN and declared['main_execution_sha256'] == EXECUTION,
            'Wrong main release for submission package')
    return dict(declared, revision_sha256=fingerprint(declared))


def validate_plan(plan):
    from fe01_a01_compare.contracts import ROOT as original_root
    require(Path(os.environ['ON_EVAL_RELEASE_ROOT']).resolve() == original_root.resolve(), 'Wrong original import root')
    require(plan['source_identity']['result_commit'] == MAIN
            and plan['source_identity']['execution_sha256'] == EXECUTION, 'Exact frozen 50672e1 release required')
    require(plan['run_id'] == os.environ['FE01_V4_RUN_ID'], 'Run ID differs')
    require(Path(plan['env_file']).resolve() == Path(os.environ['FE01_V4_ENV_FILE']).resolve(), 'Private env differs')


def bind_attempt(original, callback):
    namespace = dict(original.__globals__)
    require('attempt' in namespace, 'Original worker has no attempt binding')
    namespace['attempt'] = callback
    result = types.FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    result.__kwdefaults__ = original.__kwdefaults__
    require(result.__code__ is original.__code__ and all(namespace[k] is v for k, v in original.__globals__.items()
                                                        if k != 'attempt'), 'Original worker binding changed')
    return result


def evidence_attempt(original_attempt, evidence):
    @contextlib.contextmanager
    def revised(session, stage, task, identity, **kwargs):
        require(stage == evidence['stage'] and task == evidence['logical_task'], 'Logical task differs from mapping')
        extra = dict(identity, runtime_revision_sha256=evidence['revision']['revision_sha256'],
                     runtime_revision_commit=evidence['revision']['commit'],
                     operational_pool_manifest_sha256=evidence['manifest_sha256'])
        with original_attempt(session, stage, task, extra, **kwargs) as out:
            if out is not None:
                write(Path(out) / 'operational_submission.json', evidence)
            yield out
    return revised
