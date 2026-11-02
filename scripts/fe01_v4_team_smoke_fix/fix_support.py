"""Identity and diagnostics for an external, smoke-only operational revision."""
import contextlib
import hashlib
import json
from pathlib import Path
import types

ORIGINAL_COMMIT = '50672e131d876a604432b283162d5b0189d40cf0'
ORIGINAL_EXECUTION = 'b9dd0368adc7041bb0c1e36a6899d49014ac9545ae7f44850274099bab6dc15c'
TASK = 3
ATOL = 1e-5  # Unchanged original absolute gate, no relative tolerance is used to pass.
ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def operational_identity(root=None):
    root = Path(root or ROOT)
    marker = root / 'TEAM_SMOKE_FIX_RELEASE.json'
    declared = json.loads(marker.read_text())
    actual = {str(p.relative_to(root)): digest(p)
              for p in sorted((root / 'scripts/fe01_v4_team_smoke_fix').rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts}
    require(actual == declared['files_sha256'], 'Operational fix source changed')
    require(declared['original_commit'] == ORIGINAL_COMMIT and
            declared['original_execution_sha256'] == ORIGINAL_EXECUTION, 'Wrong original release for repair')
    return dict(**declared, revision_sha256=fingerprint(declared), marker_sha256=digest(marker))


def validate_plan(plan, execution_root):
    from fe01_a01_compare.contracts import ROOT as original_root
    require(Path(execution_root).resolve() == original_root.resolve(), 'Original release import path differs')
    source = plan['source_identity']
    require(source['result_commit'] == ORIGINAL_COMMIT and source['execution_sha256'] == ORIGINAL_EXECUTION,
            'This repair requires the exact 50672e1 source release')
    require(plan['tasks']['model_smoke'][TASK] == dict(index=TASK, family='TEAM'), 'TEAM task mapping differs')


def bind_smoke_callback(original, callback):
    """Reuse the exact runner bytecode with a private callback namespace.

    No original module/class/global is patched. Every other binding stays the
    identical object; model loading, event selection and inference are unchanged.
    """
    namespace = dict(original.__globals__)
    require('verify_smoke' in namespace, 'Original runner has no smoke callback')
    namespace['verify_smoke'] = callback
    bound = types.FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    bound.__kwdefaults__ = original.__kwdefaults__
    require(bound.__code__ is original.__code__, 'Runner bytecode differs')
    require(set(namespace) == set(original.__globals__) and
            all(namespace[k] is v for k, v in original.__globals__.items() if k != 'verify_smoke'),
            'Unexpected runner binding change')
    return bound


def feature_comparison(reference, candidate):
    import numpy as np
    def array(value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        return np.asarray(value)
    rows = []
    for i in range(max(len(reference), len(candidate))):
        if i >= len(reference) or i >= len(candidate):
            rows.append(dict(index=i, passed=False, reason='capture count differs', max_abs=None))
            continue
        a, b = array(reference[i]), array(candidate[i])
        if a.shape != b.shape or a.dtype != b.dtype or not np.isfinite(a).all() or not np.isfinite(b).all():
            rows.append(dict(index=i, passed=False, reason='shape/dtype/nonfinite mismatch', max_abs=None))
            continue
        error = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.
        rows.append(dict(index=i, passed=error <= ATOL, max_abs=error, shape=list(a.shape), dtype=str(a.dtype),
                         max_reference_abs=float(np.max(np.abs(a))) if a.size else 0.,
                         reference_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                         candidate_sha256=hashlib.sha256(b.tobytes()).hexdigest()))
    return dict(passed=bool(rows) and all(r['passed'] for r in rows), atol=ATOL, rtol=0.,
                captures_reference=len(reference), captures_candidate=len(candidate),
                max_abs=max((r['max_abs'] or 0. for r in rows), default=0.),
                largest_differences=sorted(rows, key=lambda r: r['max_abs'] or 0., reverse=True)[:10],
                failures=[r for r in rows if not r['passed']][:20],
                failure_count=sum(not r['passed'] for r in rows))


def prediction_comparison(reference, candidate):
    import numpy as np
    a, b = reference.sort_values('station_id'), candidate.sort_values('station_id')
    require(a.station_id.tolist() == b.station_id.tolist(), 'Repeat control query identities differ')
    rows = {}
    for name in ('prediction', 'mdn_logits', 'mdn_mu', 'mdn_sigma'):
        aa = a[name].to_numpy() if name == 'prediction' else np.asarray([json.loads(x) for x in a[name]])
        bb = b[name].to_numpy() if name == 'prediction' else np.asarray([json.loads(x) for x in b[name]])
        require(aa.shape == bb.shape and np.isfinite(aa).all() and np.isfinite(bb).all(), 'Invalid repeat MDN: ' + name)
        rows[name] = float(np.max(np.abs(aa.astype(float) - bb.astype(float))))
    return dict(passed=all(v <= ATOL for v in rows.values()), atol=ATOL, rtol=0., fields_max_abs=rows)


class FeatureAuditRunner:
    def __init__(self, original, directory):
        self.original, self.directory, self.baseline, self.count = original, Path(directory), None, 0

    def __getattr__(self, name):
        return getattr(self.original, name)

    @contextlib.contextmanager
    def capture_features(self):
        with self.original.capture_features() as features:
            yield features
        index = self.count
        self.count += 1
        label = ('baseline', 'future_NaN', 'future_Inf', 'future_pulse')[index] if index < 4 else 'extra_capture'
        if self.baseline is None:
            self.baseline = features
        comparison = feature_comparison(self.baseline, features)
        # This is diagnostic only. The original callback still enforces its own gate.
        write(self.directory / (f'{index + 3:02d}_{label}.json'), comparison)
        print('TEAM_SMOKE_FEATURE', label, 'max_abs=' + str(comparison['max_abs']),
              'passed=' + str(comparison['passed']), flush=True)


class AuditForward:
    def __init__(self, original, directory):
        self.original = original
        self.runner = FeatureAuditRunner(original.runner, directory)

    def __getattr__(self, name):
        return getattr(self.original, name)


def backend_settings(torch):
    return dict(cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
                cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
                matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32))


def controlled_smoke(original, paired_execution, torch, root):
    """Wrap the unchanged original gate with same-input controls and diagnostics."""
    root = Path(root)
    def verify(forward, event, group, legacy, family, checks):
        require(family == 'TEAM', 'This operational audit is TEAM-only')
        request = group.iloc[0]
        identity = {k: str(request[k]) for k in ('dataset_id', 'event_id', 'geometry_protocol', 'elapsed_time')}
        directory = root / fingerprint(identity)[:24]
        directory.mkdir(parents=True, exist_ok=False)
        before = backend_settings(torch)
        write(directory / '00_scope.json', dict(request=identity, backend_before=before,
            absolute_tolerance=ATOL, relative_tolerance=0., original_future_gates_unchanged=True))
        try:
            with paired_execution(forward.device):
                write(directory / '01_backend.json', backend_settings(torch))
                with forward.runner.capture_features() as first:
                    first_output, _ = forward.predict(event, group)
                with forward.runner.capture_features() as second:
                    second_output, _ = forward.predict(event, group)
                features = feature_comparison(first, second)
                prediction = prediction_comparison(first_output, second_output)
                write(directory / '02_same_input_repeat.json', dict(features=features, prediction=prediction))
                print('TEAM_SMOKE_SAME_INPUT', identity, 'feature_max_abs=' + str(features['max_abs']),
                      'passed=' + str(features['passed'] and prediction['passed']), flush=True)
                require(features['passed'] and prediction['passed'],
                        'SAME_INPUT_REPEATABILITY_FAILED; see smoke_diagnostics/02_same_input_repeat.json')
                original(AuditForward(forward, directory), event, group, legacy, family, checks)
            require(backend_settings(torch) == before, 'Audit changed backend state outside its scope')
            write(directory / '99_pass.json', dict(status='PASS', backend_after=backend_settings(torch),
                backend_restored=backend_settings(torch) == before))
        except BaseException as exc:
            write(directory / '99_failure.json', dict(status='FAIL', type=type(exc).__name__, reason=str(exc),
                backend_after=backend_settings(torch), backend_restored=backend_settings(torch) == before))
            raise
        checks.append(dict(check='TEAM_controlled_same_input_repeat', status='PASS',
                           feature_max_abs=features['max_abs'], prediction=prediction,
                           scope='audit-only backend controls restored; original future/query/cache/label/legacy gates retained'))
    return verify
