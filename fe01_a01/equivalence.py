"""Numerical evidence for paired ON executions; never changes training defaults."""
import contextlib
import hashlib

import torch


ATOL = 1e-6
RTOL = 1e-5


def compare_tensors(reference, candidate, atol=ATOL, rtol=RTOL):
    """Keep names, shapes, dtypes and gradient presence strict; compare FP values."""
    rows = []
    for name in sorted(set(reference) | set(candidate)):
        row = dict(name=name, passed=False, max_abs=None)
        if name not in reference or name not in candidate:
            row['reason'] = 'missing tensor'
        else:
            a, b = reference[name], candidate[name]
            if a is None or b is None:
                row.update(passed=a is None and b is None, reason='gradient presence', max_abs=0.)
            elif a.shape != b.shape or a.dtype != b.dtype:
                row.update(reason='shape/dtype differs', reference_shape=list(a.shape),
                           candidate_shape=list(b.shape), reference_dtype=str(a.dtype), candidate_dtype=str(b.dtype))
            elif a.numel() == 0:
                row.update(passed=True, max_abs=0., reason='empty')
            elif a.is_floating_point():
                a, b = a.detach().to('cpu', torch.float64), b.detach().to('cpu', torch.float64)
                finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                if finite:
                    error = (a-b).abs()
                    allowed = atol + rtol*a.abs()
                    row.update(passed=bool((error <= allowed).all()), max_abs=float(error.max()),
                               max_tolerance_ratio=float((error/allowed).max()),
                               mismatched_elements=int((error > allowed).sum()), reason='finite FP comparison')
                else:
                    row['reason'] = 'nonfinite tensor'
            else:
                row.update(passed=torch.equal(a.detach().cpu(), b.detach().cpu()),
                           max_abs=0. if torch.equal(a.detach().cpu(), b.detach().cpu()) else None,
                           reason='nonfloating values must be exact')
        rows.append(row)
    failed = [row['name'] for row in rows if not row['passed']]
    largest = sorted(rows, key=lambda row: row.get('max_abs') or 0., reverse=True)[:10]
    return dict(passed=not failed, atol=atol, rtol=rtol, tensors=len(rows),
                failed_tensors=failed, max_abs=max((row.get('max_abs') or 0. for row in rows), default=0.),
                largest_differences=largest,
                failures=[row for row in rows if not row['passed']][:20])


def outputs_comparison(reference, candidate):
    return compare_tensors({str(i): x for i, x in enumerate(reference)},
                           {str(i): x for i, x in enumerate(candidate)})


def capture_rng(devices):
    return dict(cpu=torch.get_rng_state().clone(),
                cuda={i: torch.cuda.get_rng_state(i).clone() for i in devices})


def restore_rng(state):
    torch.set_rng_state(state['cpu'])
    for device, value in state['cuda'].items():
        torch.cuda.set_rng_state(value, device)


def rng_identity(state):
    digest = lambda value: hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()
    return dict(cpu=digest(state['cpu']), cuda={str(i): digest(value) for i, value in state['cuda'].items()})


@contextlib.contextmanager
def paired_execution(device):
    """Scope RNG/backend controls to the audit, restoring them even on failure."""
    device = torch.device(device)
    # Both factories call manual_seed(), which also seeds every visible GPU.
    devices = list(range(torch.cuda.device_count())) if device.type == 'cuda' or torch.cuda.is_initialized() else []
    settings = [(torch.backends.cudnn, 'benchmark', False),
                (torch.backends.cudnn, 'deterministic', True),
                (torch.backends.cudnn, 'allow_tf32', False),
                (torch.backends.cuda.matmul, 'allow_tf32', False)]
    previous = [(owner, name, getattr(owner, name)) for owner, name, _ in settings]
    with torch.random.fork_rng(devices=devices):
        try:
            for owner, name, value in settings:
                setattr(owner, name, value)
            yield devices
        finally:
            for owner, name, value in previous:
                setattr(owner, name, value)
