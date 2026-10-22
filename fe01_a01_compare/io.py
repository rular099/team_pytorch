"""Strict artifacts, immutable attempts, and content identities (stdlib only)."""
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import tempfile


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: a successful or failed attempt is never rewritten.
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def csv_read(path, **kwargs):
    import pandas as pd
    return pd.read_csv(path, float_precision='round_trip',
                       dtype={'dataset_id': str, 'event_id': str, 'EVENT': str, 'station_id': str}, **kwargs)


def csv_write(frame, path):
    path = Path(path)
    require(not path.exists(), 'Preserve existing output: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression={'method': 'gzip', 'mtime': 0}
                 if str(path).endswith('.gz') else None)


def files_manifest(root):
    root = Path(root)
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p != root/'SEAL.json'}


def seal(root, **identity):
    write_json(Path(root) / 'SEAL.json', dict(status='COMPLETE', **identity,
               files_sha256=files_manifest(root)))


def verify_seal(root, expected=None):
    root = Path(root)
    s = read_json(root / 'SEAL.json')
    require(s['status'] == 'COMPLETE', 'Not a complete stage: ' + str(root))
    if expected:
        for k, v in expected.items():
            require(s.get(k) == v, 'Stale seal identity: ' + k)
    require(files_manifest(root) == s['files_sha256'], 'Sealed files changed: ' + str(root))
    return s


@contextlib.contextmanager
def attempt(session, stage, task, identity, blocking=False):
    import fcntl
    base = Path(session) / 'stages' / stage
    base.mkdir(parents=True, exist_ok=True)
    final = base / str(task)
    with (base / (str(task) + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB)
        if final.exists():
            verify_seal(final, identity)
            yield None
            return
        attempts = Path(session) / 'attempts' / stage
        attempts.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=str(task) + '_', dir=attempts))
        write_json(temp / 'ATTEMPT.json', dict(stage=stage, task=task, identity=identity,
                   job_id=os.environ.get('SLURM_JOB_ID'),
                   started_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        try:
            yield temp
            seal(temp, **identity)
            # Lock serializes publication; never replace a previous success.
            require(not final.exists(), 'Success already exists')
            temp.rename(final)
        except BaseException as exc:
            write_json(temp / 'FAILURE.json', dict(type=type(exc).__name__, reason=str(exc)))
            raise
