"""Explicit experiment hooks; FE01 default engine remains unchanged when omitted."""
from pathlib import Path
import json

from fe01 import engine as core
from fe01.config import fingerprint, sha256
from . import BASE_COMMIT
from .config import validate
from .model import build_model, extra_audit, model_audit


def code_identity():
    root = Path(__file__).resolve().parents[1]
    paths = list((root/'fe02').glob('*.py')) + list((root/'scripts/fe02').glob('*'))
    paths += list((root/'configs/fe02').rglob('*.json'))
    paths += [root/'configs/fe01/formal/formal__diting_pretrained_frozen__seed42.json']
    return fingerprint(dict(backend=core.code_identity(),
        files={str(p.relative_to(root)): sha256(p) for p in paths if p.is_file()}))


def check_evaluation_contract(cfg, checkpoint_path):
    validate(cfg)
    lock=json.loads((Path(checkpoint_path).parent/'protocol.lock.json').read_text())
    if lock['status'] != 'AUDIT_PASS' or lock['variant_id'] != cfg['variant_id']:
        raise ValueError('FE02 evaluation/audit variant mismatch')
    if sha256(cfg['data']['split_manifest']) != lock['data_identity']['split_manifest_sha256']:
        raise ValueError('FE02 evaluation split changed since training')
    current=core.data_identity(cfg,with_hash=False)
    if set(current['shards']) != set(lock['data_identity']['shards']):
        raise ValueError('FE02 evaluation shard set changed')
    for path, record in lock['data_identity']['shards'].items():
        stat=Path(path).stat()
        if stat.st_size != record['bytes'] or stat.st_mtime_ns != record['mtime_ns']:
            raise ValueError('FE02 evaluation data changed since training')
    if sha256(cfg['pretrained_manifest']) != lock['pretrained_manifest_sha256']:
        raise ValueError('FE02 evaluation weight manifest changed since training')
