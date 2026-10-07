"""Register an authenticated existing DiTing encoder in an A01-only manifest."""
import fcntl
import os
from pathlib import Path

from fe01.config import sha256
from .provenance import read_json,write_json,require


def ensure_diting_manifest(cfg):
    if cfg['model_family']!='diting_pretrained_frozen':return
    path=Path(cfg['pretrained_manifest'])
    if path.exists():return
    checkpoint=Path(os.environ.get('A01_DITING_CHECKPOINT',''))
    expected=os.environ.get('A01_DITING_ENCODER_SHA256','')
    require(checkpoint.is_file() and len(expected)==64,
            'DiTing NOT_RUN: supply existing encoder and its frozen EVAL1 SHA in private env; no random replacement')
    require(sha256(checkpoint)==expected,'Registered DiTing bytes differ from frozen encoder SHA')
    root=Path(cfg['output_root']).resolve()
    require(root in path.resolve().parents,'New manifest must live in A01 outputs, not original weights')
    path.parent.mkdir(parents=True,exist_ok=True)
    with (path.parent/'.registration.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        if path.exists():return
        write_json(path,dict(models=dict(diting=dict(files=dict(weights=dict(path=str(checkpoint.resolve()),
            sha256=expected,bytes=checkpoint.stat().st_size)),architecture='registered original MAE 1200M encoder',
            native_n_samples=10000,component_order='NEZ',sampling_rate=100,
            source='existing encoder authenticated by frozen FE01-EVAL1 inventory; not RT55 downstream state',
            overlap_status='unknown; no new pretraining claim',device_forward_status='A01 audit required'))))
