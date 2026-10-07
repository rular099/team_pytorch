from pathlib import Path
import pytest
from fe01.config import sha256
from fe01_a01.assets import ensure_diting_manifest
from fe01_a01.provenance import read_json
from test_fe01_a01_helpers import config


def test_new_manifest_never_changes_original_weights(tmp_path,monkeypatch):
    cfg=config(tmp_path,'diting_pretrained_frozen')
    cfg['pretrained_manifest']=str(tmp_path/'weights/diting_manifest.json')
    original=tmp_path/'original_manifest.json';original.write_text('{"models": {}}\n')
    original_sha=sha256(original)
    # Synthetic bytes only exercise registration; no synthetic production encoder.
    encoder=tmp_path/'synthetic_encoder';encoder.write_bytes(b'synthetic registration fixture')
    monkeypatch.setenv('A01_DITING_CHECKPOINT',str(encoder))
    monkeypatch.setenv('A01_DITING_ENCODER_SHA256','0'*64)
    with pytest.raises(ValueError,match='SHA'):ensure_diting_manifest(cfg)
    monkeypatch.setenv('A01_DITING_ENCODER_SHA256',sha256(encoder))
    ensure_diting_manifest(cfg);ensure_diting_manifest(cfg)
    manifest=read_json(cfg['pretrained_manifest'])
    assert manifest['models']['diting']['files']['weights']['sha256']==sha256(encoder)
    assert sha256(original)==original_sha


def test_missing_actual_diting_is_explicit_NOT_RUN(tmp_path,monkeypatch):
    cfg=config(tmp_path,'diting_pretrained_frozen');cfg['pretrained_manifest']=str(tmp_path/'weights/missing.json')
    monkeypatch.delenv('A01_DITING_CHECKPOINT',raising=False);monkeypatch.delenv('A01_DITING_ENCODER_SHA256',raising=False)
    with pytest.raises(ValueError,match='NOT_RUN'):ensure_diting_manifest(cfg)
