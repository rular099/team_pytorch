#!/usr/bin/env python
"""Explicitly hash/register a local DiTing checkpoint; never loads or trains it."""
import argparse
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fe01.config import sha256,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--source',required=True,help='Actual origin, training corpus/time span or explicitly unknown')
    p.add_argument('--copy',action='store_true',help='Copy large checkpoint into a portable offline directory')
    a=p.parse_args();path=Path(a.manifest);checkpoint=Path(a.checkpoint).resolve()
    manifest=json.loads(path.read_text())
    if 'diting' in manifest['models']: raise ValueError('DiTing already registered; preserve existing manifest')
    if a.copy:
        target=path.parent/'diting'/checkpoint.name
        if target.exists(): raise FileExistsError(target)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(checkpoint,target)
        checkpoint=target.resolve();record_path=str(checkpoint.relative_to(path.parent.resolve()))
    else: record_path=str(checkpoint)
    manifest['models']['diting']=dict(files=dict(weights=dict(path=record_path,sha256=sha256(checkpoint),bytes=checkpoint.stat().st_size)),
        architecture='MAE 1200M backbone_attn_pool; key/shape audit required',native_n_samples=10000,
        component_order='NEZ',sampling_rate=100,source=a.source,overlap_status='unknown; corpus audit required',
        pretraining_normalization='unknown; actual run metadata required',device_forward_status='NOT_RUN')
    manifest.pop('missing_diting',None)
    # Preserve the original manifest for review and refuse to overwrite backup.
    backup=path.with_suffix('.before_diting.json')
    if backup.exists(): raise FileExistsError(backup)
    shutil.copyfile(path,backup);write_json(path,manifest)
    print('Registered exact checkpoint; real weight forward audit remains mandatory')

if __name__=='__main__': main()
