#!/usr/bin/env python
"""Register only a real DiTing checkpoint into a NEW FE02 manifest; torch-free."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fe01.config import sha256, write_json


def register(checkpoint, output, source):
    checkpoint=Path(checkpoint).resolve();output=Path(output)
    if output.exists(): raise FileExistsError('Preserve existing weight manifest: '+str(output))
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    write_json(output, dict(models=dict(diting=dict(
        files=dict(weights=dict(path=str(checkpoint),sha256=sha256(checkpoint),bytes=checkpoint.stat().st_size)),
        source=source,architecture='MAE 1200M backbone_attn_pool; strict-load/device audit required',
        component_order='NEZ',sampling_rate=100,native_n_samples=10000,
        pretraining_normalization='unknown',overlap_status='unknown; corpus-level audit required',
        device_forward_status='NOT_RUN'))))
    print('Registered file SHA only. Strict loading, frozen-state and real forward remain NOT_RUN.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--source',required=True);a=p.parse_args()
    register(a.checkpoint,a.output,a.source)
