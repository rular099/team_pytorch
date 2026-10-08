#!/usr/bin/env python
"""Render only the five DiTing architecture groups. Torch-free; no submissions."""
import argparse
import copy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fe01.config import fingerprint, write_json
from fe02.config import ARCH_KEYS, VARIANTS, baseline, expand_architecture, validate


def make_config(variant, seed=42, stage='formal'):
    cfg = baseline()
    cfg.update(fe02=dict(enabled=True, version=1), seed=seed, stage=stage, variant_id=variant,
        run_id='fe02__'+stage+'__'+variant+'__seed'+str(seed),
        pretrained_manifest='${FE02_WEIGHTS_ROOT}/pretrained_manifest.json',
        diting_config='${FE02_CODE_ROOT}/diting/config/diting_1200m_backbone_attnpool.yml',
        output_root='${FE02_OUTPUT_ROOT}')
    cfg['data'].update(root='${FE02_DATA_ROOT}', split_manifest='${FE02_SPLIT_MANIFEST}')
    cfg['spatial']['manifest']='${FE02_OUTPUT_ROOT}/unused_spatial_manifest.csv'
    for key in ARCH_KEYS:
        cfg['model_params'].pop(key, None)
    if stage == 'pilot':
        cfg['training'].update(epochs=1, max_updates=4, global_batch=16, microbatch=1)
        cfg['audit_limits']=dict(train_events=16, val_events=8)
    cfg = expand_architecture(cfg)
    validate(cfg)
    return cfg


def render(destination):
    destination = Path(destination)
    configs = []
    for stage in ('formal', 'pilot'):
        for seed in ([42,43,44] if stage == 'formal' else [42]):
            for variant in VARIANTS:
                cfg = make_config(variant, seed, stage)
                relative = stage+'/'+cfg['run_id']+'.json'
                path = destination/relative
                if path.exists():
                    import json
                    if json.loads(path.read_text()) != cfg:
                        raise FileExistsError('Refusing to overwrite changed config: '+str(path))
                else:
                    write_json(path, cfg)
                configs.append(dict(stage=stage,seed=seed,variant_id=variant,variant_name=cfg['variant_name'],
                    model_family=cfg['model_family'],run_id=cfg['run_id'],config='configs/fe02/'+relative,
                    protocol_sha256=fingerprint(cfg)))
    import csv
    destination.mkdir(parents=True, exist_ok=True)
    with (destination/'runs.tsv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(configs[0]),delimiter='\t',lineterminator='\n')
        writer.writeheader();writer.writerows(configs)
    print('FE02: 20 configs prepared (15 formal + 5 optional pilot); default submission is ONLY seed42 (5 formal runs). No jobs submitted.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='configs/fe02')
    render(p.parse_args().output)
