#!/usr/bin/env python
"""Manual FE02 training/evaluation. Held-out test is intentionally unavailable."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fe01.config import load
from fe02.config import validate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['audit','train','eval'])
    p.add_argument('--config',required=True);p.add_argument('--output');p.add_argument('--audit-dir')
    p.add_argument('--reuse-data-audit');p.add_argument('--device',default='cuda')
    p.add_argument('--checkpoint');p.add_argument('--resume',action='store_true')
    p.add_argument('--random-times-manifest');a=p.parse_args()
    cfg=load(a.config);validate(cfg)
    # Formal numerical work is imported only after torch-free argument checks.
    from fe01 import engine as core
    from fe02 import engine as experiment
    if a.action=='audit':
        if not a.output: p.error('--output required')
        core.audit(cfg,a.output,a.device,a.reuse_data_audit,experiment=experiment)
    elif a.action=='train':
        if not a.audit_dir: p.error('--audit-dir required')
        core.train(cfg,a.audit_dir,a.resume,experiment=experiment)
    else:
        if not a.output or not a.checkpoint: p.error('--output and --checkpoint required')
        core.export_evaluation(cfg,a.checkpoint,a.output,a.device,
            random_manifest=a.random_times_manifest,experiment=experiment)

if __name__=='__main__': main()
