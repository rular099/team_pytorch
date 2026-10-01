#!/usr/bin/env python
"""Manual FE01 audit/train/eval/cases/replay. This CLI never submits Slurm jobs."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fe01.config import load


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['audit','train','eval','cases','replay'])
    parser.add_argument('--config',required=True)
    parser.add_argument('--output')
    parser.add_argument('--device',default='cuda' if os.environ.get('FE01_DEVICE')=='cuda' else 'cpu')
    parser.add_argument('--audit-dir')
    parser.add_argument('--reuse-data-audit')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--checkpoint')
    parser.add_argument('--split',choices=['val','test'],default='val')
    parser.add_argument('--allow-test',action='store_true')
    parser.add_argument('--protocol-lock-sha256')
    parser.add_argument('--test-exposure-ledger')
    parser.add_argument('--random-times-manifest')
    parser.add_argument('--cases-manifest')
    parser.add_argument('--reference',choices=['first_p_pick','origin_time'],default='first_p_pick')
    parser.add_argument('--rolling',action='store_true')
    parser.add_argument('--start',type=int,default=1)
    parser.add_argument('--end',type=int,default=90)
    parser.add_argument('--spacing-km',type=float,default=20)
    parser.add_argument('--no-grid',action='store_true')
    args=parser.parse_args()
    cfg=load(args.config)
    if args.action=='audit':
        if not args.output and not args.audit_dir: parser.error('--output or --audit-dir required')
        from fe01.engine import audit
        audit(cfg,args.output or args.audit_dir,args.device,args.reuse_data_audit)
    elif args.action=='train':
        if not args.audit_dir:
            parser.error('--audit-dir required; manual data/weight audit must pass first')
        from fe01.engine import train
        train(cfg,args.audit_dir,args.resume)
    elif args.action=='eval':
        if not args.checkpoint or not args.output:
            parser.error('--checkpoint and --output required')
        from fe01.engine import export_evaluation
        export_evaluation(cfg,args.checkpoint,args.output,args.device,args.split,args.allow_test,
                          args.protocol_lock_sha256,args.random_times_manifest,args.test_exposure_ledger)
    elif args.action=='cases':
        if not args.output: parser.error('--output required')
        from fe01.replay import select_cases
        select_cases(cfg,args.output)
    else:
        if not 1<=args.start<=args.end<=90: parser.error('Replay range must be within 1..90')
        if not args.checkpoint or not args.output or not args.cases_manifest:
            parser.error('--checkpoint, --output and --cases-manifest required')
        from fe01.replay import replay
        replay(cfg,args.checkpoint,args.cases_manifest,args.output,args.device,args.reference,args.rolling,
               args.start,args.end,args.spacing_km,not args.no_grid)

if __name__=='__main__':
    main()
