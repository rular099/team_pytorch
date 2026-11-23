#!/usr/bin/env python
"""Compute-node entry. Offline analysis does not construct the encoder."""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fe02_lite.config import resolve, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['audit','train','eval','analyze','compare','diagnose'])
    p.add_argument('--config')
    p.add_argument('--dataset-view')
    p.add_argument('--variant')
    p.add_argument('--seed',type=int)
    p.add_argument('--stage')
    p.add_argument('--output')
    p.add_argument('--audit-dir')
    p.add_argument('--checkpoint',choices=['best','last'],default='best')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--device',default='cuda')
    p.add_argument('--evaluation',nargs='+')
    p.add_argument('--run-dir')
    p.add_argument('--diagnostics')
    p.add_argument('--bootstrap-draws',type=int,default=1000)
    a = p.parse_args()
    if a.action in ('analyze','compare','diagnose'):
        from fe02_lite import analysis
        if a.action=='compare': analysis.compare(a.evaluation,a.output,a.bootstrap_draws)
        elif a.action=='diagnose': analysis.diagnose(a.diagnostics,a.output)
        else: analysis.summarize_run(a.evaluation[0],a.output,a.run_dir,a.bootstrap_draws)
        return
    cfg = resolve(a.config,a.dataset_view,a.variant,a.seed,a.stage)
    audit_dir = Path(a.audit_dir or Path(cfg['output_root'])/'audits'/cfg['run_id'])
    run_dir = Path(cfg['output_root'])/cfg['run_id']
    from fe02_lite import engine
    if a.action=='audit':
        # Preserve failed attempts. Publish a reusable lock only on successful audit.
        if (audit_dir/'protocol.lock.json').exists():
            engine.check_lock(cfg,audit_dir)
            print('Existing successful audit verified')
            return
        if audit_dir.exists():
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
            audit_dir.rename(audit_dir.with_name(audit_dir.name+'.failed_'+stamp))
        engine.audit(cfg,audit_dir,a.device)
    elif a.action=='train':
        engine.train(cfg,audit_dir,a.resume)
    else:
        from fe02_lite.evaluation import evaluate
        from fe02_lite.analysis import summarize_run
        checkpoint_path = run_dir/(a.checkpoint+'.pth')
        model,checkpoint,effective = engine.load_for_evaluation(cfg,audit_dir,checkpoint_path,a.device)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        output = Path(a.output or run_dir/'evaluations'/(a.checkpoint+'_'+stamp))
        evaluate(model,effective,output,dict(**checkpoint['artifact_identity'],checkpoint_sha256=sha256(checkpoint_path),
            checkpoint_epoch=checkpoint['epoch'],checkpoint_step=checkpoint['step'],checkpoint_tag=a.checkpoint),a.device)
        summarize_run(output,output/'analysis',run_dir,a.bootstrap_draws)
        print(json.dumps(dict(evaluation=str(output),checkpoint=a.checkpoint)))


if __name__=='__main__': main()
