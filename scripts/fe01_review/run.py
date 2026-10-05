#!/usr/bin/env python3
"""Read-only FE01-EVAL1 CLI. Training, resume and test have no entry points."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fe01_review.provenance import (read_json,write_json,new_output,manifest_entry,training_identity,
    provenance,seal,require)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['inventory','plan','verify','export-existing','evaluate','reference','analyze','replay','pack'])
    for name in ('source-root','output','manifest','run-id','plan','request-root','verification','reference','input'):
        parser.add_argument('--'+name)
    parser.add_argument('--hydrate',action='store_true')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--scope',choices=['fixed','random','long'],default='fixed')
    args=parser.parse_args(argv)
    require(args.output,'--output required')
    output=new_output(args.output)
    try:
        training_identity()
        if args.action=='inventory':
            from fe01_review.checkpoint_inventory import inventory
            require(args.manifest,'--manifest required');inventory(args.manifest,output)
            write_json(output/'provenance.json',provenance(action=args.action))
        elif args.action=='plan':
            from fe01_review.requests import freeze_plan,lock_new_requests
            require(args.source_root,'--source-root required')
            if args.hydrate:
                from fe01_review.runners import relocate,data_gate
                cfg=relocate(read_json(Path(args.source_root)/'formal__team_original_scratch__seed42/resolved_config.json'))
                data_gate(cfg,read_json(Path(args.source_root)/'formal__team_original_scratch__seed42/protocol.lock.json'))
                require(args.plan,'--plan frozen local metadata required');lock_new_requests(cfg,args.plan,output)
            else: freeze_plan(args.source_root,output)
            write_json(output/'provenance.json',provenance(action=args.action,hydrated=args.hydrate))
        elif args.action in ('verify','evaluate','replay'):
            entry=manifest_entry(args.manifest,args.run_id)
            require(args.plan and args.source_root,'--plan and --source-root required')
            if args.action=='verify':
                from fe01_review.runners import verify
                verify(entry,args.plan,args.source_root,output,args.device,args.request_root)
            elif args.action=='evaluate':
                from fe01_review.runners import evaluate
                require(args.verification,'--verification required')
                evaluate(entry,args.plan,args.request_root,args.source_root,output,args.verification,args.scope,args.device)
            else:
                from fe01_review.replay import replay
                replay(entry,args.plan,args.request_root,args.source_root,output,args.verification,args.device)
        elif args.action=='reference':
            from fe01_review.reference import build_reference
            require(args.source_root,'--source-root required');build_reference(args.source_root,output,args.manifest)
        elif args.action in ('analyze','export-existing'):
            from fe01_review.pipeline import analyze
            require(args.source_root,'--source-root required')
            analyze(args.source_root,output,args.reference,args.verification)
        elif args.action=='pack':
            from fe01_review.pack import pack
            require(args.input,'--input required');pack(args.input,output)
        seal(output)
    except Exception as exc:
        write_json(output/'failure.json',dict(action=args.action,run_id=args.run_id,status='BLOCKED',reason=type(exc).__name__+': '+str(exc)))
        if args.action=='verify':write_json(output/'verification.json',dict(status='BLOCKED',reason=str(exc),run_id=args.run_id))
        seal(output)
        raise


if __name__=='__main__':main()
