"""Explicit commands only. Import/prepare/status/test never submits a scheduler job."""
import argparse
import json
from pathlib import Path
from .io import require


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    init=sub.add_parser('init-env')
    init.add_argument('--env',required=True)
    init.add_argument('--inherit-session',required=True)
    for name in ('prepare','status','submit','resume'):
        s=sub.add_parser(name)
        s.add_argument('--run-id',required=True)
        s.add_argument('--env',required=True)
        if name in ('submit','resume'):
            s.add_argument('--stage',required=True)
            s.add_argument('--geometry',choices=['normal','random'],default='normal')
    work=sub.add_parser('worker')
    work.add_argument('--session',required=True)
    work.add_argument('--stage',required=True)
    work.add_argument('--index',type=int,required=True)
    work.add_argument('--device',default='cuda')
    release=sub.add_parser('build-bundle')
    release.add_argument('--output',required=True)
    compare=sub.add_parser('compare-existing-local')
    compare.add_argument('--source',required=True)
    compare.add_argument('--output',required=True)
    compare.add_argument('--draws',type=int,default=2000)
    interaction=sub.add_parser('interaction-existing-local')
    interaction.add_argument('--source',required=True)
    interaction.add_argument('--output',required=True)
    interaction.add_argument('--draws',type=int,default=2000)
    demo=sub.add_parser('synthetic-e2e')
    demo.add_argument('--output',required=True)
    a=p.parse_args()
    from .orchestration import initialize_env,prepare,status,session_path,submit
    if a.command=='init-env':
        initialize_env(a.env,a.inherit_session)
    elif a.command=='prepare':
        prepare(a.run_id,a.env)
    elif a.command=='status':
        status(session_path(a.run_id))
    elif a.command in ('submit','resume'):
        submit(session_path(a.run_id),a.stage,a.geometry,resume=a.command=='resume')
    elif a.command=='worker':
        from .stages import execute
        execute(a.session,a.stage,a.index,a.device)
    elif a.command=='build-bundle':
        from .packaging import source_bundle
        source_bundle(a.output)
    elif a.command=='compare-existing-local':
        from .comparison import analyze_existing
        from .contracts import source_identity,verify_frozen_core
        from .io import write_json,seal
        out=Path(a.output)
        out.mkdir(parents=True,exist_ok=False)
        write_json(out/'source_identity.json',source_identity())
        write_json(out/'core_verification.json',verify_frozen_core())
        analyze_existing(a.source,out,a.draws)
        seal(out,scope='REAL_JAPAN_SAVED_PREDICTIONS; local CPU only')
    elif a.command=='synthetic-e2e':
        from .synthetic import run
        run(a.output)
    elif a.command=='interaction-existing-local':
        from .interactions import analyze
        from .contracts import source_identity
        from .io import write_json,seal
        analyze(a.source,Path(a.output),draws=a.draws)
        write_json(Path(a.output)/'source_identity.json',source_identity())
        seal(a.output,scope='REAL_JAPAN_EXISTING_PREDICTIONS; joint event interaction contrasts')


if __name__=='__main__':
    main()
