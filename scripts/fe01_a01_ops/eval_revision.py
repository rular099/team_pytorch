#!/usr/bin/env python3
"""Append-only execution revision for a live, immutable eval session (stdlib)."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

FILES=('eval_pack.py','eval_pack.sh','eval_revision.py')
PURPOSE='TEAM scratch trained encoder identity correction v1'
DITING_PURPOSE='DiTing OFF serialized ON request identity correction v1'
LEGACY_OPERATIONS={
    'eval_pack.py':'25abba3dfd7960d2d83016d72999f92235934e12e2a5c3e0a20a929be06b732d',
    'eval_pack.sh':'9418235a2e0a56594bfd82b461e7e6495650c1f3a3f694352e3b688d3d97d0cd',
}


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def parent_plan(session):
    session=Path(session).resolve();plan=read(session/'plan.json')
    root=Path(os.environ['A01_CODE_ROOT']).resolve()
    sys.path[:0]=[str(root/'vendor'),str(root)]
    spec=importlib.util.spec_from_file_location('a01_revision_preflight',root/'scripts/fe01_a01/preflight.py')
    pf=importlib.util.module_from_spec(spec);spec.loader.exec_module(pf)
    require(str(root)==plan['code_root'] and pf.current_code_sha()==plan['source_core_sha256'],'Parent core source changed')
    require(plan['operations_sha256']==LEGACY_OPERATIONS,'This revision requires the original 20261014 eval/pack session')
    for name,digest in LEGACY_OPERATIONS.items():
        require(sha256(session/'tools'/name)==digest,'Parent frozen tool changed: '+name)
    require(plan['runs'][0]['config']['model_family']=='team_original_scratch','Index0 is not the registered TEAM scratch run')
    return session,plan


def policy(purpose):
    if purpose==PURPOSE:return dict(files=FILES,indices=[0],prefix='team_eval_fix_')
    if purpose==DITING_PURPOSE:
        return dict(files=FILES+('serialized_on.py',),indices=[4],prefix='diting_eval_fix_')
    raise ValueError('Unsupported execution revision policy')


def validate_revision(session,path,expected_sha256=None,index=None):
    session,plan=parent_plan(session);path=Path(path).resolve()
    require((session/'operations') in path.parents and path.name=='revision.json','Revision path escapes session operations')
    if expected_sha256 is not None:require(sha256(path)==expected_sha256,'Revision metadata SHA differs')
    revision=read(path)
    selected=policy(revision['purpose'])
    require(revision['schema']==1 and revision['allowed_eval_indices']==selected['indices'],
            'Unsupported execution revision policy')
    require(revision['parent_plan_sha256']==sha256(session/'plan.json') and
            revision['parent_operations_sha256']==plan['operations_sha256'] and
            revision['source_core_sha256']==plan['source_core_sha256'],'Revision parent identity differs')
    require(set(revision['operations_sha256'])==set(selected['files']),'Revision source manifest incomplete')
    if index is not None:
        require(index in revision['allowed_eval_indices'],
                'This correction may evaluate only index'+str(selected['indices'][0]))
    for name,digest in revision['operations_sha256'].items():
        require(sha256(path.parent/name)==digest,'Revision source changed: '+name)
    return revision


def register(session,tool_dir,purpose=PURPOSE):
    session,plan=parent_plan(session);tool_dir=Path(tool_dir)
    selected=policy(purpose)
    operations={name:sha256(tool_dir/name) for name in selected['files']}
    content=hashlib.sha256(json.dumps(operations,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    base=session/'operations';base.mkdir(exist_ok=True)
    target=base/(selected['prefix']+content[:12])
    if target.exists():
        revision=validate_revision(session,target/'revision.json')
        require(revision['operations_sha256']==operations,'Existing revision content differs')
        return target
    temporary=Path(tempfile.mkdtemp(prefix='.'+selected['prefix'],dir=base))
    for name in selected['files']:
        shutil.copy2(tool_dir/name,temporary/name)
        (temporary/name).chmod(0o500 if name.endswith('.sh') else 0o400)
    changes=(['TEAM: verify trained encoder bytes against its checkpoint SHA',
              'Every family: preserve common initialization identity',
              'Frozen families: preserve initial encoder identity'] if purpose==PURPOSE else
             ['DiTing OFF: read sealed ON predictions with round_trip floating parser',
              'Before each OFF export: authenticate ON bytes and full unchanged production population gate',
              'No label replacement, intersection, tolerance widening or source seal rewrite'])
    revision=dict(schema=1,purpose=purpose,allowed_eval_indices=selected['indices'],
                  parent_plan_sha256=sha256(session/'plan.json'),parent_operations_sha256=plan['operations_sha256'],
                  source_core_sha256=plan['source_core_sha256'],operations_sha256=operations,
                  changes=changes+['Preserve live parent plan/tools and accept sealed mixed execution identities for pack'])
    path=temporary/'revision.json';path.write_text(json.dumps(revision,indent=2)+'\n');path.chmod(0o400)
    os.rename(temporary,target)
    validate_revision(session,target/'revision.json')
    return target


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['register'])
    parser.add_argument('--session',required=True);parser.add_argument('--tool-dir',required=True)
    parser.add_argument('--policy',choices=['team','diting_off'],default='team')
    args=parser.parse_args()
    print(register(args.session,args.tool_dir,DITING_PURPOSE if args.policy=='diting_off' else PURPOSE))


if __name__=='__main__':main()
