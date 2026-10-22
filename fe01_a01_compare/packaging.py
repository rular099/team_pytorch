"""Read-only numerical/visual index and a light review archive with independent seals."""
import json
import os
import re
import io
from pathlib import Path
import shutil
import subprocess
import tarfile
from .contracts import ROOT,source_identity,contract
from .io import files_manifest,fingerprint,read_json,require,sha256,verify_seal,write_json


def redact(value):
    if isinstance(value,dict):
        return {k:redact(v) for k,v in value.items()}
    if isinstance(value,list):
        return [redact(v) for v in value]
    if isinstance(value,str) and value.startswith('/'):
        return 'PRIVATE_PATH/'+Path(value).name
    if isinstance(value,str):
        return re.sub(r'/(?:public|home|tmp|opt|mnt|run|var)/[^\s"\']+',lambda m:'PRIVATE_PATH/'+Path(m.group(0)).name,value)
    return value


def review_bundle(session,out):
    from .orchestration import verify_plan,status
    from .stages import all_stage_directories
    from .plotting import build_index
    plan=verify_plan(session)
    review=out/'review'
    review.mkdir()
    index=[]
    for p in sorted(Path(session).rglob('SEAL.json')):
        directory=p.parent
        if 'stages' not in directory.relative_to(session).parts:
            continue
        s=verify_seal(directory)
        for relative,digest in s['files_sha256'].items():
            source=directory/relative
            index.append(dict(path=str(source.relative_to(session)),sha256=digest,bytes=source.stat().st_size,
                              scope='full numeric/visuals retained on HPC',source_seal_sha256=sha256(p)))
    write_json(review/'full_artifact_index.json',index)
    for key in ('analysis_contract.json','source_identity.json','plan.json','local_core_verification.json'):
        write_json(review/key,redact(read_json(Path(session)/key)))
    write_json(review/'status_summary.json',status(session,False))
    write_json(review/'failure_attempts.json',redact([dict(path=str(p.relative_to(session)),failure=read_json(p))
        for p in sorted((Path(session)/'attempts').rglob('FAILURE.json'))]))
    # Exact original seals are preserved. Redacted copies have a separate review manifest.
    partial=os.environ.get('ON_EVAL_PACK_PARTIAL')=='1'
    for stage in ('compare_existing','inventory','analyze','latency'):
        if not (Path(session)/'stages'/stage).exists():
            continue
        directories=[p for p in (Path(session)/'stages'/stage).iterdir() if p.is_dir() and (p/'SEAL.json').is_file()] if partial else all_stage_directories(session,stage)
        for directory in directories:
            verify_seal(directory)
            target=review/stage/directory.name
            target.mkdir(parents=True)
            for source in directory.rglob('*'):
                if not source.is_file() or source.name in ('shadow_station_outcomes.csv.gz','map_readiness_by_frame.csv.gz','event_effects.csv.gz','spatial_fields.csv.gz'):
                    continue
                dest=target/source.relative_to(directory)
                dest.parent.mkdir(parents=True,exist_ok=True)
                if source.suffix=='.json':
                    write_json(dest,redact(read_json(source)))
                elif source.stat().st_size<100*1024*1024:
                    shutil.copy2(source,dest)
    if not partial:
        from .delivery_analysis import analyze_rendered
        derived=out/'render_delivery_full'
        analyze_rendered(session,derived)
        target=review/'render_delivery'
        target.mkdir()
        for source in derived.iterdir():
            if source.name!='shadow_station_outcomes_rendered.csv.gz':
                shutil.copy2(source,target/source.name)
        write_json(target/'full_rendered_outcomes_index.json',dict(path='render_delivery_full/shadow_station_outcomes_rendered.csv.gz',
            sha256=sha256(derived/'shadow_station_outcomes_rendered.csv.gz'),scope='retained beside review tar in pack output'))
    # 12 metadata/hash selected cases; complete non-selected source remains indexed.
    visual=[]
    directories=[p for p in (Path(session)/'stages/render_normal').glob('*') if p.is_dir() and (p/'SEAL.json').is_file()] if partial else all_stage_directories(session,'render_normal')
    for directory in directories:
        for item in read_json(directory/'visuals_index.json'):
            item=dict(item,source_directory=directory,case_hash=fingerprint(['V4_REPORT_CASE',item['dataset_id'],item['event_id']]))
            visual.append(item)
    if partial:
        selected=sorted(visual,key=lambda x:x['case_hash'])[:12]
    else:
        from .selection import freeze
        selected_events={tuple(e) for e in freeze(session)['report_events']}
        selected=[x for x in visual if (x['dataset_id'],x['event_id']) in selected_events]
    final=Path(session)/'stages/pack/0'
    build_index([dict(href=os.path.relpath(x['source_directory']/x['href'],final),label=x['label']) for x in visual],
                out/'full_visuals_index.html',title='Full normal event gallery: '+str(len(visual))+' events; '+('PARTIAL' if partial else 'all locked events'))
    write_json(review/'full_visuals_index.json',[dict(dataset_id=x['dataset_id'],event_id=x['event_id'],
        geometry=x['geometry'],path=str((x['source_directory']/x['href']).relative_to(session))) for x in visual])
    for item in selected:
        source=item['source_directory']/Path(item['href']).parent
        shutil.copytree(source,review/'cases'/source.name)
    build_index([dict(href='cases/'+Path(x['href']).parent.name+'/index.html',label=x['label']) for x in selected],review/'map_index.html',
                title='12 metadata-selected review cases; all events listed in full_visuals_index.json')
    write_json(review/'selected_report_cases.json',redact([{k:v for k,v in x.items() if k!='source_directory'} for x in selected]))
    (review/'limitations.md').write_text('validation/seed42；所有新端点是探索性研究敏感性，不是工程认证。\n'
        '全量图源和数值原地保留，完整SHA索引见full_artifact_index.json。代表案例按元数据/hash选取，非误差筛选。\n'
        '原私有seal与脱敏review字节不同；本包另存新manifest。站点支撑不认证连续区域真值，面积NA。\n'
        'UPSTREAM_CAUSALITY_UNKNOWN；通信和连续检测尚未知。\n')
    write_json(review/'REVIEW_MANIFEST.json',dict(execution=source_identity(),files_sha256=files_manifest(review),
        completion_scope='PARTIAL_WITH_FAILURES' if partial else 'REQUIRED_SCOPE_COMPLETE',
        partial_reason=os.environ.get('ON_EVAL_PACK_PARTIAL_REASON') if partial else None,
        original_seal_semantics='full_artifact_index pins original bytes; copied JSON is independently redacted'))
    archive=out/'fe01_on_eval_v4_review.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        tar.add(review,arcname='fe01_on_eval_v4_review')
    (out/'fe01_on_eval_v4_review.tar.gz.sha256').write_text(sha256(archive)+'  '+archive.name+'\n')
    print('REVIEW='+str(archive))


def source_bundle(output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
    require(not dirty,'Commit only task files and use a clean isolated worktree before source release')
    top='on_eval_v4_'+commit[:12]
    archive=output/('fe01_on_eval_v4_'+commit[:12]+'.tar.gz')
    raw=output/'committed_source.tar'
    with raw.open('xb') as f:
        subprocess.run(['git','-C',str(ROOT),'archive','--format=tar','--prefix='+top+'/',commit],stdout=f,check=True)
    marker=dict(commit=commit,execution=source_identity(),external_vendor='read-only original HPC vendor; original115 runtime gate required')
    with tarfile.open(raw,'r:') as src,tarfile.open(archive,'w:gz') as dst:
        for member in src:
            dst.addfile(member,src.extractfile(member) if member.isfile() else None)
        payload=json.dumps(marker,indent=2,allow_nan=False).encode()
        member=tarfile.TarInfo(top+'/FE01_ON_EVAL_SOURCE_RELEASE.json')
        member.size=len(payload);member.mode=0o444
        dst.addfile(member,io.BytesIO(payload))
    raw.unlink()  # Only the newly created intermediate git archive, never an input artifact.
    (output/(archive.name+'.sha256')).write_text(sha256(archive)+'  '+archive.name+'\n')
    write_json(output/'SOURCE_RELEASE.json',dict(commit=commit,branch=subprocess.check_output(
        ['git','-C',str(ROOT),'branch','--show-current'],text=True).strip(),archive=archive.name,
        sha256=sha256(archive),execution=source_identity(),top_directory=top,
        external_dependency='original immutable HPC vendor directory via ON_EVAL_INHERITED_CODE_ROOT/vendor; validated against original115 before model use',
        private_env='excluded',hpc_status='NOT_SUBMITTED'))
    print(str(archive))
