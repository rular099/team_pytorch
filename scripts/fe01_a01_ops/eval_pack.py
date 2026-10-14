#!/usr/bin/env python3
"""Read-only training inputs, explicit multi-root validation and review stages.

Login preparation and packing use stdlib plus the frozen torch-free config and
pack helpers. Only the allocated eval process imports torch. No scheduler here.
"""
import argparse
import copy
import csv
import datetime
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def core():
    root = Path(os.environ['A01_CODE_ROOT']).resolve()
    sys.path[:0] = [str(root/'vendor'), str(root)]
    spec = importlib.util.spec_from_file_location('a01_eval_preflight', root/'scripts/fe01_a01/preflight.py')
    pf = importlib.util.module_from_spec(spec);spec.loader.exec_module(pf)
    return root, pf


def config_from_record(template, recorded):
    """Infer old path expansions, then authenticate the complete source config.

    Effective audit fields are not copied into the source config. Constants come
    from the frozen template; every reconstructed config must match audit SHA.
    No shell env files from discovered directories are executed.
    """
    pattern = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)')
    values = {}
    def infer(raw, actual):
        if isinstance(raw, dict) and isinstance(actual, dict):
            for key in raw.keys() & actual.keys():
                infer(raw[key], actual[key])
        elif isinstance(raw, list) and isinstance(actual, list):
            for a, b in zip(raw, actual):infer(a, b)
        elif isinstance(raw, str) and isinstance(actual, str):
            matches = list(pattern.finditer(raw))
            if not matches:return
            require(len(matches) == 1, 'Ambiguous path template: '+raw)
            match = matches[0];prefix, suffix = raw[:match.start()], raw[match.end():]
            require(actual.startswith(prefix) and actual.endswith(suffix), 'Recorded path/template mismatch: '+raw)
            value = actual[len(prefix):len(actual)-len(suffix) if suffix else None]
            key = match.group(1) or match.group(2)
            require(key not in values or values[key] == value, 'Conflicting recorded path variable: '+key)
            values[key] = value
    infer(template, recorded)
    def expand(value):
        if isinstance(value, dict):return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list):return [expand(v) for v in value]
        if isinstance(value, str):
            def replace(match):
                key = match.group(1) or match.group(2)
                require(key in values, 'Cannot recover original path variable: '+key)
                return values[key]
            return pattern.sub(replace, value)
        return value
    return expand(template)


def file_stat(path):
    stat = Path(path).stat()
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def candidate(index, directory):
    root, pf = core();directory = Path(directory).resolve()
    entry = pf.read(root/'configs/fe01_a01/matrix.json')['new_training'][index]
    require(directory.name == entry['run_id'], 'Wrong run directory name')
    recorded = read(directory/'resolved_config.json')
    cfg = config_from_record(pf.load(root/entry['config'], expand=False), recorded)
    audit = directory.parent/'audits'/cfg['run_id']
    if not audit.is_dir():audit = Path(cfg['output_root'])/'audits'/cfg['run_id']
    lock = read(audit/'protocol.lock.json');gate_root = audit.parent.parent
    code = pf.current_code_sha()
    from fe01_review.provenance import TRAIN_CODE
    require(lock['code_sha256'] == TRAIN_CODE, 'Original115 training identity differs')
    require(lock['status'] == 'AUDIT_PASS' and lock['a01_code_sha256'] == code, 'Audit source/status mismatch')
    require(lock['config_sha256'] == pf.fingerprint(cfg), 'Reconstructed source config SHA differs')
    effective = copy.deepcopy(cfg);effective.update(lock['effective_config'])
    require(recorded == effective, 'Recorded effective config differs from original audit')
    require(lock['world_size'] == 16 and cfg['training']['epochs'] == 12, 'Wrong world/epoch protocol')
    require(pf.sha256(directory/'protocol.lock.json') == pf.sha256(audit/'protocol.lock.json'), 'Run/audit lock mismatch')
    environment = read(gate_root/'environment/environment.json')
    require(environment['status'] == 'PASS' and environment['source']['a01_code_sha256'] == code, 'Environment gate mismatch')
    for stage in ('diagnostics', 'pilot'):
        gate = read(gate_root/stage/cfg['run_id']/(stage+'.json'))
        require(gate['status'] == 'PASS' and gate['a01_code_sha256'] == code and
                gate['audit_lock_sha256'] == pf.sha256(audit/'protocol.lock.json'), stage+' gate mismatch')
    with (directory/'training_curves.csv').open() as stream:curves = list(csv.DictReader(stream))
    require([int(r['epoch']) for r in curves] == list(range(1,13)), 'Training curves are not complete epochs1..12')
    budget = lock['sampling']['budget']
    require(int(curves[-1]['updates']) == budget['total_updates'] == 2544, 'Training update budget incomplete')
    for i, row in enumerate(curves,1):
        require(int(row['updates']) == i*budget['updates_per_epoch'], 'Training curve updates differ')
    stats = {name: file_stat(directory/name) for name in ('best.pth','last.pth')}
    require(all(r['bytes'] > 0 for r in stats.values()), 'Empty checkpoint')
    return dict(index=index, family=entry['family'], mode=entry['mode'], run_id=cfg['run_id'],
                run_dir=str(directory), audit_dir=str(audit), gate_root=str(gate_root), config=cfg,
                audit_lock_sha256=pf.sha256(audit/'protocol.lock.json'), checkpoint_stats=stats,
                curve_sha256=pf.sha256(directory/'training_curves.csv'), metadata_epoch=12, metadata_updates=2544)


def pair_gate(rows):
    by_index = {r['index']:r for r in rows}
    if not {3,4}.issubset(by_index):return
    left, right = (read(Path(by_index[i]['audit_dir'])/'protocol.lock.json') for i in (3,4))
    for key in ('sampling','data_identity','validation_population_sha256','effective_config','world_size'):
        require(left[key] == right[key], 'Selected DiTing ON/OFF differ: '+key)
    for key in ('state_sha256','common_sha256','encoder_sha256'):
        require(left['initial_state'][key] == right['initial_state'][key], 'Selected DiTing init differs: '+key)
    a,b = (copy.deepcopy(by_index[i]['config']) for i in (3,4))
    for cfg in (a,b):
        for key in ('run_id','output_root','absolute_amplitude_mode','pretrained_manifest'):cfg.pop(key,None)
    require(a == b, 'Selected DiTing training semantics differ')


def discover(project):
    root, pf = core();matrix = read(root/'configs/fe01_a01/matrix.json')['new_training']
    overrides = read(os.environ['A01_EVAL_RUN_MAP']) if os.environ.get('A01_EVAL_RUN_MAP') else {}
    search = {Path(os.environ['A01_OUTPUT_ROOT']).resolve()}
    for pattern in ('outputs/*', '.a01_releases/*/outputs/*'):
        search.update(p.resolve() for p in Path(project).glob(pattern) if p.is_dir())
    search.update(Path(p).resolve() for p in os.environ.get('A01_EVAL_SEARCH_ROOTS','').split(os.pathsep) if p)
    rows=[];inventory=[]
    for index, entry in enumerate(matrix):
        paths = {p/entry['run_id'] for p in search if (p/entry['run_id']).is_dir()}
        if str(index) in overrides:paths = {Path(overrides[str(index)]).resolve()}
        accepted=[]
        for path in sorted(paths):
            try:
                row=candidate(index,path);accepted.append(row)
                inventory.append(dict(index=index,path=str(path),status='COMPLETE_METADATA'))
            except (ValueError, KeyError, OSError, TypeError) as error:
                inventory.append(dict(index=index,path=str(path),status='REJECTED',reason=str(error)))
        rows.append(accepted)
    return rows,inventory


def prepare(project, tool_dir, env_file):
    tool_dir=Path(tool_dir);pointer=tool_dir/'eval_session.path'
    if pointer.exists():
        session=Path(pointer.read_text().strip());show(read(session/'plan.json'));return
    choices, inventory=discover(project)
    write(tool_dir/'eval_candidates.json', inventory)
    for index, options in enumerate(choices):
        require(len(options)==1, f'Index {index}: found {len(options)} complete candidates. See eval_candidates.json; '
                'missing/incomplete inputs block submission; multiple candidates require explicit A01_EVAL_RUN_MAP JSON.')
    rows=[options[0] for options in choices];pair_gate(rows)
    destination=Path(project)/'outputs';destination.mkdir(exist_ok=True)
    session=Path(tempfile.mkdtemp(prefix='a01_eval_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S')+'_',dir=destination))
    (session/'tools').mkdir();(session/'slurm').mkdir()
    files=('eval_pack.py','eval_pack.sh')
    if (tool_dir/'eval_revision.py').is_file():files+=('eval_revision.py',)
    for name in files:
        shutil.copy2(tool_dir/name,session/'tools'/name)
        (session/'tools'/name).chmod(0o500 if name.endswith('.sh') else 0o400)
    root,pf=core()
    manifest=dict(schema=1,status='PREPARED_METADATA_ONLY',source_core_sha256=pf.current_code_sha(),
                  code_root=str(root),session=str(session),runs=rows,
                  operations_sha256={name:pf.sha256(session/'tools'/name) for name in files})
    write(session/'plan.json',manifest)
    (session/'plan.json').chmod(0o400)
    # Runtime cache/output settings are a new private snapshot. Input cfgs remain
    # byte-authenticated against their original locks in plan.json.
    import shlex
    settings=Path(env_file).read_text()+'\nexport A01_OUTPUT_ROOT='+shlex.quote(str(session))+'\n'
    settings+='export A01_EVAL_SESSION='+shlex.quote(str(session))+'\n'
    private=session/'runtime.private.env';private.write_text(settings);private.chmod(0o600)
    pointer.write_text(str(session)+'\n');pointer.chmod(0o600)
    show(manifest)


def show(manifest):
    print('INDEX  FAMILY    MODE  EPOCH UPDATES  TRAINING_WEIGHTS_DIRECTORY')
    for row in manifest['runs']:
        print(f"{row['index']:5}  {row['family']:8}  {row['mode']:4}  12    2544     {row['run_dir']}")
    print('EVALUATION_OUTPUT='+manifest['session'])
    print('PREPARED: metadata checked; checkpoint tensor identity is checked on the allocated compute node.')


def load_plan(session):
    session=Path(session).resolve();manifest=read(session/'plan.json');root,pf=core()
    require(str(root)==manifest['code_root'] and pf.current_code_sha()==manifest['source_core_sha256'], 'Prepared core source changed')
    for name,digest in manifest['operations_sha256'].items():
        require(pf.sha256(session/'tools'/name)==digest,'Prepared operations source changed: '+name)
    return manifest


def revision_module():
    path=Path(__file__).with_name('eval_revision.py')
    spec=importlib.util.spec_from_file_location('a01_eval_revision',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def execution_identity(session,index=None):
    manifest=load_plan(session)
    directory=os.environ.get('A01_EVAL_REVISION_DIRECTORY')
    if directory:
        module=revision_module();path=Path(directory)/'revision.json'
        revision=module.validate_revision(session,path,index=index)
        _,pf=core()
        require(pf.sha256(__file__)==revision['operations_sha256']['eval_pack.py'],'Executed revision helper differs')
        return dict(execution_operations_sha256=revision['operations_sha256'],
                    operations_revision=dict(path=str(path.relative_to(Path(session))),sha256=pf.sha256(path)))
    _,pf=core()
    require(pf.sha256(__file__)==manifest['operations_sha256']['eval_pack.py'],'Executed helper differs from original plan')
    return dict(execution_operations_sha256=manifest['operations_sha256'])


def result_gate(session,index):
    _,pf=core();directory=Path(session)/'results'/f'index{index}'
    gate=read(directory/'EVAL_COMPLETE.json')
    require(gate['status']=='PASS' and gate['index']==index,'Incomplete evaluation gate')
    manifest=load_plan(session)
    require(gate['plan_sha256']==pf.sha256(Path(session)/'plan.json'),'Evaluation plan differs')
    expected=manifest['operations_sha256']
    proof=gate.get('operations_revision')
    if proof:
        path=(Path(session)/proof['path']).resolve()
        revision=revision_module().validate_revision(session,path,expected_sha256=proof['sha256'],index=index)
        expected=revision['operations_sha256']
    require(gate['split']=='validation' and gate['source_core_sha256']==manifest['source_core_sha256'] and
            gate['execution_operations_sha256']==expected,'Evaluation provenance differs')
    run_id=manifest['runs'][index]['run_id'];mandatory=set()
    for label in ('selected','epoch12'):
        mandatory.update(f'evaluation/{run_id}/{label}/{name}' for name in
                         ('predictions.csv.gz','support.csv','population.json','scores.csv','fields.csv.gz','checkpoint_identity.json'))
        if index<3:
            mandatory.update(f'evaluation/{run_id}_reused_ON/{label}/{name}' for name in
                             ('predictions.csv.gz','population.json','checkpoint_identity.json'))
        if index!=3:
            mandatory.update(f'comparisons/{label}/{name}' for name in
                             ('comparison.json','paired_event_bootstrap.csv','ON_scores.csv','OFF_scores.csv'))
    require(mandatory.issubset(gate['files_sha256']),'Evaluation seal lacks required artifacts')
    for name,digest in gate['files_sha256'].items():
        path=(directory/name).resolve()
        require(directory.resolve() in path.parents and pf.sha256(path)==digest,'Evaluation artifact changed: '+name)
    if index!=3:
        for label in ('selected','epoch12'):
            require(read(directory/'comparisons'/label/'comparison.json')['status']=='PASS','Comparison did not pass')
    return directory


def preflight(session, stage):
    manifest=load_plan(session)
    if os.environ.get('A01_EVAL_REVISION_DIRECTORY'):
        require(stage in ('eval_team','pack'),'TEAM correction supports only eval_team or pack')
    execution_identity(session,0 if stage=='eval_team' else None)
    if stage=='pack':
        require(not (Path(session)/'review.tar.gz').exists(),'Review package already exists; preserved')
        for index in range(5):result_gate(session,index)
        return
    indices=(0,) if stage=='eval_team' else (0,1,2,3) if stage=='eval_first' else (4,)
    if stage=='eval_diting_off':result_gate(session,3)
    pending=[]
    for index in indices:
        if (Path(session)/'results'/f'index{index}').exists():
            result_gate(session,index)
            continue
        frozen=manifest['runs'][index];actual=candidate(index,frozen['run_dir'])
        require(actual==frozen,'Prepared input metadata changed: index'+str(index))
        pending.append(index)
    pair_gate(manifest['runs'])
    return pending


def checkpoint_metadata(path, row, lock):
    import torch
    _,pf=core();cfg=row['config'];effective=copy.deepcopy(cfg);effective.update(lock['effective_config'])
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    epoch=checkpoint['epoch'];budget=lock['sampling']['budget']
    require(0<epoch<=12 and checkpoint['updates']==epoch*budget['updates_per_epoch'],'Checkpoint epoch/update mismatch')
    require(checkpoint['config']==effective and checkpoint['source_config_sha256']==pf.fingerprint(cfg),'Checkpoint config mismatch')
    require(checkpoint['a01_code_sha256']==lock['a01_code_sha256'] and checkpoint['world_size']==lock['world_size']==16,'Checkpoint source/world mismatch')
    require(checkpoint['identity']==pf.fingerprint(dict(config=effective,audit_lock=lock,world_size=16)),'Checkpoint protocol identity mismatch')
    require(checkpoint['absolute_amplitude_mode']==effective['absolute_amplitude_mode'],'Checkpoint amplitude mode mismatch')
    require(checkpoint['common_initial_state_sha256']==lock['initial_state']['common_sha256'],
            'Checkpoint common initialization mismatch')
    family=effective['model_family']
    encoder_identity=dict(family=family,recorded_state_sha256=checkpoint['encoder_state_sha256'],
                          initial_state_sha256=lock['initial_state']['encoder_sha256'])
    if family=='team_original_scratch':
        from fe01.model import state_fingerprint
        prefix='waveform_model.encoder.'
        state={key[len(prefix):]:value for key,value in checkpoint['model_state_dict'].items() if key.startswith(prefix)}
        require(bool(state),'TEAM encoder state missing')
        actual=state_fingerprint(state)
        require(actual==checkpoint['encoder_state_sha256'],'Trained TEAM encoder bytes/metadata mismatch')
        encoder_identity.update(policy='trainable scratch; authenticate trained bytes',actual_state_sha256=actual)
    elif family in ('phasenet_pretrained_frozen','eqt_pretrained_frozen','diting_pretrained_frozen'):
        require(checkpoint['encoder_state_sha256']==lock['initial_state']['encoder_sha256'],'Frozen encoder mismatch')
        encoder_identity.update(policy='frozen; initial equality plus original loader tensor verification')
    else:
        raise ValueError('Unknown checkpoint model family: '+str(family))
    require([r['epoch'] for r in checkpoint['curve_records']]==list(range(1,epoch+1)),'Checkpoint curves incomplete')
    result={k:checkpoint[k] for k in ('epoch','updates','identity','best_epoch','best_selection_score','curve_records',
                                    'a01_code_sha256','a01_source_commit','world_size','absolute_amplitude_mode')}
    del checkpoint
    result.update(path=str(path),file_sha256=pf.sha256(path),encoder_identity=encoder_identity)
    return result


def evaluate_one(session,index,device='cuda'):
    core()
    from fe01_a01.engine import check_audit
    from fe01_a01.evaluation import export,export_old_ON,compare
    from fe01_review.requests import read_frame
    manifest=load_plan(session);row=manifest['runs'][index]
    executed=execution_identity(session,index)
    require(candidate(index,row['run_dir'])==row,'Evaluation input metadata changed')
    pair_gate(manifest['runs'])
    final=Path(session)/'results'/f'index{index}'
    require(not final.exists(),'Complete evaluation exists')
    if index==4:result_gate(session,3)
    cfg=row['config'];audit=Path(row['audit_dir']);lock=check_audit(cfg,audit)
    run=Path(row['run_dir'])
    last=checkpoint_metadata(run/'last.pth',row,lock);best=checkpoint_metadata(run/'best.pth',row,lock)
    require(last['epoch']==12 and last['updates']==lock['sampling']['budget']['total_updates'],'Last checkpoint incomplete')
    chosen=min(last['curve_records'],key=lambda r:r['validation_selection'])
    require(best['epoch']==last['best_epoch']==chosen['epoch'],'Selected checkpoint differs from preregistered validation selection')
    require(best['best_selection_score']==last['best_selection_score']==chosen['validation_selection'],'Selected checkpoint score mismatch')
    attempts=Path(session)/'attempts';attempts.mkdir(exist_ok=True)
    attempt=Path(tempfile.mkdtemp(prefix=f'eval_index{index}_',dir=attempts))
    for label,metadata in (('selected',best),('epoch12',last)):
        target=attempt/'evaluation'/cfg['run_id']/label
        frame=export(cfg,audit,Path(metadata['path']),metadata['epoch'],target,device)
        write(target/'checkpoint_identity.json',metadata)
        if index<3:
            on=export_old_ON(cfg,audit,label,attempt/'evaluation'/(cfg['run_id']+'_reused_ON')/label,device)
            compare(on,frame,attempt/'comparisons'/label)
        elif index==4:
            on_id=manifest['runs'][3]['run_id']
            on=read_frame(Path(session)/'results/index3/evaluation'/on_id/label/'predictions.csv.gz')
            compare(on,frame,attempt/'comparisons'/label)
    _,pf=core()
    records={str(p.relative_to(attempt)):pf.sha256(p) for p in sorted(attempt.rglob('*')) if p.is_file()}
    write(attempt/'EVAL_COMPLETE.json',dict(status='PASS',index=index,split='validation',plan_sha256=pf.sha256(Path(session)/'plan.json'),
          source_core_sha256=manifest['source_core_sha256'],files_sha256=records,**executed))
    final.parent.mkdir(exist_ok=True);os.rename(attempt,final)
    print('A01_EVAL_COMPLETE index='+str(index)+' output='+str(final),flush=True)


def copy_evidence(source,destination,training_only=False):
    source=Path(source)
    for path in sorted(source.rglob('*')):
        rel=path.relative_to(source)
        if not path.is_file() or path.is_symlink() or any(p in ('traces','sample_journals','.cache') for p in rel.parts):continue
        if path.suffix not in ('.json','.csv','.gz','.png','.md','.tsv','.sha256','.out','.err'):continue
        if path.name.endswith('.tar.gz'):continue
        if training_only and (len(rel.parts)>1 or path.name.startswith('validation_epoch')):continue
        target=Path(destination)/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)


def package(session):
    preflight(session,'pack');manifest=load_plan(session)
    from fe01_a01.pack import review_package
    _,pf=core();attempts=Path(session)/'pack_attempts';attempts.mkdir(exist_ok=True)
    attempt=Path(tempfile.mkdtemp(prefix='review_',dir=attempts));evidence=attempt/'evidence';evidence.mkdir()
    copy_evidence(Path(session)/'results',evidence/'results')
    copy_evidence(Path(session)/'slurm',evidence/'slurm')
    write(evidence/'evaluation_plan.json',manifest)
    write(evidence/'package_execution.json',execution_identity(session))
    proof_paths=set()
    for index in range(5):
        gate=read(Path(session)/'results'/f'index{index}'/'EVAL_COMPLETE.json')
        if gate.get('operations_revision'):proof_paths.add(gate['operations_revision']['path'])
    package_proof=execution_identity(session).get('operations_revision')
    if package_proof:proof_paths.add(package_proof['path'])
    for relative in sorted(proof_paths):
        path=Path(session)/relative
        target=evidence/'evaluation_execution'/path.parent.name/'revision.json'
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    if (Path(session)/'jobs.tsv').is_file():shutil.copy2(Path(session)/'jobs.tsv',evidence/'jobs.tsv')
    write(evidence/'PACKAGE_SCOPE.json',dict(
        split='validation',checkpoint_policy='original selected best plus fixed epoch12',
        source_seals='EVAL_COMPLETE files_sha256 refer to original unredacted evaluation artifacts',
        archive_seal='A01_REVIEW_MANIFEST refers to actual redacted bytes inside this tar',
        excluded=['weights','raw waveforms','failed evaluation attempts','sample journals','trace arrays','private env files'],
    ))
    proof_records=[]
    for row in manifest['runs']:
        index=row['index'];gate_root=Path(row['gate_root'])
        copy_evidence(row['run_dir'],evidence/'training'/f'index{index}',training_only=True)
        for stage in ('audits','diagnostics','pilot'):
            copy_evidence(gate_root/stage/row['run_id'],evidence/'training'/f'index{index}'/stage)
        for operation in sorted((gate_root/'operations').glob('dcu_*')):
            for name in (f'execution_identity_index{index}.json',f'train_action_{index}.json',
                         f'archive_index{index}.json',f'checkpoint_before_{index}/manifest.json'):
                path=operation/name
                if path.is_file():
                    target=evidence/'training_execution'/f'index{index}'/operation.name/name
                    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
                    proof_records.append(dict(index=index,path=str(path),sha256=pf.sha256(path)))
    write(evidence/'training_execution_provenance.json',dict(records=proof_records,
          limitation='Collected operation evidence; absence never implies identical executed source or successful training.'))
    review_package(evidence)
    target=Path(session)/'review.tar.gz';require(not target.exists(),'Package already exists')
    os.rename(evidence/'review.tar.gz',target)
    shutil.copy2(evidence/'review.tar.gz.sha256',Path(session)/'review.tar.gz.sha256')
    print('A01_REVIEW_PACKAGE='+str(target),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','preflight','evaluate','pack','show'])
    parser.add_argument('--project');parser.add_argument('--tool-dir');parser.add_argument('--env-file')
    parser.add_argument('--session');parser.add_argument('--stage',choices=['eval_first','eval_diting_off','eval_team','pack'])
    parser.add_argument('--index',type=int,choices=range(5))
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.project,args.tool_dir,args.env_file)
    elif args.mode=='preflight':
        indices=preflight(args.session,args.stage)
        print(','.join(map(str,indices or [])))
    elif args.mode=='evaluate':evaluate_one(args.session,args.index)
    elif args.mode=='pack':package(args.session)
    else:show(load_plan(args.session))


if __name__=='__main__':main()
