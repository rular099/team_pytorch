"""Explicit successor run for the a5c677c inventory manifest-path bug.

Only unchanged, sealed CPU results can be transported. The old run and its
source are read-only; an import receipt distinguishes transport from execution.
"""
import argparse
import ast
import datetime as dt
import fcntl
import os
from pathlib import Path
import shlex
import shutil
import tempfile
import uuid

from .contracts import ROOT, source_identity
from .io import attempt, fingerprint, read_json, require, sha256, verify_seal, write_json
from . import orchestration as ops

BASE_RELEASE = 'a5c677c06ae67ed5c47b0cab0905908e1275cf81'
BASE_EXECUTION = '7a5d3704020a42097365ac4aeb5321363bacf6b63eac55a3f5e68dceebbbf868'
CPU_REUSE = ('compare_existing', 'covariates', 'crossing_truth')
ADDED_EXECUTION_FILES = {'fe01_a01_compare/recovery.py',
                         'scripts/fe01_a01_compare/recover_inventory.sh'}


def cpu_source_compatibility(old_root, old_identity, new_root, new_identity):
    """Prove every old execution file unchanged except two model-only functions."""
    require(old_identity['result_commit'] == BASE_RELEASE and
            old_identity['execution_sha256'] == BASE_EXECUTION,
            'Recovery supports only the exact a5c677c source release')
    old_files, new_files = old_identity['files_sha256'], new_identity['files_sha256']
    require(fingerprint(old_files) == BASE_EXECUTION, 'Old execution manifest changed')
    require(fingerprint(new_files) == new_identity['execution_sha256'], 'New execution manifest changed')
    require(set(new_files) - set(old_files) == ADDED_EXECUTION_FILES and
            not set(old_files) - set(new_files), 'Unexpected added/removed execution source')
    permitted = 'fe01_a01_compare/sources.py'
    for name, digest in old_files.items():
        require(sha256(Path(old_root) / name) == digest, 'Original release changed: ' + name)
        require(sha256(Path(new_root) / name) == new_files[name], 'Recovery release changed: ' + name)
        if name != permitted:
            require(new_files[name] == digest, 'CPU dependency changed; reuse forbidden: ' + name)
    for name in ADDED_EXECUTION_FILES:
        require(sha256(Path(new_root) / name) == new_files[name], 'Recovery source changed: ' + name)
    def unchanged_ast(path):
        tree = ast.parse(path.read_text())
        allowed = {'inspect', 'runtime_config'}
        require({n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in allowed} == allowed,
                'Missing model-only recovery functions')
        tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name in allowed)]
        return ast.dump(tree, include_attributes=False)
    require(unchanged_ast(Path(old_root) / permitted) == unchanged_ast(Path(new_root) / permitted),
            'CPU source functions changed; reuse forbidden')
    return dict(status='PASS', original_execution_sha256=BASE_EXECUTION,
                execution_sha256=new_identity['execution_sha256'],
                permitted_function_changes=['sources.inspect', 'sources.runtime_config'],
                unchanged_cpu_stages=list(CPU_REUSE))


def verify_original(session):
    session = Path(session).resolve()
    require(sha256(session / 'plan.json') == read_json(session / 'plan_pin.json')['sha256'],
            'Original run plan changed')
    plan = read_json(session / 'plan.json')
    require(read_json(session / 'source_identity.json') == plan['source_identity'], 'Original source record changed')
    require(fingerprint(read_json(session / 'analysis_contract.json')) == plan['contract_sha256'],
            'Original contract changed')
    require(sha256(Path(plan['source_session']) / 'plan.json') == plan['source_plan_sha256'],
            'Original A01 plan changed')
    require(ops.scientific_environment() == plan['scientific_environment'],
            'Scientific environment changed; refuse to transport results')
    require(Path(os.environ['ON_EVAL_SOURCE_SESSION']).resolve() == Path(plan['source_session']).resolve(),
            'A01 source session differs')
    release = Path(plan['release_root'])
    marker = read_json(release / 'FE01_ON_EVAL_SOURCE_RELEASE.json')
    require(marker['commit'] == BASE_RELEASE and
            marker['execution']['files_sha256'] == plan['source_identity']['files_sha256'],
            'Original release marker differs')
    return plan, cpu_source_compatibility(release, plan['source_identity'], ROOT, source_identity())


def original_stage_identity(plan, session, stage, index):
    require(not ops.dependency_keys(stage), 'Only independent CPU stages can be reused')
    return dict(execution_sha256=plan['source_identity']['execution_sha256'],
                contract_sha256=plan['contract_sha256'], plan_sha256=sha256(Path(session) / 'plan.json'),
                stage=stage, task=index, dependency_seals_sha256={})


def import_cpu_results(original_session, session, old_plan, new_plan):
    """Transport completed CPU bytes into a new sealed attempt, with both identities."""
    rows = []
    for stage in CPU_REUSE:
        require(old_plan['tasks'][stage] == new_plan['tasks'][stage], 'Task population differs: ' + stage)
        for task in old_plan['tasks'][stage]:
            index = task['index']
            source = Path(original_session) / 'stages' / stage / str(index)
            if not source.exists():
                final = Path(session) / 'stages' / stage / str(index)
                if final.exists():
                    verify_seal(final, ops.stage_identity(session, stage, index))
                rows.append(dict(stage=stage, index=index, status='ALREADY_COMPLETE_IN_SUCCESSOR'
                                 if final.exists() else 'WAIT_FOR_ORIGINAL_RESULT'))
                continue
            old_seal = verify_seal(source, original_stage_identity(old_plan, original_session, stage, index))
            original_seal_sha = sha256(source / 'SEAL.json')
            identity = ops.stage_identity(session, stage, index)
            with attempt(session, stage, index, identity) as out:
                if out is None:
                    rows.append(dict(stage=stage, index=index, status='ALREADY_COMPLETE_IN_SUCCESSOR'))
                    continue
                provenance = out / 'REUSE_PROVENANCE'
                require(not (source / 'REUSE_PROVENANCE').exists(), 'Nested recovery is unsupported')
                provenance.mkdir()
                shutil.copy2(source / 'SEAL.json', provenance / 'source_SEAL.json')
                methods = {}
                for name in old_seal['files_sha256']:
                    src = source / name
                    require(not src.is_symlink(), 'Symlink in original sealed output: ' + name)
                    dest = provenance / 'source_ATTEMPT.json' if name == 'ATTEMPT.json' else out / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(src, dest)
                        methods[name] = 'hard_link_read_only_result'
                    except OSError:
                        shutil.copy2(src, dest)
                        methods[name] = 'copy_derived_result'
                    require(sha256(dest) == old_seal['files_sha256'][name], 'Transport bytes differ: ' + name)
                verify_seal(source, original_stage_identity(old_plan, original_session, stage, index))
                require(sha256(source / 'SEAL.json') == original_seal_sha, 'Source seal changed during import')
                write_json(provenance / 'receipt.json', dict(
                    kind='UNCHANGED_CPU_RESULT_TRANSPORT; not recomputed by successor source',
                    original_session=str(Path(original_session).resolve()), original_seal_sha256=original_seal_sha,
                    original_identity=original_stage_identity(old_plan, original_session, stage, index),
                    original_execution=old_plan['source_identity'], transport_identity=identity,
                    original_files_sha256=old_seal['files_sha256'], storage_methods=methods,
                    original_attempt_location='REUSE_PROVENANCE/source_ATTEMPT.json'))
            rows.append(dict(stage=stage, index=index, status='IMPORTED_SEAL_VERIFIED'))
    return rows


def import_data_caches(original_session, session):
    """These stat/SHA caches have no run/source dependency; retain their exact seal."""
    imported = []
    for source in sorted((Path(original_session) / 'inputs/stages/data_identity').glob('*')):
        if not source.is_dir():
            continue
        s = verify_seal(source)
        expected = {k: v for k, v in s.items() if k not in ('status', 'files_sha256')}
        parent = Path(session) / 'inputs/stages/data_identity'
        parent.mkdir(parents=True, exist_ok=True)
        with (parent / (source.name + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            final = parent / source.name
            if final.exists():
                verify_seal(final, expected)
                continue
            attempts = Path(session) / 'inputs/attempts/reused_data_identity'
            attempts.mkdir(parents=True, exist_ok=True)
            temp = Path(tempfile.mkdtemp(prefix=source.name + '_', dir=attempts))
            shutil.copytree(source, temp, dirs_exist_ok=True)
            verify_seal(temp, expected)
            require(sha256(temp / 'SEAL.json') == sha256(source / 'SEAL.json'), 'Cache seal changed')
            temp.rename(final)
            imported.append(source.name)
    return imported


def recover(original_session, old_env, run_id, env_file, submit_inventory=False):
    old_session = Path(original_session).resolve()
    old, proof = verify_original(old_session)
    require(Path(old['env_file']).resolve() == Path(old_env).resolve(), 'Wrong original private environment file')
    session = ops.session_path(run_id)
    require(session.resolve() != old_session, 'Use a distinct successor run ID')
    env_file = Path(env_file).resolve()
    require(env_file != Path(old_env).resolve(), 'Never overwrite original environment')
    text = '# Inventory fix successor; original environment is read-only.\nsource ' + shlex.quote(str(Path(old_env).resolve()))
    text += '\nexport ON_EVAL_RELEASE_ROOT=' + shlex.quote(str(ROOT)) + '\n'
    if env_file.exists():
        require(env_file.read_text() == text, 'Existing successor private file differs')
    else:
        env_file.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(os.open(env_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as f:
            f.write(text)
    ops.prepare(run_id, env_file)
    new = ops.verify_plan(session)
    for key in ('cohort', 'tasks', 'source_session', 'source_plan_sha256', 'contract_sha256',
                'scientific_environment', 'block_events', 'pilot_events'):
        require(new[key] == old[key], 'Successor population/input plan differs: ' + key)
    require(sha256(session / 'static_grid_layout.csv') == sha256(old_session / 'static_grid_layout.csv'),
            'Grid layout changed')
    caches = import_data_caches(old_session, session)
    rows = import_cpu_results(old_session, session, old, new)
    receipt = dict(source_compatibility=proof, original_session=str(old_session), successor_session=str(session),
                   imported_caches=caches, tasks=rows, scope='CPU result transport; inventory must rerun')
    write_json(session / 'recoveries' / (dt.datetime.now().strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex + '.json'), receipt)
    from collections import Counter
    for stage in CPU_REUSE:
        print('RECOVERY', stage, dict(Counter(r['status'] for r in rows if r['stage'] == stage)), flush=True)
    print('SUCCESSOR_SESSION=' + str(session), flush=True)
    print('SUCCESSOR_ENV=' + str(env_file), flush=True)
    print('原目录和在跑作业保持原状。WAIT_FOR_ORIGINAL_RESULT 项等原作业成功后再次运行本恢复脚本。', flush=True)
    if submit_inventory:
        # This requires the explicitly invoked recovery submission script/flag.
        ops.submit(session, 'inventory', resume=True)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-env', required=True)
    p.add_argument('--from-run-id', default='on_v4_20261022')
    p.add_argument('--run-id', default='on_v4_20261025_inventory_fix')
    p.add_argument('--env', required=True)
    p.add_argument('--submit-inventory', action='store_true')
    a = p.parse_args()
    recover(ops.session_path(a.from_run_id), a.old_env, a.run_id, a.env, a.submit_inventory)


if __name__ == '__main__':
    main()
