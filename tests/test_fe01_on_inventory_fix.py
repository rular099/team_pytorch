import copy
import json
from pathlib import Path

import pytest

from fe01_a01_compare import recovery
from fe01_a01_compare.io import fingerprint, read_json, seal, sha256, verify_seal, write_json
from fe01_a01_compare.sources import runtime_config


@pytest.mark.parametrize('family,filename', [
    ('diting_pretrained_frozen', 'diting_manifest.json'),
    ('phasenet_pretrained_frozen', 'pretrained_manifest.json'),
    ('eqt_pretrained_frozen', 'pretrained_manifest.json'),
    ('team_original_scratch', 'pretrained_manifest.json')])
def test_mixed_inventory_preserves_recorded_manifest_and_strict_sha(tmp_path, monkeypatch, family, filename):
    weights = tmp_path / 'weights'; weights.mkdir()
    (weights / 'pretrained_manifest.json').write_text('{"models": ["phasenet", "eqtransformer"]}\n')
    (weights / 'diting_manifest.json').write_text('{"models": ["diting"]}\n')
    cfg = {'model_family': family, 'pretrained_manifest': '/original/' + filename,
           'diting_config': '/original/diting.yml', 'data': {'root': '/original/data', 'split_manifest': '/original/split'}}
    saved = copy.deepcopy(cfg)
    monkeypatch.delenv('A01_DITING_MANIFEST', raising=False)
    monkeypatch.setenv('FE01_WEIGHTS_ROOT', str(weights))
    monkeypatch.setenv('FE01_CODE_ROOT', str(tmp_path / 'code'))
    monkeypatch.setenv('DATA_ROOT', str(tmp_path / 'data'))
    monkeypatch.setenv('FE01_SPLIT_MANIFEST', str(tmp_path / 'split.csv'))
    expected = None if family == 'team_original_scratch' else sha256(weights / filename)
    actual = runtime_config(cfg, {'pretrained_manifest_sha256': expected})
    assert actual['pretrained_manifest'] == str(weights / filename)
    assert actual['data'] == {'root': str(tmp_path / 'data'), 'split_manifest': str(tmp_path / 'split.csv')}
    assert actual['diting_config'].endswith('/code/diting/config/diting_1200m_backbone_attnpool.yml')
    assert cfg == saved
    if family == 'diting_pretrained_frozen':
        # Exact historical failure: a general relocation picks the other valid JSON.
        assert sha256(weights / 'pretrained_manifest.json') != expected


def test_no_manifest_search_fallback_or_sha_relaxation(tmp_path, monkeypatch):
    monkeypatch.delenv('A01_DITING_MANIFEST', raising=False)
    monkeypatch.setenv('FE01_WEIGHTS_ROOT', str(tmp_path))
    cfg = {'model_family': 'diting_pretrained_frozen', 'pretrained_manifest': '/old/diting_manifest.json', 'data': {}}
    correct = tmp_path / 'pretrained_manifest.json'; correct.write_text('the original bytes')
    expected = sha256(correct)
    manifest = tmp_path / 'diting_manifest.json'; manifest.write_text('wrong bytes')
    with pytest.raises(ValueError, match=r'family=diting_pretrained_frozen.*diting_manifest.json.*expected=.*actual='):
        runtime_config(cfg, {'pretrained_manifest_sha256': expected})
    manifest.unlink()
    with pytest.raises(ValueError, match='manifest missing'):
        runtime_config(cfg, {'pretrained_manifest_sha256': expected})


def test_original_a01_manifest_outside_general_weights_directory_is_preserved(tmp_path, monkeypatch):
    general = tmp_path / 'general'; general.mkdir()
    (general / 'pretrained_manifest.json').write_text('phasenet/eqt manifest')
    original = tmp_path / 'a01_run/weights/diting_manifest.json'
    original.parent.mkdir(parents=True); original.write_text('exact original DiTing manifest')
    monkeypatch.setenv('FE01_WEIGHTS_ROOT', str(general))
    cfg = {'model_family': 'diting_pretrained_frozen', 'pretrained_manifest': str(original), 'data': {}}
    lock = {'pretrained_manifest_sha256': sha256(original)}
    assert runtime_config(cfg, lock)['pretrained_manifest'] == str(original)
    original.write_text('tampered')
    (general / 'diting_manifest.json').write_text('exact original DiTing manifest')
    with pytest.raises(ValueError, match='SHA differs'):
        runtime_config(cfg, lock)  # No fallback even when another path has the expected bytes.


def test_missing_original_uses_existing_a01_manifest_setting(tmp_path, monkeypatch):
    actual = tmp_path / 'diting_manifest.json'; actual.write_text('pinned manifest')
    monkeypatch.setenv('A01_DITING_MANIFEST', str(actual))
    monkeypatch.setenv('FE01_WEIGHTS_ROOT', str(tmp_path / 'general'))
    cfg = {'model_family': 'diting_pretrained_frozen', 'pretrained_manifest': '/gone/diting_manifest.json', 'data': {}}
    assert runtime_config(cfg, {'pretrained_manifest_sha256': sha256(actual)})['pretrained_manifest'] == str(actual)


def test_scratch_does_not_read_manifest_and_no_environment_keeps_paths(monkeypatch):
    for key in ('FE01_WEIGHTS_ROOT', 'FE01_CODE_ROOT', 'DATA_ROOT', 'FE01_SPLIT_MANIFEST'):
        monkeypatch.delenv(key, raising=False)
    cfg = {'model_family': 'team_original_scratch', 'pretrained_manifest': '/missing/pretrained_manifest.json', 'data': {}}
    assert runtime_config(cfg, {'pretrained_manifest_sha256': None}) == cfg


def compatibility_fixture(tmp_path, monkeypatch):
    old, new = tmp_path / 'old_source', tmp_path / 'new_source'
    originals = {'fe01_a01_compare/sources.py': '"""same globals"""\nVALUE=1\ndef load_export():\n    return 1\ndef inspect():\n    return 1\ndef runtime_config():\n    return "pretrained_manifest.json"\n',
                 'fe01_a01_compare/statistics.py': 'def score():\n    return 1\n'}
    for name, text in originals.items():
        for directory in (old, new):
            p = directory / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
    source = new / 'fe01_a01_compare/sources.py'
    source.write_text(source.read_text().replace('return "pretrained_manifest.json"', 'return "preserved basename"'))
    for name in recovery.ADDED_EXECUTION_FILES:
        p = new / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# recovery transport only\n')
    old_files = {name: sha256(old / name) for name in originals}
    monkeypatch.setattr(recovery, 'BASE_EXECUTION', fingerprint(old_files))
    def identities():
        new_files = {name: sha256(new / name) for name in set(originals) | recovery.ADDED_EXECUTION_FILES}
        return dict(result_commit=recovery.BASE_RELEASE, execution_sha256=fingerprint(old_files), files_sha256=old_files), dict(
            execution_sha256=fingerprint(new_files), files_sha256=new_files)
    return old, new, identities


def test_cpu_reuse_requires_exact_original_and_unchanged_analysis(tmp_path, monkeypatch):
    old, new, identities = compatibility_fixture(tmp_path, monkeypatch)
    old_id, new_id = identities()
    assert recovery.cpu_source_compatibility(old, old_id, new, new_id)['status'] == 'PASS'
    old_id['result_commit'] = 'other'
    with pytest.raises(ValueError, match='exact a5c677c'):
        recovery.cpu_source_compatibility(old, old_id, new, new_id)
    (new / 'fe01_a01_compare/statistics.py').write_text('def score():\n    return 2\n')
    with pytest.raises(ValueError, match='CPU dependency changed'):
        recovery.cpu_source_compatibility(old, identities()[0], new, identities()[1])


def test_cpu_reuse_rejects_changed_reader_and_original_source(tmp_path, monkeypatch):
    old, new, identities = compatibility_fixture(tmp_path, monkeypatch)
    source = new / 'fe01_a01_compare/sources.py'
    source.write_text(source.read_text().replace('def load_export():\n    return 1', 'def load_export():\n    return 2'))
    with pytest.raises(ValueError, match='CPU source functions changed'):
        recovery.cpu_source_compatibility(old, identities()[0], new, identities()[1])
    (old / 'fe01_a01_compare/statistics.py').write_text('tampered')
    with pytest.raises(ValueError, match='Original release changed'):
        recovery.cpu_source_compatibility(old, identities()[0], new, identities()[1])


def result_fixture(tmp_path, monkeypatch):
    old, new = tmp_path / 'old', tmp_path / 'new'
    old.mkdir(); new.mkdir()
    write_json(old / 'plan.json', {'old': 'immutable plan'})
    tasks = {stage: [{'index': 0}, {'index': 1}] for stage in recovery.CPU_REUSE}
    plan = {'tasks': tasks, 'source_identity': {'execution_sha256': 'old-execution'}, 'contract_sha256': 'same-contract'}
    def target_identity(session, stage, index):
        return {'execution_sha256': 'new-execution', 'plan_sha256': 'new-plan', 'stage': stage, 'task': index}
    monkeypatch.setattr(recovery.ops, 'stage_identity', target_identity)
    for stage in recovery.CPU_REUSE:
        p = old / 'stages' / stage / '0'; p.mkdir(parents=True)
        write_json(p / 'ATTEMPT.json', {'old_execution': 'original computation'})
        (p / 'result.csv').write_text('event_id,value\n00001,0.2\n')
        seal(p, **recovery.original_stage_identity(plan, old, stage, 0))
    before = {str(p.relative_to(old)): sha256(p) for p in old.rglob('*') if p.is_file()}
    return old, new, plan, before


def test_result_transport_preserves_old_bytes_and_provenance_and_is_idempotent(tmp_path, monkeypatch):
    old, new, plan, before = result_fixture(tmp_path, monkeypatch)
    rows = recovery.import_cpu_results(old, new, plan, plan)
    assert sum(r['status'] == 'IMPORTED_SEAL_VERIFIED' for r in rows) == 3
    assert sum(r['status'] == 'WAIT_FOR_ORIGINAL_RESULT' for r in rows) == 3
    for stage in recovery.CPU_REUSE:
        p = new / 'stages' / stage / '0'
        verify_seal(p, recovery.ops.stage_identity(new, stage, 0))
        receipt = read_json(p / 'REUSE_PROVENANCE/receipt.json')
        assert 'not recomputed' in receipt['kind']
        assert receipt['original_identity']['execution_sha256'] == 'old-execution'
        assert receipt['transport_identity']['execution_sha256'] == 'new-execution'
        assert read_json(p / 'REUSE_PROVENANCE/source_ATTEMPT.json') == {'old_execution': 'original computation'}
    assert before == {str(p.relative_to(old)): sha256(p) for p in old.rglob('*') if p.is_file()}
    again = recovery.import_cpu_results(old, new, plan, plan)
    assert sum(r['status'] == 'ALREADY_COMPLETE_IN_SUCCESSOR' for r in again) == 3
    assert len(list((new / 'attempts').glob('*/*'))) == 0  # successes moved, no duplicate attempts


def test_transport_rejects_corruption_or_different_population(tmp_path, monkeypatch):
    old, new, plan, before = result_fixture(tmp_path, monkeypatch)
    changed = copy.deepcopy(plan); changed['tasks']['compare_existing'][0]['events'] = [['different', 'event']]
    with pytest.raises(ValueError, match='Task population differs'):
        recovery.import_cpu_results(old, new, plan, changed)
    (old / 'stages/compare_existing/0/result.csv').write_text('changed')
    with pytest.raises(ValueError, match='Sealed files changed'):
        recovery.import_cpu_results(old, new, plan, plan)
    assert not (new / 'stages').exists()


def test_shared_data_cache_transport_retains_exact_seal(tmp_path):
    old, new = tmp_path / 'old', tmp_path / 'new'
    source = old / 'inputs/stages/data_identity/key'; source.mkdir(parents=True)
    write_json(source / 'data_identity.json', {'exact stat/sha': 'unchanged'})
    seal(source, stat_signature_sha256='stat', lock_sha256='original-lock')
    before = sha256(source / 'SEAL.json')
    assert recovery.import_data_caches(old, new) == ['key']
    target = new / 'inputs/stages/data_identity/key'
    assert sha256(target / 'SEAL.json') == before
    assert recovery.import_data_caches(old, new) == []
    (target / 'data_identity.json').write_text('changed')
    with pytest.raises(ValueError, match='Sealed files changed'):
        recovery.import_data_caches(old, new)


def test_successor_recovery_prepares_reuses_and_submits_only_inventory(tmp_path, monkeypatch):
    import os
    import pandas as pd
    from fe01_a01_compare.sources import pins
    from fe01_a01_compare.io import attempt
    from test_fe01_on_runtime import fake_scheduler
    log = fake_scheduler(tmp_path, monkeypatch)
    source = tmp_path / 'a01'; source.mkdir()
    write_json(source / 'plan.json', {'source': 'fake immutable A01 plan'})
    pin = next(x for x in pins() if x['family'] == 'TEAM' and x['checkpoint_policy'] == 'selected')
    export = source / pin['source']; export.parent.mkdir(parents=True)
    pd.DataFrame([dict(dataset_id='japan_2020.hdf5', event_id=f'{i:014d}', station_id=f'S{j}', latitude=35+j*.01,
                      longitude=139+j*.01, input_count=j+1, magnitude=3+i/1310)
                  for i in range(1310) for j in range(2)]).to_csv(export, index=False)
    monkeypatch.setenv('ON_EVAL_SOURCE_SESSION', str(source))
    monkeypatch.setenv('ON_EVAL_OUTPUT_ROOT', str(tmp_path / 'runs'))
    monkeypatch.setattr('fe01_a01_compare.sources.verify_export_seals', lambda p: [])
    old_env = tmp_path / 'old.private.env'; old_env.write_text('# explicit old environment\n')
    recovery.ops.prepare('original', old_env)
    old_session = recovery.ops.session_path('original')
    old_plan = read_json(old_session / 'plan.json')
    with attempt(old_session, 'covariates', 0, recovery.ops.stage_identity(old_session, 'covariates', 0)) as out:
        (out / 'physical_metadata.csv').write_text('fixed output\n')
    before = {str(p.relative_to(old_session)): sha256(p) for p in old_session.rglob('*') if p.is_file()}
    # The full original-source compatibility gate is exercised separately above.
    monkeypatch.setattr(recovery, 'verify_original', lambda p: (old_plan, {'status': 'PASS_FIXTURE'}))
    env = tmp_path / 'successor.private.env'
    receipt = recovery.recover(old_session, old_env, 'fixed', env, submit_inventory=True)
    assert any(r['stage'] == 'covariates' and r['status'] == 'IMPORTED_SEAL_VERIFIED' for r in receipt['tasks'])
    fixed = recovery.ops.session_path('fixed')
    assert recovery.ops.completed(fixed, 'covariates', 0)
    jobs = list((fixed / 'jobs').glob('*.json'))
    assert len(jobs) == 1 and read_json(jobs[0])['stage'] == 'inventory'
    assert len(log.read_text().splitlines()) == 1
    assert env.stat().st_mode & 0o777 == 0o600
    assert before == {str(p.relative_to(old_session)): sha256(p) for p in old_session.rglob('*') if p.is_file()}
    monkeypatch.setenv('FAKE_QUEUE', '999_0|RUNNING')
    recovery.recover(old_session, old_env, 'fixed', env, submit_inventory=True)
    assert len(log.read_text().splitlines()) == 1
    with attempt(old_session, 'covariates', 1, recovery.ops.stage_identity(old_session, 'covariates', 1)) as out:
        (out / 'physical_metadata.csv').write_text('finished after recovery\n')
    recovery.recover(old_session, old_env, 'fixed', env, submit_inventory=True)
    assert recovery.ops.completed(fixed, 'covariates', 1)
    assert len(log.read_text().splitlines()) == 1
