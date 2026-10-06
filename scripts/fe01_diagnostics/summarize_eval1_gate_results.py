#!/usr/bin/env python3
"""Summarize downloaded EVAL1 gates without importing Torch or opening HDF5.

Raw artifacts are read-only. Large request/label tables and caches stay local;
portable JSON, scheduler logs and derived counts are exported for review.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import re
import statistics
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fe01_review.provenance import training_identity, evaluation_identity


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return stream_digest(stream)


def stream_digest(stream):
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        result.update(chunk)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def table(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--training-root', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    src, out = args.input, args.output
    require(not out.exists(), 'Output exists; choose a new directory')
    source_files = sorted(p for p in src.rglob('*') if p.is_file())
    source_hashes = {str(p.relative_to(src)): digest(p) for p in source_files}
    seals = []
    for manifest in sorted(src.rglob('artifact_manifest.sha256')):
        count = 0
        for line in manifest.read_text().splitlines():
            expected, name = line.split('  ', 1)
            path = manifest.parent / name
            require(path.resolve().is_relative_to(src.resolve()), 'Manifest path escapes source')
            require(path.is_file() and digest(path) == expected, 'Artifact seal differs: ' + str(path))
            count += 1
        seals.append(dict(manifest=str(manifest.relative_to(src)), entries=count, status='PASS'))
    archive = None
    if args.archive:
        with tarfile.open(args.archive) as packed:
            members = [m for m in packed.getmembers() if m.isfile()]
            archive_paths = set()
            for member in members:
                path = str(Path(member.name).relative_to(src.name))
                require(path not in archive_paths, 'Duplicate archive member: ' + path)
                archive_paths.add(path)
                require(path in source_hashes, 'Archive member absent from download: ' + path)
                with packed.extractfile(member) as stream:
                    require(stream_digest(stream) == source_hashes[path], 'Archive byte mismatch: ' + path)
            require(archive_paths == set(source_hashes), 'Archive and folder file lists differ')
        archive = dict(name=args.archive.name, bytes=args.archive.stat().st_size,
                       sha256=digest(args.archive), files=len(members), folder_matches='PASS')

    jobs = read(src / 'jobs_manifest.json')
    frozen = read(src / 'identity/frozen_checkpoint_manifest.json')['models']
    inventory = {x['run_id']: x['selected'] for x in read(src / 'identity/checkpoint_inventory.json')}
    expected_module = jobs['evaluation_module_sha']
    current_training = training_identity()
    current_evaluation = evaluation_identity()
    require(current_evaluation['evaluation_module_sha'] == expected_module, 'Current evaluation source differs')
    source_checks = []
    for path in [src / 'identity/provenance.json', src / 'requests/provenance.json',
                 src / 'reference/provenance.json', *sorted((src / 'verification').glob('*/verification.json'))]:
        data = read(path)
        if data.get('status') == 'BLOCKED':
            continue
        require(data['evaluation_module_sha'] == expected_module, 'Evidence module SHA differs')
        require(data['training_code_sha'] == current_training['code_sha256'], 'Training source SHA differs')
        for name, expected in data['files_sha256'].items():
            require(digest(ROOT / name) == expected, 'Recorded source byte SHA differs: ' + name)
        require(digest(ROOT / 'eval_checkpoint.py') == data['legacy_loader_sha256'], 'Legacy loader differs')
        source_checks.append(dict(path=str(path.relative_to(src)), files=len(data['files_sha256']), status='PASS'))
    require(read(src / 'requests/provenance.json')['hydrated'] is True, 'Request planning unfinished')
    require(digest(args.plan / 'fixed_requests.csv.gz') == jobs['fixed_request_sha256'], 'Fixed request plan differs')
    require(digest(args.plan / 'random_time_draws.csv') == jobs['random_draw_sha256'], 'Random draw plan differs')
    probes = table(args.plan / 'verification_decisions.csv')
    probe_keys = {(x['dataset_id'], x['event_id'], float(x['elapsed_time']), x['geometry_protocol']) for x in probes}
    rows, lock_checks = [], []
    for model in frozen:
        name = model['run_id']
        selected = inventory[name]
        require(selected['status'] == model['inventory_status'] == 'IDENTITY_PASS', 'Checkpoint identity failed: ' + name)
        require(selected['internal_epoch'] == model['epoch'], 'Checkpoint epoch differs: ' + name)
        require(selected['checkpoint_sha256'] == model['checkpoint_sha256'], 'Frozen checkpoint SHA differs')
        gate = read(src / 'verification' / name / 'verification.json')
        checks = gate.get('checks', [])
        if gate['status'] == 'PASS':
            require(gate['checkpoint_sha256'] == selected['checkpoint_sha256'] and gate['checkpoint_epoch'] == model['epoch'], 'Verification weight differs')
            require(gate['request_manifest_sha256'] == jobs['fixed_request_sha256'], 'Verification plan differs')
            keys = {(c['dataset_id'], c['event_id'], c['elapsed_time'], c['geometry_protocol']) for c in checks}
            require(keys == probe_keys and len(keys) == gate['decisions'] == len(checks), 'Probe selection differs')
            require(all(c['clock_and_labels_exact'] for c in checks) and gate['unchanged_buffers'], 'Clock/buffer check failed')
            for c in checks:
                if 'L0_same_tensor_original_wrapper' in c:
                    require(all(c[k] == 'PASS' for k in ['L0_same_tensor_original_wrapper', 'query_batch_reverse_single', 'query_append_unrelated', 'same_T_cache']) and c['different_T_cache_rejected'], 'Query/cache check failed')
                if 'future_NaN' in c:
                    require(c['future_NaN'].startswith('PASS') and c['future_large_pulse'].startswith('PASS'), 'Future perturbation check failed')
        item = dict(run_id=name, kind=model['kind'], model_family=model.get('model_family', model['kind']),
                    seed=model.get('seed'), epoch=model['epoch'], identity_status=selected['status'],
                    forward_status=gate['status'], reason=gate.get('reason', ''), checkpoint_sha256=selected['checkpoint_sha256'],
                    probe_events=len({(c['dataset_id'], c['event_id']) for c in checks}), decisions=len(checks),
                    targets=sum(c['targets'] for c in checks), query_cache_decisions=sum('same_T_cache' in c for c in checks),
                    future_perturbation_decisions=sum('future_NaN' in c for c in checks),
                    atol=gate.get('atol'), rtol=gate.get('rtol'),
                    prediction_max_abs_delta=None, mdn_weights_max_abs_delta=None,
                    mdn_mu_max_abs_delta=None, mdn_sigma_max_abs_delta=None)
        for key in ['prediction_max_abs_delta', 'mdn_weights_max_abs_delta', 'mdn_mu_max_abs_delta', 'mdn_sigma_max_abs_delta']:
            values = [c[key] for c in checks if key in c]
            item[key] = max(values) if values else None
        rows.append(item)
        if model.get('model_family') == 'team_original_scratch':
            path = args.training_root / name / 'protocol.lock.json'
            lock = read(path)
            require(digest(path) == selected['parent_lock_sha256'], 'Original TEAM lock differs')
            require(lock['model_family'] == 'team_original_scratch' and lock['pretrained_manifest_sha256'] is None, 'Scratch lock expectation differs')
            lock_checks.append(dict(run_id=name, original_lock_sha256=digest(path),
                                    inventory_lock_matches=True, model_family=lock['model_family'],
                                    pretrained_manifest_sha256=lock['pretrained_manifest_sha256']))

    ref = read(src / 'reference/train_only_reference.json')
    ref_provenance = read(src / 'reference/provenance.json')
    require(digest(src / 'reference/train_only_reference.json') == ref_provenance['reference_sha256'], 'Reference SHA differs')
    labels = table(src / 'reference/training_final_labels.csv.gz')
    require({x['split'] for x in labels} == {'train'}, 'Reference labels contain non-training split')
    label_keys = {(x['dataset_id'], x['event_id'], x['station_id']) for x in labels}
    require(len(label_keys) == len(labels) == ref['final_labels'], 'Reference label population differs/duplicates')
    require(len({x['event_id'] for x in labels}) == ref['train_events'], 'Reference event count differs')
    require(len({x['station_id'] for x in labels}) == ref['train_stations'], 'Reference station count differs')
    truths = [float(x['truth']) for x in labels]
    original_cfg = read(args.training_root / 'formal__team_original_scratch__seed42/resolved_config.json')
    require(digest(args.training_root / 'formal__team_original_scratch__seed42/resolved_config.json') == ref['source_config_sha256'], 'Reference original config differs')
    normalization = original_cfg['target_normalization']
    mean, std = statistics.mean(truths), statistics.pstdev(truths)
    require(abs(mean - normalization['mean']) < 1e-10 and abs(std - normalization['std']) < 1e-10, 'Reference normalization differs')
    exposure_identity = read(src / 'reference/training_exposure_identity.json')
    exposure = table(src / 'reference/actual_training_station_exposure.csv')
    metadata = table(src / 'requests/fixed_station_metadata.csv.gz')
    val_stations = {x['station_id'] for x in metadata}
    exposure_rows = []
    for status in exposure_identity:
        name = status['run_id']
        group = [x for x in exposure if x['run_id'] == name]
        if status['status'] == 'PASS':
            require(status['checkpoint_sha256'] == inventory[name]['checkpoint_sha256'], 'Exposure checkpoint differs')
            require(len(status['committed_journals_sha256']) == inventory[name]['committed_journals_count'], 'Journal count differs')
        input_stations = {x['station_id'] for x in group if x['station_seen_as_input'] == 'True'}
        query_stations = {x['station_id'] for x in group if x['station_seen_as_query'] == 'True'}
        exposure_rows.append(dict(run_id=name, status=status['status'], reason=status.get('reason', ''),
                                  journals=len(status.get('committed_journals_sha256', {})), train_events=status.get('unique_train_events'),
                                  stations=len(group) if group else None, input_stations=len(input_stations) if group else None,
                                  query_stations=len(query_stations) if group else None,
                                  validation_stations=len(val_stations),
                                  validation_seen_as_input=len(val_stations & input_stations) if group else None,
                                  validation_seen_as_query=len(val_stations & query_stations) if group else None))
    populations = []
    for scope in ['random', 'long', 'replay']:
        data = table(src / 'requests' / (scope + '_requests.csv.gz'))
        require({x['split'] for x in data} == {'val'}, 'Request split differs')
        keys = {(x['dataset_id'], x['event_id'], x['elapsed_time'], x['geometry_protocol'], x['input_mode'], x['station_id']) for x in data}
        require(len(keys) == len(data), 'Duplicate physical request')
        population = dict(scope=scope, rows=len(data), events=len({(x['dataset_id'], x['event_id']) for x in data}),
                          decisions=len({k[:-1] for k in keys}),
                          noninput_rows=sum(x['target_role'] in ('triggered_noninput', 'untriggered_noninput') for x in data),
                          observed_input_rows=sum(x['target_role'] == 'observed_input' for x in data),
                          triggered_noninput_rows=sum(x['target_role'] == 'triggered_noninput' for x in data),
                          untriggered_noninput_rows=sum(x['target_role'] == 'untriggered_noninput' for x in data),
                          minimum_elapsed_seconds=min(float(x['elapsed_time']) for x in data),
                          maximum_elapsed_seconds=max(float(x['elapsed_time']) for x in data))
        populations.append(population)
        require(digest(src / 'requests' / (scope + '_requests.csv.gz')) == read(src / 'requests/new_requests_identity.json')['request_sha256'][scope], 'New request SHA differs')

    # Replace private roots; preserve model names, numbers and original source hashes.
    code = next(x.removeprefix('--chdir=') for x in jobs['jobs'][0]['command'] if x.startswith('--chdir='))
    train = str(Path(next(m['run_dir'] for m in frozen if m['kind'] == 'fe01')).parent)
    replacements = {code: '${FE01_CODE_ROOT}', train: '${FE01_TRAIN_OUTPUT_ROOT}',
                    str(Path(original_cfg['pretrained_manifest']).parent): '${FE01_WEIGHTS_ROOT}',
                    original_cfg['data']['root']: '${DATA_ROOT}'}
    for model in frozen:
        if model.get('encoder'):
            replacements[model['encoder']] = '${LEGACY_ENCODER_CHECKPOINT}'

    def portable(text):
        for old, new in sorted(replacements.items(), key=lambda pair: -len(pair[0])):
            text = text.replace(old, new)
        text = re.sub(r'/public/home/[^/\s"\x27]+', '${HPC_HOME}', text)
        require('/public/home/' not in text, 'Unredacted private home path')
        return text

    out.mkdir(parents=True)
    export_map = {}
    for path in source_files:
        rel = path.relative_to(src)
        if rel.parts[0] == 'cache' or path.name == 'artifact_manifest.sha256':
            continue
        if path.suffix == '.json' or rel.parts[0] == 'logs' or path.name == 'actual_training_station_exposure.csv':
            target = Path('evidence') / (Path('scheduler_logs') / rel.name if rel.parts[0] == 'logs' else rel)
            destination = out / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(portable(path.read_text()))
            export_map[str(rel)] = str(target)
    for check in lock_checks:
        name = check['run_id']
        write_json(out / 'evidence/training_locks' / name / 'lock_fields.json', check)
    write_csv(out / 'model_gates.csv', rows)
    write_csv(out / 'training_exposure_summary.csv', exposure_rows)
    write_csv(out / 'request_population_summary.csv', populations)
    write_csv(out / 'source_inventory.csv', [dict(path=str(p.relative_to(src)), bytes=p.stat().st_size,
                                               sha256=source_hashes[str(p.relative_to(src))],
                                               exported_as=export_map.get(str(p.relative_to(src)), ''),
                                               transformation='private roots redacted' if str(p.relative_to(src)) in export_map else 'retained locally') for p in source_files])
    modules, runtime = [], []
    for p in sorted((src / 'logs').glob('*.out')):
        lines = p.read_text().splitlines()
        modules += [s.split(': ', 1)[1] for s in lines if s.startswith('FE01_MODULES_READY: ')]
        runtime += [dict(log=p.name, **json.loads(s.split(': ', 1)[1])) for s in lines if s.startswith('FE01_RUNTIME_READY: ')]
    summary = dict(task_id='20261004-fe01-eval1-existing-model-comparison', report_date='2026-10-06',
                   repo='rular099/team_pytorch', branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
                   analysis_base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   hpc_evaluation_source_sha=read(src / 'identity/provenance.json')['evaluation_source_sha'],
                   evaluation_module_sha=expected_module, training_code_sha=current_training['code_sha256'],
                   training_source_files=len(current_training['files_sha256']), generator_sha256=digest(Path(__file__)),
                   source=dict(folder_name=src.name, files=len(source_files), bytes=sum(p.stat().st_size for p in source_files), archive=archive),
                   split='val', unit='log10(m/s^2)', artifact_status=dict(identity='ARTIFACTS_VERIFIED', request_planning='ARTIFACTS_VERIFIED', reference='ARTIFACTS_VERIFIED'),
                   model_counts=dict(identity_pass=len(rows), forward_pass=sum(x['forward_status'] == 'PASS' for x in rows), forward_blocked=sum(x['forward_status'] == 'BLOCKED' for x in rows)),
                   models=rows, jobs=[{k: j.get(k) for k in ['stage', 'job_id', 'execution_status', 'returncode', 'model_ids', 'run_ids']} for j in jobs['jobs']],
                   scheduler_state='UNKNOWN: no sacct artifact provided; SUBMITTED is submission receipt only',
                   modules_ready=sorted(set(modules)), runtime=runtime,
                   reference={k: v for k, v in ref.items() if k not in ('train_site_effects', 'train_event_effects')},
                   independently_checked_labels=dict(rows=len(labels), events=ref['train_events'], stations=ref['train_stations'], mean=mean, population_std=std, fit_split='train'),
                   validation_station_coverage=dict(unique_stations=len(val_stations), seen_in_reference_train=len(val_stations & set(ref['train_site_effects'])), unseen_in_reference_train=len(val_stations - set(ref['train_site_effects']))),
                   team_blocker=dict(observed_reason='Original pretrained manifest byte SHA changed',
                                     original_locks=lock_checks,
                                     diagnosis='Unconditional pretrained manifest SHA check compares a SHA256 string to null for scratch TEAM; it does not establish changed bytes or failed forward equivalence.',
                                     source_locations=['fe01_review/runners.py:124', 'fe01/engine.py:359', 'fe01/model.py:130'],
                                     correction_status='PROPOSED_ONLY: no runner or old lock modified'),
                   request_populations=populations, training_exposure=exposure_rows,
                   next_stages={stage: 'NOT_SUBMITTED_IN_PROVIDED_MANIFEST' for stage in ['legacy_fixed', 'random', 'long', 'replay', 'analyze', 'pack']},
                   limits=['Gate probes are not full-population performance scores.', 'Legacy L0 checks same tensors, not historical loader recreation.',
                           'Only 4 decisions per PASS model run extended query/cache checks; only 2 run future perturbations.',
                           'Train-only reference uses retrospective catalog metadata, not a realtime competitor.',
                           'Journal hashes are recorded by HPC; raw journals are not in this download and are not independently rehashed here.',
                           'HDF5 hashes are recorded/reused from saved audit at matched bytes/mtime, not locally rehashed.',
                           'Upstream offline causality remains uncertified; no new training or held-out test was run.'])
    write_json(out / 'summary.json', summary)
    write_json(out / 'integrity_checks.json', dict(status='PASS', source_archive=archive, original_artifact_manifests=seals,
                                                original_sealed_entries=sum(x['entries'] for x in seals), current_source_checks=source_checks,
                                                training_source_files=len(current_training['files_sha256']),
                                                evaluation_source_files=len(current_evaluation['files_sha256']),
                                                checkpoint_identity_and_probe_pins='PASS', training_label_population_and_normalization='PASS',
                                                original_team_lock_byte_pins='PASS', raw_source_modified=False))
    require(all(digest(src / name) == value for name, value in source_hashes.items()), 'Source changed while summarizing')
    print(json.dumps(dict(output=str(out), models=summary['model_counts'], seals=len(seals),
                          sealed_entries=sum(x['entries'] for x in seals), archive_matches=archive['folder_matches'] if archive else 'NOT_CHECKED'), ensure_ascii=False))


if __name__ == '__main__':
    main()
