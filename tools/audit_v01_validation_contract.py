#!/usr/bin/env python3
"""Read existing V01 artifacts, trace real idx7, and gate validation-only work.

No cache materialization, model training, checkpoint writes, or test access.
The prepare subcommand uses only the standard library on login nodes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CELLS = ('vfull__vfull__normal', 'vfull__vmissing__normal',
         'vmissing__vfull__normal', 'vmissing__vmissing__normal', 'apair__apair__normal')
AA_RANDOM_REASON = ('15 duplicate event/time groups contain 45 extra rows; all 15 '
                    'have nonidentical predictions. Offline deduplication is not valid.')


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    from tools.v01_validation_contract import write_json as write
    write(path, data)


def prepare(args):
    # Deliberately do not import numpy/torch, even indirectly, on login nodes.
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Refusing existing closure directory: ' + str(output))
    manifest = json.loads((REPO/'tools/v01_closure_source_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        actual = file_sha(REPO/name)
        if actual != expected:
            raise ValueError('Uploaded source mismatch: ' + name)
    cells = list(CELLS) + (['apair__apair__random'] if args.include_aa else [])
    prepared = {}
    for cell in cells:
        source = args.run_root/'eval_retry1'/(cell + '.config.json')
        config = json.loads(source.read_text())
        if config.get('v01_validation_protocol') != cell.split('__')[-1]:
            raise ValueError('Wrong original protocol: ' + str(source))
        training = config['training_params']
        weight = Path(training['weight_path'])/'full_model_last.pth'
        if not weight.is_file():
            raise FileNotFoundError(weight)
        if training.get('v01_validation_closure'):
            raise ValueError('Original config already has closure changes')
        training['v01_validation_closure'] = True
        # Separate lightweight loader CSVs; never overwrite historical caches.
        training['metadata_cache_dir'] = str(output/'metadata_cache'/cell)
        config['v01_evaluation_protocol_version'] = 'v01-validation-closure-v1'
        prepared[cell] = (config, source, weight)
    output.mkdir(parents=True)
    for directory in ('configs', 'logs', 'eval', 'audit', 'source_evidence'):
        (output/directory).mkdir()
    for cell, (config, source, weight) in prepared.items():
        (output/'configs'/(cell + '.json')).write_text(json.dumps(config, indent=2) + '\n')
        shutil.copy2(source, output/'source_evidence'/source.name)
    shutil.copy2(REPO/'tools/v01_closure_source_manifest.json', output/'source_manifest.json')
    record = {'protocol': 'v01-validation-closure-v1', 'cells': cells,
              'run_root': str(args.run_root.resolve()), 'closure_root': str(output),
              'aa_random_reason': AA_RANDOM_REASON if args.include_aa else None,
              'source_manifest_sha256': file_sha(output/'source_manifest.json'),
              'old_outputs_untouched': True,
              'historical_runtime_identity': 'NOT_AVAILABLE unless verified in copied original evidence',
              'effective_config_sha256': {cell: file_sha(output/'configs'/(cell+'.json')) for cell in cells}}
    (output/'submission_plan.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


def verify_sources(output):
    manifest = json.loads((output/'source_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if file_sha(REPO/name) != expected:
            raise ValueError('Runtime source drift since submission: ' + name)
    plan = json.loads((output/'submission_plan.json').read_text())
    if file_sha(output/'source_manifest.json') != plan['source_manifest_sha256']:
        raise ValueError('Submitted source manifest changed')
    for cell, expected in plan['effective_config_sha256'].items():
        if file_sha(output/'configs'/(cell+'.json')) != expected:
            raise ValueError('Resolved closure config changed: ' + cell)


def checkpoint_contract(run_root):
    import gc
    import torch
    from tools.v01_validation_contract import CHECKPOINTS
    result = []
    for arm, (directory, expected) in CHECKPOINTS.items():
        path = run_root/directory/'full_model_last.pth'
        actual = file_sha(path)
        if actual != expected:
            raise ValueError('Checkpoint hash changed: ' + str(path))
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        init_sha = file_sha(run_root/directory/'full_model_init.pth')
        if init_sha != '6e91102989e1d6440dd878c6339001c1000e9a737e1071a7c6255bbe54d26577':
            raise ValueError('Original initialization checkpoint changed: '+arm)
        epoch = int(checkpoint['epoch'])
        states = checkpoint['optimizer_state_dict']['state'].values()
        steps = [float(s['step']) for s in states if 'step' in s]
        if epoch != 8 or not steps or min(steps) != 1496 or max(steps) != 1496:
            raise ValueError('Fixed epoch8/step1496 contract failed: ' + arm)
        result.append({'arm': arm, 'file': str(path), 'sha256': actual, 'init_sha256': init_sha,
                       'epoch': epoch, 'step_min': min(steps), 'step_max': max(steps),
                       'saved_encoder_source': checkpoint.get('encoder_source'),
                       'historical_encoder_file_sha256': 'NOT_AVAILABLE'})
        del checkpoint, states
        gc.collect()
    return result


def child_request(dataset, index):
    if hasattr(dataset, 'generators'):
        gid, local = dataset.indexes[index]
        return dataset.generators[gid], local
    return dataset, index


def signature(sample):
    import numpy as np
    digest = hashlib.sha256()
    # All model inputs/labels and all old info fields. New identity is non-numeric.
    for value in list(sample[0]) + list(sample[1]) + [sample[2][k] for k in sorted(sample[2])
                                                    if k not in ('v01_query_sensor_id', 'v01_source_sensor_id',
                                                                 'v01_source_paired_acc_id', 'v01_absolute_cutoff_utc',
                                                                 'v01_cache_plan_hash', 'v01_requested_event_id',
                                                                 'v01_protocol', 'v01_crop_start')]:
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        digest.update(str((array.dtype, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def trace_idx7(dataset):
    import h5py
    import numpy as np
    from gemini_util_light import _select_wave_idx_rows
    from tools.v01_validation_contract import cache_identity, json_value
    generator = dataset.generators[0] if hasattr(dataset, 'generators') else dataset
    index = next(i for i in range(len(generator))
                 if generator.describe_request(i)['base_event_index'] == 7)
    request = generator.describe_request(index)
    with h5py.File(generator.data_path, 'r') as h5:
        group = h5['data'][request['event_id']]
        metadata = generator.event_metadata.get_group(generator.event_keys[7])
        selector = _select_wave_idx_rows(metadata, group)
        rows = np.arange(group['waveforms'].shape[0])[selector]
        identity = cache_identity(group, rows)
        physical = {key: json_value(identity[key]) for key in
                    ('station_codes', 'v01_source_sensor_id', 'v01_paired_acc_sensor_id',
                     'coords', 'pga', 'p_picks', 'v01_source_role', 'query_valid',
                     'absolute_window_start_timestamp', 'plan_hash')}
    original = generator._select_realtime_cutout
    original_target_sampler = generator._sample_realtime_pga_targets
    captured = []
    target_selections = []
    def traced(picks, mask, rng, context, length):
        before = mask.copy()
        clock = original(picks, mask, rng, context, length)
        captured.append({'p_samples': picks.tolist(), 'mask_before': before.tolist(),
                         'mask_after': mask.tolist(), 'changed_slots': int(np.sum(before != mask)),
                         'clock': clock})
        return clock
    def traced_targets(*args, **kwargs):
        selection = original_target_sampler(*args, **kwargs)
        target_selections.append({'selected_query_count': int(len(selection[0])),
                                  'sampler_slots_not_sensor_ids': selection[0].tolist()})
        return selection
    generator._select_realtime_cutout = traced
    generator._sample_realtime_pga_targets = traced_targets
    result = {'request': request, 'row_selector': rows.tolist(), 'physical_rows': physical}
    try:
        generator.v01_validation_closure = False
        try:
            generator._get_one(index)
            result['before_error'] = None
        except Exception as exc:
            result['before_error'] = repr(exc)
        result['before_clock'] = list(captured)
        result['before_target_selection'] = list(target_selections)
        captured.clear()
        target_selections.clear()
        generator.v01_validation_closure = True
        from tools.v01_validation_contract import V01NoPrediction
        try:
            sample = generator[index]
            result['after'] = {'outcome': 'predicted', 'actual_event_id': sample[2]['event_id'],
                               'input_count': int(sample[0][2].sum()),
                               'query_count': int(sample[0][4].sum()),
                               'identity': json_value({k: v for k,v in sample[2].items() if k.startswith('v01_')})}
        except V01NoPrediction as exc:
            if exc.outcome != 'no_available_source':
                raise
            # Source abstention is legitimate, but only if production sampling
            # actually restored finite queries. Do not require a fabricated prediction.
            selected = max((x['selected_query_count'] for x in target_selections), default=0)
            result['after'] = {'outcome': exc.outcome, 'reason': exc.reason,
                               'actual_event_id': None, 'input_count': 0,
                               'query_count': selected, 'request_not_substituted': True}
        result['after_clock'] = list(captured)
        result['after_target_selection'] = list(target_selections)
        result['confirmed'] = (
            'Found event without PGA' in str(result['before_error'])
            and len(result['before_clock']) == len(result['after_clock']) == 1
            and result['before_clock'][0]['changed_slots'] > 0
            and result['after_clock'][0]['changed_slots'] == 0
            and result['before_clock'][0]['clock'] == result['after_clock'][0]['clock']
            and result['after']['query_count'] > 0)
    except Exception as exc:
        result['confirmed'] = False
        result['after_error'] = repr(exc)
    finally:
        generator._select_realtime_cutout = original
        generator._sample_realtime_pga_targets = original_target_sampler
        generator.v01_validation_closure = True
    return result


def npz_arrays(path):
    import numpy as np
    with np.load(path, allow_pickle=True) as data:
        if any(k.startswith('test_') for k in data.files):
            raise ValueError('Test NPZ is forbidden')
        return {k[4:]: np.asarray(data[k].tolist()) for k in data.files if k.startswith('val_')}


def audit_aa_duplicates(path):
    import numpy as np
    arrays = npz_arrays(path)
    groups = {}
    for row, key in enumerate(zip(arrays['event_id'], arrays['realtime_requested_elapsed_time'])):
        groups.setdefault(tuple(key), []).append(row)
    records = []
    for (event, time), rows in groups.items():
        if len(rows) < 2:
            continue
        different = []
        for key, value in arrays.items():
            if key == 'event_index':
                continue  # this is the requested index, not actual event identity
            equal = all(np.array_equal(value[rows[0]], value[i], equal_nan=True)
                        if value.dtype.kind in 'fci' else np.array_equal(value[rows[0]], value[i])
                        for i in rows[1:])
            if not equal:
                different.append(key)
        records.append({'event_id': str(event), 'time_s': float(time), 'rows': rows,
                        'requested_indices': arrays['event_index'][rows].tolist(),
                        'nonidentical_fields': different})
    return {'file': str(path), 'sha256': file_sha(path), 'duplicate_groups': records,
            'extra_rows': sum(len(r['rows'])-1 for r in records),
            'can_deduplicate': bool(records) and not any(r['nonidentical_fields'] for r in records),
            'decision': 'rerun_one_AA_random' if any(r['nonidentical_fields'] for r in records)
                        else 'needs_input_identity_and_missing_request_verification_before_dedup'}


def plan_and_sidecar(dataset, old_npz, output, name):
    """Align original requests and exported rows to real cache identities.

    Only metadata is read here. No waveform bytes and no test metrics are read.
    Ambiguous coordinate-to-ID matches abort; coordinates are not treated as IDs.
    """
    import h5py
    import numpy as np
    from gemini_util_light import _select_wave_idx_rows
    from tools.v01_validation_contract import cache_identity, json_value
    arrays = npz_arrays(old_npz) if old_npz is not None else None
    by_request = {} if arrays is None else {int(x): row for row,x in enumerate(arrays['event_index'])}
    requested_keys = {}
    records = []
    cached = {}
    for index in range(len(dataset)):
        generator, local = child_request(dataset, index)
        request = dataset.describe_request(index)
        request['global_index'] = index
        event = request['event_id']
        key = (str(generator.data_path), event)
        if key not in cached:
            with h5py.File(generator.data_path, 'r') as h5:
                group = h5['data'][event]
                event_group = generator.event_metadata.get_group(generator.event_keys[request['base_event_index']])
                selector = _select_wave_idx_rows(event_group, group)
                rows = np.arange(group['waveforms'].shape[0])[selector]
                identity = cache_identity(group, rows)
                if group['waveforms'].shape[1] != generator.trace_length or generator.decimate != 1:
                    raise ValueError('Unverified legacy crop/decimation: identity sidecar needs explicit trace')
                identity['reference'] = int(group['v01_reference_p_pick'][0])
                identity['length'] = int(group['waveforms'].shape[1])
                raw = identity['coords'][None].copy()
                identity['export_coords'] = generator.location_transformation(
                    raw, station_valid=np.ones(raw.shape[:2], bool))[0].astype(np.float32)
            cached[key] = identity
        identity = cached[key]
        current = int(np.clip(identity['reference'] + round(request['time_s']*generator.sampling_rate),
                              0, identity['length']-1))
        source = identity['v01_source_role'].astype(bool)
        query = identity['query_valid']
        request.update(absolute_cutoff_utc=float(identity['absolute_window_start_timestamp'])
                       + current/generator.sampling_rate, cache_plan_hash=identity['plan_hash'],
                       requested_query_sensor_ids=identity['station_codes'][query].tolist(),
                       requested_source_sensor_ids=identity['station_codes'][source].tolist(),
                       paired_acc_sensor_ids=identity['v01_paired_acc_sensor_id'][source].tolist(),
                       expected_query_count=int(query.sum()), original_rows=identity['original_rows'].tolist())
        actual_key = (event, request['time_s'])
        if actual_key in requested_keys:
            raise ValueError('Frozen request plan contains duplicate event/time: ' + str(actual_key))
        requested_keys[actual_key] = index
        record = dict(request)
        if arrays is not None:
            if index not in by_request:
                raise ValueError('Old full-evaluation request index missing: ' + str(index))
            row = by_request[index]
            record.update(actual_event_id=str(arrays['event_id'][row]),
                          actual_time_s=float(arrays['realtime_requested_elapsed_time'][row]))
            if (record['actual_event_id'], record['actual_time_s']) != actual_key:
                record['outcome'] = 'legacy_neighbour_substitution'
            else:
                if int(arrays['realtime_current_sample'][row]) != current:
                    raise ValueError('Legacy absolute cutoff changed')
                valid = arrays['pga_target_valid'][row].astype(bool)
                query_ids = [''] * len(valid)
                for slot in np.flatnonzero(valid):
                    coords = arrays['pga_target_abs'][row, slot]
                    match = np.flatnonzero(query & (identity['export_coords'] == coords).all(axis=1))
                    if len(match) != 1:
                        raise ValueError('Cannot uniquely certify physical sensor ID from cache/NPZ')
                    pos = int(match[0])
                    label = float(np.asarray(arrays['pga_label'][row,slot]).reshape(-1)[0])
                    if np.float32(identity['pga'][pos]) != np.float32(label):
                        raise ValueError('Cache query label differs from saved NPZ')
                    query_ids[slot] = str(identity['station_codes'][pos])
                original_rows = arrays['selected_original_input_indices'][row].astype(int)
                source_valid = arrays['station_valid'][row].astype(bool)
                src_ids = [''] * len(source_valid); paired_ids = src_ids.copy()
                for slot in np.flatnonzero(source_valid):
                    found = np.flatnonzero(identity['original_rows'] == original_rows[slot])
                    if len(found) != 1 or not source[found[0]]:
                        raise ValueError('Selected input identity cannot be certified')
                    src_ids[slot] = str(identity['station_codes'][found[0]])
                    paired_ids[slot] = str(identity['v01_paired_acc_sensor_id'][found[0]])
                record.update(outcome='predicted', query_sensor_ids=query_ids,
                              source_sensor_ids=src_ids, source_paired_acc_ids=paired_ids,
                              selected_valid_query_count=int(valid.sum()))
        else:
            record['outcome'] = 'requested_not_yet_evaluated'
        records.append(record)
    if arrays is not None and len(by_request) != len(records):
        raise ValueError('Old NPZ has unplanned request indices')
    # Full plan + identity evidence: retained even for legacy substitutions.
    with (output/(name + '.plan.jsonl')).open('x', encoding='utf8') as stream:
        for record in records:
            stream.write(json.dumps(json_value(record), ensure_ascii=False) + '\n')
    return records


def audit(args):
    import h5py
    import numpy as np
    from eval_checkpoint import build_datasets
    from tools.v01_validation_contract import json_value
    output = args.output/'audit'
    verify_sources(args.output)
    write_json(output/'checkpoints_before.json', checkpoint_contract(args.run_root))
    evidence = args.output/'source_evidence'
    availability = []
    candidates = list((args.run_root/'derived_cache').glob('*.json')) + list((args.run_root/'derived_cache').glob('*.csv'))
    for directory in ('weights_vfull_retry1', 'weights_vmissing', 'weights_apair'):
        candidates += [args.run_root/directory/k for k in ('config.json', 'manifest.json', 'split_events.csv', 'split_stations.csv')]
    for source in candidates:
        if source.is_file():
            relative = source.relative_to(args.run_root)
            target = evidence/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            availability.append({'file': str(relative), 'sha256': file_sha(source)})
    for name in ('protocol_lock.json', 'preflight_summary.json', 'cohort_counts_by_year.csv', 'cohort_audit.csv'):
        source = args.run_root/'derived_cache'/name
        if not source.is_file():
            availability.append({'file': 'derived_cache/' + name, 'status': 'NOT_AVAILABLE'})
    configs = {cell: json.loads((args.output/'configs'/(cell + '.json')).read_text()) for cell in CELLS}
    first = next(iter(configs.values()))['training_params']
    for field in ('frozen_split_manifest', 'load_model_path'):
        source = Path(first[field])
        availability.append({'field': field, 'file': str(source),
                             'current_sha256': file_sha(source) if source.is_file() else 'NOT_AVAILABLE',
                             'historical_sha256': 'NOT_AVAILABLE unless recorded in original manifest'})
        if field == 'frozen_split_manifest':
            if not source.is_file():
                raise FileNotFoundError('Original frozen split manifest is mandatory')
            shutil.copy2(source, evidence/'frozen_split_events.csv')
    encoder = os.environ.get('DITING_PRETRAINED')
    availability.append({'field': 'runtime_encoder', 'file': encoder,
                         'current_sha256': file_sha(encoder) if encoder and Path(encoder).is_file() else 'NOT_AVAILABLE',
                         'historical_sha256': 'NOT_AVAILABLE'})
    # Reuse the original encoder path, but do not misrepresent today's file
    # hash as a recorded historical hash.
    before = json.loads((output/'checkpoints_before.json').read_text())
    if not encoder or not Path(encoder).is_file():
        raise FileNotFoundError('Existing runtime encoder required')
    for checkpoint in before:
        saved = checkpoint['saved_encoder_source']
        if not saved or not Path(saved).is_file() or not os.path.samefile(saved, encoder):
            raise ValueError('Do not change the saved encoder source for validation closure')
    availability.append({'field': 'runtime_diting_config', 'file': os.environ.get('DITING_CONFIG'),
                         'current_sha256': file_sha(os.environ['DITING_CONFIG']), 'historical_sha256': 'NOT_AVAILABLE'})
    write_json(output/'source_provenance.json', availability)
    counts = {}
    for config in configs.values():
        for path in config['training_params']['data_path']:
            if path in counts:
                continue
            split_counts = {}
            with h5py.File(path, 'r') as h5:
                for event, group in h5['data'].items():
                    split = json_value(group.attrs.get('split', 'NOT_AVAILABLE'))
                    if split not in ('train', 'dev'):
                        raise ValueError('Unexpected/test event in derived cache: ' + event)
                    record = split_counts.setdefault(split, {'events': 0, 'source_rows': 0, 'query_rows': 0})
                    role = np.asarray(group['v01_source_role']).astype(bool)
                    record['events'] += 1
                    record['source_rows'] += int(role.sum())
                    record['query_rows'] += int((~role).sum())
            counts[path] = split_counts
    write_json(output/'cache_train_dev_counts.json', {'counts': counts,
               'definition': 'materialized cache rows before loader filters, not effective training samples'})
    traces = {}
    loader_counts = []
    for cell in ('vfull__vfull__normal', 'vfull__vmissing__normal', 'apair__apair__normal'):
        dataset = build_datasets(configs[cell], splits=['val'])['val']
        trace = trace_idx7(dataset)
        traces[cell] = trace
        write_json(output/(cell + '.idx7_trace.json'), trace)
        if not trace['confirmed']:
            raise RuntimeError('Real idx7 alias hypothesis not confirmed; do not launch evaluations: ' + cell)
        plan_and_sidecar(dataset, None, output, cell)
        training_config = json.loads(json.dumps(configs[cell]))
        training_config['training_params']['v01_validation_closure'] = False
        train_dataset = build_datasets(training_config, splits=['train'])['train']
        for split, obj in (('train', train_dataset), ('dev', dataset)):
            children = obj.generators if hasattr(obj, 'generators') else [obj]
            loader_counts.append({'view': cell.split('__')[1], 'split': split,
                                  'eligible_events': sum(len(g.event_keys) for g in children),
                                  'selected_station_metadata_rows': sum(len(g.event_metadata.obj) for g in children),
                                  'meaning': 'real loader filters, before training resampling/augmentation'})
    write_json(output/'actual_loader_train_dev_counts.json', loader_counts)
    random_signatures = []
    reused_artifacts = []
    for cell in ('vfull__vfull__random', 'vfull__vmissing__random', 'vmissing__vfull__random', 'vmissing__vmissing__random'):
        config = json.loads((args.run_root/'eval_retry1'/(cell + '.config.json')).read_text())
        config['training_params']['v01_validation_closure'] = True
        config['training_params']['metadata_cache_dir'] = str(args.output/'metadata_cache'/cell)
        dataset = build_datasets(config, splits=['val'])['val']
        records = plan_and_sidecar(dataset, args.run_root/'eval_retry1'/(cell + '.npz'), output, cell)
        reused_artifacts.append({'cell': cell, 'file': str(args.run_root/'eval_retry1'/(cell+'.npz')),
                                 'sha256': file_sha(args.run_root/'eval_retry1'/(cell+'.npz'))})
        shutil.copy2(args.run_root/'eval_retry1'/(cell+'.config.json'), evidence/(cell+'.config.json'))
        if any(r['outcome'] != 'predicted' for r in records):
            raise RuntimeError('Existing velocity random contains substituted requests; cannot reuse')
        # Bounded real-cache signatures: idx7 at all seven times in shard0,
        # plus the first request of every shard. No encoder/model forward.
        for index in range(len(dataset)):
            generator, local = child_request(dataset, index)
            request = generator.describe_request(local)
            shard = request.get('shard_id', dataset.describe_request(index).get('shard_id', 0))
            selected = ((shard == 0 and request['base_event_index'] == 7)
                        or (request['base_event_index'] == 0
                            and request['time_s'] == generator.realtime_training['val_times'][0]))
            if not selected:
                continue
            generator.v01_validation_closure = False
            before = signature(generator._get_one(local))
            generator.v01_validation_closure = True
            after = signature(generator._get_one(local))
            if before != after:
                raise RuntimeError('Existing random numerics changed: stop without four-cell rerun')
            random_signatures.append({'cell': cell, 'request_index': index, 'sha256': before,
                                      'model_inputs_labels_old_info_equal': True})
    write_json(output/'random_numeric_signatures.json', random_signatures)
    write_json(output/'reused_random_artifacts.json', reused_artifacts)
    aa = audit_aa_duplicates(args.run_root/'eval_retry1/apair__apair__random.npz')
    config = json.loads((args.run_root/'eval_retry1/apair__apair__random.config.json').read_text())
    config['training_params']['v01_validation_closure'] = True
    config['training_params']['metadata_cache_dir'] = str(args.output/'metadata_cache/aa_identity')
    dataset = build_datasets(config, splits=['val'])['val']
    records = plan_and_sidecar(dataset, args.run_root/'eval_retry1/apair__apair__random.npz', output, 'apair__apair__random_legacy')
    substitutions = []
    from tools.v01_validation_contract import V01NoPrediction
    for record in records:
        if record['outcome'] != 'legacy_neighbour_substitution':
            continue
        try:
            dataset[record['global_index']]
            record['strict_original_outcome'] = 'predictable_under_closure'
        except V01NoPrediction as exc:
            record['strict_original_outcome'] = exc.outcome
            record['strict_reason'] = exc.reason
        substitutions.append(record)
    aa['substituted_requests'] = substitutions
    write_json(output/'aa_duplicate_audit.json', aa)
    write_json(output/'gate_passed.json', {'protocol': 'v01-validation-closure-v1',
               'real_idx7_confirmed': True, 'random_numeric_signatures_equal': True,
               'aa_random_requires_rerun': aa['decision'] == 'rerun_one_AA_random',
               'checkpoints_unchanged': True})
    print('PASS: real idx7 alias confirmed; old random inputs/labels unchanged; evaluation gate opened.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'audit', 'verify', 'audit-aa'])
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-aa', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args)
    elif args.action == 'audit':
        audit(args)
    elif args.action == 'audit-aa':
        args.output.mkdir(parents=True, exist_ok=False)
        record = audit_aa_duplicates(args.run_root/'eval_retry1/apair__apair__random.npz')
        record.update(task_id='20261004-v01-validation-closure',
                      base_commit='74c55aa5442b4200961c88ceee3d11af5033275d',
                      real_cache_trace_status='NOT_RUN: derived cache exists only on HPC',
                      sensor_identity_certification='NOT_AVAILABLE locally; must run audit gate on HPC',
                      offline_dedup_performed=False)
        write_json(args.output/'aa_offline_duplicate_audit.json', record)
        print(json.dumps(record, indent=2))
    else:
        verify_sources(args.output)
        if not (args.output/'audit/gate_passed.json').is_file():
            raise RuntimeError('Audit gate has not passed')
        print('PASS: audited source identity unchanged')


if __name__ == '__main__':
    main()
