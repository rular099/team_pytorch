"""V01-only validation identities and non-substituting request accounting."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROTOCOL = 'v01-validation-closure-v1'
OUTCOMES = ('predicted', 'no_available_source', 'invalid_label_or_metadata',
            'implementation_error')
CHECKPOINTS = {
    'vfull': ('weights_vfull_retry1', '09c65503ff6e7d848899081110142487c0622f1cc5fd143df3579d55e38f4763'),
    'vmissing': ('weights_vmissing', '9173d8d60d657b6b6eebb6a1dedef6c4af139319328c787c233187b2acde3b8a'),
    'apair': ('weights_apair', 'a2837d575e4a81e6386e2cb004669a3b0052e9ab093aa3893fcda7b76c74cb32'),
}


class V01NoPrediction(Exception):
    def __init__(self, outcome, reason):
        if outcome not in OUTCOMES[1:3]:
            raise ValueError(outcome)
        self.outcome, self.reason = outcome, reason
        super().__init__(reason)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return json_value(value.item())
    if isinstance(value, (list, tuple)):
        return [json_value(x) for x in value]
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, bytes):
        return value.decode('utf8')
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(json_value(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def read_reference_pick(group):
    """Read an event scalar, also accepting legacy single-element arrays.

    The production cache builder writes a zero-dimensional int64 dataset.
    HDF5 [()] reads both representations; [0] is illegal on a scalar.
    """
    values = np.asarray(group['v01_reference_p_pick'][()]).reshape(-1)
    if values.size != 1:
        raise ValueError('v01_reference_p_pick must be an event scalar')
    value = values[0]
    numeric = (np.issubdtype(values.dtype, np.integer)
               or np.issubdtype(values.dtype, np.floating))
    if not numeric or not np.isfinite(value) or value != int(value):
        raise ValueError('v01_reference_p_pick must be a finite integer sample')
    return int(value)


def cache_identity(group, rows=None):
    """Read metadata only. IDs come from cache rows, never sampler slots."""
    required = ('station_codes', 'v01_source_sensor_id', 'v01_paired_acc_sensor_id',
                'coords', 'pga', 'p_picks', 'v01_source_role')
    missing = [k for k in required if k not in group]
    if missing:
        raise ValueError('V01 identity datasets missing: ' + ','.join(missing))
    rows = np.arange(len(group['pga'])) if rows is None else np.asarray(rows)
    result = {k: np.asarray(group[k])[rows] for k in required}
    for key in required[:3]:
        result[key] = np.asarray([json_value(x) for x in result[key]], dtype=str)
    role = result['v01_source_role'].astype(bool)
    result['query_valid'] = (~role & np.isfinite(result['pga'])
                             & np.isfinite(result['coords']).all(axis=1))
    result['original_rows'] = rows.astype(int)
    for key in ('absolute_window_start_timestamp', 'plan_hash', 'split'):
        if key not in group.attrs:
            raise ValueError('Missing V01 cache attribute: ' + key)
        result[key] = json_value(group.attrs[key])
    if result['split'] not in ('train', 'dev'):
        raise ValueError('Held-out test cache cannot enter V01 closure')
    return result


def selected_identity(identity, request, input_rows, station_valid, target_row_map,
                      target_slots, target_valid, clock, crop_start, rate, decimate):
    source_ids = np.full(len(input_rows), '', dtype=object)
    paired_ids = source_ids.copy()
    ok = np.asarray(station_valid, bool)
    rows = np.asarray(input_rows)[ok]
    if np.any(rows < 0) or not identity['v01_source_role'][rows].all():
        raise ValueError('Query-only row entered valid input slots')
    source_ids[ok] = identity['station_codes'][rows]
    paired_ids[ok] = identity['v01_paired_acc_sensor_id'][rows]
    query_ids = np.full(len(target_slots), '', dtype=object)
    q_ok = np.asarray(target_valid, bool)
    query_rows = np.asarray(target_row_map)[np.asarray(target_slots)[q_ok]]
    if np.any(query_rows < 0) or not identity['query_valid'][query_rows].all():
        raise ValueError('Invalid query row mapping')
    query_ids[q_ok] = identity['station_codes'][query_rows]
    if len(set(query_ids[q_ok])) != int(q_ok.sum()):
        raise ValueError('Duplicate physical query IDs in a request')
    cutoff = (float(identity['absolute_window_start_timestamp'])
              + (int(crop_start) + int(clock['current_sample'])) / float(rate))
    # crop_start/current_sample are in decimated samples; absolute start is raw UTC.
    return {'v01_query_sensor_id': query_ids, 'v01_source_sensor_id': source_ids,
            'v01_source_paired_acc_id': paired_ids,
            'v01_absolute_cutoff_utc': cutoff,
            'v01_cache_plan_hash': str(identity['plan_hash']),
            'v01_requested_event_id': request['event_id'],
            'v01_protocol': PROTOCOL, 'v01_crop_start': int(crop_start)}


def run_closure_inference(model, dataset, device, config, indices, ledger_path, run_one):
    """Reuse the unchanged evaluator once per strict request. Errors abort."""
    results = defaultdict(list)
    counts = Counter()
    seen = set()
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    class SingleSample:
        def __getitem__(self, _index):
            return self.sample
    holder = SingleSample()
    with ledger_path.open('x', encoding='utf8') as ledger:
        for index in indices:
            request = dataset.describe_request(index)
            request['global_index'] = int(index)
            key = (request['cache_path'], request['event_id'], request['time_s'])
            if key in seen:
                raise ValueError('Duplicate requested cache/event/time')
            seen.add(key)
            record = {**request, 'protocol': PROTOCOL, 'actual_event_id': None}
            try:
                holder.sample = dataset[index]
                info = holder.sample[2]
                if str(info['event_id']) != request['event_id']:
                    raise ValueError('Actual event differs from requested event')
                if float(info['realtime_requested_elapsed_time']) != request['time_s']:
                    raise ValueError('Actual cutoff request differs from ledger')
                one = run_one(model, holder, device, config, indices=[index])
                for name, values in one.items():
                    results[name].extend(values)
                record.update(outcome='predicted', actual_event_id=str(info['event_id']),
                              reason=None, identity=json_value({k: v for k, v in info.items()
                                                               if k.startswith('v01_')}))
            except V01NoPrediction as exc:
                record.update(outcome=exc.outcome, reason=exc.reason)
            except Exception as exc:
                record.update(outcome='implementation_error', reason=repr(exc))
                counts[record['outcome']] += 1
                ledger.write(json.dumps(json_value(record), ensure_ascii=False) + '\n')
                ledger.flush()
                write_json(str(ledger_path) + '.summary.json',
                           {'protocol': PROTOCOL, 'complete': False, 'processed': sum(counts.values()),
                            'planned_requests': len(indices),
                            'pending_requests_not_executed': len(indices)-sum(counts.values()),
                            'outcomes': dict(counts), 'error': repr(exc)})
                raise
            counts[record['outcome']] += 1
            ledger.write(json.dumps(json_value(record), ensure_ascii=False) + '\n')
            ledger.flush()
    summary = {'protocol': PROTOCOL, 'complete': True, 'requested': len(seen),
               'outcomes': {key: counts[key] for key in OUTCOMES}, 'ledger': str(ledger_path)}
    if summary['requested'] != sum(summary['outcomes'].values()):
        raise AssertionError('Request accounting failed')
    write_json(str(ledger_path) + '.summary.json', summary)
    return dict(results), summary
