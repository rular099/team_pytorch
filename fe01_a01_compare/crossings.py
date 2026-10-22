"""Offline truth only: first crossing of the label-consistent NEZ vector norm."""
import numpy as np
import pandas as pd
from .io import csv_write, write_json
from .covariates import physical_event


def first_crossing(wave, mask, start, length, fs, threshold, label, clipping=None):
    wave = np.asarray(wave, float)
    support = np.asarray(mask, bool)
    if support.ndim == 2:
        support = support.all(0)
    stop = min(len(support), int(start+length))
    indices = np.arange(len(support))
    valid = support & (indices >= start) & (indices < stop) & np.isfinite(wave).all(0)
    amplitude = np.linalg.norm(np.where(valid[None, :], wave, 0), axis=0)
    base = dict(status='unknown_crossing', crossing_sample=None, crossing_record_seconds=None,
                left_censored=False, clipping_status='known_clipping' if clipping else 'not_detected' if clipping is False else 'UNKNOWN',
                label_log10=float(label), retrospective_only=True, available_at_decision=False)
    if not valid.any():
        return dict(base, reason='missing_waveform')
    peak = float(amplitude[valid].max())
    y = float(np.log10(peak+1e-10))
    base.update(recomputed_log10_peak=y, peak_label_abs_difference=abs(y-float(label)))
    if abs(y-float(label)) > 1e-5:
        return dict(base, reason='peak_label_definition_mismatch')
    if clipping:
        return dict(base, reason='clipping')
    if (float(label)>=threshold)!=(peak>=10**threshold):
        return dict(base,reason='label_threshold_boundary_disagreement')
    exceed = np.flatnonzero(valid & (amplitude >= 10**threshold))
    if not len(exceed):
        return dict(base, status='not_exceeded' if valid[start:stop].all() else 'unknown_crossing',
                    reason='' if valid[start:stop].all() else 'missing_support_could_hide_crossing')
    first = int(exceed[0])
    if first == start:
        return dict(base, status='left_censored', reason='record_starts_above_threshold', left_censored=True,
                    crossing_upper_sample=first)
    if not valid[start:first].all():
        return dict(base, reason='gap_before_first_observed_crossing', crossing_upper_sample=first)
    return dict(base, status='exact_within_record', reason='', crossing_sample=first,
                crossing_record_seconds=first/fs,
                boundary='first within verified stored record; pre-record physical history unobserved')


def extract(rows, thresholds, out):
    result, events = [], []
    for _, row in rows.iterrows():
        e = physical_event(row, full=True)
        for i in e['queries']:
            for threshold in thresholds:
                r = first_crossing(e['waveform'][i], e['storage'][i], int(e['starts'][i]), int(e['lengths'][i]),
                                   e['fs'], threshold, float(e['pga'][i]),
                                   bool(e['clipping'][i]) if e['clipping_known'] else None)
                crossing = r.get('crossing_sample')
                result.append(dict(dataset_id=e['dataset_id'], event_id=e['event_id'], station_id=str(e['ids'][i]),
                    threshold_logpga=threshold, truth=float(e['pga'][i]), affected=float(e['pga'][i]) >= threshold,
                    crossing_first_p_seconds=(crossing-e['reference_sample'])/e['fs'] if crossing is not None else None,
                    crossing_utc=e['record_start_utc']+crossing/e['fs'] if crossing is not None and e['record_start_utc'] is not None else None,
                    clock_status=e['clock']['status'], **r))
        events.append(dict(dataset_id=e['dataset_id'], event_id=e['event_id'], source_attributes=e['source_attributes']))
    f = pd.DataFrame(result)
    csv_write(f, out/'crossing_truth.csv.gz')
    csv_write(f.groupby(['threshold_logpga','status','reason'], dropna=False).size().rename('stations').reset_index(), out/'threshold_crossing_audit.csv')
    write_json(out/'label_definition_audit.json', dict(
        definition='peak sqrt(NS^2+EW^2+UD^2), stored resampled/demeaned acceleration, m/s^2; log10(peak+1e-10)',
        production_chain=['japan_dataset_builder.py:_compute_pga_stats,_resample_waveform,build_event',
                          'build_japan_flat_training_data.py: waveform demean after resample; stationheight(m)/1000'],
        filter='stored source upstream processing only; no new filter or demeaning in crossing truth',
        upstream='UPSTREAM_CAUSALITY_UNKNOWN', magnitude_scale='UNKNOWN_catalog_native', events=events,
        full_target_waveforms='offline scoring module only; never part of input selection or normalization'))
