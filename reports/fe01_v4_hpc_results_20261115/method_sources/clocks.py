"""One delivery clock for TTUM and shadow; communication remains unmeasured."""
import numpy as np
from .io import require


def jst_utc(value):
    import pandas as pd
    t = pd.Timestamp(str(value))
    if t.tz is None:
        t = t.tz_localize('Asia/Tokyo')
    return float(t.tz_convert('UTC').timestamp())


def audit_clock(starts_jst, starts_sample, fs, origin=None, reference_sample=None):
    base = dict(status='unknown_clock', record_start_utc=None, origin_utc=None, first_p_utc=None,
                first_p_minus_origin_seconds=None, timezone='naive *_jst is Asia/Tokyo then UTC')
    try:
        times = np.asarray([jst_utc(v) for v in starts_jst])-np.asarray(starts_sample)/fs
        require(len(times) > 0 and np.ptp(times) <= 1/fs+1e-5, 'station UTC axes disagree')
        record = float(times[0])
        base.update(status='record_utc_verified', record_start_utc=record,
                    first_p_utc=record+reference_sample/fs if reference_sample is not None else None)
        if origin is not None and str(origin) not in ('nan', '', 'None'):
            o = jst_utc(origin)
            base.update(status='verified', origin_utc=o,
                        first_p_minus_origin_seconds=base['first_p_utc']-o if base['first_p_utc'] is not None else None)
    except (ValueError, TypeError, OverflowError) as e:
        base['reason'] = str(e)
    return base


def fifo(cutoffs, service_seconds, render_seconds=None):
    require(len(cutoffs) == len(service_seconds), 'Queue length mismatch')
    render_seconds = render_seconds if render_seconds is not None else [0.]*len(cutoffs)
    last = -np.inf
    out = []
    for cutoff, service, render in zip(cutoffs, service_seconds, render_seconds):
        require(np.isfinite([cutoff, service, render]).all() and service >= 0 and render >= 0, 'Invalid queue durations')
        begin = max(float(cutoff), last)
        ready = begin+service
        rendered = ready+render
        out.append(dict(cutoff_seconds=float(cutoff), compute_start_seconds=begin,
                        queue_wait_seconds=begin-cutoff, numeric_ready_seconds=ready,
                        rendered_ready_seconds=rendered, service_seconds=float(service), render_seconds=float(render),
                        communication_latency='UNKNOWN', scene='measured_compute_replay'))
        # Shared sequential FIFO service, including render when requested in this scenario.
        last = rendered
    return out
