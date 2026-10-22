"""Physical quality uses actual masks, independent of model native padding."""
import numpy as np
from .io import require


def station_snr(wave, mask, pick, cutoff, fs=100, clipping=None):
    wave = np.asarray(wave)
    precision=np.finfo(wave.dtype) if np.issubdtype(wave.dtype,np.floating) else np.finfo(np.float64)
    wave=wave.astype(np.float64,copy=False)
    mask = np.asarray(mask, bool)
    if mask.ndim == 1:
        mask = np.broadcast_to(mask[None, :], wave.shape)
    stop = min(int(cutoff), wave.shape[-1])
    wave, mask = wave[:, :stop], mask[:, :stop]
    def window(a, b):
        lo, hi = int(np.ceil(a)), int(np.ceil(b))
        total = max(hi-lo, 0)
        lo, hi = max(lo, 0), min(hi, stop)
        good = mask[:, lo:hi].all(0)
        return wave[:, lo:hi][:, good], int(good.sum())/fs, good.sum()/total if total else 0
    base = dict(snr_name='causal_first_second_snr_proxy_db', status='Unknown', snr_db=None,
                available_at_decision=True, retrospective_only=False)
    if not np.isfinite(pick) or pick <= 0:
        return dict(base, reason='unknown_P_clock')
    noise, ns, nc = window(pick-5*fs, pick-fs)
    # Signal interval has an inclusive physical end; cutoff is exclusive.
    signal_end=int(np.floor(min(pick+fs,cutoff-1)))+1
    signal, ss, sc = window(pick, signal_end)
    base.update(noise_valid_seconds=ns, noise_coverage=nc, signal_valid_seconds=ss, signal_coverage=sc,
                signal_endpoint='inclusive min(station_P+1s,current_sample); raw cutoff remains exclusive',
                clipping_status='not_detected' if clipping is False else 'known_clipping' if clipping else 'UNKNOWN_no_sensor_limit_metadata')
    if clipping:
        return dict(base, reason='clipping')
    if ns < 2 or nc < .8 or ss < .5 or sc < .8:
        return dict(base, reason='insufficient_valid_noise_or_signal')
    if not np.isfinite(noise).all() or not np.isfinite(signal).all():
        return dict(base, reason='nonfinite_received_samples')
    mean = noise.mean(1, keepdims=True)
    en = float(np.mean((noise-mean)**2))
    es = float(np.mean((signal-mean)**2))
    if en <= 0 or es <= 0 or not np.isfinite(en+es):
        return dict(base, reason='zero_or_invalid_energy')
    floor=max(float(precision.tiny),float(precision.eps)**2*max(es,float(np.mean(noise**2))))
    if en<=floor:
        return dict(base,reason='noise_energy_below_numeric_resolution',noise_energy=en,numeric_resolution_floor=floor)
    return dict(base, status='OK', snr_db=float(10*(np.log10(es)-np.log10(en))), reason='',
                noise_energy=en, observed_signal_including_noise_energy=es)


def decision_quality(event, selected, elapsed):
    from fe01.windows import clock
    current, cutoff = clock(event['reference_sample'], elapsed, event['fs'])
    start = int(np.floor(event['reference_sample']))-500
    required = cutoff-start
    rows, missing, durations = [], [], []
    for i in selected:
        lo, hi = max(0, start), min(cutoff, event['waveform'].shape[-1])
        valid = event['storage'][i, lo:hi]
        if valid.ndim == 2:
            valid = valid.all(0)
        count = int(valid.sum())
        fraction = 1-count/required
        post = max(lo, int(np.ceil(event['picks'][i])))
        vpost = event['storage'][i, post:hi]
        post_seconds = float(vpost.sum()/event['fs'])
        if vpost.ndim == 2:
            post_seconds = float(vpost.all(0).sum()/event['fs'])
        q = station_snr(event['waveform'][i], event['storage'][i], event['picks'][i], cutoff, event['fs'])
        q.update(station_id=str(event['ids'][i]), valid_seconds=count/event['fs'],
                 missing_fraction=float(fraction), post_p_valid_seconds=post_seconds,
                 physical_history_samples=required, native_padding_not_counted_as_missing=True)
        import hashlib
        q['received_joint_mask_sha256']=hashlib.sha256(np.asarray(valid,dtype=np.uint8).tobytes()).hexdigest()
        rows.append(q)
        missing.append(fraction)
        durations.append(post_seconds)
    good = [r['snr_db'] for r in rows if r['status'] == 'OK']
    coverage = len(good)/len(selected) if len(selected) else 0
    summary = dict(input_count=len(selected), snr_available_inputs=len(good), snr_available_fraction=coverage,
                   snr_db=float(np.median(good)) if good and coverage >= .8 else np.nan,
                   snr_lower_quartile_db=float(np.quantile(good, .25)) if good else np.nan,
                   physical_missing_fraction=float(np.mean(missing)) if missing else np.nan,
                   post_p_valid_seconds_min=float(min(durations)) if durations else np.nan,
                   post_p_valid_seconds_median=float(np.median(durations)) if durations else np.nan,
                   inputs_post_p_at_least_1s=sum(v >= 1 for v in durations),
                   snr_status='OK' if coverage >= .8 and good else 'Unknown')
    return rows, summary
