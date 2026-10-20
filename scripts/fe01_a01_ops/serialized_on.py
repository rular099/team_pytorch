#!/usr/bin/env python3
"""Read sealed exported floats exactly, then apply the unchanged production gate."""
from pathlib import Path


def read_serialized_on(path, expected_sha256, report_path):
    import pandas as pd
    from fe01.config import sha256, write_json
    from fe01_review.requests import population_audit, read_frame, IDENTITY_KEYS, EXPECTED_POP, EXPECTED_TRUTH
    path = Path(path)
    report = dict(status='STARTED', source=str(path), source_expected_sha256=expected_sha256,
                  parser=dict(engine='c', float_precision='round_trip'),
                  expected_population_sha256=EXPECTED_POP, expected_truth_input_clock_sha256=EXPECTED_TRUTH,
                  policy='Read existing CSV bytes exactly; unchanged production audit; no replacement or tolerance')
    frame = None
    try:
        report['source_actual_sha256'] = sha256(path)
        if report['source_actual_sha256'] != expected_sha256:
            raise ValueError('Sealed ON prediction bytes changed')
        frame = pd.read_csv(path, engine='c', float_precision='round_trip',
                            dtype={'dataset_id': str, 'event_id': str, 'station_id': str})
        # This is deliberately the original production=True gate, including the
        # locked hashes, 194265 rows, 1310 events, role counts and physical clocks.
        report['strict_population_audit'] = population_audit(frame)
        if sha256(path) != expected_sha256:
            raise ValueError('Sealed ON prediction bytes changed during read')
        ordinary = read_frame(path)
        report['default_parser_identity_differences'] = {
            name: int((~frame[name].eq(ordinary[name])).sum()) for name in IDENTITY_KEYS
            if not frame[name].equals(ordinary[name])}
        report['status'] = 'PASS'
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report.update(status='FAIL', error=str(exc))
        if frame is not None:
            try:
                # Diagnostics only. Never substitute this for the production gate.
                report['observed_population'] = population_audit(frame, production=False)
            except (ValueError, KeyError, TypeError) as diagnostic:
                report['observed_population_error'] = str(diagnostic)
        write_json(report_path, report)
        print('A01_SERIALIZED_ON_CHECK=FAIL details='+str(report_path), flush=True)
        raise ValueError('Serialized ON identity check failed; details: '+str(report_path)) from exc
    write_json(report_path, report)
    print('A01_SERIALIZED_ON_CHECK=PASS details='+str(report_path), flush=True)
    return frame
