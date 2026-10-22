"""Authenticate old bytes before computing; no model import on CPU score path."""
import copy
import json
import os
from pathlib import Path
import numpy as np
from .contracts import CONFIG, EPOCHS, FAMILIES, ON, ROOT
from .io import csv_read, fingerprint, read_json, require, sha256, write_json


def pins():
    return read_json(CONFIG / 'source_pins.json')['exports']


def load_export(source, family, policy, include_mdn=False):
    p = next(x for x in pins() if x['family'] == family and x['checkpoint_policy'] == policy)
    path = Path(source) / p['source']
    require(sha256(path) == p['prediction_sha256'], 'Saved prediction byte SHA mismatch: ' + str(path))
    # Every source column is retained; default CSV decoding is expressly forbidden.
    # Statistics never need hundreds of MB of repeated raw-MDN JSON strings.
    # Categorical physical strings preserve exact values/leading zeros and share repeated decision metadata.
    import pandas as pd
    from fe01_review.requests import IDENTITY_KEYS
    wanted=set(IDENTITY_KEYS+['split','units','status','seed','model_family','absolute_amplitude_mode',
        'latitude','longitude','event_latitude','event_longitude','depth','magnitude','input_coords',
        'prediction','nll','crps','brier','covered68','covered95','predictive_sigma','width95','q16','q84'])
    if include_mdn:
        wanted.update(['mdn_logits','mdn_weights','mdn_mu','mdn_sigma'])
    f=pd.read_csv(path,float_precision='round_trip',usecols=lambda c:c in wanted,
        dtype={'dataset_id':str,'event_id':str,'station_id':str,'input_ids':'category','input_coords':'category'})
    # Pairing uses plain string columns; intern categories after parsing without changing their values.
    for column in ('input_ids','input_coords'):
        f[column]=f[column].astype(object)
    from fe01_review.requests import population_audit
    population_audit(f)
    require(f.absolute_amplitude_mode.eq(ON).all(), 'Not ON')
    require(f.seed.eq(42).all(), 'Not seed42')
    require(f.model_family.eq(FAMILIES[family]).all(), 'Wrong model family')
    require(f.status.eq('supported').all(), 'Missing legacy output')
    return f.sort_values(['dataset_id', 'event_id', 'elapsed_time', 'station_id', 'geometry_protocol']).reset_index(drop=True)


def verify_export_seals(source):
    source = Path(source)
    out = []
    for p in pins():
        relative = p['source']
        path = source / relative
        require(sha256(path) == p['prediction_sha256'], 'Wrong export: ' + relative)
        gate = read_json(source / 'results' / ('index' + str(p['source_index'])) / 'EVAL_COMPLETE.json')
        declared_gate=read_json(ROOT/'reports/fe01_a01_hpc_results_20261021/source_metadata/results'/
                                ('index'+str(p['source_index']))/'EVAL_COMPLETE.json')
        require(gate==declared_gate,'Original source/revision/plan seal differs from published provenance')
        local = '/'.join(Path(relative).parts[2:])
        require(gate['status'] == 'PASS' and gate['split'] == 'validation', 'Wrong legacy stage gate')
        require(gate['files_sha256'][local] == p['prediction_sha256'], 'Legacy prediction seal differs')
        identity = read_json(ROOT / 'reports/fe01_a01_hpc_results_20261021' / p['checkpoint_identity_source'])
        require(identity.get('file_sha256', identity.get('checkpoint_sha256')) == p['checkpoint_sha256'], 'Wrong exported weight identity')
        out.append(dict(family=p['family'], policy=p['checkpoint_policy'], epoch=p['epoch'],
                        prediction_sha256=p['prediction_sha256'], checkpoint_sha256=p['checkpoint_sha256']))
    return out


def checkpoint_spec(session, family, policy='selected'):
    """Use the old frozen multi-root plan, never a filename/mtime/loss search."""
    plan = read_json(Path(session) / 'plan.json')
    i = {'TEAM': 0, 'PhaseNet': 1, 'EQT': 2, 'DiTing': 3}[family]
    row = plan['runs'][i]
    run = Path(row['run_dir']) if family == 'DiTing' else Path(row['config']['a01']['original_run'])
    override = os.environ.get('ON_EVAL_CHECKPOINT_MAP')
    if override:
        mapping = read_json(override)
        run = Path(mapping.get(family, str(run)))
    path = run / ('best.pth' if policy == 'selected' else 'last.pth')
    pin = next(p for p in pins() if p['family'] == family and p['checkpoint_policy'] == policy)
    return dict(family=family, policy=policy, run_dir=str(run), path=str(path),
                epoch=pin['epoch'], checkpoint_sha256=pin['checkpoint_sha256'],
                config_path=str(run / 'resolved_config.json'), lock_path=str(run / 'protocol.lock.json'),
                audit_dir=row['audit_dir'] if family == 'DiTing' else str(run))


def metadata_gate(payload, spec, cfg, lock):
    """Torch-free metadata rejection, in addition to the mandatory byte/tensor checks."""
    require(payload['epoch'] == spec['epoch'] and payload['config'] == cfg, 'Wrong epoch/config')
    require(cfg['model_family'] == FAMILIES[spec['family']] and cfg['seed'] == 42, 'Wrong config family/seed')
    require(cfg['window']['protocol'] == 'native_prefix_v2', 'Rolling forbidden')
    require(payload['source_config_sha256'] == lock['config_sha256'], 'Wrong source config')
    require(lock['code_sha256'] == '3e537ff6693971d92b139e517f370b998237bd8b29989194e9ddcc06cc9dcfea', 'Wrong original source')
    if spec['family'] != 'TEAM':
        import re
        require(bool(re.fullmatch(r'[0-9a-f]{64}', str(lock.get('pretrained_manifest_sha256', '')))), 'Missing pretrained identity')
    else:
        require(lock.get('pretrained_manifest_sha256') is None, 'TEAM scratch has a contradictory pretrained identity')
    if spec['family'] == 'DiTing':
        require(payload['a01_code_sha256'] == lock['a01_code_sha256'] ==
                '3e1e4398b01d724f793f88b19be2a2c03ca67a71e881d876c173a8fbc5b8177f', 'Wrong A01 source')
        require(payload['absolute_amplitude_mode'] == cfg['absolute_amplitude_mode'] == ON, 'Wrong amplitude mode')


def inspect(spec, return_payload=False):
    """CPU checkpoint content authentication, before any model construction."""
    import torch
    from fe01.model import state_fingerprint
    from fe01_review.provenance import training_identity, TRAIN_CODE
    from fe01_review.checkpoint_inventory import inspect_checkpoint
    training_identity()  # authenticates all 115, including the external vendor snapshot
    require(sha256(spec['path']) == spec['checkpoint_sha256'], 'Checkpoint bytes differ from original HPC export')
    cfg = read_json(spec['config_path'])
    lock = read_json(spec['lock_path'])
    require(cfg['model_family'] == FAMILIES[spec['family']] and cfg['seed'] == 42, 'Wrong config family/seed')
    require(cfg['window']['protocol'] == 'native_prefix_v2', 'Rolling forbidden')
    require(lock['code_sha256'] == TRAIN_CODE, 'Wrong training code')
    p = torch.load(spec['path'], map_location='cpu', weights_only=False)
    metadata_gate(p, spec, cfg, lock)
    state = p['model_state_dict']
    if spec['family'] != 'DiTing':
        old = inspect_checkpoint(spec['path'], spec['epoch'], 'fe01', spec['config_path'], spec['run_dir'])
        require(old['status'] == 'IDENTITY_PASS', old['reason'])
        expected = read_json(ROOT / 'reports/fe01_a01_hpc_results_20261021' / next(
            x['checkpoint_identity_source'] for x in pins() if x['family'] == spec['family'] and x['checkpoint_policy'] == spec['policy']))
        require(old['state_content_sha256'] == expected['state_content_sha256'], 'Old tensor state mismatch')
    else:
        require(p['a01_code_sha256'] == lock['a01_code_sha256'] ==
                '3e1e4398b01d724f793f88b19be2a2c03ca67a71e881d876c173a8fbc5b8177f', 'Wrong A01 source')
        require(p['source_config_sha256'] == lock['config_sha256'], 'Wrong source config')
        require(p['identity'] == fingerprint(dict(config=cfg, audit_lock=lock, world_size=p['world_size'])), 'Wrong lock/world identity')
        require(p['absolute_amplitude_mode'] == cfg['absolute_amplitude_mode'] == ON, 'Wrong amplitude policy')
        require(p['common_initial_state_sha256'] == lock['initial_state']['common_sha256'], 'Wrong initialization')
        require(p['updates'] == spec['epoch'] * 212 and p['world_size'] == 16, 'Wrong training budget')
    encoder = {k[len('waveform_model.encoder.'):]: v for k, v in state.items() if k.startswith('waveform_model.encoder.')}
    require(bool(encoder), 'Missing encoder state')
    encoder_sha = state_fingerprint(encoder)
    if spec['family'] == 'DiTing':
        require(encoder_sha == p['encoder_state_sha256'] == lock['initial_state']['encoder_sha256'], 'Frozen encoder mismatch')
    # The whole original trained TEAM state is pinned above. It is never compared to epoch0.
    if spec['family'] != 'TEAM':
        from fe01_a01.identity import check_family
        check_family(runtime_config(cfg), lock)
    result = dict(**spec, status='IDENTITY_PASS', config_sha256=fingerprint(cfg),
                  lock_sha256=sha256(spec['lock_path']), state_sha256=state_fingerprint(state),
                  encoder_sha256=encoder_sha, encoder_policy='trained scratch' if spec['family'] == 'TEAM' else 'frozen pretrained',
                  normalization=cfg['target_normalization'])
    return (result, p) if return_payload else result


def runtime_config(recorded):
    """Relocate path fields only, after the original config has been authenticated."""
    from fe01_review.runners import relocate
    return relocate(recorded)


def load_spatial_export(source,family,policy):
    """Reuse original field exports; validate their immutable saved byte seals."""
    import csv
    index={'TEAM':0,'PhaseNet':1,'EQT':2,'DiTing':4}[family]
    relative=f'results/index{index}/comparisons/{policy}/spatial_fields.csv.gz'
    with (ROOT/'reports/fe01_a01_hpc_results_20261021/source_inventory.csv').open() as stream:
        digests={r['path']:r['sha256'] for r in csv.DictReader(stream)}
    path=Path(source)/relative
    require(sha256(path)==digests[relative],'Saved spatial byte SHA differs')
    gate=read_json(Path(source)/'results'/f'index{index}'/'EVAL_COMPLETE.json')
    require(gate['files_sha256'][f'comparisons/{policy}/spatial_fields.csv.gz']==digests[relative],'Spatial source seal differs')
    f=csv_read(path)
    return f[f.model_mode.eq('ON') & f.status.eq('supported')].copy()
