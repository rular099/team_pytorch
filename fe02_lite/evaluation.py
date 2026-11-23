"""Full per-target mixtures and one source row per observation, streamed to NPZ."""
import copy
import json
from pathlib import Path

import numpy as np
import torch

from fe01.data import geometry
from fe01.windows import build_window, CAPABILITIES
from fe01.metrics import decode, score_rows, checkpoint_selection
from .config import fingerprint, sampling_config, write_json
from .data import catalog, read_event, prepare, transform_coordinates
from .artifacts import TableWriter, read_table, save_diagnostic, rng_state, restore_rng


def to_device(sample, device):
    return [x[None].to(device) for x in sample['inputs']]


def source_row(encoded, info, cfg, identity):
    loc = encoded['source_loc'][0].detach().cpu().numpy()
    mag = encoded['source_mag'][0].detach().cpu().numpy()
    mu = transform_coordinates(loc[:,1:4],cfg,source=True,inverse=True)
    sigma = loc[:,4:7]*np.asarray(cfg['coordinates']['scale'])
    weights = torch.softmax(encoded['source_loc'][0,:,0],-1).detach().cpu().numpy()
    return dict(**identity,observation_id=info['observation_id'],canonical_event_id=info['canonical_event_id'],
        event_id=info['event_id'],dataset_id=info['dataset_id'],elapsed_time=info['actual_elapsed_time'],
        geometry_protocol=info['geometry_protocol'],input_set_id=info['input_set_id'],
        input_ids=json.dumps(info['input_ids']),input_count=info['input_count'],
        absolute_cutoff_utc=info['absolute_decision_utc'],cutout_exclusive=info['cutout_exclusive'],
        absolute_exclusive_end_utc=info['record_axis_start_utc']+info['cutout_exclusive']/100,
        time_reference=info['time_reference'],truth_location=json.dumps(info['event_location']),
        truth_magnitude=info['magnitude'],location_mdn_normalized=json.dumps(loc.tolist()),
        location_weights=json.dumps(weights.tolist()),location_mu=json.dumps(mu.tolist()),
        location_sigma=json.dumps(sigma.tolist()),magnitude_mdn=json.dumps(mag.tolist()),
        location_units='latitude deg, longitude deg, depth km positive down')


@torch.no_grad()
def evaluate(model, cfg, directory, identity, device='cpu', times=None):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    predictions, sources, statuses = (TableWriter(directory/name) for name in ('predictions','sources','statuses'))
    identity = dict(identity, dataset_view=cfg['dataset_view'],run_id=cfg['run_id'],seed=cfg['seed'],
        variant=cfg['variant'],stage=cfg['stage'],split='val',config_sha256=fingerprint(cfg),
        model_dim=cfg['architecture']['model_dim'],amplitude_dim=cfg['architecture']['amplitude_dim'],
        station_pool_queries=cfg['architecture']['station_pool_queries'],
        source_aux_enabled=cfg['loss']['location']>0 or cfg['loss']['magnitude']>0,
        trainable_task_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    was_training = model.training
    model.eval()
    try:
        for _,row in catalog(cfg,'val').iterrows():
            for elapsed in times or cfg['realtime']['fixed_times']:
                event = read_event(row,cfg,elapsed)
                _,mask,window = build_window(event['waveform'],event['storage'],event['reference_sample'],elapsed,
                                             CAPABILITIES['diting_pretrained_frozen'])
                for protocol in ('normal','random'):
                    _,universe,roles,actual = geometry(event,mask.all(1),sampling_config(cfg),protocol)
                    if window['status'] != 'supported' or not len(universe):
                        statuses.append(dict(**identity,event_id=row.event_id,elapsed_time=elapsed,
                            geometry_protocol=actual,status=window['status'],reason=window.get('reason','no_target')))
                        continue
                    chunk = cfg['sampling']['query_chunk_size']
                    first = prepare(event,cfg,elapsed,protocol,queries=universe[:chunk])
                    info = first['info']
                    if first['inputs'] is None:
                        statuses.append(dict(**identity,event_id=row.event_id,elapsed_time=elapsed,
                            geometry_protocol=actual,status=info['status'],reason=info['reason']))
                        continue
                    inputs = to_device(first,device)
                    encoded = model.encode_observations(inputs[0],inputs[1],inputs[2],inputs[5])
                    sources.append(source_row(encoded,info,cfg,identity))
                    statuses.append(dict(**identity,event_id=row.event_id,elapsed_time=elapsed,
                        geometry_protocol=actual,status='supported',reason='',observation_id=info['observation_id']))
                    for start in range(0,len(universe),chunk):
                        query = universe[start:start+chunk]
                        target_coords = torch.from_numpy(transform_coordinates(event['coords'][query],cfg))[None].to(device)
                        output,_ = model.read_targets(encoded,target_coords)
                        mdn = output[0].cpu().numpy()
                        w,mu,sigma = decode(mdn,cfg['target_normalization'])
                        if not np.isfinite(mdn).all() or not (sigma>0).all():
                            raise FloatingPointError('Non-finite MDN; evaluation cannot omit failures')
                        scores = score_rows(event['pga'][query],w,mu,sigma,threshold=cfg['selection']['threshold_log10_pga'])
                        for i,j in enumerate(query):
                            record = dict(**identity,event_id=row.event_id,canonical_event_id=row.canonical_event_id,
                                dataset_id=row.dataset_id,observation_id=info['observation_id'],
                                input_set_id=info['input_set_id'],station_id=event['ids'][j],
                                physical_station_id=event['physical_ids'][j],
                                record_id=f'{row.dataset_id}:{row.event_id}:{event["ids"][j]}',
                                sensor_class=event['sensor_class'],target_role=roles[j],input_count=info['input_count'],
                                input_ids=json.dumps(info['input_ids']),input_coords=json.dumps(info['input_coords']),
                                geometry_protocol=actual,elapsed_time=float(elapsed),actual_elapsed_time=info['actual_elapsed_time'],
                                absolute_cutoff_utc=info['absolute_decision_utc'],time_reference=info['time_reference'],
                                absolute_exclusive_end_utc=event['record_start_utc']+info['cutout_exclusive']/100,
                                record_axis_start_utc=event['record_start_utc'],reference_sample=event['reference_sample'],
                                current_sample=info['current_sample'],cutout_exclusive=info['cutout_exclusive'],
                                history_start_sample=info['history_start_sample'],
                                latitude=float(event['coords'][j,0]),longitude=float(event['coords'][j,1]),height_km=float(event['coords'][j,2]),
                                event_latitude=float(event['event_location'][0]),event_longitude=float(event['event_location'][1]),
                                depth=float(event['event_location'][2]),magnitude=event['magnitude'],
                                truth=float(event['pga'][j]),units='log10(m/s^2)',status='supported',reason='',target_valid=True,
                                normalization_mean=cfg['target_normalization']['mean'],normalization_std=cfg['target_normalization']['std'],
                                normalization_sha256=fingerprint(cfg['target_normalization']),coordinate_sha256=fingerprint(cfg['coordinates']))
                            for k in range(3):
                                record.update({f'w{k}':float(w[i,k]),f'mu{k}':float(mu[i,k]),f'sigma{k}':float(sigma[i,k])})
                            record.update({name:float(value[i]) for name,value in scores.items()})
                            predictions.append(record)
    finally:
        model.train(was_training)
    for writer in (predictions,sources,statuses):
        writer.close(identity)
    frame = read_table(directory/'predictions')
    selection = checkpoint_selection(frame,cfg['realtime']['common_times'])
    write_json(directory/'selection.json',dict(score=selection,units='dex; log-space mixture mean MAE',
        rule=cfg['selection']['rule'],targets=len(frame),events=frame.event_id.nunique(),identity=identity))
    return selection, frame


@torch.no_grad()
def export_diagnostics(model, cfg, audit_dir, destination, tag, identity, device):
    if not cfg['diagnostics']['enabled']:
        return
    before = rng_state()
    mode = model.training
    model.eval()
    try:
        plan = json.loads((Path(audit_dir)/'diagnostic_plan.json').read_text())
        table = catalog(cfg,'val').set_index(['dataset_id','event_id'])
        for record in plan['events']:
            row = table.loc[(record['dataset_id'],record['event_id'])].copy()
            row['dataset_id'],row['event_id'] = record['dataset_id'],record['event_id']
            for elapsed in plan['times']:
                event = read_event(row,cfg,elapsed)
                for protocol in plan['protocols']:
                    sample = prepare(event,cfg,elapsed,protocol)
                    if sample['inputs'] is None:
                        raise ValueError('Frozen diagnostic support changed')
                    for count in plan['input_counts']:
                        info = copy.deepcopy(sample['info'])
                        inputs = [x.clone() for x in to_device(sample,device)]
                        if count == 'single':
                            inputs[2][:,1:] = False
                            inputs[5][:,1:] = False
                            info['input_ids'] = info['input_ids'][:1]
                            info['input_coords'] = info['input_coords'][:1]
                            info['input_count'] = 1
                            info['target_roles'] = [('triggered_noninput' if role=='observed_input' and sensor not in info['input_ids'] else role)
                                for sensor,role in zip(info['query_ids'],info['target_roles'])]
                        info['input_set_id'] = fingerprint(sorted(info['input_ids']))
                        info['observation_id'] = fingerprint(dict(view=cfg['dataset_view'],event=event['canonical_event_id'],
                            dataset_id=event['dataset_id'],cutoff=info['cutout_exclusive'],axis=event['record_start_utc'],
                            inputs=sorted(info['input_ids']),geometry=info['geometry_protocol']))
                        native_debug = cfg['diagnostics'].get('native_feature_debug',False)
                        if native_debug and len(plan['events'])>1:
                            raise ValueError('Native debug export is restricted to one fixed event')
                        outputs,tensors = model(*inputs,diagnostics=True,native_debug=native_debug)
                        tensors.update(input_coords=inputs[1],target_coords=inputs[3],target_valid=inputs[4])
                        sample_tag = f'{tag}_{record["priority"][:12]}_t{elapsed}_{protocol}_{count}'
                        save_diagnostic(destination,sample_tag,tensors,dict(**identity,**info,
                            count_rule=count,checkpoint_tag=tag,token_start_in_window=list(range(0,10000,50)),
                            token_start_on_record_axis=[info['cutout_exclusive']-10000+i for i in range(0,10000,50)],
                            station_order='input_ids then masked padding slots',summary_order='station-major then pool slot',
                            source_coordinates=cfg['coordinates'],normalization=cfg['target_normalization']),
                            cfg['diagnostics']['max_bytes'])
    finally:
        model.train(mode)
        restore_rng(before)
