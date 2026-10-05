"""Separate FE01 and legacy system runners over locked physical requests."""
import contextlib
import copy
import json
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fe01.config import fingerprint, sha256
from fe01.data import read_event, split_catalog
from fe01.metrics import decode, score_rows
from fe01.windows import CAPABILITIES, build_window
from .checkpoint_inventory import inspect_checkpoint, load_cpu
from .provenance import require, read_json, write_json, training_identity, provenance, expand, seal
from .requests import read_frame, DECISION_KEYS, PAIR_KEYS, population_audit

ATOL = RTOL = 1e-5


def relocate(cfg):
    cfg = copy.deepcopy(cfg)
    for key, env in [('pretrained_manifest', 'FE01_WEIGHTS_ROOT'), ('diting_config', 'FE01_CODE_ROOT')]:
        import os
        if env in os.environ:
            cfg[key] = str(Path(os.environ[env]) / ('pretrained_manifest.json' if key == 'pretrained_manifest' else 'diting/config/diting_1200m_backbone_attnpool.yml'))
    import os
    if 'DATA_ROOT' in os.environ: cfg['data']['root'] = os.environ['DATA_ROOT']
    if 'FE01_SPLIT_MANIFEST' in os.environ: cfg['data']['split_manifest'] = os.environ['FE01_SPLIT_MANIFEST']
    return cfg


def sync(device):
    if torch.device(device).type == 'cuda': torch.cuda.synchronize(device)


def buffer_pin(model):
    from fe01.model import state_fingerprint
    return state_fingerprint(dict(model.named_buffers()))


def tensor_pin(inputs):
    from fe01.model import state_fingerprint
    return state_fingerprint({str(i): x for i, x in enumerate(inputs)})


class SystemRunner:
    """Wrapper has no registered parameters and never rewrites an original class."""
    def __init__(self, model, normalization, kind, device='cpu'):
        self.model, self.normalization, self.kind, self.device = model, normalization, kind, device
        self.model.requires_grad_(False).eval()
        self.active_cutoff = None
        self.cache = None
        self.station_index = 0
        self.encoder_seconds = 0.
        self.preprocessor_id = ('fe01_native_prefix_shape_scale_v2' if kind == 'fe01' else
            'legacy_strict_prefix_valid_loader_demean_then_original_masked_std_and_ln_scale_v1')

    @contextlib.contextmanager
    def cutoff_cache(self, token):
        require(self.active_cutoff is None, 'Nested or different-T cache reuse is forbidden')
        self.active_cutoff, self.cache = token, {}
        if self.kind == 'fe01':
            manager = self.model.same_cutoff_query_cache()
            try:
                with manager: yield
            finally:
                self.active_cutoff = self.cache = None
        else:
            original = self.model._encode_station_waveform
            def cached(model, waveform, **kwargs):
                index = self.station_index
                self.station_index += 1
                pin = tensor_pin([waveform] + [v for v in kwargs.values() if torch.is_tensor(v)])
                if index in self.cache:
                    old_pin, result = self.cache[index]
                    require(pin == old_pin, 'Cache received history/mask/geometry changed')
                    return result
                sync(self.device); begin = time.perf_counter()
                result = original(waveform, **kwargs)
                sync(self.device); self.encoder_seconds += time.perf_counter() - begin
                self.cache[index] = (pin, result)
                return result
            self.model._encode_station_waveform = types.MethodType(cached, self.model)
            try:
                yield
            finally:
                self.model._encode_station_waveform = original
                self.active_cutoff = self.cache = None

    def forward(self, inputs, token=None):
        if self.active_cutoff is not None:
            require(token == self.active_cutoff, 'Cannot reuse cache across different cutoff/request')
        self.station_index, self.encoder_seconds = 0, 0.
        with torch.no_grad(): return self.model(*inputs)

    @contextlib.contextmanager
    def capture_features(self):
        original=self.model._encode_station_waveform
        collected=[]
        def capture(model,*args,**kwargs):
            result=original(*args,**kwargs)
            collected.append(result[0].detach().cpu().clone())
            return result
        self.model._encode_station_waveform=types.MethodType(capture,self.model)
        try:yield collected
        finally:self.model._encode_station_waveform=original


def load_runner(entry, device):
    training_identity()
    identity = inspect_checkpoint(entry['checkpoint'], entry['epoch'], entry['kind'], entry.get('config'), entry.get('run_dir'))
    require(identity['status'] == 'IDENTITY_PASS', 'BLOCKED identity: ' + identity['reason'])
    require(entry.get('inventory_status')=='IDENTITY_PASS' and entry.get('checkpoint_sha256')==identity['checkpoint_sha256'],'CPU frozen inventory gate absent/stale')
    if entry['kind'] == 'fe01':
        from fe01.model import build_model
        checkpoint = load_cpu(entry['checkpoint'])
        original = checkpoint['config']
        cfg = relocate(original)
        parent_lock=read_json(Path(entry['run_dir'])/'protocol.lock.json')
        require(sha256(cfg['pretrained_manifest'])==parent_lock['pretrained_manifest_sha256'],'Original pretrained manifest byte SHA changed')
        model = build_model(cfg, device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        normalization = cfg['target_normalization']
        del checkpoint
    else:
        from eval_checkpoint import build_model_and_load
        from train_light import build_diting_args, CHECKPOINT_ENCODER_PREFIXES
        original = read_json(entry['config'])
        if isinstance(original, list): original = original[0]
        cfg = copy.deepcopy(original)
        checkpoint=load_cpu(entry['checkpoint'])
        require(tuple(checkpoint.get('excluded_prefixes',()))==tuple(CHECKPOINT_ENCODER_PREFIXES),'Legacy missing-prefix scope differs from original encoder-only contract')
        require(checkpoint.get('checkpoint_format')=='non_encoder_v1','Legacy checkpoint format must be authenticated non_encoder_v1')
        del checkpoint
        encoder = Path(entry.get('encoder', ''))
        require(encoder.is_file(), 'BLOCKED: legacy checkpoint requires its original external encoder')
        require(entry.get('encoder_sha256') and sha256(encoder) == entry['encoder_sha256'], 'External legacy encoder SHA must be explicitly pinned')
        # build_model_and_load performs the original strict non-encoder load and
        # restores RT61's exact immutable parent readout payload. No new parent.
        args = build_diting_args(entry['diting_config'], device=device, pretrained_override=str(encoder))
        model = build_model_and_load(cfg, args, entry['checkpoint'], device)
        normalization = cfg['training_params']['pga_target_normalization']
    runner = SystemRunner(model, normalization, entry['kind'], device)
    return runner, cfg, identity


def pack_inputs(event, request, cfg, native_length, kind='fe01', queries=None, grid_points=None):
    """Only received input waves and coordinates enter forward; no label tensors."""
    from dataclasses import replace
    capability = replace(CAPABILITIES['team_original_scratch'], native_n_samples=native_length)
    raw, cmask, info = build_window(event['waveform'], event['storage'], event['reference_sample'], float(request.elapsed_time), capability)
    require(info['status'] == 'supported', 'unsupported_history')
    for key in ('current_sample','cutout_exclusive','history_start_sample','history_end_sample'):
        require(info[key] == int(request[key]), 'Clock differs from locked request: ' + key)
    if 'absolute_decision_utc' in request and pd.notna(request.absolute_decision_utc):
        require(event['record_start_utc'] is not None,'Locked absolute clock lacks HDF record time')
        absolute=event['record_start_utc']+info['current_sample']/100
        require(abs(absolute-float(request.absolute_decision_utc))<=1e-6,'Absolute clock differs beyond UTC float serialization precision')
    ids = json.loads(request.input_ids)
    index = {str(s): i for i,s in enumerate(event['ids'])}
    require(len(ids) == len(set(ids)) and all(s in index for s in ids), 'Locked input IDs absent/duplicated')
    selected = np.asarray([index[s] for s in ids])
    require(np.array_equal(event['coords'][selected], np.asarray(json.loads(request.input_coords),dtype=np.float32)), 'Physical input coordinates differ')
    mask = cmask.all(1)
    require(mask[selected].any(1).all(), 'Locked input has no received physical support')
    require(((event['picks'][selected] > 0) & (event['picks'][selected] <= info['current_sample'])).all(), 'Input precedes trigger')
    parameters = cfg['model_params']
    stations, query_capacity = parameters['max_stations'], parameters['n_pga_targets']
    require(0 < len(selected) <= stations, 'Locked input count exceeds model capacity')
    wave = np.zeros((stations,3,native_length),dtype=np.float32)
    sample_mask = np.zeros((stations,native_length),dtype=bool)
    coords = np.zeros((stations,3),dtype=np.float32)
    valid = np.zeros(stations,dtype=bool)
    wave[:len(selected)] = np.where(mask[selected,None,:], raw[selected], 0)
    sample_mask[:len(selected)] = mask[selected]
    if kind != 'fe01':
        # Loader demeaning is prefix-valid only. FullModel retains its original
        # per-channel std normalization and natural-log raw scale statistics.
        count = sample_mask.sum(-1).clip(min=1)[:,None,None]
        mean = wave.sum(-1,keepdims=True)/count
        wave = np.where(sample_mask[:,None,:],wave-mean,0).astype(np.float32)
    offset = np.asarray([37,140,0],dtype=np.float32)
    coords[:len(selected)] = event['coords'][selected]-offset
    valid[:len(selected)] = True
    target_coords = np.zeros((query_capacity,3),dtype=np.float32)
    target_valid = np.zeros(query_capacity,dtype=bool)
    if grid_points is None:
        query_indices = np.asarray([index[str(s)] for s in queries])
        query_coords = event['coords'][query_indices]
    else:
        query_coords = np.asarray(grid_points,dtype=np.float32)
    require(len(query_coords) <= query_capacity, 'Query chunk exceeds original model capacity')
    target_coords[:len(query_coords)] = query_coords-offset
    target_valid[:len(query_coords)] = True
    inputs = [torch.from_numpy(a).unsqueeze(0) for a in (wave,coords,valid,target_coords,target_valid,sample_mask)]
    physical_pin = fingerprint(dict(input_ids=ids, input_coords=event['coords'][selected].tolist(),
        cutoff=info['cutout_exclusive'], history_start=info['history_start_sample'],
        raw_sha256=tensor_pin([torch.from_numpy(raw[selected][..., -info['required_n_samples']:]),
            torch.from_numpy(mask[selected][..., -info['required_n_samples']:])])) )
    return inputs, dict(**info, physical_received_history_sha256=physical_pin, model_input_sha256=tensor_pin(inputs),
                       query_coordinates=query_coords.tolist(), input_count=len(ids), preprocessor_id=kind,
                       physical_input_centroid=event['coords'][selected].mean(0).tolist(),
                       model_coordinate_center=(event['coords'][selected]-offset).mean(0).tolist())


def decision_token(request):
    return fingerprint([request.dataset_id, request.event_id, float(request.elapsed_time), request.geometry_protocol,
        request.get('input_mode','natural'), int(request.cutout_exclusive), request.input_ids])


def predict_decision(runner, cfg, event, requests, native_length, chunk=15, grid_points=None, use_cache=True, reuse_active=False):
    request = requests.iloc[0]
    if float(request.elapsed_time)*100+501 > native_length+1e-6:
        # Integer clock below is authoritative; do not extrapolate native domain.
        _, _, info = build_window(event['waveform'],event['storage'],event['reference_sample'],float(request.elapsed_time),
                                  __import__('dataclasses').replace(CAPABILITIES['team_original_scratch'],native_n_samples=native_length))
        if info['status'] != 'supported': return pd.DataFrame(), dict(status='unsupported_history', reason='native capacity')
    token = decision_token(request)
    records=[];timings=dict(preprocess_seconds=0.,encoder_seconds=0.,forward_seconds=0.,distribution_seconds=0.)
    manager=runner.cutoff_cache(token) if use_cache and not reuse_active else contextlib.nullcontext()
    with manager:
        count = len(grid_points) if grid_points is not None else len(requests)
        for start in range(0,count,chunk):
            subset = requests.iloc[start:start+chunk] if grid_points is None else requests
            points = None if grid_points is None else grid_points[start:start+chunk]
            begin=time.perf_counter()
            inputs, info = pack_inputs(event,request,cfg,native_length,runner.kind,
                                      queries=subset.station_id.tolist() if points is None else None,grid_points=points)
            timings['preprocess_seconds']+=time.perf_counter()-begin
            inputs=[x.to(runner.device) for x in inputs]
            sync(runner.device);begin=time.perf_counter()
            outputs=runner.forward(inputs,token if use_cache else None)
            sync(runner.device);timings['forward_seconds']+=time.perf_counter()-begin
            timings['encoder_seconds']+=getattr(runner.model,'_fe01_encoder_seconds',0.) if runner.kind=='fe01' else runner.encoder_seconds
            mdn=outputs[runner.model.output_layout.index('pga')][0,:len(points) if points is not None else len(subset)].detach().cpu().numpy()
            require(np.isfinite(mdn).all(), 'Nonfinite MDN; do not drop numerical failure')
            weights,mu,sigma=decode(mdn,runner.normalization)
            require((sigma>0).all(),'Nonpositive physical sigma')
            truth=subset.truth.to_numpy() if points is None else np.zeros(len(points))
            begin=time.perf_counter();scored=score_rows(truth,weights,mu,sigma);timings['distribution_seconds']+=time.perf_counter()-begin
            require(all(np.isfinite(scored[name]).all() for name in ('prediction','nll','crps','brier','predictive_sigma','q025','q975','q16','q84')),'Nonfinite scoring; failed targets cannot be omitted')
            for i in range(len(truth)):
                row=subset.iloc[i].to_dict() if points is None else dict(dataset_id=request.dataset_id,event_id=request.event_id,elapsed_time=request.elapsed_time,
                    geometry_protocol=request.geometry_protocol,input_mode=request.get('input_mode','natural'),latitude=float(points[i,0]),longitude=float(points[i,1]),elevation=float(points[i,2]),
                    split='val',units='log10(m/s^2)',target_group='grid_display_only',truth=None)
                if points is not None:
                    row.update(**{k:request[k] for k in ('magnitude','depth','event_latitude','event_longitude')})
                row.update(status='supported',**{k:float(v[i]) for k,v in scored.items() if points is None or k in ('prediction','q025','q975','q16','q84','width95','predictive_sigma','linear_mixture_mean_mps2','linear_mixture_median_mps2')})
                row.update(mdn_weights=json.dumps(weights[i].tolist()),mdn_mu=json.dumps(mu[i].tolist()),mdn_sigma=json.dumps(sigma[i].tolist()),
                           query_elevation=float(info['query_coordinates'][i][2]),physical_received_history_sha256=info['physical_received_history_sha256'],
                           model_input_sha256=info['model_input_sha256'],preprocessor_id=runner.preprocessor_id,
                           physical_input_centroid=json.dumps(info['physical_input_centroid']),model_coordinate_center=json.dumps(info['model_coordinate_center']))
                if points is None:
                    index={str(s):j for j,s in enumerate(event['ids'])}[str(row['station_id'])]
                    require(event['pga'][index] == np.float32(row['truth']), 'Query label differs from locked request at stored float32 precision')
                    require(np.array_equal(event['coords'][index,:2],np.asarray([row['latitude'],row['longitude']],dtype=np.float32)), 'Query coordinates differ')
                records.append(row)
    return pd.DataFrame(records),dict(status='supported',**timings)


def catalog_row(catalog, dataset, event_id):
    row = catalog.loc[(dataset,str(event_id))].copy(); row['dataset_id'],row['event_id']=dataset,str(event_id)
    return row


def data_gate(cfg, lock):
    # Original byte hashes can be reused only at their recorded size AND mtime.
    # Changed/copied shards are independently hashed before any new inference.
    require(sha256(cfg['data']['split_manifest']) == lock['data_identity']['split_manifest_sha256'], 'Frozen split byte SHA mismatch')
    records=[]
    for original, info in lock['data_identity']['shards'].items():
        p=Path(cfg['data']['root'])/Path(original).parent.name/Path(original).name
        require(p.is_file(),'Missing original Japan HDF5: '+str(p))
        require(p.stat().st_size==info['bytes'],'Japan shard byte size changed')
        unchanged=p.stat().st_mtime_ns==info['mtime_ns']
        if not unchanged: require(sha256(p)==info['sha256'],'Japan shard SHA changed')
        records.append(dict(relative_path=str(p.relative_to(cfg['data']['root'])),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,original_sha256=info['sha256'],verification='saved audit byte SHA at same bytes/mtime' if unchanged else 'independent byte SHA'))
    return records


def verify(entry, plan, source_root, output, device='cuda',request_root=None):
    require(request_root,'CPU hydrated request metadata required before forward verification')
    metadata_path=Path(request_root)/'fixed_station_metadata.csv.gz'
    metadata=read_frame(metadata_path)
    new_identity=read_json(Path(request_root)/'new_requests_identity.json')
    require(sha256(metadata_path)==new_identity['fixed_station_metadata_sha256'],'Station metadata pin differs from CPU planning')
    runner,cfg,identity=load_runner(entry,device)
    base_cfg=relocate(read_json(Path(source_root)/'formal__team_original_scratch__seed42/resolved_config.json'))
    lock=read_json(Path(source_root)/'formal__team_original_scratch__seed42/protocol.lock.json')
    data_identity=data_gate(base_cfg,lock)
    catalog=split_catalog(base_cfg,'val').set_index(['dataset_id','event_id'])
    requests=read_frame(Path(plan)/'fixed_requests.csv.gz')
    probes=read_frame(Path(plan)/'verification_decisions.csv')
    selected=requests.merge(probes[DECISION_KEYS],on=DECISION_KEYS,validate='many_to_one')
    native=cfg['native_n_samples'] if entry['kind']=='fe01' else 10000
    original=None
    if entry['kind']=='fe01':
        original=read_frame(Path(entry['run_dir'])/f"validation_epoch{entry['epoch']}.csv.gz")
        population_audit(original)
    checks=[];before=buffer_pin(runner.model)
    for decision_index,(key,group) in enumerate(selected.groupby(DECISION_KEYS)):
        row=catalog_row(catalog,key[0],key[1]);event=read_event(row,base_cfg,float(key[2]))
        batch,info=predict_decision(runner,cfg,event,group,native)
        require(info['status']=='supported','Probe decision unavailable')
        coordinates=batch.merge(metadata[['dataset_id','event_id','station_id','query_elevation']],on=['dataset_id','event_id','station_id'],how='left',suffixes=('_prediction','_locked'),validate='many_to_one')
        require(np.array_equal(coordinates.query_elevation_prediction.to_numpy(dtype=np.float32),coordinates.query_elevation_locked.to_numpy(dtype=np.float32)),'Real query elevation differs from CPU locked metadata')
        report=dict(zip(DECISION_KEYS,key),targets=len(batch),clock_and_labels_exact=True)
        if original is not None:
            old=original.merge(group[PAIR_KEYS],on=PAIR_KEYS,validate='one_to_one').set_index('station_id').loc[batch.station_id].reset_index()
            for field in ('prediction','mdn_weights','mdn_mu','mdn_sigma'):
                a=np.asarray([json.loads(v) for v in batch[field]]) if field.startswith('mdn_') else batch[field].to_numpy()
                b=np.asarray([json.loads(v) for v in old[field]]) if field.startswith('mdn_') else old[field].to_numpy()
                require(np.allclose(a,b,atol=ATOL,rtol=RTOL),'Existing CSV forward verification failed: '+field)
                report[field+'_max_abs_delta']=float(np.max(np.abs(a-b)))
        if decision_index < 4:
            singles,_=predict_decision(runner,cfg,event,group,native,chunk=1)
            reverse,_=predict_decision(runner,cfg,event,group.iloc[::-1],native)
            # Appending an arbitrary real query is tested via tensors, without inventing its scoring label.
            query_ids=group.station_id.head(2).tolist()
            unrelated=next((str(s) for s in event['ids'] if str(s) not in query_ids),None)
            require(unrelated is not None,'No unrelated real station for append probe')
            inputs,_=pack_inputs(event,group.iloc[0],cfg,native,runner.kind,queries=query_ids)
            direct_inputs=[t.to(device) for t in inputs]
            with torch.no_grad(): direct=runner.model(*direct_inputs)
            wrapped=runner.forward(direct_inputs)
            for a,b in zip(direct,wrapped): torch.testing.assert_close(a,b,atol=ATOL,rtol=RTOL)
            extended,_=pack_inputs(event,group.iloc[0],cfg,native,runner.kind,queries=query_ids+[unrelated])
            appended=runner.forward([t.to(device) for t in extended])
            pga_index=runner.model.output_layout.index('pga')
            torch.testing.assert_close(wrapped[pga_index][:,:len(query_ids)],appended[pga_index][:,:len(query_ids)],atol=ATOL,rtol=RTOL)
            for other in (singles,reverse):
                aligned=other.set_index('station_id').loc[batch.station_id]
                for field in ('prediction','mdn_weights','mdn_mu','mdn_sigma'):
                    a=np.array([json.loads(x) for x in batch[field]]) if field.startswith('mdn_') else batch[field].to_numpy()
                    b=np.array([json.loads(x) for x in aligned[field]]) if field.startswith('mdn_') else aligned[field].to_numpy()
                    require(np.allclose(a,b,atol=ATOL,rtol=RTOL),'Query reorder/single mismatch')
            uncached,_=predict_decision(runner,cfg,event,group,native,use_cache=False)
            for field in ('prediction','mdn_weights','mdn_mu','mdn_sigma'):
                a=np.array([json.loads(x) for x in batch[field]]) if field.startswith('mdn_') else batch[field].to_numpy()
                b=np.array([json.loads(x) for x in uncached[field]]) if field.startswith('mdn_') else uncached[field].to_numpy()
                require(np.allclose(a,b,atol=ATOL,rtol=RTOL),'Cached/uncached mismatch: '+field)
            with runner.cutoff_cache('cutoff-A'):
                try: runner.forward(direct_inputs,'cutoff-B')
                except ValueError: pass
                else: raise AssertionError('Different cutoff cache accepted')
            report.update(L0_same_tensor_original_wrapper='PASS',query_batch_reverse_single='PASS',query_append_unrelated='PASS',same_T_cache='PASS',different_T_cache_rejected=True)
        if decision_index < 2:
            full=read_event(row,base_cfg,float(key[2]),full_for_audit=True)
            with runner.capture_features() as baseline_features:
                predict_decision(runner,cfg,event,group,native)
            for name,value in [('future_NaN',np.nan),('future_large_pulse',1e10)]:
                mutated=copy.deepcopy(full);mutated['waveform'][...,int(group.cutout_exclusive.iloc[0]):]=value
                with runner.capture_features() as changed_features:
                    changed,_=predict_decision(runner,cfg,mutated,group,native)
                require(len(baseline_features)==len(changed_features),'Future feature trace length changed')
                for a,b in zip(baseline_features,changed_features):torch.testing.assert_close(a,b,atol=ATOL,rtol=RTOL)
                for field in ('prediction','mdn_weights','mdn_mu','mdn_sigma'):
                    a=np.array([json.loads(x) for x in batch[field]]) if field.startswith('mdn_') else batch[field].to_numpy()
                    b=np.array([json.loads(x) for x in changed[field]]) if field.startswith('mdn_') else changed[field].to_numpy()
                    require(np.allclose(a,b,atol=ATOL,rtol=RTOL),'Future perturbation changed '+field)
                require(batch.model_input_sha256.tolist()==changed.model_input_sha256.tolist(),'Future changed preprocessing')
                report[name]='PASS preprocessing/features/MDN'
        checks.append(report)
    require(buffer_pin(runner.model)==before,'Model eval changed buffers')
    report=provenance(run_id=entry['run_id'],status='PASS',checkpoint_sha256=identity['checkpoint_sha256'],
        checkpoint_epoch=entry['epoch'],parent_lock_sha256=identity.get('parent_lock_sha256','not applicable: legacy original config'),
        common_request_lock_sha256=sha256(Path(source_root)/'formal__team_original_scratch__seed42/protocol.lock.json'),
        training_source_sha=identity['training_source'],request_manifest_sha256=sha256(Path(plan)/'fixed_requests.csv.gz'),
        fixed_station_metadata_sha256=sha256(metadata_path),
        reused_csv_sha256=sha256(Path(entry['run_dir'])/f"validation_epoch{entry['epoch']}.csv.gz") if entry['kind']=='fe01' else None,
        L0='same input tensor equivalence, not historical loader recreation', C1=runner.preprocessor_id,
        atol=ATOL,rtol=RTOL,decisions=len(checks),checks=checks,data_identity=data_identity,
        unchanged_buffers=True,upstream_offline_causality='uncertified')
    write_json(Path(output)/'verification.json',report);seal(output)


def evaluate(entry, plan, request_root, source_root, output, verification, scope, device='cuda'):
    require(scope!='fixed' or entry['kind']!='fe01','FE01 fixed-time full rerun is disabled; use export-existing')
    gate=read_json(Path(verification)/'verification.json')
    require(gate['status']=='PASS' and gate['checkpoint_sha256']==sha256(entry['checkpoint']),'Forward verification gate absent/stale')
    from .provenance import evaluation_identity
    require(gate['evaluation_module_sha']==evaluation_identity()['evaluation_module_sha'],'Verification source identity is stale')
    require(gate['request_manifest_sha256']==sha256(Path(plan)/'fixed_requests.csv.gz'),'Verification request identity is stale')
    runner,cfg,identity=load_runner(entry,device)
    base_cfg=relocate(read_json(Path(source_root)/'formal__team_original_scratch__seed42/resolved_config.json'))
    data_gate(base_cfg,read_json(Path(source_root)/'formal__team_original_scratch__seed42/protocol.lock.json'))
    request_path=Path(plan if scope=='fixed' else request_root)/(scope+'_requests.csv.gz')
    if scope!='fixed':
        locked=read_json(Path(request_root)/'new_requests_identity.json')
        require(sha256(request_path)==locked['request_sha256'][scope],'CPU request manifest changed after planning')
    requests=read_frame(request_path)
    native=cfg['native_n_samples'] if entry['kind']=='fe01' else 10000
    catalog=split_catalog(base_cfg,'val').set_index(['dataset_id','event_id'])
    frames,statuses=[],[];before=buffer_pin(runner.model)
    for key,group in requests.groupby(DECISION_KEYS):
        if int(group.history_end_sample.iloc[0])-int(group.history_start_sample.iloc[0])+1>native:
            statuses.append(dict(zip(DECISION_KEYS,key),status='unsupported_history',reason='native capacity; no truncation',requested_targets=len(group)))
            continue
        row=catalog_row(catalog,key[0],key[1]);event=read_event(row,base_cfg,float(key[2]))
        frame,status=predict_decision(runner,cfg,event,group,native)
        statuses.append(dict(zip(DECISION_KEYS,key),**status,requested_targets=len(group)))
        if len(frame): frame['run_id']=entry['run_id'];frames.append(frame)
    frame=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    if scope in ('fixed','random'): require(all(s['status']=='supported' for s in statuses),'Common domain incomplete; blocked official paired comparison')
    if scope=='fixed': population_audit(frame)
    require(buffer_pin(runner.model)==before,'Evaluation changed model buffers')
    frame.to_csv(Path(output)/'predictions.csv.gz',index=False)
    pd.DataFrame(statuses).to_csv(Path(output)/'support_status.csv',index=False)
    write_json(Path(output)/'provenance.json',provenance(run_id=entry['run_id'],checkpoint_sha256=identity['checkpoint_sha256'],
        checkpoint_epoch=entry['epoch'],scope=scope,preprocessor_id=runner.preprocessor_id,system_role=entry.get('system_role','fe01_comparison'),
        request_manifest_sha256=sha256(request_path),verification_sha256=sha256(Path(verification)/'verification.json'),
        training_source_sha=identity['training_source'],parent_lock_sha256=identity.get('parent_lock_sha256','not applicable: legacy original config'),
        common_request_lock_sha256=sha256(Path(source_root)/'formal__team_original_scratch__seed42/protocol.lock.json'),
        unchanged_buffers=True,rows=len(frame),not_historical_pipeline=(entry['kind']!='fe01')))
    seal(output)
