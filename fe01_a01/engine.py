"""A01 training loop: explicit factory, frozen FE01 sampler/loss/DDP semantics.

The loop is a static copy of the base FE01 implementation with only factory,
initial-state restoration and A01 identity fields changed. No monkeypatch.
"""
import contextlib
import copy
import datetime
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from fe01.config import fingerprint, sha256, safe_output
from fe01.data import TrainingDataset, collate, split_catalog, read_event, prepare_sample
from fe01.engine import (distributed_device, runtime, evaluate, loss, inputs_to,
                         data_identity, audit_population, query_audit)
from fe01.metrics import checkpoint_selection
from fe01.model import common_state, model_audit, state_fingerprint
from .model import build_model
from .provenance import validate, require, read_json, write_json, source_identity, new_output
from .identity import (check_family, old_checkpoint, training_budget, restore_initial,
                       reuse_equivalence)


def sample_plan(cfg, destination, world):
    """Shared family ON/OFF event/epoch/draw schedule, including actual DDP drops."""
    dataset = TrainingDataset(cfg)
    rows = []
    for epoch in range(cfg['training']['epochs']):
        for i,row in dataset.catalog.iterrows():
            for draw in range(cfg['realtime']['draws_per_event']):
                elapsed = dataset.sampler.sample(row.dataset_id,row.event_id,epoch,draw,cfg['sampling_seed'])
                rows.append(dict(dataset_id=row.dataset_id,event_id=row.event_id,epoch=epoch,
                    draw=draw,elapsed_time=elapsed,geometry_stream='mixed',sampling_seed=cfg['sampling_seed']))
    table = pd.DataFrame(rows)
    table.to_csv(destination/'sampling_manifest.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    order = {}
    budget = training_budget(len(dataset.catalog),cfg,world)
    used = budget['updates_per_epoch']*budget['accumulation']*cfg['training']['microbatch']
    for epoch in range(cfg['training']['epochs']):
        order[str(epoch)] = {}
        for rank in range(world):
            sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,
                seed=cfg['sampling_seed'],drop_last=True)
            sampler.set_epoch(epoch)
            order[str(epoch)][str(rank)] = fingerprint(list(sampler)[:used])
    write_json(destination/'ddp_sampling_order.json',order)
    return dict(plan_sha256=fingerprint(rows),ddp_order_sha256=fingerprint(order),budget=budget)


def audit(cfg,destination,device='cpu',world=16):
    validate(cfg)
    destination=new_output(destination)
    source=source_identity()
    from .assets import ensure_diting_manifest
    ensure_diting_manifest(cfg)
    original_dir=Path(cfg['a01']['original_run'])
    original_audit=Path(cfg['a01']['population_audit'])
    old_lock=read_json(original_dir/'protocol.lock.json')
    old_cfg=read_json(original_dir/'resolved_config.json')
    for key in ('training','realtime','geometry','window','model_params'):
        require(cfg[key]==old_cfg[key],'Common FE01 semantics changed: '+key)
    # Cohorts, normalization and selection population are reused only after data verification.
    require(old_lock['code_sha256']==source['original_training']['code_sha256'],'Old training code differs')
    data=read_json(original_audit/'data_split_audit.json')
    require(sha256(cfg['data']['split_manifest'])==data['split_manifest_sha256'],'Split changed')
    actual=data_identity(cfg,with_hash=False)
    require(set(actual['shards'])==set(data['shards']),'Runtime HDF paths differ from original audited data')
    require(cfg['data']==old_cfg['data'] and cfg['spatial']==old_cfg['spatial'], 'Data/mask/spatial contract changed')
    for path,record in data['shards'].items():
        stat=Path(path).stat()
        require(stat.st_size==record['bytes'] and stat.st_mtime_ns==record['mtime_ns'],'Data changed: redo independent hash audit')
    effective={k:old_cfg[k] for k in ('audited_cohorts','target_normalization')}
    resolved=copy.deepcopy(cfg);resolved.update(effective)
    require(len(effective['audited_cohorts']['train'])==9084 and len(effective['audited_cohorts']['val'])==1310,
            'Production original cohort differs')
    write_json(destination/'data_split_audit.json',data)
    population=original_audit/'validation_population.csv.gz'
    (destination/'validation_population.csv.gz').write_bytes(population.read_bytes())
    dep=dict(pretrained_manifest_sha256=(sha256(cfg['pretrained_manifest'])
              if cfg['model_family']!='team_original_scratch' else None))
    dependency=check_family(resolved,dep)
    model=build_model(resolved,device)
    original_common=read_json(original_dir/'model_interface_audit.json')['common_initial_state_sha256']
    require(state_fingerprint(common_state(model))==original_common,'Reconstructed common initial state differs from original FE01')
    native_initial=state_fingerprint(model.state_dict())
    reuse=dict(status='NEW_MATCHED_PAIR',reason='DiTing has no old common-downstream ON comparator')
    if cfg['model_family']!='diting_pretrained_frozen':
        require(old_cfg['model_family']==cfg['model_family'],'Wrong original family')
        epoch=cfg['a01']['original_selected_epoch']
        selected,selected_identity=old_checkpoint(original_dir/'best.pth',epoch,original_dir,resolved)
        if (original_dir/'init.pth').is_file():
            initial,initial_identity=old_checkpoint(original_dir/'init.pth',0,original_dir,resolved)
        else:
            pins=cfg['a01']['original_checkpoint_pins']
            require(native_initial==pins['init']['state_sha256'] and
                    state_fingerprint(common_state(model))==cfg['a01']['original_common_initial_sha256'],
                    'Missing init: reconstruction fingerprint does not match frozen inventory')
            initial=dict(model_state_dict=copy.deepcopy(model.state_dict()))
            initial_identity=dict(status='RECONSTRUCTED_FINGERPRINT_PASS',state_content_sha256=native_initial,
                authenticated_world_size=selected_identity['authenticated_world_size'])
        # Restore saved epoch0, not the trained ON final checkpoint.
        model.load_state_dict(initial['model_state_dict'],strict=True)
        require(native_initial==state_fingerprint(initial['model_state_dict']),
                'Reconstructed full frontend/adapter/common initial state differs')
        last,last_identity=old_checkpoint(original_dir/'last.pth',12,original_dir,resolved)
        del last
        row=split_catalog(resolved,'val').iloc[0]
        sample=prepare_sample(read_event(row,resolved,3),resolved,3,'normal')
        runtime_old=copy.deepcopy(old_cfg)
        for key in ('pretrained_manifest','diting_config'):
            runtime_old[key]=resolved[key]
        reuse=reuse_equivalence(resolved,runtime_old,initial['model_state_dict'],inputs_to(sample,device),
                                report_path=destination/'ON_equivalence.json')
        # Check trained ON forward too, without using it to initialize the OFF run.
        legacy_model=__import__('fe01.model',fromlist=['build_model']).build_model(runtime_old,device)
        legacy_model.load_state_dict(selected['model_state_dict'],strict=True)
        model.load_state_dict(selected['model_state_dict'],strict=True)
        legacy_model.eval();model.eval()
        from . import ON
        original_mode=model.absolute_amplitude_mode;model.absolute_amplitude_mode=ON
        from .equivalence import paired_execution, capture_rng, restore_rng, outputs_comparison
        with paired_execution(device) as devices, torch.no_grad():
            packed=inputs_to(sample,device);paired_rng=capture_rng(devices)
            restore_rng(paired_rng);a=legacy_model(*packed)
            restore_rng(paired_rng);b=model(*packed)
        trained=outputs_comparison(a,b)
        write_json(destination/'trained_ON_forward_equivalence.json',trained)
        require(trained['passed'],'Trained ON forward differs; details: '+str(destination/'trained_ON_forward_equivalence.json'))
        reuse['trained_ON_forward_max_abs']=trained['max_abs']
        reuse['trained_ON_forward_comparison']=trained
        model.absolute_amplitude_mode=original_mode
        model.load_state_dict(initial['model_state_dict'],strict=True)
        del legacy_model,selected,initial
        previous=training_budget(9084,old_cfg,initial_identity['authenticated_world_size'])
        current=training_budget(9084,resolved,world)
        require(previous['world_size']==current['world_size'], 'DDP world changed: per-rank loss denominators/order cannot be assumed equivalent')
        require(previous['total_updates']==current['total_updates'] and previous['consumed_samples_per_epoch']==current['consumed_samples_per_epoch'],
                'Original update budget differs')
        reuse.update(original_init=initial_identity,selected=selected_identity,last=last_identity,
                     original_budget=previous,matching_optimizer_loss_scheduler=True)
    # Independent matched init for each new pair; OFF and ON have identical bytes.
    torch.save(dict(epoch=0,model_state_dict=model.state_dict(),config=resolved),destination/'matched_init.pth')
    initial=dict(path=str(destination/'matched_init.pth'),file_sha256=sha256(destination/'matched_init.pth'),
                 state_sha256=state_fingerprint(model.state_dict()),common_sha256=state_fingerprint(common_state(model)),
                 encoder_sha256=state_fingerprint(model.waveform_model.encoder.state_dict()))
    independence=query_audit(model,resolved,split_catalog(resolved,'val').iloc[0],device)
    effective.update(query_independence_verified=True,query_chunk_size=cfg['query_chunk_size'])
    plan=sample_plan(resolved,destination,world)
    lock=dict(config_sha256=fingerprint(cfg),a01_code_sha256=source['a01_code_sha256'],
        source_commit=source['source_commit'],
        code_sha256=source['original_training']['code_sha256'],effective_config=effective,
        data_identity=data,initial_state=initial,pretrained_manifest_sha256=dep['pretrained_manifest_sha256'],
        dependency=dependency,reuse_gate=reuse,sampling=plan,world_size=world,status='AUDIT_PASS',
        validation_population_sha256=sha256(population),query_independence=independence)
    write_json(destination/'source_identity.json',source)
    write_json(destination/'model_interface_audit.json',model_audit(model))
    write_json(destination/'resolved_config.json',{**cfg,**effective})
    write_json(destination/'protocol.lock.json',lock)
    return lock


def check_matched_pair(cfg,lock):
    if cfg['model_family']!='diting_pretrained_frozen':return
    from . import ON,OFF
    other_mode='off' if cfg['absolute_amplitude_mode']==ON else 'on'
    name=cfg['run_id'].replace('__on__','__'+other_mode+'__').replace('__off__','__'+other_mode+'__')
    other=read_json(Path(cfg['output_root'])/'audits'/name/'protocol.lock.json')
    require(other['status']=='AUDIT_PASS' and other['a01_code_sha256']==lock['a01_code_sha256'],'Matched DiTing pair gate missing/stale')
    for key in ('state_sha256','common_sha256','encoder_sha256'):
        require(other['initial_state'][key]==lock['initial_state'][key],'DiTing pair init differs: '+key)
    require(other['sampling']==lock['sampling'],'DiTing ON/OFF draw/DDP budget differs')


def check_audit(cfg,audit_dir):
    validate(cfg)
    lock=read_json(Path(audit_dir)/'protocol.lock.json')
    require(lock['status']=='AUDIT_PASS' and lock['config_sha256']==fingerprint(cfg),'Stale config/audit gate')
    source=source_identity()
    require(lock['a01_code_sha256']==source['a01_code_sha256'] and lock['code_sha256']==source['original_training']['code_sha256'],
            'Source changed: invalidate identity/diagnostics/pilot gates')
    check_family(cfg,lock)
    data=lock['data_identity']
    require(sha256(cfg['data']['split_manifest'])==data['split_manifest_sha256'],'Split changed')
    for path,record in data['shards'].items():
        stat=Path(path).stat()
        require(stat.st_size==record['bytes'] and stat.st_mtime_ns==record['mtime_ns'],'Data changed after audit')
    require(sha256(Path(audit_dir)/'validation_population.csv.gz')==lock['validation_population_sha256'],'Validation plan changed')
    return lock


def pilot(cfg,audit_dir,destination,device='cuda'):
    lock=check_audit(cfg,audit_dir)
    check_matched_pair(cfg,lock)
    destination=new_output(destination)
    resolved=copy.deepcopy(cfg);resolved.update(lock['effective_config'])
    model=build_model(resolved,device);restore_initial(model,resolved,lock)
    model.train()
    frozen_before=state_fingerprint(model.waveform_model.encoder.state_dict())
    dataset=TrainingDataset(resolved)
    part=collate([dataset[i] for i in range(min(cfg['training']['microbatch'],len(dataset)))])
    inputs=[x.to(device) for x in part['inputs']];labels=[x.to(device) for x in part['labels']]
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay'])
    value=loss(model(*inputs),labels,model,resolved,inputs[4])
    require(torch.isfinite(value),'Pilot nonfinite loss')
    value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training']['gradient_clip']);optimizer.step()
    if cfg['model_family']!='team_original_scratch':
        require(frozen_before==state_fingerprint(model.waveform_model.encoder.state_dict()) and not model.waveform_model.encoder.training,
                'Frozen encoder changed/in train mode')
    result=dict(status='PASS',scope='one local microbatch update, NOT formal training/selection',loss=float(value.detach()),
        audit_lock_sha256=sha256(Path(audit_dir)/'protocol.lock.json'),a01_code_sha256=lock['a01_code_sha256'])
    write_json(destination/'pilot.json',result)
    return result


def train(cfg,audit_dir,resume=False):
    validate(cfg)
    lock=check_audit(cfg,audit_dir)
    check_matched_pair(cfg,lock)
    source_config_sha=fingerprint(cfg)
    cfg=copy.deepcopy(cfg);cfg.update(lock['effective_config'])
    device,rank,world=distributed_device()
    output=safe_output(cfg['output_root'],Path(cfg['output_root'])/cfg['run_id'])
    random.seed(cfg['seed']+rank);np.random.seed(cfg['seed']+rank);torch.manual_seed(cfg['seed']+rank)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(cfg['seed']+rank)
    pilot=read_json(Path(audit_dir).parent.parent/'pilot'/cfg['run_id']/'pilot.json')
    require(pilot['status']=='PASS' and pilot['audit_lock_sha256']==sha256(Path(audit_dir)/'protocol.lock.json'), 'Matching pilot required')
    existing=output.exists() and any(output.iterdir())
    if existing and not resume:
        raise FileExistsError('Nonempty FE01 output; explicit resume required')
    if resume and not (output/'last.pth').is_file():
        raise FileNotFoundError('Resume requires last.pth; never falls back to best')
    if rank==0:
        output.mkdir(parents=True,exist_ok=True)
    if world>1:
        dist.barrier()
    identity=fingerprint(dict(config=cfg,audit_lock=lock,world_size=world))
    dataset=TrainingDataset(cfg)
    if cfg.get('audit_limits',{}).get('train_events'):
        dataset.catalog=dataset.catalog.iloc[:cfg['audit_limits']['train_events']]
    batch=cfg['training']['microbatch']
    global_batch=cfg['training']['global_batch']
    if global_batch%(batch*world):
        raise ValueError('microbatch*world must divide the preregistered global batch')
    accumulation=global_batch//(batch*world)
    sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,
                               seed=cfg['sampling_seed'],drop_last=True)
    loader=DataLoader(dataset,batch_size=batch,sampler=sampler,collate_fn=collate,
                      num_workers=cfg['training']['workers'],drop_last=True)
    updates_per_epoch=len(loader)//accumulation
    if updates_per_epoch<1:
        raise ValueError('Cohort too small for effective batch')
    total_updates=updates_per_epoch*cfg['training']['epochs']
    if cfg['training']['max_updates']:
        total_updates=min(total_updates,cfg['training']['max_updates'])
    core=build_model(cfg,device)
    restore_initial(core,cfg,lock)
    model=DistributedDataParallel(core,device_ids=[device.index] if device.type=='cuda' else None,
                                  find_unused_parameters=True) if world>1 else core
    optimizer=torch.optim.Adam([p for p in core.parameters() if p.requires_grad],
                               lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay'])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=total_updates)
    start_epoch=0;updates=0;best=float('inf');committed_journals=[];curves=[];best_epoch=0
    def payload(epoch):
        return dict(epoch=epoch,updates=updates,model_state_dict=core.state_dict(),
                    optimizer=optimizer.state_dict(),scheduler=schedule.state_dict(),
                    identity=identity,best_selection_score=best,
                    world_size=world,a01_code_sha256=lock['a01_code_sha256'],
                    a01_source_commit=lock['source_commit'],
                    absolute_amplitude_mode=cfg['absolute_amplitude_mode'],
                    encoder_state_sha256=state_fingerprint(core.waveform_model.encoder.state_dict()),
                    source_config_sha256=source_config_sha,
                    curve_records=curves,best_epoch=best_epoch,
                    rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                    rank_rng=None,config=cfg,common_initial_state_sha256=initial)
    initial=state_fingerprint(common_state(core))
    if resume:
        checkpoint=torch.load(output/'last.pth',map_location=device,weights_only=False)
        if checkpoint['identity']!=identity:
            raise ValueError('Resume identity differs: config/data/weights/protocol/world')
        core.load_state_dict(checkpoint['model_state_dict'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer']);schedule.load_state_dict(checkpoint['scheduler'])
        start_epoch=checkpoint['epoch'];updates=checkpoint['updates'];best=checkpoint['best_selection_score']
        initial=checkpoint['common_initial_state_sha256']
        committed_journals=checkpoint.get('committed_journals',[])
        curves=checkpoint['curve_records'];best_epoch=checkpoint['best_epoch']
        if rank==0 and best_epoch==start_epoch:
            # Recover a crash between last publication and best publication.
            stored_best=torch.load(output/'best.pth',map_location='cpu',weights_only=False) if (output/'best.pth').is_file() else None
            if stored_best is None or stored_best['epoch']!=best_epoch:
                torch.save(checkpoint,output/'best.pth.tmp');os.replace(output/'best.pth.tmp',output/'best.pth')
        rng=checkpoint.get('rank_rng')
        rng= rng[rank] if rng is not None else checkpoint
        torch.set_rng_state(rng['rng'].cpu());random.setstate(rng['python_rng']);np.random.set_state(rng['numpy_rng'])
        if device.type=='cuda' and rng['cuda_rng']:
            torch.cuda.set_rng_state_all(rng['cuda_rng'])
    elif rank==0:
        write_json(output/'model_interface_audit.json',model_audit(core))
        torch.save(payload(0),output/'init.pth')
    if rank==0:
        write_json(output/'resolved_config.json',cfg)
        if not (output/'runtime.json').exists(): write_json(output/'runtime.json',runtime())
        write_json(output/'protocol.lock.json',lock)
    # Each attempt has its own journal. Only a published epoch checkpoint commits
    # its samples; interrupted attempts remain available as forensic evidence.
    attempt=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')+f'_rank{rank}'
    if rank==0: write_json(output/f'runtime_{attempt}.json',runtime())
    journal=output/'sample_journals';journal.mkdir(exist_ok=True)
    curve_path=output/'training_curves.csv'
    if rank==0 and curves: pd.DataFrame(curves).to_csv(curve_path,index=False)
    for epoch in range(start_epoch,cfg['training']['epochs']):
        if updates>=total_updates:
            break
        dataset.epoch=epoch;sampler.set_epoch(epoch);model.train();optimizer.zero_grad()
        sum_loss=0.;microsteps=0;started=time.perf_counter()
        records_path=journal/f'epoch{epoch+1:04d}_{attempt}.jsonl'
        with records_path.open('w') as stream:
            for micro,part in enumerate(loader):
                if micro>=updates_per_epoch*accumulation or updates>=total_updates:
                    break
                inputs=[x.to(device) for x in part['inputs']];labels=[x.to(device) for x in part['labels']]
                synchronize_grad=(micro+1)%accumulation==0
                context=model.no_sync() if world>1 and not synchronize_grad else contextlib.nullcontext()
                with context:
                    outputs=model(*inputs)
                    value=loss(outputs,labels,core,cfg,inputs[4])
                    if not torch.isfinite(value):
                        raise FloatingPointError('Non-finite FE01 loss')
                    (value/accumulation).backward()
                for info in part['info']:
                    stream.write(json.dumps(dict(epoch=epoch+1,rank=rank,microstep=micro,optimizer_update=updates+1,**info))+'\n')
                sum_loss+=float(value.detach());microsteps+=1
                if synchronize_grad:
                    torch.nn.utils.clip_grad_norm_(core.parameters(),cfg['training']['gradient_clip'])
                    optimizer.step();schedule.step();optimizer.zero_grad();updates+=1
        if world>1:
            dist.barrier()
        selection=None
        if rank==0:
            frame,status=evaluate(core,cfg,device=device,times=cfg['realtime']['common_times'],
                                  event_limit=cfg.get('audit_limits',{}).get('val_events'))
            if not status.status.eq('supported').all():
                raise ValueError('Incomplete common validation support; checkpoint selection refused')
            keys=['dataset_id','event_id','elapsed_time','geometry_protocol','station_id','target_role','input_ids']
            planned=pd.read_csv(Path(audit_dir)/'validation_population.csv.gz',dtype={'event_id':str,'station_id':str})
            if fingerprint(frame[keys].sort_values(keys).to_dict('records'))!=fingerprint(planned[keys].sort_values(keys).to_dict('records')):
                raise ValueError('Validation population differs from preregistered audit')
            selection=checkpoint_selection(frame,cfg['realtime']['common_times'])
            frame.to_csv(output/f'validation_epoch{epoch+1}.csv.gz',index=False)
            improved=selection<best
            if improved:
                best=selection;best_epoch=epoch+1
            row=dict(epoch=epoch+1,updates=updates,train_loss_rank0=sum_loss/max(microsteps,1),
                     validation_selection=selection,lr=optimizer.param_groups[0]['lr'],
                     elapsed_seconds=time.perf_counter()-started,global_batch=global_batch,
                     planned_samples=len(dataset),actual_samples=microsteps*batch*world,
                     dropped_for_equal_update_budget=len(dataset)-microsteps*batch*world)
            curves.append(row)
            write_json(output/f'epoch{epoch+1:04d}_{attempt}.json',row)
        scores=[best,selection]
        if world>1:
            dist.broadcast_object_list(scores,src=0)
        best=scores[0]
        local_rng=dict(rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                       sample_journal=str(records_path.relative_to(output)))
        rank_rng=[None]*world if rank==0 else None
        if world>1:
            dist.gather_object(local_rng,rank_rng,dst=0)
        else:
            rank_rng=[local_rng]
        if rank==0:
            checkpoint=payload(epoch+1);checkpoint['rank_rng']=rank_rng
            checkpoint['committed_journals']=committed_journals+[r['sample_journal'] for r in rank_rng]
            # Atomic publication with existing outputs preserved on interruption.
            torch.save(checkpoint,output/'last.pth.tmp');os.replace(output/'last.pth.tmp',output/'last.pth')
            if improved:
                torch.save(checkpoint,output/'best.pth.tmp');os.replace(output/'best.pth.tmp',output/'best.pth')
            pd.DataFrame(curves).to_csv(curve_path,index=False)
            print(json.dumps(row),flush=True)
            committed_journals=checkpoint['committed_journals']
        if world>1:
            dist.barrier()
    if world>1:
        dist.destroy_process_group()
