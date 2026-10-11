"""Static A01 loop copy with CPU RNG restoration and startup observations.

Source delta is checked against the immutable engine before execution. Import
bindings are checked by identity; no original module or torch API is replaced.
The loop's SHA is recorded separately from the authenticated core source SHA.
"""
from fe01_a01.engine import (
    contextlib, copy, datetime, json, os, Path, random, time, np, pd, torch,
    dist, DistributedDataParallel, DataLoader, DistributedSampler, validate,
    check_audit, check_matched_pair, fingerprint, distributed_device, safe_output,
    read_json, require, sha256, TrainingDataset, collate, build_model,
    restore_initial, state_fingerprint, common_state, write_json, model_audit,
    runtime, evaluate, loss, checkpoint_selection,
)


def train(cfg,audit_dir,resume=False,*,observe):
    observe('validate.begin')
    validate(cfg)
    lock=check_audit(cfg,audit_dir)
    check_matched_pair(cfg,lock)
    source_config_sha=fingerprint(cfg)
    cfg=copy.deepcopy(cfg);cfg.update(lock['effective_config'])
    observe('validate.complete')
    observe('distributed_init.begin')
    device,rank,world=distributed_device()
    observe('distributed_init.complete')
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
    observe('initial_barrier.begin')
    if world>1:
        dist.barrier()
    observe('initial_barrier.complete')
    identity=fingerprint(dict(config=cfg,audit_lock=lock,world_size=world))
    observe('dataset.begin')
    dataset=TrainingDataset(cfg)
    observe('dataset.complete')
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
    observe('model_build.begin')
    core=build_model(cfg,device)
    observe('model_build.complete')
    observe('audited_initial_state.begin')
    restore_initial(core,cfg,lock)
    observe('audited_initial_state.complete')
    observe('ddp_constructor.begin')
    model=DistributedDataParallel(core,device_ids=[device.index] if device.type=='cuda' else None,
                                  find_unused_parameters=True) if world>1 else core
    observe('ddp_constructor.complete')
    observe('optimizer.begin')
    optimizer=torch.optim.Adam([p for p in core.parameters() if p.requires_grad],
                               lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay'])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=total_updates)
    observe('optimizer.complete')
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
        observe('checkpoint_load.begin')
        checkpoint=torch.load(output/'last.pth',map_location=device,weights_only=False)
        observe('checkpoint_load.complete')
        if checkpoint['identity']!=identity:
            raise ValueError('Resume identity differs: config/data/weights/protocol/world')
        observe('checkpoint_model_restore.begin')
        core.load_state_dict(checkpoint['model_state_dict'],strict=True)
        observe('checkpoint_model_restore.complete')
        observe('optimizer_restore.begin')
        optimizer.load_state_dict(checkpoint['optimizer']);schedule.load_state_dict(checkpoint['scheduler'])
        observe('optimizer_restore.complete')
        start_epoch=checkpoint['epoch'];updates=checkpoint['updates'];best=checkpoint['best_selection_score']
        initial=checkpoint['common_initial_state_sha256']
        committed_journals=checkpoint.get('committed_journals',[])
        curves=checkpoint['curve_records'];best_epoch=checkpoint['best_epoch']
        if rank==0 and best_epoch==start_epoch:
            # Recover a crash between last publication and best publication.
            stored_best=torch.load(output/'best.pth',map_location='cpu',weights_only=False) if (output/'best.pth').is_file() else None
            if stored_best is None or stored_best['epoch']!=best_epoch:
                torch.save(checkpoint,output/'best.pth.tmp');os.replace(output/'best.pth.tmp',output/'best.pth')
        observe('rng_restore.begin')
        rng=checkpoint.get('rank_rng')
        rng= rng[rank] if rng is not None else checkpoint
        torch.set_rng_state(rng['rng'].cpu());random.setstate(rng['python_rng']);np.random.set_state(rng['numpy_rng'])
        if device.type=='cuda' and rng['cuda_rng']:
            torch.cuda.set_rng_state_all([state.cpu() for state in rng['cuda_rng']])
        observe('rng_restore.complete')
    elif rank==0:
        observe('init_publish.begin')
        write_json(output/'model_interface_audit.json',model_audit(core))
        torch.save(payload(0),output/'init.pth')
        observe('init_publish.complete')
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
    observe('training_loop.enter')
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
                if micro == 0 and epoch == start_epoch:
                    observe('first_batch.ready')
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
                    if micro == 0 and epoch == start_epoch:
                        observe('first_update.complete')
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
    observe('training.complete')
