"""FE02 sampling/selection, small task-only checkpoints, strict compute-node gates."""
import contextlib
import copy
import datetime
import gzip
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .config import fingerprint, sha256, write_json, code_identity, validate
from .data import audit_data, catalog, read_event, prepare, TrainingDataset, collate
from .model import build_model, parameter_audit
from .losses import loss_terms
from .artifacts import (atomic_save, state_digest, rng_state, restore_rng, runtime)
from .evaluation import to_device, evaluate, export_diagnostics


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def gradient_modules(model):
    groups = {}
    for name,p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = name.split('.')[0]
        groups[key] = groups.get(key,0.0) + (float(p.grad.detach().float().square().sum()) if p.grad is not None else 0)
    return {name:math.sqrt(value) for name,value in groups.items()}


def audit_model(model, cfg, sample, device):
    """One bounded update, then restore task state/RNG/mode exactly."""
    original = {n:t.detach().clone() for n,t in model.task_state().items()}
    rng,mode = rng_state(),model.training
    initial = state_digest(model.encoder.state_dict())
    inputs = to_device(sample,device)
    # Audit only at most two active stations, independent of full production budget.
    inputs[2][:,2:] = False
    inputs[5][:,2:] = False
    labels = [x[None].to(device) for x in sample['labels']]
    try:
        model.eval()
        with torch.no_grad():
            ordinary = model(*inputs)
            recorded,tensors = model(*inputs,diagnostics=True)
            if not all(torch.equal(a,b) for a,b in zip(ordinary,recorded)):
                raise ValueError('Diagnostics changed outputs')
            future = [x.clone() for x in inputs]
            future[0] = torch.where(inputs[5][:,:,None],inputs[0],torch.full_like(inputs[0],float('nan')))
            changed = model(*future)
            if not all(torch.equal(a,b) for a,b in zip(ordinary,changed)):
                raise ValueError('Future/invalid samples affect outputs')
            n = inputs[3].shape[1]
            encoded = model.encode_observations(inputs[0],inputs[1],inputs[2],inputs[5])
            chunks = torch.cat([model.read_targets(encoded,inputs[3][:,i:i+1])[0] for i in range(n)],1)
            if not torch.allclose(chunks,ordinary[2],atol=2e-5,rtol=2e-5):
                raise ValueError('Query chunk dependence')
            permuted = [x.flip(1) if i in (0,1,2,5) else x for i,x in enumerate(inputs)]
            if not torch.allclose(model(*permuted)[2],ordinary[2],atol=2e-5,rtol=2e-5):
                raise ValueError('Station permutation changes predictions')
        model.train()
        if model.encoder.training or any(p.requires_grad for p in model.encoder.parameters()):
            raise ValueError('Encoder is not frozen/eval')
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['training']['lr'])
        value,terms = loss_terms(model(*inputs),labels,inputs[4],cfg)
        if not torch.isfinite(value):
            raise FloatingPointError('Non-finite audit loss')
        value.backward()
        gradients = gradient_modules(model)
        required = ['projection','amplitude','pool_queries','pool_key_norm','coordinates','joint','event',
                    'spatial','target_query_norm','target_key_norm','target_attention','pga']
        if cfg['loss']['location']:
            required += ['location']
        if cfg['loss']['magnitude']:
            required += ['magnitude']
        if any(not math.isfinite(gradients[n]) or gradients[n]<=0 for n in required):
            raise ValueError('Missing/nonfinite required gradient: '+str(gradients))
        optimizer.step()
        if any(p.grad is not None for p in model.encoder.parameters()) or state_digest(model.encoder.state_dict()) != initial:
            raise ValueError('Frozen encoder parameter/buffer mutation')
        return dict(status='REAL_MODEL_AUDIT_PASS',parameter_audit=parameter_audit(model),
            encoder_identity=model.encoder_identity,encoder_state_sha256=initial,
            module_gradient_norms=gradients,loss_terms={k:float(v.detach()) for k,v in terms.items()},
            checks=['strict keys/shapes','real native shape','patch start/stride/padding/order',
                    'causal invalid mask','encoder parameters and buffers frozen','task gradients',
                    'query chunks','station permutation','diagnostics output equality'],
            audit_input_stations=int(inputs[2].sum()),audit_not_charged_to_training=True)
    finally:
        model.load_task_state(original)
        model.zero_grad(set_to_none=True)
        model.train(mode)
        restore_rng(rng)
        if state_digest(model.task_state()) != state_digest(original):
            raise ValueError('Audit failed to restore initialization')


def audit(cfg, directory, device):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    data = audit_data(cfg,directory/'data')
    effective = copy.deepcopy(cfg)
    effective.update(data['effective_config'])
    model = build_model(effective,device)
    row = catalog(effective,'train').iloc[0]
    event = read_event(row,effective,3)
    sample = prepare(event,effective,3,'normal')
    report = audit_model(model,effective,sample,device)
    write_json(directory/'model_audit.json',report)
    lock = dict(status='AUDIT_PASS',config_sha256=fingerprint(cfg),code_sha256=code_identity(),
        data=data,encoder_identity=model.encoder_identity,task_initial_sha256=state_digest(model.task_state()),
        effective_config=data['effective_config'],runtime=runtime())
    write_json(directory/'protocol.lock.json',lock)
    return lock


def check_lock(cfg, directory):
    lock = json.loads((Path(directory)/'protocol.lock.json').read_text())
    if lock['status'] != 'AUDIT_PASS' or lock['config_sha256'] != fingerprint(cfg):
        raise ValueError('Audit/config identity mismatch')
    if lock['code_sha256'] != code_identity():
        raise ValueError('Audited source/dependency changed')
    if sha256(cfg['data']['split_manifest']) != lock['data']['split_manifest']['sha256']:
        raise ValueError('Frozen split changed')
    for path,record in lock['data']['shards'].items():
        stat = Path(path).stat()
        if stat.st_size != record['bytes'] or stat.st_mtime_ns != record['mtime_ns']:
            raise ValueError('Audited shard changed: '+path)
    return lock


def distributed_device():
    world,rank,local = (int(os.environ.get(k,d)) for k,d in [('WORLD_SIZE','1'),('RANK','0'),('LOCAL_RANK','0')])
    device = torch.device('cuda',local) if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world>1:
        dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',init_method='env://')
    return device,rank,world


def train(cfg, audit_dir, resume=False, model_factory=build_model):
    validate(cfg)
    lock = check_lock(cfg,audit_dir)
    cfg = copy.deepcopy(cfg)
    cfg.update(lock['effective_config'])
    device,rank,world = distributed_device()
    output = Path(cfg['output_root'])/cfg['run_id']
    if (output/'COMPLETED.json').exists():
        raise FileExistsError('Completed run is immutable; evaluation has a separate entry')
    if output.exists() and not resume:
        raise FileExistsError('Preserving existing run; use explicit resume from last')
    if resume and not (output/'last.pth').is_file():
        raise FileNotFoundError('Resume requires last.pth; never best')
    if rank==0:
        output.mkdir(parents=True,exist_ok=resume)
    if world>1: dist.barrier()
    random.seed(cfg['seed']+rank)
    np.random.seed(cfg['seed']+rank)
    torch.manual_seed(cfg['seed']+rank)
    if device.type=='cuda': torch.cuda.manual_seed_all(cfg['seed']+rank)
    dataset = TrainingDataset(cfg)
    t = cfg['training']
    if t['global_batch'] % (t['microbatch']*world):
        raise ValueError('microbatch*world must divide effective batch')
    accumulation = t['global_batch']//(t['microbatch']*world)
    sampler = DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,seed=cfg['sampling_seed'],drop_last=True)
    loader = DataLoader(dataset,batch_size=t['microbatch'],sampler=sampler,collate_fn=collate,
        num_workers=t['workers'],drop_last=True)
    steps_epoch = len(loader)//accumulation
    if steps_epoch<1: raise ValueError('Cohort too small for one effective batch')
    total_steps = steps_epoch*t['epochs']
    if t['max_updates']: total_steps = min(total_steps,t['max_updates'])
    core = model_factory(cfg,device)
    if core.encoder_identity != lock['encoder_identity']:
        raise ValueError('Encoder/YAML identity differs from audit')
    if state_digest(core.task_state()) != lock['task_initial_sha256']:
        raise ValueError('Paired initialization differs from audit')
    model = DistributedDataParallel(core,device_ids=[device.index] if device.type=='cuda' else None,
        broadcast_buffers=False,find_unused_parameters=True) if world>1 else core
    parameters = [(n,p) for n,p in core.named_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam([p for _,p in parameters],lr=t['lr'],weight_decay=t['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=total_steps)
    resume_environment = dict(torch=torch.__version__,numpy=np.__version__,python=sys.version,
                              hip=torch.version.hip,device_type=device.type)
    identity = fingerprint(dict(config=cfg,lock=lock,world=world,accumulation=accumulation,
                                total_steps=total_steps,environment=resume_environment))
    artifact_identity = dict(data_shards_sha256=fingerprint(lock['data']['shards']),
        split_manifest_sha256=lock['data']['split_manifest']['sha256'],
        view_manifest_sha256=lock['data']['view_manifest_sha256'],cohort_sha256=lock['data']['cohort_sha256'],
        encoder_checkpoint_sha256=core.encoder_identity.get('checkpoint_sha256','SYNTHETIC'),
        encoder_yaml_sha256=core.encoder_identity.get('yaml_sha256','SYNTHETIC'),code_sha256=lock['code_sha256'])
    start_epoch,step,best,best_epoch = 0,0,float('inf'),0
    committed,curves = [],[]
    if resume:
        checkpoint = torch.load(output/'last.pth',map_location='cpu',weights_only=False)
        if checkpoint['identity'] != identity:
            raise ValueError('Resume identity changed: config/data/source/encoder/world/budget')
        core.load_task_state(checkpoint['task_state'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch,step = checkpoint['epoch'],checkpoint['step']
        best,best_epoch = checkpoint['best'],checkpoint['best_epoch']
        committed,curves = checkpoint['committed_files'],checkpoint['curves']
        for record in committed:
            if sha256(output/record['path']) != record['sha256']:
                raise ValueError('Committed journal/log is missing or changed')
        # If last was published just before a crash, finish publishing best from it.
        if rank==0 and best_epoch == start_epoch:
            stored = torch.load(output/'best.pth',map_location='cpu',weights_only=False) if (output/'best.pth').is_file() else None
            if stored is None or stored['task_state_sha256'] != checkpoint['task_state_sha256'] or stored['epoch'] != start_epoch:
                atomic_save(output/'best.pth',checkpoint)
        restore_rng(checkpoint['rank_rng'][rank])
    attempt = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    if rank==0:
        write_json(output/f'runtime_{attempt}.json',dict(**runtime(),global_batch=t['global_batch'],
            microbatch=t['microbatch'],gradient_accumulation=accumulation,steps_per_epoch=steps_epoch,total_steps=total_steps))
        write_json(output/'resolved_config.json',cfg)
        write_json(output/'protocol.lock.json',lock)
        write_json(output/'parameters.json',parameter_audit(core))
    def head(epoch):
        return dict(task_state={n:v.detach().cpu() for n,v in core.task_state().items()},
            epoch=epoch,step=step,identity=identity,encoder_identity=core.encoder_identity,config=cfg,
            task_state_sha256=state_digest(core.task_state()),kind='task_head_only',artifact_identity=artifact_identity)
    if not resume and rank==0:
        atomic_save(output/'init.pth',head(0))
        export_diagnostics(core,cfg,Path(audit_dir)/'data',output/'diagnostics','init',
            dict(checkpoint_sha256=sha256(output/'init.pth'),run_id=cfg['run_id']),device)
    if world>1: dist.barrier()
    for epoch in range(start_epoch,t['epochs']):
        if step>=total_steps: break
        dataset.epoch = epoch
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        journal_dir,log_dir = output/'sample_journals',output/'step_logs'
        journal_dir.mkdir(exist_ok=True)
        log_dir.mkdir(exist_ok=True)
        stem = f'epoch{epoch+1:04d}_{attempt}_rank{rank}'
        journal_path,log_path = journal_dir/(stem+'.jsonl.gz'),log_dir/(stem+'.jsonl.gz')
        # Every attempt is retained. Only files linked by last are committed.
        samples_seen,targets_seen,station_sum,loss_sum,micro_count = 0,0,0,0.0,0
        term_sums = {}
        pending = {}
        last_end = time.perf_counter()
        with gzip.open(journal_path,'wt') as journal,gzip.open(log_path,'wt') as logs:
            for micro,part in enumerate(loader):
                if micro>=steps_epoch*accumulation or step>=total_steps: break
                data_seconds = time.perf_counter()-last_end
                batch_start = time.perf_counter()
                inputs = [x.to(device) for x in part['inputs']]
                labels = [x.to(device) for x in part['labels']]
                update = (micro+1)%accumulation==0
                context = model.no_sync() if world>1 and not update else contextlib.nullcontext()
                with context:
                    value,terms = loss_terms(model(*inputs),labels,inputs[4],cfg)
                    if not torch.isfinite(value):
                        logs.write(json.dumps(dict(epoch=epoch+1,step=step+1,status='nonfinite_loss'))+'\n')
                        raise FloatingPointError('Non-finite training loss')
                    (value/accumulation).backward()
                for info in part['info']:
                    journal.write(json.dumps(dict(epoch=epoch+1,step=step+1,micro=micro,rank=rank,**info))+'\n')
                for name,v in terms.items():
                    scalar = float(v.detach())
                    term_sums[name] = term_sums.get(name,0)+scalar
                    pending[name] = pending.get(name,0)+scalar/accumulation
                samples_seen += len(part['info'])
                targets_seen += int(inputs[4].sum())
                station_sum += int(inputs[2].sum())
                loss_sum += float(value.detach())
                micro_count += 1
                if update:
                    measure_start = time.perf_counter()
                    measure = (step+1)%cfg['logging']['module_frequency']==0 or step==0
                    gradients = gradient_modules(core) if measure else None
                    grad_norm = float(torch.nn.utils.clip_grad_norm_([p for _,p in parameters],t['gradient_clip']))
                    if not math.isfinite(grad_norm): raise FloatingPointError('Non-finite gradient')
                    before = {n:p.detach().clone() for n,p in parameters} if measure else None
                    measurement_seconds = time.perf_counter()-measure_start
                    lr_used = optimizer.param_groups[0]['lr']
                    optimizer.step()
                    scheduler.step()
                    step += 1
                    updates = {}
                    if measure:
                        measure_start = time.perf_counter()
                        for name,p in parameters:
                            key = name.split('.')[0]
                            updates[key] = updates.get(key,0)+float((p.detach()-before[name]).square().sum())
                        updates = {k:math.sqrt(v) for k,v in updates.items()}
                        measurement_seconds += time.perf_counter()-measure_start
                    sync(device)
                    record = dict(epoch=epoch+1,step=step,rank=rank,lr=lr_used,scope='this rank; accumulated update',
                        losses=pending,gradient_norm_before_clip=grad_norm,gradient_clip=t['gradient_clip'],
                        module_gradient_norms=gradients,module_update_norms=updates,
                        data_seconds_last_micro=data_seconds,compute_seconds_last_micro=time.perf_counter()-batch_start,
                        module_measurement_seconds=measurement_seconds,
                        cumulative_event_instances=samples_seen,cumulative_targets=targets_seen,
                        last_micro_elapsed_times=[i['actual_elapsed_time'] for i in part['info']],
                        peak_device_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                        status='finite')
                    logs.write(json.dumps(record)+'\n')
                    pending = {}
                    optimizer.zero_grad(set_to_none=True)
                last_end = time.perf_counter()
        sync(device)
        term_names = sorted(term_sums)
        totals = torch.tensor([samples_seen,targets_seen,station_sum,loss_sum,micro_count]+
                              [term_sums[k] for k in term_names],dtype=torch.float64,device=device)
        if world>1:
            dist.all_reduce(totals)
            dist.barrier()
        training_seconds = time.perf_counter()-started
        improved = False
        if rank==0:
            epoch_path = output/f'epoch{epoch+1:04d}.pth'
            # Preserve uncommitted failed epoch snapshots under their attempt name.
            if epoch_path.exists(): epoch_path.rename(output/f'epoch{epoch+1:04d}_orphan_{attempt}.pth')
            atomic_save(epoch_path,head(epoch+1))
            validation_dir = output/'validation'/f'epoch{epoch+1:04d}_{attempt}'
            selection,frame = evaluate(core,cfg,validation_dir,dict(**artifact_identity,checkpoint_epoch=epoch+1,checkpoint_step=step,
                checkpoint_sha256=sha256(epoch_path),checkpoint_task_sha256=state_digest(core.task_state())),device,
                times=cfg['realtime']['common_times'])
            keys = ['dataset_id','event_id','elapsed_time','geometry_protocol','station_id','target_role','input_ids']
            planned = pd.read_csv(Path(audit_dir)/'data/validation_population.csv.gz',dtype={'event_id':str,'station_id':str})
            # CSV integer 1 and float 1.0 are canonicalized explicitly.
            for table in (planned,frame): table['elapsed_time'] = table.elapsed_time.astype(float)
            if fingerprint(frame[keys].sort_values(keys).to_dict('records')) != fingerprint(planned[keys].sort_values(keys).to_dict('records')):
                raise ValueError('Validation target/input population drift')
            improved = selection<best
            if improved: best,best_epoch = selection,epoch+1
            record = dict(epoch=epoch+1,step=step,selection=selection,best_epoch=best_epoch,
                training_seconds=training_seconds,validation_seconds=time.perf_counter()-started-training_seconds,
                global_event_instances=int(totals[0]),global_targets=int(totals[1]),
                mean_stations=float(totals[2]/totals[0]),train_loss=float(totals[3]/totals[4]),
                planned_event_instances=len(dataset),dropped_instances=len(dataset)-int(totals[0]),
                scope='all ranks full epoch',validation_path=str(validation_dir.relative_to(output)))
            record.update({'train_'+name:float(totals[5+i]/totals[4]) for i,name in enumerate(term_names)})
            curves.append(record)
            print(json.dumps(record),flush=True)
        local = dict(rng=rng_state(),files=[dict(path=str(p.relative_to(output)),sha256=sha256(p)) for p in (journal_path,log_path)])
        rank_states = [None]*world if rank==0 else None
        if world>1:
            dist.gather_object(local,rank_states,dst=0)
        else: rank_states = [local]
        if rank==0:
            committed += [f for r in rank_states for f in r['files']]
            checkpoint = dict(head(epoch+1),kind='full_resume',optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),amp=None,rank_rng=[r['rng'] for r in rank_states],
                committed_files=committed,curves=curves,best=best,best_epoch=best_epoch,
                sampling_position=dict(next_epoch=epoch+1,next_micro=0,epoch_boundary_resume=True))
            atomic_save(output/'last.pth',checkpoint)
            if improved: atomic_save(output/'best.pth',checkpoint)
            pd.DataFrame(curves).to_csv(output/'training_curves.csv',index=False)
        if world>1: dist.barrier()
    if rank==0:
        for tag in ('best','last'):
            checkpoint = torch.load(output/(tag+'.pth'),map_location='cpu',weights_only=False)
            core.load_task_state(checkpoint['task_state'])
            export_diagnostics(core,cfg,Path(audit_dir)/'data',output/'diagnostics',tag,
                dict(checkpoint_sha256=sha256(output/(tag+'.pth')),checkpoint_epoch=checkpoint['epoch'],
                     checkpoint_step=checkpoint['step'],run_id=cfg['run_id']),device)
        write_json(output/'TRAINING_COMPLETE.json',dict(step=step,epochs=t['epochs'],best_epoch=best_epoch,
            identity=identity,committed_files=committed))
    if world>1:
        dist.barrier()
        dist.destroy_process_group()
    return output


def load_for_evaluation(cfg, audit_dir, checkpoint_path, device):
    lock = check_lock(cfg,audit_dir)
    checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    effective = copy.deepcopy(cfg)
    effective.update(lock['effective_config'])
    if checkpoint['config'] != effective or checkpoint['encoder_identity'] != lock['encoder_identity']:
        raise ValueError('Evaluation checkpoint identity mismatch')
    model = build_model(effective,device)
    if model.encoder_identity != checkpoint['encoder_identity']:
        raise ValueError('Pretrained encoder identity changed')
    model.load_task_state(checkpoint['task_state'])
    return model,checkpoint,effective
