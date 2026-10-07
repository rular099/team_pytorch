"""Family-aware identity checks. Empty pretrained SHA is legal only for TEAM."""
import copy
import re
from pathlib import Path

import torch

from fe01.config import fingerprint, sha256
from fe01.extractors import checked_asset
from fe01.model import state_fingerprint, common_state
from fe01_review.checkpoint_inventory import inspect_checkpoint
from . import FAMILIES, ON
from .provenance import require, read_json, source_identity, validate


def check_family(cfg, lock):
    family = cfg['model_family']
    require(family in FAMILIES, 'Unknown checkpoint family')
    declared = lock.get('pretrained_manifest_sha256')
    if family == 'team_original_scratch':
        require(declared is None, 'Scratch TEAM has contradictory pretrained identity')
        return dict(family=family, pretrained_manifest_sha256=None, policy='scratch: no pretrained dependency')
    require(isinstance(declared, str) and re.fullmatch(r'[0-9a-f]{64}', declared),
            'Pretrained family requires a legal nonempty SHA')
    require(sha256(cfg['pretrained_manifest']) == declared, 'Pretrained manifest SHA differs')
    asset = {'phasenet_pretrained_frozen': 'phasenet', 'eqt_pretrained_frozen': 'eqtransformer',
             'diting_pretrained_frozen': 'diting'}[family]
    entry, files = checked_asset(cfg['pretrained_manifest'], asset)
    return dict(family=family, pretrained_manifest_sha256=declared,
                encoder_assets_sha256={k: sha256(p) for k, p in files.items()})


def old_checkpoint(path, epoch, run_dir, runtime_cfg):
    """Authenticate original bytes/config/lock BEFORE using relocated runtime paths."""
    path, run_dir = Path(path), Path(run_dir)
    pins=runtime_cfg.get('a01',{}).get('original_checkpoint_pins')
    if pins is not None:
        role='init' if epoch==0 else ('last' if path.name=='last.pth' else 'selected')
        require(sha256(path)==pins[role]['file_sha256'],'Original checkpoint bytes differ from frozen inventory')
    result = inspect_checkpoint(path, epoch, 'fe01', run_dir/'resolved_config.json', run_dir)
    require(result['status'] == 'IDENTITY_PASS', result['reason'])
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    recorded = checkpoint['config']
    require(recorded['model_family'] == runtime_cfg['model_family'], 'Wrong encoder/family')
    require(recorded['model_params'] == runtime_cfg['model_params'], 'Model config differs from original')
    family = check_family(runtime_cfg, read_json(run_dir/'protocol.lock.json'))
    return checkpoint, dict(**result, dependency=family)


def a01_checkpoint(path, epoch, cfg, audit_dir, device='cpu'):
    from .model import build_model
    from .engine import check_audit
    lock = check_audit(cfg, audit_dir)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    require(checkpoint.get('epoch') == epoch, 'Wrong checkpoint epoch')
    require(checkpoint.get('source_config_sha256') == fingerprint(cfg), 'Wrong checkpoint config')
    effective = copy.deepcopy(cfg); effective.update(lock['effective_config'])
    require(checkpoint['config'] == effective, 'Resolved checkpoint config differs')
    require(checkpoint.get('a01_code_sha256') == source_identity()['a01_code_sha256'], 'Wrong A01 training source')
    require(checkpoint.get('absolute_amplitude_mode') == effective['absolute_amplitude_mode'], 'Amplitude mode differs')
    world = checkpoint.get('world_size')
    require(checkpoint['identity'] == fingerprint(dict(config=effective, audit_lock=lock, world_size=world)),
            'Checkpoint protocol identity mismatch')
    check_family(effective, lock)
    model = build_model(effective, device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if effective['model_family'] != 'team_original_scratch':
        actual=state_fingerprint(model.waveform_model.encoder.state_dict())
        require(actual == checkpoint['encoder_state_sha256'] == lock['initial_state']['encoder_sha256'],
                'Wrong frozen encoder state')
    model.eval()
    return model, checkpoint


def training_budget(events, cfg, world):
    from torch.utils.data import DistributedSampler, DataLoader
    from fe01.data import collate
    samples = events * cfg['realtime']['draws_per_event']
    micro, global_batch = cfg['training']['microbatch'], cfg['training']['global_batch']
    require(global_batch % (micro*world) == 0, 'Global batch not divisible by microbatch*world')
    sampler = DistributedSampler(range(samples), num_replicas=world, rank=0, drop_last=True,
                                 shuffle=True, seed=cfg['sampling_seed'])
    # Use the actual PyTorch sampler/loader length rules, without reading data.
    loader = DataLoader(range(samples), batch_size=micro, sampler=sampler, drop_last=True)
    accumulation = global_batch // (micro*world)
    updates = len(loader)//accumulation
    return dict(events=events, samples_per_epoch=samples, world_size=world,
                sampler_per_rank=len(sampler), microsteps_per_rank=len(loader), accumulation=accumulation,
                updates_per_epoch=updates, total_updates=updates*cfg['training']['epochs'],
                consumed_samples_per_epoch=updates*global_batch,
                dropped_samples_per_epoch=samples-updates*global_batch)


def restore_initial(model, cfg, lock):
    path = Path(lock['initial_state']['path'])
    require(sha256(path) == lock['initial_state']['file_sha256'], 'Initial checkpoint changed')
    state = torch.load(path, map_location='cpu', weights_only=False)['model_state_dict']
    model.load_state_dict(state, strict=True)
    require(state_fingerprint(state) == lock['initial_state']['state_sha256'], 'Initial state content differs')
    require(state_fingerprint(common_state(model)) == lock['initial_state']['common_sha256'], 'Common init differs')


def reuse_equivalence(cfg, original_cfg, initial_state, inputs, report_path=None):
    """Exact init plus paired numerical forward/gradient/Adam-update equivalence."""
    from .equivalence import paired_execution
    with paired_execution(inputs[0].device) as devices:
        return _reuse_equivalence(cfg, original_cfg, initial_state, inputs, devices, report_path)


def canonical_reuse_equivalence(cfg, original_cfg, initial_state, inputs, report_path=None):
    """Compare the real epoch0 model/update exactly on one CPU thread.

    DCU forward compatibility is checked independently by engine.audit. This
    isolates implementation equivalence from backend reduction/Adam roundoff.
    The formal training device, optimizer, gradient clipping and DDP are intact.
    """
    from .provenance import write_json
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        result = reuse_equivalence(cfg, original_cfg, initial_state,
                                   [value.detach().cpu() for value in inputs], report_path)
        exact = (result['initial_equal'] and result['update_state_equal'] and
                 result['losses'][0] == result['losses'][1] and
                 all(check['max_abs'] == 0. for check in result['checks'].values()))
        result['canonical_reference'] = dict(device='cpu', threads=1, exact_state_update=exact,
            policy='same actual initial weights/input; exact CPU forward/gradient/Adam update; separate DCU forward gate')
        if not exact:
            result['status'] = 'FAIL'
            result['failed_checks'].append('canonical_CPU_exact_equivalence')
        if report_path is not None:
            write_json(report_path, result)
        require(exact, 'Canonical CPU ON equivalence failed; details: '+str(report_path))
        return result
    finally:
        torch.set_num_threads(previous_threads)


def _reuse_equivalence(cfg, original_cfg, initial_state, inputs, devices, report_path):
    from fe01.model import build_model as old_build
    from .model import build_model
    from fe01.engine import loss
    from .equivalence import compare_tensors, outputs_comparison, capture_rng, restore_rng, rng_identity
    from .provenance import write_json
    require(cfg['training'] == original_cfg['training'], 'ON equivalence training config differs')
    new_cfg = copy.deepcopy(cfg); new_cfg['absolute_amplitude_mode'] = ON
    old = old_build(original_cfg, inputs[0].device)
    new = build_model(new_cfg, inputs[0].device)
    old.load_state_dict(initial_state, strict=True); new.load_state_dict(initial_state, strict=True)
    initial_pins = dict(requested=state_fingerprint(initial_state),
                        legacy=state_fingerprint(old.state_dict()), a01=state_fingerprint(new.state_dict()))
    initial_equal = len(set(initial_pins.values())) == 1
    old.eval(); new.eval()
    eval_rng = capture_rng(devices)
    with torch.no_grad():
        restore_rng(eval_rng); a = old(*inputs)
        restore_rng(eval_rng); b = new(*inputs)
    forward = outputs_comparison(a, b)
    # Labels are synthetic and never used to claim production performance.
    labels = [torch.full((len(inputs[0]),1),4.,device=inputs[0].device),
              torch.zeros((len(inputs[0]),3),device=inputs[0].device),
              torch.zeros((*inputs[4].shape,1),device=inputs[0].device)]
    losses = []; updates = []; gradients = []; states = []; training_outputs = []; optimizers = []
    train_rng = capture_rng(devices)
    for model, config in ((old,original_cfg),(new,new_cfg)):
        restore_rng(train_rng)
        model.train()
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
            lr=config['training']['lr'],weight_decay=config['training']['weight_decay'])
        count = len(config.get('audited_cohorts',{}).get('train',[])) or 1
        budget = training_budget(count,config,1)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1,budget['total_updates']))
        opt.zero_grad(); output = model(*inputs)
        training_outputs.append({str(i): x.detach().cpu().clone() for i,x in enumerate(output)})
        value = loss(output,labels,model,config,inputs[4])
        value.backward()
        gradients.append({name: None if p.grad is None else p.grad.detach().cpu().clone()
                          for name,p in model.named_parameters() if p.requires_grad})
        torch.nn.utils.clip_grad_norm_(model.parameters(),config['training']['gradient_clip'])
        opt.step(); schedule.step()
        losses.append(float(value.detach())); updates.append(state_fingerprint(model.state_dict()))
        states.append({name: value.detach().cpu().clone() for name,value in model.state_dict().items()})
        optimizers.append(dict(lr=opt.param_groups[0]['lr'], scheduler_last_epoch=schedule.last_epoch,
                               scheduler_T_max=schedule.T_max))
    checks = dict(eval_forward=forward, train_forward=compare_tensors(*training_outputs),
                  loss=compare_tensors({'loss':torch.tensor(losses[0],dtype=torch.float64)},
                                       {'loss':torch.tensor(losses[1],dtype=torch.float64)}),
                  gradients=compare_tensors(*gradients), updated_state=compare_tensors(*states))
    failures = [name for name, check in checks.items() if not check['passed']]
    if not initial_equal: failures.append('initial_state_SHA')
    if optimizers[0] != optimizers[1]: failures.append('optimizer_scheduler')
    result = dict(status='FAIL' if failures else 'PASS', initial_equal=initial_equal,
                  initial_state_sha256=initial_pins, forward_max_abs=forward['max_abs'], losses=losses,
                  update_state_equal=updates[0] == updates[1], update_state_sha256=updates,
                  update_numerically_equivalent=checks['updated_state']['passed'], checks=checks,
                  failed_checks=failures, optimizer_scheduler=optimizers,
                  execution=dict(device=str(inputs[0].device), torch=torch.__version__, rocm=torch.version.hip,
                                 rng_replayed=True, train_rng_sha256=rng_identity(train_rng),
                                 cudnn_deterministic=True, cudnn_benchmark=False, tf32=False,
                                 controls_scope='audit only; RNG and backend settings restored on exit'),
                  scope='initial checkpoint + fixed mini-batch; no formal training')
    if report_path is not None:
        write_json(report_path, result)
    require(not failures, 'Legacy ON / A01 ON equivalence gate failed: '+', '.join(failures)+
            ('; details: '+str(report_path) if report_path is not None else ''))
    return result
