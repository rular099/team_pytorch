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


def reuse_equivalence(cfg, original_cfg, initial_state, inputs):
    """Exact legacy/A01 ON initial state, forward, loss and one optimizer update."""
    from fe01.model import build_model as old_build
    from .model import build_model
    from fe01.engine import loss
    new_cfg = copy.deepcopy(cfg); new_cfg['absolute_amplitude_mode'] = ON
    old = old_build(original_cfg, inputs[0].device)
    new = build_model(new_cfg, inputs[0].device)
    old.load_state_dict(initial_state, strict=True); new.load_state_dict(initial_state, strict=True)
    initial_equal = state_fingerprint(old.state_dict()) == state_fingerprint(new.state_dict())
    old.eval(); new.eval()
    with torch.no_grad():
        a, b = old(*inputs), new(*inputs)
    forward_error = max(float((x-y).abs().max()) for x,y in zip(a,b))
    # Labels are synthetic and never used to claim production performance.
    labels = [torch.full((len(inputs[0]),1),4.,device=inputs[0].device),
              torch.zeros((len(inputs[0]),3),device=inputs[0].device),
              torch.zeros((*inputs[4].shape,1),device=inputs[0].device)]
    losses = []; updates = []
    for model, config in ((old,original_cfg),(new,new_cfg)):
        model.train()
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
            lr=config['training']['lr'],weight_decay=config['training']['weight_decay'])
        count = len(config.get('audited_cohorts',{}).get('train',[])) or 1
        budget = training_budget(count,config,1)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1,budget['total_updates']))
        opt.zero_grad(); value = loss(model(*inputs),labels,model,config,inputs[4])
        value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),config['training']['gradient_clip'])
        opt.step(); schedule.step()
        losses.append(float(value.detach())); updates.append(state_fingerprint(model.state_dict()))
    require(initial_equal and forward_error <= 1e-6 and losses[0] == losses[1] and updates[0] == updates[1],
            'Legacy ON / A01 ON equivalence gate failed')
    return dict(status='PASS', initial_equal=True, forward_max_abs=forward_error,
                losses=losses, update_state_equal=True, scope='initial checkpoint + fixed mini-batch; no formal training')
