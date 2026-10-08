"""Use the actual FE01 strict-loaded DiTing frontend, with FE02 readout auditing."""
import inspect

import torch
from torch import nn

import gemini_models as legacy
from fe01.config import sha256
from fe01.model import build_model as build_base, model_audit as audit_base, state_fingerprint
from .config import validate


def build_model(cfg, device='cpu'):
    validate(cfg)
    # Legacy factory accepts **kwargs; reject typos instead of its warning fallback.
    accepted = set(inspect.signature(legacy.build_transformer_model).parameters) - {'kwargs'}
    unknown = set(cfg['model_params']) - accepted
    if unknown:
        raise ValueError('Unknown factory arguments: ' + ', '.join(sorted(unknown)))
    model = build_base(cfg, device)
    # Match every unchanged downstream tensor at a paired seed, even when an
    # extra PGA layer consumed different RNG draws during construction.
    from fe01.model import FE01FullModel
    reference_params = dict(cfg['model_params'])
    reference_params.update(pga_readout_mode='target_cross_attention', pga_use_event_context=False,
                            pga_event_memory=False, pga_output_mlp_dims=None)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg['seed'])
        reference = legacy.build_transformer_model(**reference_params, trace_length=10000,
            station_waveform_model=model.waveform_model, full_model_class=FE01FullModel)
    state = model.state_dict()
    for name, tensor in reference.state_dict().items():
        if (not name.startswith(('waveform_model.', 'mlp_pga.'))
                and name in state and state[name].shape == tensor.shape):
            state[name] = tensor.to(device)
    model.load_state_dict(state, strict=True)
    del reference, state
    model._fe02_variant_id = cfg['variant_id']
    return model


def model_audit(model, cfg):
    from train_light import build_diting_args
    from fe01.extractors import checked_asset
    args = build_diting_args(cfg['diting_config'], device='cpu', pretrained_override='')
    entry, files = checked_asset(cfg['pretrained_manifest'], 'diting')
    if (entry.get('native_n_samples') != 10000 or entry.get('sampling_rate') != 100
            or entry.get('component_order') != 'NEZ'
            or entry['files']['weights'].get('bytes') != files['weights'].stat().st_size):
        raise ValueError('Registered DiTing native shape/component order/bytes mismatch')
    if (args.diting_frontend != 'backbone_attn_pool' or args.target_width != 1792
            or args.model_depth != 24 or args.in_samples != 10000 or args.patch_size != 50):
        raise ValueError('FE02 requires the fixed real 1200M backbone_attn_pool YAML')
    if not isinstance(model.waveform_model.diting_adapter, legacy.BackboneAttentionPoolAdapter):
        raise ValueError('Actual DiTing adapter is not BackboneAttentionPoolAdapter')
    result = audit_base(model)
    front = model.waveform_model
    result.update(variant_id=cfg['variant_id'], variant_name=cfg['variant_name'],
        unchanged_downstream_initial_sha256=state_fingerprint({n: t for n, t in model.state_dict().items()
            if not n.startswith(('waveform_model.', 'mlp_pga.', 'pga_event_context_', 'pga_event_memory_'))}),
        adapter_initial_sha256=state_fingerprint(front.diting_adapter.state_dict()),
        event_fusion=cfg['event_fusion'], encoder_class=type(front.encoder).__module__+'.'+type(front.encoder).__name__,
        adapter_class=type(front.diting_adapter).__module__+'.'+type(front.diting_adapter).__name__,
        effective_diting_args=vars(args), diting_yaml_sha256=sha256(cfg['diting_config']),
        encoder_checkpoint_sha256=entry['files']['weights']['sha256'],
        encoder_checkpoint_bytes=files['weights'].stat().st_size, encoder_source=entry.get('source', 'unknown'),
        pretraining_overlap=entry.get('overlap_status', 'unknown'),
        raw_input_units='m/s^2 acceleration; masked demeaning/joint peak before encoder',
        native_n_samples=10000, sampling_rate=100, component_order='NEZ', output_dim=front.output_dim,
        pga_mlp=str(model.mlp_pga), magnitude_mlp=str(model.mlp_mag), location_mlp=str(model.mlp_loc),
        pga_readout_mode=model.pga_readout_mode, event_readout_mode=model.event_readout_mode,
        station_context_mode=model.station_context_mode, pga_use_event_context=model.pga_use_event_context,
        pga_event_memory=model.pga_event_memory,
        event_mapper=str(model.pga_event_memory_mapper if model.pga_event_memory else model.pga_event_context_proj),
        event_post_add_gate=float(model.pga_event_context_gate.detach()) if model.pga_use_event_context else None,
        event_memory_gate='none: always-valid single event key/value' if model.pga_event_memory else None,
        capacity_note='C bypasses PGA cross-attention; registered parameters are not active capacity')
    return result


def extra_audit(model, cfg, row, device, destination):
    """One real audit update, rolled back. Never an extra training data exposure."""
    from fe01.config import write_json
    from fe01.data import prepare_sample, read_event
    from fe01.engine import inputs_to, loss
    before = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
    encoder_before = state_fingerprint(model.waveform_model.encoder.state_dict())
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.device(device).type == 'cuda' else None
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg['training']['lr'])
    optimizer_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
    if any(id(p) in optimizer_ids for p in model.waveform_model.encoder.parameters()):
        raise AssertionError('Encoder entered optimizer')
    gradients = {}
    observed_shapes = []
    def shape(value):
        if torch.is_tensor(value): return list(value.shape)
        if isinstance(value, dict): return {k: shape(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [shape(v) for v in value]
        return str(type(value))
    hook = model.waveform_model.encoder.register_forward_hook(lambda m, x, y: observed_shapes.append(shape(y)))
    try:
        model.train()
        if any(m.training for m in model.waveform_model.encoder.modules()):
            raise AssertionError('Frozen encoder not eval after parent.train()')
        sample = prepare_sample(read_event(row, cfg, 3), cfg, 3)
        outputs = model(*inputs_to(sample, device))
        value = loss(outputs, [t.unsqueeze(0).to(device) for t in sample['labels']], model, cfg, sample['inputs'][4].unsqueeze(0).to(device))
        if not torch.isfinite(value): raise FloatingPointError('Nonfinite audit loss')
        value.backward()
        for name, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all(): raise FloatingPointError('Nonfinite gradient: '+name)
            gradients[name] = dict(present=p.grad is not None, nonzero=p.grad is not None and bool(p.grad.abs().max()>0))
        if any(p.grad is not None for p in model.waveform_model.encoder.parameters()):
            raise AssertionError('Frozen encoder has gradient')
        if not any(v['nonzero'] for n, v in gradients.items() if n.startswith('waveform_model.diting_adapter.')):
            raise AssertionError('DiTing adapter has no nonzero gradient')
        for prefix in ('mlp_pga.', 'event_cross_attention.'):
            if not any(v['present'] for n, v in gradients.items() if n.startswith(prefix)):
                raise AssertionError('Missing downstream gradient: '+prefix)
        if model.pga_event_memory and not any(v['nonzero'] for n, v in gradients.items() if n.startswith('pga_event_memory_mapper.')):
            raise AssertionError('Memory mapper has no gradient')
        optimizer.step()
        if state_fingerprint(model.waveform_model.encoder.state_dict()) != encoder_before:
            raise AssertionError('Frozen encoder parameter/buffer changed')
        write_json(destination/'freeze_gradient_audit.json', dict(status='PASS',
            encoder_not_in_optimizer=True, parent_train_encoder_eval=True,
            encoder_parameters_and_buffers_unchanged=True, encoder_state_sha256=encoder_before,
            observed_native_feature_shapes=observed_shapes, gradient_by_parameter=gradients,
            active_gradient_parameters=sum(p.numel() for n, p in model.named_parameters() if gradients[n]['present']),
            nonzero_gradient_parameters=sum(p.numel() for n, p in model.named_parameters() if gradients[n]['nonzero']),
            scope='one real train-event audit update; complete model state and RNG restored; not formal training'))
    finally:
        hook.remove()
        model.load_state_dict(before, strict=True)
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(rng)
        if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)
        model.eval()
