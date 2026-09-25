"""RT61 immutable-reference, trainable-delta and objective helpers."""

from __future__ import annotations

import hashlib
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor

from tools.rt60_contrast_objective import (
    READOUT_PREFIX,
    READOUT_STATE_KEYS,
    readout_state_dict,
    rt60_contrast_objective,
    tensor_mapping_sha256,
)


TASK_ID = '20260925-rt61-wave-geometry-residual-conditioning'
ADAPTER_PREFIX = (
    'pga_anchor_residual_transport_head.transport.'
    'rt61_wave_geometry_adapter.'
)
TRAINABLE_PREFIXES = (READOUT_PREFIX, ADAPTER_PREFIX)
ADAPTER_STATE_KEYS = (
    'wave_projection.weight',
    'geometry_projection.weight',
    'output_projection.weight',
)


def _raw_model(model):
    return model.module if hasattr(model, 'module') else model


def _transport(model):
    raw = _raw_model(model)
    head = getattr(raw, 'pga_anchor_residual_transport_head', None)
    if head is None:
        raise RuntimeError('RT61 requires pga_anchor_residual_transport_head.')
    transport = head.transport
    if not getattr(transport, 'use_rt61_wave_geometry_adapter', False):
        raise RuntimeError('RT61 wave-geometry adapter is not enabled.')
    if not getattr(transport, 'enable_rt60_reference', False):
        raise RuntimeError('RT61 immutable reference mode is not enabled.')
    return transport


def adapter_state_dict(model) -> Dict[str, Tensor]:
    state = _transport(model).rt61_wave_geometry_adapter.state_dict()
    if tuple(state) != ADAPTER_STATE_KEYS:
        raise RuntimeError(f'RT61 adapter schema changed: {tuple(state)!r}.')
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def trainable_state_dict(model) -> Dict[str, Tensor]:
    raw = _raw_model(model)
    state = {}
    for key, value in raw.state_dict().items():
        if key.startswith(TRAINABLE_PREFIXES):
            state[key] = value.detach().cpu().clone()
    expected = len(READOUT_STATE_KEYS) + len(ADAPTER_STATE_KEYS)
    if len(state) != expected:
        raise RuntimeError(
            f'RT61 trainable state must contain {expected} tensors, got {len(state)}.'
        )
    return state


def trainable_manifest(model) -> Dict[str, object]:
    state = trainable_state_dict(model)
    adapter = adapter_state_dict(model)
    return {
        'tensor_count': len(state),
        'scalar_count': sum(value.numel() for value in state.values()),
        'readout_tensor_count': len(READOUT_STATE_KEYS),
        'readout_scalar_count': sum(
            value.numel() for value in readout_state_dict(model).values()
        ),
        'adapter_tensor_count': len(adapter),
        'adapter_scalar_count': sum(value.numel() for value in adapter.values()),
        'parameter_names': list(state),
        'state_sha256': tensor_mapping_sha256(state),
    }


def _shared_tensor_mapping(state: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    filtered = {}
    for key, value in state.items():
        canonical = str(key)[7:] if str(key).startswith('module.') else str(key)
        if canonical.startswith(TRAINABLE_PREFIXES):
            continue
        if torch.is_tensor(value):
            filtered[canonical] = value
    if not filtered:
        raise ValueError('RT61 shared-state fingerprint has no tensors.')
    return filtered


def checkpoint_shared_fingerprint(checkpoint: Mapping[str, object]) -> str:
    state = checkpoint.get('model_state_dict')
    if not isinstance(state, Mapping):
        raise ValueError('RT61 parent checkpoint lacks model_state_dict.')
    return tensor_mapping_sha256(_shared_tensor_mapping(state))


def model_shared_fingerprint(model, keys=None) -> str:
    shared = _shared_tensor_mapping(_raw_model(model).state_dict())
    if keys is not None:
        wanted = set(keys)
        missing = sorted(wanted - set(shared))
        if missing:
            raise ValueError(f'RT61 model lacks parent shared keys: {missing[:10]}')
        shared = {key: shared[key] for key in sorted(wanted)}
    return tensor_mapping_sha256(shared)


def build_reference_payload(
        model,
        parent_checkpoint: Mapping[str, object],
        parent_checkpoint_sha256: str,
        normalization: Optional[Mapping[str, object]],
        source_identity: Mapping[str, object],
        expected_parent_task_id: str,
        expected_parent_epoch: int = 8,
        expected_readout_sha256: Optional[str] = None) -> Dict[str, object]:
    if len(parent_checkpoint_sha256) != 64:
        raise ValueError('RT61 parent checkpoint SHA-256 must contain 64 characters.')
    try:
        int(parent_checkpoint_sha256, 16)
    except ValueError as exc:
        raise ValueError('RT61 parent checkpoint SHA-256 is not hexadecimal.') from exc
    epoch = int(parent_checkpoint.get('epoch', -1))
    if epoch != int(expected_parent_epoch):
        raise ValueError(
            f'RT61 requires parent epoch {expected_parent_epoch}, got {epoch}.'
        )
    recorded_task = parent_checkpoint.get('task_id')
    if recorded_task is not None and str(recorded_task) != str(expected_parent_task_id):
        raise ValueError(
            f'RT61 parent task_id mismatch: {recorded_task!r} != '
            f'{expected_parent_task_id!r}.'
        )
    state = readout_state_dict(model)
    readout_sha = tensor_mapping_sha256(state)
    if expected_readout_sha256 and readout_sha != expected_readout_sha256.lower():
        raise ValueError(
            'RT61 immutable readout SHA-256 mismatch: '
            f'expected {expected_readout_sha256.lower()}, got {readout_sha}.'
        )
    parent_state = _shared_tensor_mapping(parent_checkpoint['model_state_dict'])
    model_fingerprint = model_shared_fingerprint(model)
    checkpoint_fingerprint = tensor_mapping_sha256(parent_state)
    loaded_parent_fingerprint = model_shared_fingerprint(
        model, keys=parent_state.keys()
    )
    if loaded_parent_fingerprint != checkpoint_fingerprint:
        raise ValueError(
            'RT61 loaded shared state does not exactly match the parent checkpoint.'
        )
    payload = {
        'schema_version': 2,
        'readout_state_dict': state,
        'readout_sha256': readout_sha,
        'parent_checkpoint_sha256': parent_checkpoint_sha256.lower(),
        'parent_shared_fingerprint': model_fingerprint,
        'parent_checkpoint_shared_fingerprint': checkpoint_fingerprint,
        'loaded_parent_tensor_fingerprint': loaded_parent_fingerprint,
        'parent_epoch': epoch,
        'parent_task_id': str(recorded_task or expected_parent_task_id),
        'parent_task_id_source': (
            'checkpoint_metadata' if recorded_task is not None
            else 'exact_checkpoint_sha_plus_rt59_task_contract'
        ),
        'normalization': dict(normalization or {}),
        'source_identity': dict(source_identity),
        'trainable_manifest_at_initialization': trainable_manifest(model),
    }
    restore_reference_payload(model, payload)
    return payload


def restore_reference_payload(model, payload: Mapping[str, object]) -> None:
    if not isinstance(payload, Mapping) or int(payload.get('schema_version', -1)) != 2:
        raise ValueError('missing or unsupported RT61 reference payload.')
    state = payload.get('readout_state_dict')
    if not isinstance(state, Mapping) or tuple(state) != READOUT_STATE_KEYS:
        raise ValueError('invalid RT61 reference readout state schema.')
    if tensor_mapping_sha256(state) != payload.get('readout_sha256'):
        raise ValueError('RT61 reference readout SHA-256 mismatch.')
    _transport(model).set_rt60_reference_state(dict(state))


def export_reference_payload(model, identity: Mapping[str, object]) -> Dict[str, object]:
    state = _transport(model).export_rt60_reference_state()
    if state is None:
        raise RuntimeError('RT61 immutable reference has not been configured.')
    payload = dict(identity)
    payload['readout_state_dict'] = state
    payload['readout_sha256'] = tensor_mapping_sha256(state)
    return payload


def build_trainable_delta_payload(model, identity: Mapping[str, object]) -> Dict[str, object]:
    current_shared = model_shared_fingerprint(model)
    if current_shared != identity.get('parent_shared_fingerprint'):
        raise RuntimeError(
            'RT61 frozen shared parameters/buffers changed since parent loading.'
        )
    state = trainable_state_dict(model)
    return {
        'schema_version': 1,
        'task_id': TASK_ID,
        'parent_checkpoint_sha256': identity['parent_checkpoint_sha256'],
        'parent_shared_fingerprint': identity['parent_shared_fingerprint'],
        'frozen_shared_state_verified_unchanged': True,
        'trainable_state_dict': state,
        'trainable_state_sha256': tensor_mapping_sha256(state),
        'trainable_manifest': trainable_manifest(model),
        'reference_identity': export_reference_payload(model, identity),
    }


def restore_trainable_delta(model, payload: Mapping[str, object]) -> None:
    if int(payload.get('schema_version', -1)) != 1:
        raise ValueError('unsupported RT61 trainable-delta schema.')
    if payload.get('task_id') != TASK_ID:
        raise ValueError('RT61 trainable-delta task_id mismatch.')
    if model_shared_fingerprint(model) != payload.get('parent_shared_fingerprint'):
        raise ValueError('RT61 delta parent shared-state fingerprint mismatch.')
    state = payload.get('trainable_state_dict')
    if not isinstance(state, Mapping):
        raise ValueError('RT61 delta lacks trainable_state_dict.')
    if tensor_mapping_sha256(state) != payload.get('trainable_state_sha256'):
        raise ValueError('RT61 delta trainable-state SHA-256 mismatch.')
    current = _raw_model(model).state_dict()
    expected_keys = set(trainable_state_dict(model))
    if set(state) != expected_keys:
        raise ValueError('RT61 delta trainable tensor keys do not match the model.')
    for key, value in state.items():
        if tuple(value.shape) != tuple(current[key].shape):
            raise ValueError(f'RT61 delta tensor shape mismatch for {key}.')
        current[key] = value.detach().to(
            device=current[key].device, dtype=current[key].dtype
        )
    _raw_model(model).load_state_dict(current, strict=True)
    restore_reference_payload(model, payload['reference_identity'])
    if model_shared_fingerprint(model) != payload.get('parent_shared_fingerprint'):
        raise RuntimeError('RT61 delta reconstruction changed frozen shared state.')


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rt61_contrast_objective(
        model,
        outputs,
        labels,
        output_layout,
        query_valid: Tensor,
        p_picks,
        cfg: Mapping[str, object],
        pga_target_normalization: Optional[Mapping[str, float]] = None
        ) -> Tuple[Tensor, Dict[str, Tensor]]:
    return rt60_contrast_objective(
        model,
        outputs,
        labels,
        output_layout,
        query_valid,
        p_picks,
        cfg,
        pga_target_normalization=pga_target_normalization,
        reference_attribute='_last_rt61_reference_mdn',
    )
