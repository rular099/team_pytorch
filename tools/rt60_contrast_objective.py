"""RT60 final-field contrast loss and immutable RT59 teacher helpers.

The teacher readout is checkpoint metadata, not a registered model component.
This preserves all RT55--RT59 state-dict keys and shapes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from tools.rt59_dual_objective import (
    GROUP_WEIGHTS,
    PointLossConfig,
    _target_from_labels,
    group_codes,
    group_counts,
    local_group_objective,
    point_losses,
)


READOUT_PREFIX = 'pga_anchor_residual_transport_head.transport.residual_head.'
READOUT_STATE_KEYS = (
    '0.weight', '0.bias', '1.weight', '1.bias', '3.weight', '3.bias',
)
FIELD_BUCKET_NAMES = (
    'random_single', 'random_multi', 'normal_remote_single', 'normal_remote_multi',
)
FIELD_BUCKET_WEIGHTS = (0.25, 0.25, 0.125, 0.125)


def _raw_model(model):
    return model.module if hasattr(model, 'module') else model


def _readout(model):
    raw = _raw_model(model)
    head = getattr(raw, 'pga_anchor_residual_transport_head', None)
    if head is None:
        raise RuntimeError('RT60 requires pga_anchor_residual_transport_head.')
    transport = head.transport
    if not getattr(transport, 'enable_rt60_reference', False):
        raise RuntimeError('RT60 reference mode is not enabled by the model config.')
    return transport.residual_head, transport


def tensor_mapping_sha256(state: Mapping[str, Tensor]) -> str:
    """Stable hash over names, dtype/shape and exact contiguous tensor bytes."""
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not torch.is_tensor(value):
            raise TypeError(f'non-tensor state value for {key}.')
        cpu = value.detach().cpu().contiguous()
        header = json.dumps(
            {'key': key, 'dtype': str(cpu.dtype), 'shape': list(cpu.shape)},
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
        digest.update(len(header).to_bytes(8, 'big'))
        digest.update(header)
        digest.update(cpu.numpy().tobytes(order='C'))
    return digest.hexdigest()


def readout_state_dict(model) -> Dict[str, Tensor]:
    readout, _ = _readout(model)
    state = readout.state_dict()
    if tuple(state.keys()) != READOUT_STATE_KEYS:
        raise RuntimeError(
            f'RT60 production readout schema changed: {tuple(state.keys())!r}.'
        )
    if len(state) != 6 or sum(value.numel() for value in state.values()) != 67077:
        raise RuntimeError(
            'RT60 production readout must contain exactly 6 tensors/67077 scalars.'
        )
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def checkpoint_non_readout_fingerprint(checkpoint: Mapping[str, object]) -> str:
    state = checkpoint.get('model_state_dict')
    if not isinstance(state, Mapping):
        raise ValueError('parent checkpoint lacks model_state_dict.')
    filtered = {}
    for key, value in state.items():
        canonical = str(key)[7:] if str(key).startswith('module.') else str(key)
        if canonical.startswith(READOUT_PREFIX):
            continue
        if torch.is_tensor(value):
            filtered[canonical] = value
    if not filtered:
        raise ValueError('parent checkpoint has no non-readout tensors to fingerprint.')
    return tensor_mapping_sha256(filtered)


def model_non_readout_fingerprint(model) -> str:
    """Fingerprint every loaded parent model tensor except the six readout tensors."""
    filtered = {
        key: value
        for key, value in _raw_model(model).state_dict().items()
        if not key.startswith(READOUT_PREFIX)
    }
    if not filtered:
        raise ValueError('loaded parent model has no non-readout tensors.')
    return tensor_mapping_sha256(filtered)


def build_reference_payload(
        model,
        parent_checkpoint: Mapping[str, object],
        parent_checkpoint_sha256: str,
        normalization: Optional[Mapping[str, object]],
        source_identity: Mapping[str, object],
        expected_parent_task_id: str,
        expected_parent_epoch: int = 8) -> Dict[str, object]:
    if len(parent_checkpoint_sha256) != 64:
        raise ValueError('RT60 parent checkpoint SHA-256 must contain 64 hex characters.')
    try:
        int(parent_checkpoint_sha256, 16)
    except ValueError as exc:
        raise ValueError('RT60 parent checkpoint SHA-256 is not hexadecimal.') from exc
    epoch = int(parent_checkpoint.get('epoch', -1))
    if epoch != int(expected_parent_epoch):
        raise ValueError(
            f'RT60 requires parent epoch {expected_parent_epoch}, got {epoch}.'
        )
    recorded_task = parent_checkpoint.get('task_id')
    if recorded_task is not None and str(recorded_task) != str(expected_parent_task_id):
        raise ValueError(
            f'RT60 parent task_id mismatch: {recorded_task!r} != '
            f'{expected_parent_task_id!r}.'
        )
    state = readout_state_dict(model)
    payload = {
        'schema_version': 1,
        'readout_state_dict': state,
        'readout_sha256': tensor_mapping_sha256(state),
        'parent_checkpoint_sha256': parent_checkpoint_sha256.lower(),
        'parent_non_readout_fingerprint': model_non_readout_fingerprint(model),
        'parent_non_readout_checkpoint_fingerprint': (
            checkpoint_non_readout_fingerprint(parent_checkpoint)
        ),
        'parent_epoch': epoch,
        'parent_task_id': str(recorded_task or expected_parent_task_id),
        'parent_task_id_source': (
            'checkpoint_metadata' if recorded_task is not None
            else 'exact_checkpoint_sha_plus_rt59_task_contract'
        ),
        'normalization': dict(normalization or {}),
        'source_identity': dict(source_identity),
    }
    restore_reference_payload(model, payload)
    return payload


def restore_reference_payload(model, payload: Mapping[str, object]) -> None:
    if not isinstance(payload, Mapping) or int(payload.get('schema_version', -1)) != 1:
        raise ValueError('missing or unsupported RT60 reference payload.')
    state = payload.get('readout_state_dict')
    if not isinstance(state, Mapping) or tuple(state.keys()) != READOUT_STATE_KEYS:
        raise ValueError('invalid RT60 reference readout state schema.')
    actual = tensor_mapping_sha256(state)
    if actual != payload.get('readout_sha256'):
        raise ValueError('RT60 reference readout SHA-256 mismatch.')
    _, transport = _readout(model)
    transport.set_rt60_reference_state(dict(state))


def export_reference_payload(model, identity: Mapping[str, object]) -> Dict[str, object]:
    """Refresh tensor copies while preserving immutable teacher identity."""
    _, transport = _readout(model)
    state = transport.export_rt60_reference_state()
    if state is None:
        raise RuntimeError('RT60 reference snapshot has not been configured.')
    payload = dict(identity)
    payload['readout_state_dict'] = state
    payload['readout_sha256'] = tensor_mapping_sha256(state)
    return payload


def _all_reduce_counts(local_counts: Tensor) -> Tuple[Tensor, int]:
    counts = local_counts.detach().clone()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return counts, world_size


def field_contrast_values(
        prediction: Tensor,
        target: Tensor,
        groups: Tensor,
        actual_station_count: Tensor) -> Tuple[Tensor, Tensor]:
    """O(Q) per-field half pair-MSE and four RT60 bucket codes.

    For n>=2 valid errors e_j, the returned value is
    sum_j (e_j - mean(e))^2 / (n - 1), exactly equal to the average
    over j<k of 0.5 * (e_j - e_k)^2.
    """
    if prediction.shape != target.shape or target.shape != groups.shape:
        raise ValueError('RT60 contrast target shapes differ.')
    if actual_station_count.reshape(-1).shape[0] != prediction.shape[0]:
        raise ValueError('actual station count must have one value per field.')
    values = []
    buckets = []
    counts = actual_station_count.reshape(-1)
    for row in range(prediction.shape[0]):
        for group, single_bucket, multi_bucket in ((0, 0, 1), (2, 2, 3)):
            selected = groups[row] == group
            n_target = int(selected.sum().item())
            if n_target < 2:
                continue
            # Mask before arithmetic so invalid NaN padding cannot contaminate.
            error = prediction[row, selected] - target[row, selected]
            if not torch.isfinite(error).all():
                raise ValueError('valid RT60 contrast errors must be finite.')
            centered = error - error.mean()
            values.append(centered.square().sum() / float(n_target - 1))
            buckets.append(single_bucket if int(counts[row].item()) == 1 else multi_bucket)
    if not values:
        zero = prediction.reshape(-1)[:0].sum().reshape(1)
        code = torch.full((1,), -1, device=prediction.device, dtype=torch.long)
        return zero, code
    return (
        torch.stack(values),
        torch.as_tensor(buckets, device=prediction.device, dtype=torch.long),
    )


def field_bucket_counts(buckets: Tensor) -> Tensor:
    if ((buckets < -1) | (buckets > 3)).any():
        raise ValueError('field bucket codes must be -1 or 0..3.')
    return torch.stack([(buckets == bucket).sum() for bucket in range(4)])


def local_field_objective(
        values: Tensor,
        buckets: Tensor,
        global_counts: Tensor,
        world_size: int,
        weights: Sequence[float] = FIELD_BUCKET_WEIGHTS) -> Tensor:
    if values.shape != buckets.shape or global_counts.shape != (4,):
        raise ValueError('invalid RT60 field reduction shapes.')
    if len(weights) != 4 or world_size < 1:
        raise ValueError('invalid RT60 field weights/world size.')
    result = values.reshape(-1)[:0].sum()
    counts = global_counts.detach().to(device=values.device, dtype=torch.float64)
    if (counts < field_bucket_counts(buckets).to(counts)).any():
        raise ValueError('global field counts cannot be smaller than local counts.')
    for bucket, weight in enumerate(weights):
        active = (counts[bucket] > 0).to(values.dtype)
        denominator = counts[bucket].clamp_min(1).to(values.dtype)
        result = result + (
            float(weight) * float(world_size) * active
            * values[buckets == bucket].sum() / denominator
        )
    return result


def rt60_contrast_objective(
        model,
        outputs,
        labels,
        output_layout,
        query_valid: Tensor,
        p_picks,
        cfg: Mapping[str, object],
        pga_target_normalization: Optional[Mapping[str, float]] = None
        ) -> Tuple[Tensor, Dict[str, Tensor]]:
    if not cfg or not cfg.get('enabled', False):
        raise ValueError('RT60 contrast objective is not enabled.')
    if not isinstance(p_picks, dict) or 'causal_random_mask_applied' not in p_picks:
        raise ValueError('RT60 objective requires causal_random_mask_applied metadata.')
    raw = _raw_model(model)
    reference_mdn = getattr(raw, '_last_rt60_reference_mdn', None)
    observed = getattr(raw, '_last_rt59_route_observed', None)
    station_valid = getattr(raw, '_last_station_valid', None)
    if reference_mdn is None or observed is None or station_valid is None:
        raise RuntimeError('RT60 forward/reference diagnostics are incomplete.')
    final_mdn = outputs[output_layout.index('pga')]
    target = _target_from_labels(
        labels, output_layout, final_mdn, pga_target_normalization
    )
    valid = query_valid.to(final_mdn.device).bool()
    groups = group_codes(
        valid,
        observed,
        p_picks['causal_random_mask_applied'].to(final_mdn.device),
    )
    loss_cfg = PointLossConfig(
        smooth_weight=float(cfg.get('smooth_weight', 0.10)),
        mse_weight=float(cfg.get('mse_weight', 0.10)),
        regret_weight=0.0,
        smooth_beta_model=float(cfg.get('smooth_beta_model', 1.0)),
    )
    point = point_losses(reference_mdn.detach(), final_mdn, target, groups, loss_cfg)
    reference_point = point_losses(
        reference_mdn.detach(), reference_mdn.detach(), target, groups, loss_cfg
    )
    global_target_counts, world_size = _all_reduce_counts(group_counts(groups))
    group_weights = tuple(float(value) for value in cfg.get(
        'group_weights', GROUP_WEIGHTS
    ))
    point_loss = local_group_objective(
        point['main'], groups, global_target_counts, world_size, group_weights
    )
    remote = (groups == 0) | (groups == 2)
    regret_values = torch.where(
        remote,
        F.relu(point['smooth'] - reference_point['smooth'].detach()),
        torch.zeros_like(point['smooth']),
    )
    regret_loss = local_group_objective(
        regret_values, groups, global_target_counts, world_size, group_weights
    )

    actual_station_count = station_valid.to(final_mdn.device).bool().sum(dim=1)
    contrast_values, contrast_buckets = field_contrast_values(
        point['mean'], target, groups, actual_station_count
    )
    global_field_counts, field_world_size = _all_reduce_counts(
        field_bucket_counts(contrast_buckets)
    )
    if field_world_size != world_size:
        raise RuntimeError('RT60 DDP world size changed between fixed-order reductions.')
    field_weights = tuple(float(value) for value in cfg.get(
        'field_bucket_weights', FIELD_BUCKET_WEIGHTS
    ))
    contrast_loss = local_field_objective(
        contrast_values,
        contrast_buckets,
        global_field_counts,
        world_size,
        field_weights,
    )
    total = (
        point_loss
        + float(cfg.get('regret_weight', 0.05)) * regret_loss
        + float(cfg.get('contrast_weight', 0.40)) * contrast_loss
    )
    row_counts = torch.stack([
        (groups == group).any(dim=1).sum() for group in range(3)
    ])
    stats = {
        'target_counts': global_target_counts.detach(),
        'local_row_counts': row_counts.detach(),
        'field_counts': global_field_counts.detach(),
        'point': point_loss.detach(),
        'nll': local_group_objective(
            point['nll'].detach(), groups, global_target_counts, world_size, group_weights
        ).detach(),
        'smooth': local_group_objective(
            point['smooth'].detach(), groups, global_target_counts, world_size, group_weights
        ).detach(),
        'mse': local_group_objective(
            point['mse'].detach(), groups, global_target_counts, world_size, group_weights
        ).detach(),
        'regret': regret_loss.detach(),
        'contrast': contrast_loss.detach(),
        'total': total.detach(),
        'groups': groups.detach(),
    }
    return total, stats
