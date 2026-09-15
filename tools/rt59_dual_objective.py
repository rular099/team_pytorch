"""RT59-v3 observable routing and globally normalized loss helpers.

All PGA tensors handled here use the train-normalized coordinate.  The module
is intentionally independent of the TEAM model so its routing and DDP algebra
can be tested without constructing the DiTing encoder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


GROUP_NAMES = ('random', 'normal_observed', 'normal_remote')
GROUP_WEIGHTS = (0.50, 0.25, 0.25)


def unique_observed_match(
        station_coords: Tensor,
        query_coords: Tensor,
        station_valid: Tensor,
        query_valid: Tensor,
        atol: float = 1e-6) -> Tuple[Tensor, Tensor, Tensor]:
    """Return unique exact-coordinate matches, observed and ambiguous masks."""
    if atol < 0 or not math.isfinite(atol):
        raise ValueError('atol must be finite and nonnegative.')
    if station_coords.ndim != 3 or query_coords.ndim != 3:
        raise ValueError('coordinates must be (B,S,3)/(B,Q,3).')
    if (
        station_coords.shape[0] != query_coords.shape[0]
        or station_coords.shape[-1] != 3
        or query_coords.shape[-1] != 3
    ):
        raise ValueError('incompatible coordinate shapes.')
    if station_coords.shape[:2] != station_valid.shape:
        raise ValueError('station coordinate/mask shape mismatch.')
    if query_coords.shape[:2] != query_valid.shape:
        raise ValueError('query coordinate/mask shape mismatch.')
    sv = station_valid.bool()
    qv = query_valid.bool()
    if not torch.isfinite(station_coords[sv]).all():
        raise ValueError('valid station coordinates must be finite.')
    if not torch.isfinite(query_coords[qv]).all():
        raise ValueError('valid query coordinates must be finite.')
    xs = torch.where(sv[..., None], station_coords, torch.zeros_like(station_coords))
    xq = torch.where(qv[..., None], query_coords, torch.zeros_like(query_coords))
    match = (xq[:, :, None, :] - xs[:, None, :, :]).abs().le(atol).all(dim=-1)
    match = match & qv[:, :, None] & sv[:, None, :]
    count = match.sum(dim=-1)
    observed = qv & (count == 1)
    ambiguous = qv & (count > 1)
    return match & observed[..., None], observed, ambiguous


def route_correction(
        local_delta: Tensor,
        remote_delta: Tensor,
        unique_match: Tensor,
        observed: Tensor,
        query_valid: Tensor) -> Tensor:
    if unique_match.shape != (*remote_delta.shape, local_delta.shape[1]):
        raise ValueError('routing shape mismatch.')
    matched = torch.where(
        unique_match,
        local_delta[:, None, :],
        torch.zeros_like(local_delta[:, None, :]),
    ).sum(dim=-1)
    correction = torch.where(observed, matched, remote_delta)
    return torch.where(query_valid.bool(), correction, torch.zeros_like(correction))


def shift_mdn(base_mdn: Tensor, delta: Tensor) -> Tensor:
    if base_mdn.ndim != 4 or base_mdn.shape[-1] != 3:
        raise ValueError('expected base MDN shape (B,Q,K,3).')
    if delta.shape != base_mdn.shape[:2]:
        raise ValueError('delta must have shape (B,Q).')
    return torch.stack(
        (base_mdn[..., 0], base_mdn[..., 1] + delta[..., None], base_mdn[..., 2]),
        dim=-1,
    )


def group_codes(query_valid: Tensor, observed: Tensor, random_applied: Tensor) -> Tensor:
    """Map valid targets to R=0, NI=1, NO=2; invalid targets are -1."""
    if query_valid.shape != observed.shape:
        raise ValueError('query/observed shape mismatch.')
    if random_applied.reshape(-1).shape != query_valid.shape[:1]:
        raise ValueError('random protocol metadata shape mismatch.')
    rv = random_applied.bool().reshape(-1, 1)
    valid = query_valid.bool()
    if (rv & observed & valid).any():
        raise ValueError(
            'RT59 random protocol unexpectedly contains an observed input query.'
        )
    code = torch.where(rv, 0, torch.where(observed, 1, 2)).long()
    return torch.where(valid, code, torch.full_like(code, -1))


@dataclass(frozen=True)
class PointLossConfig:
    smooth_weight: float = 0.10
    mse_weight: float = 0.10
    regret_weight: float = 0.05
    smooth_beta_model: float = 1.0


def point_losses(
        base_mdn: Tensor,
        final_mdn: Tensor,
        target: Tensor,
        groups: Tensor,
        cfg: PointLossConfig = PointLossConfig()) -> Mapping[str, Tensor]:
    """Return unreduced RT59 main loss terms, masking before arithmetic."""
    if base_mdn.shape != final_mdn.shape:
        raise ValueError('base/final MDN shapes differ.')
    if target.shape != groups.shape or final_mdn.shape[:2] != target.shape:
        raise ValueError('loss shapes mismatch.')
    valid = groups >= 0
    for value in (base_mdn, final_mdn):
        if not torch.isfinite(value[valid]).all():
            raise ValueError('valid mixture parameters must be finite.')
        if (value[valid][..., 2] <= 0).any():
            raise ValueError('valid mixture sigmas must be positive.')
    if not torch.isfinite(target[valid]).all():
        raise ValueError('valid PGA targets must be finite.')
    safe_target = torch.where(valid, target, torch.zeros_like(target))

    def stats(mixture: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        fill = torch.zeros_like(mixture)
        fill[..., 2] = 1.0
        safe = torch.where(valid[..., None, None], mixture, fill)
        logits, means, sigmas = safe.unbind(dim=-1)
        sigmas = sigmas.clamp_min(1e-6)
        log_weights = F.log_softmax(logits, dim=-1)
        predictive_mean = (log_weights.exp() * means).sum(dim=-1)
        nll = -torch.logsumexp(
            log_weights
            - sigmas.log()
            - 0.5 * math.log(2.0 * math.pi)
            - 0.5 * ((safe_target[..., None] - means) / sigmas).square(),
            dim=-1,
        )
        smooth = F.smooth_l1_loss(
            predictive_mean,
            safe_target,
            beta=cfg.smooth_beta_model,
            reduction='none',
        )
        mse = (predictive_mean - safe_target).square()
        return predictive_mean, nll, smooth, mse

    mean, nll, smooth, mse = stats(final_mdn)
    _, _, base_smooth, _ = stats(base_mdn.detach())
    normal = (groups == 1) | (groups == 2)
    regret = torch.where(
        normal,
        F.relu(smooth - base_smooth.detach()),
        torch.zeros_like(smooth),
    )
    main = (
        nll
        + cfg.smooth_weight * smooth
        + cfg.mse_weight * mse
        + cfg.regret_weight * regret
    )
    values = {
        'main': main,
        'nll': nll,
        'smooth': smooth,
        'mse': mse,
        'regret': regret,
        'mean': mean,
    }
    return {
        name: torch.where(valid, value, torch.zeros_like(value))
        for name, value in values.items()
    }


def group_counts(groups: Tensor) -> Tensor:
    if ((groups < -1) | (groups > 2)).any():
        raise ValueError('group codes must be -1, 0, 1, or 2.')
    return torch.stack([(groups == group).sum() for group in range(3)])


def _all_reduce_counts(local_counts: Tensor) -> Tuple[Tensor, int]:
    counts = local_counts.detach().clone()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return counts, world_size


def local_group_objective(
        values: Tensor,
        groups: Tensor,
        global_counts: Tensor,
        world_size: int = 1,
        group_weights: Tuple[float, float, float] = GROUP_WEIGHTS) -> Tensor:
    """Differentiable local numerator scaled for default DDP averaging."""
    if values.shape != groups.shape or global_counts.shape != (3,):
        raise ValueError('invalid grouped reduction shapes.')
    if world_size < 1 or len(group_weights) != 3:
        raise ValueError('invalid world size or group weights.')
    counts = global_counts.detach().to(device=values.device, dtype=torch.float64)
    if not torch.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('global counts must be finite and nonnegative.')
    if (counts < group_counts(groups).to(counts)).any():
        raise ValueError('global counts cannot be smaller than local counts.')
    result = values.reshape(-1)[:0].sum()
    for group, weight in enumerate(group_weights):
        numerator = values[groups == group].sum()
        active = (counts[group] > 0).to(values.dtype)
        denominator = counts[group].clamp_min(1).to(values.dtype)
        result = result + float(weight) * world_size * active * numerator / denominator
    return result


def ddp_group_objective(
        values: Tensor,
        groups: Tensor,
        group_weights: Tuple[float, float, float] = GROUP_WEIGHTS,
        global_counts: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    if global_counts is None:
        global_counts, world_size = _all_reduce_counts(group_counts(groups))
    else:
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    return (
        local_group_objective(values, groups, global_counts, world_size, group_weights),
        global_counts,
    )


def _target_from_labels(
        labels,
        output_layout,
        reference: Tensor,
        normalization: Optional[Mapping[str, float]]) -> Tensor:
    label_layout = [name for name in output_layout if name in ('mag', 'loc', 'pga')]
    if 'pga' not in label_layout:
        raise ValueError('RT59 objective requires PGA labels.')
    target = labels[label_layout.index('pga')].to(
        device=reference.device,
        dtype=reference.dtype,
    )
    target = target.reshape(target.shape[0], target.shape[1], -1)[..., 0]
    if normalization and normalization.get('enabled', False):
        mean = float(normalization['mean'])
        std = max(float(normalization['std']), 1e-8)
        target = (target - mean) / std
    return target


def _masked_pair_query_loss(
        prediction: Tensor,
        target: Tensor,
        pair_mask: Tensor,
        beta: float) -> Tuple[Tensor, Tensor]:
    """Mean stations per query; return graph values and valid query mask."""
    safe_prediction = torch.where(pair_mask, prediction, torch.zeros_like(prediction))
    safe_target = torch.where(pair_mask, target, torch.zeros_like(target))
    per_pair = F.smooth_l1_loss(
        safe_prediction,
        safe_target,
        beta=float(beta),
        reduction='none',
    )
    count = pair_mask.sum(dim=-1)
    query_value = (per_pair * pair_mask.to(per_pair.dtype)).sum(dim=-1) / count.clamp_min(1)
    return query_value, count > 0


def _within_event_values(
        prediction: Tensor,
        target: Tensor,
        groups: Tensor,
        beta: float,
        max_pairs: int) -> Tuple[Tensor, Tensor]:
    """Return one mean pair loss per event/group and its group codes."""
    values = []
    codes = []
    for batch_index in range(prediction.shape[0]):
        for group in (0, 2):
            valid_idx = torch.nonzero(
                groups[batch_index] == group,
                as_tuple=False,
            ).squeeze(-1)
            if valid_idx.numel() < 2:
                continue
            pairs = torch.triu_indices(
                valid_idx.numel(),
                valid_idx.numel(),
                offset=1,
                device=prediction.device,
            )
            if max_pairs > 0 and pairs.shape[1] > max_pairs:
                pairs = pairs[:, :max_pairs]
            left = valid_idx[pairs[0]]
            right = valid_idx[pairs[1]]
            pred_diff = prediction[batch_index, left] - prediction[batch_index, right]
            target_diff = target[batch_index, left] - target[batch_index, right]
            values.append(F.smooth_l1_loss(
                pred_diff,
                target_diff,
                beta=float(beta),
                reduction='mean',
            ))
            codes.append(group)
    if not values:
        zero = prediction.reshape(-1)[:0].sum().reshape(1)
        return zero, torch.full((1,), -1, device=prediction.device, dtype=torch.long)
    return torch.stack(values), torch.as_tensor(codes, device=prediction.device, dtype=torch.long)


def rt59_grouped_objective(
        model,
        outputs,
        labels,
        output_layout,
        query_valid: Tensor,
        p_picks,
        cfg: Mapping[str, object],
        pga_target_normalization: Optional[Mapping[str, float]] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Compute the complete RT59-v3 objective and detached accounting stats."""
    if not cfg or not cfg.get('enabled', False):
        raise ValueError('RT59 grouped objective is not enabled.')
    if not isinstance(p_picks, dict) or 'causal_random_mask_applied' not in p_picks:
        raise ValueError('RT59 objective requires causal_random_mask_applied metadata.')
    if 'pga' not in output_layout:
        raise ValueError('RT59 objective requires a PGA output.')
    base_mdn = getattr(model, '_last_rt59_base_mdn', None)
    observed = getattr(model, '_last_rt59_route_observed', None)
    relative = getattr(model, '_last_rt59_relative_transfer', None)
    candidate = getattr(model, '_last_rt59_candidate', None)
    if any(value is None for value in (base_mdn, observed, relative, candidate)):
        raise RuntimeError('RT59 forward diagnostics are incomplete.')
    final_mdn = outputs[output_layout.index('pga')]
    target = _target_from_labels(
        labels,
        output_layout,
        final_mdn,
        pga_target_normalization,
    )
    valid = query_valid.to(final_mdn.device).bool()
    random_applied = p_picks['causal_random_mask_applied'].to(final_mdn.device)
    groups = group_codes(valid, observed, random_applied)
    loss_cfg = PointLossConfig(
        smooth_weight=float(cfg.get('smooth_weight', 0.10)),
        mse_weight=float(cfg.get('mse_weight', 0.10)),
        regret_weight=float(cfg.get('regret_weight', 0.05)),
        smooth_beta_model=float(cfg.get('smooth_beta_model', 1.0)),
    )
    weights = tuple(float(value) for value in cfg.get('group_weights', GROUP_WEIGHTS))
    if len(weights) != 3:
        raise ValueError('RT59 group_weights must have exactly three entries.')
    point = point_losses(base_mdn, final_mdn, target, groups, loss_cfg)
    global_target_counts, world_size = _all_reduce_counts(group_counts(groups))
    total = local_group_objective(
        point['main'], groups, global_target_counts, world_size, weights
    )

    station_valid = getattr(model, '_last_station_valid', None)
    input_base = getattr(model, '_last_rt59_input_base_mean', None)
    public_base = getattr(model, '_last_rt59_base_mean', None)
    if station_valid is None or input_base is None or public_base is None:
        raise RuntimeError('RT59 frozen station/public base tensors are incomplete.')
    if 'input_pga_values' not in p_picks or 'input_pga_valid' not in p_picks:
        raise ValueError('RT59 transport losses require input PGA label metadata.')
    input_target = p_picks['input_pga_values'].to(
        device=final_mdn.device,
        dtype=final_mdn.dtype,
    )
    if pga_target_normalization and pga_target_normalization.get('enabled', False):
        mean = float(pga_target_normalization['mean'])
        std = max(float(pga_target_normalization['std']), 1e-8)
        input_target = (input_target - mean) / std
    else:
        std = 1.0
    input_valid = p_picks['input_pga_valid'].to(final_mdn.device).bool()
    input_valid = input_valid & station_valid.to(final_mdn.device).bool()
    if not torch.isfinite(input_target[input_valid]).all():
        raise ValueError('valid RT59 input-station PGA labels must be finite.')
    transport_query = (groups == 0) | (groups == 2)
    pair_mask = transport_query[..., None] & input_valid[:, None, :]
    safe_query_target = torch.where(valid, target, torch.zeros_like(target))
    safe_input_target = torch.where(
        input_valid, input_target, torch.zeros_like(input_target)
    )
    safe_public_base = torch.where(
        valid,
        public_base.squeeze(-1),
        torch.zeros_like(public_base.squeeze(-1)),
    )
    safe_input_base = torch.where(
        input_valid,
        input_base.squeeze(-1),
        torch.zeros_like(input_base.squeeze(-1)),
    )
    relative_target = (
        safe_query_target[..., None]
        - safe_input_target[:, None, :]
        - (safe_public_base[..., None] - safe_input_base[:, None, :])
    )
    candidate_target = safe_query_target[..., None].expand_as(candidate)
    relative_value, relative_query_valid = _masked_pair_query_loss(
        relative,
        relative_target,
        pair_mask,
        float(cfg.get('relative_beta_dex', 0.20)) / std,
    )
    candidate_value, candidate_query_valid = _masked_pair_query_loss(
        candidate,
        candidate_target,
        pair_mask,
        float(cfg.get('candidate_beta_dex', 0.20)) / std,
    )
    auxiliary_groups = torch.where(
        transport_query,
        groups,
        torch.full_like(groups, -1),
    )
    relative_groups = torch.where(
        relative_query_valid,
        auxiliary_groups,
        torch.full_like(groups, -1),
    )
    candidate_groups = torch.where(
        candidate_query_valid,
        auxiliary_groups,
        torch.full_like(groups, -1),
    )
    relative_counts, _ = _all_reduce_counts(group_counts(relative_groups))
    candidate_counts, _ = _all_reduce_counts(group_counts(candidate_groups))
    relative_loss = local_group_objective(
        relative_value,
        relative_groups,
        relative_counts,
        world_size,
        weights,
    )
    candidate_loss = local_group_objective(
        candidate_value,
        candidate_groups,
        candidate_counts,
        world_size,
        weights,
    )
    total = total + float(cfg.get('relative_weight', 0.20)) * relative_loss
    total = total + float(cfg.get('candidate_weight', 0.05)) * candidate_loss

    final_mean = point['mean']
    difference_values, difference_groups = _within_event_values(
        final_mean,
        target,
        groups,
        float(cfg.get('difference_beta_dex', 0.15)) / std,
        int(cfg.get('difference_max_pairs', 105)),
    )
    difference_counts, _ = _all_reduce_counts(group_counts(difference_groups))
    difference_loss = local_group_objective(
        difference_values,
        difference_groups,
        difference_counts,
        world_size,
        weights,
    )
    total = total + float(cfg.get('difference_weight', 0.40)) * difference_loss

    row_counts = torch.stack([
        (groups == group).any(dim=1).sum() for group in range(3)
    ])
    stats = {
        'target_counts': global_target_counts.detach(),
        'local_row_counts': row_counts.detach(),
        'relative_query_counts': relative_counts.detach(),
        'candidate_query_counts': candidate_counts.detach(),
        'difference_field_counts': difference_counts.detach(),
        'main': local_group_objective(
            point['main'].detach(), groups, global_target_counts, world_size, weights
        ).detach(),
        'nll': local_group_objective(
            point['nll'].detach(), groups, global_target_counts, world_size, weights
        ).detach(),
        'smooth': local_group_objective(
            point['smooth'].detach(), groups, global_target_counts, world_size, weights
        ).detach(),
        'mse': local_group_objective(
            point['mse'].detach(), groups, global_target_counts, world_size, weights
        ).detach(),
        'regret': local_group_objective(
            point['regret'].detach(), groups, global_target_counts, world_size, weights
        ).detach(),
        'relative': relative_loss.detach(),
        'candidate': candidate_loss.detach(),
        'difference': difference_loss.detach(),
        'total': total.detach(),
        'groups': groups.detach(),
    }
    return total, stats
