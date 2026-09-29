#!/usr/bin/env python3
"""Deterministic, mask-first P-prefix interventions for V01.

This module deliberately contains no model code.  The intervention operates on
the storage-valid mask before centering/normalization and never removes the P
sample or any post-P sample.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


PROTOCOL_VERSION = "v01-prep-padding-v1"


def stable_index(parts: Iterable[object], size: int, seed: int = 42) -> int:
    if size <= 0:
        raise ValueError("size must be positive")
    text = "|".join(str(part) for part in (seed, *parts))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % int(size)


def assign_retained_prep_samples(
    template_samples: np.ndarray,
    event_id: str,
    source_sensor_id: str,
    *,
    seed: int = 42,
) -> int:
    values = np.asarray(template_samples, dtype=np.int64).reshape(-1)
    if values.size == 0:
        raise ValueError("The train-only P-prefix template library is empty")
    return int(values[stable_index((event_id, source_sensor_id), values.size, seed)])


def prep_keep_mask(
    sample_valid: np.ndarray,
    p_pick_samples: np.ndarray,
    retained_prep_samples: np.ndarray,
    source_role: np.ndarray | None = None,
) -> np.ndarray:
    """Return the V_missing support mask.

    For each source, valid samples earlier than ``P-retained`` are deleted.
    Samples at P and later are always inherited unchanged from ``sample_valid``.
    Non-source/query rows are never altered.
    """
    valid = np.asarray(sample_valid, dtype=bool)
    if valid.ndim != 2:
        raise ValueError(f"sample_valid must be station-by-time, got {valid.shape}")
    picks = np.asarray(p_pick_samples, dtype=np.int64).reshape(-1)
    retained = np.asarray(retained_prep_samples, dtype=np.int64).reshape(-1)
    if picks.size != valid.shape[0] or retained.size != valid.shape[0]:
        raise ValueError("picks/retained values must align with station rows")
    if source_role is None:
        roles = np.ones(valid.shape[0], dtype=bool)
    else:
        roles = np.asarray(source_role, dtype=bool).reshape(-1)
        if roles.size != valid.shape[0]:
            raise ValueError("source_role must align with station rows")

    keep = valid.copy()
    timeline = np.arange(valid.shape[1], dtype=np.int64)
    for row in np.where(roles)[0]:
        p_pick = int(picks[row])
        if p_pick < 0:
            continue
        boundary = max(0, p_pick - max(0, int(retained[row])))
        keep[row, timeline < boundary] = False
    return keep


def apply_prep_intervention(
    waveforms: np.ndarray,
    sample_valid: np.ndarray,
    p_pick_samples: np.ndarray,
    retained_prep_samples: np.ndarray,
    source_role: np.ndarray | None = None,
    *,
    fill_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    waves = np.asarray(waveforms)
    valid = np.asarray(sample_valid, dtype=bool)
    if waves.ndim != 3 or waves.shape[:2] != valid.shape:
        raise ValueError(
            f"waveforms/sample_valid mismatch: {waves.shape} versus {valid.shape}"
        )
    keep = prep_keep_mask(
        valid,
        p_pick_samples,
        retained_prep_samples,
        source_role,
    )
    out = waves.copy()
    out[~keep] = fill_value
    return out, keep


__all__ = [
    "PROTOCOL_VERSION",
    "apply_prep_intervention",
    "assign_retained_prep_samples",
    "prep_keep_mask",
    "stable_index",
]
