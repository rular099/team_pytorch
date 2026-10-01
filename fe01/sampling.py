"""A discrete density conditioned on native capacity, independent of model RNG."""
import hashlib
import json

import numpy as np

DEFAULT_BINS = [(1, 3), (3, 5), (5, 10), (10, 20), (20, 40), (40, 90)]
DEFAULT_PROBABILITIES = [0.20, 0.20, 0.20, 0.15, 0.15, 0.10]


def stable_rng(*identity):
    value = json.dumps(identity, separators=(',', ':'), ensure_ascii=True).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(value).digest()[:8], 'little'))


class TimeSampler:
    def __init__(self, sampling_rate=100, max_sample=9000, bins=DEFAULT_BINS,
                 probabilities=DEFAULT_PROBABILITIES):
        self.rate = int(sampling_rate)
        probabilities = np.asarray(probabilities, dtype=float)
        if len(bins) != len(probabilities) or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1):
            raise ValueError('Bin probabilities must be finite, nonnegative, and sum to one')
        ticks, weights, indices = [], [], []
        previous = None
        for i, ((lo, hi), probability) in enumerate(zip(bins, probabilities)):
            if lo < 1 or hi > 90 or hi <= lo or (previous is not None and lo != previous):
                raise ValueError('Bins must partition 1..90 without overlaps or gaps')
            # Half-open bins; the last includes the global endpoint 90 s.
            full = np.arange(int(lo * self.rate), int(hi * self.rate) + (i == len(bins) - 1))
            selected = full[full <= int(max_sample)]
            ticks.extend(selected.tolist())
            weights.extend([probability / len(full)] * len(selected))
            indices.extend([i] * len(selected))
            previous = hi
        if bins[0][0] != 1 or bins[-1][1] != 90 or not ticks:
            raise ValueError('No supported 1..90 second sampling domain')
        self.ticks = np.asarray(ticks)
        self.bin_ids = np.asarray(indices)
        self.retained_mass = float(np.sum(weights))
        self.probabilities = np.asarray(weights) / self.retained_mass
        self.cdf = np.cumsum(self.probabilities)
        self.cdf[-1] = 1

    def sample(self, dataset_id, event_id, epoch, draw, seed=42, stream='train-time'):
        u = stable_rng(seed, dataset_id, str(event_id), epoch, draw, stream).random()
        index = np.searchsorted(self.cdf, u, side='right')
        return int(self.ticks[index]) / self.rate

    def audit(self, draws=100000, seed=20261001):
        if draws < 100000:
            raise ValueError('The sampler audit requires at least 100000 draws')
        rng = stable_rng(seed, 'sampler-only-audit')
        indices = np.searchsorted(self.cdf, rng.random(draws), side='right')
        counts = np.bincount(self.bin_ids[indices], minlength=6)
        expected_probability = np.bincount(self.bin_ids, weights=self.probabilities, minlength=6)
        expected = expected_probability * draws
        tolerance = 6 * np.sqrt(draws * expected_probability * (1 - expected_probability)) + 5
        return dict(draws=draws, sample_type='sampler only; no real labels', seed=seed,
                    expected=expected.tolist(), observed=counts.tolist(), tolerance=tolerance.tolist(),
                    passed=bool(np.all(np.abs(counts - expected) <= tolerance)),
                    retained_probability_mass=self.retained_mass,
                    min_seconds=float(self.ticks.min() / self.rate), max_seconds=float(self.ticks.max() / self.rate),
                    duplicate_tick_fraction=float(1 - len(np.unique(self.ticks[indices])) / draws),
                    per_event_three_draw_duplicate_fraction=float(np.mean(np.any(np.diff(np.sort(self.ticks[indices[:draws//3*3]].reshape(-1,3),axis=1),axis=1)==0,axis=1))),
                    bin_boundaries='left inclusive / right exclusive; 90 included in final bin')
