"""Uncertainty estimation.

A golden set of a few hundred questions produces metric differences that are
routinely smaller than the sampling noise. Every number this harness reports
therefore carries an interval, and every A/B claim goes through a paired test.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Resampling in blocks keeps peak memory bounded for large golden sets.
_MAX_CELLS = 20_000_000


@dataclass(frozen=True)
class Interval:
    value: float
    low: float
    high: float
    n: int


@dataclass(frozen=True)
class Comparison:
    """Result of comparing one metric across two runs on the same questions."""

    metric: str
    baseline: float
    candidate: float
    delta: float
    delta_low: float
    delta_high: float
    p_value: float
    n: int

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    @property
    def direction(self) -> str:
        if not self.significant:
            return "no change"
        return "improved" if self.delta > 0 else "regressed"


def _resample_means(
    values: NDArray[np.float64], n_resamples: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    n = values.size
    block = max(1, min(n_resamples, _MAX_CELLS // max(n, 1)))
    means = np.empty(n_resamples, dtype=np.float64)
    done = 0
    while done < n_resamples:
        size = min(block, n_resamples - done)
        idx = rng.integers(0, n, size=(size, n))
        means[done : done + size] = values[idx].mean(axis=1)
        done += size
    return means


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap interval for the mean of ``values``."""
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return Interval(value=0.0, low=0.0, high=0.0, n=0)
    mean = float(data.mean())
    if data.size == 1 or np.allclose(data, data[0]):
        return Interval(value=mean, low=mean, high=mean, n=int(data.size))

    rng = np.random.default_rng(seed)
    means = _resample_means(data, n_resamples, rng)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return Interval(value=mean, low=float(low), high=float(high), n=int(data.size))


def paired_permutation_test(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    n_resamples: int = 10000,
    seed: int = 0,
) -> float:
    """Two-sided p-value for a paired difference, by random sign flipping.

    The questions are the same in both runs, so the pairing carries most of the
    signal. Sign flipping is the exact randomisation test for that design and
    assumes nothing about the metric's distribution, which matters because
    recall@k and exact match are bounded and far from normal.
    """
    a = np.asarray(baseline, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired test requires runs over the same questions")
    diffs = b - a
    nonzero = diffs[diffs != 0.0]
    if nonzero.size == 0:
        return 1.0

    observed = abs(float(diffs.mean()))
    rng = np.random.default_rng(seed)
    n = nonzero.size
    base = float(diffs.sum() - nonzero.sum())

    block = max(1, min(n_resamples, _MAX_CELLS // n))
    extreme = 0
    done = 0
    while done < n_resamples:
        size = min(block, n_resamples - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, n))
        sampled = np.abs((signs * nonzero).sum(axis=1) + base) / diffs.size
        extreme += int((sampled >= observed - 1e-12).sum())
        done += size
    # Add-one correction: a randomisation test never reports p == 0.
    return (extreme + 1) / (n_resamples + 1)


def compare(
    metric: str,
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    seed: int = 0,
    n_resamples: int = 2000,
) -> Comparison:
    """Paired comparison of one metric between two runs."""
    a = np.asarray(baseline, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired comparison requires runs over the same questions")
    delta = bootstrap_ci(list(b - a), n_resamples=n_resamples, seed=seed)
    return Comparison(
        metric=metric,
        baseline=float(a.mean()) if a.size else 0.0,
        candidate=float(b.mean()) if b.size else 0.0,
        delta=delta.value,
        delta_low=delta.low,
        delta_high=delta.high,
        p_value=paired_permutation_test(a.tolist(), b.tolist(), seed=seed),
        n=int(a.size),
    )


def cohens_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Chance-corrected agreement between two sets of categorical labels."""
    x = np.asarray(a)
    y = np.asarray(b)
    if x.shape != y.shape:
        raise ValueError("label sequences must be the same length")
    if x.size == 0:
        return 0.0

    labels = np.unique(np.concatenate([x, y]))
    if labels.size == 1:
        return 1.0

    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((labels.size, labels.size), dtype=np.float64)
    for xi, yi in zip(x, y, strict=True):
        matrix[index[xi], index[yi]] += 1
    matrix /= matrix.sum()

    observed = float(np.trace(matrix))
    expected = float(matrix.sum(axis=0) @ matrix.sum(axis=1))
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)
