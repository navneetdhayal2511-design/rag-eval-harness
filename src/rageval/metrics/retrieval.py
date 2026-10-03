"""Ranking quality metrics.

Each function scores a single query. Aggregation and confidence intervals are
the harness's job, so that per-question scores stay available for paired tests.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def hit_rate_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant chunk made the top k. The floor the system must clear."""
    return float(any(c in relevant for c in retrieved[:k]))


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the gold evidence that the top k actually surfaced."""
    if not relevant:
        return 0.0
    found = sum(1 for c in retrieved[:k] if c in relevant)
    return found / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top k that is gold. Low precision is what burns context budget."""
    window = retrieved[:k]
    if not window:
        return 0.0
    return sum(1 for c in window if c in relevant) / len(window)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant chunk, 0 if none was retrieved."""
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance.

    Unlike recall, nDCG cares where in the list the evidence landed, which is the
    thing that actually matters once a context window forces a cutoff.
    """
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved[:k], start=1)
        if chunk_id in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0
