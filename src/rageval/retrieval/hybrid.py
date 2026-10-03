"""Fusion of lexical and dense rankings."""

from __future__ import annotations

from typing import Literal

from rageval.corpus import Corpus
from rageval.retrieval.base import Retriever
from rageval.types import Retrieval, ScoredChunk

FusionMode = Literal["rrf", "weighted"]


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int, smoothing: int = 60, weights: list[float] | None = None
) -> list[ScoredChunk]:
    """Combine rankings by rank position rather than by score.

    Rank fusion is used by default because BM25 scores and cosine similarities
    live on incomparable scales, and any attempt to normalise them is sensitive
    to the score distribution of whichever query happens to be running.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights and rankings must be the same length")

    totals: dict[str, float] = {}
    for weight, ranking in zip(weights, rankings, strict=True):
        for rank, chunk_id in enumerate(ranking, start=1):
            totals[chunk_id] = totals.get(chunk_id, 0.0) + weight / (smoothing + rank)
    ordered = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    return [ScoredChunk(chunk_id=cid, score=score) for cid, score in ordered[:k]]


def weighted_score_fusion(
    results: list[Retrieval], k: int, weights: list[float] | None = None
) -> list[ScoredChunk]:
    """Min-max normalise each ranker's scores, then take a weighted sum.

    Keeps score magnitude information that rank fusion throws away, at the cost
    of being sensitive to outliers in the score distribution.
    """
    if weights is None:
        weights = [1.0] * len(results)
    if len(weights) != len(results):
        raise ValueError("weights and results must be the same length")

    totals: dict[str, float] = {}
    for weight, result in zip(weights, results, strict=True):
        if not result.chunks:
            continue
        values = [c.score for c in result.chunks]
        lo, hi = min(values), max(values)
        spread = hi - lo
        for chunk in result.chunks:
            scaled = 1.0 if spread <= 0 else (chunk.score - lo) / spread
            totals[chunk.chunk_id] = totals.get(chunk.chunk_id, 0.0) + weight * scaled
    ordered = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    return [ScoredChunk(chunk_id=cid, score=score) for cid, score in ordered[:k]]


class HybridRetriever(Retriever):
    """Runs several retrievers and fuses their rankings."""

    name = "hybrid"

    def __init__(
        self,
        retrievers: list[Retriever],
        mode: FusionMode = "rrf",
        weights: list[float] | None = None,
        candidate_multiplier: int = 4,
        rrf_smoothing: int = 60,
    ) -> None:
        if not retrievers:
            raise ValueError("hybrid retrieval needs at least one retriever")
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least 1")
        self.retrievers = retrievers
        self.mode = mode
        self.weights = weights
        self.candidate_multiplier = candidate_multiplier
        self.rrf_smoothing = rrf_smoothing

    def index(self, corpus: Corpus) -> None:
        for retriever in self.retrievers:
            retriever.index(corpus)

    def search(self, query: str, k: int) -> Retrieval:
        # Each arm must go deeper than k, or fusion has nothing to disagree about.
        depth = k * self.candidate_multiplier
        results = [r.search(query, depth) for r in self.retrievers]
        if self.mode == "rrf":
            chunks = reciprocal_rank_fusion(
                [r.chunk_ids for r in results],
                k=k,
                smoothing=self.rrf_smoothing,
                weights=self.weights,
            )
        else:
            chunks = weighted_score_fusion(results, k=k, weights=self.weights)
        return Retrieval(chunks=chunks)
