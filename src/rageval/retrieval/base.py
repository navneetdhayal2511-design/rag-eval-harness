"""Retriever interface and shared ranking helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from rageval.corpus import Corpus
from rageval.types import Retrieval, ScoredChunk


def top_k_indices(scores: NDArray[np.float32], k: int) -> NDArray[np.int64]:
    """Indices of the ``k`` highest scores, best first.

    Uses a partial selection rather than a full sort: ranking only ever needs the
    head of the list, and the corpus side of that comparison is the large one.
    Ties break on index so runs stay byte-for-byte reproducible.
    """
    k = min(k, scores.size)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    candidates = np.argpartition(-scores, k - 1)[:k] if k < scores.size else np.arange(scores.size)
    order = np.lexsort((candidates, -scores[candidates]))
    return candidates[order].astype(np.int64, copy=False)


class Retriever(ABC):
    """Ranks corpus chunks against a query."""

    name: str

    @abstractmethod
    def index(self, corpus: Corpus) -> None: ...

    @abstractmethod
    def search(self, query: str, k: int) -> Retrieval: ...

    def _to_retrieval(self, chunk_ids: list[str], scores: NDArray[np.float32], k: int) -> Retrieval:
        picks = top_k_indices(scores, k)
        return Retrieval(
            chunks=[
                ScoredChunk(chunk_id=chunk_ids[int(i)], score=float(scores[int(i)])) for i in picks
            ]
        )
