"""Second-stage reranking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from rageval.corpus import Corpus
from rageval.embeddings import Embedder
from rageval.retrieval.base import Retriever
from rageval.types import Retrieval, ScoredChunk


class Reranker(ABC):
    name: str

    def index(self, corpus: Corpus) -> None:
        """Optional hook for rerankers that need corpus state."""

    @abstractmethod
    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]: ...


class MMRReranker(Reranker):
    """Maximal marginal relevance.

    First-stage retrieval over an overlapping chunking tends to return several
    near-duplicate passages, which wastes the generator's context window on the
    same sentence three times. MMR trades a little relevance for coverage:
    ``score = lambda * sim(q, c) - (1 - lambda) * max sim(c, already_selected)``.
    """

    name = "mmr"

    def __init__(self, embedder: Embedder, diversity: float = 0.3) -> None:
        if not 0.0 <= diversity <= 1.0:
            raise ValueError("diversity must lie in [0, 1]")
        self.embedder = embedder
        self.diversity = diversity
        self._texts: dict[str, str] = {}

    def index(self, corpus: Corpus) -> None:
        self._texts = {c.chunk_id: c.text for c in corpus.chunks}
        self.embedder.fit([c.text for c in corpus.chunks])

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        if len(candidates) <= 1:
            return candidates[:k]

        ids = [c.chunk_id for c in candidates]
        vectors = self.embedder.encode([self._texts[cid] for cid in ids])
        query_vector = self.embedder.encode([query])[0]
        relevance = vectors @ query_vector
        similarity = vectors @ vectors.T

        lam = 1.0 - self.diversity
        selected: list[int] = []
        remaining = set(range(len(ids)))
        best_sim = np.full(len(ids), -np.inf, dtype=np.float32)

        while remaining and len(selected) < k:
            pool = np.fromiter(sorted(remaining), dtype=np.int64)
            if selected:
                objective = lam * relevance[pool] - self.diversity * best_sim[pool]
            else:
                objective = relevance[pool]
            pick = int(pool[int(np.argmax(objective))])
            selected.append(pick)
            remaining.discard(pick)
            # Running maximum avoids rescoring the whole selected set each round.
            np.maximum(best_sim, similarity[pick], out=best_sim)

        return [ScoredChunk(chunk_id=ids[i], score=float(relevance[i])) for i in selected]


class CrossEncoderReranker(Reranker):
    """Cross-encoder reranking, when a local model is available."""

    name = "cross-encoder"

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model
        self._model: Any = None
        self._texts: dict[str, str] = {}

    def index(self, corpus: Corpus) -> None:
        self._texts = {c.chunk_id: c.text for c in corpus.chunks}

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        pairs = [(query, self._texts[c.chunk_id]) for c in candidates]
        scores = np.asarray(self._model.predict(pairs), dtype=np.float32)
        order = np.lexsort((np.arange(len(candidates)), -scores))[:k]
        return [
            ScoredChunk(chunk_id=candidates[int(i)].chunk_id, score=float(scores[int(i)]))
            for i in order
        ]


class RerankingRetriever(Retriever):
    """Wraps a first-stage retriever with a reranking second stage."""

    name = "reranked"

    def __init__(self, base: Retriever, reranker: Reranker, candidate_multiplier: int = 4) -> None:
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least 1")
        self.base = base
        self.reranker = reranker
        self.candidate_multiplier = candidate_multiplier

    def index(self, corpus: Corpus) -> None:
        self.base.index(corpus)
        self.reranker.index(corpus)

    def search(self, query: str, k: int) -> Retrieval:
        candidates = self.base.search(query, k * self.candidate_multiplier)
        return Retrieval(chunks=self.reranker.rerank(query, candidates.chunks, k))
