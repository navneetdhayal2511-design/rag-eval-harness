"""Dense retrieval over L2-normalised embeddings."""

from __future__ import annotations

import numpy as np

from rageval.corpus import Corpus
from rageval.embeddings import Embedder, Matrix
from rageval.retrieval.base import Retriever
from rageval.types import Retrieval


class DenseRetriever(Retriever):
    """Cosine similarity search.

    Vectors are normalised once at index time, so similarity is a single
    matrix-vector product and no per-query normalisation is needed.
    """

    name = "dense"

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._matrix: Matrix | None = None
        self._chunk_ids: list[str] = []

    def index(self, corpus: Corpus) -> None:
        texts = [chunk.text for chunk in corpus.chunks]
        self.embedder.fit(texts)
        self._matrix = self.embedder.encode(texts)
        self._chunk_ids = corpus.chunk_ids

    def search(self, query: str, k: int) -> Retrieval:
        if self._matrix is None:
            raise RuntimeError("DenseRetriever.search called before index")
        vector = self.embedder.encode([query])[0]
        scores = (self._matrix @ vector).astype(np.float32, copy=False)
        return self._to_retrieval(self._chunk_ids, scores, k)

    @property
    def matrix(self) -> Matrix:
        if self._matrix is None:
            raise RuntimeError("retriever is not indexed")
        return self._matrix
