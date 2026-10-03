"""Okapi BM25 with the document side of the formula precomputed."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from rageval.corpus import Corpus
from rageval.retrieval.base import Retriever
from rageval.text import analyze
from rageval.types import Retrieval


class BM25Retriever(Retriever):
    """Sparse lexical retrieval.

    Every factor in the BM25 score except the choice of query terms depends only
    on the document, so the whole weight matrix is built once at index time and a
    query becomes a column gather and a row sum. That turns per-query work from
    "loop over terms, loop over postings" into one sparse operation.
    """

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0:
            raise ValueError("k1 must not be negative")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must lie in [0, 1]")
        self.k1 = k1
        self.b = b
        self._vocab: dict[str, int] = {}
        self._weights: sparse.csc_matrix | None = None
        self._chunk_ids: list[str] = []

    def index(self, corpus: Corpus) -> None:
        docs = [analyze(chunk.text) for chunk in corpus.chunks]
        self._chunk_ids = corpus.chunk_ids

        vocab: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        tfs: list[float] = []
        lengths = np.zeros(len(docs), dtype=np.float32)

        for row, tokens in enumerate(docs):
            lengths[row] = len(tokens)
            counts: dict[int, int] = {}
            for token in tokens:
                col = vocab.setdefault(token, len(vocab))
                counts[col] = counts.get(col, 0) + 1
            rows.extend([row] * len(counts))
            cols.extend(counts.keys())
            tfs.extend(float(v) for v in counts.values())

        n_docs = len(docs)
        n_terms = len(vocab)
        if n_terms == 0:
            raise ValueError("corpus produced no indexable terms")

        tf = sparse.csr_matrix(
            (np.asarray(tfs, dtype=np.float32), (rows, cols)),
            shape=(n_docs, n_terms),
            dtype=np.float32,
        )

        avgdl = float(lengths.mean()) or 1.0
        # Per-document denominator term, broadcast across each row's postings.
        norm = (self.k1 * (1.0 - self.b + self.b * lengths / avgdl)).astype(np.float32)

        df = np.diff(tf.tocsc().indptr).astype(np.float32)
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        weights = tf.copy()
        row_index = np.repeat(np.arange(n_docs), np.diff(weights.indptr))
        data = weights.data
        weights.data = (data * (self.k1 + 1.0)) / (data + norm[row_index])
        weights.data *= idf[weights.indices]

        self._vocab = vocab
        # CSC because scoring slices columns, never rows.
        self._weights = weights.tocsc()

    def search(self, query: str, k: int) -> Retrieval:
        if self._weights is None:
            raise RuntimeError("BM25Retriever.search called before index")
        cols = [self._vocab[t] for t in analyze(query) if t in self._vocab]
        if not cols:
            return Retrieval(chunks=[])
        scores = self.score_columns(cols)
        return self._to_retrieval(self._chunk_ids, scores, k)

    def score_columns(self, cols: list[int]) -> NDArray[np.float32]:
        """Sum precomputed weights over the query's term columns."""
        assert self._weights is not None
        selected = self._weights[:, cols]
        return np.asarray(selected.sum(axis=1), dtype=np.float32).ravel()
