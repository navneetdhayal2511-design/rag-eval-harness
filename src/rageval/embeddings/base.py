"""Embedder interface and the offline default."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rageval.text import analyze

Matrix = NDArray[np.float32]


def l2_normalise(matrix: Matrix) -> Matrix:
    """Row-normalise so cosine similarity reduces to a single matrix product."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    normalised: Matrix = (matrix / norms).astype(np.float32, copy=False)
    return normalised


class Embedder(ABC):
    """Maps text to L2-normalised dense vectors."""

    name: str

    def fit(self, corpus: Sequence[str]) -> None:
        """Optional corpus-fitting hook. No-op for pretrained models."""

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> Matrix: ...


class TfidfSvdEmbedder(Embedder):
    """Latent semantic indexing: TF-IDF over word and character n-grams, then SVD.

    This is the default because it needs no network, no API key and no GPU, yet
    still produces genuinely distributional vectors, so a dense run measures
    something different from BM25. Results are deterministic given a seed, which
    is what makes the regression gate meaningful.
    """

    name = "tfidf-svd"

    def __init__(self, dim: int = 256, seed: int = 0, min_df: int = 1) -> None:
        if dim < 2:
            raise ValueError("dim must be at least 2")
        self.dim = dim
        self.seed = seed
        self.min_df = min_df
        self._vectorizer: Any = None
        self._svd: Any = None

    def fit(self, corpus: Sequence[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not corpus:
            raise ValueError("cannot fit an embedder on an empty corpus")

        vectorizer = TfidfVectorizer(
            analyzer=analyze,
            sublinear_tf=True,
            min_df=min(self.min_df, len(corpus)),
        )
        tfidf = vectorizer.fit_transform(corpus)
        # SVD cannot produce more components than the matrix has rank.
        n_components = min(self.dim, min(tfidf.shape) - 1)
        if n_components < 1:
            raise ValueError("corpus is too small to fit a latent semantic model")
        svd = TruncatedSVD(n_components=n_components, random_state=self.seed, algorithm="arpack")
        svd.fit(tfidf)
        self._vectorizer = vectorizer
        self._svd = svd

    def encode(self, texts: Sequence[str]) -> Matrix:
        if self._vectorizer is None or self._svd is None:
            raise RuntimeError("TfidfSvdEmbedder.encode called before fit")
        tfidf = self._vectorizer.transform(texts)
        dense = self._svd.transform(tfidf)
        return l2_normalise(np.asarray(dense, dtype=np.float32))

    @property
    def explained_variance(self) -> float:
        if self._svd is None:
            return 0.0
        return float(self._svd.explained_variance_ratio_.sum())
