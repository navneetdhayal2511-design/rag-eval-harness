"""Hosted and pretrained embedders.

These are optional extras. The harness runs end to end without them so that CI
and a fresh clone never depend on a key or a model download.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np

from rageval.embeddings.base import Embedder, Matrix, l2_normalise


class OpenAIEmbedder(Embedder):
    """OpenAI embeddings, batched and disk-cached by content hash."""

    name = "openai"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 128,
        cache_dir: str | None = ".rageval_cache",
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self._cache_dir = cache_dir
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set; use the default tfidf-svd embedder or export a key"
                )
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def encode(self, texts: Sequence[str]) -> Matrix:
        from rageval.harness.cache import DiskCache

        cache = DiskCache(self._cache_dir) if self._cache_dir else None
        vectors: list[list[float]] = [[] for _ in texts]
        pending: list[int] = []

        for i, text in enumerate(texts):
            hit = cache.get("embed", self.model, text) if cache else None
            if hit is None:
                pending.append(i)
            else:
                vectors[i] = hit["vector"]

        if pending:
            client = self._get_client()
            for start in range(0, len(pending), self.batch_size):
                batch_idx = pending[start : start + self.batch_size]
                batch = [texts[i] for i in batch_idx]
                response = client.embeddings.create(model=self.model, input=batch)
                for i, item in zip(batch_idx, response.data, strict=True):
                    vectors[i] = list(item.embedding)
                    if cache:
                        cache.put("embed", self.model, texts[i], {"vector": vectors[i]})

        return l2_normalise(np.asarray(vectors, dtype=np.float32))


class SentenceTransformerEmbedder(Embedder):
    """Local sentence-transformers model, for when a real encoder is available."""

    name = "sbert"

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> Matrix:
        model = self._load()
        vectors = model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        return l2_normalise(np.asarray(vectors, dtype=np.float32))
