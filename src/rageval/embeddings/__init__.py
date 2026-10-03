from typing import Any

from rageval.embeddings.base import Embedder, Matrix, TfidfSvdEmbedder, l2_normalise

__all__ = ["Embedder", "Matrix", "TfidfSvdEmbedder", "build_embedder", "l2_normalise"]


def build_embedder(name: str, **kwargs: Any) -> Embedder:
    """Construct an embedder by name, importing optional providers lazily."""
    if name == TfidfSvdEmbedder.name:
        return TfidfSvdEmbedder(**kwargs)
    if name == "openai":
        from rageval.embeddings.providers import OpenAIEmbedder

        return OpenAIEmbedder(**kwargs)
    if name == "sbert":
        from rageval.embeddings.providers import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(**kwargs)
    raise ValueError(f"unknown embedder {name!r}; available: ['tfidf-svd', 'openai', 'sbert']")
