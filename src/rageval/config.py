"""Declarative pipeline configuration.

A pipeline is data, not code, so an ablation is a sweep over config files and
the exact configuration that produced a number is recorded alongside it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rageval.corpus import Chunker, build_chunker
from rageval.embeddings import build_embedder
from rageval.generation import Answerer, build_answerer
from rageval.metrics.judge import Judge, build_judge
from rageval.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankingRetriever,
    Retriever,
)
from rageval.retrieval.rerank import CrossEncoderReranker, MMRReranker, Reranker


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComponentConfig(Base):
    name: str
    options: dict[str, Any] = Field(default_factory=dict)


class RetrieverConfig(Base):
    type: Literal["bm25", "dense", "hybrid"]
    options: dict[str, Any] = Field(default_factory=dict)
    embedder: ComponentConfig | None = None
    # Only used when type == "hybrid".
    arms: list[RetrieverConfig] = Field(default_factory=list)


RetrieverConfig.model_rebuild()


class RerankConfig(Base):
    type: Literal["mmr", "cross-encoder"]
    options: dict[str, Any] = Field(default_factory=dict)
    embedder: ComponentConfig | None = None
    candidate_multiplier: int = 4


class DatasetConfig(Base):
    corpus_dir: Path
    questions: Path
    human_labels: Path | None = None


class PipelineConfig(Base):
    """One fully specified retrieval-and-answering pipeline."""

    name: str
    dataset: DatasetConfig
    chunker: ComponentConfig = ComponentConfig(name="section")
    retriever: RetrieverConfig
    rerank: RerankConfig | None = None
    answerer: ComponentConfig = ComponentConfig(name="extractive")
    judge: ComponentConfig = ComponentConfig(name="lexical")
    top_k: int = 5
    # Cutoffs the retrieval metrics are reported at.
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5])
    # Characters of retrieved text the generator is allowed to see. When set,
    # the harness additionally reports metrics at this budget, which is the only
    # fair way to compare chunking strategies: at a fixed k, a coarse chunker is
    # handed several times more text than a fine one and wins on recall for that
    # reason alone.
    context_chars: int | None = None
    workers: int = 8
    seed: int = 0

    @model_validator(mode="after")
    def _check_cutoffs(self) -> PipelineConfig:
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        # A cutoff deeper than the retrieved list silently duplicates the
        # deepest real cutoff, which reads as a second data point and is not one.
        over = [k for k in self.k_values if k > self.top_k]
        if over:
            raise ValueError(
                f"k_values {over} exceed top_k={self.top_k}; raise top_k or lower the cutoffs"
            )
        if self.context_chars is not None and self.context_chars < 1:
            raise ValueError("context_chars must be positive when set")
        return self

    @classmethod
    def load(cls, path: Path) -> PipelineConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path} must contain a YAML mapping")
        config = cls.model_validate(raw)
        # Dataset paths are written relative to the config file so sweeps are portable.
        root = path.parent
        return config.model_copy(
            update={
                "dataset": config.dataset.model_copy(
                    update={
                        "corpus_dir": (root / config.dataset.corpus_dir).resolve(),
                        "questions": (root / config.dataset.questions).resolve(),
                        "human_labels": (
                            (root / config.dataset.human_labels).resolve()
                            if config.dataset.human_labels
                            else None
                        ),
                    }
                )
            }
        )

    def fingerprint(self) -> dict[str, Any]:
        """The config as plain data, stored in every run artefact."""
        return self.model_dump(mode="json", exclude={"workers"})


def build_chunker_from(config: ComponentConfig) -> Chunker:
    return build_chunker(config.name, **config.options)


def _build_retriever(config: RetrieverConfig, seed: int) -> Retriever:
    if config.type == "bm25":
        return BM25Retriever(**config.options)
    if config.type == "dense":
        spec = config.embedder or ComponentConfig(name="tfidf-svd")
        options = dict(spec.options)
        if spec.name == "tfidf-svd":
            options.setdefault("seed", seed)
        return DenseRetriever(build_embedder(spec.name, **options))
    if not config.arms:
        raise ValueError("hybrid retriever requires at least one arm")
    arms = [_build_retriever(arm, seed) for arm in config.arms]
    return HybridRetriever(arms, **config.options)


def _build_reranker(config: RerankConfig, seed: int) -> Reranker:
    if config.type == "mmr":
        spec = config.embedder or ComponentConfig(name="tfidf-svd")
        options = dict(spec.options)
        if spec.name == "tfidf-svd":
            options.setdefault("seed", seed)
        return MMRReranker(build_embedder(spec.name, **options), **config.options)
    return CrossEncoderReranker(**config.options)


def build_retriever(config: PipelineConfig) -> Retriever:
    retriever = _build_retriever(config.retriever, config.seed)
    if config.rerank is not None:
        retriever = RerankingRetriever(
            retriever,
            _build_reranker(config.rerank, config.seed),
            candidate_multiplier=config.rerank.candidate_multiplier,
        )
    return retriever


def build_answerer_from(config: ComponentConfig) -> Answerer:
    return build_answerer(config.name, **config.options)


def build_judge_from(config: ComponentConfig) -> Judge:
    return build_judge(config.name, **config.options)
