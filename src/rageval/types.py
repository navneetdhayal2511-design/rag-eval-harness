"""Core data model shared by the corpus, retrieval, generation and metric layers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Document(Frozen):
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(Frozen):
    """A contiguous slice of a document.

    ``start`` and ``end`` are character offsets into ``Document.text`` and are what
    allow gold evidence to be resolved against any chunking strategy.
    """

    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    start: int
    end: int

    @model_validator(mode="after")
    def _check_span(self) -> Chunk:
        if self.end <= self.start:
            raise ValueError(f"chunk {self.chunk_id} has empty span [{self.start}, {self.end})")
        return self

    def overlap(self, start: int, end: int) -> int:
        """Number of characters shared with the span ``[start, end)``."""
        return max(0, min(self.end, end) - max(self.start, start))


class GoldEvidence(Frozen):
    """A verbatim quote from a source document that supports the reference answer.

    Quotes are resolved to character spans at load time, which keeps the labels
    independent of how the corpus happens to be chunked.
    """

    doc_id: str
    quote: str


class EvalQuestion(Frozen):
    qid: str
    question: str
    reference_answer: str
    evidence: list[GoldEvidence]
    category: str = "general"
    # Questions whose correct behaviour is to refuse / say the corpus does not cover it.
    unanswerable: bool = False

    @model_validator(mode="after")
    def _check_evidence(self) -> EvalQuestion:
        if not self.unanswerable and not self.evidence:
            raise ValueError(f"question {self.qid} is answerable but has no gold evidence")
        return self


class ResolvedEvidence(Frozen):
    """Gold evidence after its quote has been located inside the document text."""

    doc_id: str
    quote: str
    start: int
    end: int


class ScoredChunk(Frozen):
    chunk_id: str
    score: float


class Retrieval(Frozen):
    """Ranked retrieval output for a single query."""

    chunks: list[ScoredChunk]

    @property
    def chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self.chunks]

    def top(self, k: int) -> list[str]:
        return self.chunk_ids[:k]


class Answer(Frozen):
    text: str
    # Chunk ids the answer claims to be grounded in.
    citations: list[str] = Field(default_factory=list)
    abstained: bool = False


class QueryRecord(Frozen):
    """Everything produced for one question by one pipeline configuration."""

    qid: str
    retrieval: Retrieval
    answer: Answer
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.retrieval_ms + self.generation_ms


class MetricValue(Frozen):
    """A metric point estimate with a bootstrap confidence interval."""

    name: str
    value: float
    ci_low: float
    ci_high: float
    n: int

    def format(self) -> str:
        return f"{self.value:.3f} [{self.ci_low:.3f}, {self.ci_high:.3f}]"


class RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    pipeline: str
    config: dict[str, Any]
    records: list[QueryRecord]
    metrics: dict[str, MetricValue]
    # Raw per-question scores, kept so two runs can be compared with a paired test.
    per_question: dict[str, dict[str, float]]
