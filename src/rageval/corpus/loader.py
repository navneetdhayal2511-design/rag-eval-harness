"""Corpus loading, gold-evidence resolution and dataset validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from rageval.corpus.chunking import Chunker
from rageval.types import Chunk, Document, EvalQuestion, GoldEvidence, ResolvedEvidence

_WS_RE = re.compile(r"\s+")
_TITLE_RE = re.compile(r"^#[ \t]+(.+)$", re.MULTILINE)


class DatasetError(ValueError):
    """Raised when a dataset is internally inconsistent."""


def _normalise(text: str) -> tuple[str, list[int]]:
    """Lowercase and collapse whitespace, keeping a map back to original offsets.

    Returns ``(normalised_text, offsets)`` where ``offsets[i]`` is the index in
    the original string of the character that produced ``normalised_text[i]``.
    This lets gold quotes be matched tolerantly (line wrapping, double spaces)
    while still yielding exact spans in the source document.
    """
    out: list[str] = []
    offsets: list[int] = []
    pending_space = False
    for idx, ch in enumerate(text):
        if ch.isspace():
            pending_space = bool(out)
            continue
        if pending_space:
            out.append(" ")
            offsets.append(idx)
            pending_space = False
        out.append(ch.lower())
        offsets.append(idx)
    return "".join(out), offsets


@dataclass(frozen=True)
class Corpus:
    """Documents plus one concrete chunking of them."""

    documents: dict[str, Document]
    chunks: list[Chunk]
    _by_id: dict[str, Chunk] = field(init=False, repr=False, compare=False)
    _by_doc: dict[str, list[Chunk]] = field(init=False, repr=False, compare=False)
    _norm: dict[str, tuple[str, list[int]]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_id", {c.chunk_id: c for c in self.chunks})
        by_doc: dict[str, list[Chunk]] = {}
        for chunk in self.chunks:
            by_doc.setdefault(chunk.doc_id, []).append(chunk)
        object.__setattr__(self, "_by_doc", by_doc)
        object.__setattr__(
            self,
            "_norm",
            {doc_id: _normalise(doc.text) for doc_id, doc in self.documents.items()},
        )

    @property
    def chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self.chunks]

    def chunk(self, chunk_id: str) -> Chunk:
        return self._by_id[chunk_id]

    def resolve(self, evidence: GoldEvidence) -> ResolvedEvidence:
        """Locate a gold quote inside its document."""
        if evidence.doc_id not in self.documents:
            raise DatasetError(f"evidence references unknown document {evidence.doc_id!r}")
        hay, offsets = self._norm[evidence.doc_id]
        needle, _ = _normalise(evidence.quote)
        if not needle:
            raise DatasetError(f"empty gold quote for document {evidence.doc_id!r}")
        pos = hay.find(needle)
        if pos < 0:
            raise DatasetError(
                f"gold quote not found in {evidence.doc_id!r}: {evidence.quote[:70]!r}"
            )
        if hay.find(needle, pos + 1) >= 0:
            raise DatasetError(
                f"gold quote is ambiguous in {evidence.doc_id!r} (matches more than once): "
                f"{evidence.quote[:70]!r}"
            )
        return ResolvedEvidence(
            doc_id=evidence.doc_id,
            quote=evidence.quote,
            start=offsets[pos],
            end=offsets[pos + len(needle) - 1] + 1,
        )

    def relevant_chunks(self, question: EvalQuestion, min_overlap: float = 0.5) -> set[str]:
        """Chunk ids that carry enough of a gold quote to count as relevant.

        A chunk is relevant when it contains at least ``min_overlap`` of some gold
        span. Partial credit is deliberately excluded: half a clause is usually
        not enough for a reader to answer from.
        """
        relevant: set[str] = set()
        for evidence in question.evidence:
            span = self.resolve(evidence)
            length = span.end - span.start
            for chunk in self._by_doc.get(span.doc_id, []):
                if chunk.overlap(span.start, span.end) >= min_overlap * length:
                    relevant.add(chunk.chunk_id)
        return relevant


def load_documents(directory: Path) -> dict[str, Document]:
    """Load every ``.md`` file in ``directory`` as one document."""
    docs: dict[str, Document] = {}
    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise DatasetError(f"no .md documents found under {directory}")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        match = _TITLE_RE.search(text)
        docs[path.stem] = Document(
            doc_id=path.stem,
            title=match.group(1).strip() if match else path.stem,
            text=text,
            metadata={"source": path.name},
        )
    return docs


def load_questions(path: Path) -> list[EvalQuestion]:
    """Load the golden set from JSONL."""
    questions: list[EvalQuestion] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path.name}:{lineno} is not valid JSON: {exc}") from exc
        question = EvalQuestion.model_validate(payload)
        if question.qid in seen:
            raise DatasetError(f"{path.name}:{lineno} duplicate qid {question.qid!r}")
        seen.add(question.qid)
        questions.append(question)
    if not questions:
        raise DatasetError(f"{path} contains no questions")
    return questions


def build_corpus(directory: Path, chunker: Chunker) -> Corpus:
    documents = load_documents(directory)
    chunks: list[Chunk] = []
    for doc in documents.values():
        chunks.extend(chunker.split(doc))
    if not chunks:
        raise DatasetError("chunking produced no chunks")
    return Corpus(documents=documents, chunks=chunks)


def validate_dataset(corpus: Corpus, questions: list[EvalQuestion]) -> list[str]:
    """Check that every gold quote resolves and is actually reachable.

    Returns a list of human-readable problems; empty means the dataset is sound.
    """
    problems: list[str] = []
    for question in questions:
        if question.unanswerable:
            if question.evidence:
                problems.append(f"{question.qid}: marked unanswerable but carries gold evidence")
            continue
        try:
            relevant = corpus.relevant_chunks(question)
        except DatasetError as exc:
            problems.append(f"{question.qid}: {exc}")
            continue
        if not relevant:
            problems.append(
                f"{question.qid}: gold evidence resolves but no chunk covers it, so the "
                "question is unreachable under this chunking"
            )
    return problems
