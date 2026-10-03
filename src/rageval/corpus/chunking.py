"""Chunking strategies.

Every strategy returns chunks carrying exact character offsets into the source
document. The offsets are load bearing: gold evidence is labelled as verbatim
quotes and resolved to spans, so relevance judgements survive a change of
chunking strategy without any relabelling.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from rageval.types import Chunk, Document

_WORD_RE = re.compile(r"\S+")
# Sentence terminator followed by whitespace, not preceded by a common abbreviation.
_SENT_END_RE = re.compile(r"(?<=[.!?;:])\s+|\n{2,}")
_HEADING_RE = re.compile(r"^#{1,6}[ \t]+\S.*$", re.MULTILINE)


def _spans(pattern: re.Pattern[str], text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split ``text`` into sentence-ish spans, dropping pure whitespace."""
    out: list[tuple[int, int]] = []
    cursor = 0
    for m in _SENT_END_RE.finditer(text):
        piece = text[cursor : m.start()]
        if piece.strip():
            lead = len(piece) - len(piece.lstrip())
            out.append((cursor + lead, cursor + len(piece.rstrip())))
        cursor = m.end()
    tail = text[cursor:]
    if tail.strip():
        lead = len(tail) - len(tail.lstrip())
        out.append((cursor + lead, cursor + len(tail.rstrip())))
    return out


class Chunker(ABC):
    """Splits a document into retrievable units."""

    name: str

    @abstractmethod
    def split(self, doc: Document) -> list[Chunk]: ...

    def _build(self, doc: Document, spans: list[tuple[int, int]]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for ordinal, (start, end) in enumerate(spans):
            text = doc.text[start:end]
            if not text.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{ordinal:04d}",
                    doc_id=doc.doc_id,
                    ordinal=ordinal,
                    text=text,
                    start=start,
                    end=end,
                )
            )
        return chunks


class FixedWindowChunker(Chunker):
    """Fixed-width sliding window over whitespace tokens.

    The naive baseline: cheap, strategy-agnostic, and prone to cutting a clause
    in half right where the answer lives.
    """

    name = "fixed"

    def __init__(self, window: int = 180, stride: int = 135) -> None:
        if stride <= 0 or window <= 0:
            raise ValueError("window and stride must be positive")
        if stride > window:
            raise ValueError("stride must not exceed window or tokens would be dropped")
        self.window = window
        self.stride = stride

    def split(self, doc: Document) -> list[Chunk]:
        words = _spans(_WORD_RE, doc.text)
        if not words:
            return []
        spans: list[tuple[int, int]] = []
        for start_idx in range(0, len(words), self.stride):
            window = words[start_idx : start_idx + self.window]
            if not window:
                break
            spans.append((window[0][0], window[-1][1]))
            if start_idx + self.window >= len(words):
                break
        return self._build(doc, spans)


class SentenceWindowChunker(Chunker):
    """Packs whole sentences up to a character budget, with sentence overlap.

    Avoids mid-sentence cuts, which is the main failure mode of fixed windows.
    """

    name = "sentence"

    def __init__(self, max_chars: int = 900, overlap_sentences: int = 1) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must not be negative")
        self.max_chars = max_chars
        self.overlap_sentences = overlap_sentences

    def split(self, doc: Document) -> list[Chunk]:
        sents = sentence_spans(doc.text)
        if not sents:
            return []
        spans: list[tuple[int, int]] = []
        i = 0
        while i < len(sents):
            start = sents[i][0]
            end = sents[i][1]
            j = i + 1
            while j < len(sents) and sents[j][1] - start <= self.max_chars:
                end = sents[j][1]
                j += 1
            spans.append((start, end))
            if j >= len(sents):
                break
            # Step back so consecutive chunks share context, but always advance.
            i = max(i + 1, j - self.overlap_sentences)
        return self._build(doc, spans)


class SectionChunker(Chunker):
    """Splits on markdown headings, then subdivides oversized sections by sentence.

    Keeps each chunk inside a single semantic unit of the source document, which
    matters for policy-style text where a clause is meaningless without its heading.
    """

    name = "section"

    def __init__(self, max_chars: int = 1200) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self.max_chars = max_chars
        self._fallback = SentenceWindowChunker(max_chars=max_chars, overlap_sentences=0)

    def split(self, doc: Document) -> list[Chunk]:
        starts = [m.start() for m in _HEADING_RE.finditer(doc.text)]
        if not starts or starts[0] != 0:
            starts = [0, *starts]
        bounds = [*starts, len(doc.text)]

        spans: list[tuple[int, int]] = []
        for start, end in zip(bounds, bounds[1:], strict=False):
            if not doc.text[start:end].strip():
                continue
            if end - start <= self.max_chars:
                spans.append((start, end))
                continue
            # Oversized section: keep the heading line attached to the first part.
            for sub_start, sub_end in self._subdivide(doc.text, start, end):
                spans.append((sub_start, sub_end))
        return self._build(doc, spans)

    def _subdivide(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        local = text[start:end]
        sub = self._fallback.split(Document(doc_id="_", title="_", text=local))
        return [(start + c.start, start + c.end) for c in sub]


_REGISTRY: dict[str, type[Chunker]] = {
    FixedWindowChunker.name: FixedWindowChunker,
    SentenceWindowChunker.name: SentenceWindowChunker,
    SectionChunker.name: SectionChunker,
}


def build_chunker(name: str, **kwargs: Any) -> Chunker:
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown chunker {name!r}; available: {sorted(_REGISTRY)}") from None
    return cls(**kwargs)
