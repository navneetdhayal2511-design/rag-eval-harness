"""Answer generation.

The default answerer is extractive and deterministic. That is a deliberate
choice: a regression gate is only meaningful if the same input produces the same
output, and it keeps the whole pipeline runnable with no API key.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from rageval.corpus.chunking import sentence_spans
from rageval.text import analyze
from rageval.types import Answer, Chunk

ABSTAIN_TEXT = "The provided documents do not contain enough information to answer this."


class Answerer(ABC):
    name: str

    @abstractmethod
    def answer(self, question: str, contexts: list[Chunk]) -> Answer: ...


class ExtractiveAnswerer(Answerer):
    """Selects the sentences from the retrieved context that best cover the query.

    Sentences are scored by IDF-weighted query overlap, with IDF estimated over
    the retrieved context itself so that terms repeated across every passage stop
    driving the selection. Length normalisation keeps a long sentence from
    winning purely by containing more words.

    ``min_score`` is a floor for the case where nothing in the context relates to
    the query at all. It is not a calibrated abstention signal: a lexical scorer
    cannot tell "the corpus does not cover this" from "the question is phrased
    unusually", so this answerer abstains only in the degenerate case. See the
    abstention section of the README for the measurements behind that.
    """

    name = "extractive"

    def __init__(self, max_sentences: int = 3, min_score: float = 0.35) -> None:
        if max_sentences < 1:
            raise ValueError("max_sentences must be at least 1")
        self.max_sentences = max_sentences
        self.min_score = min_score

    def answer(self, question: str, contexts: list[Chunk]) -> Answer:
        candidates: list[tuple[int, int, str, list[str]]] = []
        for ctx_rank, chunk in enumerate(contexts):
            for start, end in sentence_spans(chunk.text):
                text = chunk.text[start:end].strip()
                if len(text) < 20:
                    continue
                candidates.append((ctx_rank, len(candidates), text, analyze(text)))

        if not candidates:
            return Answer(text=ABSTAIN_TEXT, citations=[], abstained=True)

        idf = self._idf([tokens for *_, tokens in candidates])
        query_tokens = set(analyze(question))

        scored: list[tuple[float, int, int]] = []
        for ctx_rank, order, _, tokens in candidates:
            if not tokens:
                continue
            matched = query_tokens & set(tokens)
            if not matched:
                continue
            weight = sum(idf[t] for t in matched)
            # Mild length normalisation, plus a small preference for better-ranked context.
            score = weight / math.sqrt(len(tokens)) * (1.0 / (1.0 + 0.08 * ctx_rank))
            scored.append((score, ctx_rank, order))

        if not scored or max(s for s, _, _ in scored) < self.min_score:
            return Answer(text=ABSTAIN_TEXT, citations=[], abstained=True)

        scored.sort(key=lambda item: (-item[0], item[2]))
        chosen = sorted({order for _, _, order in scored[: self.max_sentences]})

        text = " ".join(candidates[order][2] for order in chosen)
        citations: list[str] = []
        for order in chosen:
            chunk_id = contexts[candidates[order][0]].chunk_id
            if chunk_id not in citations:
                citations.append(chunk_id)
        return Answer(text=text, citations=citations, abstained=False)

    @staticmethod
    def _idf(documents: list[list[str]]) -> dict[str, float]:
        n = len(documents)
        df: dict[str, int] = {}
        for tokens in documents:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1
        return {token: math.log(1.0 + n / count) for token, count in df.items()}
