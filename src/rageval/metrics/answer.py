"""Answer-quality and attribution metrics."""

from __future__ import annotations

from collections.abc import Sequence

from rageval.text import analyze, token_f1
from rageval.types import Answer


def exact_match(prediction: str, reference: str) -> float:
    """Match after tokenisation, so punctuation and casing do not decide correctness."""
    return float(analyze(prediction) == analyze(reference))


def answer_f1(prediction: str, reference: str) -> float:
    return token_f1(prediction, reference)[2]


def citation_precision(citations: Sequence[str], relevant: set[str]) -> float:
    """Fraction of cited chunks that really do contain gold evidence.

    This is the number that catches a system confidently pointing at the wrong
    clause, which reads as authoritative to a user and is worse than no citation.
    """
    if not citations:
        return 0.0
    return sum(1 for c in citations if c in relevant) / len(citations)


def citation_recall(citations: Sequence[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    cited = set(citations)
    return sum(1 for c in relevant if c in cited) / len(relevant)


def groundedness(answer: Answer, cited_texts: Sequence[str]) -> float:
    """Fraction of the answer's content tokens that appear in the text it cites.

    A lexical proxy for attribution, not a claim-level entailment check. It is
    cheap, deterministic and catches the common failure of an answer drifting
    beyond its sources; it will not catch a fluent contradiction built from the
    source's own vocabulary. The LLM judge covers that case.
    """
    if answer.abstained:
        return 1.0
    tokens = analyze(answer.text)
    if not tokens:
        return 0.0
    supported = set()
    for text in cited_texts:
        supported |= set(analyze(text))
    return sum(1 for t in tokens if t in supported) / len(tokens)


def abstention_correct(answer: Answer, unanswerable: bool) -> float:
    """Did the system refuse exactly when it should have?

    Scored for every question, not just the unanswerable ones, because a system
    that abstains on everything would otherwise look perfect.
    """
    return float(answer.abstained == unanswerable)


def evidence_coverage(prediction: str, quotes: Sequence[str]) -> float:
    """Mean fraction of each gold quote's content tokens that the answer reproduces.

    Catches answers that cite the right passage but omit the operative detail,
    such as the number of days in a claims deadline.
    """
    if not quotes:
        return 0.0
    predicted = set(analyze(prediction))
    scores = []
    for quote in quotes:
        tokens = set(analyze(quote))
        if not tokens:
            continue
        scores.append(len(tokens & predicted) / len(tokens))
    return sum(scores) / len(scores) if scores else 0.0
