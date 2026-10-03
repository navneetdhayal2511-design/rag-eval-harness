"""Shared text analysis.

One analyzer is used by every lexical component so that BM25, the TF-IDF
embedder and the citation metrics all agree on what a token is. Divergent
tokenisation between the retriever and the metric is a classic source of
evaluation results that cannot be reproduced.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

# Deliberately small: domain corpora are full of words a large stop list would
# discard (for example "no", "not", "all" matter in policy exclusions).
STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "for",
        "on",
        "at",
        "by",
        "with",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "and",
        "or",
        "if",
        "then",
        "than",
        "so",
        "such",
        "there",
        "here",
    ]
)


def _singularise(token: str) -> str:
    """Light plural folding. Avoids a stemmer dependency and is easy to reason about."""
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def analyze(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Lowercase, tokenise and fold plurals."""
    tokens = (_singularise(t) for t in _TOKEN_RE.findall(text.lower()))
    if drop_stopwords:
        return [t for t in tokens if t not in STOPWORDS]
    return list(tokens)


def token_set(text: str) -> set[str]:
    return set(analyze(text))


def token_f1(prediction: str, reference: str) -> tuple[float, float, float]:
    """Bag-of-tokens precision, recall and F1 between two strings."""
    pred = analyze(prediction)
    ref = analyze(reference)
    if not pred or not ref:
        return (0.0, 0.0, 0.0)

    counts: dict[str, int] = {}
    for token in ref:
        counts[token] = counts.get(token, 0) + 1
    overlap = 0
    for token in pred:
        if counts.get(token, 0) > 0:
            counts[token] -= 1
            overlap += 1
    if overlap == 0:
        return (0.0, 0.0, 0.0)

    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return (precision, recall, 2 * precision * recall / (precision + recall))


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0
