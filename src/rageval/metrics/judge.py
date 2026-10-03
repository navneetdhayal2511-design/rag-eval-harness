"""Automatic correctness judging, and the calibration that makes it usable.

An automatic judge is a measuring instrument, and an uncalibrated instrument
produces numbers that look rigorous and are not. Nothing here reports a judge
score without also offering the agreement it reached with human labels.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rageval.metrics.stats import cohens_kappa
from rageval.text import token_f1
from rageval.types import Answer

# Landis & Koch bands for kappa.
_SUBSTANTIAL = 0.6
_MODERATE = 0.4


class Judge(ABC):
    name: str

    @abstractmethod
    def score(self, question: str, answer: Answer, reference: str) -> int:
        """Return 1 if the answer is judged correct, else 0."""


class LexicalJudge(Judge):
    """Deterministic judge: content overlap with the reference answer.

    Weak on paraphrase by construction. It is the default so the pipeline runs
    offline, and its calibration output is expected to show that weakness rather
    than hide it.
    """

    name = "lexical"

    def __init__(self, threshold: float = 0.45) -> None:
        self.threshold = threshold

    def score(self, question: str, answer: Answer, reference: str) -> int:  # noqa: ARG002
        if answer.abstained:
            return 0
        _, recall, f1 = token_f1(answer.text, reference)
        # Extractive answers carry surrounding context, so recall of the
        # reference's content matters more than precision against it.
        return int(max(f1, recall * 0.9) >= self.threshold)


class LLMJudge(Judge):
    """Model-graded correctness. Optional, cached, and temperature zero."""

    name = "llm"

    _PROMPT = (
        "You grade whether a candidate answer conveys the same information as a "
        "reference answer. Extra detail is acceptable. Missing or contradicting "
        "the reference's key facts is not. Reply with exactly CORRECT or INCORRECT."
    )

    def __init__(
        self, model: str = "gpt-4o-mini", cache_dir: str | None = ".rageval_cache"
    ) -> None:
        self.model = model
        self._cache_dir = cache_dir
        self._client: Any = None

    def score(self, question: str, answer: Answer, reference: str) -> int:
        from rageval.harness.cache import DiskCache

        if answer.abstained:
            return 0
        user = (
            f"Question: {question}\n\nReference answer: {reference}\n\n"
            f"Candidate answer: {answer.text}"
        )
        cache = DiskCache(self._cache_dir) if self._cache_dir else None
        hit = cache.get("judge", self.model, user) if cache else None
        if hit is not None:
            return int(hit["label"])

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set; use the default lexical judge")
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=5,
            messages=[
                {"role": "system", "content": self._PROMPT},
                {"role": "user", "content": user},
            ],
        )
        verdict = (response.choices[0].message.content or "").strip().upper()
        label = int(verdict.startswith("CORRECT"))
        if cache:
            cache.put("judge", self.model, user, {"label": label})
        return label


@dataclass(frozen=True)
class Calibration:
    """How far an automatic judge can be trusted, measured against human labels."""

    n: int
    agreement: float
    kappa: float
    judge_positive_rate: float
    human_positive_rate: float
    false_positives: int
    false_negatives: int

    @property
    def verdict(self) -> str:
        if self.kappa >= _SUBSTANTIAL:
            return "usable: substantial agreement with human labels"
        if self.kappa >= _MODERATE:
            return "use with caution: only moderate agreement, report the interval"
        return "not usable as a headline metric: agreement is near chance"

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "n": self.n,
            "agreement": round(self.agreement, 4),
            "kappa": round(self.kappa, 4),
            "judge_positive_rate": round(self.judge_positive_rate, 4),
            "human_positive_rate": round(self.human_positive_rate, 4),
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "verdict": self.verdict,
        }


def calibrate(judge_labels: Sequence[int], human_labels: Sequence[int]) -> Calibration:
    """Compare a judge's verdicts against human ground truth."""
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge and human label sequences must be the same length")
    n = len(judge_labels)
    if n == 0:
        raise ValueError("calibration needs at least one labelled question")

    agree = sum(1 for j, h in zip(judge_labels, human_labels, strict=True) if j == h)
    fp = sum(1 for j, h in zip(judge_labels, human_labels, strict=True) if j == 1 and h == 0)
    fn = sum(1 for j, h in zip(judge_labels, human_labels, strict=True) if j == 0 and h == 1)
    return Calibration(
        n=n,
        agreement=agree / n,
        kappa=cohens_kappa(judge_labels, human_labels),
        judge_positive_rate=sum(judge_labels) / n,
        human_positive_rate=sum(human_labels) / n,
        false_positives=fp,
        false_negatives=fn,
    )


def load_human_labels(path: Path) -> dict[str, int]:
    """Read JSONL rows of ``{"qid": ..., "correct": 0|1}``."""
    labels: dict[str, int] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "qid" not in row or "correct" not in row:
            raise ValueError(f"{path.name}:{lineno} needs both 'qid' and 'correct'")
        labels[str(row["qid"])] = int(bool(row["correct"]))
    if not labels:
        raise ValueError(f"{path} contains no labels")
    return labels


def build_judge(name: str, **kwargs: Any) -> Judge:
    if name == LexicalJudge.name:
        return LexicalJudge(**kwargs)
    if name == LLMJudge.name:
        return LLMJudge(**kwargs)
    raise ValueError(f"unknown judge {name!r}; available: ['lexical', 'llm']")


_WHITESPACE_RE = re.compile(r"\s+")


def normalise_for_display(text: str, limit: int = 160) -> str:
    """Collapse an answer onto one line for report tables."""
    flat = _WHITESPACE_RE.sub(" ", text).strip()
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


__all__ = [
    "Calibration",
    "Judge",
    "LLMJudge",
    "LexicalJudge",
    "build_judge",
    "calibrate",
    "load_human_labels",
    "normalise_for_display",
]
