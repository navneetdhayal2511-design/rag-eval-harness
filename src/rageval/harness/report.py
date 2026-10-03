"""Run persistence, markdown reporting and A/B comparison."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from rageval.metrics.stats import Comparison, compare
from rageval.types import RunResult

# Metrics where a smaller number is better.
LOWER_IS_BETTER = frozenset({"latency_ms"})
# Diagnostics reported on their own scale rather than as a 0-1 score.
COUNTS = frozenset({"latency_ms", "chunks_in_budget"})

# Budget-relative metrics come first: they are the ones comparable across
# chunking strategies. Fixed-k metrics are reported too but only mean something
# between runs that chunk the corpus the same way.
HEADLINE = (
    "recall@budget",
    "ndcg@budget",
    "recall@5",
    "ndcg@5",
    "mrr",
    "citation_precision",
    "judge_correct",
    "chunks_in_budget",
    "latency_ms",
)


def save_run(result: RunResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_run(path: Path) -> RunResult:
    return RunResult.model_validate_json(path.read_text(encoding="utf-8"))


def _fmt(value: float, metric: str) -> str:
    return f"{value:.1f}" if metric in COUNTS else f"{value:.3f}"


def render_run(result: RunResult, *, headline_only: bool = False) -> str:
    """Markdown summary of a single run."""
    names = (
        [m for m in HEADLINE if m in result.metrics] if headline_only else sorted(result.metrics)
    )
    lines = [
        f"# Run `{result.run_id}`",
        "",
        f"Pipeline: `{result.pipeline}`  |  Questions: {len(result.records)}",
        "",
        "| metric | value | 95% CI | n |",
        "| --- | ---: | :---: | ---: |",
    ]
    for name in names:
        metric = result.metrics[name]
        lines.append(
            f"| {name} | {_fmt(metric.value, name)} | "
            f"[{_fmt(metric.ci_low, name)}, {_fmt(metric.ci_high, name)}] | {metric.n} |"
        )
    return "\n".join(lines) + "\n"


def compare_runs(
    baseline: RunResult,
    candidate: RunResult,
    *,
    metrics: Sequence[str] | None = None,
    seed: int = 0,
) -> list[Comparison]:
    """Paired comparison over the questions both runs answered."""
    shared = sorted(set(baseline.per_question) & set(candidate.per_question))
    if not shared:
        raise ValueError("runs share no questions, so they cannot be compared")

    names = list(metrics) if metrics else sorted(baseline.metrics.keys() & candidate.metrics.keys())
    out: list[Comparison] = []
    for name in names:
        pairs = [
            (baseline.per_question[q][name], candidate.per_question[q][name])
            for q in shared
            if name in baseline.per_question[q] and name in candidate.per_question[q]
        ]
        if not pairs:
            continue
        out.append(compare(name, [a for a, _ in pairs], [b for _, b in pairs], seed=seed))
    return out


def render_comparison(
    baseline: RunResult, candidate: RunResult, comparisons: Sequence[Comparison]
) -> str:
    lines = [
        f"# `{candidate.run_id}` vs `{baseline.run_id}`",
        "",
        "Paired over the questions both runs cover. `p` is a two-sided sign-flip",
        "permutation test; a delta whose interval spans zero is noise.",
        "",
        "| metric | baseline | candidate | delta | 95% CI | p | verdict |",
        "| --- | ---: | ---: | ---: | :---: | ---: | --- |",
    ]
    for c in comparisons:
        better_when_lower = c.metric in LOWER_IS_BETTER
        if not c.significant:
            verdict = "no change"
        elif (c.delta < 0) == better_when_lower:
            verdict = "improved"
        else:
            verdict = "regressed"
        lines.append(
            f"| {c.metric} | {_fmt(c.baseline, c.metric)} | {_fmt(c.candidate, c.metric)} | "
            f"{c.delta:+.3f} | [{c.delta_low:+.3f}, {c.delta_high:+.3f}] | "
            f"{c.p_value:.3f} | {verdict} |"
        )
    return "\n".join(lines) + "\n"


def render_leaderboard(results: Sequence[RunResult], metrics: Sequence[str] = HEADLINE) -> str:
    """One row per configuration, for an ablation sweep."""
    usable = [m for m in metrics if any(m in r.metrics for r in results)]
    lines = [
        "| pipeline | " + " | ".join(usable) + " |",
        "| --- | " + " | ".join("---:" for _ in usable) + " |",
    ]
    for result in results:
        cells = []
        for name in usable:
            metric = result.metrics.get(name)
            cells.append(_fmt(metric.value, name) if metric else "—")
        lines.append(f"| {result.pipeline} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_json(payload: object, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
