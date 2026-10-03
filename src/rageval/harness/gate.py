"""The regression gate that CI runs on every pull request."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from rageval.harness.report import LOWER_IS_BETTER, compare_runs
from rageval.types import RunResult


class Threshold(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str
    # Absolute floor the metric must clear regardless of history.
    min: float | None = None
    # Largest tolerated move in the wrong direction against the baseline.
    max_drop: float | None = None


class GateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thresholds: list[Threshold] = Field(default_factory=list)
    # Require a drop to be statistically significant before failing the build.
    # Without this, resampling noise on a few hundred questions turns the gate
    # into a flaky test that people start ignoring, which is worse than no gate.
    require_significance: bool = True
    alpha: float = 0.05
    seed: int = 0

    @classmethod
    def load(cls, path: Path) -> GateConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw or {})


@dataclass(frozen=True)
class Violation:
    metric: str
    reason: str


def evaluate(
    candidate: RunResult, baseline: RunResult | None, config: GateConfig
) -> list[Violation]:
    """Check a run against absolute floors and against the stored baseline."""
    violations: list[Violation] = []

    for threshold in config.thresholds:
        metric = candidate.metrics.get(threshold.metric)
        if metric is None:
            violations.append(
                Violation(threshold.metric, "metric is missing from the candidate run")
            )
            continue
        if threshold.min is None:
            continue
        lower_better = threshold.metric in LOWER_IS_BETTER
        breached = metric.value > threshold.min if lower_better else metric.value < threshold.min
        if breached:
            comparator = "above" if lower_better else "below"
            violations.append(
                Violation(
                    threshold.metric,
                    f"{metric.value:.3f} is {comparator} the floor of {threshold.min:.3f}",
                )
            )

    drops = {t.metric: t.max_drop for t in config.thresholds if t.max_drop is not None}
    if baseline is not None and drops:
        comparisons = compare_runs(baseline, candidate, metrics=sorted(drops), seed=config.seed)
        for c in comparisons:
            # Normalise so a positive `regression` always means "got worse".
            regression = c.delta if c.metric in LOWER_IS_BETTER else -c.delta
            limit = drops[c.metric]
            if limit is None or regression <= limit:
                continue
            if config.require_significance and c.p_value >= config.alpha:
                continue
            violations.append(
                Violation(
                    c.metric,
                    f"dropped {regression:.3f} against the baseline "
                    f"(limit {limit:.3f}, p={c.p_value:.3f})",
                )
            )

    return violations
