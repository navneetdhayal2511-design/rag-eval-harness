from pathlib import Path

import pytest

from rageval.harness import DiskCache, GateConfig, aggregate, evaluate
from rageval.harness.gate import Threshold
from rageval.harness.report import compare_runs, load_run, render_run, save_run
from rageval.types import Answer, MetricValue, QueryRecord, Retrieval, RunResult


def _run(run_id: str, recall: list[float], latency: float = 10.0) -> RunResult:
    per_question = {
        f"q{i}": {"recall@budget": value, "latency_ms": latency} for i, value in enumerate(recall)
    }
    return RunResult(
        run_id=run_id,
        pipeline=run_id,
        config={},
        records=[
            QueryRecord(qid=qid, retrieval=Retrieval(chunks=[]), answer=Answer(text=""))
            for qid in per_question
        ],
        metrics=aggregate(per_question),
        per_question=per_question,
    )


def test_aggregation_averages_each_metric_over_the_questions_that_have_it() -> None:
    # Retrieval metrics are undefined for unanswerable questions, so a metric
    # must not be imputed as zero where it simply does not apply.
    per_question = {
        "q1": {"recall@budget": 1.0, "latency_ms": 5.0},
        "q2": {"recall@budget": 0.0, "latency_ms": 7.0},
        "u1": {"latency_ms": 9.0},
    }
    metrics = aggregate(per_question)
    assert metrics["recall@budget"].n == 2
    assert metrics["recall@budget"].value == pytest.approx(0.5)
    assert metrics["latency_ms"].n == 3


def test_gate_fails_on_an_absolute_floor() -> None:
    config = GateConfig(thresholds=[Threshold(metric="recall@budget", min=0.8)])
    violations = evaluate(_run("candidate", [0.5] * 20), None, config)
    assert len(violations) == 1 and "floor" in violations[0].reason


def test_gate_fails_on_a_significant_drop_against_the_baseline() -> None:
    config = GateConfig(thresholds=[Threshold(metric="recall@budget", max_drop=0.05)])
    violations = evaluate(_run("cand", [0.0] * 30), _run("base", [1.0] * 30), config)
    assert len(violations) == 1 and "dropped" in violations[0].reason


def test_gate_ignores_a_drop_that_is_within_the_noise() -> None:
    # One question out of thirty moving must not break the build.
    baseline = _run("base", [1.0] * 30)
    candidate = _run("cand", [1.0] * 29 + [0.0])
    config = GateConfig(thresholds=[Threshold(metric="recall@budget", max_drop=0.0)])
    assert evaluate(candidate, baseline, config) == []


def test_gate_can_be_told_to_ignore_significance() -> None:
    baseline = _run("base", [1.0] * 30)
    candidate = _run("cand", [1.0] * 29 + [0.0])
    config = GateConfig(
        thresholds=[Threshold(metric="recall@budget", max_drop=0.0)],
        require_significance=False,
    )
    assert len(evaluate(candidate, baseline, config)) == 1


def test_gate_reads_latency_in_the_direction_where_lower_is_better() -> None:
    config = GateConfig(thresholds=[Threshold(metric="latency_ms", min=50.0)])
    assert evaluate(_run("fast", [1.0] * 10, latency=5.0), None, config) == []
    assert len(evaluate(_run("slow", [1.0] * 10, latency=500.0), None, config)) == 1


def test_gate_flags_a_metric_the_candidate_never_reported() -> None:
    # A renamed metric would otherwise silently disable its own threshold.
    config = GateConfig(thresholds=[Threshold(metric="ndcg@budget", min=0.5)])
    violations = evaluate(_run("candidate", [1.0] * 10), None, config)
    assert len(violations) == 1 and "missing" in violations[0].reason


def test_runs_survive_a_save_and_load_round_trip(tmp_path: Path) -> None:
    original = _run("candidate", [0.5, 1.0, 0.0])
    reloaded = load_run(save_run(original, tmp_path / "run.json"))
    assert reloaded.per_question == original.per_question
    assert reloaded.metrics["recall@budget"].value == pytest.approx(0.5)


def test_comparison_only_uses_questions_present_in_both_runs() -> None:
    baseline = _run("base", [1.0] * 5)
    candidate = _run("cand", [1.0] * 3)
    assert compare_runs(baseline, candidate, metrics=["recall@budget"])[0].n == 3


def test_comparing_runs_with_no_shared_questions_is_an_error() -> None:
    baseline = _run("base", [1.0])
    empty = RunResult(
        run_id="empty", pipeline="empty", config={}, records=[], metrics={}, per_question={}
    )
    with pytest.raises(ValueError, match="share no questions"):
        compare_runs(baseline, empty)


def test_report_renders_the_interval_next_to_the_point_estimate() -> None:
    rendered = render_run(_run("candidate", [0.0, 1.0] * 10))
    assert "recall@budget" in rendered and "95% CI" in rendered


def test_metric_formatting_shows_both_bounds() -> None:
    metric = MetricValue(name="recall@budget", value=0.5, ci_low=0.4, ci_high=0.6, n=10)
    assert metric.format() == "0.500 [0.400, 0.600]"


def test_cache_round_trips_by_content(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    assert cache.get("judge", "m", "prompt") is None
    cache.put("judge", "m", "prompt", {"label": 1})
    assert cache.get("judge", "m", "prompt") == {"label": 1}
    # A different payload must not collide with the stored entry.
    assert cache.get("judge", "m", "other prompt") is None
    cache.close()
