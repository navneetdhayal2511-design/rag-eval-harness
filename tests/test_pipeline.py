"""End-to-end checks against the shipped dataset and configs."""

from pathlib import Path

import pytest

from rageval.config import PipelineConfig, build_chunker_from
from rageval.corpus import build_corpus, load_questions, validate_dataset
from rageval.harness.runner import Pipeline
from rageval.metrics.judge import load_human_labels

from .conftest import DATASET, REPO

CONFIGS = sorted(p for p in (REPO / "configs").glob("*.yaml") if p.name != "gates.yaml")


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_every_gold_quote_resolves_under_every_chunking(path: Path) -> None:
    # The dataset is labelled once and reused across chunking strategies, so a
    # quote that only resolves under one of them is a latent false negative.
    config = PipelineConfig.load(path)
    corpus = build_corpus(config.dataset.corpus_dir, build_chunker_from(config.chunker))
    assert validate_dataset(corpus, load_questions(config.dataset.questions)) == []


def test_cutoffs_deeper_than_the_retrieved_list_are_rejected() -> None:
    # recall@10 on a 5-item list silently repeats recall@5 and reads as a
    # second, independent data point.
    config = PipelineConfig.load(CONFIGS[0]).model_dump()
    config["k_values"] = [1, 5, 50]
    with pytest.raises(ValueError, match="exceed top_k"):
        PipelineConfig.model_validate(config)


def test_baseline_pipeline_runs_and_scores_every_question() -> None:
    config = PipelineConfig.load(REPO / "configs" / "bm25-fixed.yaml")
    result = Pipeline.build(config, strict=True).run()
    assert len(result.records) == 65
    assert result.metrics["recall@budget"].n == 60
    assert result.metrics["recall@budget"].value > 0.5


def test_quality_metrics_are_deterministic() -> None:
    # Without this the regression gate measures run-to-run noise rather than
    # changes. Latency is wall clock and is excluded by design.
    config = PipelineConfig.load(REPO / "configs" / "hybrid-rrf.yaml")
    pipeline = Pipeline.build(config)

    def quality(result: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        return {
            qid: {m: v for m, v in scores.items() if m != "latency_ms"}
            for qid, scores in result.items()
        }

    assert quality(pipeline.run().per_question) == quality(pipeline.run().per_question)


def test_the_context_budget_is_respected() -> None:
    config = PipelineConfig.load(REPO / "configs" / "bm25-section.yaml")
    pipeline = Pipeline.build(config)
    budget = config.context_chars
    assert budget is not None
    for question in pipeline.questions[:10]:
        record = pipeline.run_one(question)
        kept = pipeline._within_budget(record.retrieval.chunk_ids)
        used = sum(len(pipeline.corpus.chunk(cid).text) for cid in kept)
        assert len(kept) == 1 or used <= budget


def test_answers_only_cite_chunks_that_were_retrieved() -> None:
    # A citation to something outside the context window would make
    # citation_precision meaningless.
    config = PipelineConfig.load(REPO / "configs" / "bm25-fixed.yaml")
    pipeline = Pipeline.build(config)
    for question in pipeline.questions[:15]:
        record = pipeline.run_one(question)
        assert set(record.answer.citations) <= set(record.retrieval.chunk_ids)


def test_human_labels_line_up_with_the_golden_set() -> None:
    labels = load_human_labels(DATASET / "human_labels.jsonl")
    answerable = {q.qid for q in load_questions(DATASET / "questions.jsonl") if not q.unanswerable}
    assert set(labels) == answerable


def test_the_committed_baseline_passes_its_own_gate() -> None:
    from rageval.harness.gate import GateConfig, evaluate
    from rageval.harness.report import load_run

    baseline = load_run(REPO / "baselines" / "bm25-fixed.json")
    config = GateConfig.load(REPO / "configs" / "gates.yaml")
    assert evaluate(baseline, baseline, config) == []
