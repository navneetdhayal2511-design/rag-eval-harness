"""Pipeline execution and metric aggregation."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from rageval import metrics as M
from rageval.config import (
    PipelineConfig,
    build_answerer_from,
    build_chunker_from,
    build_judge_from,
    build_retriever,
)
from rageval.corpus import (
    Corpus,
    DatasetError,
    build_corpus,
    load_questions,
    validate_dataset,
)
from rageval.generation import Answerer
from rageval.metrics.judge import Judge
from rageval.metrics.stats import bootstrap_ci
from rageval.retrieval import Retriever
from rageval.types import EvalQuestion, MetricValue, QueryRecord, RunResult


class Pipeline:
    """An indexed, ready-to-query system under evaluation."""

    def __init__(
        self,
        config: PipelineConfig,
        corpus: Corpus,
        questions: list[EvalQuestion],
        problems: list[str],
    ) -> None:
        self.config = config
        self.corpus = corpus
        self.questions = questions
        self._problems = problems
        self._retriever: Retriever = build_retriever(config)
        self._retriever.index(corpus)
        self._answerer: Answerer = build_answerer_from(config.answerer)
        self._judge: Judge = build_judge_from(config.judge)

    @classmethod
    def build(cls, config: PipelineConfig, *, strict: bool = False) -> Pipeline:
        chunker = build_chunker_from(config.chunker)
        corpus = build_corpus(config.dataset.corpus_dir, chunker)
        questions = load_questions(config.dataset.questions)

        problems = validate_dataset(corpus, questions)
        if problems and strict:
            listing = "\n  ".join(problems[:10])
            raise ValueError(
                f"dataset is not usable with the '{config.chunker.name}' chunker "
                f"({len(problems)} problems):\n  {listing}"
            )
        return cls(config, corpus, questions, problems)

    def _within_budget(self, chunk_ids: list[str]) -> list[str]:
        """Longest prefix of the ranking that fits the configured character budget."""
        budget = self.config.context_chars
        if budget is None:
            return chunk_ids
        kept: list[str] = []
        used = 0
        for chunk_id in chunk_ids:
            size = len(self.corpus.chunk(chunk_id).text)
            if kept and used + size > budget:
                break
            kept.append(chunk_id)
            used += size
        return kept

    def run_one(self, question: EvalQuestion) -> QueryRecord:
        started = time.perf_counter()
        retrieval = self._retriever.search(question.question, self.config.top_k)
        retrieved_ms = (time.perf_counter() - started) * 1000.0

        # The generator only ever sees what fits in the context budget.
        contexts = [self.corpus.chunk(cid) for cid in self._within_budget(retrieval.chunk_ids)]
        started = time.perf_counter()
        answer = self._answerer.answer(question.question, contexts)
        generated_ms = (time.perf_counter() - started) * 1000.0

        return QueryRecord(
            qid=question.qid,
            retrieval=retrieval,
            answer=answer,
            retrieval_ms=retrieved_ms,
            generation_ms=generated_ms,
        )

    def score_one(self, question: EvalQuestion, record: QueryRecord) -> dict[str, float]:
        """Per-question scores. Kept raw so two runs can be compared pairwise.

        Retrieval metrics are reported for every answerable question, including
        ones whose gold evidence no chunk happens to cover. Skipping those would
        quietly reward a chunking strategy for making questions unanswerable.
        """
        retrieved = record.retrieval.chunk_ids
        scores: dict[str, float] = {}

        if not question.unanswerable:
            try:
                relevant = self.corpus.relevant_chunks(question)
            except DatasetError:
                # Already surfaced by validation; score it as a miss rather than
                # aborting a sweep halfway through.
                relevant = set()
            for k in self.config.k_values:
                scores[f"recall@{k}"] = M.recall_at_k(retrieved, relevant, k)
                scores[f"precision@{k}"] = M.precision_at_k(retrieved, relevant, k)
                scores[f"ndcg@{k}"] = M.ndcg_at_k(retrieved, relevant, k)
                scores[f"hit_rate@{k}"] = M.hit_rate_at_k(retrieved, relevant, k)
            if self.config.context_chars is not None:
                budgeted = self._within_budget(retrieved)
                depth = len(budgeted)
                scores["recall@budget"] = M.recall_at_k(budgeted, relevant, depth)
                scores["precision@budget"] = M.precision_at_k(budgeted, relevant, depth)
                scores["ndcg@budget"] = M.ndcg_at_k(budgeted, relevant, depth)
                scores["chunks_in_budget"] = float(depth)
            scores["mrr"] = M.reciprocal_rank(retrieved, relevant)
            scores["citation_precision"] = M.citation_precision(record.answer.citations, relevant)
            scores["citation_recall"] = M.citation_recall(record.answer.citations, relevant)
            scores["answer_f1"] = M.answer_f1(record.answer.text, question.reference_answer)
            scores["exact_match"] = M.exact_match(record.answer.text, question.reference_answer)
            scores["evidence_coverage"] = M.evidence_coverage(
                record.answer.text, [e.quote for e in question.evidence]
            )
            scores["judge_correct"] = float(
                self._judge.score(question.question, record.answer, question.reference_answer)
            )

        cited_texts = [self.corpus.chunk(cid).text for cid in record.answer.citations]
        scores["groundedness"] = M.groundedness(record.answer, cited_texts)
        scores["abstention_correct"] = M.abstention_correct(record.answer, question.unanswerable)
        scores["latency_ms"] = record.total_ms
        return scores

    def run(self, run_id: str | None = None) -> RunResult:
        workers = max(1, self.config.workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(self.run_one, self.questions))

        per_question = {
            question.qid: self.score_one(question, record)
            for question, record in zip(self.questions, records, strict=True)
        }
        return RunResult(
            run_id=run_id or self.config.name,
            pipeline=self.config.name,
            config=self.config.fingerprint(),
            records=records,
            metrics=aggregate(per_question, seed=self.config.seed),
            per_question=per_question,
        )

    @property
    def dataset_problems(self) -> list[str]:
        return self._problems


def aggregate(
    per_question: dict[str, dict[str, float]], *, seed: int = 0, n_resamples: int = 2000
) -> dict[str, MetricValue]:
    """Mean and bootstrap interval for every metric, over the questions that have it.

    Retrieval metrics are undefined for unanswerable questions, so each metric is
    averaged only over the questions where it applies rather than being silently
    imputed as zero.
    """
    columns: dict[str, list[float]] = {}
    for scores in per_question.values():
        for metric, value in scores.items():
            columns.setdefault(metric, []).append(value)

    out: dict[str, MetricValue] = {}
    for metric in sorted(columns):
        interval = bootstrap_ci(columns[metric], seed=seed, n_resamples=n_resamples)
        out[metric] = MetricValue(
            name=metric,
            value=interval.value,
            ci_low=interval.low,
            ci_high=interval.high,
            n=interval.n,
        )
    return out
