import math

import pytest

from rageval.metrics import (
    abstention_correct,
    answer_f1,
    citation_precision,
    citation_recall,
    evidence_coverage,
    groundedness,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rageval.types import Answer

RANKING = ["c0", "c1", "c2", "c3", "c4"]
RELEVANT = {"c1", "c3"}


def test_recall_counts_only_the_gold_inside_the_cutoff() -> None:
    assert recall_at_k(RANKING, RELEVANT, 2) == pytest.approx(0.5)
    assert recall_at_k(RANKING, RELEVANT, 4) == pytest.approx(1.0)


def test_precision_divides_by_the_cutoff_window() -> None:
    assert precision_at_k(RANKING, RELEVANT, 4) == pytest.approx(0.5)


def test_hit_rate_is_satisfied_by_a_single_gold_chunk() -> None:
    assert hit_rate_at_k(RANKING, RELEVANT, 2) == 1.0
    assert hit_rate_at_k(RANKING, RELEVANT, 1) == 0.0


def test_reciprocal_rank_uses_the_first_gold_position() -> None:
    assert reciprocal_rank(RANKING, RELEVANT) == pytest.approx(0.5)
    assert reciprocal_rank(RANKING, {"zzz"}) == 0.0


def test_ndcg_matches_a_hand_computed_value() -> None:
    # Gold at ranks 2 and 4; ideal places both at ranks 1 and 2.
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RANKING, RELEVANT, 5) == pytest.approx(dcg / idcg)


def test_ndcg_rewards_putting_the_gold_higher() -> None:
    assert ndcg_at_k(["c1", "c3", "a", "b"], RELEVANT, 4) > ndcg_at_k(RANKING, RELEVANT, 4)


def test_retrieval_metrics_are_zero_when_nothing_relevant_exists() -> None:
    # An answerable question whose gold is unreachable must score as a miss, not
    # be quietly treated as satisfied.
    assert recall_at_k(RANKING, set(), 5) == 0.0
    assert ndcg_at_k(RANKING, set(), 5) == 0.0
    assert hit_rate_at_k(RANKING, set(), 5) == 0.0


def test_citation_precision_punishes_pointing_at_the_wrong_clause() -> None:
    assert citation_precision(["c1", "c9"], RELEVANT) == pytest.approx(0.5)
    assert citation_precision([], RELEVANT) == 0.0


def test_citation_recall_measures_gold_coverage() -> None:
    assert citation_recall(["c1"], RELEVANT) == pytest.approx(0.5)


def test_answer_f1_is_insensitive_to_casing_and_punctuation() -> None:
    assert answer_f1("Ninety days.", "ninety days") == pytest.approx(1.0)


def test_groundedness_falls_when_the_answer_leaves_its_sources() -> None:
    supported = Answer(text="the deductible is 500 units", citations=["c1"])
    invented = Answer(text="the deductible is 500 units and premiums double", citations=["c1"])
    source = ["The deductible is 500 units per policy year."]
    assert groundedness(supported, source) == pytest.approx(1.0)
    assert groundedness(invented, source) < 1.0


def test_abstention_is_scored_on_answerable_questions_too() -> None:
    # Otherwise a system that refuses everything would look perfect.
    refusal = Answer(text="cannot answer", citations=[], abstained=True)
    reply = Answer(text="ninety days", citations=["c1"])
    assert abstention_correct(refusal, unanswerable=True) == 1.0
    assert abstention_correct(refusal, unanswerable=False) == 0.0
    assert abstention_correct(reply, unanswerable=True) == 0.0


def test_evidence_coverage_catches_an_answer_that_drops_the_key_detail() -> None:
    quotes = ["filed within ninety days of treatment"]
    full = evidence_coverage("claims must be filed within ninety days of treatment", quotes)
    vague = evidence_coverage("claims must be filed promptly", quotes)
    assert full > vague
