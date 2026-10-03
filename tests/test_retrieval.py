import math

import numpy as np
import pytest

from rageval.corpus import Corpus, SectionChunker
from rageval.retrieval import BM25Retriever, reciprocal_rank_fusion, top_k_indices
from rageval.retrieval.hybrid import weighted_score_fusion
from rageval.text import analyze
from rageval.types import Document, Retrieval, ScoredChunk


def _corpus(texts: list[str]) -> Corpus:
    docs = {
        f"d{i}": Document(doc_id=f"d{i}", title=f"d{i}", text=f"# D{i}\n\n{t}\n")
        for i, t in enumerate(texts)
    }
    chunks = [c for d in docs.values() for c in SectionChunker(max_chars=5000).split(d)]
    return Corpus(documents=docs, chunks=chunks)


def _reference_bm25(query: str, docs: list[list[str]], k1: float, b: float) -> list[float]:
    """Textbook Okapi BM25, scored one posting at a time."""
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n
    scores = []
    for doc in docs:
        total = 0.0
        for term in analyze(query):
            freq = doc.count(term)
            if freq == 0:
                continue
            df = sum(1 for other in docs if term in other)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            total += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * len(doc) / avgdl))
        scores.append(total)
    return scores


def test_vectorised_bm25_matches_the_textbook_formula() -> None:
    # The index-time precomputation is the main optimisation in this repo; it is
    # only worth having if it is numerically identical to the obvious version.
    texts = [
        "The deductible is 500 units and the deductible resets each year.",
        "Claims must be filed within ninety days of treatment.",
        "Cosmetic surgery is excluded from every plan and every rider.",
        "The deductible for families is 1500 units in total.",
    ]
    corpus = _corpus(texts)
    retriever = BM25Retriever(k1=1.5, b=0.75)
    retriever.index(corpus)

    query = "what is the deductible"
    got = {c.chunk_id: c.score for c in retriever.search(query, len(corpus.chunks)).chunks}
    expected = _reference_bm25(query, [analyze(c.text) for c in corpus.chunks], k1=1.5, b=0.75)
    for chunk, want in zip(corpus.chunks, expected, strict=True):
        assert got[chunk.chunk_id] == pytest.approx(want, abs=1e-4)


def test_bm25_ranks_the_passage_that_answers_the_query_first() -> None:
    corpus = _corpus(
        [
            "Cosmetic surgery is excluded from every plan.",
            "The maternity waiting period is ten months from the cover start date.",
            "Interest accrues on late settlements at six per cent.",
        ]
    )
    retriever = BM25Retriever()
    retriever.index(corpus)
    top = retriever.search("how long is the maternity waiting period", 1).chunks[0]
    assert "maternity waiting period" in corpus.chunk(top.chunk_id).text


def test_bm25_returns_nothing_when_no_query_term_is_in_the_vocabulary() -> None:
    corpus = _corpus(["Cosmetic surgery is excluded."])
    retriever = BM25Retriever()
    retriever.index(corpus)
    assert retriever.search("acupuncture", 5).chunks == []


def test_searching_before_indexing_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match="before index"):
        BM25Retriever().search("anything", 3)


def test_invalid_bm25_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="b must lie"):
        BM25Retriever(b=1.5)


def test_top_k_selection_orders_by_score_and_breaks_ties_by_index() -> None:
    scores = np.array([0.2, 0.9, 0.9, 0.1, 0.5], dtype=np.float32)
    assert top_k_indices(scores, 3).tolist() == [1, 2, 4]


def test_top_k_selection_handles_k_above_the_population() -> None:
    scores = np.array([0.4, 0.8], dtype=np.float32)
    assert top_k_indices(scores, 10).tolist() == [1, 0]


def test_reciprocal_rank_fusion_matches_the_formula() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=2, smoothing=60)
    expected = 1 / 61 + 1 / 62
    assert {c.chunk_id for c in fused} == {"a", "b"}
    assert all(c.score == pytest.approx(expected) for c in fused)


def test_fusion_promotes_what_both_rankers_agree_on() -> None:
    fused = reciprocal_rank_fusion([["x", "shared"], ["y", "shared"]], k=3, smoothing=1)
    assert fused[0].chunk_id == "shared"


def test_weighted_fusion_respects_arm_weights() -> None:
    lexical = Retrieval(
        chunks=[ScoredChunk(chunk_id="a", score=10.0), ScoredChunk(chunk_id="b", score=0.0)]
    )
    dense = Retrieval(
        chunks=[ScoredChunk(chunk_id="b", score=1.0), ScoredChunk(chunk_id="a", score=0.0)]
    )
    assert weighted_score_fusion([lexical, dense], k=2, weights=[0.9, 0.1])[0].chunk_id == "a"
    assert weighted_score_fusion([lexical, dense], k=2, weights=[0.1, 0.9])[0].chunk_id == "b"
