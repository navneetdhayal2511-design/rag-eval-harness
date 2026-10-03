import pytest

from rageval.corpus import (
    Corpus,
    DatasetError,
    FixedWindowChunker,
    SectionChunker,
    validate_dataset,
)
from rageval.types import Document, EvalQuestion, GoldEvidence


def test_quote_resolves_to_an_exact_span(toy_corpus: Corpus) -> None:
    span = toy_corpus.resolve(GoldEvidence(doc_id="policy", quote="deductible is 500 units"))
    source = toy_corpus.documents["policy"].text
    assert source[span.start : span.end] == "deductible is 500 units"


def test_quote_matching_tolerates_rewrapped_whitespace(toy_corpus: Corpus) -> None:
    # Labels are typed by hand; a line break in the source must not break them.
    span = toy_corpus.resolve(
        GoldEvidence(doc_id="policy", quote="Claims  must be\n  filed within ninety days")
    )
    assert "ninety days" in toy_corpus.documents["policy"].text[span.start : span.end]


def test_missing_quote_is_rejected(toy_corpus: Corpus) -> None:
    with pytest.raises(DatasetError, match="not found"):
        toy_corpus.resolve(GoldEvidence(doc_id="policy", quote="premiums are waived"))


def test_ambiguous_quote_is_rejected() -> None:
    # A quote matching twice would make relevance depend on which copy was meant.
    doc = Document(doc_id="d", title="d", text="# T\n\nIt is excluded. Also it is excluded.\n")
    corpus = Corpus(documents={"d": doc}, chunks=SectionChunker().split(doc))
    with pytest.raises(DatasetError, match="ambiguous"):
        corpus.resolve(GoldEvidence(doc_id="d", quote="it is excluded"))


def test_unknown_document_is_rejected(toy_corpus: Corpus) -> None:
    with pytest.raises(DatasetError, match="unknown document"):
        toy_corpus.resolve(GoldEvidence(doc_id="nope", quote="anything"))


def test_relevance_survives_a_change_of_chunking(toy_document: Document) -> None:
    # The whole point of labelling quotes instead of chunk ids: the same gold
    # text must stay reachable when the chunker changes.
    question = EvalQuestion(
        qid="q1",
        question="What is the deductible?",
        reference_answer="500 units per year.",
        evidence=[GoldEvidence(doc_id="policy", quote="deductible is 500 units per year")],
    )
    for chunker in (SectionChunker(max_chars=200), FixedWindowChunker(window=10, stride=7)):
        corpus = Corpus(documents={"policy": toy_document}, chunks=chunker.split(toy_document))
        relevant = corpus.relevant_chunks(question)
        assert relevant, f"gold text unreachable under {chunker.name}"
        assert any("deductible" in corpus.chunk(cid).text for cid in relevant)


def test_a_chunk_holding_a_sliver_of_the_quote_is_not_relevant(toy_document: Document) -> None:
    question = EvalQuestion(
        qid="q1",
        question="What is the deductible?",
        reference_answer="500 units.",
        evidence=[GoldEvidence(doc_id="policy", quote="deductible is 500 units per year")],
    )
    corpus = Corpus(
        documents={"policy": toy_document},
        chunks=FixedWindowChunker(window=4, stride=2).split(toy_document),
    )
    for chunk_id in corpus.relevant_chunks(question, min_overlap=0.9):
        assert "500 units per year" in corpus.chunk(chunk_id).text


def test_answerable_question_without_evidence_is_rejected() -> None:
    with pytest.raises(ValueError, match="no gold evidence"):
        EvalQuestion(qid="q1", question="?", reference_answer="x", evidence=[])


def test_validate_reports_an_unreachable_question(toy_document: Document) -> None:
    question = EvalQuestion(
        qid="q1",
        question="What is the deductible?",
        reference_answer="500 units.",
        evidence=[GoldEvidence(doc_id="policy", quote="no such clause appears here")],
    )
    corpus = Corpus(documents={"policy": toy_document}, chunks=SectionChunker().split(toy_document))
    problems = validate_dataset(corpus, [question])
    assert len(problems) == 1 and "q1" in problems[0]
