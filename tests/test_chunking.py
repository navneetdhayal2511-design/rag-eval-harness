import pytest

from rageval.corpus import FixedWindowChunker, SectionChunker, SentenceWindowChunker
from rageval.corpus.chunking import build_chunker, sentence_spans
from rageval.types import Document

CHUNKERS = [
    FixedWindowChunker(window=12, stride=8),
    SentenceWindowChunker(max_chars=120, overlap_sentences=1),
    SectionChunker(max_chars=200),
]


@pytest.mark.parametrize("chunker", CHUNKERS, ids=lambda c: c.name)
def test_offsets_address_the_original_text(chunker, toy_document: Document) -> None:
    # Gold evidence is resolved through these offsets, so a drift here would
    # silently mislabel relevance for every question.
    for chunk in chunker.split(toy_document):
        assert chunk.text == toy_document.text[chunk.start : chunk.end]


@pytest.mark.parametrize("chunker", CHUNKERS, ids=lambda c: c.name)
def test_every_word_survives_chunking(chunker, toy_document: Document) -> None:
    covered = "".join(c.text for c in chunker.split(toy_document))
    for word in ("deductible", "ninety", "Cosmetic", "accident"):
        assert word in covered


def test_fixed_window_overlaps_by_the_configured_stride(toy_document: Document) -> None:
    chunks = FixedWindowChunker(window=12, stride=8).split(toy_document)
    assert len(chunks) > 1
    assert chunks[1].start < chunks[0].end


def test_stride_wider_than_window_is_rejected() -> None:
    # Would drop tokens between consecutive windows without any error.
    with pytest.raises(ValueError, match="stride"):
        FixedWindowChunker(window=10, stride=20)


def test_sentence_chunker_does_not_cut_mid_sentence(toy_document: Document) -> None:
    # The failure this guards against is a chunk ending mid-clause, which strands
    # the operative half of a rule in the next chunk.
    boundaries = {end for _, end in sentence_spans(toy_document.text)}
    for chunk in SentenceWindowChunker(max_chars=120).split(toy_document):
        assert chunk.end in boundaries, f"chunk ends mid-sentence: {chunk.text!r}"


def test_section_chunker_keeps_each_heading_with_its_body(toy_document: Document) -> None:
    chunks = SectionChunker(max_chars=200).split(toy_document)
    exclusions = [c for c in chunks if "Cosmetic surgery" in c.text]
    assert exclusions and "## Exclusions" in exclusions[0].text


def test_section_chunker_subdivides_an_oversized_section() -> None:
    body = " ".join(f"Clause {i} states a rule." for i in range(60))
    doc = Document(doc_id="d", title="d", text=f"# Long\n\n{body}\n")
    chunks = SectionChunker(max_chars=200).split(doc)
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 for c in chunks)


def test_build_chunker_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown chunker"):
        build_chunker("sliding-magic")
