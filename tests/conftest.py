from pathlib import Path

import pytest

from rageval.corpus import Corpus, SectionChunker
from rageval.types import Document

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "datasets" / "aurora-health"


@pytest.fixture
def toy_document() -> Document:
    text = (
        "# Policy\n\n"
        "The deductible is 500 units per year. Claims must be filed within ninety days.\n\n"
        "## Exclusions\n\n"
        "Cosmetic surgery is excluded. Dental work is excluded unless it follows an accident.\n"
    )
    return Document(doc_id="policy", title="Policy", text=text)


@pytest.fixture
def toy_corpus(toy_document: Document) -> Corpus:
    chunks = SectionChunker(max_chars=200).split(toy_document)
    return Corpus(documents={toy_document.doc_id: toy_document}, chunks=chunks)
