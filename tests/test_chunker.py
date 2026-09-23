import pytest

from app.rag.chunker import TextChunker
from app.rag.models import Document


def test_chunker_splits_text_and_preserves_metadata() -> None:
    document = Document(
        text="alpha beta gamma delta epsilon zeta",
        filename="paper.pdf",
        page=3,
        metadata={"author": "Ada"},
    )

    chunks = TextChunker(chunk_size=18, chunk_overlap=5).split_documents([document])

    assert len(chunks) > 1
    assert all(chunk.filename == "paper.pdf" for chunk in chunks)
    assert all(chunk.page == 3 for chunk in chunks)
    assert all(chunk.metadata == {"author": "Ada"} for chunk in chunks)
    assert chunks[0].chunk_id == "doc-0-chunk-0"


def test_chunker_rejects_overlap_as_large_as_chunk() -> None:
    with pytest.raises(ValueError, match="smaller"):
        TextChunker(chunk_size=10, chunk_overlap=10)
