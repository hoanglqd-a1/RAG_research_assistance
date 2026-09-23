from app.rag.chunker import TextChunker
from app.rag.document_loader import DocumentLoader
from app.rag.pipeline import RetrievalPipeline
from app.rag.vector_store import FaissVectorStore
from tests.fakes import KeywordEmbeddingService


def test_pipeline_ingests_file_and_retrieves_source_metadata(tmp_path) -> None:
    path = tmp_path / "facts.txt"
    path.write_text(
        "Python is useful for software and data science. "
        "The ocean is a large body of salt water.",
        encoding="utf-8",
    )
    pipeline = RetrievalPipeline(
        DocumentLoader(),
        TextChunker(chunk_size=48, chunk_overlap=8),
        KeywordEmbeddingService(),
        FaissVectorStore(),
    )

    chunks = pipeline.ingest(path)
    results = pipeline.query("ocean", top_k=1)

    assert len(chunks) >= 2
    assert results[0].chunk.filename == "facts.txt"
    assert "ocean" in results[0].chunk.text.lower()
