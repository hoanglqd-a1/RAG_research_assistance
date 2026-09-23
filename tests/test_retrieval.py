from app.rag.models import TextChunk
from app.rag.retriever import Retriever
from app.rag.vector_store import FaissVectorStore
from tests.fakes import KeywordEmbeddingService


def test_query_retrieves_semantically_matching_chunk() -> None:
    chunks = [
        TextChunk("Python is a programming language.", "code.txt", None, "c-1"),
        TextChunk("The ocean contains salt water.", "sea.txt", None, "c-2"),
        TextChunk("Astronomy studies stars.", "space.txt", None, "c-3"),
    ]
    embedder = KeywordEmbeddingService()
    store = FaissVectorStore()
    store.add_documents(chunks, embedder.embed_documents([c.text for c in chunks]))

    results = Retriever(embedder, store).retrieve("Tell me about ocean water", top_k=2)

    assert results[0].chunk.chunk_id == "c-2"
    assert results[0].chunk.filename == "sea.txt"
    assert results[0].score == 1.0


def test_empty_store_returns_no_results() -> None:
    results = Retriever(KeywordEmbeddingService(), FaissVectorStore()).retrieve(
        "python", top_k=3
    )
    assert results == []
