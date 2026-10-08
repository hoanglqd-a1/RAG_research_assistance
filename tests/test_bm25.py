import pytest

from app.rag.bm25_store import BM25Store, tokenize
from app.rag.chunker import TextChunker
from app.rag.document_loader import DocumentLoader
from app.rag.models import TextChunk
from app.rag.pipeline import RetrievalPipeline
from app.rag.retriever import BM25Retriever
from app.rag.vector_store import FaissVectorStore
from tests.fakes import KeywordEmbeddingService


def test_technical_query_returns_original_source() -> None:
    target = TextChunk("SplineGS reports 100 FPS with MACP.", "paper.pdf", 6, "a")
    store = BM25Store()
    store.add_documents([
        target,
        TextChunk("Camera pose estimation.", "other.pdf", 2, "b"),
        TextChunk("Depth estimation results.", "other.pdf", 3, "c"),
    ])
    results = store.search("SPLINEGS FPS?", 1)
    assert results[0].chunk is target
    assert results[0].score > 0
    assert tokenize("mLPIPS, 3D") == ["mlpips", "3d"]


def test_updates_rebuild_index_and_no_matches_return_empty() -> None:
    store = BM25Store()
    assert store.search("ocean") == []
    store.add_documents([TextChunk("Python programming", "a.txt", None, "a")])
    target = TextChunk("Ocean water", "b.txt", None, "b")
    store.add_documents([target, TextChunk("Stars", "c.txt", None, "c")])
    assert store.search("ocean")[0].chunk is target
    assert store.search("unmatched") == []
    assert store.search("!!!") == []


def test_common_terms_are_not_discarded_for_nonpositive_scores() -> None:
    store = BM25Store()
    store.add_documents([TextChunk("ocean ocean", "a.txt", 1, "a")])
    assert len(store.search("ocean")) == 1


def test_empty_text_and_invalid_queries() -> None:
    store = BM25Store()
    store.add_documents([TextChunk("!!!", "a.txt", None, "a")])
    assert store.search("ocean") == []
    with pytest.raises(ValueError):
        store.search(" ")
    with pytest.raises(ValueError):
        store.search("ocean", 0)


@pytest.mark.parametrize("k1,b", [(0, 0.75), (float("nan"), 0.75), (1.5, -1), (1.5, 2)])
def test_invalid_parameters(k1, b) -> None:
    with pytest.raises(ValueError):
        BM25Store(k1, b)


def test_pipeline_updates_shared_bm25_retriever(tmp_path) -> None:
    store = BM25Store()
    retriever = BM25Retriever(store)
    pipeline = RetrievalPipeline(
        DocumentLoader(), TextChunker(), KeywordEmbeddingService(),
        FaissVectorStore(), retriever=retriever, bm25_store=store,
    )
    first = tmp_path / "first.txt"
    first.write_text("Python programming")
    pipeline.ingest(first)
    second = tmp_path / "second.txt"
    second.write_text("Ocean water")
    pipeline.ingest(second)
    assert pipeline.query("ocean", 1) == retriever.retrieve("ocean", 1)
    assert retriever.retrieve("ocean", 1)[0].chunk.filename == "second.txt"
