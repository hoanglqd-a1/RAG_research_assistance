"""Offline hybrid tests: no model downloads or LLM calls."""

import pytest

from app.core.config import Settings
from app.rag.bm25_store import BM25Store
from app.rag.chunker import TextChunker
from app.rag.document_loader import DocumentLoader
from app.rag.models import SearchResult, TextChunk
from app.rag.pipeline import RetrievalPipeline
from app.rag.retriever import (
    BM25Retriever, HybridRetriever, VectorRetriever, build_retriever,
)
from app.rag.vector_store import FaissVectorStore
from tests.fakes import KeywordEmbeddingService


class StubRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


def result(chunk_id: str, filename: str = "paper.pdf", score: float = 1.0) -> SearchResult:
    return SearchResult(TextChunk(chunk_id, filename, 3, chunk_id), score)


def test_shared_chunk_combines_rank_contributions_not_raw_scores() -> None:
    shared = result("shared", score=0.1)
    dense = StubRetriever([result("dense", score=0.99), shared])
    lexical = StubRetriever([result("lexical", score=999), shared])
    hits = HybridRetriever(dense, lexical, candidate_k=3).retrieve("question", 3)

    assert [hit.chunk.chunk_id for hit in hits] == ["shared", "dense", "lexical"]
    assert hits[0].score == pytest.approx(2 / 62)
    assert hits[1].score == pytest.approx(1 / 61)
    assert hits[0].chunk is shared.chunk
    assert hits[0].chunk.filename == "paper.pdf"
    assert hits[0].chunk.page == 3
    assert dense.calls == lexical.calls == [("question", 3)]


def test_same_local_id_in_different_files_is_not_merged() -> None:
    dense = StubRetriever([result("doc-0-chunk-0", "a.pdf")])
    lexical = StubRetriever([result("doc-0-chunk-0", "b.pdf")])
    hits = HybridRetriever(dense, lexical).retrieve("question", 5)
    assert [hit.chunk.filename for hit in hits] == ["a.pdf", "b.pdf"]
    assert all(hit.score == pytest.approx(1 / 61) for hit in hits)


def test_duplicate_within_one_ranking_is_not_double_counted() -> None:
    shared = result("same")
    hits = HybridRetriever(
        StubRetriever([shared, shared]), StubRetriever([]),
    ).retrieve("question")
    assert len(hits) == 1
    assert hits[0].score == pytest.approx(1 / 61)


def test_empty_branch_and_empty_corpus() -> None:
    dense = StubRetriever([])
    lexical = StubRetriever([result("only")])
    assert HybridRetriever(dense, lexical).retrieve("question")[0].chunk.chunk_id == "only"
    assert HybridRetriever(dense, dense).retrieve("question") == []


def test_candidate_pool_can_be_larger_than_final_top_k() -> None:
    dense = StubRetriever([result(str(i)) for i in range(10)])
    lexical = StubRetriever([])
    hits = HybridRetriever(dense, lexical, candidate_k=10).retrieve("question", 2)
    assert len(hits) == 2
    assert dense.calls == lexical.calls == [("question", 10)]


def test_top_k_larger_than_configured_pool_expands_both_searches() -> None:
    dense = StubRetriever([result(str(i)) for i in range(5)])
    lexical = StubRetriever([])
    hits = HybridRetriever(dense, lexical, candidate_k=2).retrieve("question", 5)
    assert len(hits) == 5
    assert dense.calls == lexical.calls == [("question", 5)]


@pytest.mark.parametrize("query,top_k", [(" ", 5), ("question", 0), ("question", -1)])
def test_invalid_query_does_not_call_backends(query, top_k) -> None:
    backend = StubRetriever([])
    with pytest.raises(ValueError):
        HybridRetriever(backend, backend).retrieve(query, top_k)
    assert backend.calls == []


@pytest.mark.parametrize("candidate_k,rrf_k", [
    (0, 60), (-1, 60), (20, 0), (20, -1), (20, float("nan")), (20, float("inf")),
])
def test_invalid_fusion_settings(candidate_k, rrf_k) -> None:
    with pytest.raises(ValueError):
        HybridRetriever(StubRetriever([]), StubRetriever([]), candidate_k, rrf_k)


@pytest.mark.parametrize("mode,expected_type", [
    ("dense", VectorRetriever), ("bm25", BM25Retriever), ("hybrid", HybridRetriever),
])
def test_factory_preserves_all_modes(mode, expected_type) -> None:
    retriever = build_retriever(mode, KeywordEmbeddingService(), FaissVectorStore(), BM25Store())
    assert isinstance(retriever, expected_type)


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_retriever("unknown", KeywordEmbeddingService(), FaissVectorStore(), BM25Store())


def test_settings_accept_hybrid_and_validate_parameters() -> None:
    assert Settings(retrieval_mode="hybrid").retrieval_mode == "hybrid"
    for values in [
        {"retrieval_mode": "unknown"}, {"hybrid_candidate_k": 0},
        {"hybrid_rrf_k": float("nan")},
    ]:
        with pytest.raises(ValueError):
            Settings(**values)


def test_ingestion_and_real_bm25_faiss_fusion(tmp_path) -> None:
    embedder = KeywordEmbeddingService()
    vectors = FaissVectorStore()
    lexical = BM25Store()
    retriever = build_retriever("hybrid", embedder, vectors, lexical)
    pipeline = RetrievalPipeline(
        DocumentLoader(), TextChunker(), embedder, vectors,
        retriever=retriever, bm25_store=lexical,
    )
    for name, text in [
        ("code.txt", "Python is a programming language."),
        ("sea.txt", "The ocean contains salt water."),
        ("space.txt", "Astronomy studies stars."),
    ]:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        pipeline.ingest(path)

    hits = pipeline.query("ocean water", 3)
    assert hits[0].chunk.filename == "sea.txt"
    assert hits[0].score == pytest.approx(2 / 61)
    assert len(hits) == 3
    assert len({hit.chunk.filename for hit in hits}) == 3


def test_benchmark_pipeline_uses_hybrid_settings(monkeypatch, tmp_path) -> None:
    from app.evaluation.retrieval_benchmark import build_pipeline

    monkeypatch.setattr(
        "app.rag.embeddings.EmbeddingService", lambda name: KeywordEmbeddingService(),
    )
    pipeline = build_pipeline(Settings(
        retrieval_mode="hybrid", hybrid_candidate_k=2, hybrid_rrf_k=10,
    ))
    # A pipeline built by the benchmark should return the same RRF scores
    # as the application, rather than silently falling back to dense search.
    for name, text in [("sea.txt", "ocean water"), ("code.txt", "python"), ("sky.txt", "astronomy")]:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        pipeline.ingest(path)
    hits = pipeline.query("ocean", 1)
    assert hits[0].chunk.filename == "sea.txt"
    assert hits[0].score == pytest.approx(2 / 11)
