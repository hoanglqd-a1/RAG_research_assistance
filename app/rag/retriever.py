"""Dense, lexical, and rank-fused retrieval, separate from generation."""

import logging
import math
from typing import Protocol

from app.rag.bm25_store import BM25Store
from app.rag.embeddings import EmbeddingProvider
from app.rag.models import SearchResult
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

class RetrievalProvider(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]: ...


class BM25Retriever:
    def __init__(self, store: BM25Store) -> None:
        self._store = store

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return self._store.search(query, top_k)


class VectorRetriever:
    """Embed one question and retrieve its most similar text chunks."""

    def __init__(
        self, embedding_service: EmbeddingProvider, vector_store: VectorStore
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        query_embedding = self._embedding_service.embed_query(query)
        return self._vector_store.search(query_embedding, top_k)


class HybridRetriever:
    """Combine dense and BM25 rankings with equal-weight Reciprocal Rank Fusion.

    A chunk at one-based rank r contributes 1 / (rrf_k + r). A chunk found
    by both searches receives both contributions. Raw backend scores are not
    combined because BM25 scores and cosine similarities have different scales.
    """

    def __init__(
        self,
        dense: RetrievalProvider,
        lexical: RetrievalProvider,
        candidate_k: int = 20,
        rrf_k: float = 60.0,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("Hybrid candidate_k must be greater than zero")
        if not math.isfinite(rrf_k) or rrf_k <= 0:
            raise ValueError("Hybrid rrf_k must be finite and greater than zero")
        self._dense = dense
        self._lexical = lexical
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        # Search a wider pool than the final output. Ensure requests larger
        # than the configured candidate count can still return enough results.
        pool_size = max(self._candidate_k, top_k)
        dense_results = self._dense.retrieve(query, pool_size)
        lexical_results = self._lexical.retrieve(query, pool_size)
        scores: dict[tuple[str, int | None, str], float] = {}
        originals: dict[tuple[str, int | None, str], SearchResult] = {}

        for ranking in (dense_results, lexical_results):
            seen: set[tuple[str, int | None, str]] = set()
            for rank, result in enumerate(ranking, start=1):
                chunk = result.chunk
                # Chunk IDs alone are not globally unique in the current
                # chunker: different files can both have doc-0-chunk-0.
                key = (chunk.filename, chunk.page, chunk.chunk_id)
                if key in seen:
                    continue
                seen.add(key)
                originals.setdefault(key, result)
                scores[key] = scores.get(key, 0.0) + 1.0 / (self._rrf_k + rank)

        # Equal scores retain insertion order: dense first, then lexical-only
        # candidates. This deterministic tie-break does not use raw scores.
        ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
        results = [
            SearchResult(chunk=originals[key].chunk, score=scores[key])
            for key in ranked_keys[:top_k]
        ]
        logger.debug(
            "Hybrid retrieval: dense=%d lexical=%d unique=%d returned=%d",
            len(dense_results), len(lexical_results), len(scores), len(results),
        )
        return results


def build_retriever(
    mode: str,
    embedding_service: EmbeddingProvider,
    vector_store: VectorStore,
    bm25_store: BM25Store,
    candidate_k: int = 20,
    rrf_k: float = 60.0,
) -> RetrievalProvider:
    """Use the same mode selection for application and benchmark services."""
    dense = VectorRetriever(embedding_service, vector_store)
    lexical = BM25Retriever(bm25_store)
    if mode == "dense":
        return dense
    if mode == "bm25":
        return lexical
    if mode == "hybrid":
        return HybridRetriever(dense, lexical, candidate_k, rrf_k)
    raise ValueError("Retrieval mode must be dense, bm25, or hybrid")
