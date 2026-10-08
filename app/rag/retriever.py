"""Query embedding and vector search, kept separate from future generation."""

from typing import Protocol

from app.rag.bm25_store import BM25Store
from app.rag.embeddings import EmbeddingProvider
from app.rag.models import SearchResult
from app.rag.vector_store import VectorStore


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
