"""FAISS storage with an explicit mapping from vectors back to text chunks."""

from typing import Protocol, Sequence

import faiss
import numpy as np
from numpy.typing import NDArray

from app.rag.models import SearchResult, TextChunk


class VectorStore(Protocol):
    """Minimal contract that a future Qdrant or pgvector adapter can implement."""

    def add_documents(
        self, chunks: Sequence[TextChunk], embeddings: NDArray[np.float32]
    ) -> None: ...

    def search(
        self, query_embedding: NDArray[np.float32], top_k: int
    ) -> list[SearchResult]: ...


class FaissVectorStore:
    """In-memory cosine similarity index plus associated chunk data."""

    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        self._chunks: list[TextChunk] = []

    def add_documents(
        self, chunks: Sequence[TextChunk], embeddings: NDArray[np.float32]
    ) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required")
        vectors = self._as_matrix(embeddings)
        if len(chunks) != vectors.shape[0]:
            raise ValueError("Each chunk must have exactly one embedding")

        if self._index is None:
            self._index = faiss.IndexFlatIP(vectors.shape[1])
        elif self._index.d != vectors.shape[1]:
            raise ValueError(
                f"Embedding dimension {vectors.shape[1]} does not match index "
                f"dimension {self._index.d}"
            )

        # After L2 normalization, inner product equals cosine similarity.
        faiss.normalize_L2(vectors)
        self._index.add(vectors)
        self._chunks.extend(chunks)

    def search(
        self, query_embedding: NDArray[np.float32], top_k: int = 5
    ) -> list[SearchResult]:
        if self._index is None or not self._chunks:
            return []
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query = self._as_matrix(query_embedding)
        if query.shape[0] != 1:
            raise ValueError("Search expects exactly one query embedding")
        if query.shape[1] != self._index.d:
            raise ValueError("Query embedding dimension does not match the index")

        faiss.normalize_L2(query)
        count = min(top_k, len(self._chunks))
        scores, indices = self._index.search(query, count)
        return [
            SearchResult(chunk=self._chunks[index], score=float(score))
            for score, index in zip(scores[0], indices[0], strict=True)
            if index >= 0
        ]

    @staticmethod
    def _as_matrix(values: NDArray[np.float32]) -> NDArray[np.float32]:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] == 0:
            raise ValueError("Embeddings must be a non-empty 1D or 2D array")
        if not np.all(np.isfinite(array)):
            raise ValueError("Embeddings must contain only finite numbers")
        return np.ascontiguousarray(array.copy())
