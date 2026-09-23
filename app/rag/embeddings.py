"""Embedding abstraction backed by a sentence-transformer model."""

from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


FloatMatrix = NDArray[np.float32]


class EmbeddingProvider(Protocol):
    """Interface used by the pipeline and retriever (also easy to fake in tests)."""

    def embed_documents(self, texts: Sequence[str]) -> FloatMatrix: ...

    def embed_query(self, query: str) -> FloatMatrix: ...


class EmbeddingService:
    """Map document text and queries into the same semantic vector space.

    `embed_documents` returns shape (number_of_texts, embedding_dimension).
    `embed_query` returns shape (embedding_dimension,). Because the same model
    embeds both, vectors with similar meaning should be close to one another.
    """

    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: Sequence[str]) -> FloatMatrix:
        if not texts:
            raise ValueError("At least one document text is required")
        vectors = self._model.encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=False
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> FloatMatrix:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        vector = self._model.encode(
            query, convert_to_numpy=True, normalize_embeddings=False
        )
        return np.asarray(vector, dtype=np.float32)
