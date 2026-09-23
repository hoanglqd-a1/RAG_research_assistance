"""Offline test doubles that avoid downloading an embedding model."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


class KeywordEmbeddingService:
    """Represent text by counts for three known concepts."""

    vocabulary = ("python", "ocean", "astronomy")

    def _embed(self, text: str) -> NDArray[np.float32]:
        lowered = text.lower()
        return np.asarray(
            [lowered.count(word) for word in self.vocabulary], dtype=np.float32
        )

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.vstack([self._embed(text) for text in texts])

    def embed_query(self, query: str) -> NDArray[np.float32]:
        return self._embed(query)
