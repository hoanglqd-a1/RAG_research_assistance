"""In-memory lexical retrieval over the same chunks used by dense search."""

import math
import re
from typing import Sequence

from rank_bm25 import BM25Okapi

from app.rag.models import SearchResult, TextChunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


class BM25Store:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("BM25 k1 must be finite and greater than zero")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 b must be between zero and one")
        self._k1 = k1
        self._b = b
        self._chunks: list[TextChunk] = []
        self._tokens: list[list[str]] = []
        self._index: BM25Okapi | None = None

    def add_documents(self, chunks: Sequence[TextChunk]) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required")
        added = [(chunk, tokenize(chunk.text)) for chunk in chunks]
        searchable = [(chunk, tokens) for chunk, tokens in added if tokens]
        if not searchable:
            return
        combined_chunks = self._chunks + [chunk for chunk, tokens in searchable]
        combined_tokens = self._tokens + [tokens for chunk, tokens in searchable]
        index = BM25Okapi(combined_tokens, k1=self._k1, b=self._b)
        self._chunks = combined_chunks
        self._tokens = combined_tokens
        self._index = index

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        query_tokens = tokenize(query)
        if self._index is None or not query_tokens:
            return []
        scores = self._index.get_scores(query_tokens)
        query_terms = set(query_tokens)
        matches = [
            index for index, tokens in enumerate(self._tokens)
            if query_terms.intersection(tokens)
        ]
        ranked = sorted(matches, key=lambda index: (-float(scores[index]), index))
        return [
            SearchResult(chunk=self._chunks[index], score=float(scores[index]))
            for index in ranked[:top_k]
        ]
