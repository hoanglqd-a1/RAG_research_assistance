"""Small domain models passed between retrieval components."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """Text extracted from one document page (or one complete text file)."""

    text: str
    filename: str
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextChunk:
    """A searchable piece of a document with source metadata."""

    text: str
    filename: str
    page: int | None
    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk and its backend-specific relevance score."""

    chunk: TextChunk
    score: float
