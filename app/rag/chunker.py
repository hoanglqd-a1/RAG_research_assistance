"""Simple character-based text chunking."""

from dataclasses import dataclass

from app.rag.models import Document, TextChunk


@dataclass(frozen=True)
class TextChunker:
    """Split documents into overlapping, approximately fixed-size chunks.

    Overlap repeats a small amount of text between neighboring chunks so that
    facts near a boundary are not separated from their surrounding context.
    """

    chunk_size: int = 500
    chunk_overlap: int = 75

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def split_documents(self, documents: list[Document]) -> list[TextChunk]:
        """Convert extracted documents into chunks with stable source metadata."""

        chunks: list[TextChunk] = []
        for document_index, document in enumerate(documents):
            for local_index, text in enumerate(self._split_text(document.text)):
                chunks.append(
                    TextChunk(
                        text=text,
                        filename=document.filename,
                        page=document.page,
                        chunk_id=f"doc-{document_index}-chunk-{local_index}",
                        metadata=dict(document.metadata),
                    )
                )
        return chunks

    def _split_text(self, text: str) -> list[str]:
        """Split on a nearby whitespace when possible, otherwise at max size."""

        text = text.strip()
        if not text:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(text):
            proposed_end = min(start + self.chunk_size, len(text))
            end = proposed_end

            if proposed_end < len(text):
                # Avoid tiny chunks by only seeking a boundary in the latter half.
                boundary = text.rfind(" ", start + self.chunk_size // 2, proposed_end)
                if boundary > start:
                    end = boundary

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break

            next_start = end - self.chunk_overlap
            # Move past boundary whitespace without removing intentional overlap.
            start = max(start + 1, next_start)

        return chunks
