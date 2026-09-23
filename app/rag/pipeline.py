"""Orchestrate ingestion and retrieval while keeping every step inspectable."""

import logging
from pathlib import Path

from app.rag.chunker import TextChunker
from app.rag.document_loader import DocumentLoader
from app.rag.embeddings import EmbeddingProvider
from app.rag.models import SearchResult, TextChunk
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Connect loading, chunking, embedding, indexing, and retrieval."""

    def __init__(
        self,
        loader: DocumentLoader,
        chunker: TextChunker,
        embedding_service: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._loader = loader
        self._chunker = chunker
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._retriever = Retriever(embedding_service, vector_store)

    def ingest(self, path: str | Path) -> list[TextChunk]:
        """Load and index one file, returning the chunks that were added."""

        documents = self._loader.load(path)
        chunks = self._chunker.split_documents(documents)
        if not chunks:
            raise ValueError(f"No extractable text found in '{Path(path).name}'")
        embeddings = self._embedding_service.embed_documents(
            [chunk.text for chunk in chunks]
        )
        self._vector_store.add_documents(chunks, embeddings)
        logger.info("Indexed %d chunks from %s", len(chunks), Path(path).name)
        return chunks

    def query(self, question: str, top_k: int = 5) -> list[SearchResult]:
        """Retrieve relevant chunks; generation intentionally comes later."""

        return self._retriever.retrieve(question, top_k)
