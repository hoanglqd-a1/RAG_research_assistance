"""Index a text/PDF file and print the chunks most relevant to a question."""

import argparse
import logging

from app.core.config import settings
from app.rag.chunker import TextChunker
from app.rag.bm25_store import BM25Store
from app.rag.document_loader import DocumentLoader
from app.rag.embeddings import EmbeddingService
from app.rag.pipeline import RetrievalPipeline
from app.rag.retriever import build_retriever
from app.rag.vector_store import FaissVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", help="Path to a PDF or UTF-8 text file")
    parser.add_argument("question", help="Question used for similarity search")
    parser.add_argument("--top-k", type=int, default=settings.default_top_k)
    args = parser.parse_args()


    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    embedder = EmbeddingService(settings.embedding_model_name)
    vectors = FaissVectorStore()
    lexical = BM25Store(settings.bm25_k1, settings.bm25_b)
    retriever = build_retriever(
        settings.retrieval_mode, embedder, vectors, lexical,
        candidate_k=settings.hybrid_candidate_k,
        rrf_k=settings.hybrid_rrf_k,
    )
    pipeline = RetrievalPipeline(
        loader=DocumentLoader(),
        chunker=TextChunker(settings.chunk_size, settings.chunk_overlap),
        embedding_service=embedder,
        vector_store=vectors,
        retriever=retriever,
        bm25_store=lexical,
    )
    chunks = pipeline.ingest(args.document)
    print(f"\nIndexed {len(chunks)} chunks. Mode: {settings.retrieval_mode}. Retrieval results:\n")
    for rank, result in enumerate(pipeline.query(args.question, args.top_k), start=1):
        source = result.chunk.filename
        if result.chunk.page is not None:
            source += f", page {result.chunk.page}"
        print(
            f"{rank}. score={result.score:.3f} | {source} | "
            f"{result.chunk.chunk_id}\n{result.chunk.text}\n"
        )


if __name__ == "__main__":
    main()
