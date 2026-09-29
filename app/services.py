"""Composition root: construct shared in-memory retrieval and agent services."""

from dataclasses import dataclass

from app.agent.agent import Agent
from app.agent.factory import build_tool_registry
from app.agent.state import ConversationStore
from app.agent.tools.summarize_document import DocumentSummarizer
from app.core.config import Settings
from app.llm.openai_client import OpenAIClient
from app.rag.chunker import TextChunker
from app.rag.document_loader import DocumentLoader
from app.rag.embeddings import EmbeddingService
from app.rag.generator import RagGenerator
from app.rag.pipeline import RetrievalPipeline
from app.rag.retriever import Retriever
from app.rag.vector_store import FaissVectorStore


@dataclass
class ApplicationServices:
    pipeline: RetrievalPipeline
    retriever: Retriever
    vector_store: FaissVectorStore
    rag_generator: RagGenerator
    agent: Agent
    conversations: ConversationStore


def build_services(settings: Settings) -> ApplicationServices:
    """Create one shared in-memory index and all consumers of it."""

    if not settings.llm_api_key:
        raise ValueError("LLM_API_KEY is missing; add it to .env")
    embedding_service = EmbeddingService(settings.embedding_model_name)
    vector_store = FaissVectorStore()
    retriever = Retriever(embedding_service, vector_store)
    pipeline = RetrievalPipeline(
        DocumentLoader(),
        TextChunker(settings.chunk_size, settings.chunk_overlap),
        embedding_service,
        vector_store,
    )
    llm = OpenAIClient(settings.llm_api_key, settings.llm_model)
    summarizer = DocumentSummarizer(
        vector_store,
        llm,
        batch_chars=settings.summary_batch_chars,
        max_batches=settings.max_summary_batches,
    )
    registry = build_tool_registry(retriever, vector_store, summarizer)
    return ApplicationServices(
        pipeline=pipeline,
        retriever=retriever,
        vector_store=vector_store,
        rag_generator=RagGenerator(llm),
        agent=Agent(llm, registry, max_steps=settings.max_agent_steps),
        conversations=ConversationStore(),
    )
