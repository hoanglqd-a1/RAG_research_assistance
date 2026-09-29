"""Build the standard tool registry from existing application services."""

from app.agent.registry import ToolRegistry
from app.agent.tools.calculator import create_calculator_tool
from app.agent.tools.document_metadata import create_document_metadata_tool
from app.agent.tools.document_search import create_search_documents_tool
from app.agent.tools.summarize_document import (
    DocumentSummarizer,
    create_summarize_document_tool,
)
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore


def build_tool_registry(
    retriever: Retriever,
    vector_store: VectorStore,
    summarizer: DocumentSummarizer,
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(create_search_documents_tool(retriever))
    registry.register(create_calculator_tool())
    registry.register(create_document_metadata_tool(vector_store))
    registry.register(create_summarize_document_tool(summarizer))
    return registry
