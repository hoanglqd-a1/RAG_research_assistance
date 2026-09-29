import numpy as np

from app.agent.registry import ToolRegistry
from app.agent.tools.document_metadata import create_document_metadata_tool
from app.agent.tools.summarize_document import (
    DocumentSummarizer,
    create_summarize_document_tool,
)
from app.rag.models import TextChunk
from app.rag.vector_store import FaissVectorStore


class RecordingGenerator:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return f"summary-{len(self.prompts)}"


def _store() -> FaissVectorStore:
    store = FaissVectorStore()
    chunks = [
        TextChunk("first section", "paper.pdf", 1, "p-1", {"author": "Ada"}),
        TextChunk("second section", "paper.pdf", 2, "p-2", {"author": "Ada"}),
        TextChunk("other document", "notes.txt", None, "n-1"),
    ]
    store.add_documents(chunks, np.eye(3, dtype=np.float32))
    return store


def test_metadata_lists_and_filters_documents() -> None:
    registry = ToolRegistry()
    registry.register(create_document_metadata_tool(_store()))

    all_result = registry.execute("get_document_metadata", {})
    one_result = registry.execute(
        "get_document_metadata", {"filename": "paper.pdf"}
    )

    assert all_result.ok and all_result.data["count"] == 2
    assert one_result.data["documents"] == [
        {
            "filename": "paper.pdf",
            "chunk_count": 2,
            "pages": [1, 2],
            "metadata": {"author": "Ada"},
        }
    ]


def test_summary_uses_bounded_map_reduce_and_returns_sources() -> None:
    generator = RecordingGenerator()
    summarizer = DocumentSummarizer(
        _store(), generator, batch_chars=14, max_batches=3
    )
    registry = ToolRegistry()
    registry.register(create_summarize_document_tool(summarizer))

    result = registry.execute("summarize_document", {"filename": "paper.pdf"})

    assert result.ok
    assert result.data["summary"] == "summary-3"
    assert result.data["sources"] == [
        {"filename": "paper.pdf", "page": 1},
        {"filename": "paper.pdf", "page": 2},
    ]
    assert len(generator.prompts) == 3  # two map calls plus one reduce call


def test_summary_reports_missing_document_as_tool_error() -> None:
    registry = ToolRegistry()
    registry.register(
        create_summarize_document_tool(DocumentSummarizer(_store(), RecordingGenerator()))
    )

    result = registry.execute("summarize_document", {"filename": "missing.pdf"})

    assert not result.ok
    assert "No indexed document" in result.error
