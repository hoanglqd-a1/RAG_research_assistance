import pytest

from app.agent.registry import ToolRegistry
from app.agent.tools.calculator import create_calculator_tool, evaluate_arithmetic
from app.agent.tools.document_search import create_search_documents_tool
from app.rag.models import TextChunk
from app.rag.retriever import VectorRetriever
from app.rag.vector_store import FaissVectorStore
from tests.fakes import KeywordEmbeddingService


def _search_registry() -> ToolRegistry:
    chunks = [
        TextChunk("Python is a programming language.", "code.txt", None, "c-1"),
        TextChunk("The ocean contains salt water.", "sea.txt", None, "c-2"),
    ]
    embedder = KeywordEmbeddingService()
    store = FaissVectorStore()
    store.add_documents(chunks, embedder.embed_documents([c.text for c in chunks]))
    registry = ToolRegistry()
    registry.register(create_search_documents_tool(VectorRetriever(embedder, store)))
    return registry


def test_search_documents_returns_source_metadata() -> None:
    result = _search_registry().execute(
        "search_documents", {"query": "ocean", "top_k": 1}
    )

    assert result.ok
    assert result.data["results"][0]["filename"] == "sea.txt"
    assert result.data["results"][0]["chunk_id"] == "c-2"


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("15.2 / 3.4", pytest.approx(4.470588)), ("2 * (3 + 4)", 14), ("2 ** 5", 32)],
)
def test_calculator(expression, expected) -> None:
    assert evaluate_arithmetic(expression) == expected


@pytest.mark.parametrize(
    "expression", ["__import__('os').system('dir')", "abs(-2)", "2 ** 1000"]
)
def test_calculator_rejects_unsafe_or_excessive_input(expression: str) -> None:
    with pytest.raises(ValueError):
        evaluate_arithmetic(expression)


def test_registry_returns_validation_error() -> None:
    registry = ToolRegistry()
    registry.register(create_calculator_tool())

    result = registry.execute("calculator", {"wrong_field": "2 + 2"})

    assert not result.ok
    assert "Invalid arguments" in result.error


def test_registry_rejects_unknown_tool() -> None:
    result = ToolRegistry().execute("delete_everything", {})
    assert not result.ok
    assert result.error == "Unknown tool: delete_everything"
