from collections import deque

import pytest

from app.agent.agent import Agent
from app.agent.registry import ToolRegistry
from app.agent.schemas import AgentModelResponse, ModelToolCall, ToolResultEvent
from app.agent.state import AgentState
from app.agent.tools.calculator import create_calculator_tool
from app.agent.tools.document_search import create_search_documents_tool
from app.rag.models import TextChunk
from app.rag.retriever import Retriever
from app.rag.vector_store import FaissVectorStore
from tests.fakes import KeywordEmbeddingService


class ScriptedAgentLLM:
    """Return predetermined model responses and record the observed state."""

    def __init__(self, responses: list[AgentModelResponse]) -> None:
        self.responses = deque(responses)
        self.observed_event_counts: list[int] = []

    def respond(self, state, tool_definitions):
        self.observed_event_counts.append(len(state.events))
        return self.responses.popleft()


def test_agent_executes_tool_then_returns_final_answer() -> None:
    llm = ScriptedAgentLLM(
        [
            AgentModelResponse(
                tool_calls=[
                    ModelToolCall(
                        call_id="call-1",
                        name="calculator",
                        arguments={"expression": "25.4 - 27.1"},
                    )
                ]
            ),
            AgentModelResponse(text="The absolute difference is 1.7."),
        ]
    )
    registry = ToolRegistry()
    registry.register(create_calculator_tool())
    state = AgentState()

    result = Agent(llm, registry).run("What is the difference?", state)

    assert result.answer == "The absolute difference is 1.7."
    assert result.llm_calls == 2
    assert result.tool_calls[0].tool == "calculator"
    assert result.tool_calls[0].success
    assert llm.observed_event_counts == [1, 3]
    tool_result = next(e for e in state.events if isinstance(e, ToolResultEvent))
    assert tool_result.result.data["result"] == pytest.approx(-1.7)


def test_agent_stops_at_maximum_steps() -> None:
    repeated_call = AgentModelResponse(
        tool_calls=[
            ModelToolCall(
                call_id="repeated",
                name="calculator",
                arguments={"expression": "1 + 1"},
            )
        ]
    )
    llm = ScriptedAgentLLM([repeated_call, repeated_call])
    registry = ToolRegistry()
    registry.register(create_calculator_tool())

    result = Agent(llm, registry, max_steps=2).run("Keep calculating")

    assert result.stopped_due_to_limit
    assert result.llm_calls == 2
    assert len(result.tool_calls) == 2


def test_agent_searches_documents_then_preserves_sources() -> None:
    embedder = KeywordEmbeddingService()
    store = FaissVectorStore()
    chunk = TextChunk("The ocean contains salt water.", "sea.pdf", 4, "sea-4")
    store.add_documents([chunk], embedder.embed_documents([chunk.text]))
    registry = ToolRegistry()
    registry.register(create_search_documents_tool(Retriever(embedder, store)))
    llm = ScriptedAgentLLM(
        [
            AgentModelResponse(
                tool_calls=[
                    ModelToolCall(
                        call_id="search-1",
                        name="search_documents",
                        arguments={"query": "ocean", "top_k": 1},
                    )
                ]
            ),
            AgentModelResponse(text="The ocean is salty [sea.pdf, page 4]."),
        ]
    )

    result = Agent(llm, registry).run("What does the paper say about the ocean?")

    assert result.answer.endswith("[sea.pdf, page 4].")
    assert result.sources[0].filename == "sea.pdf"
    assert result.sources[0].page == 4
