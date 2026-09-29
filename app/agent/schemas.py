"""Typed data exchanged by the agent, LLM adapter, and tools."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    """A user or assistant message kept in request/session state."""

    type: Literal["message"] = "message"
    role: Literal["user", "assistant"]
    content: str


class ToolRequestEvent(BaseModel):
    """One function call selected by the model."""

    type: Literal["tool_request"] = "tool_request"
    call_id: str
    name: str
    arguments: dict[str, Any]


class ToolExecutionResult(BaseModel):
    """JSON-compatible success or error envelope returned by every tool."""

    ok: bool
    data: Any = None
    error: str | None = None


class ToolResultEvent(BaseModel):
    """A tool result associated with the model's call ID."""

    type: Literal["tool_result"] = "tool_result"
    call_id: str
    name: str
    result: ToolExecutionResult


AgentEvent = ConversationMessage | ToolRequestEvent | ToolResultEvent


class ModelToolCall(BaseModel):
    """A normalized tool request returned by an LLM implementation."""

    call_id: str
    name: str
    arguments: dict[str, Any]


class AgentModelResponse(BaseModel):
    """Normalized LLM response: final text, tool calls, or both."""

    text: str | None = None
    tool_calls: list[ModelToolCall] = Field(default_factory=list)


class SourceReference(BaseModel):
    """Source metadata captured from a document tool result."""

    filename: str
    page: int | None = None
    chunk_id: str | None = None
    score: float | None = None


class ToolCallTrace(BaseModel):
    """Development trace without hidden model reasoning or secrets."""

    step: int
    tool: str
    arguments: dict[str, Any]
    success: bool
    duration_ms: float
    error: str | None = None


class AgentRunResult(BaseModel):
    """Final answer plus observable tool activity and grounded sources."""

    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    llm_calls: int = 0
    stopped_due_to_limit: bool = False
