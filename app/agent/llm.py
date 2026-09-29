"""Protocol separating the agent loop from a particular model provider."""

from typing import Protocol

from app.agent.schemas import AgentModelResponse
from app.agent.state import AgentState


class AgentLLM(Protocol):
    def respond(
        self, state: AgentState, tool_definitions: list[dict[str, object]]
    ) -> AgentModelResponse: ...
