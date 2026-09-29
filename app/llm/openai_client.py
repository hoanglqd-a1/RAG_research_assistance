"""Small OpenAI Responses API adapter for text generation and tool calling."""

import json
from typing import Any

from openai import OpenAI

from app.agent.prompts import AGENT_SYSTEM_PROMPT
from app.agent.schemas import (
    AgentModelResponse,
    ConversationMessage,
    ModelToolCall,
    ToolRequestEvent,
    ToolResultEvent,
)
from app.agent.state import AgentState


class OpenAIClient:
    """Translate application-owned state to OpenAI Responses API items.

    The agent loop remains provider-independent and visible in `Agent`; this
    class only handles the wire-format conversion.
    """

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("LLM_API_KEY is required for OpenAI generation")
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        """Generate plain text for direct RAG and summarization."""

        response = self._client.responses.create(model=self._model, input=prompt)
        return response.output_text.strip()

    def respond(
        self, state: AgentState, tool_definitions: list[dict[str, object]]
    ) -> AgentModelResponse:
        """Ask the model either for function calls or a final answer."""

        response = self._client.responses.create(
            model=self._model,
            instructions=AGENT_SYSTEM_PROMPT,
            input=self._to_openai_input(state),
            tools=tool_definitions,
        )
        tool_calls: list[ModelToolCall] = []
        for item in response.output:
            if getattr(item, "type", None) != "function_call":
                continue
            raw_arguments = getattr(item, "arguments", "{}")
            try:
                arguments = json.loads(raw_arguments)
                if not isinstance(arguments, dict):
                    arguments = {}
            except json.JSONDecodeError:
                # An empty object will fail required-field validation and the
                # structured validation error is returned to the model.
                arguments = {}
            tool_calls.append(
                ModelToolCall(
                    call_id=item.call_id,
                    name=item.name,
                    arguments=arguments,
                )
            )
        return AgentModelResponse(
            text=response.output_text.strip() or None,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _to_openai_input(state: AgentState) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for event in state.events:
            if isinstance(event, ConversationMessage):
                items.append({"role": event.role, "content": event.content})
            elif isinstance(event, ToolRequestEvent):
                items.append(
                    {
                        "type": "function_call",
                        "call_id": event.call_id,
                        "name": event.name,
                        "arguments": json.dumps(event.arguments),
                    }
                )
            elif isinstance(event, ToolResultEvent):
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": event.call_id,
                        "output": event.result.model_dump_json(),
                    }
                )
        return items
