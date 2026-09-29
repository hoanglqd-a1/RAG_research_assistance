"""Ollama adapter for text generation and agent tool calling."""

import json
import re
from typing import Any
from uuid import uuid4

from app.agent.prompts import AGENT_SYSTEM_PROMPT
from app.agent.schemas import (
    AgentModelResponse,
    ConversationMessage,
    ModelToolCall,
    ToolRequestEvent,
    ToolResultEvent,
)
from app.agent.state import AgentState


THINKING_BLOCK_PATTERN = re.compile(
    r"<think>.*?</think>", re.DOTALL | re.IGNORECASE
)


class OllamaClient:
    """Translate application-owned state to Ollama chat/generate calls."""

    def __init__(self, model: str, host: str = "http://localhost:11434") -> None:
        try:
            from ollama import Client
        except ImportError as exc:
            raise RuntimeError(
                "The 'ollama' package is required. Install requirements and make "
                "sure the Ollama server is running."
            ) from exc

        self._client = Client(host=host)
        self._model = model

    def generate(self, prompt: str) -> str:
        """Generate plain text for direct RAG and summarization."""

        response = self._client.generate(model=self._model, prompt=prompt)
        return self._clean_text(str(self._get_value(response, "response", "")))

    def respond(
        self, state: AgentState, tool_definitions: list[dict[str, object]]
    ) -> AgentModelResponse:
        """Ask the model either for function calls or a final answer."""

        response = self._client.chat(
            model=self._model,
            messages=self._to_ollama_messages(state),
            tools=self._to_ollama_tools(tool_definitions),
        )
        message = self._get_value(response, "message", {})
        raw_tool_calls = self._get_value(message, "tool_calls", []) or []
        tool_calls = [
            self._to_model_tool_call(raw_tool_call)
            for raw_tool_call in raw_tool_calls
        ]
        return AgentModelResponse(
            text=(
                self._clean_text(str(self._get_value(message, "content", "")))
                or None
            ),
            tool_calls=tool_calls,
        )

    @classmethod
    def _to_model_tool_call(cls, raw_tool_call: Any) -> ModelToolCall:
        function = cls._get_value(raw_tool_call, "function", {})
        name = str(cls._get_value(function, "name", ""))
        raw_arguments = cls._get_value(function, "arguments", {})
        arguments = cls._parse_arguments(raw_arguments)
        call_id = str(
            cls._get_value(raw_tool_call, "id", "") or f"ollama-{uuid4()}"
        )
        return ModelToolCall(call_id=call_id, name=name, arguments=arguments)

    @classmethod
    def _to_ollama_messages(cls, state: AgentState) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT}
        ]
        pending_tool_calls: list[dict[str, Any]] = []

        def flush_tool_calls() -> None:
            if pending_tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": pending_tool_calls.copy(),
                    }
                )
                pending_tool_calls.clear()

        for event in state.events:
            if isinstance(event, ConversationMessage):
                flush_tool_calls()
                messages.append({"role": event.role, "content": event.content})
            elif isinstance(event, ToolRequestEvent):
                pending_tool_calls.append(
                    {
                        "function": {
                            "name": event.name,
                            "arguments": event.arguments,
                        }
                    }
                )
            elif isinstance(event, ToolResultEvent):
                flush_tool_calls()
                messages.append(
                    {
                        "role": "tool",
                        "content": event.result.model_dump_json(),
                        "tool_name": event.name,
                    }
                )

        flush_tool_calls()
        return messages

    @staticmethod
    def _to_ollama_tools(
        tool_definitions: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        tools: list[dict[str, object]] = []
        for definition in tool_definitions:
            if "function" in definition:
                tools.append(definition)
                continue
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": definition.get("name"),
                        "description": definition.get("description", ""),
                        "parameters": definition.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    },
                }
            )
        return tools

    @staticmethod
    def _parse_arguments(raw_arguments: Any) -> dict[str, Any]:
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if isinstance(raw_arguments, str):
            try:
                parsed = json.loads(raw_arguments)
            except json.JSONDecodeError:
                return {}
            if isinstance(parsed, dict):
                return parsed
        return {}

    @staticmethod
    def _clean_text(text: str) -> str:
        return THINKING_BLOCK_PATTERN.sub("", text).strip()

    @staticmethod
    def _get_value(container: Any, key: str, default: Any = None) -> Any:
        if isinstance(container, dict):
            return container.get(key, default)
        return getattr(container, key, default)
