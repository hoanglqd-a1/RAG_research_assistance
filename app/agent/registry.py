"""Validated tool registration and dispatch without a large conditional block."""

from collections.abc import Callable
import json
import logging
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from app.agent.schemas import ToolExecutionResult

logger = logging.getLogger(__name__)
InputModel = TypeVar("InputModel", bound=BaseModel)


class Tool(Generic[InputModel]):
    """One model-visible function and its validated Python handler."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_model: type[InputModel],
        handler: Callable[[InputModel], Any],
    ) -> None:
        self.name = name
        self.description = description
        self.input_model = input_model
        self.handler = handler

    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.input_model.model_json_schema(),
        }

    def execute(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        try:
            validated = self.input_model.model_validate(arguments)
            return ToolExecutionResult(ok=True, data=self.handler(validated))
        except ValidationError as exc:
            return ToolExecutionResult(
                ok=False,
                error=f"Invalid arguments for {self.name}: {exc}",
            )
        except Exception as exc:  # Tool failures must be observable to the model.
            logger.exception("Tool %s failed", self.name)
            return ToolExecutionResult(
                ok=False,
                error=f"Tool {self.name} failed: {type(exc).__name__}: {exc}",
            )


class ToolRegistry:
    """Lookup table used for both LLM definitions and safe dispatch."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool[Any]] = {}

    def register(self, tool: Tool[Any]) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def definitions(self) -> list[dict[str, Any]]:
        return [tool.definition() for tool in self._tools.values()]

    def execute(
        self, name: str, arguments: dict[str, Any] | str
    ) -> ToolExecutionResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolExecutionResult(ok=False, error=f"Unknown tool: {name}")
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as exc:
                return ToolExecutionResult(
                    ok=False, error=f"Tool arguments are not valid JSON: {exc}"
                )
            if not isinstance(parsed, dict):
                return ToolExecutionResult(
                    ok=False, error="Tool arguments must be a JSON object"
                )
            arguments = parsed
        return tool.execute(arguments)
