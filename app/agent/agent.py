"""Explicit single-agent tool execution loop."""

import logging
import time
from typing import Any

from app.agent.llm import AgentLLM
from app.agent.registry import ToolRegistry
from app.agent.schemas import (
    AgentRunResult,
    SourceReference,
    ToolCallTrace,
    ToolRequestEvent,
    ToolResultEvent,
)
from app.agent.state import AgentState

logger = logging.getLogger(__name__)


class Agent:
    """Ask an LLM what to do, execute requested tools, and repeat safely."""

    def __init__(
        self, llm: AgentLLM, registry: ToolRegistry, max_steps: int = 6
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")
        self._llm = llm
        self._registry = registry
        self._max_steps = max_steps

    def run(self, message: str, state: AgentState | None = None) -> AgentRunResult:
        if not message.strip():
            raise ValueError("message cannot be empty")
        state = state or AgentState()
        state.add_user_message(message)
        traces: list[ToolCallTrace] = []
        sources: list[SourceReference] = []
        llm_calls = 0

        for step in range(1, self._max_steps + 1):
            logger.info("Agent step %d/%d", step, self._max_steps)
            response = self._llm.respond(state, self._registry.definitions())
            llm_calls += 1

            if not response.tool_calls:
                answer = response.text or "I could not produce an answer."
                state.add_assistant_message(answer)
                logger.info(
                    "Agent finished: llm_calls=%d tool_calls=%d",
                    llm_calls,
                    len(traces),
                )
                return AgentRunResult(
                    answer=answer,
                    sources=self._deduplicate_sources(sources),
                    tool_calls=traces,
                    llm_calls=llm_calls,
                )

            # Preserve the provider's turn structure: all function-call requests
            # belong to one assistant turn, followed by their tool outputs.
            for call in response.tool_calls:
                logger.info("Agent selected tool=%s arguments=%s", call.name, call.arguments)
                state.events.append(
                    ToolRequestEvent(
                        call_id=call.call_id,
                        name=call.name,
                        arguments=call.arguments,
                    )
                )

            for call in response.tool_calls:
                started = time.perf_counter()
                result = self._registry.execute(call.name, call.arguments)
                duration_ms = (time.perf_counter() - started) * 1000
                state.events.append(
                    ToolResultEvent(
                        call_id=call.call_id,
                        name=call.name,
                        result=result,
                    )
                )
                traces.append(
                    ToolCallTrace(
                        step=step,
                        tool=call.name,
                        arguments=call.arguments,
                        success=result.ok,
                        duration_ms=duration_ms,
                        error=result.error,
                    )
                )
                sources.extend(self._extract_sources(result.data))
                logger.info(
                    "Tool %s success=%s duration_ms=%.2f",
                    call.name,
                    result.ok,
                    duration_ms,
                )

        answer = (
            "I could not complete the request within the configured agent step limit. "
            "Please make the request more specific or try again."
        )
        state.add_assistant_message(answer)
        logger.warning(
            "Agent reached step limit: llm_calls=%d tool_calls=%d",
            llm_calls,
            len(traces),
        )
        return AgentRunResult(
            answer=answer,
            sources=self._deduplicate_sources(sources),
            tool_calls=traces,
            llm_calls=llm_calls,
            stopped_due_to_limit=True,
        )

    @staticmethod
    def _extract_sources(data: Any) -> list[SourceReference]:
        if not isinstance(data, dict):
            return []
        raw_sources = data.get("results", data.get("sources", []))
        if not isinstance(raw_sources, list):
            return []
        sources: list[SourceReference] = []
        for item in raw_sources:
            if isinstance(item, dict) and isinstance(item.get("filename"), str):
                sources.append(
                    SourceReference(
                        filename=item["filename"],
                        page=item.get("page"),
                        chunk_id=item.get("chunk_id"),
                        score=item.get("score"),
                    )
                )
        return sources

    @staticmethod
    def _deduplicate_sources(sources: list[SourceReference]) -> list[SourceReference]:
        unique: dict[tuple[str, int | None, str | None], SourceReference] = {}
        for source in sources:
            key = (source.filename, source.page, source.chunk_id)
            unique[key] = source
        return list(unique.values())
