"""Conversation state retained during one request or in-memory session."""

from dataclasses import dataclass, field
from threading import Lock
from uuid import uuid4

from app.agent.schemas import AgentEvent, ConversationMessage


@dataclass
class AgentState:
    """Ordered observable events sent back to the model on each step."""

    events: list[AgentEvent] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        self.events.append(ConversationMessage(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        self.events.append(ConversationMessage(role="assistant", content=content))


class ConversationStore:
    """Process-local conversation storage; intentionally not long-term memory."""

    def __init__(self) -> None:
        self._states: dict[str, AgentState] = {}
        self._lock = Lock()

    def get_or_create(self, conversation_id: str | None) -> tuple[str, AgentState]:
        with self._lock:
            identifier = conversation_id or str(uuid4())
            state = self._states.setdefault(identifier, AgentState())
            return identifier, state
