from types import SimpleNamespace

from app.agent.schemas import ToolExecutionResult, ToolRequestEvent, ToolResultEvent
from app.agent.state import AgentState
from app.llm.ollama_client import OllamaClient


class FakeOllamaClient:
    def __init__(self, chat_response=None, generate_response=None) -> None:
        self.chat_response = chat_response
        self.generate_response = generate_response
        self.chat_kwargs = None
        self.generate_kwargs = None

    def chat(self, **kwargs):
        self.chat_kwargs = kwargs
        return self.chat_response

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return self.generate_response


def test_ollama_adapter_translates_function_calls_and_results() -> None:
    model_response = {
        "message": {
            "content": "",
            "tool_calls": [
                {
                    "id": "call-2",
                    "function": {
                        "name": "calculator",
                        "arguments": {"expression": "2 + 2"},
                    },
                }
            ],
        }
    }
    fake_client = FakeOllamaClient(chat_response=model_response)
    client = object.__new__(OllamaClient)
    client._model = "test-model"
    client._client = fake_client
    state = AgentState()
    state.add_user_message("Continue")
    state.events.append(
        ToolRequestEvent(
            call_id="call-1", name="calculator", arguments={"expression": "1 + 1"}
        )
    )
    state.events.append(
        ToolResultEvent(
            call_id="call-1",
            name="calculator",
            result=ToolExecutionResult(ok=True, data={"result": 2}),
        )
    )

    response = client.respond(state, [{"type": "function", "name": "calculator"}])

    assert response.tool_calls[0].call_id == "call-2"
    assert response.tool_calls[0].name == "calculator"
    assert response.tool_calls[0].arguments == {"expression": "2 + 2"}
    assert fake_client.chat_kwargs["tools"][0]["function"]["name"] == "calculator"
    assert fake_client.chat_kwargs["messages"][0]["role"] == "system"
    previous_tool_call = fake_client.chat_kwargs["messages"][2]["tool_calls"][0]
    assert previous_tool_call["function"]["name"] == "calculator"
    assert fake_client.chat_kwargs["messages"][3]["role"] == "tool"


def test_ollama_adapter_handles_object_responses_and_strips_thinking() -> None:
    model_response = SimpleNamespace(
        message=SimpleNamespace(
            content="<think>private draft</think> Final answer.",
            tool_calls=[],
        )
    )
    fake_client = FakeOllamaClient(chat_response=model_response)
    client = object.__new__(OllamaClient)
    client._model = "test-model"
    client._client = fake_client
    state = AgentState()
    state.add_user_message("Hi")

    response = client.respond(state, [])

    assert response.text == "Final answer."
    assert not response.tool_calls


def test_ollama_generate_strips_thinking_blocks() -> None:
    fake_client = FakeOllamaClient(
        generate_response={"response": "<think>notes</think> Summary."}
    )
    client = object.__new__(OllamaClient)
    client._model = "test-model"
    client._client = fake_client

    response = client.generate("Summarize this")

    assert response == "Summary."
    assert fake_client.generate_kwargs == {
        "model": "test-model",
        "prompt": "Summarize this",
    }
