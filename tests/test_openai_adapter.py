from types import SimpleNamespace

from app.agent.schemas import ToolExecutionResult, ToolRequestEvent, ToolResultEvent
from app.agent.state import AgentState
from app.llm.openai_client import OpenAIClient


class FakeResponses:
    def __init__(self, response) -> None:
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_openai_adapter_translates_function_calls_and_results() -> None:
    model_response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                call_id="call-2",
                name="calculator",
                arguments='{"expression":"2 + 2"}',
            )
        ],
        output_text="",
    )
    fake_responses = FakeResponses(model_response)
    client = object.__new__(OpenAIClient)
    client._model = "test-model"
    client._client = SimpleNamespace(responses=fake_responses)
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

    assert response.tool_calls[0].arguments == {"expression": "2 + 2"}
    assert fake_responses.kwargs["input"][1]["type"] == "function_call"
    assert fake_responses.kwargs["input"][2]["type"] == "function_call_output"
