import json
from types import SimpleNamespace

from genai_engineering.core.providers import OpenAIProvider
from genai_engineering.task3_agent.agent import ToolAgent
from genai_engineering.task3_agent.provider import AgentToolCall, AgentTurn


def _call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def test_agent_handles_multiple_tools_and_returns_a_final_answer():
    first = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        _call("1", "calculate", {"expression": "6 * 7"}),
                        _call("2", "read_file", {"path": "report.txt"}),
                    ],
                )
            )
        ]
    )
    final = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="The calculation is 42.", tool_calls=None
                )
            )
        ]
    )
    replies = iter([first, final])
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return next(replies)

    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    provider.model = "mock-model"
    reader = SimpleNamespace(read=lambda path: f"Contents of {path}")

    result = ToolAgent(provider, file_reader=reader).run("Calculate and inspect a file")

    assert result.answer == "The calculation is 42."
    assert [step.tool for step in result.trace] == ["calculate", "read_file"]
    assert result.trace[0].observation == "42"
    assert len(calls) == 2
    assert all(step.status == "success" for step in result.trace)
    assert all(step.duration_ms >= 0 for step in result.trace)


def test_agent_enforces_total_tool_call_budget():
    class Adapter:
        model = "mock"

        def respond(self, messages, tools):
            return AgentTurn(
                "",
                [
                    AgentToolCall("1", "calculate", '{"expression":"1+1"}'),
                    AgentToolCall("2", "calculate", '{"expression":"2+2"}'),
                ],
            )

    result = ToolAgent(object(), agent_provider=Adapter(), max_tool_calls=1).run(
        "calculate"
    )

    assert result.trace[0].observation == "2"
    assert result.trace[1].status == "budget_exhausted"
    assert "tool call limit" in result.answer


def test_malformed_arguments_are_reported_as_tool_errors():
    class Adapter:
        model = "mock"
        turns = iter(
            [
                AgentTurn("", [AgentToolCall("1", "calculate", "not-json")]),
                AgentTurn("Handled the invalid tool call.", []),
            ]
        )

        def respond(self, messages, tools):
            return next(self.turns)

    result = ToolAgent(object(), agent_provider=Adapter()).run("calculate")

    assert result.trace[0].status == "error"
    assert "error" in result.trace[0].observation
    assert result.answer == "Handled the invalid tool call."


def test_agent_forwards_provider_stream_tokens_and_tool_progress():
    class Adapter:
        model = "mock"

        def respond(self, messages, tools, on_token=None):
            on_token("Streamed answer")
            return AgentTurn("Streamed answer", [])

    events = []
    result = ToolAgent(object(), agent_provider=Adapter()).run(
        "answer", on_event=events.append
    )

    assert result.answer == "Streamed answer"
    assert events == [{"type": "token", "text": "Streamed answer"}]
