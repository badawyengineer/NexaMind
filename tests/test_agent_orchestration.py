import json
from types import SimpleNamespace

from genai_engineering.core.providers import OpenAIProvider
from genai_engineering.task3_agent.agent import ToolAgent


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
