import json
from types import SimpleNamespace

from google.genai import types

from genai_engineering.task3_agent.provider import (
    GeminiAgentProvider,
    OpenAIAgentProvider,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculate safely",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    }
]


def test_openai_agent_adapter_normalizes_tool_calls():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            function=SimpleNamespace(
                                name="calculate", arguments='{"expression":"2+2"}'
                            ),
                        )
                    ],
                )
            )
        ]
    )
    seen = {}

    class Completions:
        def create(self, **kwargs):
            seen.update(kwargs)
            return response

    provider = SimpleNamespace(
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
        model="mock-openai",
    )
    result = OpenAIAgentProvider(provider).respond(
        [{"role": "user", "content": "calculate", "_private": "ignored"}], TOOLS
    )

    assert result.calls[0].call_id == "call-1"
    assert result.calls[0].name == "calculate"
    assert json.loads(result.calls[0].arguments) == {"expression": "2+2"}
    assert "_private" not in seen["messages"][0]


def test_openai_agent_adapter_streams_final_tokens_in_same_request():
    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="Answer", tool_calls=None)
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=" ready", tool_calls=None)
                )
            ]
        ),
    ]
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return iter(chunks)

    provider = SimpleNamespace(
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
        model="mock-openai",
    )
    received = []
    result = OpenAIAgentProvider(provider).respond(
        [{"role": "user", "content": "question"}],
        TOOLS,
        on_token=received.append,
    )

    assert result.content == "Answer ready"
    assert received == ["Answer", " ready"]
    assert len(calls) == 1
    assert calls[0]["stream"] is True


def test_gemini_agent_adapter_disables_automatic_execution_and_maps_calls():
    function_call = types.FunctionCall(name="calculate", args={"expression": "2+2"})
    native = types.Content(
        role="model", parts=[types.Part(function_call=function_call)]
    )
    response = SimpleNamespace(
        function_calls=[function_call],
        candidates=[SimpleNamespace(content=native)],
        text=None,
    )
    seen = {}

    class Models:
        def generate_content(self, **kwargs):
            seen.update(kwargs)
            return response

    provider = SimpleNamespace(
        client=SimpleNamespace(models=Models()), model="mock-gemini"
    )
    result = GeminiAgentProvider(provider).respond(
        [
            {"role": "system", "content": "Be safe"},
            {"role": "user", "content": "calculate"},
        ],
        TOOLS,
    )

    assert result.calls[0].name == "calculate"
    assert json.loads(result.calls[0].arguments) == {"expression": "2+2"}
    assert result.native_content is native
    assert seen["config"].automatic_function_calling.disable is True
    assert seen["config"].system_instruction == "Be safe"


def test_gemini_agent_adapter_streams_tokens_from_one_request():
    chunks = [
        SimpleNamespace(text="Answer", function_calls=None, candidates=[]),
        SimpleNamespace(text=" ready", function_calls=None, candidates=[]),
    ]
    calls = []

    class Models:
        def generate_content_stream(self, **kwargs):
            calls.append(kwargs)
            return iter(chunks)

    provider = SimpleNamespace(
        client=SimpleNamespace(models=Models()), model="mock-gemini"
    )
    received = []
    result = GeminiAgentProvider(provider).respond(
        [{"role": "user", "content": "question"}],
        TOOLS,
        on_token=received.append,
    )

    assert result.content == "Answer ready"
    assert received == ["Answer", " ready"]
    assert len(calls) == 1
