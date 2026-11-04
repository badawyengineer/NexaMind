"""Provider-neutral tool-call records and OpenAI/Gemini protocol adapters."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from genai_engineering.core.errors import ProviderError


@dataclass(frozen=True)
class AgentToolCall:
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class AgentTurn:
    content: str
    calls: list[AgentToolCall]
    native_content: object | None = None


class AgentProvider(Protocol):
    model: str

    def respond(
        self,
        messages: list[dict],
        tools: list[dict],
        on_token: Callable[[str], None] | None = None,
    ) -> AgentTurn: ...


class OpenAIAgentProvider:
    def __init__(self, provider):
        self.client = provider.client
        self.model = provider.model

    def respond(
        self,
        messages: list[dict],
        tools: list[dict],
        on_token: Callable[[str], None] | None = None,
    ) -> AgentTurn:
        try:
            safe_messages = [
                {
                    key: value
                    for key, value in message.items()
                    if not key.startswith("_")
                }
                for message in messages
            ]
            kwargs = {
                "model": self.model,
                "messages": safe_messages,
                "tools": tools,
                "tool_choice": "auto",
                "temperature": 0,
            }
            if on_token is None:
                result = self.client.chat.completions.create(**kwargs)
                message = result.choices[0].message
                calls = [
                    AgentToolCall(call.id, call.function.name, call.function.arguments)
                    for call in (message.tool_calls or [])
                ]
                return AgentTurn(message.content or "", calls)

            result = self.client.chat.completions.create(**kwargs, stream=True)
            content = []
            calls_by_index = {}
            for chunk in result:
                delta = chunk.choices[0].delta
                if delta.content:
                    content.append(delta.content)
                    on_token(delta.content)
                for call_delta in delta.tool_calls or []:
                    call = calls_by_index.setdefault(
                        call_delta.index,
                        {"id": "", "name": "", "arguments": ""},
                    )
                    if call_delta.id:
                        call["id"] = call_delta.id
                    if call_delta.function and call_delta.function.name:
                        call["name"] += call_delta.function.name
                    if call_delta.function and call_delta.function.arguments:
                        call["arguments"] += call_delta.function.arguments
            calls = [
                AgentToolCall(
                    item["id"] or str(uuid4()), item["name"], item["arguments"]
                )
                for _, item in sorted(calls_by_index.items())
            ]
            return AgentTurn("".join(content), calls)
        except Exception as exc:
            raise ProviderError("OpenAI agent request failed") from exc


class GeminiAgentProvider:
    def __init__(self, provider):
        self.client = provider.client
        self.model = provider.model

    def respond(
        self,
        messages: list[dict],
        tools: list[dict],
        on_token: Callable[[str], None] | None = None,
    ) -> AgentTurn:
        try:
            from google.genai import types

            contents, config = self._build_request(messages, tools)
            if on_token is None:
                response = self.client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
                response_calls = response.function_calls or []
                response_text = response.text or ""
                native = response.candidates[0].content if response_calls else None
            else:
                response_calls = []
                response_text_parts = []
                native_parts = []
                for response in self.client.models.generate_content_stream(
                    model=self.model, contents=contents, config=config
                ):
                    if response.text:
                        response_text_parts.append(response.text)
                        on_token(response.text)
                    if response.function_calls:
                        response_calls = response.function_calls
                    for candidate in response.candidates or []:
                        if candidate.content:
                            native_parts.extend(candidate.content.parts or [])
                response_text = "".join(response_text_parts)
                native = (
                    types.Content(role="model", parts=native_parts)
                    if response_calls
                    else None
                )
            calls = [
                AgentToolCall(
                    str(getattr(call, "id", None) or uuid4()),
                    call.name,
                    json.dumps(call.args or {}),
                )
                for call in response_calls
            ]
            return AgentTurn(response_text, calls, native)
        except Exception as exc:
            raise ProviderError("Gemini agent request failed") from exc

    @staticmethod
    def _build_request(messages: list[dict], tools: list[dict]):
        from google.genai import types

        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        contents = []
        for message in messages:
            role = message["role"]
            if role == "system":
                continue
            if role == "tool":
                payload = json.loads(message["content"])
                contents.append(
                    types.Content(
                        role="tool",
                        parts=[
                            types.Part.from_function_response(
                                name=message["name"], response={"result": payload}
                            )
                        ],
                    )
                )
            elif role == "assistant" and message.get("_native_content") is not None:
                contents.append(message["_native_content"])
            else:
                contents.append(
                    types.Content(
                        role="model" if role == "assistant" else "user",
                        parts=[types.Part.from_text(text=message.get("content", ""))],
                    )
                )
        declarations = [
            types.FunctionDeclaration(
                name=item["function"]["name"],
                description=item["function"]["description"],
                parameters_json_schema=item["function"]["parameters"],
            )
            for item in tools
        ]
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=declarations)],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            temperature=0,
        )
        return contents, config
