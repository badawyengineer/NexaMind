"""Small provider boundary shared by extraction, RAG and the agent."""

from collections.abc import Iterator
from typing import Protocol

from genai_engineering.core.config import Settings, get_settings
from genai_engineering.core.errors import ConfigurationError, ProviderError


class LLMProvider(Protocol):
    def generate(
        self, prompt: str, *, system: str = "", json_mode: bool = False
    ) -> str: ...
    def stream(self, prompt: str, *, system: str = "") -> Iterator[str]: ...


class OpenAIProvider:
    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai"
            )
        from openai import OpenAI

        self.client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )
        self.model = settings.openai_model
        self.temperature = settings.temperature

    def generate(
        self, prompt: str, *, system: str = "", json_mode: bool = False
    ) -> str:
        try:
            messages = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": prompt}
            ]
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            result = self.client.chat.completions.create(**kwargs)
            return result.choices[0].message.content or ""
        except Exception as exc:
            raise ProviderError("OpenAI generation failed") from exc

    def stream(self, prompt: str, *, system: str = "") -> Iterator[str]:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        try:
            result = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stream=True,
            )
            for part in result:
                token = part.choices[0].delta.content
                if token:
                    yield token
        except Exception as exc:
            raise ProviderError("OpenAI streaming failed") from exc


class GeminiProvider:
    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise ConfigurationError(
                "GEMINI_API_KEY is required when LLM_PROVIDER=gemini"
            )
        from google import genai

        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options={"timeout": int(settings.request_timeout * 1000)},
        )
        self.model = settings.gemini_model

    def generate(
        self, prompt: str, *, system: str = "", json_mode: bool = False
    ) -> str:
        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                system_instruction=system or None,
                response_mime_type="application/json" if json_mode else "text/plain",
            )
            result = self.client.models.generate_content(
                model=self.model, contents=prompt, config=config
            )
            return result.text or ""
        except Exception as exc:
            raise ProviderError("Gemini generation failed") from exc

    def stream(self, prompt: str, *, system: str = "") -> Iterator[str]:
        try:
            from google.genai import types

            config = types.GenerateContentConfig(system_instruction=system or None)
            for part in self.client.models.generate_content_stream(
                model=self.model, contents=prompt, config=config
            ):
                if part.text:
                    yield part.text
        except Exception as exc:
            raise ProviderError("Gemini streaming failed") from exc


def create_provider(settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    if settings.llm_provider == "openai":
        return OpenAIProvider(settings)
    if settings.llm_provider == "gemini":
        return GeminiProvider(settings)
    raise ConfigurationError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
