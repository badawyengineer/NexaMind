import json
import logging

from pydantic import BaseModel, ValidationError

from genai_engineering.core.errors import ValidationFailure
from genai_engineering.core.providers import LLMProvider
from genai_engineering.task1_structured_extraction.prompts import SYSTEM, build_prompt
from genai_engineering.task1_structured_extraction.schemas import Invoice, Resume

logger = logging.getLogger(__name__)


class Extractor:
    def __init__(self, provider: LLMProvider, max_attempts: int = 2):
        if max_attempts < 1 or max_attempts > 4:
            raise ValueError("max_attempts must be between 1 and 4")
        self.provider, self.max_attempts = provider, max_attempts

    def extract(self, text: str, kind: str) -> BaseModel:
        if not text or not text.strip():
            raise ValueError("Input text must not be empty")
        if len(text) > 1_000_000:
            raise ValueError("Input text exceeds the 1,000,000 character limit")
        model = {"invoice": Invoice, "resume": Resume}.get(kind)
        if model is None:
            raise ValueError("kind must be 'invoice' or 'resume'")
        last_error = None
        for attempt in range(self.max_attempts):
            prompt = build_prompt(kind, text, model, last_error)
            raw = self.provider.generate(prompt, system=SYSTEM, json_mode=True)
            try:
                return model.model_validate_json(raw)
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                logger.warning(
                    "Structured output validation failed (attempt %d/%d)",
                    attempt + 1,
                    self.max_attempts,
                )
        raise ValidationFailure(
            f"Could not produce valid {kind} JSON after {self.max_attempts} attempts"
        ) from None

    @staticmethod
    def to_json(value: BaseModel) -> str:
        return value.model_dump_json(indent=2)
