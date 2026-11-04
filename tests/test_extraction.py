import pytest

from genai_engineering.core.errors import ValidationFailure
from genai_engineering.task1_structured_extraction.extractor import Extractor
from genai_engineering.task1_structured_extraction.schemas import Invoice, Resume


def test_invoice_validated(fake_provider):
    result = Extractor(
        fake_provider(['{"invoice_number":"I-1","total":"12.50"}'])
    ).extract("Invoice I-1 total 12.50", "invoice")
    assert result.invoice_number == "I-1"
    assert str(result.total) == "12.50"


def test_invalid_output_gets_bounded_repair(fake_provider):
    provider = fake_provider(['{"unexpected": 1}', '{"invoice_number":"ok"}'])
    result = Extractor(provider, max_attempts=2).extract("invoice", "invoice")
    assert result.invoice_number == "ok"
    assert "Previous response validation error" in provider.prompts[1]


def test_bad_json_fails_after_limit(fake_provider):
    with pytest.raises(ValidationFailure):
        Extractor(
            fake_provider(["not json", "still not json"]), max_attempts=2
        ).extract("resume text", "resume")


def test_empty_input_rejected(fake_provider):
    with pytest.raises(ValueError):
        Extractor(fake_provider([])).extract(" ", "invoice")


def test_example_json_outputs_match_schemas():
    from pathlib import Path

    base = Path(__file__).parents[1] / "data" / "samples"
    Invoice.model_validate_json((base / "example_invoice.json").read_text())
    Resume.model_validate_json((base / "example_resume.json").read_text())
