import json

import pytest

from genai_engineering.core.observability import observe


def test_observability_records_sanitized_status_and_duration(caplog):
    with caplog.at_level("INFO", logger="genai_engineering.events"):
        with observe("test.operation", provider="mock", model="model") as event:
            event["input_tokens"] = 3

    payload = json.loads(caplog.records[-1].message)
    assert payload["status"] == "success"
    assert payload["duration_ms"] >= 0
    assert payload["input_tokens"] == 3
    assert "prompt" not in payload


def test_observability_marks_failures(caplog):
    with caplog.at_level("INFO", logger="genai_engineering.events"):
        with pytest.raises(RuntimeError), observe("test.failure"):
            raise RuntimeError("expected")

    assert json.loads(caplog.records[-1].message)["status"] == "error"
