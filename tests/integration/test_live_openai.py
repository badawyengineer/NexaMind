import os

import pytest

from genai_engineering.core.config import Settings
from genai_engineering.core.providers import create_provider
from genai_engineering.task1_structured_extraction.extractor import Extractor

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1" or not os.getenv("OPENAI_API_KEY"),
    reason="Set RUN_LIVE_TESTS=1 and OPENAI_API_KEY to enable paid integration tests",
)


def test_live_invoice_extraction():
    settings = Settings(
        llm_provider="openai", openai_api_key=os.environ["OPENAI_API_KEY"]
    )
    result = Extractor(create_provider(settings)).extract(
        "Invoice #L-1. Vendor: Example Ltd. Total USD 20.00.", "invoice"
    )
    assert result.invoice_number == "L-1"
