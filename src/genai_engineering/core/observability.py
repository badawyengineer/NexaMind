"""Sanitized JSON events and OpenTelemetry API spans for core operations."""

import json
import logging
from contextlib import contextmanager
from time import perf_counter
from typing import Iterator
from uuid import uuid4

from opentelemetry import trace

logger = logging.getLogger("genai_engineering.events")
tracer = trace.get_tracer("genai_engineering")


@contextmanager
def observe(operation: str, **attributes) -> Iterator[dict]:
    """Record duration/status without including prompts, documents, or secrets."""
    event = {"request_id": str(uuid4()), "operation": operation, **attributes}
    started = perf_counter()
    with tracer.start_as_current_span(operation) as span:
        for key, value in attributes.items():
            if isinstance(value, (str, int, float, bool)):
                span.set_attribute(key, value)
        try:
            yield event
            event["status"] = "success"
            span.set_status(trace.StatusCode.OK)
        except Exception as exc:
            event["status"] = "error"
            span.set_attribute("exception.type", type(exc).__name__)
            span.set_status(trace.StatusCode.ERROR)
            raise
        finally:
            event["duration_ms"] = round((perf_counter() - started) * 1000, 3)
            logger.info(json.dumps(event, sort_keys=True, default=str))
